"""Run the pilot per research/PILOT_SPEC_v1.md.

Modes:
  --dry            cost estimate from live OpenRouter prices, no model calls
  --smoke          clusters 0-1 only (all arms incl. downstream)
  --full           all clusters (requires --i-have-user-approval)

Provider pinning: official-or-≥FP8 via provider.quantizations allowlist.
Traces are append-only JSONL in traces/, one file per invocation.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["deepseek/deepseek-v4-flash", "z-ai/glm-5.2", "openai/gpt-5.6-luna"] #openrouter name 
QUANTS = ["fp8", "bf16", "fp16", "fp32", "unknown"]  # unknown = official first-party
SYS = json.loads((ROOT / "data" / "manifest_v1.json").read_text())["system_template"]
MAX_TOK = {"behavioral": 400, "e1": 350, "l2": 350, "explicit": 200, "e2": 400, "l2e2": 400}


def live_prices() -> dict:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        data = json.load(r)["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
            for m in data if m["id"] in MODELS}


def estimate(cases: list[dict]) -> None:
    prices = live_prices()
    missing = [m for m in MODELS if m not in prices]
    if missing:
        sys.exit(f"model IDs not on OpenRouter: {missing} — fix before running")
    n_downstream = sum(1 for c in cases if c["arm"] in ("e1", "l2"))
    total_calls = (len(cases) + n_downstream) * len(MODELS)
    in_tok, out_tok = 420, 280  # envelope per spec
    cost = sum((in_tok * p_in + out_tok * p_out) * (len(cases) + n_downstream)
               for p_in, p_out in prices.values())
    print(f"calls: {total_calls} ({len(cases)} stimuli + {n_downstream} downstream, x{len(MODELS)} models)")
    for m, (pi, po) in prices.items():
        print(f"  {m}: ${pi * 1e6:.3f}/M in, ${po * 1e6:.2f}/M out")
    print(f"projected cost @ {in_tok}in/{out_tok}out per call: ${cost:.2f}")
    if cost > 25:
        sys.exit("ABORT: projected cost exceeds $25 hard cap (spec).")


def call(client: OpenAI, model: str, history: str, query: str, max_tokens: int) -> str:
    msgs = [{"role": "system", "content": SYS.format(history=history)},
            {"role": "user", "content": query}]
    delay = 5.0
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0, max_tokens=max_tokens,
                extra_body={"provider": {"quantizations": QUANTS, "allow_fallbacks": False}})
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
            raise RuntimeError("empty content")
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 3
    raise RuntimeError("unreachable")


def run_one(client: OpenAI, model: str, c: dict) -> list[dict]:
    out = []
    try:
        resp = call(client, model, c["history"], c["query"], MAX_TOK[c["arm"]])
    except Exception as e:  # noqa: BLE001
        resp = f"<ERROR: {e}>"
    base = {k: c[k] for k in ("case_id", "frame", "cluster", "form", "gap",
                              "witness", "wit_token", "fact_noun", "arm")}
    out.append(base | dict(model=model, response=resp))
    if c["arm"] in ("e1", "l2") and not resp.startswith("<ERROR"):
        notes_hist = f"[Memory notes written {'2025-12-09'}]\n{resp}\n"
        arm2 = "e2" if c["arm"] == "e1" else "l2e2"
        try:
            resp2 = call(client, model, notes_hist, c["downstream_query"], MAX_TOK[arm2])
        except Exception as e:  # noqa: BLE001
            resp2 = f"<ERROR: {e}>"
        out.append(base | dict(arm=arm2, case_id=c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1),
                               model=model, response=resp2))
    return out


def main() -> None:
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--i-have-user-approval", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated OpenRouter IDs (default: all three)")
    ap.add_argument("--resume", default=None, metavar="TRACE",
                    help="existing trace file: skip (model, case_id) pairs already "
                         "completed without error; new records append to THAT file")
    a = ap.parse_args()
    MODELS = [m.strip() for m in a.models.split(",") if m.strip()]
    cases = [json.loads(ln) for ln in (ROOT / "data" / "stimuli_v1.jsonl").read_text().splitlines()]
    if a.smoke:
        cases = [c for c in cases if c["cluster"] < 2]
    if a.dry:
        estimate(cases)
        return
    if a.full and not a.i_have_user_approval:
        sys.exit("full run requires --i-have-user-approval (workspace cost gate)")
    if not (a.smoke or a.full):
        sys.exit("pick --dry, --smoke or --full")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    done_keys: set[tuple[str, str]] = set()
    if a.resume:
        trace = Path(a.resume)
        for ln in trace.read_text().splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if not rec["response"].startswith("<ERROR"):
                done_keys.add((rec["model"], rec["case_id"]))
        print(f"resume: {len(done_keys)} completed records found in {trace.name}",
              file=sys.stderr)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = "smoke" if a.smoke else "full"
        trace = ROOT / "traces" / f"run-{tag}-{stamp}.jsonl"
    trace.parent.mkdir(exist_ok=True)

    def complete(m: str, c: dict) -> bool:
        if (m, c["case_id"]) not in done_keys:
            return False
        if c["arm"] in ("e1", "l2"):  # stimulus also owes its downstream record
            arm2 = "e2" if c["arm"] == "e1" else "l2e2"
            return (m, c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1)) in done_keys
        return True

    jobs = [(m, c) for m in MODELS for c in cases if not complete(m, c)]
    done = 0
    with trace.open("a") as f, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, client, m, c) for m, c in jobs]
        for fut in cf.as_completed(futs):
            for rec in fut.result():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} stimuli done", file=sys.stderr)
    errs = sum(1 for ln in trace.read_text().splitlines()
               if '"<ERROR' in ln or '"response": "<ERROR' in ln)
    print(f"trace: {trace} | stimuli: {len(jobs)} | error responses: {errs}")


if __name__ == "__main__":
    main()
