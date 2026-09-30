"""Run v2 per research/PILOT_SPEC_v2.md (frozen 2026-08-12).

The frozen builder is deterministic with --eval-date required; RUN builds use
the run date. Build run stimuli OUTSIDE the repo (the builder writes to
<parent-of-tools>/data, so copy tools/ to a scratch dir and run it there) to
keep the committed reference build (eval-date 2026-08-11) byte-intact.

Modes:
  --dry     cost estimate from live OpenRouter prices, no model calls
  --smoke   spec §8 smoke subset (~260 stimuli + derived; deterministic
            lowest-cluster selection over every NEW cell type; dev split only)
  --full    all run rows (requires --i-have-user-approval — launch gate)

Downstream (e2/l2e2) calls use the ORIGINAL row's session date (gap_date) as
the notes-written date, not a hardcoded constant (v1 runner's '2025-12-09'
was correct only because v1 had a single stale gap).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["deepseek/deepseek-v4-flash"]
QUANTS = ["fp8", "bf16", "fp16", "fp32", "unknown"]
MAX_TOK = {"behavioral": 400, "e1": 350, "l2": 350, "explicit": 200,
           "e2": 400, "l2e2": 400, "anchor": 450}
NEW_FRAMES = ("equipment", "affiliate_role", "project")


def smoke_select(cases: list[dict]) -> list[dict]:
    """Spec §8: lowest-cluster coverage of every NEW cell type (dev split)."""
    keep = []
    for c in cases:
        comp, fr, cl, form, gap = (c["component"], c["frame"], c["cluster"],
                                   c["form"], c["gap"])
        new_frame = fr in NEW_FRAMES
        if comp == "behavioral_core" and new_frame and cl < 2:
            keep.append(c)                                   # 3f×2g×3fr×2cl=36
        elif comp == "e1_primary" and new_frame and cl < 2:
            keep.append(c)                                   # 12 (+12 e2)
        elif comp == "explicit" and new_frame and cl < 2:
            keep.append(c)                                   # 12
        elif comp == "l2" and new_frame and cl < 1:
            keep.append(c)                                   # 6 (+6 l2e2)
        elif comp == "perf" and (cl < 2 if c["arm"] == "e1" else cl < 2):
            keep.append(c)                                   # beh 32 + e1 32
        elif comp == "behavioral_boundary" and cl < 1:
            keep.append(c)                                   # 3f×8fr=24
        elif comp == "e1_boundary" and cl < 1:
            keep.append(c)                                   # 2f×8fr=16
        elif comp == "gap_gradient" and form == "prog" and cl < 1:
            keep.append(c)                                   # 3g×8fr=24
        elif comp == "anchor" and cl < 1:
            keep.append(c)                                   # 3f×2g×8fr=48
        elif comp == "carrier_behavioral_p2" and cl < 1 \
                and form == "prog" and fr in ("lodging", "equipment"):
            keep.append(c)                                   # 5car×2fr=10
        elif comp == "carrier_fresh_p3" and cl < 1 \
                and fr in ("lodging", "equipment"):
            keep.append(c)                                   # 5car×2fr=10
        elif comp == "lexeme_subject" and cl < 1:
            keep.append(c)                                   # 9
    assert all(c["split"] == "dev" for c in keep), "smoke must stay in dev"
    return keep


def live_prices() -> dict:
    import urllib.request
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models",
                                timeout=30) as r:
        data = json.load(r)["data"]
    return {m["id"]: (float(m["pricing"]["prompt"]),
                      float(m["pricing"]["completion"]))
            for m in data if m["id"] in MODELS}


def estimate(cases: list[dict], hard_cap: float) -> None:
    prices = live_prices()
    missing = [m for m in MODELS if m not in prices]
    if missing:
        sys.exit(f"model IDs not on OpenRouter: {missing} — fix before running")
    n_down = sum(1 for c in cases if "downstream_query" in c)
    total = (len(cases) + n_down) * len(MODELS)
    in_tok, out_tok = 420, 280
    cost = sum((in_tok * pi + out_tok * po) * (len(cases) + n_down)
               for pi, po in prices.values())
    print(f"calls: {total} ({len(cases)} stimuli + {n_down} derived, "
          f"x{len(MODELS)} models)")
    for m, (pi, po) in prices.items():
        print(f"  {m}: ${pi * 1e6:.3f}/M in, ${po * 1e6:.2f}/M out")
    print(f"projected cost @ {in_tok}in/{out_tok}out per call: ${cost:.2f}")
    if cost > hard_cap:
        sys.exit(f"ABORT: projected cost exceeds ${hard_cap} cap.")


EXTRA_BODY = {"provider": {"quantizations": QUANTS,
                           "allow_fallbacks": False}}


def call(client: OpenAI, model: str, sys_prompt: str, query: str,
         max_tokens: int) -> str:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}]
    delay = 5.0
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=0,
                max_tokens=max_tokens,
                extra_body=EXTRA_BODY)
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


def run_one(client: OpenAI, model: str, c: dict, sys_tmpl: str) -> list[dict]:
    out = []
    sys_prompt = sys_tmpl.format(history=c["history"])
    try:
        resp = call(client, model, sys_prompt, c["query"], MAX_TOK[c["arm"]])
    except Exception as e:  # noqa: BLE001
        resp = f"<ERROR: {e}>"
    base = {k: c[k] for k in ("case_id", "frame", "cluster", "form", "gap",
                              "gap_date", "witness", "wit_token", "fact_noun",
                              "arm", "component", "carrier", "tier", "split")}
    out.append(base | dict(model=model, response=resp))
    if "downstream_query" in c and c["component"] != "v1_grid_extra" \
            and not resp.startswith("<ERROR"):
        notes_hist = f"[Memory notes written {c['gap_date']}]\n{resp}\n"
        arm2 = "e2" if c["arm"] == "e1" else "l2e2"
        try:
            resp2 = call(client, model, sys_tmpl.format(history=notes_hist),
                         c["downstream_query"], MAX_TOK[arm2])
        except Exception as e:  # noqa: BLE001
            resp2 = f"<ERROR: {e}>"
        out.append(base | dict(
            arm=arm2, model=model, response=resp2,
            case_id=c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1)))
    return out


def main() -> None:
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True,
                    help="run-build stimuli_v2.jsonl (built at the run date)")
    ap.add_argument("--manifest", required=True,
                    help="matching manifest_v2.json (for system_template)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--i-have-user-approval", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--resume", default=None, metavar="TRACE")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                    help="OpenAI-compatible endpoint; e.g. a local vllm serve "
                         "for the A100 open-weights rows (spec model table)")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="chat_template_kwargs enable_thinking=false "
                         "(Qwen3 local serving; keeps stimuli byte-identical)")
    a = ap.parse_args()
    MODELS = [m.strip() for m in a.models.split(",") if m.strip()]
    local = "openrouter.ai" not in a.base_url
    global EXTRA_BODY
    if local:
        EXTRA_BODY = {}          # provider pinning is OpenRouter-specific
    if a.disable_thinking:
        EXTRA_BODY = EXTRA_BODY | {
            "chat_template_kwargs": {"enable_thinking": False}}
    manifest = json.loads(Path(a.manifest).read_text())
    sys_tmpl = manifest["system_template"]
    cases = [json.loads(ln) for ln in Path(a.stimuli).read_text().splitlines()]
    cases = [c for c in cases if c["component"] != "v1_grid_extra"]
    if a.smoke:
        cases = smoke_select(cases)
    if a.dry:
        if local:
            n_down = sum(1 for c in cases if "downstream_query" in c)
            print(f"local endpoint {a.base_url}: "
                  f"{(len(cases) + n_down) * len(MODELS)} calls, $0")
        else:
            estimate(cases, hard_cap=2.0 if a.smoke else 50.0)
        return
    if a.full and not a.i_have_user_approval:
        sys.exit("full run requires --i-have-user-approval (launch gate)")
    if not (a.smoke or a.full):
        sys.exit("pick --dry, --smoke or --full")
    client = OpenAI(base_url=a.base_url,
                    api_key=os.environ["OPENROUTER_API_KEY"] if not local
                    else os.environ.get("LOCAL_API_KEY", "local"))
    done_keys: set[tuple[str, str]] = set()
    if a.resume:
        trace = Path(a.resume)
        for ln in trace.read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if not rec["response"].startswith("<ERROR"):
                    done_keys.add((rec["model"], rec["case_id"]))
        print(f"resume: {len(done_keys)} completed in {trace.name}",
              file=sys.stderr)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = "smoke" if a.smoke else "full"
        trace = ROOT / "traces" / f"run-v2-{tag}-{stamp}.jsonl"
    trace.parent.mkdir(exist_ok=True)

    def complete(m: str, c: dict) -> bool:
        if (m, c["case_id"]) not in done_keys:
            return False
        if "downstream_query" in c:
            arm2 = "e2" if c["arm"] == "e1" else "l2e2"
            return (m, c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1)) \
                in done_keys
        return True

    jobs = [(m, c) for m in MODELS for c in cases if not complete(m, c)]
    done = 0
    with trace.open("a") as f, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, client, m, c, sys_tmpl) for m, c in jobs]
        for fut in cf.as_completed(futs):
            for rec in fut.result():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} stimuli done", file=sys.stderr)
    errs = sum(1 for ln in trace.read_text().splitlines()
               if '"response": "<ERROR' in ln)
    print(f"trace: {trace} | stimuli: {len(jobs)} | error responses: {errs}")


if __name__ == "__main__":
    main()
