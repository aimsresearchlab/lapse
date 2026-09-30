#!/usr/bin/env python3
"""Run qualifier-ablation stimuli through pinned cheap writers. Append-only trace; resumable; hard cost cap."""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUANTS = ["fp8", "bf16", "fp16", "fp32", "unknown"]
EXTRA_BODY = {"provider": {"quantizations": QUANTS, "allow_fallbacks": False}}

def prices(models):
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    cat = {m["id"]: m for m in json.load(urllib.request.urlopen(req, timeout=30))["data"]}
    return {m: (float(cat[m]["pricing"]["prompt"]), float(cat[m]["pricing"]["completion"])) for m in models}

def estimate(stims, models, pr, sys_tmpl, max_tokens):
    tot = 0.0
    for m in models:
        pi, po = pr[m]
        for s in stims:
            toks = (len(sys_tmpl.format(history=s["history"])) + len(s["query"])) / 3.5 + 20
            tot += toks * pi + max_tokens * po
    return tot

def call(client, model, sys_prompt, query, max_tokens):
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
    delay = 5.0
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=max_tokens, extra_body=EXTRA_BODY)
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty content")
            u = r.usage
            return text, {"provider": getattr(r, "provider", None), "served_model": r.model, "id": r.id,
                          "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return f"<ERROR: {e}>", {"provider": None, "served_model": None, "id": None, "prompt_tokens": None, "completion_tokens": None}
            time.sleep(delay); delay *= 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, default=HERE / "stimuli.jsonl"); ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    ap.add_argument("--trace", type=Path, default=HERE / "trace.jsonl"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cap-usd", type=float, required=True); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--models", default=None)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text()); models = a.models.split(",") if a.models else man["models"]
    stims = [json.loads(l) for l in a.stimuli.read_text().splitlines()]
    if a.smoke:
        stims = [s for s in stims if s["cluster"] == 0 and s["frame"] in ("lodging", "vehicle")]
    done = set()
    if a.trace.exists():
        for l in a.trace.read_text().splitlines():
            r = json.loads(l)
            if not r["response"].startswith("<ERROR"):
                done.add((r["model"], r["case_id"]))
    jobs = [(m, s) for m in models for s in stims if (m, s["case_id"]) not in done]
    pr = prices(models); est = estimate([s for _, s in jobs], models, pr, man["system_template"], man["max_tokens"]) / max(1, len(models)) * 1  # per-job already includes model loop
    est = sum(((len(man["system_template"].format(history=s["history"])) + len(s["query"])) / 3.5 + 20) * pr[m][0] + man["max_tokens"] * pr[m][1] for m, s in jobs)
    print(json.dumps({"jobs": len(jobs), "skipped_done": len(done), "estimated_usd_worst_case": round(est, 4), "cap_usd": a.cap_usd, "prices": pr}))
    if est > a.cap_usd:
        raise SystemExit("cap exceeded; refusing")
    if a.dry:
        return
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    def one(job):
        m, s = job
        text, meta = call(client, m, man["system_template"].format(history=s["history"]), s["query"], man["max_tokens"])
        return {k: s[k] for k in ("case_id", "frame", "cluster", "form", "source_case_id", "gap", "gap_date", "eval_date", "witness", "wit_token", "fact_noun", "arm", "carrier", "marker")} | {"model": m, "response": text, "meta": meta, "ts": time.time(), "stimuli_sha256": man["stimuli_sha256"]}
    with a.trace.open("a") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        n = 0
        for rec in ex.map(one, jobs):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush(); n += 1
    errs = sum(1 for l in a.trace.read_text().splitlines() if json.loads(l)["response"].startswith("<ERROR"))
    print(json.dumps({"written": n, "trace": str(a.trace), "errors_in_trace": errs}))

if __name__ == "__main__":
    main()
