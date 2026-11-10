#!/usr/bin/env python3
"""Writer step (research/STRUCTURED_RECORD_SPEC_2026-10-10.md, sections 4, 9 and 10).

DeepSeek V4 Flash (OpenRouter, Baidu fp8 only; deviation D1: official API had no balance), writes memory notes for the 120 verification-tool facts in
progressive and simple-present form, under the free-text instruction (F, the paper's e1 query) and
the structured-record instruction (S). Session and writer date 2026-01-06.

    python write_notes.py build            # writer_inputs.jsonl
    python write_notes.py dev              # one session per instruction, prints raw output
    python write_notes.py run [--workers 32]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRAMES = ("affiliate_role", "class", "equipment", "project", "vehicle", "workplace")
DATE = "2026-01-06"
SYSTEM = json.loads((ROOT / "data/manifest_v2.json").read_text())["system_template"].replace("2026-08-11", DATE)
CARRIER = "hey! quick life update — {core}. also, any podcast recommendations for my commute?"
ASSISTANT = "Nice! For podcasts, try 99% Invisible or Radiolab — both great for commutes."
E1 = "Before we continue: write concise memory notes about me for your future sessions. Just the notes."
S_EXTRA = (" Write each note as a record with exactly these four fields, one per line: said: the user's own words,"
           " quoted exactly; fact: the fact in a short normalized form; form: the grammatical form of what the user"
           " said (simple present, present progressive, past, future, or other); observed: the session date.")
QUERY = {"F": E1, "S": E1 + S_EXTRA}
MODEL, BASE_URL, MAX_TOKENS = "deepseek/deepseek-v4-flash", "https://openrouter.ai/api/v1", 600
INPUTS, TRACE = HERE / "writer_inputs.jsonl", HERE / "writer_trace.jsonl"


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def build():
    assert "2026-08-11" not in SYSTEM and DATE in SYSTEM
    sent = {}
    for r in load(ROOT / "data/stimuli_v2.jsonl"):
        if r["frame"] in FRAMES and r["carrier"] == "c0_original" and r["subject"] == "first" and r["form"] in ("prog", "simple"):
            m = re.search(r"quick life update — (.*?)\. also", r["history"])
            k = (r["frame"], r["cluster"], r["form"])
            assert sent.get(k, (m.group(1), r["witness"], r["wit_token"])) == (m.group(1), r["witness"], r["wit_token"])
            sent[k] = (m.group(1), r["witness"], r["wit_token"])
    assert len(sent) == 240
    rows = []
    for (frame, cl, form), (core, w, tok) in sorted(sent.items()):
        hist = f"[Session dated {DATE}]\nUser: {CARRIER.format(core=core)}\nAssistant: {ASSISTANT}\n"
        for instr in ("F", "S"):
            rows.append({"id": f"W-{frame}-{cl:02d}-{form.upper()}-{instr}", "frame": frame, "cluster": cl,
                         "content_key": f"{frame}:{cl:02d}", "form": "progressive" if form == "prog" else "simple",
                         "instruction": instr, "user_sentence": core + ".", "witness": w, "wit_token": tok,
                         "system": SYSTEM.format(history=hist), "query": QUERY[instr]})
    INPUTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(INPUTS.read_bytes()).hexdigest()}))


def client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def call(c, row, extra):
    t = time.time()
    rec = {"item_id": row["id"], "model": MODEL, "base_url": BASE_URL, "max_tokens": MAX_TOKENS, "temperature": 0,
           "extra_body": extra, "timestamp_unix": t, "error": None}
    try:
        r = c.chat.completions.create(model=MODEL, temperature=0, max_tokens=MAX_TOKENS, extra_body=extra,
                                      messages=[{"role": "system", "content": row["system"]}, {"role": "user", "content": row["query"]}])
        m = r.choices[0].message
        rec.update({"response": (m.content or "").strip(), "reasoning_content": getattr(m, "reasoning_content", None),
                    "finish_reason": r.choices[0].finish_reason, "served_model": r.model,
                    "usage": r.usage.model_dump() if r.usage else None,
                    "served_provider": getattr(r, "provider", None)})
        if not rec["response"]:
            rec["error"] = "empty content"
    except Exception as e:  # kept as a failed attempt
        rec["error"] = repr(e)
    rec["elapsed_seconds"] = round(time.time() - t, 3)
    return rec


EXTRA = {"provider": {"only": ["Baidu"], "quantizations": ["fp8"], "allow_fallbacks": False},
         "reasoning": {"enabled": False}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "dev", "run"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    rows = load(INPUTS)
    c = client()
    if a.cmd == "dev":
        for row in [r for r in rows if r["content_key"] == "workplace:00" and r["form"] == "progressive"]:
            rec = call(c, row, EXTRA)
            print(json.dumps({k: rec.get(k) for k in ("item_id", "error", "finish_reason", "served_model", "usage")}))
            print(rec.get("reasoning_content") and "REASONING PRESENT", "\n" + str(rec.get("response")) + "\n")
        return
    for attempt in range(3):
        done = {r["item_id"] for r in load(TRACE) if not r["error"]} if TRACE.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            break
        with cf.ThreadPoolExecutor(a.workers) as ex, TRACE.open("a") as f:
            for rec in ex.map(lambda r: call(c, r, EXTRA), todo):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    final = {r["item_id"]: r for r in load(TRACE) if not r["error"]}
    print(json.dumps({"completed": len(final), "of": len(rows), "error_attempts": sum(1 for r in load(TRACE) if r["error"]),
                      "finish": sorted({r["finish_reason"] for r in final.values()})}))


if __name__ == "__main__":
    sys.exit(main())
