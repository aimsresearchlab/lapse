"""Part A of research/READ_TIME_LADDER_SPEC_2026-09-02.md: read-time ladder.

Reader sees ONE memory note, its write date, and today's date. Three note
forms (FLAT / PROG / BOUND) x two gaps (stale 245 d, fresh 2 d) x 8 frames x
12 witnesses x two tasks:

  A1  "probability this is still true today", one integer 0-100
  A2  fill in a field, or UNKNOWN if it may be outdated (COMMIT / ABSTAIN)

Witnesses = clusters 0-11 of the frozen v2 grid (data/stimuli_v2.jsonl,
memory-write arm, c0_original carrier). Note text is the grid's c0 utterance
recast as a third-person memory line, one per frame (NOTES below).

Deviations from the spec text, fixed before any call (2026-09-08):
  * BOUND month follows the frozen bound rule (write month + 3, skip Dec/Jan)
    instead of the literal "March 2026", because today is 2026-09-08 and a
    fresh note bounded by March 2026 would already be expired. Stale ->
    "until April 2026" (expired), fresh -> "until February 2027" (ahead).
  * The year is always written (the "until December" ambiguity in
    Limitations is avoided by design).
  * Cell count is 8 x 12 x 3 x 2 x 2 = 1,152 per reader, 3,456 total
    (spec estimated ~5,200; nothing was dropped, the estimate was loose).

Usage:
    python run_ladder.py --build            # write inputs.jsonl, no calls
    python run_ladder.py --dry              # 2 calls/reader, print raw
    python run_ladder.py --full             # all, resumable, 3 readers
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TODAY = dt.date(2026, 9, 8)
GAPS = {"stale": 245, "fresh": 2}          # manifest_v2 gap_offsets_days
READERS = ["deepseek/deepseek-v4-flash", "openai/gpt-5.6-luna", "z-ai/glm-5.2"]
QUANTS = ["fp8", "bf16", "fp16", "fp32", "unknown"]
EXTRA_BODY = {"provider": {"quantizations": QUANTS, "allow_fallbacks": False}}
# Official first-party endpoint for deepseek (added 2026-09-08 after the temp-0
# noise finding: OpenRouter has no first-party DeepSeek host for this model, and
# the unpinned traces cannot say which of its 15 third-party hosts served a call).
DIRECT = {"deepseek/deepseek-v4-flash": ("https://api.deepseek.com", "DEEPSEEK_API_KEY",
                                         "deepseek-v4-flash", "deepseek-official")}
MAX_TOK = {"A1": 1600, "A2": 1600}   # reasoning readers spend tokens before content
N_CLUSTERS = 12

# frame -> (flat, prog, A2 field label). {w} = witness as in the grid.
NOTES = {
    "lodging":        ("User lives at {w} on Fifth Street.",
                       "User is staying at {w} on Fifth Street.",
                       "User's current home address"),
    "workplace":      ("User works at {w}.",
                       "User is working at {w}.",
                       "User's current employer"),
    "vehicle":        ("User drives a {w}.",
                       "User is driving a {w}.",
                       "User's current car"),
    "class":          ("User takes a ceramics class at {w} on Tuesdays.",
                       "User is taking a ceramics class at {w} on Tuesdays.",
                       "User's current class studio"),
    "equipment":      ("User uses a {w} on loan from the conservatory.",
                       "User is using a {w} on loan from the conservatory.",
                       "User's current instrument"),
    "affiliate_role": ("User lectures at {w} as an affiliate.",
                       "User is lecturing at {w} as an affiliate.",
                       "User's current college affiliation"),
    "project":        ("User works on {w}.",
                       "User is working on {w}.",
                       "User's current project"),
    "household":      ("User lives with their cousin {w}.",
                       "User is staying with their cousin {w}.",
                       "User's current housemate"),
}

SYS = ("You are a personal assistant with long-term memory of your "
       "conversations with the user. Today's date: {today}. One stored "
       "memory note is shown below together with the date it was written.\n\n"
       "--- Memory note (written {written}) ---\n{note}\n--- End of note ---")
Q_A1 = ("What is the probability that this memory note is still true today? "
        "Answer with a single integer from 0 to 100 and nothing else.")
Q_A2 = ("Using only the memory note above, fill in the field below. If the "
        "information may be outdated, write UNKNOWN instead. Reply with the "
        "field value only.\n\n{label}:")


def bound_phrase(written: dt.date) -> str:
    """Frozen bound rule: write month + 3, never Dec/Jan (skip forward)."""
    m, y = written.month + 3, written.year
    while True:
        if m > 12:
            m -= 12
            y += 1
        if m in (12, 1):
            m += 1
            continue
        break
    return f"until {dt.date(y, m, 1).strftime('%B')} {y}"


def witnesses() -> dict[str, list[tuple[str, str]]]:
    rows = [json.loads(l) for l in (ROOT / "data" / "stimuli_v2.jsonl").read_text().splitlines()]
    out: dict[str, dict[int, tuple[str, str]]] = {}
    for r in rows:
        if (r["arm"] == "e1" and r["carrier"] == "c0_original" and r["form"] == "prog"
                and r["gap"] == "stale" and r.get("subject") == "first"
                and r["cluster"] < N_CLUSTERS):
            out.setdefault(r["frame"], {})[r["cluster"]] = (r["witness"], r["wit_token"])
    return {f: [out[f][c] for c in range(N_CLUSTERS)] for f in NOTES}


def build() -> list[dict]:
    items = []
    for frame, (flat, prog, label) in NOTES.items():
        for cl, (w, tok) in enumerate(witnesses()[frame]):
            for gap, days in GAPS.items():
                written = TODAY - dt.timedelta(days=days)
                notes = {"FLAT": flat.format(w=w), "PROG": prog.format(w=w),
                         "BOUND": prog.format(w=w)[:-1] + " " + bound_phrase(written) + "."}
                for form, note in notes.items():
                    for task in ("A1", "A2"):
                        items.append({
                            "id": f"A-{frame}-{cl:02d}-{form}-{gap}-{task}",
                            "frame": frame, "cluster": cl, "form": form, "gap": gap,
                            "task": task, "witness": w, "wit_token": tok,
                            "written": written.isoformat(), "today": TODAY.isoformat(),
                            "note": note, "label": label,
                            "system": SYS.format(today=TODAY.isoformat(), written=written.isoformat(), note=note),
                            "query": Q_A1 if task == "A1" else Q_A2.format(label=label),
                        })
    return items


def parse(it: dict, text: str) -> dict:
    t = text.strip()
    if it["task"] == "A1":
        m = re.search(r"\b(100|\d{1,2})\b", t)
        return {"p": int(m.group(1)) if m else None,
                "parse": "OK" if m else "UNPARSED"}
    has_w = it["wit_token"].lower() in t.lower()
    has_u = re.search(r"\bunknown\b", t, re.I) is not None
    lab = ("COMMIT" if has_w and not has_u else "ABSTAIN" if has_u and not has_w
           else "MIXED" if has_w and has_u else "UNPARSED")
    return {"parse": lab}


def call(client: OpenAI, model: str, it: dict) -> tuple[str, dict]:
    msgs = [{"role": "system", "content": it["system"]},
            {"role": "user", "content": it["query"]}]
    direct = model in DIRECT
    delay = 5.0
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=DIRECT[model][2] if direct else model, messages=msgs,
                temperature=0, max_tokens=MAX_TOK[it["task"]],
                extra_body={} if direct else EXTRA_BODY)
            text = (r.choices[0].message.content or "").strip()
            d = r.model_dump()
            meta = {"provider": DIRECT[model][3] if direct else d.get("provider"),
                    "served_model": d.get("model"), "gen_id": d.get("id")}
            if text:
                return text, meta
            raise RuntimeError("empty content")
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 3
    raise RuntimeError("unreachable")


def run(reader: str, items: list[dict], limit: int | None, workers: int, suffix: str = "") -> None:
    tag = re.sub(r"[^a-z0-9]+", "-", reader.lower())
    out_path = HERE / f"trace{suffix}_{tag}.jsonl"
    done = set()
    if out_path.exists() and limit is None:
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()}
    todo = [it for it in items if it["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    if reader in DIRECT:
        base, keyenv, _, tag = DIRECT[reader][0], DIRECT[reader][1], None, tag + "-official"
        client = OpenAI(base_url=base, api_key=os.environ[keyenv])
        out_path = HERE / f"trace{suffix}_{tag}.jsonl"
        if out_path.exists() and limit is None:
            done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()}
            todo = [it for it in items if it["id"] not in done]
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
    out = open(out_path if limit is None else os.devnull, "a")

    def one(it: dict) -> dict:
        t0 = time.time()
        try:
            resp, meta = call(client, reader, it)
            err = None
        except Exception as e:  # noqa: BLE001
            resp, meta, err = "", {}, repr(e)
        rec = {k: it[k] for k in ("id", "frame", "cluster", "form", "gap", "task",
                                  "witness", "wit_token", "written", "today", "note")}
        rec.update(reader=reader, response=resp, error=err, **meta,
                   secs=round(time.time() - t0, 1), **parse(it, resp))
        return rec

    n = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            n += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if limit is not None or n % 50 == 0 or rec["error"]:
                print(f"[{tag}] {n}/{len(todo)} {rec['id']} -> {rec['response']!r} "
                      f"{rec.get('parse')}{' ERR ' + rec['error'] if rec['error'] else ''}",
                      file=sys.stderr)
    out.close()
    print(f"[{tag}] done {n} (skipped {len(done)} already present)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--readers", nargs="*", default=READERS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--task", choices=["A1", "A2"], help="restrict to one task")
    ap.add_argument("--suffix", default="", help="trace file suffix, e.g. _rep2 for a replicate")
    a = ap.parse_args()
    items = build()
    if a.task:
        items = [i for i in items if i["task"] == a.task]
    (HERE / "inputs.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    print(f"{len(items)} items/reader; {len(items) * len(a.readers)} calls total", file=sys.stderr)
    if a.build:
        return
    for r in a.readers:
        run(r, items, 4 if a.dry else None, a.workers, a.suffix)


if __name__ == "__main__":
    main()
