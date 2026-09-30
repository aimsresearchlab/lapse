"""Reader step of research/UNRECOVERABILITY_SPEC_2026-09-08.md.

Builds reader items from writer_trace_deepseek_official.jsonl (mem0 output)
and runs them through the Part A reader format (run_ladder.call/parse).

    python read_pipeline.py --build          # items + edit check, no calls
    python read_pipeline.py --dry            # 4 calls per reader
    python read_pipeline.py --full [--readers ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "read-time-ladder-2026-09-08"))
from run_ladder import (DIRECT, NOTES, READERS, SYS, Q_A1, Q_A2, call, parse)  # noqa: E402
from openai import OpenAI  # noqa: E402

TODAY = "2026-09-08"
WRITTEN = "2026-01-06"
POLICY = (" If the note describes an ongoing or temporary situation rather "
          "than a settled fact, write UNKNOWN.")

# progressive <-> simple verb table per frame (mem0 writes third person)
VERBS = {
    "lodging":        [("is staying at", "lives at"), ("is currently staying at", "lives at"), ("is living at", "lives at")],
    "workplace":      [("is working at", "works at"), ("is currently working at", "works at"), ("is employed at", "works at")],
    "vehicle":        [("is driving a", "drives a"), ("is currently driving a", "drives a"), ("is driving an", "drives an"), ("is renting a", "rents a"), ("is driving", "drives")],
    "class":          [("is taking a", "takes a"), ("is currently taking a", "takes a"), ("is attending a", "attends a"), ("is enrolled in a", "takes a")],
    "equipment":      [("is using a", "uses a"), ("is currently using a", "uses a"), ("is playing a", "plays a"), ("is borrowing a", "borrows a"), ("is borrowing an", "borrows an")],
    "affiliate_role": [("is lecturing at", "lectures at"), ("is currently lecturing at", "lectures at")],
    "project":        [("is working on", "works on"), ("is currently working on", "works on")],
    "household":      [("is staying with", "lives with"), ("is currently staying with", "lives with"), ("is living with", "lives with")],
}
PROG_RE = re.compile(r"\b(is|are|has been|have been)\s+(currently\s+)?\w+ing\b", re.I)
TEMP_RE = re.compile(r"\b(currently|temporar|for now|at the moment|on loan|until|as of)\b", re.I)


def label_writer(text: str) -> str:
    """PRESERVED if the stored line keeps a progressive or an explicit ongoing marker."""
    if PROG_RE.search(text):
        return "PRESERVED"
    if re.search(r"\b(currently|temporarily|for now|at the moment)\b", text, re.I):
        return "PRESERVED"
    return "FLATTENED"


def edit(text: str, frame: str, to: str) -> str | None:
    """to='FLAT': progressive -> simple; to='PROG': simple -> progressive. None if no rule matched."""
    for prog, simple in VERBS[frame]:
        if to == "FLAT":
            m = re.search(re.escape(prog), text, re.I)
            if m:
                return text[:m.start()] + simple + text[m.end():]
        else:
            m = re.search(r"\b" + re.escape(simple), text, re.I)
            if m:
                return text[:m.start()] + prog + text[m.end():]
    return None


def build() -> tuple[list[dict], dict]:
    rows = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "writer_trace_deepseek_official.jsonl").read_text().splitlines()}
    items, stats = [], {"dropped": [], "edit_fail": []}
    for frame in NOTES:
        for cl in range(12):
            base = {}
            for form in ("PROG", "SIMPLE", "BOUND"):
                r = rows.get(f"U-{frame}-{cl:02d}-{form}")
                if not r or not r["witness_memory"]:
                    stats["dropped"].append(f"U-{frame}-{cl:02d}-{form}")
                    continue
                base[form] = r
            if not base:
                continue
            w, tok = next(iter(base.values()))["witness"], next(iter(base.values()))["wit_token"]
            notes = {}
            for form, arm in (("PROG", "PIPE-PROG"), ("SIMPLE", "PIPE-SIMPLE"), ("BOUND", "PIPE-BOUND")):
                if form in base:
                    notes[arm] = base[form]["witness_memory"]
            if "SIMPLE" in base:
                e = edit(base["SIMPLE"]["witness_memory"], frame, "PROG")
                if e: notes["EDIT-PROG"] = e
                else: stats["edit_fail"].append(("EDIT-PROG", base["SIMPLE"]["witness_memory"]))
            if "PROG" in base and label_writer(base["PROG"]["witness_memory"]) == "PRESERVED":
                # EDIT-FLAT only makes sense where the pipeline kept the form
                e = edit(base["PROG"]["witness_memory"], frame, "FLAT")
                if e: notes["EDIT-FLAT"] = e
                else: stats["edit_fail"].append(("EDIT-FLAT", base["PROG"]["witness_memory"]))
            wl = label_writer(base["PROG"]["witness_memory"]) if "PROG" in base else None
            label = NOTES[frame][2]
            for arm, note in notes.items():
                for task in ("A1", "A2"):
                    items.append({"id": f"R-{frame}-{cl:02d}-{arm}-{task}", "frame": frame, "cluster": cl,
                                  "arm": arm, "task": task, "witness": w, "wit_token": tok,
                                  "writer_label": wl, "written": WRITTEN, "today": TODAY, "note": note,
                                  "system": SYS.format(today=TODAY, written=WRITTEN, note=note),
                                  "query": Q_A1 if task == "A1" else Q_A2.format(label=label)})
            for arm in ("PIPE-PROG", "PIPE-SIMPLE"):
                if arm in notes:
                    q = Q_A2.format(label=label).replace(" Reply with the field value only.", POLICY + " Reply with the field value only.")
                    items.append({"id": f"R-{frame}-{cl:02d}-POLICY-{arm[5:]}-A2", "frame": frame, "cluster": cl,
                                  "arm": "POLICY-" + arm[5:], "task": "A2", "witness": w, "wit_token": tok,
                                  "writer_label": wl, "written": WRITTEN, "today": TODAY, "note": notes[arm],
                                  "system": SYS.format(today=TODAY, written=WRITTEN, note=notes[arm]), "query": q})
    return items, stats


def run(reader: str, items: list[dict], limit: int | None, workers: int) -> None:
    tag = re.sub(r"[^a-z0-9]+", "-", reader.lower())
    if reader in DIRECT:
        base, keyenv, _, _ = DIRECT[reader]
        client, tag = OpenAI(base_url=base, api_key=os.environ[keyenv]), tag + "-official"
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out_path = HERE / f"reader_trace_{tag}.jsonl"
    done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()} if out_path.exists() and limit is None else set()
    todo = [it for it in items if it["id"] not in done][: limit or None]
    out = open(out_path if limit is None else os.devnull, "a")

    def one(it):
        t0 = time.time()
        try:
            resp, meta = call(client, reader, it); err = None
        except Exception as e:  # noqa: BLE001
            resp, meta, err = "", {}, repr(e)
        rec = {k: it[k] for k in ("id", "frame", "cluster", "arm", "task", "witness", "wit_token", "writer_label", "note")}
        rec.update(reader=reader, response=resp, error=err, **meta, secs=round(time.time() - t0, 1), **parse(it, resp))
        return rec

    n = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            n += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if limit or n % 100 == 0 or rec["error"]:
                print(f"[{tag}] {n}/{len(todo)} {rec['id']} -> {rec['response'][:60]!r} {rec.get('parse')}{' ERR ' + rec['error'] if rec['error'] else ''}", file=sys.stderr)
    out.close()
    print(f"[{tag}] done {n} (skipped {len(done)})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--readers", nargs="*", default=READERS); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items, stats = build()
    (HERE / "reader_inputs.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    print(f"{len(items)} items/reader; dropped {len(stats['dropped'])} {stats['dropped'][:8]}; edit failures {len(stats['edit_fail'])}", file=sys.stderr)
    for ef in stats["edit_fail"]:
        print("  EDIT FAIL", ef, file=sys.stderr)
    if a.build:
        return
    for r in a.readers:
        run(r, items, 4 if a.dry else None, a.workers)


if __name__ == "__main__":
    main()
