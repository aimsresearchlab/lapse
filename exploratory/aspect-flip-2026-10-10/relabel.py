"""Relabel the scored memory-write traces under the aspect-flip definition ($0, no API calls).

Spec/record: exploratory/stative-only-recount-2026-10-10/RESULTS.md (why) and README.md here.
For every memory-write row (arm e1, or l2 = guided memory write) with form prog, perf_prog or
simple (the main test, carriers, same-lexeme, frame, instruction and perfect-form analyses):
  - RESIDUAL and WITNESS-DROPPED stay as they are (same exclusions, same n).
  - prog, perf_prog: COERCED-STATIVE stays (progressive rewritten as a simple-present state);
            every other label becomes PRESERVED (MANUFACTURE no longer counts).
  - simple: becomes COERCED-STATIVE if the witness line gains a progressive
            (tools/score.py PROGRESSIVE or PERF_PROG, dated or not), else PRESERVED.
The frozen label is kept in `label_frozen`. Re-using the COERCED-STATIVE name lets every frozen
analysis script (DESTROYED = COERCED-STATIVE or MANUFACTURE) compute the aspect-flip count
unchanged. All other rows are copied unchanged.

Output: root/traces/<base>.scored.jsonl, with root/data, root/tools, root/exploratory linked to
the repo so the frozen scripts can run with ROOT pointed here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
import score as SC  # noqa: E402

EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
OUT = HERE / "root"


def gains_progressive(rec: dict) -> bool:
    t = SC.norm(rec["response"])
    ln = " ".join(l for l in t.splitlines() if SC.wit_re(rec["wit_token"]).search(l))
    return bool(SC.PROGRESSIVE.search(ln) or SC.PERF_PROG.search(ln))


def relabel(r: dict) -> dict:
    if r.get("arm") not in ("e1", "l2") or r.get("form") not in ("prog", "perf_prog", "simple") \
            or r["label"] in EXCLUDED:
        return r
    r = dict(r, label_frozen=r["label"])
    if r["form"] in ("prog", "perf_prog"):
        r["label"] = "COERCED-STATIVE" if r["label_frozen"] == "COERCED-STATIVE" else "PRESERVED"
    else:
        r["label"] = "COERCED-STATIVE" if gains_progressive(r) else "PRESERVED"
    return r


def main() -> None:
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    for name in ("data", "tools", "exploratory"):
        link = OUT / name
        if not link.exists():
            os.symlink(os.path.relpath(REPO / name, OUT), link)
    n = 0
    for p in sorted((REPO / "traces").glob("*.scored.jsonl")):
        rows = [relabel(json.loads(l)) for l in p.read_text().splitlines() if l.strip()]
        (OUT / "traces" / p.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        n += 1
    print(f"relabeled {n} traces -> {OUT / 'traces'}")


if __name__ == "__main__":
    main()
