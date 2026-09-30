"""Part A analysis: per reader, A1 mean probability and A2 commit rate by
form x gap; controls; PROG-vs-FLAT contrasts (paired Wilcoxon on A1,
McNemar on A2), per-frame presentation. Descriptive tier."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon, binomtest

HERE = Path(__file__).resolve().parent
FORMS = ["FLAT", "PROG", "BOUND"]


def load(p: Path) -> dict:
    d = {}
    for l in p.read_text().splitlines():
        r = json.loads(l)
        d[r["id"]] = r
    return d


def key(r, form=None):
    return f"A-{r['frame']}-{r['cluster']:02d}-{form or r['form']}-{r['gap']}-{r['task']}"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return binomtest(b, n, 0.5).pvalue if n else float("nan")


def report(p: Path) -> None:
    recs = load(p)
    reader = next(iter(recs.values()))["reader"]
    errs = sum(1 for r in recs.values() if r["error"])
    unp = sum(1 for r in recs.values() if r["parse"] in ("UNPARSED", "MIXED"))
    print(f"\n=== {reader}: {len(recs)} rows, {errs} errors, {unp} UNPARSED/MIXED")

    # cell table
    print(f"{'gap':6} {'form':6} {'A1 mean':>8} {'A1 n':>5} {'A2 commit':>10} {'A2 abst':>8}")
    for gap in ("fresh", "stale"):
        for form in FORMS:
            a1 = [r["p"] for r in recs.values() if r["task"] == "A1" and r["gap"] == gap and r["form"] == form and r["p"] is not None]
            a2 = [r["parse"] for r in recs.values() if r["task"] == "A2" and r["gap"] == gap and r["form"] == form]
            n2 = len(a2)
            print(f"{gap:6} {form:6} {sum(a1)/len(a1) if a1 else float('nan'):8.1f} {len(a1):5d} "
                  f"{a2.count('COMMIT'):4d}/{n2:<4d} {a2.count('ABSTAIN'):4d}/{n2}")

    # controls
    print("controls: fresh FLAT should be high P / high commit; stale BOUND low P / >=50% abstain")

    # contrasts PROG vs FLAT, per gap, pooled and per frame
    for gap in ("fresh", "stale"):
        d_all, frames = [], defaultdict(list)
        b = c = 0
        fb, fc = defaultdict(int), defaultdict(int)
        for r in recs.values():
            if r["form"] != "PROG" or r["gap"] != gap:
                continue
            flat = recs.get(key(r, "FLAT"))
            if not flat:
                continue
            if r["task"] == "A1" and r["p"] is not None and flat["p"] is not None:
                d = r["p"] - flat["p"]
                d_all.append(d)
                frames[r["frame"]].append(d)
            if r["task"] == "A2" and {r["parse"], flat["parse"]} <= {"COMMIT", "ABSTAIN"}:
                if flat["parse"] == "COMMIT" and r["parse"] == "ABSTAIN":
                    b += 1; fb[r["frame"]] += 1
                elif flat["parse"] == "ABSTAIN" and r["parse"] == "COMMIT":
                    c += 1; fc[r["frame"]] += 1
        if d_all:
            nz = [d for d in d_all if d != 0]
            pw = wilcoxon(nz).pvalue if len(nz) >= 5 else float("nan")
            print(f"\n[{gap}] A1 PROG-FLAT: mean diff {sum(d_all)/len(d_all):+.1f} "
                  f"(n={len(d_all)}, nonzero={len(nz)}, PROG<FLAT in {sum(d<0 for d in d_all)}) Wilcoxon p={pw:.3g}")
            print("  per frame: " + ", ".join(f"{f} {sum(v)/len(v):+.0f}" for f, v in frames.items()))
        print(f"[{gap}] A2 discordant: FLAT commit/PROG abstain = {b}, FLAT abstain/PROG commit = {c}, McNemar p={mcnemar(b, c):.3g}")
        print("  per frame (b/c): " + ", ".join(f"{f} {fb[f]}/{fc[f]}" for f in sorted(set(fb) | set(fc))))

    # per-frame A2 commit by form (stale)
    print("\nstale A2 COMMIT by frame:")
    for frame in sorted({r["frame"] for r in recs.values()}):
        cells = []
        for form in FORMS:
            a2 = [r["parse"] for r in recs.values() if r["task"] == "A2" and r["gap"] == "stale" and r["form"] == form and r["frame"] == frame]
            cells.append(f"{form} {a2.count('COMMIT')}/{len(a2)}")
        print(f"  {frame:15} " + "  ".join(cells))


if __name__ == "__main__":
    paths = [p for p in sorted(HERE.glob("trace_*.jsonl"))
             if "rep2" not in p.name and "unpinned" not in p.name]
    for p in paths:
        report(p)
