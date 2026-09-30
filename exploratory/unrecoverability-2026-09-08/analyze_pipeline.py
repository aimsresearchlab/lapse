"""Unrecoverability arm analysis: predictions P2-P6 per reader."""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent

def mcn(b, c): return binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")

def report(p):
    R = {}
    for l in p.read_text().splitlines():
        r = json.loads(l); R[r["id"]] = r
    reader = next(iter(R.values()))["reader"]
    errs = sum(1 for r in R.values() if r["error"]); unp = sum(1 for r in R.values() if r["parse"] in ("UNPARSED", "MIXED"))
    print(f"\n=== {reader}: {len(R)} rows, {errs} errors, {unp} UNPARSED/MIXED")
    def cell(arm, task, sub=None):
        return [r for r in R.values() if r["arm"] == arm and r["task"] == task and (sub is None or r["writer_label"] == sub)]
    print(f"{'arm':14}{'A1 mean':>8}{'commit':>8}{'abstain':>8}{'n':>5}")
    for arm in ("PIPE-SIMPLE", "PIPE-PROG", "PIPE-BOUND", "EDIT-PROG", "EDIT-FLAT", "POLICY-SIMPLE", "POLICY-PROG"):
        a1 = [r["p"] for r in cell(arm, "A1") if r["p"] is not None]; a2 = [r["parse"] for r in cell(arm, "A2")]
        print(f"{arm:14}{(sum(a1)/len(a1) if a1 else float('nan')):8.1f}{a2.count('COMMIT'):8d}{a2.count('ABSTAIN'):8d}{len(a2):5d}")
    def contrast(armA, armB, sub, name):
        b = c = 0; d = []
        for r in cell(armA, "A2", sub):
            o = R.get(r["id"].replace(armA, armB))
            if not o: continue
            if r["parse"] == "ABSTAIN" and o["parse"] == "COMMIT": b += 1
            elif r["parse"] == "COMMIT" and o["parse"] == "ABSTAIN": c += 1
        for r in cell(armA, "A1", sub):
            o = R.get(r["id"].replace(armA, armB))
            if o and r["p"] is not None and o["p"] is not None: d.append(r["p"] - o["p"])
        n = len(cell(armA, "A2", sub))
        ca = sum(r["parse"] == "COMMIT" for r in cell(armA, "A2", sub)); cb = sum(r["parse"] == "COMMIT" for r in cell(armB, "A2", sub))
        nz = [x for x in d if x]
        pw = wilcoxon(nz).pvalue if len(nz) >= 5 else float("nan")
        print(f"  {name}: n={n} commit {armA} {ca} vs {armB} {cb} | {armA}-abstain/{armB}-commit={b}, reverse={c}, McNemar p={mcn(b,c):.3g} | A1 diff {sum(d)/len(d) if d else float('nan'):+.1f} p={pw:.2g}")
    print("P2 PRESERVED subset:"); contrast("PIPE-PROG", "PIPE-SIMPLE", "PRESERVED", "PIPE-PROG vs PIPE-SIMPLE")
    print("P3 FLATTENED subset:"); contrast("PIPE-PROG", "PIPE-SIMPLE", "FLATTENED", "PIPE-PROG vs PIPE-SIMPLE")
    print("P4 EDIT-PROG vs PIPE-SIMPLE, FLATTENED frames:"); contrast("EDIT-PROG", "PIPE-SIMPLE", "FLATTENED", "restored form on flattened frames")
    print("   EDIT-FLAT vs PIPE-PROG, PRESERVED frames:"); contrast("EDIT-FLAT", "PIPE-PROG", "PRESERVED", "hand-flattened vs kept")
    print("P5 POLICY:")
    for sub in ("PRESERVED", "FLATTENED"):
        for arm in ("POLICY-PROG", "POLICY-SIMPLE"):
            a2 = [r["parse"] for r in cell(arm, "A2", sub)]
            print(f"  {sub:9} {arm:13} abstain {a2.count('ABSTAIN')}/{len(a2)}")
    a2 = [r["parse"] for r in cell("PIPE-BOUND", "A2")]
    print(f"P6 PIPE-BOUND abstain {a2.count('ABSTAIN')}/{len(a2)}")
    print("per-frame PIPE-PROG vs PIPE-SIMPLE commit (A2):")
    for fr in sorted({r["frame"] for r in R.values()}):
        pp = [r for r in cell("PIPE-PROG", "A2") if r["frame"] == fr]; ps = [r for r in cell("PIPE-SIMPLE", "A2") if r["frame"] == fr]
        lab = pp[0]["writer_label"] if pp else "?"
        print(f"  {fr:15} {sum(r['parse']=='COMMIT' for r in pp)}/{len(pp)} vs {sum(r['parse']=='COMMIT' for r in ps)}/{len(ps)}  (writer mostly {lab})")

if __name__ == "__main__":
    for p in sorted(HERE.glob("reader_trace_*.jsonl")): report(p)
