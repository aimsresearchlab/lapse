#!/usr/bin/env python3
"""Predeclared metrics for the installed-mem0 scale-up (research/MEM0_SCALE_SPEC_2026-09-12.md). No model calls."""
import json, re, sys, glob, hashlib, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qualifier-ablation-2026-09-12")); sys.path.insert(0, str(HERE.parents[1] / "tools"))
import score_ablation as SA  # noqa: E402
import score as cascade  # noqa: E402
PROG = re.compile(r"\b(?:is|are|am|'s|'re|'m|has been|have been|been)\s+(?:currently\s+|now\s+)?\w+ing\b", re.I)
from math import comb
def cp(x, n, a=0.05):
    def cdf(p, k): return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))
    def bis(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0: hi = mid
            else: lo = mid
        return (lo + hi) / 2
    return (0.0 if x == 0 else bis(lambda p: 1 - cdf(p, x - 1) - a / 2), 1.0 if x == n else bis(lambda p: cdf(p, x) - a / 2))
rows = [json.loads(l) for f in sorted(glob.glob(str(HERE / "trace_shard*.jsonl"))) for l in open(f)]
def wit_mem(r): return "\n".join(m for m in r["memories"] if cascade.wit_re(r["wit_token"]).search(cascade.norm(m)))
out = {"n_rows": len(rows), "errors": sum(1 for r in rows if r["error"]), "forms": {}}
for form in ("prog", "simple", "simple_fornow", "simple_atm"):
    sub = [r for r in rows if r["form"] == form and not r["error"]]
    stored = [r for r in sub if wit_mem(r)]
    d = {"n": len(sub), "witness_stored": len(stored), "per_frame": {}}
    if form == "prog":
        flat = [r for r in stored if not PROG.search(wit_mem(r))]
        d["flattened"] = len(flat); d["flattened_ci95"] = cp(len(flat), len(stored)) if stored else None; d["flattened_rate"] = len(flat) / len(stored) if stored else None
        for fr in sorted({r["frame"] for r in sub}):
            s = [r for r in stored if r["frame"] == fr]; d["per_frame"][fr] = {"flattened": sum(1 for r in s if not PROG.search(wit_mem(r))), "n": len(s)}
        d["examples_flattened"] = [(r["text"][:90], wit_mem(r)[:120]) for r in flat[:6]]
    elif form == "simple":
        man = [r for r in stored if PROG.search(wit_mem(r)) or cascade.CURRENCY.search(wit_mem(r))]
        d["manufacture"] = len(man); d["manufacture_ci95"] = cp(len(man), len(stored)) if stored else None; d["examples"] = [wit_mem(r)[:120] for r in man[:6]]
    else:
        labs = collections.Counter(); ql = 0
        for r in sub:
            lab = SA.marker_label("\n".join(r["memories"]), r["wit_token"], form); labs[lab] += 1; r["marker_label"] = lab
            ql += lab in ("DROPPED", "DATE-ONLY")
        d["labels"] = dict(labs); nn = sum(v for k, v in labs.items() if k != "WITNESS-DROPPED")
        d["qual_lost"] = ql; d["qual_lost_rate"] = ql / nn if nn else None; d["qual_lost_ci95"] = cp(ql, nn) if nn else None
        for fr in sorted({r["frame"] for r in sub}):
            s = [r for r in sub if r["frame"] == fr]; d["per_frame"][fr] = {"qual_lost": sum(r["marker_label"] in ("DROPPED", "DATE-ONLY") for r in s), "n": len(s)}
        d["examples_kept"] = [wit_mem(r)[:120] for r in sub if r["marker_label"].startswith("MARKER")][:6]
        d["examples_lost"] = [wit_mem(r)[:120] for r in sub if r["marker_label"] in ("DROPPED", "DATE-ONLY")][:6]
        d["residual"] = [wit_mem(r)[:160] for r in sub if r["marker_label"] == "RESIDUAL"]
    d["date_anchor"] = sum(1 for r in stored if cascade.DATE_ANCHOR.search(wit_mem(r)) or re.search(r"\bas of\b", wit_mem(r), re.I))
    out["forms"][form] = d
# overlap with 08-27 run (clusters 0-3, prog)
old = {}
for l in open(HERE.parent / "mem0-pipeline-2026-08-27" / "mem0_pipeline_trace.jsonl"):
    r = json.loads(l)
    if r["form"] == "prog": old[(r["frame"], int(r["cluster"]))] = not PROG.search("\n".join(m for m in r["memories_after_add"] if r["wit_token"].lower() in m.lower()))
agree = tot = 0; old_flat = new_flat = 0
for r in rows:
    if r["form"] == "prog" and (r["frame"], r["cluster"]) in old and not r["error"]:
        nf = not PROG.search(wit_mem(r)) if wit_mem(r) else None
        if nf is None: continue
        tot += 1; agree += nf == old[(r["frame"], r["cluster"])]; old_flat += old[(r["frame"], r["cluster"])]; new_flat += nf
out["overlap_with_0827"] = {"n": tot, "agree": agree, "old_flattened": old_flat, "new_flattened": new_flat}
out["hashes"] = {Path(f).name: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(str(HERE / "trace_shard*.jsonl")))}
(HERE / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
for form, d in out["forms"].items():
    print(f"## {form}: n={d['n']} witness_stored={d['witness_stored']} date_anchor={d['date_anchor']}")
    for k in ("flattened", "flattened_rate", "flattened_ci95", "manufacture", "manufacture_ci95", "labels", "qual_lost", "qual_lost_rate", "qual_lost_ci95", "per_frame"):
        if k in d: print("   ", k, d[k])
    for k in ("examples_flattened", "examples", "examples_kept", "examples_lost", "residual"):
        if k in d: print("   ", k, d[k][:4])
print("overlap with 08-27:", out["overlap_with_0827"]); print("errors:", out["errors"])
