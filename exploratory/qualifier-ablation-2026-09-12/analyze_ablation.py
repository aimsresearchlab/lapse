#!/usr/bin/env python3
"""Predeclared analysis for the qualifier ablation (spec: research/QUALIFIER_ABLATION_SPEC_2026-09-12.md)."""
from __future__ import annotations
import json, sys, collections, hashlib
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = {"COERCED-STATIVE", "MANUFACTURE"}; EXCL = {"RESIDUAL", "WITNESS-DROPPED", "ERROR"}
QL = {"DROPPED", "DATE-ONLY"}; MEXCL = {"RESIDUAL", "WITNESS-DROPPED", "ERROR"}

def binom_ge(b, n): return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n if n else 1.0
def two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)
def cp(x, n, a=0.05):
    from math import log
    # exact Clopper-Pearson via bisection on binomial tails
    def cdf(p, k): return sum(comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))
    lo = 0.0 if x == 0 else _bis(lambda p: 1 - cdf(p, x-1) - a/2, 0, 1)
    hi = 1.0 if x == n else _bis(lambda p: cdf(p, x) - a/2, 0, 1)
    return lo, hi
def _bis(f, lo, hi):
    for _ in range(60):
        mid = (lo+hi)/2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return (lo+hi)/2
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); m = len(ps); run = 0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i]); out[i] = min(1.0, run)
    return out

TRACE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "trace.scored.jsonl"
rows = [json.loads(l) for l in TRACE.read_text().splitlines()]
models = sorted({r["model"] for r in rows})
stim = {}
for l in (ROOT / "data" / "stimuli_v2.jsonl").read_text().splitlines():
    s = json.loads(l); stim[s["case_id"]] = s
frozen = {}
for base in ("run-v2-full-20260825-010312", "run-v2-full-20260825-010323", "run-v2-full-20260825-234313"):
    for l in (ROOT / "traces" / f"{base}.scored.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["arm"] == "e1" and r["component"] == "e1_primary" and r["carrier"] == "c0_original" and r["gap"] == "stale" and r["form"] in ("prog", "simple"):
            frozen[(r["model"], r["frame"], r["cluster"], r["form"])] = r["label"]
out = {"models": {}, "tests": []}
ps, tests = [], []
for m in models:
    cell = collections.defaultdict(dict)
    for r in rows:
        if r["model"] == m: cell[(r["frame"], r["cluster"])][r["form"]] = r["label"]
    lab = collections.Counter((r["form"], r["label"]) for r in rows if r["model"] == m)
    # (1) replication prog vs simple
    b = c = n = 0
    for k, v in cell.items():
        if v.get("prog") in EXCL or v.get("simple") in EXCL or "prog" not in v or "simple" not in v: continue
        n += 1; pd, sd = v["prog"] in D, v["simple"] in D
        b += pd and not sd; c += sd and not pd
    fb = fc = fn = 0
    for (fm, fr, cl, fo), l in frozen.items():
        pass
    fkeys = {(fr, cl) for (fm, fr, cl, fo) in frozen if fm == m}
    for (fr, cl) in fkeys:
        lp, ls = frozen.get((m, fr, cl, "prog")), frozen.get((m, fr, cl, "simple"))
        if lp is None or ls is None or lp in EXCL or ls in EXCL: continue
        fn += 1; fb += (lp in D) and not (ls in D); fc += (ls in D) and not (lp in D)
    rep = {"n": n, "b_prog_only": b, "c_simple_only": c, "p_one_sided": binom_ge(b, b + c),
           "frozen_run": {"n": fn, "b": fb, "c": fc, "p_one_sided": binom_ge(fb, fb + fc)}}
    # (2) ablation per marker
    abl = {}
    for form in ("simple_fornow", "simple_atm"):
        bb = cc = nn = 0; lost = tot = 0
        for k, v in cell.items():
            if v.get("prog") in EXCL or v.get(form) in MEXCL or "prog" not in v or form not in v: continue
            nn += 1; pd, ql = v["prog"] in D, v[form] in QL
            bb += pd and not ql; cc += ql and not pd
        for k, v in cell.items():
            if v.get(form) in MEXCL or form not in v: continue
            tot += 1; lost += v[form] in QL
        pdn = sum(1 for v in cell.values() if v.get("prog") not in EXCL and "prog" in v); pdd = sum(1 for v in cell.values() if v.get("prog") in D)
        p = two_sided(bb, cc); ps.append(p); tests.append((m, form))
        abl[form] = {"n_pairs": nn, "prog_destroyed_not_marker_lost": bb, "marker_lost_not_prog_destroyed": cc, "p_two_sided": p,
                     "qual_lost_rate": {"x": lost, "n": tot, "rate": lost / tot if tot else None, "ci95": cp(lost, tot) if tot else None},
                     "prog_destroyed_rate": {"x": pdd, "n": pdn, "rate": pdd / pdn if pdn else None, "ci95": cp(pdd, pdn) if pdn else None},
                     "labels": {l: lab[(form, l)] for l in sorted({k[1] for k in lab if k[0] == form})},
                     "per_frame": {}}
        for fr in sorted({k[0] for k in cell}):
            sub = [v for k, v in cell.items() if k[0] == fr]
            abl[form]["per_frame"][fr] = {"qual_lost": sum(v.get(form) in QL for v in sub), "prog_destroyed": sum(v.get("prog") in D for v in sub), "n": len(sub)}
    placebo = collections.Counter()
    for r in rows:
        if r["model"] == m and r.get("placebo_marker"):
            placebo[(r["form"], "T" if r["placebo_marker"]["T"] else "", "C" if r["placebo_marker"]["C"] else "")] += 1
    out["models"][m] = {"labels": {f"{f}|{l}": v for (f, l), v in sorted(lab.items())}, "replication_prog_vs_simple": rep, "ablation": abl,
                        "placebo_marker_hits_on_unmarked_inputs": {"|".join(k): v for k, v in placebo.items()},
                        "errors": sum(1 for r in rows if r["model"] == m and r["label"] == "ERROR")}
hp = holm(ps)
for (m, f), p, h in zip(tests, ps, hp):
    a = out["models"][m]["ablation"][f]; a["holm_p_family6"] = h
    diff = a["prog_destroyed_rate"]["rate"] - a["qual_lost_rate"]["rate"]
    a["rate_difference_prog_minus_marker"] = diff
    a["reading"] = ("aspect-specific" if (h < .05 and diff > .20) else "general-qualifier-loss" if (diff < .10 or h >= .05) and diff < .20 else "intermediate")
out["input_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (TRACE, HERE / "stimuli.jsonl", HERE / "manifest.json")}
(TRACE.with_name(TRACE.name.replace(".scored.jsonl", "") + "_results.json") if TRACE.name != "trace.scored.jsonl" else HERE / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True))
for m in models:
    M = out["models"][m]; r = M["replication_prog_vs_simple"]
    print(f"\n## {m}  errors={M['errors']}")
    print(f"  replication prog>simple: n={r['n']} b={r['b_prog_only']} c={r['c_simple_only']} p={r['p_one_sided']:.2e} | frozen run: n={r['frozen_run']['n']} b={r['frozen_run']['b']} c={r['frozen_run']['c']}")
    for f, a in M["ablation"].items():
        q, pdr = a["qual_lost_rate"], a["prog_destroyed_rate"]
        print(f"  {f:13s} QUAL-LOST {q['x']}/{q['n']} ({q['rate']:.2f}, CI {q['ci95'][0]:.2f}-{q['ci95'][1]:.2f}) vs prog D {pdr['x']}/{pdr['n']} ({pdr['rate']:.2f}) | paired b={a['prog_destroyed_not_marker_lost']} c={a['marker_lost_not_prog_destroyed']} p={a['p_two_sided']:.2e} Holm={a['holm_p_family6']:.2e} diff={a['rate_difference_prog_minus_marker']:+.2f} -> {a['reading']}")
        print(f"      labels: {a['labels']}")
        print("      per-frame qual_lost/prog_destroyed/n:", {k: (v['qual_lost'], v['prog_destroyed'], v['n']) for k, v in a['per_frame'].items()})
    print("  placebo marker hits on prog/simple outputs:", M["placebo_marker_hits_on_unmarked_inputs"])
