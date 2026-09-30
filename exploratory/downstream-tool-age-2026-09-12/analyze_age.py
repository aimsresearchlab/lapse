#!/usr/bin/env python3
"""Predeclared analysis for the note-age run (research/DOWNSTREAM_TOOL_AGE_SPEC_2026-09-12.md)."""
import json, hashlib, collections
from math import comb
from pathlib import Path
HERE = Path(__file__).resolve().parent; R2 = HERE.parent / "downstream-tool-2026-09-11"
def two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(idx): run = max(run, (len(ps) - r) * ps[i]); out[i] = min(1.0, run)
    return out
def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines()]
rows = load(HERE / "trace_age_targets.jsonl")
for r in load(R2 / "trace_target_r2.jsonl"): r["item"]["age_tag"] = "M8"; rows.append(r)
cells = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    it = r["item"]; cells[(r["configured_model"], it["age_tag"])][it["pair_id"].replace("-M1", "").replace("-M3", "")][it["form"]] = (r["score"], it["frame"])
res = {}; tests = []; ps = []
for (m, age), pairs in sorted(cells.items()):
    ex = {"simple": 0, "progressive": 0}; n = {"simple": 0, "progressive": 0}; unp = {"simple": 0, "progressive": 0}
    b = c = comp = 0; per_frame = collections.defaultdict(lambda: [0, 0, 0])
    for pid, d in pairs.items():
        for f in ("simple", "progressive"):
            if f in d:
                n[f] += 1; ex[f] += d[f][0] == "EXECUTE"; unp[f] += d[f][0] == "UNPARSED"
        if "simple" in d and "progressive" in d and d["simple"][0] != "UNPARSED" and d["progressive"][0] != "UNPARSED":
            comp += 1; s, p = d["simple"][0] == "EXECUTE", d["progressive"][0] == "EXECUTE"
            b += s and not p; c += p and not s
            fr = d["simple"][1]; per_frame[fr][0] += s; per_frame[fr][1] += p; per_frame[fr][2] += 1
    p = two_sided(b, c)
    rate = {f: ex[f] / n[f] for f in n}
    res[f"{m}|{age}"] = {"reader": m, "age": age, "n_pairs": len(pairs), "complete_pairs": comp, "execute": ex, "n": n, "unparsed": unp, "rate": rate,
                         "flat_only": b, "prog_only": c, "p_two_sided": p, "risk_difference": (b - c) / comp if comp else None,
                         "saturation": [f for f in n if rate[f] < .10 or rate[f] > .90], "per_frame": {k: {"flat_exec": v[0], "prog_exec": v[1], "n": v[2]} for k, v in sorted(per_frame.items())}}
    if age != "M8": tests.append(f"{m}|{age}"); ps.append(p)
for k, h in zip(tests, holm(ps)): res[k]["holm_p_family6"] = h
out = {"cells": res, "hashes": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in ("trace_age_targets.jsonl", "trace_age_controls.jsonl", "age_targets.jsonl")},
       "r2_target_trace_sha256": hashlib.sha256((R2 / "trace_target_r2.jsonl").read_bytes()).hexdigest()}
(HERE / "results_age.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print(f"{'reader':36s} {'age':4s} {'flat EXEC':>10s} {'prog EXEC':>10s} {'b:c':>7s} {'p':>9s} {'Holm':>9s} {'RD':>6s} sat")
for k in sorted(res, key=lambda k: (res[k]['reader'], {'M1': 1, 'M3': 3, 'M8': 8}[res[k]['age']])):
    r = res[k]
    print(f"{r['reader']:36s} {r['age']:4s} {r['execute']['simple']:3d}/{r['n']['simple']:<6d} {r['execute']['progressive']:3d}/{r['n']['progressive']:<6d} {r['flat_only']:3d}:{r['prog_only']:<3d} {r['p_two_sided']:9.2e} {r.get('holm_p_family6', float('nan')):9.2e} {r['risk_difference'] if r['risk_difference'] is None else round(r['risk_difference'], 3)!s:>6s} {r['saturation']}")
    if r['age'] != 'M8': print("      per-frame flat/prog/n:", {f: (v['flat_exec'], v['prog_exec'], v['n']) for f, v in r['per_frame'].items()})
