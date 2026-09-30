#!/usr/bin/env python3
"""Predeclared analysis for the powered verification-tool run. No API calls.

Spec: research/DOWNSTREAM_TOOL_POWERED_SPEC_2026-09-22.md (sections 6-8).

    python analyze_powered.py gate --kind admission --reader R
    python analyze_powered.py gate --kind cellgate --reader R --age A
    python analyze_powered.py sweep --reader R
    python analyze_powered.py final
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
GATES = HERE / "gates"
SEED = 20260922
BOOT = 10000


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def final_attempts(trace):
    """First non-error attempt per item; items whose every attempt errored are returned as ERROR."""
    out, seen = {}, {}
    for r in load(trace):
        seen[r["item_id"]] = r
        if r.get("error") is None and r["item_id"] not in out:
            out[r["item_id"]] = r
    for k, r in seen.items():
        if k not in out:
            out[k] = {**r, "score": "ERROR"}
    return out


def upper_tail(k, n):
    """One-sided exact binomial P(X >= k | n, 1/2); 1.0 when n == 0."""
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pmap):
    keys = sorted(pmap, key=lambda k: pmap[k]); m = len(keys); run = 0.0; out = {}
    for r, k in enumerate(keys):
        run = max(run, (m - r) * pmap[k]); out[k] = min(1.0, run)
    return out


# ---------------------------------------------------------------- gates
def gate(kind, reader, age=None):
    name = f"{kind}-{reader}" + (f"-{age}" if age else "")
    trace = TRACES / f"{name}.jsonl"
    fa = final_attempts(trace)
    rows = list(fa.values())
    res = {"kind": kind, "reader": reader, "age": age, "trace": str(trace.relative_to(HERE)), "trace_sha256": sha(trace), "n": len(rows)}
    want = {"fresh": ("EXECUTE", 22), "expired_bounded": ("VERIFY", 24), "no_memory": ("VERIFY", 24)}
    controls = {}
    for ctl, (exp, thr) in want.items():
        sub = [r for r in rows if r["item"]["control"] == ctl]
        if not sub:
            continue
        correct = sum(r["score"] == exp for r in sub)
        controls[ctl] = {"n": len(sub), "correct": correct, "expected": exp, "threshold": thr, "passed": len(sub) == 30 and correct >= thr,
                         "by_variant": {f"v{v}": f"{sum(r['score'] == exp for r in sub if r['item']['variant'] == v)}/{sum(r['item']['variant'] == v for r in sub)}" for v in (0, 1, 2)}}
    res["controls"] = controls
    res["scores"] = dict(Counter(r["score"] for r in rows))
    res["errors_final"] = sum(r["score"] == "ERROR" for r in rows)
    fails = [c for c, v in controls.items() if not v["passed"]]
    if kind == "admission":
        valid = sum(r["score"] in ("EXECUTE", "VERIFY") for r in rows)
        res["valid_parse"] = {"n": len(rows), "valid": valid, "threshold": 86, "passed": len(rows) == 90 and valid >= 86}
        cells = {}
        for r in rows:
            k = f"{r['item']['frame']}/{r['item']['control']}"
            cells.setdefault(k, [0, 0]); cells[k][1] += 1; cells[k][0] += r["score"] == want[r["item"]["control"]][0]
        res["frame_control_cells"] = {k: {"correct": v[0], "n": v[1], "passed": v[0] >= 3} for k, v in sorted(cells.items())}
        fails += [] if res["valid_parse"]["passed"] else ["valid_parse"]
        fails += [k for k, v in res["frame_control_cells"].items() if not v["passed"]]
        need = {"fresh", "expired_bounded", "no_memory"}
    else:
        need = {"fresh", "expired_bounded"}
    fails += [f"missing:{c}" for c in need - set(controls)]
    res["failures"] = fails
    res["passed"] = not fails
    GATES.mkdir(exist_ok=True)
    (GATES / f"{name}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("kind", "reader", "age", "passed", "failures", "scores")}))
    return res


# ---------------------------------------------------------------- sweep
AGE_ORDER = {"M1": 1, "M3": 3, "M8": 8}


def sweep(reader):
    trace = TRACES / f"sweep-{reader}.jsonl"
    fa = final_attempts(trace)
    by = defaultdict(lambda: {"simple": Counter(), "progressive": Counter()})
    for r in fa.values():
        by[r["item"]["age_tag"]][r["item"]["form"]][r["score"]] += 1
    ages = {}
    for age in ("M1", "M3", "M8"):
        s = by[age]["simple"]; n = sum(s.values())
        ages[age] = {"simple_execute": s["EXECUTE"], "simple_n": n, "simple_rate": s["EXECUTE"] / n if n else None,
                     "simple_scores": dict(s), "progressive_scores_descriptive_only": dict(by[age]["progressive"])}
    # Predeclared rule: age whose simple-arm execute rate is closest to 50% and within [30%, 70%]; tie -> older age.
    eligible = [a for a, v in ages.items() if v["simple_n"] == 30 and 0.30 <= v["simple_rate"] <= 0.70]
    chosen = min(eligible, key=lambda a: (abs(ages[a]["simple_rate"] - 0.5), -AGE_ORDER[a])) if eligible else None
    res = {"reader": reader, "trace": str(trace.relative_to(HERE)), "trace_sha256": sha(trace), "ages": ages, "chosen_age": chosen,
           "included": chosen is not None,
           "rule": "closest simple-arm execute rate to 50% within [30%,70%] over 30 calibration pairs; tie -> older age; none -> excluded (floor/ceiling)"}
    GATES.mkdir(exist_ok=True)
    (GATES / f"sweep-{reader}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"reader": reader, "chosen_age": chosen, "rates": {a: f"{v['simple_execute']}/{v['simple_n']}" for a, v in ages.items()}}))
    return res


# ---------------------------------------------------------------- confirmatory
def pairs_from(trace):
    fa = final_attempts(trace)
    pairs = defaultdict(dict)
    for r in fa.values():
        it = r["item"]
        pairs[it["pair_id"]][it["form"]] = r["score"]
        pairs[it["pair_id"]]["_meta"] = {k: it.get(k) for k in ("content_key", "frame", "variant", "in_r2", "source_actual_form", "e2e_subset", "anchor_redated")}
    return pairs, fa


def boot_rd(pairs, rng):
    """Cluster bootstrap (contents) percentile 95% CI for the complete-pair risk difference."""
    per = defaultdict(lambda: [0, 0, 0])   # b, c, complete
    for d in pairs.values():
        s, p = d.get("simple"), d.get("progressive")
        if s in ("EXECUTE", "VERIFY") and p in ("EXECUTE", "VERIFY"):
            k = d["_meta"]["content_key"]
            per[k][0] += s == "EXECUTE" and p == "VERIFY"; per[k][1] += p == "EXECUTE" and s == "VERIFY"; per[k][2] += 1
    keys = sorted(per)
    if not keys:
        return None
    vals = []
    for _ in range(BOOT):
        b = c = n = 0
        for _k in range(len(keys)):
            x = per[keys[rng.randrange(len(keys))]]; b += x[0]; c += x[1]; n += x[2]
        vals.append((b - c) / n if n else 0.0)
    vals.sort()
    return [round(vals[int(0.025 * BOOT)], 4), round(vals[int(0.975 * BOOT) - 1], 4)]


def cell_stats(pairs, rng):
    N = len(pairs)
    ex = {f: sum(d.get(f) == "EXECUTE" for d in pairs.values()) for f in ("simple", "progressive")}
    unp = {f: sum(d.get(f) not in ("EXECUTE", "VERIFY") for d in pairs.values()) for f in ("simple", "progressive")}
    b = c = comp = 0
    content = defaultdict(int)
    content_seen = set()
    for d in pairs.values():
        s, p = d.get("simple"), d.get("progressive")
        content_seen.add(d["_meta"]["content_key"])
        if s in ("EXECUTE", "VERIFY") and p in ("EXECUTE", "VERIFY"):
            comp += 1
            bi, ci = s == "EXECUTE" and p == "VERIFY", p == "EXECUTE" and s == "VERIFY"
            b += bi; c += ci; content[d["_meta"]["content_key"]] += bi - ci
    npos = sum(v > 0 for v in content.values()); nneg = sum(v < 0 for v in content.values())
    # worst/best-case bounds over unparsed members (all N pairs)
    lo = hi = 0
    for d in pairs.values():
        s, p = d.get("simple"), d.get("progressive")
        s_lo = 1 if s == "EXECUTE" else 0; p_lo = 0 if p == "VERIFY" else 1
        s_hi = 0 if s == "VERIFY" else 1; p_hi = 1 if p == "EXECUTE" else 0
        lo += s_lo - p_lo; hi += s_hi - p_hi
    rate = {f: ex[f] / N for f in ex} if N else {}
    return {"n_pairs": N, "n_contents": len(content_seen), "complete_pairs": comp, "execute": ex, "unparsed_or_error": unp,
            "rate": rate, "flat_only": b, "prog_only": c,
            "p_item_one_sided": upper_tail(b, b + c), "p_item_two_sided": two_sided(b, c),
            "contents_net_positive": npos, "contents_net_negative": nneg, "contents_tied": len(content_seen) - npos - nneg,
            "p_cluster_one_sided": upper_tail(npos, npos + nneg),
            "risk_difference": (b - c) / comp if comp else None, "rd_ci95_cluster_bootstrap": boot_rd(pairs, rng),
            "rd_missingness_bounds": [lo / N, hi / N] if N else None,
            "saturation": [f for f in rate if rate[f] < 0.10 or rate[f] > 0.90]}


def splits(pairs, field):
    out = defaultdict(lambda: {"flat_exec": 0, "prog_exec": 0, "n": 0, "flat_only": 0, "prog_only": 0})
    for d in pairs.values():
        k = str(d["_meta"].get(field)); o = out[k]
        s, p = d.get("simple"), d.get("progressive")
        o["n"] += 1; o["flat_exec"] += s == "EXECUTE"; o["prog_exec"] += p == "EXECUTE"
        o["flat_only"] += s == "EXECUTE" and p == "VERIFY"; o["prog_only"] += p == "EXECUTE" and s == "VERIFY"
    return dict(sorted(out.items()))


def final():
    frz = json.loads((HERE / "freeze_confirmatory.json").read_text())
    rng = random.Random(SEED)
    cells, e2e = {}, {}
    for cell in frz["cells"]:
        reader, age = cell.split("|")
        gate_res = json.loads((GATES / f"cellgate-{reader}-{age}.json").read_text())
        entry = {"reader": reader, "age": age, "role": frz["roles"][reader], "gate": {k: gate_res[k] for k in ("passed", "failures", "controls")}}
        t = TRACES / f"target-{reader}-{age}.jsonl"
        if gate_res["passed"] and t.exists():
            pairs, fa = pairs_from(t)
            entry.update(cell_stats(pairs, rng))
            entry["trace"] = str(t.relative_to(HERE)); entry["trace_sha256"] = sha(t)
            entry["parse_errors"] = dict(Counter(f"{r['item']['form']}:{r.get('parse_error')}" for r in fa.values() if r["score"] not in ("EXECUTE", "VERIFY")))
            entry["error_attempts"] = sum(1 for r in load(t) if r.get("error"))
            entry["per_frame"] = splits(pairs, "frame"); entry["per_variant"] = splits(pairs, "variant"); entry["per_in_r2"] = splits(pairs, "in_r2")
            entry["per_source_actual_form"] = splits(pairs, "source_actual_form")
        else:
            entry["status"] = "uninterpretable (cell gate failed)" if not gate_res["passed"] else "not run"
        cells[cell] = entry
        te = TRACES / f"e2e-{reader}-{age}.jsonl"
        if gate_res["passed"] and te.exists():
            pairs, fa = pairs_from(te)
            for sub in ("a_preserved", "b_simple_source"):
                sp = {k: v for k, v in pairs.items() if v["_meta"]["e2e_subset"] == sub}
                st = cell_stats(sp, rng)
                st["per_frame"] = splits(sp, "frame"); st["per_anchor_redated"] = splits(sp, "anchor_redated")
                e2e[f"{cell}|{sub}"] = {"reader": reader, "age": age, "subset": sub, **st, "trace": str(te.relative_to(HERE)), "trace_sha256": sha(te),
                                        "error_attempts": sum(1 for r in load(te) if r.get("error") and r["item"]["e2e_subset"] == sub)}
    run = {k: v for k, v in cells.items() if "p_item_one_sided" in v}
    hi = holm({k: v["p_item_one_sided"] for k, v in run.items()})
    hc = holm({k: v["p_cluster_one_sided"] for k, v in run.items()})
    for k in run:
        cells[k]["holm_item"] = hi[k]; cells[k]["holm_cluster"] = hc[k]
    he = holm({k: v["p_item_one_sided"] for k, v in e2e.items()})
    for k in e2e:
        e2e[k]["holm_e2e_family"] = he[k]
    prim = frz["primary_cell"]
    pc = cells.get(prim, {})
    success = bool(pc.get("holm_item", 1) < 0.05 and pc.get("holm_cluster", 1) < 0.05)
    out = {"freeze_sha256": sha(HERE / "freeze_confirmatory.json"), "cells": cells, "e2e": e2e, "primary_cell": prim,
           "success_rule": "primary cell (GLM-5.2, 8 months): Holm-adjusted one-sided p < .05 on BOTH the item-level exact McNemar and the content-level sign test",
           "success": success, "holm_family_size": len(run), "e2e_family_size": len(e2e), "bootstrap": {"seed": SEED, "reps": BOOT}}
    (HERE / "results_powered.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{'cell':52s} {'flat':>8s} {'prog':>8s} {'b:c':>7s} {'p1':>8s} {'Holm':>7s} {'clu+:-':>7s} {'pclu':>8s} {'Holmc':>7s} {'RD':>7s} CI sat")
    for k, v in cells.items():
        if "p_item_one_sided" not in v:
            print(k, v.get("status")); continue
        print(f"{k:52s} {v['execute']['simple']:>4d}/{v['n_pairs']:<3d} {v['execute']['progressive']:>4d}/{v['n_pairs']:<3d} {v['flat_only']:>3d}:{v['prog_only']:<3d} {v['p_item_one_sided']:8.2e} {v['holm_item']:7.4f} {v['contents_net_positive']:>3d}:{v['contents_net_negative']:<3d} {v['p_cluster_one_sided']:8.2e} {v['holm_cluster']:7.4f} {v['risk_difference']:+.3f} {v['rd_ci95_cluster_bootstrap']} {v['saturation']}")
    for k, v in e2e.items():
        print(f"{k:52s} {v['execute']['simple']:>4d}/{v['n_pairs']:<3d} {v['execute']['progressive']:>4d}/{v['n_pairs']:<3d} {v['flat_only']:>3d}:{v['prog_only']:<3d} {v['p_item_one_sided']:8.2e} {v['holm_e2e_family']:7.4f} {'':>7s} {'':>8s} {'':>7s} {v['risk_difference'] if v['risk_difference'] is None else round(v['risk_difference'], 3)!s:>7s} {v['rd_ci95_cluster_bootstrap']} {v['saturation']}")
    print("success:", success)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gate", "sweep", "final"))
    ap.add_argument("--kind", choices=("admission", "cellgate"))
    ap.add_argument("--reader"); ap.add_argument("--age")
    a = ap.parse_args()
    if a.cmd == "gate":
        gate(a.kind, a.reader, a.age)
    elif a.cmd == "sweep":
        sweep(a.reader)
    else:
        final()


if __name__ == "__main__":
    main()
