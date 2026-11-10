#!/usr/bin/env python3
"""Stage analysis (research/STRUCTURED_RECORD_SPEC_2026-10-10.md, section 6). Written and hashed
into freeze_confirmatory.json before any target call.

    python analyze_sr.py --reader R --age A     # results-R-A.json
"""
import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "downstream-tool-powered-2026-09-22"))
import analyze_powered as A  # noqa: E402

ARMS = ("F", "S_full", "S_noquote", "SRC")


def p_upper(k, n):
    """One-sided exact binomial P(X >= k | n, 1/2)."""
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def p_two(k, n):
    return 1.0 if n == 0 else min(1.0, 2 * min(p_upper(k, n), p_upper(n - k, n)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reader", required=True); ap.add_argument("--age", required=True)
    a = ap.parse_args()
    trace = HERE / "traces" / f"target-{a.reader}-{a.age}.jsonl"
    rows = list(A.final_attempts(trace).values())
    act = {}   # (arm, content_key, variant, form) -> 1 EXECUTE / 0 VERIFY / None otherwise
    for r in rows:
        it = r["item"]
        act[(it["arm"], it["content_key"], it["variant"], it["form"])] = {"EXECUTE": 1, "VERIFY": 0}.get(r["score"])
    base = sorted({(k[1], k[2]) for k in act})
    res = {"reader": a.reader, "age": a.age, "trace": str(trace.relative_to(HERE)), "trace_sha256": A.sha(trace),
           "calls_final": len(rows), "unparsed": sum(v is None for v in act.values()), "arms": {}}
    for arm in ARMS:
        s_act = [act.get((arm, c, v, "simple")) for c, v in base]
        p_act = [act.get((arm, c, v, "progressive")) for c, v in base]
        ok = [(s, p) for s, p in zip(s_act, p_act) if s is not None and p is not None]
        b = sum(s == 1 and p == 0 for s, p in ok); c = sum(s == 0 and p == 1 for s, p in ok)
        res["arms"][arm] = {"pairs": len(ok), "simple_acted": sum(s for s, _ in ok), "progressive_acted": sum(p for _, p in ok),
                            "simple_only": b, "progressive_only": c, "p_one_sided": p_upper(b, b + c), "p_two_sided": p_two(b, b + c)}

    # H1: form contrast larger with the full record than with the free-text note (pair-level D, content-level sign test)
    def d(arm, c, v):
        s, p = act.get((arm, c, v, "simple")), act.get((arm, c, v, "progressive"))
        return None if s is None or p is None else s - p
    by_content = defaultdict(list); pos = neg = 0
    for c, v in base:
        x, y = d("S_full", c, v), d("F", c, v)
        if x is None or y is None:
            continue
        D = x - y; by_content[c].append(D); pos += D > 0; neg += D < 0
    cm = {c: sum(v) / len(v) for c, v in by_content.items()}
    cpos, cneg = sum(m > 0 for m in cm.values()), sum(m < 0 for m in cm.values())
    res["H1"] = {"pairs": sum(len(v) for v in by_content.values()), "pair_D_pos": pos, "pair_D_neg": neg,
                 "pair_p_one_sided": p_upper(pos, pos + neg), "contents": len(cm), "content_pos": cpos, "content_neg": cneg,
                 "content_p_one_sided": p_upper(cpos, cpos + cneg)}

    # H2: on progressive-source items, agreement with the SRC decision is higher for S_full than for F
    def fidelity(arm, form):
        n10 = n01 = agree = n = 0
        for c, v in base:
            src, x, f = act.get(("SRC", c, v, form)), act.get((arm, c, v, form)), act.get(("F", c, v, form))
            if src is None or x is None or f is None:
                continue
            ax, af = x == src, f == src
            n += 1; agree += ax; n10 += ax and not af; n01 += af and not ax
        return {"items": n, "agree_arm": agree, "arm_only_agrees": n10, "F_only_agrees": n01,
                "p_one_sided": p_upper(n10, n10 + n01), "p_two_sided": p_two(n10, n10 + n01)}
    res["fidelity_F"] = {form: {"items": fidelity("F", form)["items"], "agree": sum(
        1 for c, v in base if None not in (act.get(("SRC", c, v, form)), act.get(("F", c, v, form)))
        and act[("SRC", c, v, form)] == act[("F", c, v, form)])} for form in ("progressive", "simple")}
    res["H2"] = fidelity("S_full", "progressive")
    res["H2_check_simple"] = fidelity("S_full", "simple")
    res["noquote_progressive"] = fidelity("S_noquote", "progressive")
    res["noquote_simple"] = fidelity("S_noquote", "simple")
    if any(k[0] == "SRC_retest" for k in act):   # noise ceiling (GLM only; added before GLM calls)
        res["SRC_retest"] = {form: {"items": sum(1 for c, v in base if None not in (act.get(("SRC", c, v, form)), act.get(("SRC_retest", c, v, form)))),
                                    "agree": sum(1 for c, v in base if None not in (act.get(("SRC", c, v, form)), act.get(("SRC_retest", c, v, form)))
                                                 and act[("SRC", c, v, form)] == act[("SRC_retest", c, v, form)])} for form in ("progressive", "simple")}
    out = HERE / f"results-{a.reader}-{a.age}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
