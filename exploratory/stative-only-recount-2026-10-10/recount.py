"""Stative-only recount of the main memory-write test for all 11 configurations ($0).

Why: the 2026-10-10 human check (research/MEMWRITE_ANNOTATION_SPEC_2026-10-09.md, one annotator,
64 notes) agreed with the cascade's COERCED-STATIVE label (kappa .83) but not with MANUFACTURE on
kept progressives (kappa .38 for the full destroyed set). This recounts Table allmodels with
destroyed = COERCED-STATIVE only, next to the frozen definition (COERCED-STATIVE or MANUFACTURE).
Pairs and exclusions are built exactly as in exploratory/reanalysis-2026-09-12/frame_jackknife.py.

Usage: python3 exploratory/stative-only-recount-2026-10-10/recount.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "exploratory" / "reanalysis-2026-09-12"))
from _stats import binom_ge, sha256_file  # noqa: E402
from frame_jackknife import EXCLUDED, load_stim  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import score as SC  # noqa: E402

COLUMNS = [  # Table allmodels order
    ("DeepSeek V4 Flash", "confirmatory", "run-v2-full-20260825-010312"),
    ("GPT-5.6 Luna", "confirmatory", "run-v2-full-20260825-010323"),
    ("GLM-5.2", "confirmatory", "run-v2-full-20260825-234313"),
    ("Claude Sonnet 5", "frontier", "run-v2-full-frontier-sonnet5-20260826-104357"),
    ("Qwen3-8B", "estimation", "run-v2-full-local-qwen3-8b-20260813-060538"),
    ("Mistral Small 3.2", "estimation", "run-v2-full-local-mistral-small-3.2-24b-20260813-064704"),
    ("GPT-OSS-20B", "extension", "run-v2-full-local-gpt-oss-20b-20260826-052612"),
    ("OLMo-2-32B", "extension", "run-v2-full-local-olmo-2-32b-instruct-20260826-060733"),
    ("Gemini 3.1 Pro", "frontier", "run-v2-full-frontier-gemini31pro-20260827-015332"),
    ("Grok 4.6", "frontier", "run-v2-full-frontier-grok46-20260827-015925"),
    ("Qwen3.8-Max", "frontier", "run-v2-full-frontier-qwen38max-20260827-021106"),
]
DEFS = {"frozen": {"COERCED-STATIVE", "MANUFACTURE"}, "stative_only": {"COERCED-STATIVE"}}
# The cascade never labels a simple-present source COERCED-STATIVE, so stative_only has c = 0 by
# construction. aspect_flip is the symmetric version that can show reversals: a progressive
# source destroyed = COERCED-STATIVE; a simple source destroyed = its witness line gains a
# progressive (tools/score.py PROGRESSIVE or PERF_PROG), dated or not.


def gains_progressive(rec: dict) -> bool:
    t = SC.norm(rec["response"])
    ln = " ".join(l for l in t.splitlines() if SC.wit_re(rec["wit_token"]).search(l))
    return bool(SC.PROGRESSIVE.search(ln) or SC.PERF_PROG.search(ln))


def main_pairs(recs: list, stim: dict) -> dict:
    raw: dict = {}
    for r in recs:
        if r["arm"] != "e1" or r["gap"] != "stale" or r["form"] not in ("prog", "simple"):
            continue
        if stim[r["case_id"]]["component"] != "e1_primary" or r["carrier"] != "c0_original":
            continue
        raw.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = r
    return {k: {"prog": v["prog"]["label"], "simple": v["simple"]["label"],
                "simple_gains_prog": gains_progressive(v["simple"])}
            for k, v in raw.items() if "prog" in v and "simple" in v
            and v["prog"]["label"] not in EXCLUDED and v["simple"]["label"] not in EXCLUDED}


def main() -> None:
    stim = load_stim()
    out = {"columns": [], "input_hashes": {}}
    tot = {d: Counter() for d in DEFS}
    for name, tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        out["input_hashes"][f"traces/{base}.scored.jsonl"] = sha256_file(path)
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        pairs = main_pairs(recs, stim)
        row = {"model": name, "tier": tier, "n": len(pairs),
               "reversal_labels": sorted(v["simple"] for v in pairs.values()
                                         if v["simple"] in DEFS["frozen"] and v["prog"] not in DEFS["frozen"])}
        for d, S in DEFS.items():
            b = sum(v["prog"] in S and v["simple"] not in S for v in pairs.values())
            c = sum(v["simple"] in S and v["prog"] not in S for v in pairs.values())
            sp = sum(v["simple"] in S for v in pairs.values())
            row[d] = {"b": b, "c": c, "simple_destroyed": sp, "p_one_sided": binom_ge(b, b + c)}
            tot[d].update({"n": len(pairs), "b": b, "c": c})
        b = sum(v["prog"] == "COERCED-STATIVE" and not v["simple_gains_prog"] for v in pairs.values())
        c = sum(v["simple_gains_prog"] and v["prog"] != "COERCED-STATIVE" for v in pairs.values())
        row["aspect_flip"] = {"b": b, "c": c, "p_one_sided": binom_ge(b, b + c),
                              "simple_destroyed": sum(v["simple_gains_prog"] for v in pairs.values())}
        tot.setdefault("aspect_flip", Counter()).update({"n": len(pairs), "b": b, "c": c})
        out["columns"].append(row)
    out["totals"] = {d: dict(t) for d, t in tot.items()}
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'model':18s} {'n':>4s}  frozen b:c   stative b:c  simple-destroyed frozen/stative  frozen reversal labels  aspect-flip b:c")
    for r in out["columns"]:
        f, s = r["frozen"], r["stative_only"]
        print(f"{r['model']:18s} {r['n']:4d}  {f['b']:4d}:{f['c']:<3d}    {s['b']:4d}:{s['c']:<3d}     "
              f"{f['simple_destroyed']:3d}/{s['simple_destroyed']:<3d}  {str(r['reversal_labels']):24s}"
              f"{r['aspect_flip']['b']}:{r['aspect_flip']['c']}")
    for d, t in out["totals"].items():
        print(d, t)


if __name__ == "__main__":
    main()
