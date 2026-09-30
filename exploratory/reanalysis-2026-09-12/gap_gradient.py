"""(2) Behavioral form contrast across all six gaps -- deterministic
reanalysis of committed v2 traces. NO API/model calls; stdlib only.

Reads (never writes outside this directory):
  - traces/<base>.scored.jsonl (cascade label + frame/cluster/carrier/gap/
    form/component, confirmed present on every row by inspection)
  - traces/<base>.scored.judged.local-seed-oss-36b-instruct-pv2.jsonl
    (judge label, joined on case_id)

Six-gap grid (fresh, near, boundary, expired_soon, stale, stale_long) x form
(prog, simple, bound): the arm=behavioral records that supply these six gaps
at forms {prog, simple, bound} and carrier c0_original live in exactly three
stimuli components (checked empirically against data/stimuli_v2.jsonl):
  - behavioral_core       -> gaps {fresh, stale}
  - behavioral_boundary   -> gap {boundary}
  - gap_gradient          -> gaps {near, expired_soon, stale_long}
All three are carrier c0_original only, so the (frame, cluster, carrier)
pairing key used below reduces to (frame, cluster) in practice; carrier is
still carried through and asserted constant. The other behavioral components
(carrier_behavioral_p2, carrier_fresh_p3, perf) use non-c0 carriers or a
different form set (perf_prog only) and are out of scope for this six-gap
table -- IMPLEMENTATION CHOICE, not spelled out verbatim in the task, but
the only components that actually populate all six named gaps at
prog/simple/bound.

Both-detector rule (per the task): NON-COMMIT iff cascade AND judge are both
in {HEDGE, TEMPLATE}; COMMIT iff both in {PROCEED, PROCEED-WITH-CAVEAT};
else DISAGREE (excluded from the binary tally and from McNemar, counted).
CHECK-FIRST does not occur in the behavioral arm (it is the anchor-arm
label); RESIDUAL and CONFAB-COMMIT (possible score_behavioral() outputs) are
neither commit nor non-commit under this rule and fall into DISAGREE when
either detector emits them -- disclosed per cell.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import ROOT, cp_upper, sha256_file, two_sided_binom_p  # noqa: E402

HERE = Path(__file__).resolve().parent
NONCOMMIT = {"HEDGE", "TEMPLATE"}
COMMIT = {"PROCEED", "PROCEED-WITH-CAVEAT"}
GAPS = ["fresh", "near", "boundary", "expired_soon", "stale", "stale_long"]
FORMS = ["prog", "simple", "bound"]
SIX_GAP_COMPONENTS = {"behavioral_core", "behavioral_boundary", "gap_gradient"}

COLUMNS = [  # (tier, trace basename) -- copied from tools/analysis_v2.py
    ("confirmatory", "run-v2-full-20260825-010312"),
    ("confirmatory", "run-v2-full-20260825-010323"),
    ("confirmatory", "run-v2-full-20260825-234313"),
    ("frontier-subset", "run-v2-full-frontier-sonnet5-20260826-104357"),
    ("local", "run-v2-full-local-qwen3-8b-20260813-060538"),
    ("local", "run-v2-full-local-mistral-small-3.2-24b-20260813-064704"),
    ("extension", "run-v2-full-local-gpt-oss-20b-20260826-052612"),
    ("extension", "run-v2-full-local-olmo-2-32b-instruct-20260826-060733"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def binary_label(cascade: str | None, judge: str | None) -> str:
    if cascade in NONCOMMIT and judge in NONCOMMIT:
        return "NON-COMMIT"
    if cascade in COMMIT and judge in COMMIT:
        return "COMMIT"
    return "DISAGREE"


def main() -> None:
    input_files: set[Path] = set()
    result: dict = {
        "generated_by": "exploratory/reanalysis-2026-09-12/gap_gradient.py",
        "noncommit_labels": sorted(NONCOMMIT),
        "commit_labels": sorted(COMMIT),
        "six_gap_components": sorted(SIX_GAP_COMPONENTS),
        "columns": {},
        "implementation_choices": [
            "The six-gap x {prog,simple,bound} grid is assembled from the "
            "behavioral_core + behavioral_boundary + gap_gradient stimuli "
            "components (all carrier c0_original) -- see module docstring. "
            "carrier_behavioral_p2 (c1-c5, stale only) and carrier_fresh_p3 "
            "(c1-c5, fresh only, simple only) are excluded from this table.",
            "Records with no judge counterpart (case_id absent from the "
            "judged file, e.g. any final-error retries that changed "
            "case coverage) are dropped and counted separately per cell.",
            "McNemar pairing key is (frame, cluster, carrier); since all "
            "three components use carrier=c0_original exclusively this is "
            "empirically equivalent to (frame, cluster), but carrier is "
            "still carried in the key and asserted constant.",
        ],
    }

    for tier, base in COLUMNS:
        scored_path = ROOT / "traces" / f"{base}.scored.jsonl"
        judged_path = ROOT / "traces" / f"{base}.scored.judged.local-seed-oss-36b-instruct-pv2.jsonl"
        input_files.add(scored_path)
        recs = load_jsonl(scored_path)
        model = recs[0]["model"]

        judge_of: dict = {}
        if judged_path.exists():
            input_files.add(judged_path)
            for r in load_jsonl(judged_path):
                judge_of[r["case_id"]] = r["judge"]

        beh = [r for r in recs if r["arm"] == "behavioral"
               and r["component"] in SIX_GAP_COMPONENTS
               and r["form"] in FORMS]
        assert all(r["carrier"] == "c0_original" for r in beh)

        no_judge_ct = sum(1 for r in beh if r["case_id"] not in judge_of)

        # per (gap, form) cell + per-record binary label, cached for McNemar
        cell_counts: dict = {}
        rec_binary: dict = {}   # case_id -> binary label
        rec_key: dict = {}      # case_id -> (frame, cluster, carrier)
        for gap in GAPS:
            for form in FORMS:
                sub = [r for r in beh if r["gap"] == gap and r["form"] == form]
                n = len(sub)
                nc = commit = disagree = missing_judge = 0
                cascade_label_counts: Counter = Counter()
                judge_label_counts: Counter = Counter()
                for r in sub:
                    cascade = r["label"]
                    judge = judge_of.get(r["case_id"])
                    cascade_label_counts[cascade] += 1
                    if judge is not None:
                        judge_label_counts[judge] += 1
                    if judge is None:
                        missing_judge += 1
                        b = "DISAGREE"
                    else:
                        b = binary_label(cascade, judge)
                    rec_binary[r["case_id"]] = b
                    rec_key[r["case_id"]] = (r["frame"], r["cluster"], r["carrier"])
                    if b == "NON-COMMIT":
                        nc += 1
                    elif b == "COMMIT":
                        commit += 1
                    else:
                        disagree += 1
                floor_ub = None
                if n > 0 and nc == 0:
                    floor_ub = cp_upper(0, commit + nc)  # among non-DISAGREE only
                cell_counts[f"{gap}|{form}"] = {
                    "n_total": n,
                    "non_commit": nc,
                    "commit": commit,
                    "disagree": disagree,
                    "missing_judge": missing_judge,
                    "non_commit_over_binary_n": (nc / (nc + commit)) if (nc + commit) else None,
                    "cascade_label_counts": dict(sorted(cascade_label_counts.items())),
                    "judge_label_counts": dict(sorted(judge_label_counts.items())),
                    "floor_0_of_n_one_sided_cp_upper_95": floor_ub,
                }

        # paired prog-vs-simple McNemar within each gap, keyed on
        # (frame, cluster, carrier); only records with a COMMIT/NON-COMMIT
        # binary label participate; pairs missing one side, or with either
        # side DISAGREE, are dropped and counted.
        mcnemar_by_gap = {}
        for gap in GAPS:
            pairs: dict = defaultdict(dict)
            for r in beh:
                if r["gap"] != gap or r["form"] not in ("prog", "simple"):
                    continue
                b = rec_binary.get(r["case_id"])
                if b is None or b == "DISAGREE":
                    continue
                key = rec_key[r["case_id"]]
                pairs[key][r["form"]] = b
            n_pairs = b_ct = c_ct = dropped = 0
            for v in pairs.values():
                if "prog" not in v or "simple" not in v:
                    dropped += 1
                    continue
                n_pairs += 1
                if v["prog"] == "NON-COMMIT" and v["simple"] == "COMMIT":
                    b_ct += 1
                elif v["simple"] == "NON-COMMIT" and v["prog"] == "COMMIT":
                    c_ct += 1
            p = two_sided_binom_p(b_ct, b_ct + c_ct) if (b_ct + c_ct) else float("nan")
            mcnemar_by_gap[gap] = {
                "pairs": n_pairs,
                "discordant_prog_noncommit_simple_commit": b_ct,
                "discordant_simple_noncommit_prog_commit": c_ct,
                "pairs_dropped_missing_or_disagree": dropped,
                "two_sided_exact_p": p,
            }

        result["columns"][model] = {
            "tier": tier,
            "trace_base": base,
            "n_behavioral_six_gap_rows": len(beh),
            "rows_missing_judge_label": no_judge_ct,
            "cells": cell_counts,
            "mcnemar_prog_vs_simple_by_gap": mcnemar_by_gap,
        }

    out_path = HERE / "gap_gradient_results.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(input_files)}
    (HERE / "gap_gradient_input_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"wrote {HERE / 'gap_gradient_input_hashes.json'}")


if __name__ == "__main__":
    main()
