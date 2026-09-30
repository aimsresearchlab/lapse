"""(3) Read-time ladder replicate agreement -- deterministic reanalysis of
committed traces. NO API/model calls; stdlib only.

Reads (never writes outside this directory):
  - exploratory/read-time-ladder-2026-09-08/trace_*.jsonl (run 1)
  - exploratory/read-time-ladder-2026-09-08/trace_rep2_*.jsonl (replicate)

Per exploratory/read-time-ladder-2026-09-08/run_ladder.py and
research/READ_TIME_LADDER_PARTA_2026-09-08.md: the replicate
("A1 replicate", same-day, user go-ahead) re-sent only the A1 task (576 rows
per reader -- confirmed empirically: every trace_rep2_*.jsonl file here
contains task=="A1" rows exclusively, 0 task=="A2" rows). A1's response is a
single 0-100 integer ("probability this memory note is still true today"),
parsed into fields `p` (int or None) and `parse` ("OK"/"UNPARSED").

IMPLEMENTATION CHOICE (not pinned verbatim by the task spec): the task asks
for "per-item agreement of the primary parsed outcome (percent agreement and
Cohen's kappa)" and, "if the outcome field is a numeric persistence
estimate," also mean absolute difference and fraction within 5/10 points.
Since only A1 (numeric) was replicated, there is no separate categorical
outcome to run percent-agreement/kappa on other than the parse status itself
(OK vs UNPARSED, i.e. did the reader emit a parseable integer). This script
therefore reports BOTH:
  (i)  percent agreement + Cohen's kappa on the `parse` field (OK/UNPARSED)
       -- this is expected to be near-degenerate (parse is OK on all but a
       handful of rows across the whole project), and Cohen's kappa is
       undefined (reported as null) when expected agreement pe >= 1;
  (ii) the numeric diagnostics on `p` (mean abs diff, fraction within 5,
       fraction within 10, count identical, count changed) -- this is the
       informative measure for a continuous 0-100 estimate and is the one
       research/READ_TIME_LADDER_PARTA_2026-09-08.md itself reports.

"Three readers" = the canonical confirmatory-family trio named in
READ_TIME_LADDER_PARTA_2026-09-08.md: deepseek-v4-flash on the official
DeepSeek endpoint (superseding the OpenRouter-unpinned run per that record),
gpt-5.6-luna, and glm-5.2, all via the files below. The
openrouter-unpinned deepseek pair also exists on disk and is included as a
disclosed, non-canonical fourth row (superseded, per the record, by the
official-endpoint run) -- not counted in the three-reader headline.

"The paper's A1 paired contrast (progressive vs simple)" is interpreted as
PROG vs FLAT (this ladder's baseline form is named FLAT, not "simple"; there
is no form literally called "simple" in this dataset -- see NOTES in
run_ladder.py) at gap=stale, paired on (frame, cluster), matching
research/READ_TIME_LADDER_PARTA_2026-09-08.md's own "Result" table. The
fresh-gap contrast is reported alongside for completeness.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import ROOT, cohen_kappa, sha256_file, two_sided_binom_p  # noqa: E402

HERE = Path(__file__).resolve().parent
LADDER_DIR = ROOT / "exploratory" / "read-time-ladder-2026-09-08"

READERS = [
    ("deepseek-v4-flash (official)", True,
     "trace_deepseek-deepseek-v4-flash-official.jsonl",
     "trace_rep2_deepseek-deepseek-v4-flash-official.jsonl"),
    ("gpt-5.6-luna", True,
     "trace_openai-gpt-5-6-luna.jsonl",
     "trace_rep2_openai-gpt-5-6-luna.jsonl"),
    ("glm-5.2", True,
     "trace_z-ai-glm-5-2.jsonl",
     "trace_rep2_z-ai-glm-5-2.jsonl"),
    ("deepseek-v4-flash (openrouter-unpinned, non-canonical/superseded)", False,
     "trace_deepseek-deepseek-v4-flash-openrouter-unpinned.jsonl",
     "trace_rep2_deepseek-deepseek-v4-flash-openrouter-unpinned.jsonl"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def paired_prog_flat(recs_by_id: dict, gap: str) -> dict:
    """Pair PROG vs FLAT on (frame, cluster) for the given gap, task A1."""
    pairs: dict = defaultdict(dict)
    for r in recs_by_id.values():
        if r["task"] != "A1" or r["gap"] != gap or r["form"] not in ("PROG", "FLAT"):
            continue
        if r["p"] is None:
            continue
        pairs[(r["frame"], r["cluster"])][r["form"]] = r["p"]
    diffs = []
    lt = gt = eq = 0
    for v in pairs.values():
        if "PROG" not in v or "FLAT" not in v:
            continue
        d = v["PROG"] - v["FLAT"]
        diffs.append(d)
        if d < 0:
            lt += 1
        elif d > 0:
            gt += 1
        else:
            eq += 1
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    nonzero = lt + gt
    p = two_sided_binom_p(lt, nonzero) if nonzero else float("nan")
    return {
        "n_pairs": len(diffs), "mean_diff_prog_minus_flat": mean_diff,
        "prog_lt_flat": lt, "prog_gt_flat": gt, "tied": eq,
        "sign_test_two_sided_p": p,
    }


def main() -> None:
    input_files: set[Path] = set()
    result: dict = {
        "generated_by": "exploratory/reanalysis-2026-09-12/replicate_agreement.py",
        "readers": {},
    }

    for label, canonical, main_fn, rep2_fn in READERS:
        main_path = LADDER_DIR / main_fn
        rep2_path = LADDER_DIR / rep2_fn
        input_files.add(main_path)
        input_files.add(rep2_path)

        main_recs = {r["id"]: r for r in load_jsonl(main_path) if r["task"] == "A1"}
        rep2_recs = {r["id"]: r for r in load_jsonl(rep2_path) if r["task"] == "A1"}
        common_ids = sorted(set(main_recs) & set(rep2_recs))
        only_main = sorted(set(main_recs) - set(rep2_recs))
        only_rep2 = sorted(set(rep2_recs) - set(main_recs))

        # (i) parse-field agreement (OK/UNPARSED)
        parse_pairs = [(main_recs[i]["parse"], rep2_recs[i]["parse"]) for i in common_ids]
        parse_pct_agree = (sum(1 for a, b in parse_pairs if a == b) / len(parse_pairs)
                            if parse_pairs else None)
        kappa, po, pe = cohen_kappa(parse_pairs)

        # (ii) numeric diagnostics on p, restricted to items parsed OK on
        # BOTH calls (comparing a number to a missing value is undefined)
        both_ok = [i for i in common_ids
                   if main_recs[i]["p"] is not None and rep2_recs[i]["p"] is not None]
        diffs = [abs(main_recs[i]["p"] - rep2_recs[i]["p"]) for i in both_ok]
        n_diff = len(diffs)
        identical = sum(1 for d in diffs if d == 0)
        within5 = sum(1 for d in diffs if d <= 5)
        within10 = sum(1 for d in diffs if d <= 10)
        mean_abs_diff = sum(diffs) / n_diff if n_diff else None
        max_abs_diff = max(diffs) if diffs else None
        changed = sum(1 for i in common_ids
                      if main_recs[i]["p"] != rep2_recs[i]["p"])

        contrasts = {}
        for gap in ("stale", "fresh"):
            contrasts[gap] = {
                "run1": paired_prog_flat(main_recs, gap),
                "run2_replicate": paired_prog_flat(rep2_recs, gap),
            }

        result["readers"][label] = {
            "canonical": canonical,
            "main_trace": main_fn,
            "rep2_trace": rep2_fn,
            "n_common_items": len(common_ids),
            "n_only_in_main": len(only_main),
            "n_only_in_rep2": len(only_rep2),
            "parse_field_agreement": {
                "percent_agreement": parse_pct_agree,
                "cohens_kappa": kappa if kappa == kappa else None,  # NaN -> null
                "observed_agreement_po": po,
                "expected_agreement_pe": pe,
                "note": "near-degenerate: parse is OK on almost every row "
                        "project-wide, so pe is close to 1 and kappa is "
                        "either undefined or uninformative; see numeric "
                        "diagnostics below for the substantive comparison.",
            },
            "numeric_p_agreement": {
                "n_both_parsed_ok": n_diff,
                "identical": identical,
                "identical_pct": identical / n_diff if n_diff else None,
                "within_5_points": within5,
                "within_5_pct": within5 / n_diff if n_diff else None,
                "within_10_points": within10,
                "within_10_pct": within10 / n_diff if n_diff else None,
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "n_items_changed_run1_vs_rep2": changed,
            },
            "prog_vs_flat_contrast_by_gap": contrasts,
        }

    out_path = HERE / "replicate_agreement_results.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(input_files)}
    (HERE / "replicate_agreement_input_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"wrote {HERE / 'replicate_agreement_input_hashes.json'}")


if __name__ == "__main__":
    main()
