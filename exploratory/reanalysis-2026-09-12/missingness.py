"""Exclusion/missingness table and worst-case sensitivity for the v2
confirmatory family (item T0.4 of research/OVERNIGHT_PLAN_2026-09-12.md).

Deterministic, Python-3-standard-library only. Reads `data/stimuli_v2.jsonl`
and the 8 `traces/*.scored.jsonl` columns listed in `tools/analysis_v2.py`
(3 confirmatory + 1 frontier-subset + 2 local + 2 extension = the "3
confirmatory + 5 other complete columns" of the task). No API/model calls.

Pairing/labels copied (not imported) from `tools/analysis_v2.py`: arm=e1,
gap=stale, forms prog/simple; confirmatory = component e1_primary, carrier
c0_original, paired on (frame, cluster); R1 = component lexeme_subject,
lexeme_pair=="same", paired on (frame, cluster, subject). DESTROYED =
{COERCED-STATIVE, MANUFACTURE}; EXCLUDED = {RESIDUAL, WITNESS-DROPPED}.

Implementation choices NOT pinned by the task spec (also flagged in
RESULTS_E.md):

- "Any other labels" is determined empirically per run (the label set is
  read off the data, not hardcoded) so UNPARSED or any future label would
  show up automatically if present.
- "Pairs dropped and the reason" reports, per column, each dropped
  (frame, cluster[, subject]) key together with which form(s) carried an
  EXCLUDED label and which label it was.
- Worst-case sensitivity (i)/(ii) is applied to *all 8* columns for
  completeness, but the task's "stays below .05/3" check is reported only
  for the 3 confirmatory columns, as specified. .05/3 (not Holm on the
  imputed p's) is used exactly as the task states it.
- Where a column has zero EXCLUDED-labelled members in a family (common:
  most columns have 0 WITNESS-DROPPED/RESIDUAL in the e1_primary/c0/stale
  cell), (i) and (ii) trivially reproduce the confirmatory b/c/p and this
  is reported rather than skipped, so the sensitivity table has one row
  per column regardless.
- The R1 same-lexeme worst-case sensitivity mirrors analysis_v2.py's own
  R1 pairing (drop/impute a record's contribution to its (frame, cluster,
  subject) key rather than analysis_v2's "continue"-and-let-mcnemar-drop-
  incomplete-pairs behaviour, which is extensionally identical for the
  drop case and is the natural generalization for the impute case).

Usage: python3 exploratory/reanalysis-2026-09-12/missingness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import binom_ge, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

DESTROYED = {"COERCED-STATIVE", "MANUFACTURE"}
EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
FRAMES = ["affiliate_role", "class", "equipment", "household", "lodging",
          "project", "vehicle", "workplace"]

COLUMNS = [  # copied verbatim from tools/analysis_v2.py
    ("confirmatory", "run-v2-full-20260825-010312"),
    ("confirmatory", "run-v2-full-20260825-010323"),
    ("confirmatory", "run-v2-full-20260825-234313"),
    ("frontier-subset", "run-v2-full-frontier-sonnet5-20260826-104357"),
    ("local", "run-v2-full-local-qwen3-8b-20260813-060538"),
    ("local", "run-v2-full-local-mistral-small-3.2-24b-20260813-064704"),
    ("extension", "run-v2-full-local-gpt-oss-20b-20260826-052612"),
    ("extension", "run-v2-full-local-olmo-2-32b-instruct-20260826-060733"),
]


def load_stim() -> dict:
    stim = {}
    for ln in (ROOT / "data" / "stimuli_v2.jsonl").read_text().splitlines():
        c = json.loads(ln)
        stim[c["case_id"]] = c
    return stim


def confirmatory_records(recs: list, stim: dict) -> list:
    out = []
    for r in recs:
        if r["arm"] != "e1" or r["gap"] != "stale" or r["form"] not in ("prog", "simple"):
            continue
        s = stim[r["case_id"]]
        if s["component"] != "e1_primary" or r["carrier"] != "c0_original":
            continue
        out.append(r)
    return out


def r1_records(recs: list, stim: dict) -> list:
    out = []
    for r in recs:
        if r["arm"] != "e1" or r["gap"] != "stale" or r["form"] not in ("prog", "simple"):
            continue
        s = stim[r["case_id"]]
        if s["component"] != "lexeme_subject" or s.get("lexeme_pair") != "same":
            continue
        out.append((r, s))
    return out


def label_counts_by_form_frame(records: list) -> dict:
    counts: dict = {}
    for r in records:
        key = (r["form"], r["frame"])
        counts.setdefault(key, {})
        counts[key][r["label"]] = counts[key].get(r["label"], 0) + 1
    return {f"{form}|{frame}": labs for (form, frame), labs in sorted(counts.items())}


def build_pairs_with_reason(records: list, key_fn) -> tuple[dict, list]:
    """Returns (kept_pairs, dropped) where kept_pairs maps key -> {form: label}
    for keys where neither member carries an EXCLUDED label, and dropped
    lists {key, reason} for keys where at least one member does (or a form
    is entirely missing).
    """
    raw: dict = {}
    for r in records:
        raw.setdefault(key_fn(r), {})[r["form"]] = r["label"]
    kept, dropped = {}, []
    for key, forms in raw.items():
        if "prog" not in forms or "simple" not in forms:
            dropped.append({"key": list(key), "reason": "missing form", "forms": forms})
            continue
        excl = {form: lab for form, lab in forms.items() if lab in EXCLUDED}
        if excl:
            dropped.append({"key": list(key), "reason": "excluded label", "excluded": excl})
            continue
        kept[key] = forms
    return kept, dropped


def mcnemar_from_bools(pairs: dict) -> tuple[int, int, int, float]:
    b = c = 0
    for v in pairs.values():
        if v["prog"] and not v["simple"]:
            b += 1
        elif v["simple"] and not v["prog"]:
            c += 1
    n = len(pairs)
    return n, b, c, binom_ge(b, b + c)


def worst_case(records: list, key_fn, direction: str) -> tuple[int, int, int, float]:
    """direction 'i': excluded prog -> NOT destroyed, excluded simple -> destroyed.
    direction 'ii': the opposite (excluded prog -> destroyed, excluded simple -> NOT destroyed).
    Non-excluded labels are scored normally (label in DESTROYED). Pairs
    missing a form entirely are still dropped (imputation only covers
    EXCLUDED labels present on an existing record, per the task wording
    "every excluded ... member").
    """
    raw: dict = {}
    for r in records:
        raw.setdefault(key_fn(r), {})[r["form"]] = r["label"]
    pairs = {}
    for key, forms in raw.items():
        if "prog" not in forms or "simple" not in forms:
            continue
        resolved = {}
        for form, lab in forms.items():
            if lab in EXCLUDED:
                if direction == "i":
                    resolved[form] = (form == "simple")   # prog->False, simple->True
                else:
                    resolved[form] = (form == "prog")      # prog->True, simple->False
            else:
                resolved[form] = lab in DESTROYED
        pairs[key] = resolved
    return mcnemar_from_bools(pairs)


def main() -> None:
    stim = load_stim()
    results = {"columns": [], "input_hashes": {}}
    stim_path = ROOT / "data" / "stimuli_v2.jsonl"
    results["input_hashes"]["data/stimuli_v2.jsonl"] = sha256_file(stim_path)

    sensitivity_summary = []

    for tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        results["input_hashes"][f"traces/{base}.scored.jsonl"] = sha256_file(path)
        recs = [json.loads(ln) for ln in path.read_text().splitlines()]
        model = recs[0]["model"]

        conf_recs = confirmatory_records(recs, stim)
        r1_pairs_raw = r1_records(recs, stim)
        r1_recs = [r for r, s in r1_pairs_raw]
        r1_subject = {id(r): s.get("subject") for r, s in r1_pairs_raw}

        conf_label_table = label_counts_by_form_frame(conf_recs)
        r1_label_table = label_counts_by_form_frame(r1_recs)

        conf_key = lambda r: (r["frame"], r["cluster"])
        conf_kept, conf_dropped = build_pairs_with_reason(conf_recs, conf_key)
        conf_pairs_bool = {k: {"prog": v["prog"] in DESTROYED, "simple": v["simple"] in DESTROYED}
                            for k, v in conf_kept.items()}
        n, b, c, p = mcnemar_from_bools(conf_pairs_bool)

        r1_key = lambda r: (r["frame"], r["cluster"], r1_subject[id(r)])
        r1_kept, r1_dropped = build_pairs_with_reason(r1_recs, r1_key)
        r1_pairs_bool = {k: {"prog": v["prog"] in DESTROYED, "simple": v["simple"] in DESTROYED}
                          for k, v in r1_kept.items()}
        n1, b1, c1, p1 = mcnemar_from_bools(r1_pairs_bool)

        conf_wc_i = worst_case(conf_recs, conf_key, "i")
        conf_wc_ii = worst_case(conf_recs, conf_key, "ii")
        r1_wc_i = worst_case(r1_recs, r1_key, "i")
        r1_wc_ii = worst_case(r1_recs, r1_key, "ii")

        col = {
            "tier": tier, "trace": base, "model": model,
            "confirmatory_label_counts_by_form_frame": conf_label_table,
            "confirmatory_pairs_dropped": conf_dropped,
            "confirmatory_baseline": {"n": n, "b": b, "c": c, "p_one_sided": p},
            "confirmatory_worst_case_i": dict(zip(("n", "b", "c", "p_one_sided"), conf_wc_i)),
            "confirmatory_worst_case_ii": dict(zip(("n", "b", "c", "p_one_sided"), conf_wc_ii)),
            "r1_label_counts_by_form_frame": r1_label_table,
            "r1_pairs_dropped": r1_dropped,
            "r1_baseline": {"n": n1, "b": b1, "c": c1, "p_one_sided": p1},
            "r1_worst_case_i": dict(zip(("n", "b", "c", "p_one_sided"), r1_wc_i)),
            "r1_worst_case_ii": dict(zip(("n", "b", "c", "p_one_sided"), r1_wc_ii)),
        }
        results["columns"].append(col)

        if tier == "confirmatory":
            sensitivity_summary.append({
                "model": model,
                "confirmatory_p_worst_case_i": conf_wc_i[3],
                "below_0.05_over_3": conf_wc_i[3] < 0.05 / 3,
                "r1_p_worst_case_i": r1_wc_i[3],
                "r1_below_0.05_over_3": (r1_wc_i[3] < 0.05 / 3) if r1_wc_i[1] + r1_wc_i[2] else None,
            })

    results["confirmatory_worst_case_i_summary"] = sensitivity_summary
    out_path = OUT / "missingness_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    for s in sensitivity_summary:
        print(s)


if __name__ == "__main__":
    main()
