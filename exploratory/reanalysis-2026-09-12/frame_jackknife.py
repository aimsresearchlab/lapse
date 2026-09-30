"""Frame- and cluster-level jackknife/bootstrap robustness checks for the v2
confirmatory McNemar test (item T0.2 of research/OVERNIGHT_PLAN_2026-09-12.md).

Deterministic, Python-3-standard-library only. Reads
`data/stimuli_v2.jsonl` and the 8 `traces/*.scored.jsonl` columns listed in
`tools/analysis_v2.py`'s COLUMNS. No API/model calls.

Confirmatory pairing is copied (not imported) from `tools/analysis_v2.py`
Test 1, so this script has no import dependency outside this directory:
arm=e1, gap=stale, component=e1_primary, carrier=c0_original, forms
prog vs simple, paired on (frame, cluster). DESTROYED = {COERCED-STATIVE,
MANUFACTURE}; EXCLUDED = {RESIDUAL, WITNESS-DROPPED}; a pair with an
excluded member is dropped from every analysis below, including every
resample (a dropped pair never re-enters through resampling because
resampling draws frames/clusters, not individual pairs, and the dropped
pair is simply absent from its frame's or cluster's pool).

Implementation choices NOT pinned by the task spec (also flagged in
RESULTS_E.md):

1. (b)/(c) leave-N-frames-out removes the named frame(s) entirely from the
   confirmatory 125-128-pair pool and recomputes McNemar on what remains.
2. (d) frame bootstrap: resample the 8 frame *labels* with replacement,
   8 draws per replicate (`random.choices`). A replicate's pool is the
   concatenation of all (frame, cluster) pairs for each drawn frame label;
   a frame drawn twice contributes its pairs twice (standard weighting for
   a cluster-of-clusters bootstrap where the frame is the resampled unit).
3. (e) cluster-within-frame bootstrap: each of the 8 frames is used exactly
   once (frames are not resampled here — that is structure (d)); within
   each frame, its own available clusters (<=16, fewer if any pair was
   dropped for exclusion) are resampled with replacement at the same count.
   This is a second, non-nested dependence structure, not a two-stage
   bootstrap combining both.
4. Per replicate: risk difference = mean(prog destroyed) - mean(simple
   destroyed) over the (possibly-duplicated) pooled pairs; b/(b+c) uses the
   discordant counts in the same pooled replicate. A replicate whose pool
   has zero discordant pairs has an undefined b/(b+c) and is excluded from
   that statistic's interval (count of such replicates is reported).
5. Two independent `random.Random(20260912)` instances are used, one for
   (d) and one for (e), so each bootstrap is reproducible from the stated
   seed on its own rather than depending on draw order across the two.
6. Percentile 95% CI = nearest-rank 2.5th/97.5th percentile of the sorted
   10,000 replicate statistics (index = round(q * (n-1)), no
   interpolation). This needs no numpy.
7. (f) per-carrier b/c pairs prog vs simple on (frame, cluster) *within*
   that carrier. `analysis_v2.py`'s own carrier test instead compares
   unpaired per-carrier destruction *rates* (for a sign test across
   carriers); the task here explicitly asks for "b, c", so this script
   pairs by (frame, cluster) to produce genuine discordant counts.

Usage: python3 exploratory/reanalysis-2026-09-12/frame_jackknife.py
"""
from __future__ import annotations

import json
import random
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
CARRIERS = ["c1_fact_final", "c2_bare", "c3_embedded_mid", "c4_formal",
            "c5_fact_list"]

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

N_BOOT = 10_000
SEED = 20260912


def load_stim() -> dict:
    stim = {}
    for ln in (ROOT / "data" / "stimuli_v2.jsonl").read_text().splitlines():
        c = json.loads(ln)
        stim[c["case_id"]] = c
    return stim


def mcnemar(pairs: dict) -> tuple[int, int, int, float]:
    """pairs: (frame,cluster) -> {'prog': bool, 'simple': bool}."""
    b = c = 0
    for v in pairs.values():
        if v["prog"] and not v["simple"]:
            b += 1
        elif v["simple"] and not v["prog"]:
            c += 1
    n = len(pairs)
    return n, b, c, binom_ge(b, b + c)


def build_confirmatory_pairs(recs: list, stim: dict) -> dict:
    """e1_primary c0_original stale prog/simple, keyed by (frame, cluster).

    A key is present only if both forms exist and neither carries an
    EXCLUDED label (mirrors analysis_v2.py's DROP behaviour).
    """
    raw: dict = {}
    for r in recs:
        if r["arm"] != "e1" or r["gap"] != "stale" or r["form"] not in ("prog", "simple"):
            continue
        s = stim[r["case_id"]]
        if s["component"] != "e1_primary" or r["carrier"] != "c0_original":
            continue
        raw.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = r["label"]
    pairs = {}
    for key, forms in raw.items():
        if "prog" not in forms or "simple" not in forms:
            continue
        if forms["prog"] in EXCLUDED or forms["simple"] in EXCLUDED:
            continue
        pairs[key] = {"prog": forms["prog"] in DESTROYED,
                      "simple": forms["simple"] in DESTROYED}
    return pairs


def build_carrier_pairs(recs: list, stim: dict, carrier: str) -> dict:
    raw: dict = {}
    for r in recs:
        if r["arm"] != "e1" or r["gap"] != "stale" or r["form"] not in ("prog", "simple"):
            continue
        s = stim[r["case_id"]]
        if s["component"] != "carrier_e1" or r["carrier"] != carrier:
            continue
        raw.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = r["label"]
    pairs = {}
    for key, forms in raw.items():
        if "prog" not in forms or "simple" not in forms:
            continue
        if forms["prog"] in EXCLUDED or forms["simple"] in EXCLUDED:
            continue
        pairs[key] = {"prog": forms["prog"] in DESTROYED,
                      "simple": forms["simple"] in DESTROYED}
    return pairs


def risk_diff_and_ratio(pool: list) -> tuple[float, float | None, int, int]:
    """pool: list of {'prog': bool, 'simple': bool} (duplicates allowed).

    Returns (risk_difference, b/(b+c) or None if b+c==0, b, c).
    """
    n = len(pool)
    prog_rate = sum(v["prog"] for v in pool) / n
    simple_rate = sum(v["simple"] for v in pool) / n
    b = sum(1 for v in pool if v["prog"] and not v["simple"])
    c = sum(1 for v in pool if v["simple"] and not v["prog"])
    ratio = b / (b + c) if (b + c) else None
    return prog_rate - simple_rate, ratio, b, c


def percentile(sorted_vals: list, q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. No interpolation."""
    if not sorted_vals:
        return float("nan")
    idx = round((q / 100.0) * (len(sorted_vals) - 1))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def bootstrap_frames(pairs: dict, seed: int, n_boot: int) -> dict:
    by_frame: dict = {}
    for (fr, cl), v in pairs.items():
        by_frame.setdefault(fr, []).append(v)
    frames_present = [f for f in FRAMES if f in by_frame]
    rng = random.Random(seed)
    diffs, ratios = [], []
    undefined = 0
    for _ in range(n_boot):
        drawn = rng.choices(frames_present, k=len(FRAMES))
        pool = [v for fr in drawn for v in by_frame[fr]]
        if not pool:
            undefined += 1
            continue
        rd, ratio, _, _ = risk_diff_and_ratio(pool)
        diffs.append(rd)
        if ratio is None:
            undefined += 1
        else:
            ratios.append(ratio)
    diffs.sort()
    ratios.sort()
    return {
        "n_boot": n_boot,
        "risk_diff_ci95": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
        "risk_diff_median": percentile(diffs, 50),
        "ratio_ci95": [percentile(ratios, 2.5), percentile(ratios, 97.5)] if ratios else None,
        "ratio_median": percentile(ratios, 50) if ratios else None,
        "n_undefined_ratio_replicates": undefined,
    }


def bootstrap_clusters_within_frame(pairs: dict, seed: int, n_boot: int) -> dict:
    by_frame: dict = {}
    for (fr, cl), v in pairs.items():
        by_frame.setdefault(fr, []).append(v)
    frames_present = [f for f in FRAMES if f in by_frame]
    rng = random.Random(seed)
    diffs, ratios = [], []
    undefined = 0
    for _ in range(n_boot):
        pool = []
        for fr in frames_present:
            vals = by_frame[fr]
            pool.extend(rng.choices(vals, k=len(vals)))
        if not pool:
            undefined += 1
            continue
        rd, ratio, _, _ = risk_diff_and_ratio(pool)
        diffs.append(rd)
        if ratio is None:
            undefined += 1
        else:
            ratios.append(ratio)
    diffs.sort()
    ratios.sort()
    return {
        "n_boot": n_boot,
        "risk_diff_ci95": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
        "risk_diff_median": percentile(diffs, 50),
        "ratio_ci95": [percentile(ratios, 2.5), percentile(ratios, 97.5)] if ratios else None,
        "ratio_median": percentile(ratios, 50) if ratios else None,
        "n_undefined_ratio_replicates": undefined,
    }


def main() -> None:
    stim = load_stim()
    results = {"columns": [], "seed": SEED, "n_boot": N_BOOT,
               "input_hashes": {}}

    stim_path = ROOT / "data" / "stimuli_v2.jsonl"
    results["input_hashes"]["data/stimuli_v2.jsonl"] = sha256_file(stim_path)
    results["input_hashes"]["tools/analysis_v2.py (reference, not imported)"] = \
        sha256_file(ROOT / "tools" / "analysis_v2.py")

    any_ci_includes_zero = []

    for tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        results["input_hashes"][f"traces/{base}.scored.jsonl"] = sha256_file(path)
        recs = [json.loads(ln) for ln in path.read_text().splitlines()]
        model = recs[0]["model"]

        pairs = build_confirmatory_pairs(recs, stim)
        n, b, c, p = mcnemar(pairs)

        loo = {}
        for fr in FRAMES:
            sub = {k: v for k, v in pairs.items() if k[0] != fr}
            loo[fr] = dict(zip(("n", "b", "c", "p"), mcnemar(sub)))

        two_out = {k: v for k, v in pairs.items()
                   if k[0] not in ("vehicle", "workplace")}
        n2, b2, c2, p2 = mcnemar(two_out)

        boot_frames = bootstrap_frames(pairs, SEED, N_BOOT)
        boot_clusters = bootstrap_clusters_within_frame(pairs, SEED, N_BOOT)

        carrier_bc = {}
        for carrier in CARRIERS:
            cpairs = build_carrier_pairs(recs, stim, carrier)
            cn, cb, cc, cp = mcnemar(cpairs)
            carrier_bc[carrier] = {"n": cn, "b": cb, "c": cc, "p": cp}

        loo_p_max = max(v["p"] for v in loo.values())
        loo_min_b_minus_c = min(v["b"] - v["c"] for v in loo.values())
        col_result = {
            "tier": tier,
            "trace": base,
            "model": model,
            "confirmatory": {"n": n, "b": b, "c": c, "p": p},
            "leave_one_frame_out": loo,
            "leave_one_frame_out_worst_p": loo_p_max,
            "leave_one_frame_out_min_direction_margin_b_minus_c": loo_min_b_minus_c,
            "leave_two_frames_out_vehicle_workplace": {"n": n2, "b": b2, "c": c2, "p": p2},
            "bootstrap_frame_level": boot_frames,
            "bootstrap_cluster_within_frame": boot_clusters,
            "carrier_e1_b_c": carrier_bc,
        }
        results["columns"].append(col_result)

        if tier == "confirmatory":
            ci_flags = {
                "model": model,
                "loo_p_max_over_0.05_over_3": loo_p_max >= 0.05 / 3,
                "loo_direction_ever_flips": loo_min_b_minus_c <= 0,
                "frame_boot_risk_diff_ci_includes_zero":
                    boot_frames["risk_diff_ci95"][0] <= 0 <= boot_frames["risk_diff_ci95"][1],
                "cluster_boot_risk_diff_ci_includes_zero":
                    boot_clusters["risk_diff_ci95"][0] <= 0 <= boot_clusters["risk_diff_ci95"][1],
            }
            any_ci_includes_zero.append(ci_flags)

    results["confirmatory_zero_inclusion_summary"] = any_ci_includes_zero

    out_path = OUT / "frame_jackknife_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    for flag in any_ci_includes_zero:
        print(flag)


if __name__ == "__main__":
    main()
