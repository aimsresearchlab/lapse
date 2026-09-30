"""(1) Perfect-progressive vs perfect-simple memory-write pair — deterministic
reanalysis of committed v2 traces. NO API/model calls; stdlib only.

Reads (never writes outside this directory):
  - data/stimuli_v2.jsonl              (case fields, incl. component, history)
  - traces/<base>.scored.jsonl         (cascade labels; already carry frame,
    cluster, carrier, gap, form, component, label per-row -- confirmed by
    inspection, so no join with stimuli is needed for those fields; the join
    is needed only for the `history` text used in part (b))
  - traces/<base>.scored.judged.local-seed-oss-36b-instruct-pv2.jsonl
    (used only to check whether e1 rows were ever judged; per
    research/SCORED_DEEPSEEK_V2_2026-08-27.md, judging covers
    behavioral/e2/l2e2 only)

Pairing: arm e1, component "perf", gap "stale" (the only gap the e1 arm of
the perf component has -- confirmed empirically: e1/perf stimuli are all
gap=stale; perf_prog also has a gap=fresh cell but only under arm=behavioral,
out of scope here), carrier c0_original, forms perf_prog vs perf_sim, paired
on (frame, cluster). Checked empirically: within the perf component, each
(frame, cluster, form) triple has exactly one record (household's 20 records
are all subject=third with no first-subject counterpart at the same cluster,
and the other 7 frames are all subject=first) -- so keying on (frame,
cluster) alone, as the task specifies, causes no silent collisions. This
script asserts that and would report a nonzero collision count if it ever
occurred.

COLUMNS is copied verbatim (2026-09-12) from tools/analysis_v2.py's COLUMNS
list (that file is read-only input here, not imported, so this script has no
runtime dependency on it).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import ROOT, clopper_pearson_95, sha256_file  # noqa: E402

HERE = Path(__file__).resolve().parent
DESTROYED = {"COERCED-STATIVE", "MANUFACTURE"}
EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
EXTRACTED = "EXTRACTED-AS-CURRENT"

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
# "the four frontier-subset traces run-v2-full-frontier-*.scored.jsonl" per
# the task; sonnet5 is also in COLUMNS above (frontier-subset tier) -- kept
# here too so the check is uniform and self-documenting.
FRONTIER_SUBSET_BASES = [
    "run-v2-full-frontier-gemini31pro-20260827-015332",
    "run-v2-full-frontier-grok46-20260827-015925",
    "run-v2-full-frontier-qwen38max-20260827-021106",
    "run-v2-full-frontier-sonnet5-20260826-104357",
]
DEEPSEEK_BASE = "run-v2-full-20260825-010312"

DESTROYED_KEY = "prog_destroyed"
EXTRACTED_KEY = "sim_extracted_as_current"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    input_files: set[Path] = set()

    stim_path = ROOT / "data" / "stimuli_v2.jsonl"
    input_files.add(stim_path)
    stim = {c["case_id"]: c for c in load_jsonl(stim_path)}

    result: dict = {
        "generated_by": "exploratory/reanalysis-2026-09-12/perf_pair.py",
        "destroyed_labels": sorted(DESTROYED),
        "excluded_labels": sorted(EXCLUDED),
        "columns": {},
        "frontier_subset_perf_check": {},
        "deepseek_examples": {},
        "implementation_choices": [
            "History line for part (b) = the line(s) of the case's `history` "
            "field (from data/stimuli_v2.jsonl) containing the witness token, "
            "case-insensitive; falls back to the full history text if no "
            "line matches. Not pinned by the task spec ('example source "
            "history line' is singular and the field itself is multi-line).",
            "Part (d): both RESIDUAL and WITNESS-DROPPED are excluded from "
            "the perf_sim denominator (EXCLUDED set), not RESIDUAL alone as "
            "literally written in the task -- WITNESS-DROPPED is a possible "
            "score_note() output for perf_sim too (checked before the "
            "form-specific branch in tools/score.py). Counts of each "
            "excluded label are reported separately per column.",
            "Part (e) cross-tab excludes any (frame,cluster) pair missing "
            "one side, or with an EXCLUDED label on either side; both counts "
            "are disclosed per column.",
            "Part (g) restricts the e1_primary comparison to the exact "
            "(frame,cluster) key set used in the perf pairs (clusters 0-9 of "
            "the 8 frames), not all 16 e1_primary clusters.",
        ],
    }

    for tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        input_files.add(path)
        recs = load_jsonl(path)
        model = recs[0]["model"]

        e1perf = [r for r in recs if r["arm"] == "e1" and r["component"] == "perf"
                  and r["gap"] == "stale" and r["carrier"] == "c0_original"
                  and r["form"] in ("perf_prog", "perf_sim")]

        # (a) full label distribution per form
        dist = {form: dict(sorted(Counter(
            r["label"] for r in e1perf if r["form"] == form).items()))
            for form in ("perf_prog", "perf_sim")}

        # detector provenance: does the judged file (if any) contain e1 rows?
        judged_path = ROOT / "traces" / f"{base}.scored.judged.local-seed-oss-36b-instruct-pv2.jsonl"
        if judged_path.exists():
            input_files.add(judged_path)
            jrecs = load_jsonl(judged_path)
            e1_in_judged = sum(1 for r in jrecs if r.get("arm") == "e1")
            detector = ("cascade-only (judged file exists but has 0 e1 rows)"
                        if e1_in_judged == 0 else
                        f"cascade+judge ({e1_in_judged} e1 rows found in judged file)")
        else:
            detector = "cascade-only (no judged file for this column)"

        # (c) perf_prog destroyed rate over non-excluded, CI95
        pp = [r for r in e1perf if r["form"] == "perf_prog"]
        pp_nonexcl = [r for r in pp if r["label"] not in EXCLUDED]
        pp_destroyed = sum(1 for r in pp_nonexcl if r["label"] in DESTROYED)
        pp_lo, pp_hi = clopper_pearson_95(pp_destroyed, len(pp_nonexcl))

        # (d) perf_sim EXTRACTED-AS-CURRENT rate over non-excluded, CI95
        ps = [r for r in e1perf if r["form"] == "perf_sim"]
        ps_excl_counts = dict(Counter(r["label"] for r in ps if r["label"] in EXCLUDED))
        ps_nonexcl = [r for r in ps if r["label"] not in EXCLUDED]
        ps_extracted = sum(1 for r in ps_nonexcl if r["label"] == EXTRACTED)
        ps_lo, ps_hi = clopper_pearson_95(ps_extracted, len(ps_nonexcl))

        # pairing on (frame, cluster) -- collision-checked
        pair_map: dict = defaultdict(dict)
        collisions = 0
        for r in e1perf:
            key = (r["frame"], r["cluster"])
            if r["form"] in pair_map[key]:
                collisions += 1
            pair_map[key][r["form"]] = r["label"]

        # (e) descriptive cross-tab: perf_prog destroyed x perf_sim extracted
        cross = {"destroyed_and_extracted": 0, "destroyed_and_not_extracted": 0,
                  "not_destroyed_and_extracted": 0, "neither": 0}
        used_pairs = 0
        dropped_pairs = 0
        for v in pair_map.values():
            if "perf_prog" not in v or "perf_sim" not in v:
                dropped_pairs += 1
                continue
            if v["perf_prog"] in EXCLUDED or v["perf_sim"] in EXCLUDED:
                dropped_pairs += 1
                continue
            used_pairs += 1
            a = v["perf_prog"] in DESTROYED
            b = v["perf_sim"] == EXTRACTED
            if a and b:
                cross["destroyed_and_extracted"] += 1
            elif a and not b:
                cross["destroyed_and_not_extracted"] += 1
            elif not a and b:
                cross["not_destroyed_and_extracted"] += 1
            else:
                cross["neither"] += 1

        # (f) per-frame counts
        per_frame = {}
        for fr in sorted({r["frame"] for r in e1perf}):
            pp_f = [r for r in pp_nonexcl if r["frame"] == fr]
            ps_f = [r for r in ps_nonexcl if r["frame"] == fr]
            per_frame[fr] = {
                "perf_prog_destroyed": sum(1 for r in pp_f if r["label"] in DESTROYED),
                "perf_prog_n_nonexcluded": len(pp_f),
                "perf_sim_extracted_as_current": sum(1 for r in ps_f if r["label"] == EXTRACTED),
                "perf_sim_n_nonexcluded": len(ps_f),
            }

        # (g) same-clusters e1_primary prog/simple destruction for comparison
        keys = set(pair_map.keys())
        e1primary = [r for r in recs if r["arm"] == "e1" and r["component"] == "e1_primary"
                     and r["gap"] == "stale" and r["carrier"] == "c0_original"
                     and r["form"] in ("prog", "simple")
                     and (r["frame"], r["cluster"]) in keys]
        g_collisions = 0
        g_map: dict = defaultdict(dict)
        for r in e1primary:
            k2 = (r["frame"], r["cluster"], r["form"])
            if k2 in g_map:
                g_collisions += 1
            g_map[k2] = r["label"]
        g = {}
        for form in ("prog", "simple"):
            sub = [lab for (fr, cl, f2), lab in g_map.items() if f2 == form]
            sub_nonexcl = [lab for lab in sub if lab not in EXCLUDED]
            d = sum(1 for lab in sub_nonexcl if lab in DESTROYED)
            g[form] = {"destroyed": d, "n_nonexcluded": len(sub_nonexcl),
                       "excluded": len(sub) - len(sub_nonexcl)}

        result["columns"][model] = {
            "tier": tier,
            "trace_base": base,
            "label_distribution": dist,
            "e1_detector": detector,
            "perf_prog_destroyed_rate": {
                "destroyed": pp_destroyed, "n_nonexcluded": len(pp_nonexcl),
                "excluded": len(pp) - len(pp_nonexcl),
                "rate": pp_destroyed / len(pp_nonexcl) if pp_nonexcl else None,
                "ci95": [pp_lo, pp_hi],
            },
            "perf_sim_extracted_as_current_rate": {
                "extracted": ps_extracted, "n_nonexcluded": len(ps_nonexcl),
                "excluded_by_label": ps_excl_counts,
                "rate": ps_extracted / len(ps_nonexcl) if ps_nonexcl else None,
                "ci95": [ps_lo, ps_hi],
            },
            "pairing_collisions_frame_cluster": collisions,
            "cross_tab_prog_destroyed_x_sim_extracted": cross,
            "cross_tab_pairs_used": used_pairs,
            "cross_tab_pairs_dropped": dropped_pairs,
            "cross_tab_note": "descriptive cross-tab only, NOT a directional "
                              "McNemar -- perf_prog 'destroyed' and perf_sim "
                              "'extracted-as-current' are different error "
                              "definitions over different source aspect (see "
                              "tools/score.py score_note()).",
            "per_frame": per_frame,
            "e1_primary_same_clusters_comparison": g,
        }

    for base in FRONTIER_SUBSET_BASES:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        input_files.add(path)
        recs = load_jsonl(path)
        n_perf = sum(1 for r in recs if r["arm"] == "e1" and r.get("form") in ("perf_prog", "perf_sim"))
        result["frontier_subset_perf_check"][base] = {
            "e1_perf_rows_found": n_perf,
            "included_in_columns_above": n_perf > 0,
        }

    # (b) DeepSeek examples: 2 per (form, label), truncated to 200 chars
    path = ROOT / "traces" / f"{DEEPSEEK_BASE}.scored.jsonl"
    recs = load_jsonl(path)
    e1perf_ds = [r for r in recs if r["arm"] == "e1" and r["component"] == "perf"
                 and r["gap"] == "stale" and r["carrier"] == "c0_original"
                 and r["form"] in ("perf_prog", "perf_sim")]
    examples: dict = defaultdict(list)
    for r in e1perf_ds:
        key = f"{r['form']}|{r['label']}"
        if len(examples[key]) >= 2:
            continue
        s = stim.get(r["case_id"], {})
        hist = s.get("history", "")
        wt = r["wit_token"].lower()
        matching_lines = [ln for ln in hist.splitlines() if wt in ln.lower()]
        hist_snippet = " / ".join(matching_lines) if matching_lines else hist
        examples[key].append({
            "case_id": r["case_id"],
            "history_line": hist_snippet[:200],
            "response": r["response"][:200],
        })
    result["deepseek_examples"] = dict(sorted(examples.items()))

    out_path = HERE / "perf_pair_results.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(input_files)}
    (HERE / "perf_pair_input_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"wrote {HERE / 'perf_pair_input_hashes.json'}")


if __name__ == "__main__":
    main()
