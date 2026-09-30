"""Multiplicity correction over three exploratory reader-side test families
(item T0.3 of research/OVERNIGHT_PLAN_2026-09-12.md):

1. The 15 "channel-survival" McNemar tests behind paper/sections/06_pipeline.tex
   Table tab:channel-survival (3 readers x 5 pipeline/cue subsets).
2. The 3 dated-note reader field-fill tests reported in
   paper/sections/05_write_read.tex ("Exploratory dated-note task"
   paragraph: DeepSeek 26:3, GLM 17:3, Luna 23:15).
3. The minimal-edit field-fill tests reported in
   research/UNRECOVERABILITY_2026-09-09.md P4 (also summarized in
   paper/sections/05_write_read.tex "Minimal edits to pipeline memories").

Deterministic, Python-3-standard-library only. No API/model calls.

Source-data recomputation, not table transcription: every McNemar test
below is recomputed from the per-item reader trace files (raw discordant
pairs), not from the rounded p-values printed in the paper or in the
research/*.md records. The trace files and pairing logic (id string
substitution between an arm's PROG/FLAT/EDIT variant and its SIMPLE/FLAT
partner, keyed on task "A2" = field-fill, COMMIT/ABSTAIN parse) are copied
from exploratory/unrecoverability-2026-09-08/analyze_pipeline.py,
analyze_g.py and exploratory/read-time-ladder-2026-09-08/analyze_ladder.py
(read for reference, not imported, so this script has no dependency
outside this directory). This satisfies the task's "recompute ... from
their source data" instruction; the "if raw data are not locatable" fallback
(recomputing from the paper's own discordant counts) was not needed.

Implementation choices NOT pinned by the task spec (also flagged in
RESULTS_E.md):

- Two-sided exact McNemar p (binomial sign test on discordant pairs,
  p=0.5), matching the convention already used in
  exploratory/unrecoverability-2026-09-08/analyze_pipeline.py's `mcn()`
  (scipy `binomtest`, two-sided). Reimplemented here without scipy as
  `two_sided_binom_p` in `_stats.py`: 2*min(P(X>=b), P(X<=b)), capped at 1,
  which is exact for a symmetric (p=0.5) binomial.
- Family (a) for the 15 channel-survival tests = all 15 pooled. Family (b)
  = the 5 tests belonging to one reader. For the 3 dated-note tests and the
  6 minimal-edit tests (2 contrasts x 3 readers), there is no natural
  reader-vs-pooled split as clean as the 15-test case; family (a) = all
  tests in that set, family (b) = per-reader (dated-note: 1 test per
  reader, so Holm/BH within a family of 1 is a no-op, reported anyway for
  uniformity; minimal-edit: the 2 tests belonging to one reader).
- BH q-values are reported for family (a) of each set (the pooled family);
  family-(b) BH is computed too and stored in the JSON but omitted from the
  markdown table for space (task explicitly names Holm under both
  groupings but only one "BH q" column).
- "Survives at alpha=.05" is evaluated against the Holm-adjusted p in the
  SMALLEST applicable family for that test (i.e. within-reader family (b)
  for the 15-test and minimal-edit sets, since that is arguably the
  family a reader most naturally reports "reliable" against) AND against
  the fully-pooled family (a); both are shown so neither reading is hidden.

Usage: python3 exploratory/reanalysis-2026-09-12/multiplicity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import sha256_file, two_sided_binom_p  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
UNREC = ROOT / "exploratory" / "unrecoverability-2026-09-08"
LADDER = ROOT / "exploratory" / "read-time-ladder-2026-09-08"

READERS = {
    "deepseek": ("deepseek-v4-flash (official)", "deepseek-deepseek-v4-flash-official"),
    "luna": ("gpt-5.6-luna (OpenRouter)", "openai-gpt-5-6-luna"),
    "glm": ("glm-5.2 (OpenRouter)", "z-ai-glm-5-2"),
}


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = ps[i] * m / (rank + 1)
        running_min = min(running_min, val)
        q[i] = min(1.0, running_min)
    return q


def load_jsonl(path: Path) -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()}


def mcnemar_contrast(R: dict, armA: str, armB: str, task: str, sub: str | None) -> dict:
    """b = armA ABSTAIN & armB COMMIT (predicted direction), c = reverse."""
    b = c = 0
    nA = 0
    for r in R.values():
        if r["arm"] != armA or r["task"] != task:
            continue
        if sub is not None and r.get("writer_label") != sub:
            continue
        oid = r["id"].replace(armA, armB, 1)
        o = R.get(oid)
        if not o:
            continue
        nA += 1
        if r["parse"] == "ABSTAIN" and o["parse"] == "COMMIT":
            b += 1
        elif r["parse"] == "COMMIT" and o["parse"] == "ABSTAIN":
            c += 1
    n_pairs = b + c
    p = two_sided_binom_p(b, n_pairs)
    return {"n_matched": nA, "b": b, "c": c, "p_raw": p}


def channel_survival_family(hashes: dict) -> list[dict]:
    rows = []
    specs = [
        ("mem0", "progressive verb", "PIPE-PROG", "PIPE-SIMPLE", UNREC, "reader_trace_{}.jsonl", "PRESERVED"),
        ("Graphiti", "progressive verb", "G-TEXT-PROG", "G-TEXT-SIMPLE", UNREC, "reader_g_trace_{}.jsonl", "PRESERVED"),
        ("Graphiti", "none measured", "G-TEXT-PROG", "G-TEXT-SIMPLE", UNREC, "reader_g_trace_{}.jsonl", "FLATTENED"),
        ("Letta", "“Currently” prefix", "L-TEXT-PROG", "L-TEXT-SIMPLE", UNREC, "reader_g_trace_{}.jsonl", "PRESERVED"),
        ("Letta", "none measured", "L-TEXT-PROG", "L-TEXT-SIMPLE", UNREC, "reader_g_trace_{}.jsonl", "FLATTENED"),
    ]
    for reader_key, (reader_label, fname_key) in READERS.items():
        for pipeline, cue, armA, armB, folder, pattern, sub in specs:
            path = folder / pattern.format(fname_key)
            hkey = f"exploratory/{path.relative_to(ROOT/'exploratory')}"
            if hkey not in hashes:
                hashes[hkey] = sha256_file(path)
            R = load_jsonl(path)
            res = mcnemar_contrast(R, armA, armB, "A2", sub)
            rows.append({
                "family": "channel_survival",
                "reader": reader_label,
                "reader_key": reader_key,
                "pipeline": pipeline,
                "cue": cue,
                **res,
            })
    return rows


def dated_note_family(hashes: dict) -> list[dict]:
    rows = []
    for reader_key, (reader_label, fname_key) in READERS.items():
        path = LADDER / f"trace_{fname_key}.jsonl"
        hkey = f"exploratory/{path.relative_to(ROOT/'exploratory')}"
        if hkey not in hashes:
            hashes[hkey] = sha256_file(path)
        R = load_jsonl(path)
        b = c = 0
        n_matched = 0
        for r in R.values():
            if r["form"] != "PROG" or r["gap"] != "stale" or r["task"] != "A2":
                continue
            flat_id = f"A-{r['frame']}-{r['cluster']:02d}-FLAT-{r['gap']}-{r['task']}"
            o = R.get(flat_id)
            if not o:
                continue
            n_matched += 1
            if o["parse"] == "COMMIT" and r["parse"] == "ABSTAIN":
                b += 1
            elif o["parse"] == "ABSTAIN" and r["parse"] == "COMMIT":
                c += 1
        n_pairs = b + c
        p = two_sided_binom_p(b, n_pairs)
        rows.append({
            "family": "dated_note_field_fill",
            "reader": reader_label,
            "reader_key": reader_key,
            "pipeline": "hand-written dated note (Part A ladder)",
            "cue": "progressive verb (PROG vs FLAT, stale gap, all frames)",
            "n_matched": n_matched, "b": b, "c": c, "p_raw": p,
        })
    return rows


def minimal_edit_family(hashes: dict) -> list[dict]:
    rows = []
    specs = [
        ("EDIT-PROG restored on flattened frames", "EDIT-PROG", "PIPE-SIMPLE", "FLATTENED"),
        ("EDIT-FLAT hand-flattened vs kept", "EDIT-FLAT", "PIPE-PROG", "PRESERVED"),
    ]
    for reader_key, (reader_label, fname_key) in READERS.items():
        path = UNREC / f"reader_trace_{fname_key}.jsonl"
        hkey = f"exploratory/{path.relative_to(ROOT/'exploratory')}"
        if hkey not in hashes:
            hashes[hkey] = sha256_file(path)
        R = load_jsonl(path)
        for name, armA, armB, sub in specs:
            res = mcnemar_contrast(R, armA, armB, "A2", sub)
            rows.append({
                "family": "minimal_edit",
                "reader": reader_label,
                "reader_key": reader_key,
                "pipeline": "mem0 (hand-edited notes)",
                "cue": name,
                **res,
            })
    return rows


def annotate_family(rows: list[dict], key: str) -> None:
    """Attach Holm/BH under family (a)=all rows, (b)=grouped by `key`."""
    ps_a = [r["p_raw"] for r in rows]
    holm_a = holm(ps_a)
    bh_a = benjamini_hochberg(ps_a)
    for r, ha, qa in zip(rows, holm_a, bh_a):
        r["holm_p_family_a_all"] = ha
        r["bh_q_family_a_all"] = qa

    groups: dict = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    for g in groups.values():
        ps_b = [r["p_raw"] for r in g]
        holm_b = holm(ps_b)
        bh_b = benjamini_hochberg(ps_b)
        for r, hb, qb in zip(g, holm_b, bh_b):
            r["holm_p_family_b_within_group"] = hb
            r["bh_q_family_b_within_group"] = qb


PAPER_CLAIMS = {
    # (family, reader_key, pipeline, cue) -> (paper's word, is-claimed-reliable)
    ("channel_survival", "deepseek", "mem0", "progressive verb"): True,
    ("channel_survival", "luna", "mem0", "progressive verb"): False,
    ("channel_survival", "glm", "mem0", "progressive verb"): True,
    ("channel_survival", "deepseek", "Graphiti", "progressive verb"): True,
    ("channel_survival", "luna", "Graphiti", "progressive verb"): True,
    ("channel_survival", "glm", "Graphiti", "progressive verb"): True,
    ("channel_survival", "deepseek", "Graphiti", "none measured"): False,
    ("channel_survival", "luna", "Graphiti", "none measured"): False,
    ("channel_survival", "glm", "Graphiti", "none measured"): False,
    ("channel_survival", "deepseek", "Letta", "“Currently” prefix"): True,
    ("channel_survival", "luna", "Letta", "“Currently” prefix"): False,
    ("channel_survival", "glm", "Letta", "“Currently” prefix"): True,
    ("channel_survival", "deepseek", "Letta", "none measured"): False,
    ("channel_survival", "luna", "Letta", "none measured"): False,
    ("channel_survival", "glm", "Letta", "none measured"): False,
    ("dated_note_field_fill", "deepseek", None, None): True,
    ("dated_note_field_fill", "luna", None, None): False,
    ("dated_note_field_fill", "glm", None, None): True,
    ("minimal_edit", "deepseek", None, "EDIT-PROG restored on flattened frames"): True,
    ("minimal_edit", "luna", None, "EDIT-PROG restored on flattened frames"): False,
    ("minimal_edit", "glm", None, "EDIT-PROG restored on flattened frames"): False,
}


def attach_paper_claim(r: dict) -> None:
    key = (r["family"], r["reader_key"],
           r.get("pipeline") if r["family"] == "channel_survival" else
           (None if r["family"] == "dated_note_field_fill" else r.get("pipeline")),
           r.get("cue") if r["family"] != "dated_note_field_fill" else None)
    r["paper_claims_reliable"] = PAPER_CLAIMS.get(key)


def main() -> None:
    hashes: dict = {}
    cs_rows = channel_survival_family(hashes)
    annotate_family(cs_rows, "reader_key")

    dn_rows = dated_note_family(hashes)
    annotate_family(dn_rows, "reader_key")

    me_rows = minimal_edit_family(hashes)
    annotate_family(me_rows, "reader_key")

    for rows in (cs_rows, dn_rows, me_rows):
        for r in rows:
            attach_paper_claim(r)
            r["survives_alpha05_holm_a"] = r["holm_p_family_a_all"] < 0.05
            r["survives_alpha05_holm_b"] = r["holm_p_family_b_within_group"] < 0.05
            r["survives_alpha05_bh_a"] = r["bh_q_family_a_all"] < 0.05

    results = {
        "channel_survival_15": cs_rows,
        "dated_note_field_fill_3": dn_rows,
        "minimal_edit": me_rows,
        "input_hashes": hashes,
    }
    out_path = OUT / "multiplicity_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")

    mismatches = []
    for rows in (cs_rows, dn_rows, me_rows):
        for r in rows:
            if r["paper_claims_reliable"] is None:
                continue
            if r["paper_claims_reliable"] != r["survives_alpha05_holm_a"]:
                mismatches.append((r["family"], r["reader_key"], r.get("pipeline"), r.get("cue"),
                                    r["paper_claims_reliable"], r["survives_alpha05_holm_a"]))
    print(f"{len(mismatches)} paper-claim vs Holm(all) mismatches:")
    for m in mismatches:
        print(" ", m)


if __name__ == "__main__":
    main()
