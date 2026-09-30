"""Secondary (SEC/EXP-tier) analysis over the v2 scored columns.

DESCRIPTIVE / ESTIMATION ONLY — nothing here enters the spec §4
confirmatory family (that family is closed: tools/analysis_v2.py,
research/ANALYSIS_V2_2026-08-27.md). Deterministic: committed
`traces/*.scored.jsonl` only; no model calls; stdlib only.

Implementation choices (spec's full stats-report text lost per
BUILD_NOTES_V2 §1; choices stated here and in the report header):

- A. EXTRACTION LEVER (L2): e1_primary c0 stale vs l2 stale, paired on
  (frame, cluster) within form. DESTROYED = COERCED-STATIVE or
  MANUFACTURE (EXTRACTED-AS-CURRENT counted separately, disclosed).
  Directional exact McNemar, prediction: lever reduces destruction
  (b = e1-only-destroyed > c = l2-only-destroyed).
- B. DOWNSTREAM LEVER (L2E2): e2 (own e1 notes) vs l2e2 (own l2 notes)
  at stale, paired the same way. Binary axis: NONCOMMIT = HEDGE or
  TEMPLATE; COMMIT = PROCEED or PROCEED-WITH-CAVEAT (matches §4 binary
  scoping). Directional: lever increases non-commit.
- C. ANCHOR FORCED-CHOICE: CHECK-FIRST rate per form x gap; UNPARSED
  excluded and disclosed. Paired prog-vs-simple at stale (exact
  McNemar, directional: prog draws more CHECK-FIRST).
- D. GAP GRADIENT: behavioral non-commit per gap x form (flatness
  presentation; no test).
- RESIDUAL / WITNESS-DROPPED excluded pairwise, counts disclosed.
- Frontier-tier subset columns lack all of these arms; printed as n/a.

Usage: python3 tools/analysis_v2_secondary.py > research/ANALYSIS_V2_SECONDARY_<date>.txt
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESTROYED = {"COERCED-STATIVE", "MANUFACTURE"}
EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
NONCOMMIT = {"HEDGE", "TEMPLATE"}

COLUMNS = [  # (tier, trace basename) — keep in sync with analysis_v2.py
    ("confirmatory", "run-v2-full-20260825-010312"),
    ("confirmatory", "run-v2-full-20260825-010323"),
    ("confirmatory", "run-v2-full-20260825-234313"),
    ("frontier-subset", "run-v2-full-frontier-sonnet5-20260826-104357"),
    ("local", "run-v2-full-local-qwen3-8b-20260813-060538"),
    ("local", "run-v2-full-local-mistral-small-3.2-24b-20260813-064704"),
    ("extension", "run-v2-full-local-gpt-oss-20b-20260826-052612"),
    ("extension", "run-v2-full-local-olmo-2-32b-instruct-20260826-060733"),
]


def binom_ge(b: int, n: int) -> float:
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def paired(recs_a, recs_b, bad_a, bad_b):
    """Pair on (frame, cluster, form); a-only-bad = b, b-only-bad = c."""
    m: dict = {}
    for r in recs_a:
        if r["label"] in EXCLUDED:
            m.setdefault((r["frame"], r["cluster"], r["form"]), {})["DROP"] = 1
        else:
            m.setdefault((r["frame"], r["cluster"], r["form"]), {})["a"] = bad_a(r)
    for r in recs_b:
        if r["label"] in EXCLUDED:
            m.setdefault((r["frame"], r["cluster"], r["form"]), {})["DROP"] = 1
        else:
            m.setdefault((r["frame"], r["cluster"], r["form"]), {})["b"] = bad_b(r)
    n = b = c = 0
    for v in m.values():
        if "DROP" in v or "a" not in v or "b" not in v:
            continue
        n += 1
        if v["a"] and not v["b"]:
            b += 1
        elif v["b"] and not v["a"]:
            c += 1
    return n, b, c, binom_ge(b, b + c)


def rate(recs, pred) -> str:
    ss = [r for r in recs if r["label"] not in EXCLUDED]
    k = sum(pred(r) for r in ss)
    return f"{k}/{len(ss)}" + (f" ({k / len(ss):.1%})" if ss else "")


def main() -> None:
    print("=" * 72)
    print("LAPSE v2 SECONDARY analysis (SEC/EXP tiers) — descriptive only")
    print("Confirmatory family untouched: see ANALYSIS_V2_2026-08-27.md")
    print("=" * 72)

    for tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        recs = [json.loads(ln) for ln in path.read_text().splitlines()]
        model = recs[0]["model"]
        print(f"\n### {model}  [{tier}]  ({base})")

        e1c0 = [r for r in recs if r["arm"] == "e1" and r["gap"] == "stale"
                and r["component"] == "e1_primary"
                and r["carrier"] == "c0_original"]
        l2 = [r for r in recs if r["arm"] == "l2"]
        e2 = [r for r in recs if r["arm"] == "e2"]
        l2e2 = [r for r in recs if r["arm"] == "l2e2"]
        anchor = [r for r in recs if r["arm"] == "anchor"]
        beh = [r for r in recs if r["arm"] == "behavioral"]

        if not l2:
            print("SEC arms absent (frontier-tier subset) — n/a")
            continue

        # --- A. Extraction lever ---
        print("A. EXTRACTION LEVER (e1 vs l2, destruction, stale):")
        for form in ("prog", "simple"):
            ea = [r for r in e1c0 if r["form"] == form]
            lb = [r for r in l2 if r["form"] == form]
            print(f"  {form:6s} e1 destroyed {rate(ea, lambda r: r['label'] in DESTROYED)}"
                  f" -> l2 destroyed {rate(lb, lambda r: r['label'] in DESTROYED)}")
        n, b, c, p = paired(e1c0, l2, lambda r: r["label"] in DESTROYED,
                            lambda r: r["label"] in DESTROYED)
        print(f"  paired (both forms): n={n} b(e1-only)={b} c(l2-only)={c} "
              f"one-sided p={p:.3e}")
        eac = sum(1 for r in e1c0 if r["label"] == "EXTRACTED-AS-CURRENT")
        lac = sum(1 for r in l2 if r["label"] == "EXTRACTED-AS-CURRENT")
        print(f"  EXTRACTED-AS-CURRENT (not in DESTROYED): e1={eac} l2={lac}")

        # --- B. Downstream lever ---
        print("B. DOWNSTREAM LEVER (e2 vs l2e2, non-commit, stale):")
        for form in ("prog", "simple"):
            ea = [r for r in e2 if r["form"] == form]
            lb = [r for r in l2e2 if r["form"] == form]
            print(f"  {form:6s} e2 non-commit {rate(ea, lambda r: r['label'] in NONCOMMIT)}"
                  f" -> l2e2 non-commit {rate(lb, lambda r: r['label'] in NONCOMMIT)}")
        n, b, c, p = paired(l2e2, e2, lambda r: r["label"] in NONCOMMIT,
                            lambda r: r["label"] in NONCOMMIT)
        print(f"  paired (both forms): n={n} b(l2e2-only-noncommit)={b} "
              f"c(e2-only)={c} one-sided p={p:.3e}")

        # --- C. Anchor forced-choice ---
        unp = sum(1 for r in anchor if r["label"] == "UNPARSED")
        print(f"C. ANCHOR FORCED-CHOICE (CHECK-FIRST rate; UNPARSED excl n={unp}):")
        ok = [r for r in anchor if r["label"] in ("SEND", "CHECK-FIRST")]
        for gap in ("fresh", "stale"):
            row = []
            for form in ("prog", "simple", "bound"):
                ss = [r for r in ok if r["gap"] == gap and r["form"] == form]
                k = sum(r["label"] == "CHECK-FIRST" for r in ss)
                row.append(f"{form} {k}/{len(ss)}")
            print(f"  {gap:6s} " + "  ".join(row))
        st = [r for r in ok if r["gap"] == "stale" and r["form"] in ("prog", "simple")]
        m: dict = {}
        for r in st:
            m.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = (
                r["label"] == "CHECK-FIRST")
        n = b = c = 0
        for v in m.values():
            if "prog" not in v or "simple" not in v:
                continue
            n += 1
            if v["prog"] and not v["simple"]:
                b += 1
            elif v["simple"] and not v["prog"]:
                c += 1
        print(f"  paired prog-vs-simple @stale: n={n} b(prog-only-CF)={b} "
              f"c(simple-only-CF)={c} one-sided p={binom_ge(b, b + c):.3e}")

        # --- D. Gap gradient ---
        print("D. GAP GRADIENT (behavioral non-commit per gap x form):")
        for gap in ("fresh", "near", "boundary", "expired_soon", "stale",
                    "stale_long"):
            row = []
            for form in sorted({r["form"] for r in beh}):
                ss = [r for r in beh if r["gap"] == gap and r["form"] == form
                      and r["label"] not in EXCLUDED]
                k = sum(r["label"] in NONCOMMIT for r in ss)
                row.append(f"{form} {k}/{len(ss)}")
            if any(not s.endswith("0/0") for s in row):
                print(f"  {gap:13s} " + "  ".join(row))


if __name__ == "__main__":
    main()
