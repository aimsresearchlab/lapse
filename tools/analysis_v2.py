"""Confirmatory + robustness analysis over the v2 scored columns (spec §4).

Deterministic: reads committed `traces/*.scored.jsonl` + `data/stimuli_v2.jsonl`
only; no model calls. Implements the spec §4 plan; implementation choices not
pinned by the spec are stated here and in the report header:

- DESTROYED (e1 note label) = COERCED-STATIVE or MANUFACTURE.
  RESIDUAL and WITNESS-DROPPED records are excluded from numerator and
  denominator (pairs containing one are dropped); counts disclosed.
- Confirmatory McNemar (per API model): e1_primary component, c0_original
  carrier, gap=stale, prog vs simple paired on (frame, cluster) — 128 pairs.
  Directional prediction prog>simple: one-sided exact binomial on discordant
  pairs, P(X >= b | b+c, 0.5); Holm correction across the 3 API models.
- Conditional carrier sign test (per API model): carrier_e1 component
  (c1..c5), per-carrier destruction differential (prog − simple rate);
  exact one-sided sign test across carriers, conditional on non-tied
  carriers.
- R1 (named robustness, NOT in the confirmatory family): lexeme_subject
  component, lexeme_pair == "same", prog vs simple paired on
  (frame, cluster, subject); same McNemar; per-frame presentation.
- Per-frame presentation (spec: frames are the presentation, pooled McNemar
  is the test): per-frame destruction rates for e1_primary c0 stale.
- Boundary floor: behavioral binary non-commit 0/n cells reported with
  one-sided 95% Clopper-Pearson upper bound (rule of three on exact form).

Usage: python3 tools/analysis_v2.py > research/ANALYSIS_V2_<date>.txt
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESTROYED = {"COERCED-STATIVE", "MANUFACTURE"}
EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
NONCOMMIT = {"HEDGE", "TEMPLATE"}

COLUMNS = [  # (tier, trace basename)
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
    """One-sided exact P(X >= b), X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (order-preserving)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound, exact tail search."""
    if x >= n:
        return 1.0
    lo, hi = x / n if n else 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(0, x + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def mcnemar(pairs: dict, lab: dict) -> tuple[int, int, int, float]:
    """pairs: key -> {form: destroyed_bool}. Returns (n_pairs, b, c, one-sided p)."""
    b = c = n = 0
    for v in pairs.values():
        if "prog" not in v or "simple" not in v:
            continue
        n += 1
        if v["prog"] and not v["simple"]:
            b += 1
        elif v["simple"] and not v["prog"]:
            c += 1
    return n, b, c, binom_ge(b, b + c)


def main() -> None:
    stim = {}
    for ln in (ROOT / "data" / "stimuli_v2.jsonl").read_text().splitlines():
        c = json.loads(ln)
        stim[c["case_id"]] = c

    conf_ps, conf_rows = [], []
    print("=" * 72)
    print("LAPSE v2 analysis (spec §4) — deterministic over committed files")
    print("=" * 72)

    for tier, base in COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        recs = [json.loads(ln) for ln in path.read_text().splitlines()]
        model = recs[0]["model"]
        print(f"\n### {model}  [{tier}]  ({base})")

        e1 = [r for r in recs if r["arm"] == "e1" and r["gap"] == "stale"
              and r["form"] in ("prog", "simple")]
        excl = sum(1 for r in e1 if r["label"] in EXCLUDED)

        # --- Test 1: confirmatory McNemar (e1_primary, c0, frame×cluster) ---
        pairs: dict = {}
        for r in e1:
            s = stim[r["case_id"]]
            if s["component"] != "e1_primary" or r["carrier"] != "c0_original":
                continue
            if r["label"] in EXCLUDED:
                pairs.setdefault((r["frame"], r["cluster"]), {})["DROP"] = True
                continue
            pairs.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = (
                r["label"] in DESTROYED)
        pairs = {k: v for k, v in pairs.items() if "DROP" not in v}
        n, b, c_, p = mcnemar(pairs, {})
        print(f"McNemar e1_primary c0 stale prog-vs-simple: pairs={n} "
              f"discordant b(prog-only)={b} c(simple-only)={c_} one-sided p={p:.3e}")
        if tier == "confirmatory":
            conf_ps.append(p)
            conf_rows.append((model, n, b, c_))

        # --- Test 2: conditional carrier sign test (carrier_e1, c1..c5) ---
        signs = []
        for carrier in ("c1_fact_final", "c2_bare", "c3_embedded_mid",
                        "c4_formal", "c5_fact_list"):
            rates = {}
            for form in ("prog", "simple"):
                sub = [r for r in e1 if r["carrier"] == carrier
                       and r["form"] == form and r["label"] not in EXCLUDED]
                rates[form] = (sum(r["label"] in DESTROYED for r in sub),
                               len(sub))
            if rates["prog"][1] and rates["simple"][1]:
                signs.append(rates["prog"][0] / rates["prog"][1]
                             - rates["simple"][0] / rates["simple"][1])
        if signs:
            pos = sum(1 for d in signs if d > 0)
            tied = sum(1 for d in signs if d == 0)
            psign = binom_ge(pos, len(signs) - tied)
            print(f"Carrier sign test (c1..c5): diffs={[f'{d:+.3f}' for d in signs]} "
                  f"positive={pos}/{len(signs) - tied} one-sided p={psign:.4f}")
        else:
            print("Carrier sign test: no carrier cells in this column (subset)")

        # --- R1: same-lexeme minimal pairs (robustness, not confirmatory) ---
        r1_pairs: dict = {}
        r1_frames: dict = {}
        for r in e1:
            s = stim[r["case_id"]]
            if s["component"] != "lexeme_subject" or s.get("lexeme_pair") != "same":
                continue
            if r["label"] in EXCLUDED:
                continue
            key = (r["frame"], r["cluster"], s.get("subject"))
            r1_pairs.setdefault(key, {})[r["form"]] = r["label"] in DESTROYED
            r1_frames.setdefault(r["frame"], []).append(r)
        n1, b1, c1, p1 = mcnemar(r1_pairs, {})
        print(f"R1 same-lexeme McNemar: pairs={n1} b={b1} c={c1} "
              f"one-sided p={p1:.3e}" if n1 else "R1: no paired data")
        for fr in sorted(r1_frames):
            sub = r1_frames[fr]
            for form in ("prog", "simple"):
                ss = [r for r in sub if r["form"] == form]
                if ss:
                    d = sum(r["label"] in DESTROYED for r in ss)
                    print(f"  R1 {fr} {form}: destroyed {d}/{len(ss)}")

        # --- Per-frame presentation (e1_primary c0 stale) ---
        print("Per-frame destruction (e1_primary c0 stale) prog | simple:")
        for fr in sorted({r["frame"] for r in e1}):
            row = []
            for form in ("prog", "simple"):
                ss = [r for r in e1 if r["frame"] == fr and r["form"] == form
                      and r["carrier"] == "c0_original"
                      and stim[r["case_id"]]["component"] == "e1_primary"
                      and r["label"] not in EXCLUDED]
                d = sum(r["label"] in DESTROYED for r in ss)
                row.append(f"{d}/{len(ss)}")
            print(f"  {fr:16s} {row[0]:>7s} | {row[1]:>7s}")
        print(f"e1 stale excluded (RESIDUAL/WITNESS-DROPPED): {excl}")

        # --- Boundary floor: behavioral binary non-commit per gap ---
        beh = [r for r in recs if r["arm"] == "behavioral"]
        if beh:
            print("Behavioral non-commit per gap (0/n -> 95% CP upper bound):")
            for gap in ("fresh", "near", "boundary", "expired_soon", "stale",
                        "stale_long"):
                ss = [r for r in beh if r.get("gap") == gap
                      and r["label"] in ({"PROCEED"} | NONCOMMIT)]
                nc = sum(r["label"] in NONCOMMIT for r in ss)
                ub = f" (UB {cp_upper(0, len(ss)):.4f})" if nc == 0 and ss else ""
                print(f"  {gap:13s} {nc}/{len(ss)}{ub}")

    # --- Holm over the 3 confirmatory McNemars ---
    print("\n" + "=" * 72)
    print("CONFIRMATORY FAMILY (3 API models, McNemar, Holm-adjusted):")
    for (model, n, b, c_), p, ap in zip(conf_rows, conf_ps, holm(conf_ps)):
        verdict = "PASS" if ap < 0.05 and b > c_ else "FAIL"
        print(f"  {model:28s} pairs={n:3d} b={b:3d} c={c_:2d} "
              f"p={p:.3e} holm={ap:.3e} direction={'+' if b > c_ else '-'} {verdict}")
    npass = sum(1 for (m, n, b, c_), ap in zip(conf_rows, holm(conf_ps))
                if ap < 0.05 and b > c_)
    print(f"  >=2/3 required: {npass}/3 -> "
          f"{'CONFIRMED' if npass >= 2 else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    main()
