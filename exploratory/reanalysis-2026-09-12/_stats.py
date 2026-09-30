"""Shared stdlib-only stats helpers for the 2026-09-12 deterministic reanalysis.

No API/model calls, no third-party packages. Exact methods only (math.comb).
Some of this duplicates small functions already in tools/analysis_v2.py
(binom_ge, cp_upper) by design: each script in this directory is meant to be
independently re-runnable without a sys.path dependency on tools/, and the
functions are a few lines each.
"""
from __future__ import annotations

import hashlib
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def binom_ge(b: int, n: int) -> float:
    """One-sided exact P(X >= b), X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def binom_le(b: int, n: int) -> float:
    """One-sided exact P(X <= b), X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(0, b + 1)) / 2 ** n


def two_sided_binom_p(b: int, n: int) -> float:
    """Two-sided exact p for a discordant-pair sign test, X ~ Binomial(n,0.5).

    Symmetric distribution under p=0.5, so the standard two-sided exact test
    is 2*min(P(X>=b), P(X<=b)), capped at 1.
    """
    if n == 0:
        return float("nan")
    return min(1.0, 2 * min(binom_ge(b, n), binom_le(b, n)))


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound, exact tail search."""
    if n == 0:
        return float("nan")
    if x >= n:
        return 1.0
    lo, hi = x / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(0, x + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def cp_lower(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson lower bound, exact tail search."""
    if n == 0:
        return float("nan")
    if x <= 0:
        return 0.0
    lo, hi = 0.0, x / n
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(x, n + 1))
        if tail > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def clopper_pearson_95(x: int, n: int) -> tuple[float, float]:
    """Two-sided exact 95% Clopper-Pearson CI (alpha/2 = 0.025 each tail)."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = cp_lower(x, n, alpha=0.025)
    hi = cp_upper(x, n, alpha=0.025)
    return (lo, hi)


def cohen_kappa(pairs: list[tuple]) -> tuple[float, float, float]:
    """Unweighted Cohen's kappa over paired nominal labels.

    pairs: list of (label_a, label_b) for the same item across two raters
    (here: two replicate calls to the same reader). Returns (kappa, po, pe).
    kappa is None-like (nan) if pe == 1 (degenerate, no expected disagreement
    to explain away).
    """
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    po = sum(1 for a, b in pairs if a == b) / n
    cats = sorted({a for a, b in pairs} | {b for a, b in pairs})
    ca = {c: 0 for c in cats}
    cb = {c: 0 for c in cats}
    for a, b in pairs:
        ca[a] += 1
        cb[b] += 1
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return (float("nan"), po, pe)
    kappa = (po - pe) / (1 - pe)
    return (kappa, po, pe)
