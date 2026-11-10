"""Frame-level sign test for the v2 confirmatory matched-pair result.

Treats the frame (8 per writer) as the unit of analysis, the most
conservative answer to "the McNemar p-values ignore frame clustering". For
each writer x frame cell it counts discordant pairs b (progressive only
destroyed) and c (simple only destroyed), calls the cell positive if b > c,
negative if b < c, tied otherwise, and runs a one-sided exact sign test over
the non-tied cells. Holm adjustment over the three confirmatory writers.

A logistic mixed model with frame as a random effect was the planned
statistic (todo.md); it is not estimable here because the confirmatory
writers have zero simple-only pairs, so the within-pair log-odds is
unbounded (complete separation).

Pairing, labels, and inputs are those of
exploratory/reanalysis-2026-09-12/frame_jackknife.py (imported, unmodified).
Deterministic, standard library only, no API calls.

Usage: python3 exploratory/frame-sign-test-2026-10-09/frame_sign_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "exploratory" / "reanalysis-2026-09-12"))
import frame_jackknife as fj  # noqa: E402
from _stats import binom_ge, sha256_file  # noqa: E402


def cells(pairs: dict) -> dict:
    out = {}
    for f in fj.FRAMES:
        b = sum(1 for k, v in pairs.items() if k[0] == f and v["prog"] and not v["simple"])
        c = sum(1 for k, v in pairs.items() if k[0] == f and v["simple"] and not v["prog"])
        out[f] = {"b": b, "c": c}
    return out


def sign_test(cell_counts: list) -> dict:
    pos = sum(1 for x in cell_counts if x["b"] > x["c"])
    neg = sum(1 for x in cell_counts if x["b"] < x["c"])
    ties = len(cell_counts) - pos - neg
    return {"positive": pos, "negative": neg, "tied": ties,
            "p_one_sided": binom_ge(pos, pos + neg)}


def main() -> None:
    stim = fj.load_stim()
    res = {"input_hashes": {
        "data/stimuli_v2.jsonl": sha256_file(ROOT / "data" / "stimuli_v2.jsonl"),
        "exploratory/reanalysis-2026-09-12/frame_jackknife.py":
            sha256_file(ROOT / "exploratory" / "reanalysis-2026-09-12" / "frame_jackknife.py"),
    }, "columns": []}
    all_cells, conf_cells = [], []
    for tier, base in fj.COLUMNS:
        path = ROOT / "traces" / f"{base}.scored.jsonl"
        res["input_hashes"][f"traces/{base}.scored.jsonl"] = sha256_file(path)
        recs = [json.loads(ln) for ln in path.read_text().splitlines()]
        cc = cells(fj.build_confirmatory_pairs(recs, stim))
        st = sign_test(list(cc.values()))
        res["columns"].append({"tier": tier, "trace": base, "model": recs[0]["model"],
                               "cells": cc, "sign_test": st})
        all_cells += cc.values()
        if tier == "confirmatory":
            conf_cells += cc.values()

    # Holm over the three confirmatory writers.
    conf = [c for c in res["columns"] if c["tier"] == "confirmatory"]
    order = sorted(range(len(conf)), key=lambda i: conf[i]["sign_test"]["p_one_sided"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(conf) - rank) * conf[i]["sign_test"]["p_one_sided"])
        running = max(running, adj)
        conf[i]["sign_test"]["p_holm"] = running

    res["pooled_confirmatory"] = sign_test(conf_cells)
    res["pooled_all_columns"] = sign_test(all_cells)
    (HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    for c in res["columns"]:
        print(c["model"], c["sign_test"])
    print("pooled confirmatory", res["pooled_confirmatory"])
    print("pooled all", res["pooled_all_columns"])


if __name__ == "__main__":
    main()
