"""Every main-test number the paper reports, under the frozen and the aspect-flip definitions ($0).

Reads traces/*.scored.jsonl (frozen labels) and root/traces/*.scored.jsonl (relabel.py output).
Filters as tools/analysis_v2.py: arm e1, gap stale, component e1_primary, carrier c0_original;
RESIDUAL / WITNESS-DROPPED excluded (a pair is dropped if either side is excluded).
Writes numbers.json. The frozen block must reproduce the paper's current tables (asserted).
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DESTROYED = {"COERCED-STATIVE", "MANUFACTURE"}
EXCLUDED = {"RESIDUAL", "WITNESS-DROPPED"}
FRAMES = ["affiliate_role", "class", "equipment", "household", "lodging", "project", "vehicle", "workplace"]
COLUMNS = [
    ("DeepSeek V4 Flash", "confirmatory", "run-v2-full-20260825-010312"),
    ("GPT-5.6 Luna", "confirmatory", "run-v2-full-20260825-010323"),
    ("GLM-5.2", "confirmatory", "run-v2-full-20260825-234313"),
    ("Claude Sonnet 5", "frontier", "run-v2-full-frontier-sonnet5-20260826-104357"),
    ("Qwen3-8B", "estimation", "run-v2-full-local-qwen3-8b-20260813-060538"),
    ("Mistral Small 3.2", "estimation", "run-v2-full-local-mistral-small-3.2-24b-20260813-064704"),
    ("GPT-OSS-20B", "extension", "run-v2-full-local-gpt-oss-20b-20260826-052612"),
    ("OLMo-2-32B", "extension", "run-v2-full-local-olmo-2-32b-instruct-20260826-060733"),
    ("Gemini 3.1 Pro", "frontier", "run-v2-full-frontier-gemini31pro-20260827-015332"),
    ("Grok 4.6", "frontier", "run-v2-full-frontier-grok46-20260827-015925"),
    ("Qwen3.8-Max", "frontier", "run-v2-full-frontier-qwen38max-20260827-021106"),
]


def binom_ge(k, n):
    return 1.0 if n == 0 else min(1.0, sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); out[i] = min(1.0, run)
    return out


def load_stim():
    return {json.loads(l)["case_id"]: json.loads(l) for l in (REPO / "data/stimuli_v2.jsonl").read_text().splitlines()}


def main_rows(path, stim):
    out = []
    for l in path.read_text().splitlines():
        r = json.loads(l)
        if r["arm"] == "e1" and r["gap"] == "stale" and r["form"] in ("prog", "simple") \
                and r["carrier"] == "c0_original" and stim[r["case_id"]]["component"] == "e1_primary":
            out.append(r)
    return out


def column(rows):
    elig = [r for r in rows if r["label"] not in EXCLUDED]
    rate = {f: [sum(r["label"] in DESTROYED for r in elig if r["form"] == f), sum(r["form"] == f for r in elig)]
            for f in ("prog", "simple")}
    raw = {}
    for r in rows:
        raw.setdefault((r["frame"], r["cluster"]), {})[r["form"]] = r["label"]
    pairs = {k: v for k, v in raw.items() if "prog" in v and "simple" in v and v["prog"] not in EXCLUDED and v["simple"] not in EXCLUDED}
    b = sum(v["prog"] in DESTROYED and v["simple"] not in DESTROYED for v in pairs.values())
    c = sum(v["simple"] in DESTROYED and v["prog"] not in DESTROYED for v in pairs.values())
    frames = {fr: {f: [sum(r["label"] in DESTROYED for r in elig if r["frame"] == fr and r["form"] == f),
                       sum(1 for r in elig if r["frame"] == fr and r["form"] == f)] for f in ("prog", "simple")} for fr in FRAMES}
    return {"rate": rate, "n": len(pairs), "b": b, "c": c, "p_one_sided": binom_ge(b, b + c), "frames": frames}


def build(trace_dir, stim):
    cols = []
    for name, tier, base in COLUMNS:
        row = column(main_rows(trace_dir / f"{base}.scored.jsonl", stim))
        row.update({"model": name, "tier": tier, "base": base})
        cols.append(row)
    adj = holm([c["p_one_sided"] for c in cols[:3]])
    for c, p in zip(cols[:3], adj):
        c["p_holm"] = p
    tot = {k: sum(c[k] for c in cols) for k in ("n", "b", "c")}
    conf = {k: sum(c[k] for c in cols[:3]) for k in ("n", "b", "c")}
    return {"columns": cols, "total_11": tot, "total_confirmatory": conf}


def main():
    stim = load_stim()
    frozen = build(REPO / "traces", stim)
    flip = build(HERE / "root" / "traces", stim)
    assert (frozen["total_confirmatory"], frozen["total_11"]) == ({"n": 381, "b": 244, "c": 0}, {"n": 1397, "b": 807, "c": 2}), frozen["total_11"]
    assert [tuple(c["rate"]["prog"]) for c in frozen["columns"][:3]] == [(104, 128), (79, 128), (69, 125)]
    (HERE / "numbers.json").write_text(json.dumps({"frozen": frozen, "aspect_flip": flip}, indent=1) + "\n")
    for lab, d in (("frozen", frozen), ("aspect_flip", flip)):
        print(lab, "confirmatory", d["total_confirmatory"], "all 11", d["total_11"])
        for c in d["columns"]:
            print(f"  {c['model']:18s} prog {c['rate']['prog'][0]}/{c['rate']['prog'][1]}  simple {c['rate']['simple'][0]}/{c['rate']['simple'][1]}  n={c['n']} b={c['b']} c={c['c']}  p={c['p_one_sided']:.1e}" + (f" holm={c['p_holm']:.1e}" if 'p_holm' in c else ""))


if __name__ == "__main__":
    main()
