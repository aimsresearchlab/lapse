"""Offline tests for the powered verification-tool run (no API calls). Run: python3 -m pytest -q ."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_powered as A  # noqa: E402
import build_powered as B  # noqa: E402


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


T = load(HERE / "generated/targets.jsonl")
R2 = {r["id"]: r for r in load(B.R2DIR / "generated/inputs.jsonl")}


def test_counts():
    for age in B.AGES:
        sub = [r for r in T if r["age_tag"] == age]
        assert len(sub) == 2 * 291 and len({r["content_key"] for r in sub}) == 120


def test_pairs_differ_only_in_declared_edit():
    by = {}
    for r in T:
        by.setdefault(r["pair_id"], {})[r["form"]] = r
    for pid, d in by.items():
        p, s = d["progressive"], d["simple"]
        e = p["declared_edit"]
        assert p["memory_note"].count(e["from"]) == 1
        assert s["memory_note"] == p["memory_note"].replace(e["from"], e["to"])
        assert s["system"] == p["system"].replace(p["memory_note"], s["memory_note"])
        assert p["tools"] == s["tools"] and p["user_request"] == s["user_request"]


def test_r2_v0_items_excluded_and_r2_wording_identical():
    assert not [r for r in T if r["in_r2"] and r["variant"] == 0]
    for r in T:
        if r["in_r2"] and r["age_tag"] == "M8":
            src = R2[f"DT-{r['frame']}-{r['content_key'].split(':')[1]}-{r['form'].upper()}"]
            assert r["system"] == src["system"] and r["memory_note"] == src["memory_note"]
            assert r["user_request"] != src["user_request"]
    for frame, reqs in B.REQUESTS.items():
        assert B.R2B.wrapper(frame, "X")[0] == reqs[0]


def test_role_witnesses_disjoint():
    w = lambda f: {r["witness"].casefold() for r in load(HERE / f"generated/{f}.jsonl")}
    assert not (w("targets") & w("sweep")) and not (w("targets") & w("controls")) and not (w("sweep") & w("controls"))


def test_stats_helpers():
    assert A.upper_tail(5, 5) == 1 / 32 and A.upper_tail(0, 0) == 1.0
    assert abs(A.two_sided(17, 5) - 0.0169) < 1e-3
    h = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": 0.03, "c": 0.06, "b": 0.06}
    pairs = {f"p{i}": {"simple": "EXECUTE", "progressive": "VERIFY" if i < 6 else "EXECUTE", "_meta": {"content_key": f"k{i // 2}"}} for i in range(10)}
    pairs["p9"]["progressive"] = "UNPARSED"
    st = A.cell_stats(pairs, random.Random(1))
    assert st["flat_only"] == 6 and st["prog_only"] == 0 and st["complete_pairs"] == 9
    assert st["contents_net_positive"] == 3 and st["p_cluster_one_sided"] == 1 / 8
    assert st["rd_missingness_bounds"] == [0.6, 0.7]
