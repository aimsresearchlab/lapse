"""Regression audits for data/stimuli_a2.jsonl (PILOT_SPEC_v1.md AMENDMENT A2).

These re-check, from the frozen artifact on disk, the three wrapper audits
the builder enforces at build time — so a hand-edit of the jsonl (or a
builder regression) cannot silently violate them.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_stimuli import FRAMES  # noqa: E402  (frozen v1 definitions)
from build_stimuli_a2 import BAN_RE, CARRIERS, N_CLUSTERS_A2  # noqa: E402

ROWS = [json.loads(l) for l in (ROOT / "data" / "stimuli_a2.jsonl").open()]


def key_line(row):
    return FRAMES[row["frame"]]["forms"][row["form"]].format(w=row["witness"])


def test_counts():
    assert len(ROWS) == 1400
    per = {}
    for r in ROWS:
        per[(r["arm"], r["gap"])] = per.get((r["arm"], r["gap"]), 0) + 1
    assert per == {("behavioral", "stale"): 600, ("e1", "stale"): 600,
                   ("behavioral", "fresh"): 200}
    assert {r["carrier"] for r in ROWS} == set(CARRIERS)
    assert {r["cluster"] for r in ROWS} == set(range(N_CLUSTERS_A2))


def test_key_line_present_and_v1_identical():
    for r in ROWS:
        assert key_line(r) in r["history"], r["case_id"]


def test_wrapper_temporal_lexicon_ban():
    for r in ROWS:
        wrapper = r["history"].replace(key_line(r), "")
        m = BAN_RE.search(wrapper)
        assert not m, (r["case_id"], m and m.group(0))


def test_witness_exactly_once():
    for r in ROWS:
        assert (r["history"] + r["query"]).count(r["wit_token"]) == 1, r["case_id"]


def test_within_cluster_carrier_byte_identity():
    groups = {}
    for r in ROWS:
        if r["arm"] != "behavioral" or r["gap"] != "stale":
            continue
        residue = r["history"].replace(key_line(r), "<KEY>") + r["query"]
        groups.setdefault((r["frame"], r["cluster"], r["carrier"]),
                          set()).add(residue)
    assert all(len(v) == 1 for v in groups.values())


def test_dates():
    for r in ROWS:
        want = "2026-08-09" if r["gap"] == "fresh" else "2025-12-09"
        assert f"[Session dated {want}]" in r["history"], r["case_id"]


def test_bound_only_cell_with_adverbial():
    # "until December" must appear in BOUND cells only (adverbial-leak trap).
    for r in ROWS:
        if r["form"] == "bound":
            assert "until December" in r["history"], r["case_id"]
        else:
            assert "until December" not in r["history"], r["case_id"]
