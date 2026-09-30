"""Gold-subset sampler (SCORING_V2.md item 2, todo.md "Gold-subset sampling").

Draws the 250-item human annotation sample for the behavioral-leg gold
standard, stratified by condition x cascade label with forced coverage of
every label and all former-residual shapes, then BLINDS it for annotation.

Sources (the only scored traces — v2 full traces stay unscored per spec §5):
  - v1 deepseek-v4-flash full run  (scored + judged, 1,200 behavioral items)
  - v2 smoke                        (scored, 154 behavioral items — covers the
                                     v2-only shapes: carriers c1-c5, perf,
                                     gap_gradient, lexeme cells)

Sampling design (deterministic, seed 20260813):
  A. ALL v1 items where cascade label != PROCEED           (~99: every PWC,
     HEDGE, TEMPLATE, CONFAB-COMMIT — forced label coverage incl. the
     former-residual CONFAB shape)
  B. v1 cascade-judge 4-label disagreements not already in A, stratified by
     (judge label, form), capped at CAP_B
  C. ALL v2-smoke non-PROCEED + a stratified PROCEED sample forcing every
     carrier and every v2-only component, capped at CAP_C
  D. fill to N_TOTAL with v1 PROCEED/agreeing items stratified evenly over
     form x gap x arm

Outputs:
  web/data/gold_items.js    annotator-facing, BLINDED: gid/query/response/
                            witness only, shuffled. This is all the UI loads.
  data/gold_key_v1.json     gid -> case_id/source/model/condition/cascade/
                            judge. THE UNBLINDING KEY — do not open until
                            both annotators have exported.
  data/gold_manifest_v1.json  sampling design record (strata counts, seed,
                            source file hashes) for the paper appendix.

Usage: python3 tools/sample_gold.py
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260813
N_TOTAL = 250
CAP_B = 40
CAP_C = 35
ARMS = ("behavioral", "e2", "l2e2")

V1_SCORED = ROOT / "traces/run-full-20260812-002604.scored.jsonl"
V1_JUDGED = ROOT / "traces/run-full-20260812-002604.scored.judged.jsonl"
V2_SMOKE_SCORED = ROOT / "traces/run-v2-smoke-20260812-202448.scored.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def build_query_index(stim_path: Path) -> dict[str, str]:
    """case_id -> query, including derived e2/l2e2 (downstream_query of the
    e1/l2 row, case_id arm-swapped — same join as tools/judge.py)."""
    idx: dict[str, str] = {}
    for c in load_jsonl(stim_path):
        idx[c["case_id"]] = c["query"]
        if c["arm"] in ("e1", "l2") and "downstream_query" in c:
            arm2 = "e2" if c["arm"] == "e1" else "l2e2"
            idx[c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1)] = c["downstream_query"]
    return idx


def stratified_take(items: list[dict], key_fn, n: int, rng: random.Random) -> list[dict]:
    """Round-robin across strata so every stratum is covered before any repeats."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        strata[key_fn(it)].append(it)
    for bucket in strata.values():
        rng.shuffle(bucket)
    out: list[dict] = []
    keys = sorted(strata, key=str)
    while len(out) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


def main() -> None:
    rng = random.Random(SEED)
    q_v1 = build_query_index(ROOT / "data/stimuli_v1.jsonl")
    q_v2 = build_query_index(ROOT / "data/stimuli_v2.jsonl")

    v1 = [r for r in load_jsonl(V1_SCORED) if r["arm"] in ARMS
          and not r["response"].startswith("<ERROR")]
    judge = {p["case_id"]: p["judge"] for p in load_jsonl(V1_JUDGED)
             if not p["judge"].startswith(("UNPARSED", "UNREACHABLE"))}
    v2 = [r for r in load_jsonl(V2_SMOKE_SCORED) if r["arm"] in ARMS
          and not r["response"].startswith("<ERROR")]

    picked: dict[str, dict] = {}  # (source, case_id) uniqueness via composite key

    def add(rec: dict, source: str, tier: str, queries: dict[str, str]) -> None:
        k = f"{source}:{rec['case_id']}"
        if k in picked:
            return
        picked[k] = {
            "source": source, "tier": tier, "case_id": rec["case_id"],
            "model": rec["model"], "arm": rec["arm"], "form": rec["form"],
            "gap": rec.get("gap"), "carrier": rec.get("carrier"),
            "component": rec.get("component"), "cascade": rec["label"],
            "judge": judge.get(rec["case_id"]) if source == "v1" else None,
            "query": queries[rec["case_id"]], "response": rec["response"],
            "witness": rec["witness"],
        }

    # A: every v1 non-PROCEED
    for r in v1:
        if r["label"] != "PROCEED":
            add(r, "v1", "A_forced_label", q_v1)
    n_a = len(picked)

    # B: v1 cascade-judge disagreements (not already in), stratified
    dis = [r for r in v1 if r["case_id"] in judge
           and judge[r["case_id"]] != r["label"]
           and f"v1:{r['case_id']}" not in picked]
    for r in stratified_take(dis, lambda r: (judge[r["case_id"]], r["form"]), CAP_B, rng):
        add(r, "v1", "B_disagreement", q_v1)
    n_b = len(picked) - n_a

    # C: v2 smoke — all non-PROCEED, then force carrier + component coverage
    for r in v2:
        if r["label"] != "PROCEED":
            add(r, "v2smoke", "C_v2_shapes", q_v2)
    room = CAP_C - sum(1 for p in picked.values() if p["source"] == "v2smoke")
    v2_rest = [r for r in v2 if f"v2smoke:{r['case_id']}" not in picked]
    for r in stratified_take(v2_rest, lambda r: (r.get("carrier"), r.get("component")), room, rng):
        add(r, "v2smoke", "C_v2_shapes", q_v2)
    n_c = len(picked) - n_a - n_b

    # D: fill with v1 PROCEED/agreeing, even over form x gap x arm
    fill = [r for r in v1 if r["label"] == "PROCEED"
            and judge.get(r["case_id"]) == "PROCEED"
            and f"v1:{r['case_id']}" not in picked]
    for r in stratified_take(fill, lambda r: (r["form"], r["gap"], r["arm"]),
                             N_TOTAL - len(picked), rng):
        add(r, "v1", "D_fill", q_v1)

    items = list(picked.values())
    assert len(items) == N_TOTAL, f"got {len(items)}, want {N_TOTAL}"
    rng.shuffle(items)  # presentation order must not track source or tier

    blinded, key = [], []
    for i, it in enumerate(items):
        gid = f"g{i + 1:03d}"
        blinded.append({"gid": gid, "query": it["query"],
                        "response": it["response"], "witness": it["witness"]})
        key.append({"gid": gid} | {k: it[k] for k in (
            "source", "tier", "case_id", "model", "arm", "form", "gap",
            "carrier", "component", "cascade", "judge")})

    web_data = ROOT / "web/data"
    web_data.mkdir(parents=True, exist_ok=True)
    (web_data / "gold_items.js").write_text(
        "// AUTO-GENERATED by tools/sample_gold.py — blinded annotation items.\n"
        "// Do not edit; do not add condition fields here.\n"
        "window.GOLD_ITEMS = "
        + json.dumps(blinded, ensure_ascii=False, indent=1) + ";\n")
    (ROOT / "data/gold_key_v1.json").write_text(json.dumps(key, indent=1) + "\n")

    manifest = {
        "seed": SEED, "n_total": N_TOTAL,
        "tiers": {"A_forced_label": n_a, "B_disagreement": n_b,
                  "C_v2_shapes": n_c, "D_fill": N_TOTAL - n_a - n_b - n_c},
        "sources": {
            "v1_scored": {"path": str(V1_SCORED.relative_to(ROOT)), "sha16": sha16(V1_SCORED)},
            "v1_judged": {"path": str(V1_JUDGED.relative_to(ROOT)), "sha16": sha16(V1_JUDGED)},
            "v2_smoke_scored": {"path": str(V2_SMOKE_SCORED.relative_to(ROOT)),
                                "sha16": sha16(V2_SMOKE_SCORED)},
        },
        "cascade_label_dist": dict(Counter(k["cascade"] for k in key)),
        "form_dist": dict(Counter(k["form"] for k in key)),
        "arm_dist": dict(Counter(k["arm"] for k in key)),
        "source_dist": dict(Counter(k["source"] for k in key)),
        "note": "Oversamples non-PROCEED labels and detector disagreements by "
                "design (AbstentionBench-style stratification on predicted "
                "label); report per-stratum agreement alongside overall kappa.",
    }
    (ROOT / "data/gold_manifest_v1.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"wrote {N_TOTAL} items")
    print("tiers:", manifest["tiers"])
    print("labels:", manifest["cascade_label_dist"])
    print("forms:", manifest["form_dist"])
    print("sources:", manifest["source_dist"])
    print("\nBLINDING: web/data/gold_items.js is annotator-safe; "
          "data/gold_key_v1.json is the key — do not open before export.")


if __name__ == "__main__":
    main()
