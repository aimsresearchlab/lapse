#!/usr/bin/env python3
"""Derive note-age variants of the frozen R2 verification-tool inputs. No API calls.

For each age, every R2 target row (414) and every R2 expired_bounded control row (90) is copied with ONLY
the note's written date changed (in `written`, `system`, and messages[0].content). Everything else,
including the note text, request, tools, tool order, reader, and call order, is byte-identical; asserted.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "downstream-tool-2026-09-11" / "frozen-r2"
AGES = {"M1": "2026-08-08", "M3": "2026-06-08"}   # today stays 2026-09-08 (R2)
OLD = "2026-01-06"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines()]
def js(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def redate(row, tag, new):
    r = json.loads(js(row))
    assert r["written"] == OLD and r["system"].count(f"written on {OLD}") == 1 and r["system"].count(f"(written {OLD})") == 1
    r["written"] = new
    r["system"] = r["system"].replace(f"written on {OLD}", f"written on {new}").replace(f"(written {OLD})", f"(written {new})")
    assert r["messages"][0]["role"] == "system" and r["messages"][0]["content"] == row["system"]
    r["messages"][0]["content"] = r["system"]
    r["id"] = f"{row['id']}-{tag}"; r["age_tag"] = tag; r["source_r2_id"] = row["id"]
    if "pair_id" in r: r["pair_id"] = f"{row['pair_id']}-{tag}"
    # everything else identical
    a, b = dict(row), dict(r)
    for k in ("id", "written", "system", "messages", "pair_id", "age_tag", "source_r2_id"): a.pop(k, None); b.pop(k, None)
    assert a == b, row["id"]
    return r

def main():
    targets = load(R2 / "target.jsonl"); controls = [c for c in load(R2 / "control.jsonl") if c["control"] == "expired_bounded"]
    assert len(targets) == 414 and len(controls) == 90
    out_t, out_c = [], []
    for tag, new in AGES.items():
        out_t += [redate(r, tag, new) for r in targets]
        out_c += [redate(r, tag, new) for r in controls]
    (HERE / "age_targets.jsonl").write_text("".join(js(r) + "\n" for r in out_t))
    (HERE / "age_controls.jsonl").write_text("".join(js(r) + "\n" for r in out_c))
    man = {"experiment": "downstream-tool-note-age", "ages": AGES, "today": "2026-09-08", "r2_source": {"target": str(R2 / "target.jsonl"), "target_sha256": sha(R2 / "target.jsonl"),
           "control": str(R2 / "control.jsonl"), "control_sha256": sha(R2 / "control.jsonl"), "freeze_manifest_sha256": sha(R2 / "freeze_manifest.json"),
           "calibration_gate_sha256": sha(R2.parent / "calibration_gate_r2.json"), "calibration_trace_sha256": sha(R2.parent / "trace_control_r2.jsonl")},
           "counts": {"age_targets": len(out_t), "age_controls": len(out_c)}, "hashes": {"age_targets": sha(HERE / "age_targets.jsonl"), "age_controls": sha(HERE / "age_controls.jsonl")},
           "instrument": "unchanged from R2 (prompt policy, tools, enum field, readers, temperature 0, max 1024 tokens); only the note's written date varies",
           "hard_cost_cap_usd": 1.0}
    (HERE / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n"); print(json.dumps(man["counts"]), json.dumps(man["hashes"]))

if __name__ == "__main__": main()
