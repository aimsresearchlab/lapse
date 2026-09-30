#!/usr/bin/env python3
"""Build the qualifier-ablation stimuli from the frozen v2 reference build. No API calls.

Four forms per (frame, cluster) of the confirmatory cell (e1_primary, c0_original, stale):
  prog           frozen progressive key sentence, verbatim
  simple         frozen simple-present key sentence, verbatim
  simple_fornow  simple key sentence + " for now"        (lexical temporariness marker, class T)
  simple_atm     simple key sentence + " at the moment"  (lexical present-time marker, class C)
Dates are rebuilt for a contemporaneous run: eval date TODAY, session date TODAY-245 days (the
generator's stale offset). Everything else in the history is byte-identical to the frozen row.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STIM = ROOT / "data" / "stimuli_v2.jsonl"
TODAY = dt.date(2026, 9, 12)
SESSION = TODAY - dt.timedelta(days=245)
SYS = ("You are a personal assistant with long-term memory of your conversations with the user. "
       f"Today's date: {TODAY.isoformat()}. Relevant excerpts from your conversation history are shown below."
       "\n\n--- Conversation history ---\n{history}\n--- End of history ---")
MARKERS = {"simple_fornow": " for now", "simple_atm": " at the moment"}
SPLIT = ". also, any podcast recommendations for my commute?"

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def rows():
    for ln in STIM.read_text().splitlines():
        s = json.loads(ln)
        if s["arm"] == "e1" and s["component"] == "e1_primary" and s["carrier"] == "c0_original" and s["gap"] == "stale":
            yield s

def redate(history: str) -> str:
    assert history.count("[Session dated 2025-12-09]") == 1, history
    return history.replace("[Session dated 2025-12-09]", f"[Session dated {SESSION.isoformat()}]")

def with_marker(history: str, marker: str) -> str:
    assert history.count(SPLIT) == 1, history
    return history.replace(SPLIT, marker + SPLIT)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    base = {}
    for s in rows():
        base.setdefault((s["frame"], s["cluster"]), {})[s["form"]] = s
    assert len(base) == 128 and all(set(v) == {"prog", "simple"} for v in base.values())
    out = []
    for (frame, cluster), pair in sorted(base.items()):
        for form in ("prog", "simple", "simple_fornow", "simple_atm"):
            src = pair["prog" if form == "prog" else "simple"]
            hist = redate(src["history"])
            if form in MARKERS:
                hist = with_marker(hist, MARKERS[form])
            out.append({"case_id": f"qa-{frame}-{cluster:02d}-{form}", "frame": frame, "cluster": cluster, "form": form,
                        "source_case_id": src["case_id"], "source_form": src["form"], "gap": "stale",
                        "gap_date": SESSION.isoformat(), "eval_date": TODAY.isoformat(), "witness": src["witness"],
                        "wit_token": src["wit_token"], "fact_noun": src["fact_noun"], "arm": "e1", "carrier": "c0_original",
                        "history": hist, "query": src["query"], "marker": MARKERS.get(form)})
    # pair-edit assertions: within a cluster, simple_* differ from simple only by the marker; prog differs only in the key verb phrase
    for (frame, cluster) in base:
        d = {r["form"]: r for r in out if r["frame"] == frame and r["cluster"] == cluster}
        for f, m in MARKERS.items():
            assert d[f]["history"] == d["simple"]["history"].replace(SPLIT, m + SPLIT)
        assert d["prog"]["query"] == d["simple"]["query"]
    p = a.out / "stimuli.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out))
    man = {"experiment": "qualifier-ablation", "built": TODAY.isoformat(), "session_date": SESSION.isoformat(),
           "source_stimuli": str(STIM), "source_sha256": sha(STIM), "n_stimuli": len(out), "forms": ["prog", "simple", "simple_fornow", "simple_atm"],
           "markers": MARKERS, "system_template": SYS, "max_tokens": 350, "temperature": 0,
           "models": ["deepseek/deepseek-v4-flash", "openai/gpt-5.6-luna", "z-ai/glm-5.2"],
           "provider_rule": {"quantizations": ["fp8", "bf16", "fp16", "fp32", "unknown"], "allow_fallbacks": False},
           "stimuli_sha256": sha(p)}
    (a.out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: man[k] for k in ("n_stimuli", "stimuli_sha256", "source_sha256", "session_date")}))

if __name__ == "__main__":
    main()
