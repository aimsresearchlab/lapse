#!/usr/bin/env python3
"""Predeclared scorer for the qualifier ablation. Deterministic; no API calls.

prog / simple rows: cascade v5 `score_note` from tools/score.py (paper's e1 labels).
  D (destroyed) = COERCED-STATIVE or MANUFACTURE; RESIDUAL / WITNESS-DROPPED excluded.
simple_fornow / simple_atm rows: marker-fate labels on the witness line(s):
  WITNESS-DROPPED     no line carries the witness
  MARKER-KEPT-SAME    a marker of the input's class survives (T for "for now", C for "at the moment")
  MARKER-KEPT-CROSS   only a marker of the other class survives
  PROG-CONVERTED      no lexical marker, but progressive / perfect-progressive morphology on the witness line
  DATE-ONLY           no marker, no progressive, but a date anchor on the witness line
  DROPPED             bare stative rendering; no temporal qualification of any kind
  RESIDUAL            none of the above matched (hand-read)
Primary ablation outcome: QUAL-LOST = DROPPED or DATE-ONLY.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import score as cascade  # noqa: E402

T_CLASS = re.compile(r"\b(?:for now|for the time being|temporar(?:y|ily)|for a while|short[- ]term|provisional(?:ly)?|interim|on a temporary basis|not permanent(?:ly)?|until further notice)\b", re.I)
C_CLASS = re.compile(r"\b(?:at the moment|for the moment|currently|right now|just now|at present|presently|as of now|these days|nowadays|at this time)\b", re.I)
CLASS_OF = {"simple_fornow": "T", "simple_atm": "C"}

def witness_lines(text: str, wit_token: str) -> str:
    text = cascade.norm(text)
    return " ".join(ln for ln in text.splitlines() if cascade.wit_re(wit_token).search(ln))

def marker_label(text: str, wit_token: str, form: str) -> str:
    ln = witness_lines(text, wit_token)
    if not ln:
        return "WITNESS-DROPPED"
    same = T_CLASS if CLASS_OF[form] == "T" else C_CLASS
    other = C_CLASS if CLASS_OF[form] == "T" else T_CLASS
    if same.search(ln):
        return "MARKER-KEPT-SAME"
    if other.search(ln):
        return "MARKER-KEPT-CROSS"
    if cascade.PROGRESSIVE.search(ln) or cascade.PERF_PROG.search(ln):
        return "PROG-CONVERTED"
    if cascade.STATIVE.search(ln) or cascade.PAST_EXP.search(ln):
        return "DATE-ONLY" if cascade.DATE_ANCHOR.search(ln) else "DROPPED"
    return "RESIDUAL"

def label(rec: dict) -> str:
    if rec["form"] in ("prog", "simple"):
        return cascade.score_note(rec["response"], rec["wit_token"], rec["form"])
    return marker_label(rec["response"], rec["wit_token"], rec["form"])

def placebo_marker_hits(rec: dict) -> dict:
    """Marker detector applied to rows whose input carried no lexical marker (prog/simple): false-positive audit."""
    ln = witness_lines(rec["response"], rec["wit_token"])
    return {"T": bool(T_CLASS.search(ln)), "C": bool(C_CLASS.search(ln))}

def main():
    trace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "trace.jsonl"
    out = trace.with_suffix(".scored.jsonl")
    rows = [json.loads(l) for l in trace.read_text().splitlines()]
    with out.open("w") as fh:
        for r in rows:
            r = dict(r); r["label"] = "ERROR" if r["response"].startswith("<ERROR") else label(r)
            if r["form"] in ("prog", "simple") and r["label"] != "ERROR":
                r["placebo_marker"] = placebo_marker_hits(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"scored": len(rows), "out": str(out)}))

if __name__ == "__main__":
    main()
