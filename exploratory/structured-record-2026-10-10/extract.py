#!/usr/bin/env python3
"""Extract the witness line (F) and witness record (S) from the writer trace; writer-side measures.

    python extract.py      # notes.jsonl, writer_summary.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from score import norm, score_note, wit_re  # noqa: E402

FIELD = re.compile(r"^[\s>*_\-•\d.]*(said|fact|form|observed)[\s*_]*:[\s*_]*(.*?)[\s*_]*$", re.I)


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def clean(line): return re.sub(r"\*\*|__", "", re.sub(r"^[\s>*\-•]+", "", line)).strip()
# Descriptive only: any -ing form of a frame verb, with or without an auxiliary ("I'm working",
# "Working on Project X"); score.py's PROGRESSIVE needs is/are/am and misses first person.
ING = re.compile(r"\b(?:staying|living|working|taking|driving|using|lecturing)\b", re.I)
def has_prog(s): return bool(ING.search(norm(s)))


def records(text):
    out, cur = [], {}
    for line in text.splitlines():
        m = FIELD.match(line)
        if not m:
            continue
        k, v = m.group(1).lower(), m.group(2).strip()
        if k == "said" and cur:
            out.append(cur); cur = {}
        cur[k] = v
    if cur:
        out.append(cur)
    return out


def main():
    inputs = {r["id"]: r for r in load(HERE / "writer_inputs.jsonl")}
    trace = {r["item_id"]: r for r in load(HERE / "writer_trace.jsonl") if not r["error"]}
    by = {}
    for iid, row in inputs.items():
        by.setdefault((row["content_key"], row["form"]), {})[row["instruction"]] = (row, trace[iid]["response"])
    notes, flags = [], Counter()
    for (key, form), d in sorted(by.items()):
        row, f_text = d["F"]
        _, s_text = d["S"]
        tok, w = row["wit_token"], row["witness"]
        f_lines = [clean(l) for l in f_text.splitlines() if wit_re(tok).search(norm(l))]
        recs = [r for r in records(s_text) if any(wit_re(tok).search(norm(v)) for v in r.values())]
        rec = recs[0] if recs else None
        flags["F_witness_missing"] += not f_lines
        flags["F_multi_line"] += len(f_lines) > 1
        flags["S_record_missing"] += rec is None
        flags["S_multi_record"] += len(recs) > 1
        n = {"content_key": key, "frame": row["frame"], "form": form, "witness": w, "wit_token": tok,
             "user_sentence": row["user_sentence"], "F_raw": f_text, "S_raw": s_text,
             "F_note": " ".join(f_lines) if f_lines else None, "S_record": rec}
        if rec:
            flags["S_missing_field"] += any(k not in rec for k in ("said", "fact", "form", "observed"))
            n["S_full"] = "\n".join(f"{k}: {rec[k]}" for k in ("said", "fact", "form", "observed") if k in rec)
            n["S_noquote"] = "\n".join(f"{k}: {rec[k]}" for k in ("fact", "form", "observed") if k in rec)
            # verbatim: the user's sentence appears unchanged inside the quote (some quote the whole turn)
            n["W_said_verbatim"] = norm(row["user_sentence"]).lower().rstrip(".") in norm(rec.get("said", "")).lower()
            n["W_said_prog"] = has_prog(rec.get("said", ""))
            n["W_fact_prog"] = has_prog(rec.get("fact", ""))
            fm = rec.get("form", "").lower()
            n["W_form_label"] = fm
            n["W_form_correct"] = ("progressive" in fm) if form == "progressive" else ("simple present" in fm and "progressive" not in fm)
        if n["F_note"]:
            n["W_F_label"] = score_note(n["F_note"], tok, "prog" if form == "progressive" else "simple")
            n["W_F_prog"] = has_prog(n["F_note"])
        notes.append(n)
    (HERE / "notes.jsonl").write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes))
    summ = {"flags": dict(flags)}
    for form in ("progressive", "simple"):
        g = [n for n in notes if n["form"] == form]
        s = [n for n in g if n.get("S_record")]
        summ[form] = {"n": len(g),
                      "F_keeps_progressive": sum(bool(n.get("W_F_prog")) for n in g),
                      "F_labels": dict(Counter(n.get("W_F_label") for n in g)),
                      "S_said_verbatim": sum(n["W_said_verbatim"] for n in s),
                      "S_said_progressive": sum(n["W_said_prog"] for n in s),
                      "S_fact_progressive": sum(n["W_fact_prog"] for n in s),
                      "S_form_correct": sum(n["W_form_correct"] for n in s),
                      "S_form_labels": dict(Counter(n["W_form_label"] for n in s))}
    (HERE / "writer_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
