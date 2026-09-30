"""Deterministic post-checks for Part D traces (no model calls).

Graphiti: stored text = entity-edge facts (+ node summaries as fallback);
also valid_at/invalid_at pattern per form.  Letta: stored text = the strings
the agent inserted into memory blocks (diff of blocks before/after) plus
archival passages.  Same PROG/DUR/ANCHOR regexes as Part B; the prog metric
is "progressive morphology inside the witness-bearing stored text" for grid
sets (witness = wit_token), and "anywhere in the stored text" for real sets.
"""
import glob
import json
import re
from collections import Counter, defaultdict

PROG = re.compile(r"\b(?:is|are|am|'s|'re|'m|has been|have been|been)\s+"
                  r"(?:currently\s+|now\s+)?\w+ing\b", re.I)
DUR = re.compile(r"\b(?:for (?:about |approximately |almost |the (?:past|last) )?"
                 r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
                 r"several|few|couple)\b[\w\s-]{0,12}"
                 r"(?:days?|weeks?|months?|years?)|since\s+\S+|until\b|"
                 r"for the past|for the last|recently|ago)\b", re.I)
ANCHOR = re.compile(r"\b(?:20\d\d|january|february|march|april|may|june|july|"
                    r"august|september|october|november|december|as of|since)\b",
                    re.I)


def stored_texts(r):
    if r["system"].startswith("graphiti"):
        t = [e["fact"] for e in r["edges"]]
        if not t:
            t = [n["summary"] for n in r["nodes"] if n.get("summary")]
        return t
    before = {(b["label"], b["value"]) for b in r["blocks_before"]}
    t = [b["value"] for b in r["blocks_after"] if (b["label"], b["value"]) not in before]
    return t + list(r["passages"])


def flags(r):
    texts = stored_texts(r)
    wt = (r.get("wit_token") or "").lower()
    if wt:
        wtexts = [t for t in texts if wt in t.lower()]
        target = " || ".join(wtexts)
    else:
        target = " || ".join(texts)
    return dict(n=len(texts), witness=bool(wt and target) if wt else None,
                prog=bool(PROG.search(target)), dur=bool(DUR.search(target)),
                anchor=bool(ANCHOR.search(target)), target=target)


def main():
    for f in sorted(glob.glob("trace_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        sysname, writer = rows[0]["system"], rows[0]["writer"]
        print(f"\n===== {sysname} / {writer}  ({len(rows)} items, {sum(1 for r in rows if r['error'])} errors)")
        by = defaultdict(list)
        for r in rows:
            by[r["set"]].append((r, flags(r)))
        print(f"{'set':13s} n  stored  witness  prog  dur  anchor")
        for s in ("real_lme", "real_wc", "grid_prog", "grid_simple", "grid_perfdur"):
            xs = by.get(s, [])
            if not xs:
                continue
            n = len(xs)
            st = sum(1 for _, fl in xs if fl["n"] > 0)
            wit = sum(1 for _, fl in xs if fl["witness"]) if s.startswith("grid") else "-"
            print(f"{s:13s} {n:2d}  {st:4d}    {str(wit):5s}   {sum(fl['prog'] for _, fl in xs):3d}   "
                  f"{sum(fl['dur'] for _, fl in xs):3d}   {sum(fl['anchor'] for _, fl in xs):3d}")
        for s in ("grid_prog", "grid_perfdur"):
            per = defaultdict(Counter)
            for r, fl in by.get(s, []):
                per[r["frame"]]["n"] += 1
                per[r["frame"]]["prog"] += fl["prog"]
            if per:
                print(f"  {s} per frame (prog kept/n): " + ", ".join(f"{fr} {c['prog']}/{c['n']}" for fr, c in sorted(per.items())))
        if sysname.startswith("graphiti"):
            print("  temporal fields on witness-bearing edges (valid_at==reference_time / invalid_at set / no witness edge):")
            for s in ("grid_prog", "grid_simple", "grid_perfdur"):
                same = inv = none = 0
                for r, fl in by.get(s, []):
                    wt = r["wit_token"].lower()
                    es = [e for e in r["edges"] if wt in e["fact"].lower()]
                    if not es:
                        none += 1
                        continue
                    ref = r["reference_time"][:19]
                    same += any((e["valid_at"] or "")[:19] == ref for e in es)
                    inv += any(e["invalid_at"] for e in es)
                print(f"    {s:13s} {same}/{inv}/{none}  of {len(by.get(s, []))}")
            print("  distinct valid_at offsets from reference_time (perfdur, witness edges):",
                  Counter((e["valid_at"] or "none")[:10] == r["reference_time"][:10]
                          for r, _ in by.get("grid_perfdur", []) for e in r["edges"] if r["wit_token"].lower() in e["fact"].lower()))
        with open(f.replace("trace_", "readme_real_"), "w") as o:
            for s in ("real_lme", "real_wc"):
                for r, fl in by.get(s, []):
                    o.write(f"[{r['id']}] prog={int(fl['prog'])} dur={int(fl['dur'])} anchor={int(fl['anchor'])}\n  IN : {r['text'][:300]}\n")
                    for t in stored_texts(r):
                        o.write(f"  OUT: {t}\n")
                    if r["system"].startswith("graphiti"):
                        for e in r["edges"]:
                            o.write(f"       [{e['name']}] valid_at={e['valid_at']} invalid_at={e['invalid_at']}\n")
                    o.write("\n")


if __name__ == "__main__":
    main()
