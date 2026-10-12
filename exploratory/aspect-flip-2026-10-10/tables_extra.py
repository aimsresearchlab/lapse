"""Secondary-table numbers under both definitions ($0): guided instruction (e1 vs l2), perfect
progressive loss, same-lexeme R1. Parsed from the frozen analysis outputs in research/ (frozen) and
out_*.txt here (aspect flip), plus a direct perfect-form count. Writes tables_extra.json."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
def parse_secondary(text):
    out = {}; cur = None
    for l in text.splitlines():
        m = re.match(r"### (\S+)", l)
        if m: cur = m.group(1); continue
        m = re.search(r"prog\s+e1 destroyed (\d+)/(\d+) \(([\d.]+)%\) -> l2 destroyed (\d+)/(\d+) \(([\d.]+)%\)", l)
        if m and cur: out.setdefault(cur, {})["prog"] = [int(m[1]), int(m[2]), float(m[3]), int(m[4]), int(m[5]), float(m[6])]
        m = re.search(r"paired \(both forms\): n=(\d+) b\(e1-only\)=(\d+) c\(l2-only\)=(\d+)", l)
        if m and cur: out[cur]["paired"] = [int(m[1]), int(m[2]), int(m[3])]
    return out
def parse_r1(text):
    out = {}; cur = None
    for l in text.splitlines():
        m = re.match(r"### (\S+)", l)
        if m: cur = m.group(1); continue
        m = re.search(r"R1 same-lexeme McNemar: pairs=(\d+) b=(\d+) c=(\d+)", l)
        if m and cur: out[cur] = [int(m[1]), int(m[2]), int(m[3])]
    return out
def perfect(trace_dir):
    D = {"COERCED-STATIVE", "MANUFACTURE"}; EX = {"RESIDUAL", "WITNESS-DROPPED"}; out = {}
    for p in sorted(trace_dir.glob("*.scored.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        pp = [r for r in rows if r["arm"] == "e1" and r["form"] == "perf_prog" and r["label"] not in EX]
        if pp: out[p.name.split(".")[0]] = [sum(r["label"] in D for r in pp), len(pp)]
    return out
res = {
 "frozen": {"guided": parse_secondary((REPO/"results/ANALYSIS_V2_SECONDARY_2026-08-27.txt").read_text()),
            "r1": parse_r1((REPO/"results/ANALYSIS_V2_2026-08-27.txt").read_text()), "perf_prog_lost": perfect(REPO/"traces")},
 "aspect_flip": {"guided": parse_secondary((HERE/"out_analysis_v2_secondary.txt").read_text()),
                 "r1": parse_r1((HERE/"out_analysis_v2.txt").read_text()), "perf_prog_lost": perfect(HERE/"root/traces")}}
(HERE/"tables_extra.json").write_text(json.dumps(res, indent=1) + "\n")
for d in res:
    print(d)
    for k in res[d]: print(" ", k, json.dumps(res[d][k]))
