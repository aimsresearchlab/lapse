"""Part G analysis (Graphiti + Letta): predictions G1-G4, L1-L2 per reader.
Spec: research/UNRECOVERABILITY_SPEC_2026-09-08.md, section "Part G"."""
from __future__ import annotations
import json, re
from pathlib import Path
from scipy.stats import binomtest, wilcoxon
from read_pipeline import PROG_RE, label_writer
from read_pipeline_g import g_witness_edge, l_witness_line

HERE = Path(__file__).resolve().parent
STAMP_RE = re.compile(r"\b(20\d\d|january|february|march|april|may|june|july|august|september|october|november|december|as of)\b", re.I)
LABEL_RE = re.compile(r"^[A-Z][A-Za-z /-]{1,40}:\s")   # "Currently staying at: X" / "Lives at: X"


def mcn(b, c): return binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")


def writer_side():
    G = [json.loads(l) for l in (HERE / "writer_trace_graphiti_deepseek_official.jsonl").read_text().splitlines()]
    L = [json.loads(l) for l in (HERE / "writer_trace_letta_openrouter-deepseek-deepseek-v4-flash.jsonl").read_text().splitlines()]
    print("=== WRITER SIDE ===")
    print("Graphiti: rows", len(G), "errors", sum(1 for r in G if r["error"]))
    for form in ("PROG", "SIMPLE", "BOUND"):
        rows = [r for r in G if r["form"] == form]
        edges = [(r, g_witness_edge(r)) for r in rows]
        have = [(r, e) for r, e in edges if e]
        labs = [label_writer(e["fact"]) for _, e in have]
        stamped = [bool(STAMP_RE.search(e["fact"])) for _, e in have]
        flat_stamped = sum(1 for (_, e), lab in zip(have, labs) if lab == "FLATTENED" and STAMP_RE.search(e["fact"]))
        va_write = sum(1 for _, e in have if (e["valid_at"] or "")[:10] == "2026-01-06")
        ia_set = sum(1 for _, e in have if e["invalid_at"])
        until = sum(1 for _, e in have if re.search(r"\buntil\b", e["fact"], re.I))
        print(f"  {form:6} n={len(rows)} witness-edge={len(have)} PRESERVED={labs.count('PRESERVED')} FLATTENED={labs.count('FLATTENED')} "
              f"date-in-text={sum(stamped)} (flattened&stamped={flat_stamped}) valid_at=write-time {va_write}/{len(have)} invalid_at set {ia_set} 'until' in text {until}")
    print("  G1: flattened PROG facts with a date anchor in text (prediction <= 3/n): see above")
    print("Letta: rows", len(L), "errors", sum(1 for r in L if r["error"]))
    for form in ("PROG", "SIMPLE", "BOUND"):
        rows = [r for r in L if r["form"] == form]
        lines = [(r, l_witness_line(r)) for r in rows]
        stored = [(r, s) for r, s in lines if s]
        label = sum(1 for _, s in stored if LABEL_RE.match(s))
        verb = sum(1 for _, s in stored if PROG_RE.search(s))
        marker = sum(1 for _, s in stored if label_writer(s) == "PRESERVED")
        stamped = sum(1 for _, s in stored if STAMP_RE.search(s))
        bare = sum(1 for _, s in stored if not PROG_RE.search(s) and not STAMP_RE.search(s))
        print(f"  {form:6} n={len(rows)} stored={len(stored)} not-stored={len(rows)-len(stored)} label-line={label} progressive-verb={verb} "
              f"PRESERVED(verb|marker)={marker} date-in-line={stamped} bare(no verb,no date)={bare}")
    print("  L1: PROG label line w/o verb & date >= 50%; not stored >= 20%: see above")


def report(p):
    R = {json.loads(l)["id"]: json.loads(l) for l in p.read_text().splitlines()}
    reader = next(iter(R.values()))["reader"]
    errs = sum(1 for r in R.values() if r["error"]); unp = sum(1 for r in R.values() if r["parse"] in ("UNPARSED", "MIXED"))
    print(f"\n=== {reader}: {len(R)} rows, {errs} errors, {unp} UNPARSED/MIXED")

    def cell(arm, task, sub=None):
        return [r for r in R.values() if r["arm"] == arm and r["task"] == task and (sub is None or r["writer_label"] == sub)]
    print(f"{'arm':16}{'A1 mean':>8}{'commit':>8}{'abstain':>8}{'n':>5}")
    for arm in ("G-TEXT-SIMPLE", "G-TEXT-PROG", "G-TEXT-BOUND", "G-FIELDS-SIMPLE", "G-FIELDS-PROG", "G-FIELDS-BOUND", "L-TEXT-SIMPLE", "L-TEXT-PROG", "L-TEXT-BOUND"):
        a1 = [r["p"] for r in cell(arm, "A1") if r["p"] is not None]; a2 = [r["parse"] for r in cell(arm, "A2")]
        print(f"{arm:16}{(sum(a1)/len(a1) if a1 else float('nan')):8.1f}{a2.count('COMMIT'):8d}{a2.count('ABSTAIN'):8d}{len(a2):5d}")

    def contrast(armA, armB, sub, name):
        b = c = 0; d = []
        for r in cell(armA, "A2", sub):
            o = R.get(r["id"].replace(armA, armB))
            if not o: continue
            if r["parse"] == "ABSTAIN" and o["parse"] == "COMMIT": b += 1
            elif r["parse"] == "COMMIT" and o["parse"] == "ABSTAIN": c += 1
        for r in cell(armA, "A1", sub):
            o = R.get(r["id"].replace(armA, armB))
            if o and r["p"] is not None and o["p"] is not None: d.append(r["p"] - o["p"])
        n = len(cell(armA, "A2", sub))
        ca = sum(r["parse"] == "COMMIT" for r in cell(armA, "A2", sub))
        cb = sum(R[r["id"].replace(armA, armB)]["parse"] == "COMMIT" for r in cell(armA, "A2", sub) if r["id"].replace(armA, armB) in R)
        nz = [x for x in d if x]
        pw = wilcoxon(nz).pvalue if len(nz) >= 5 else float("nan")
        print(f"  {name}: n={n} commit {armA} {ca} vs paired {armB} {cb} | A-abstain/B-commit={b}, reverse={c}, McNemar p={mcn(b,c):.3g} | A1 diff {sum(d)/len(d) if d else float('nan'):+.1f} p={pw:.2g}")
    print("G2 G-TEXT FLATTENED subset (prediction: null):"); contrast("G-TEXT-PROG", "G-TEXT-SIMPLE", "FLATTENED", "G-TEXT-PROG vs G-TEXT-SIMPLE")
    print("G3 G-TEXT PRESERVED subset (prediction: PROG commit < SIMPLE):"); contrast("G-TEXT-PROG", "G-TEXT-SIMPLE", "PRESERVED", "G-TEXT-PROG vs G-TEXT-SIMPLE")
    print("G4 G-FIELDS FLATTENED subset (prediction: still null):"); contrast("G-FIELDS-PROG", "G-FIELDS-SIMPLE", "FLATTENED", "G-FIELDS-PROG vs G-FIELDS-SIMPLE")
    print("   G-FIELDS PRESERVED subset:"); contrast("G-FIELDS-PROG", "G-FIELDS-SIMPLE", "PRESERVED", "G-FIELDS-PROG vs G-FIELDS-SIMPLE")
    print("   G-TEXT all PROG vs SIMPLE:"); contrast("G-TEXT-PROG", "G-TEXT-SIMPLE", None, "all")
    for arm in ("G-TEXT-BOUND", "G-FIELDS-BOUND", "L-TEXT-BOUND"):
        rows = cell(arm, "A2"); keep = [r for r in rows if re.search(r"\buntil\b", r["note"].split("\n")[0], re.I)]
        print(f"   {arm}: abstain {sum(r['parse']=='ABSTAIN' for r in rows)}/{len(rows)}; where text keeps 'until': abstain {sum(r['parse']=='ABSTAIN' for r in keep)}/{len(keep)} (prediction >= 80%)")
    print("L2 L-TEXT stored items, all (prediction: null):"); contrast("L-TEXT-PROG", "L-TEXT-SIMPLE", None, "L-TEXT-PROG vs L-TEXT-SIMPLE")
    print("   L-TEXT by writer label:"); contrast("L-TEXT-PROG", "L-TEXT-SIMPLE", "FLATTENED", "FLATTENED"); contrast("L-TEXT-PROG", "L-TEXT-SIMPLE", "PRESERVED", "PRESERVED")
    print("per-frame commit (A2) PROG vs SIMPLE, G-TEXT | L-TEXT:")
    for fr in sorted({r["frame"] for r in R.values()}):
        def f(arm): 
            x = [r for r in cell(arm, "A2") if r["frame"] == fr]; return f"{sum(r['parse']=='COMMIT' for r in x)}/{len(x)}"
        print(f"  {fr:15} G {f('G-TEXT-PROG'):>5} vs {f('G-TEXT-SIMPLE'):>5} | L {f('L-TEXT-PROG'):>5} vs {f('L-TEXT-SIMPLE'):>5}")


if __name__ == "__main__":
    writer_side()
    for p in sorted(HERE.glob("reader_g_trace_*.jsonl")): report(p)
