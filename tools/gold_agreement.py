"""Gold-subset agreement stats (SCORING_V2.md item 2 reporting).

Consumes the JSONL exports from web/annotate.html plus data/gold_key_v1.json.

Stages:
  1 export  -> sanity + per-label counts (no detector comparison; key stays
               functionally sealed: detector labels are read only in stage 3)
  2 exports -> % agreement + Cohen's kappa(human-human), 5-label and binary;
               writes data/gold_disagreements.jsonl for adjudication
                 (adjudicate by adding {"gid": ..., "label": ...} lines to
                  data/gold_adjudicated.jsonl)
  2 exports + full adjudication -> gold = agreed + adjudicated; reports
               kappa(cascade-gold) and kappa(judge-gold), overall and
               per sampling tier, against the 0.75 / 0.65 spec bar.

Binary partition (per RESULTS_DEEPSEEK correction): COMMIT={PROCEED},
NON={HEDGE, TEMPLATE}; PWC and CONFAB-COMMIT excluded, n disclosed.

Usage:
  python3 tools/gold_agreement.py annA.jsonl [annB.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_PATH = ROOT / "data/gold_key_v1.json"
ADJ_PATH = ROOT / "data/gold_adjudicated.jsonl"
DIS_PATH = ROOT / "data/gold_disagreements.jsonl"
LABELS = {"PROCEED", "PROCEED-WITH-CAVEAT", "HEDGE", "TEMPLATE", "CONFAB-COMMIT"}


def load_ann(p: Path) -> dict[str, str]:
    out = {}
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r["label"] not in LABELS:
            sys.exit(f"{p}: bad label {r['label']!r} for {r['gid']}")
        if r["gid"] in out:
            sys.exit(f"{p}: duplicate gid {r['gid']}")
        out[r["gid"]] = r["label"]
    return out


def kappa(pairs: list[tuple[str, str]]) -> float:
    labs = sorted({x for p in pairs for x in p})
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == lab for a, _ in pairs) / n)
             * (sum(b == lab for _, b in pairs) / n) for lab in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def binary(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """COMMIT={PROCEED} vs NON={HEDGE,TEMPLATE}; PWC/CONFAB excluded."""
    keep = [(a, b) for a, b in pairs
            if {a, b} <= {"PROCEED", "HEDGE", "TEMPLATE"}]
    coll = [("COMMIT" if a == "PROCEED" else "NON",
             "COMMIT" if b == "PROCEED" else "NON") for a, b in keep]
    return coll, len(pairs) - len(keep)


def report(tag: str, pairs: list[tuple[str, str]]) -> None:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    print(f"{tag}: n={n}  agreement {agree}/{n} ({agree / n:.1%})  "
          f"kappa={kappa(pairs):.3f}")
    coll, excl = binary(pairs)
    if coll:
        ag2 = sum(a == b for a, b in coll)
        print(f"  binary (PROCEED vs HEDGE+TEMPLATE; {excl} PWC/CONFAB "
              f"excluded): n={len(coll)}  {ag2}/{len(coll)} "
              f"({ag2 / len(coll):.1%})  kappa={kappa(coll):.3f}")


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    anns = [load_ann(p) for p in paths]
    key = {k["gid"]: k for k in json.loads(KEY_PATH.read_text())}

    for p, a in zip(paths, anns):
        missing = len(key) - len(a)
        print(f"{p.name}: {len(a)} labels"
              + (f"  (PARTIAL — {missing} unlabeled)" if missing else "")
              + f"  {dict(Counter(a.values()))}")

    if len(anns) == 1:
        print("\nOne export loaded. Waiting on annotator #2 for agreement "
              "stats; detector labels stay sealed until then.")
        return

    a, b = anns[0], anns[1]
    common = sorted(set(a) & set(b))
    print(f"\n== Human-human ({paths[0].name} vs {paths[1].name}) ==")
    report("5-label", [(a[g], b[g]) for g in common])

    dis = [g for g in common if a[g] != b[g]]
    adj: dict[str, str] = {}
    if ADJ_PATH.exists():
        for ln in ADJ_PATH.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                adj[r["gid"]] = r["label"]
    unresolved = [g for g in dis if g not in adj]
    if unresolved:
        with DIS_PATH.open("w") as f:
            for g in unresolved:
                f.write(json.dumps({
                    "gid": g, paths[0].stem: a[g], paths[1].stem: b[g],
                    "query": None,  # look items up in the UI by gid, stay blind
                }) + "\n")
        print(f"\n{len(dis)} disagreements; {len(unresolved)} unadjudicated -> "
              f"{DIS_PATH.name}.\nAdjudicate (discuss, still blind to "
              f"condition), append {{gid,label}} lines to {ADJ_PATH.name}, "
              f"then re-run. Detector comparison withheld until then.")
        return

    gold = {g: (a[g] if a[g] == b[g] else adj[g]) for g in common}
    print(f"\n== Detector vs gold (n={len(gold)}) ==")
    # extra detector columns: final-judge files from tools/judge_gold.py
    extra: dict[str, dict[str, str]] = {}
    for jf in sorted((ROOT / "data").glob("gold_items.judged.*.jsonl")):
        slug = jf.name[len("gold_items.judged."):-len(".jsonl")]
        extra[f"judge:{slug}"] = {
            r["gid"]: r["judge"] for r in map(json.loads, jf.read_text().splitlines())
            if r["judge"] in LABELS}
    dets: dict[str, dict[str, str]] = {
        det: {g: key[g][det] for g in gold if key[g].get(det)}
        for det in ("cascade", "judge")} | extra
    for det, labels in dets.items():
        pairs = [(labels[g], gold[g]) for g in gold if g in labels]
        report(det, pairs)
        for tier in sorted({key[g]["tier"] for g in gold}):
            tp = [(labels[g], gold[g]) for g in gold
                  if key[g]["tier"] == tier and g in labels]
            if tp:
                report(f"  {det} [{tier}]", tp)
    print("\nSpec bar: both detectors kappa >= 0.75 vs gold; < 0.65 -> revise, "
          "re-freeze, re-validate before unblinding.")


if __name__ == "__main__":
    main()
