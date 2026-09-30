#!/usr/bin/env python3
"""Deterministic ecological screen over local LongMemEval/LoCoMo slices.

Charter: "zero-API-cost corpus screen over local LongMemEval/LoCoMo slices
for bounded-marked user facts (ecological leg; screen ≠ prevalence rules
apply)". Finds naturally-occurring utterances whose linguistic FORM encodes
temporal validity (progressive stage-level statives, perfect progressive,
explicit bounds, temporariness adverbials). Output feeds specimen selection
for the two-sided mem0/Graphiti audit — NEVER prevalence claims.

Zero API calls. Deterministic: same slices -> same output.

Usage:
    python3 tools/screen_ecological.py \
        --out data/ecological_screen_v1.json
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SLICE_DIR = Path(__file__).resolve().parent.parent.parent / \
    "memory-eligibility-feasibility" / "dataset-slices"
SLICES = {
    "longmemeval": "longmemeval_controller_full_500_v1.jsonl",
    "locomo": "locomo_controller_full_v1.jsonl",
}

MONTHS = (r"January|February|March|April|May|June|July|August|September|"
          r"October|November|December")

# Category -> list of (pattern_name, compiled regex). Utterance-level, case
# sensitive on I/I'm to stay first-person-precise where it matters.
PATTERNS = {
    # present progressive, stage-level stative-ish lexemes about self
    "PROG": [
        ("im_prog_residence", re.compile(
            r"\bI'?a?m (?:currently |now )?"
            r"(?:staying|living|crashing|subletting|house[- ]?sitting)"
            r" (?:at|in|with|near|over|on)\b")),
        ("im_prog_use", re.compile(
            r"\bI'?a?m (?:currently |now )?"
            r"(?:renting|borrowing|using|driving)\b")),
        ("im_prog_role", re.compile(
            r"\bI'?a?m (?:currently |now )?"
            r"(?:working (?:at|for|from|out of)|interning|volunteering|"
            r"taking (?:a|an|the|this) |enrolled in)\b")),
    ],
    # perfect progressive (continuative)
    "PERF_PROG": [
        ("ive_been", re.compile(
            r"\bI'?(?:ve| have) been "
            r"(?:staying|living|renting|using|driving|working|taking|"
            r"volunteering|interning)\b")),
    ],
    # explicit right-bound
    "BOUND": [
        ("until_month", re.compile(
            r"\b(?:until|till|through) (?:" + MONTHS + r"|next "
            r"(?:week|month|year)|the end of)\b", re.IGNORECASE)),
        ("for_the_next", re.compile(
            r"\bfor the next (?:few|couple|\d+)\b", re.IGNORECASE)),
    ],
    # lexical temporariness adverbials
    "TEMP_ADV": [
        ("temp_adv", re.compile(
            r"\b(?:temporarily|for now|for the time being|"
            r"for a (?:few|couple of) (?:weeks|months|days)|short[- ]term)\b",
            re.IGNORECASE)),
        ("while_clause", re.compile(
            r"\bwhile (?:my|our|the) \w+ (?:is|are|gets?|is being)\b",
            re.IGNORECASE)),
    ],
    # third-party progressive co-residence (household frame analogue)
    "THIRD_PROG": [
        ("third_staying", re.compile(
            r"\b(?:is|are) (?:staying|living) with (?:me|us)\b",
            re.IGNORECASE)),
    ],
}

STOPWORDS = set("""a an the i i'm im my me we our you your is are was were be
been being at in on of for to with and or but so it its this that these those
have has had do does did not no yes now currently""".split())

# LongMemEval memory content: "DATE (Day) HH:MM session id | user: ... |
# assistant: ...". LoCoMo: "Speaker: text".
LME_USER = re.compile(r"\buser: (.*?)(?=\| assistant:|$)", re.DOTALL)
LOCOMO_SPK = re.compile(r"^([A-Z][a-zA-Z]+): (.*)$", re.DOTALL)


def user_texts(dataset, content):
    """Yield (speaker, user-side text) segments from a memory content."""
    if dataset == "longmemeval":
        for m in LME_USER.finditer(content):
            yield "user", m.group(1).strip()
    else:
        m = LOCOMO_SPK.match(content.strip())
        if m:
            yield m.group(1), m.group(2).strip()
        else:
            yield "unknown", content.strip()


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOPWORDS and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ecological_screen_v1.json")
    ap.add_argument("--slice-dir", default=str(SLICE_DIR))
    args = ap.parse_args()
    slice_dir = Path(args.slice_dir)

    results, seen = [], set()
    stats = {ds: Counter() for ds in SLICES}
    uniq_utt = {ds: 0 for ds in SLICES}
    provenance = {}

    for ds, fname in SLICES.items():
        path = slice_dir / fname
        provenance[ds] = {
            "file": str(path),
            "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        }
        # example rows repeat memories heavily; dedupe utterances per dataset
        for line in path.open():
            row = json.loads(line)
            mems = row.get("candidate_memories") or []
            for mem in mems:
                content = mem.get("content", "")
                for speaker, text in user_texts(ds, content):
                    key = (ds, hashlib.sha256(text.encode()).hexdigest())
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq_utt[ds] += 1
                    hits = []
                    for cat, pats in PATTERNS.items():
                        for name, rx in pats:
                            m = rx.search(text)
                            if m:
                                hits.append(
                                    {"category": cat, "pattern": name,
                                     "match": m.group(0)})
                    if not hits:
                        continue
                    for h in hits:
                        stats[ds][h["category"]] += 1
                    # is the marked fact plausibly what the row queries?
                    qa = ((row.get("query") or "") + " " +
                          str(row.get("reference_answer") or ""))
                    overlap = sorted(content_words(text) & content_words(qa))
                    results.append({
                        "dataset": ds,
                        "example_id": row.get("example_id"),
                        "memory_id": mem.get("memory_id"),
                        "speaker": speaker,
                        "utterance": text[:500],
                        "hits": hits,
                        "query_overlap_words": overlap[:12],
                        "queried_overlap_n": len(overlap),
                    })

    out = {
        "generated": "2026-08-25",
        "tool": "tools/screen_ecological.py v1.1",
        "note": ("SCREEN, NOT PREVALENCE. Regex recall is unknown and "
                 "uncalibrated; counts are lower bounds on marked forms in "
                 "these specific slices only."),
        "provenance": provenance,
        "unique_user_utterances_screened": uniq_utt,
        "hit_counts_by_category": {ds: dict(c) for ds, c in stats.items()},
        "n_hit_utterances": len(results),
        "hits": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("unique_user_utterances_screened",
                       "hit_counts_by_category", "n_hit_utterances")},
                     indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
