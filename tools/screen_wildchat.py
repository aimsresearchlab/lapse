#!/usr/bin/env python3
"""Deterministic occurrence screen over WildChat-1M user turns (spec Part C).

Zero API calls. Reuses the pattern inventory of tools/screen_ecological.py
(v1.1) plus one first-person locative progressive pattern that the
benchmark screen did not need. Output: category counts over all English
user turns, and a deterministic random sample of hits for two-annotator
adjudication. SCREEN != PREVALENCE: recall is unknown; only the
adjudicated precision times the hit count gives a lower bound.

Usage:
    python3 tools/screen_wildchat.py --out data/wildchat_screen_v1.json \
        --sample-out data/wildchat_adjudication_sample_v1.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter

import pyarrow.parquet as pq

import sys
sys.path.insert(0, os.path.dirname(__file__))
from screen_ecological import PATTERNS as BASE_PATTERNS  # noqa: E402

SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--allenai--WildChat-1M/snapshots/"
    "7d6490e462285cf85d91eabea0f9a954fbddcd1f/data")

PATTERNS = {k: list(v) for k, v in BASE_PATTERNS.items()}
# first-person locative progressive without a lexeme whitelist:
# "I'm ... at/in/with ..." with a V-ing head (excluding the idiomatic
# "I'm looking at", "I'm going to", "I'm trying to", ...)
PATTERNS["PROG"].append(("im_ving_locative", re.compile(
    r"\bI'?a?m (?:currently |now )?"
    r"(?!looking|going|trying|thinking|talking|getting|coming|planning|"
    r"asking|wondering|hoping|writing|sending|putting|feeling)"
    r"[a-z]+ing (?:at|in|with|for) (?:a|an|the|my|our)\b")))

MAX_LEN = 2000   # skip pasted documents; user facts live in short turns


def iter_user_turns():
    for f in sorted(glob.glob(os.path.join(SNAPSHOT, "*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["conversation_hash",
                                                "conversation"])
            for h, conv in zip(t.column("conversation_hash").to_pylist(),
                               t.column("conversation").to_pylist()):
                for i, m in enumerate(conv):
                    if m.get("role") != "user":
                        continue
                    if (m.get("language") or "") != "English":
                        continue
                    c = m.get("content") or ""
                    if not c or len(c) > MAX_LEN:
                        continue
                    yield h, i, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-out", required=True)
    ap.add_argument("--hits-out", default=None,
                    help="write every hit (all categories) as jsonl")
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260902)
    a = ap.parse_args()

    n_turns = n_convs = 0
    seen_convs = set()
    seen_text = set()
    cat_counts = Counter()
    pat_counts = Counter()
    hits = []
    for h, i, c in iter_user_turns():
        n_turns += 1
        if h not in seen_convs:
            seen_convs.add(h)
        th = hashlib.sha1(c.encode()).hexdigest()[:16]
        if th in seen_text:
            continue
        seen_text.add(th)
        found = []
        for cat, pats in PATTERNS.items():
            for name, rx in pats:
                m = rx.search(c)
                if m:
                    found.append({"category": cat, "pattern": name,
                                  "match": m.group(0)})
        if found:
            for x in found:
                pat_counts[x["pattern"]] += 1
            for cat in {x["category"] for x in found}:
                cat_counts[cat] += 1
            # keep a short window around the first match, not the full turn
            m0 = found[0]["match"]
            k = c.find(m0)
            lo, hi = max(0, k - 200), min(len(c), k + 300)
            hits.append({"conversation_hash": h, "turn_index": i,
                         "text_hash": th, "excerpt": c[lo:hi],
                         "excerpt_is_full": (lo == 0 and hi == len(c)),
                         "hits": found})
    n_convs = len(seen_convs)

    # stratified by first-hit category: the adverbial class dominates the
    # hit count and is mostly noise, so a proportional sample would spend
    # the annotation budget on it
    rng = random.Random(a.seed)
    per_cat = {}
    for h in hits:
        per_cat.setdefault(h["hits"][0]["category"], []).append(h)
    k = a.sample_n // len(per_cat)
    sample = []
    for cat in sorted(per_cat):
        pool = per_cat[cat]
        sample += rng.sample(pool, min(k, len(pool)))
    for j, s in enumerate(sample):
        s["sample_id"] = f"wc-{j:03d}"
        s["label_annotator1"] = None
        s["label_annotator2"] = None

    out = {"tool": "screen_wildchat.py v1.0", "corpus": "allenai/WildChat-1M",
           "snapshot": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
           "filters": {"role": "user", "language": "English",
                       "max_len_chars": MAX_LEN, "dedup": "exact text"},
           "note": "SCREEN != PREVALENCE. Recall unknown. Hit count x "
                   "adjudicated precision = lower bound on matching turns.",
           "n_conversations": n_convs, "n_user_turns_screened": n_turns,
           "n_unique_turns": len(seen_text), "n_hit_turns": len(hits),
           "hit_counts_by_category": dict(cat_counts),
           "hit_counts_by_pattern": dict(pat_counts),
           "sample_n": len(sample), "sample_seed": a.seed,
           "sample_design": "stratified, equal per first-hit category"}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    if a.hits_out:
        with open(a.hits_out, "w") as f:
            for h in hits:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
    with open(a.sample_out, "w") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith("n_") or "counts" in k}, indent=1))


if __name__ == "__main__":
    main()
