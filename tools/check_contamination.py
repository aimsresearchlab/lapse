"""Witness contamination check (spec §6 audit 6; release-eng report §6).

Queries the infini-gram API for exact-count occurrences of every witness
string (all 8 frames) in RedPajama and Dolma. Threshold: <=5 occurrences per
witness. Result is committed as data/witness_contamination_check.json and
gates spec freeze (tests/test_stimuli_v2.py::test_audit6 reads it).

Also prints witnesses needing the MANUAL web check (company-like names —
new-frame institutions and makers).

Usage: python3 tools/check_contamination.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stimuli_v2 import frames_v2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "witness_contamination_check.json"
API = "https://api.infini-gram.io/"
INDICES = {"redpajama": "v4_rpj_llama_s4", "dolma": "v4_dolma-v1_7_llama"}
THRESHOLD = 5


def count(index: str, query: str) -> int:
    body = json.dumps({"index": index, "query_type": "count",
                       "query": query}).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
            if "count" in resp:
                time.sleep(0.7)  # politeness delay (403 = rate limit)
                return resp["count"]
            raise RuntimeError(resp.get("error", str(resp)))
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _save(results: dict) -> None:
    OUT.write_text(json.dumps(dict(
        checked=date.today().isoformat(), api=API, indices=INDICES,
        threshold=THRESHOLD, witnesses=results), indent=2))


def main() -> None:
    frames = frames_v2()
    results: dict[str, dict] = {}
    if OUT.exists():  # resume: keep completed rows across rate-limit aborts
        results = json.loads(OUT.read_text())["witnesses"]
        print(f"resume: {len(results)} witnesses already checked")
    flagged = []
    for frame, spec in frames.items():
        for w in spec["witnesses"]:
            if w in results:
                if results[w]["over"]:
                    flagged.append((w, results[w]["counts"]))
                continue
            counts = {}
            for name, index in INDICES.items():
                counts[name] = count(index, w)
            over = any(c > THRESHOLD for c in counts.values())
            results[w] = dict(frame=frame, counts=counts, over=over)
            if over:
                flagged.append((w, counts))
            print(f"{frame:15s} {w:35s} {counts}"
                  + ("  <-- OVER" if over else ""))
            _save(results)  # incremental: survive rate-limit aborts
    _save(results)
    print(f"\n{len(results)} witnesses -> {OUT}")
    if flagged:
        print(f"OVER THRESHOLD ({len(flagged)}):")
        for w, c in flagged:
            print(" ", w, c)
        print("Replace these witnesses before freeze.")
    print("\nMANUAL web check still required for company-like names "
          "(institutions, makers) — spec §6.")


if __name__ == "__main__":
    main()
