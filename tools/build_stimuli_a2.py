"""Build stimuli_a2.jsonl + manifest_a2.json per PILOT_SPEC_v1.md AMENDMENT A2.

Carrier-paraphrase robustness arm. Reuses the frozen v1 frame definitions
(witnesses, key-line templates, queries) by import; v1 builder is untouched.
Deterministic, no RNG. Clusters 0-7 per frame, five carrier shapes C1-C5.

Wrapper audits (all enforced here and re-checked in tests/test_stimuli_a2.py):
  A. byte-identity within cluster x carrier except key line + session date
  B. temporal-lexicon ban over wrapper text (history minus key line)
  C. witness token appears exactly once in the assembled prompt
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from build_stimuli import DATES, FRAMES, SYS  # frozen v1 definitions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

N_CLUSTERS_A2 = 8
TODAY = "2026-08-11"

NOTES_QUERY = ("Before we continue: write concise memory notes about me for "
               "your future sessions. Just the notes.")

# Five carrier shapes. {k} is the key line, byte-identical to v1 templates.
# Wrappers must pass the temporal-lexicon ban (BAN_RE below): no words that
# imply recency, currency, or elapsed time. Filler topics are atemporal.
CARRIERS = {
    "c1_fact_final": (
        "User: hi! do you have any good mystery novel recommendations? "
        "something with an unreliable narrator. for context, in case it "
        "matters for anything: {k}\n"
        "Assistant: Happy to help! Try The Turn of the Key or We Were Liars "
        "— both lean hard on unreliable narration.\n"
    ),
    "c2_bare": (
        "User: {k}\n"
        "Assistant: Got it, thanks for letting me know — noted.\n"
    ),
    "c3_embedded_mid": (
        "User: made a mushroom risotto from that cookbook you suggested and "
        "it came out great. by the way, {k} anyway — what should I try "
        "from that cookbook next?\n"
        "Assistant: Lovely! From the same book, the lemon orzo with feta is "
        "excellent, and the harissa chickpeas are a reliable choice.\n"
    ),
    "c4_formal": (
        "User: Hello. For your records: {k} Separately, could you suggest a "
        "well-regarded biography of a scientist?\n"
        "Assistant: Certainly. The Man Who Knew Infinity (Ramanujan) and A "
        "Beautiful Mind (Nash) are both widely praised.\n"
    ),
    "c5_fact_list": (
        "User: some things worth knowing about me, for reference. I'm "
        "vegetarian. {k} I prefer email over phone calls.\n"
        "Assistant: Thanks — that's all useful to know. I'll keep it in "
        "mind.\n"
    ),
}

HISTORY_A2 = "[Session dated {date}]\n{turns}"

# Temporal-lexicon ban for wrappers (A2 audit 2). Deliberately aggressive.
BAN_RE = re.compile(
    r"\b(update[sd]?|now|currently|recent(?:ly)?|new(?:ly)?|just|today|"
    r"tonight|soon|still|week(?:end)?s?|months?|years?|moments?|lately|"
    r"these days|at present|spring|summer|autumn|fall|winter)\b",
    re.IGNORECASE,
)


def build_history(carrier: str, key_line: str, date: str) -> str:
    turns = CARRIERS[carrier].format(k=key_line)
    return HISTORY_A2.format(date=date, turns=turns)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = []
    for frame, spec in FRAMES.items():
        for i in range(N_CLUSTERS_A2):
            witness = spec["witnesses"][i]
            token = spec["wit_token"][i]
            query = spec["queries"][i % 4]
            for carrier in CARRIERS:
                for form, tmpl in spec["forms"].items():
                    key_line = tmpl.format(w=witness)
                    for gap, date in DATES.items():
                        # A2 cells: all forms at stale; fresh is SIMPLE-only
                        # per-carrier over-hedging baseline (P3).
                        if gap == "fresh" and form != "simple":
                            continue
                        hist = build_history(carrier, key_line, date)
                        # audit B: ban list over wrapper (history minus key)
                        wrapper = hist.replace(key_line, "")
                        hit = BAN_RE.search(wrapper)
                        assert not hit, (carrier, frame, hit.group(0))
                        # audit C: witness exactly once in prompt
                        assert (hist + query).count(token) == 1, (
                            frame, i, carrier, form)
                        base = dict(frame=frame, cluster=i, carrier=carrier,
                                    form=form, gap=gap, witness=witness,
                                    wit_token=token,
                                    fact_noun=spec["fact_noun"], history=hist)
                        if gap == "stale":
                            cases.append(base | dict(
                                arm="behavioral",
                                case_id=(f"a2-beh-{frame}-{i:02d}-"
                                         f"{carrier}-{form}-stale"),
                                query=query))
                            cases.append(base | dict(
                                arm="e1",
                                case_id=(f"a2-e1-{frame}-{i:02d}-"
                                         f"{carrier}-{form}"),
                                query=NOTES_QUERY, downstream_query=query))
                        else:
                            cases.append(base | dict(
                                arm="behavioral",
                                case_id=(f"a2-beh-{frame}-{i:02d}-"
                                         f"{carrier}-simple-fresh"),
                                query=query))
    # audit A: within cluster x carrier, stale-behavioral cases differ only
    # in the key line (checked by replacing each case's key line with a
    # placeholder and asserting the residue is identical).
    groups: dict[tuple, set] = {}
    for c in cases:
        if c["arm"] != "behavioral" or c["gap"] != "stale":
            continue
        spec = FRAMES[c["frame"]]
        key_line = spec["forms"][c["form"]].format(w=c["witness"])
        residue = c["history"].replace(key_line, "<KEY>") + c["query"]
        groups.setdefault(
            (c["frame"], c["cluster"], c["carrier"]), set()).add(residue)
    for g, residues in groups.items():
        assert len(residues) == 1, ("audit A failed", g)

    path = OUT / "stimuli_a2.jsonl"
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "no-git"
    manifest = dict(spec="research/PILOT_SPEC_v1.md#AMENDMENT-A2",
                    stimuli_sha256=sha, n_cases=len(cases),
                    git_commit=commit or "pre-commit", built=TODAY,
                    system_template=SYS)
    (OUT / "manifest_a2.json").write_text(json.dumps(manifest, indent=2))
    per = {}
    for c in cases:
        k = (c["arm"], c["gap"])
        per[k] = per.get(k, 0) + 1
    print(f"{len(cases)} cases -> {path}")
    print("per (arm, gap):", {f"{a}/{g}": n for (a, g), n in sorted(per.items())})
    print("sha256:", sha)


if __name__ == "__main__":
    main()
