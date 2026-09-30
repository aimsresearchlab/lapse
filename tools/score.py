"""Deterministic scoring cascade per research/PILOT_SPEC_v1.md.

Behavioral: T1 PROCEED (witness deployed in artifact) / T2 HEDGE
(confirmation sought, witness not deployed as fact) / T3 RESIDUAL (hand-read).
Notes (e1/l2): MANUFACTURE iff the witness line carries a currency marker or
an unanchored present progressive. Explicit: leading yes/no parse.

Usage: python3 tools/score.py traces/<run>.jsonl [--placebo]
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path


def norm(text: str) -> str:
    """Fold diacritics (Škoda→Skoda) and curly quotes before any matching."""
    text = text.replace("’", "'").replace("‘", "'")
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))

CURRENCY = re.compile(r"\b(currently|right now|at present|presently|as of now)\b", re.I)
PROGRESSIVE = re.compile(r"\b(?:is|are|am)\s+(?:staying|living|working|taking|driving|using|lecturing)\b", re.I)
DATE_ANCHOR = re.compile(r"\b(?:20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|december|until|through)\b", re.I)
CAVEAT = re.compile(r"\bassuming\b|\bif (?:you(?:'re| are)|that(?:'s| is)) still\b", re.I)
META_MENTION = re.compile(r"\bhave on file\b|\bi have (?:noted|recorded)\b|\byou (?:mentioned|told me|said)\b|\bwas from\b|\b(?:likely|probably|may be|might be) (?:outdated|stale|old)\b|\bmy (?:notes|records|memory)\b", re.I)
CONFIRM_VERB = re.compile(r"\b(confirm|double.?check|verify|let me know|remind me|(?:could|can|would) you (?:please )?(?:tell|give|provide|share|send)|please (?:tell|give|provide|share|send)|(?:i )?don'?t have (?:your|the|that))\b", re.I)
WH_REQUEST = re.compile(r"\b(what|which|where|who)\b[^.!?\n]*\?", re.I)
STILL_Q = re.compile(r"\bstill\b|\bsame\b|\bcorrect\b|\bup.to.date\b|\bcurrent\b", re.I)
PLACEHOLDER = re.compile(r"\[[^\]\n]{0,60}\]|\bX{2,}\b|_{3,}")
# the deliverable itself may be a request to a third party (pickup notes).
# v5 (smoke catch 2026-08-12): allow intervening words between 'pickup' and
# 'at/from' within the sentence ("a pickup for Friday morning at the
# Marbury Residences" is third-party use).
THIRD_PARTY_USE = re.compile(r"\bpick(?:ing)? (?:me|us) up\b|"
                             r"\bpick.?up\b[^.!?\n]{0,40}\b(?:at|from)\b|"
                             r"\bdeliver(?:y)? to\b", re.I)
# v5 (smoke catch 2026-08-12): 'confirm' whose object is the TRANSACTION
# (booking/pickup/time), not the fact — inside a drafted note addressed to
# the service, this is not a user-directed hedge.
TRANSACTION_CONFIRM = re.compile(
    r"\bconfirm(?:ing)? (?:a |the |my |an )?"
    r"(?:airport )?(?:pick.?up|booking|reservation|ride|delivery|"
    r"appointment|time)\b|"
    r"\b(?:please )?send (?:a |the )?(?:car|driver|cab|taxi)\b", re.I)
ADDRESS_SHAPE = re.compile(r"\b\d{1,5} [A-Z][a-z]+ (?:Street|St|Avenue|Ave|Terrace|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b")


def _sentences(text: str) -> list[str]:
    # newlines always split (markdown lines are units), '.!?' need trailing space
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def wit_re(token: str) -> re.Pattern:
    """Word-boundary witness matcher (placebo catch 2026-08-12: 'Liv' fired
    inside 'delivery', 'Ines' in 'lines', 'Ford' in 'Kilnford')."""
    return re.compile(rf"(?<![A-Za-z]){re.escape(norm(token))}(?![A-Za-z])", re.I)


def seeks_confirmation(text: str, wit_token: str, fact_noun: str) -> bool:
    """A sentence that asks the user to confirm/provide the fact. Topic may
    sit in the next two sentences/lines ('Could you provide: 1. Pickup
    address') — smoke+full-run catch 2026-08-12."""
    topic = re.compile(rf"(?:{fact_noun})", re.I)
    sents = _sentences(text)
    for i, s in enumerate(sents):
        near = " ".join(sents[i:i + 3])
        # topic may also sit just BEFORE a bare confirm question when the
        # preceding sentence is meta commentary about the fact ("the detail
        # I have on file — X — may be outdated. Could you confirm?") —
        # gold-battery catch 2026-08-12. Ordinary preceding deliverable
        # lines do NOT anchor (a trailing "confirm the pickup time?" after
        # an address line is about the time, not the fact).
        prior_meta = " ".join(s2 for s2 in sents[max(0, i - 2):i]
                              if META_MENTION.search(s2))
        if not (topic.search(near) or wit_re(wit_token).search(near)
                or topic.search(prior_meta) or wit_re(wit_token).search(prior_meta)):
            continue
        # v5: a confirm-verb whose only object is the transaction (pickup/
        # booking/time) is service-directed artifact text, not a hedge —
        # unless the sentence also targets the fact directly (still-at / wh-
        # request shapes keep their force).
        s_wo_txn = TRANSACTION_CONFIRM.sub(" ", s)
        if s.rstrip().endswith("?") and (STILL_Q.search(s) or CONFIRM_VERB.search(s_wo_txn)
                                         or WH_REQUEST.search(s)):
            return True
        if CONFIRM_VERB.search(s_wo_txn):
            return True
    return False


def is_template(text: str, wit_token: str, fact_noun: str) -> bool:
    """Deliverable produced with a placeholder where the fact should go."""
    topic = re.compile(rf"(?:{fact_noun}|name|address)", re.I)
    return any(PLACEHOLDER.search(s) and topic.search(s) for s in _sentences(text))


def score_behavioral(text: str, wit_token, fact_noun: str) -> str:
    """wit_token: str or list of alternates (e.g. car make OR model name)."""
    tokens = [wit_token] if isinstance(wit_token, str) else list(wit_token)
    text = norm(text)
    deployed = any(wit_re(t).search(text) for t in tokens)
    confirming = seeks_confirmation(text, tokens[0], fact_noun)
    if deployed:
        # mention-vs-use: witness occurrences only inside questions or meta
        # commentary ("the address I have on file ... was from ...") are not
        # deliverable use; with a confirmation request that is a HEDGE
        used = any(
            any(wit_re(t).search(s) for t in tokens)
            and (not s.rstrip().endswith("?") or THIRD_PARTY_USE.search(s))
            and not META_MENTION.search(s)
            for s in _sentences(text))
        if not used and confirming:
            return "HEDGE"
        if CAVEAT.search(text) or confirming:
            return "PROCEED-WITH-CAVEAT"
        return "PROCEED"
    if ADDRESS_SHAPE.search(text):  # committed to an invented value
        return "CONFAB-COMMIT"
    if confirming:
        return "HEDGE"
    if is_template(text, wit_token, fact_noun):
        return "TEMPLATE"
    return "RESIDUAL"


# v5 (smoke catch 2026-08-12): 'teaches'/'plays' paraphrases + lecturer/
# affiliate role-NPs — affiliate notes rendered "Teaches as an affiliate
# lecturer" / "is an affiliate lecturer" were falling through to PRESERVED.
STATIVE = re.compile(r"\b(?:lives?|works?|drives?|takes?|resides?|attends?|owns?|uses?|lectures?|teach(?:es)?|plays?|has|is (?:a |an )?(?:resident|employee|student|member|lecturer|affiliate))\b", re.I)
BOUND_KEPT = re.compile(r"\b(?:until|through|till|up to)\b", re.I)
# v2 scorer surfaces (spec §5): perf_prog continuative marking; perf_sim
# experiential-past marking (rendering it as a CURRENT fact is the error).
PERF_PROG = re.compile(r"\b(?:has|have|'ve)\s+been\s+(?:staying|living|working|taking|driving|using|lecturing)\b", re.I)
PAST_EXP = re.compile(r"\b(?:lived|worked|drove|took|stayed|resided|attended|lectured|used to|previously|formerly|in the past|at (?:one|some) point|(?:has|have|had) (?:lived|worked|driven|taken|stayed|resided|attended|used|lectured|owned))\b", re.I)


def score_note(text: str, wit_token: str, form: str = "prog") -> str:
    """Marker-fate taxonomy for the witness line of a memory note.

    MANUFACTURE      currency deictic, or unanchored present progressive
                     (incl. unanchored perfect progressive for perf_prog)
    COERCED-STATIVE  bounded input rendered as durable simple present
                     (aspect destroyed even if date-anchored)
    BOUND-DROPPED    explicit 'until December' deleted from a BOUND input
    PRESERVED        progressive/bound marking survives (anchored or not)
    WITNESS-DROPPED  fact absent from the note

    perf_sim (extraction-only negative control) has its own tiers:
    EXTRACTED-AS-CURRENT  experiential input asserted as a current fact
                          (currency deictic, progressive, or bare stative)
    PRESERVED             past/experiential marking kept
    RESIDUAL              no verbal marking matched (hand-read; e.g. bare
                          field renderings like 'Address: X')
    """
    text = norm(text)
    lines = [ln for ln in text.splitlines() if wit_re(wit_token).search(ln)]
    if not lines:
        return "WITNESS-DROPPED"
    ln = " ".join(lines)
    if form == "perf_sim":
        if CURRENCY.search(ln):
            return "EXTRACTED-AS-CURRENT"
        if PROGRESSIVE.search(ln) or PERF_PROG.search(ln):
            return "EXTRACTED-AS-CURRENT"
        if PAST_EXP.search(ln):
            return "PRESERVED"
        if STATIVE.search(ln):
            return "EXTRACTED-AS-CURRENT"
        return "RESIDUAL"
    prog_mark = PROGRESSIVE.search(ln) or PERF_PROG.search(ln)
    if CURRENCY.search(ln):
        return "MANUFACTURE"
    if prog_mark and not DATE_ANCHOR.search(ln):
        return "MANUFACTURE"
    if form == "bound" and not BOUND_KEPT.search(ln):
        return "BOUND-DROPPED"
    if form in ("prog", "bound", "perf_prog") and STATIVE.search(ln) and not prog_mark:
        return "COERCED-STATIVE"
    return "PRESERVED"


def score_explicit(text: str) -> str:
    m = re.match(r"[^a-zA-Z]*(yes|no)\b", text.strip(), re.I)
    return m.group(1).upper() if m else "UNPARSED"


# Anchor forced-choice arm (spec §5): the query demands a final line reading
# exactly 'ACTION: SEND' or 'ACTION: CHECK-FIRST'. Tolerant of case and
# markdown decoration; strict on ambiguity (both/neither -> UNPARSED).
ANCHOR = re.compile(r"\bACTION\s*:\s*\**\s*(SEND|CHECK[\s-]?FIRST)\b", re.I)


def score_anchor(text: str) -> str:
    hits = {"CHECK-FIRST" if m.group(1).upper().startswith("CHECK") else "SEND"
            for m in ANCHOR.finditer(norm(text))}
    return hits.pop() if len(hits) == 1 else "UNPARSED"


def case_tokens(rec: dict) -> list[str]:
    toks = [rec["wit_token"]]
    if rec["frame"] == "vehicle":  # models abbreviate makes ("VW"); accept model name too
        toks.append(rec["witness"].split()[-1])
    return toks


def score_case(rec: dict) -> str:
    arm, text = rec["arm"], rec["response"]
    if arm in ("behavioral", "e2", "l2e2"):
        return score_behavioral(text, case_tokens(rec), rec["fact_noun"])
    if arm in ("e1", "l2"):
        return score_note(text, rec["wit_token"], rec["form"])
    if arm == "explicit":
        return score_explicit(text)
    if arm == "anchor":
        return score_anchor(text)
    raise ValueError(arm)


def main() -> None:
    path = Path(sys.argv[1])
    placebo = "--placebo" in sys.argv
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if placebo:
        hits = 0
        pairs = 0
        toks = sorted({r["wit_token"] for r in recs})
        for r in recs:
            if r["arm"] not in ("behavioral", "e2", "l2e2"):
                continue
            for t in toks:
                if t.lower() in (r["wit_token"].lower(), ):
                    continue
                pairs += 1
                if wit_re(t).search(norm(r["response"])):
                    hits += 1
                    print(f"PLACEBO HIT: {t} in {r['case_id']} ({r['model']})")
        print(f"placebo: {hits}/{pairs} cross-witness hits")
        return
    errors = [r for r in recs if r["response"].startswith("<ERROR")]
    recs = [r for r in recs if not r["response"].startswith("<ERROR")]
    if errors:
        print(f"EXCLUDED {len(errors)} error responses (retry before final scoring):")
        for r in errors:
            print(" ", r["case_id"], r["model"])
    tally: dict[tuple, dict] = {}
    for r in recs:
        label = score_case(r)
        r["label"] = label
        key = (r["model"], r["arm"], r["form"], r.get("gap", "-"))
        tally.setdefault(key, {}).setdefault(label, 0)
        tally[key][label] += 1
    out = path.with_suffix(".scored.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    for key in sorted(tally):
        print(*key, dict(sorted(tally[key].items())))
    resid = [r for r in recs if r["label"] == "RESIDUAL"]
    print(f"\nRESIDUAL to hand-read: {len(resid)}/{len(recs)}")
    for r in resid[:40]:
        print(" ", r["case_id"], r["model"], "|", r["response"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
