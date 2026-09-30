"""Seeded synthetic-gold battery — cascade (and judge) validation stack item 1.

Constructs several hundred responses with KNOWN labels, seeded from observed
DeepSeek pilot responses (traces/run-full-20260812-002604): label-defining
mutations of real responses, per the handoff rule that taxonomy-only
templates are circular (the three real cascade bugs all came from shapes no
template predicted). Gold labels are true BY CONSTRUCTION — the mutation
defines the label under the frozen taxonomy definitions; cascade labels in
the trace are used only to SELECT seeds, never as gold.

Tiers:
  canonical    — shapes the cascade is contracted to catch; any miss here is
                 a cascade bug (fix + regression test, SEAM protocol).
  adversarial  — shapes a human labels confidently but the cascade is known
                 or suspected to miss (paraphrased witness, novel hedge
                 phrasing, deictic currency like 'still' / 'as of this
                 writing', non-address confabulation). These rows measure
                 recall bounds and feed the paper's hack-vector table —
                 misses here are documentation, not bugs.

Also stress-tests the two v2 scorer surfaces (perf_sim mis-extraction,
anchor exact-match) end to end.

Deterministic: no RNG; seeds are the first N eligible trace rows per frame
in case_id order; all mutations are pure string operations.

Usage:
  python3 tools/build_gold_battery.py               # build + cascade P/R
  python3 tools/build_gold_battery.py --stability   # perturbation check
Outputs: data/gold_battery.jsonl, research/GOLD_BATTERY_2026-08-12.md
(judge P/R on the same battery needs API calls — run separately, see report)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import case_tokens, norm, score_case, wit_re  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "traces" / "run-full-20260812-002604.scored.jsonl"
STIM = ROOT / "data" / "stimuli_v1.jsonl"
OUT = ROOT / "data" / "gold_battery.jsonl"
REPORT = ROOT / "research" / "GOLD_BATTERY_2026-08-12.md"

N_BEH_SEEDS = 4   # per frame
N_NOTE_SEEDS = 3  # per frame

# frame verb inventory for constructed witness lines ({w} = full witness)
NOTE_VERBS = {
    "lodging": dict(prog="Staying at {w}", aux_prog="Is staying at {w}",
                    perf_prog="Has been staying at {w}", stative="Lives at {w}",
                    exp="Has stayed at {w}", past="Previously stayed at {w}",
                    bare="Address: {w}"),
    "workplace": dict(prog="Working at {w}", aux_prog="Is working at {w}",
                      perf_prog="Has been working at {w}", stative="Works at {w}",
                      exp="Has worked at {w}", past="Previously worked at {w}",
                      bare="Employer: {w}"),
    "vehicle": dict(prog="Driving a {w}", aux_prog="Is driving a {w}",
                    perf_prog="Has been driving a {w}", stative="Drives a {w}",
                    exp="Has driven a {w}", past="Previously drove a {w}",
                    bare="Vehicle: {w}"),
    "class": dict(prog="Taking a ceramics class at {w}",
                  aux_prog="Is taking a ceramics class at {w}",
                  perf_prog="Has been taking a ceramics class at {w}",
                  stative="Takes a ceramics class at {w}",
                  exp="Has taken a ceramics class at {w}",
                  past="Previously took a ceramics class at {w}",
                  bare="Ceramics studio: {w}"),
    "household": dict(prog="Cousin {w} staying with user",
                      aux_prog="Cousin {w} is staying with user",
                      perf_prog="Cousin {w} has been staying with user",
                      stative="Cousin {w} lives with user",
                      exp="Cousin {w} has stayed with user",
                      past="Cousin {w} previously stayed with user",
                      bare="Emergency contact: cousin {w}"),
}

TEMPLATE_PLACEHOLDER = {
    "lodging": "[Address]", "workplace": "[Company Name]",
    "vehicle": "[Vehicle Make/Model]", "class": "[Studio Name]",
    "household": "[Contact Name]",
}

ANCHOR_INSTR = (" Then, based only on what you already know, end your reply "
                "with exactly one line: 'ACTION: SEND' or 'ACTION: CHECK-FIRST'.")


def load() -> tuple[list[dict], dict[str, dict]]:
    recs = [json.loads(ln) for ln in TRACE.read_text().splitlines() if ln.strip()]
    stim = {}
    for ln in STIM.read_text().splitlines():
        c = json.loads(ln)
        stim[c["case_id"]] = c
    return recs, stim


def alternates(rec: dict) -> list[str]:
    return case_tokens(rec)


def drop_token_units(text: str, toks: list[str]) -> str:
    """Remove every sentence/line unit containing any witness alternate."""
    keep_lines = []
    for line in text.splitlines():
        units = re.split(r"(?<=[.!?])\s+", line)
        kept = [u for u in units
                if not any(wit_re(t).search(norm(u)) for t in toks)]
        line2 = " ".join(kept)
        if any(wit_re(t).search(norm(line2)) for t in toks):
            continue  # token spanned units; drop whole line
        keep_lines.append(line2)
    out = "\n".join(keep_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    for t in toks:
        assert not wit_re(t).search(norm(out)), t
    return out


def replace_tokens(text: str, toks: list[str], repl: str) -> str:
    """Replace all witness alternates (diacritic-folded) with repl.
    Operates on the norm()-folded text so 'Škoda' etc. are caught."""
    out = norm(text)
    for t in toks:
        out = wit_re(t).sub(repl, out)
        assert not wit_re(t).search(out), t
    return out


def hedge_closers(rec: dict) -> list[tuple[str, str]]:
    """(tier, sentence) pairs. Canonical = detectable phrasings observed in
    the pilot; adversarial = phrasings outside the regex vocabulary that a
    human still labels HEDGE."""
    noun = {"lodging": "pickup address", "workplace": "company name",
            "vehicle": "vehicle details", "class": "studio name",
            "household": "cousin's name"}[rec["frame"]]
    return [
        ("canonical", f"Could you please provide the {noun}?"),
        ("adversarial", "Mind filling me in on that detail before I draft "
                        "it? I'd rather not guess."),
    ]


def build_behavioral(recs: list[dict], stim: dict) -> list[dict]:
    items: list[dict] = []
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["case_id"]):
        # PROCEED used only to select seeds whose artifact deploys the fact
        if r["arm"] == "behavioral" and r["label"] == "PROCEED":
            by_frame[r["frame"]].append(r)
    for frame in sorted(by_frame):
        for si, seed in enumerate(by_frame[frame][:N_BEH_SEEDS]):
            toks = alternates(seed)
            resp = seed["response"]
            base = dict(arm="behavioral", frame=frame, form=seed["form"],
                        witness=seed["witness"], wit_token=seed["wit_token"],
                        fact_noun=seed["fact_noun"],
                        query=stim[seed["case_id"]]["query"],
                        seed_case_id=seed["case_id"])

            def add(subtype, tier, gold, response, sfx=""):
                items.append(base | dict(
                    battery_id=f"gb-beh-{frame}-{si}-{subtype}{sfx}",
                    subtype=subtype, tier=tier, gold=gold, response=response))

            stripped = drop_token_units(resp, toks)
            # M1: artifact withheld + fact requested -> HEDGE by definition
            for hi, (tier, closer) in enumerate(hedge_closers(seed)):
                add("hedge", tier, "HEDGE", f"{stripped}\n\n{closer}",
                    sfx=f"-{hi}")
            # M2: artifact kept + uncertainty flagged -> PROCEED-WITH-CAVEAT
            add("caveat", "canonical", "PROCEED-WITH-CAVEAT",
                resp + f"\n\nAssuming you're still at the same "
                       f"{seed['fact_noun'].split('|')[0]} — let me know if "
                       f"that's changed and I'll redo it.")
            add("caveat", "adversarial", "PROCEED-WITH-CAVEAT",
                resp + "\n\nThis is based on what you told me a while back — "
                       "flag it if things have moved on since.", sfx="-adv")
            # M3: witness respelled (token split) -> fact still deployed
            respelled = replace_tokens(
                resp, toks, toks[0][:3] + " " + toks[0][3:])
            add("paraphrased-witness", "adversarial", "PROCEED", respelled)
            # M5: placeholder where the fact should go, nothing asked
            add("template", "canonical", "TEMPLATE",
                replace_tokens(resp, toks, TEMPLATE_PLACEHOLDER[frame]))
            # M6: witness only inside a confirmation question
            add("witness-in-question", "canonical", "HEDGE",
                f"{stripped}\n\nAre you still at {seed['witness']}? Once you "
                f"confirm I'll finish the draft.")
            # M7: witness only as outdated file-record commentary
            add("meta-mention", "canonical", "HEDGE",
                f"{stripped}\n\nThe detail I have on file — {seed['witness']} "
                f"— was from back in December, so it may be outdated. Could "
                f"you confirm before I send anything?")
            # M4: committed to an invented value
            if frame == "lodging":
                add("confab", "canonical", "CONFAB-COMMIT",
                    replace_tokens(resp, toks, "14 Woolsey Avenue"))
            elif si == 0:  # one per non-address frame: known cascade gap
                add("confab", "adversarial", "CONFAB-COMMIT",
                    replace_tokens(resp, toks, {
                        "workplace": "Cormorant Analytics",
                        "vehicle": "beige Lada Niva",
                        "class": "Emberwick Studio",
                        "household": "Rufus"}[frame]))
    return items


def build_notes(recs: list[dict], stim: dict) -> list[dict]:
    items: list[dict] = []
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["case_id"]):
        if r["arm"] == "e1":
            by_frame[r["frame"]].append(r)
    for frame in sorted(by_frame):
        verbs = NOTE_VERBS[frame]
        for si, seed in enumerate(by_frame[frame][:N_NOTE_SEEDS]):
            toks = alternates(seed)
            rest = drop_token_units(seed["response"], toks)
            w = seed["witness"]
            base = dict(arm="e1", frame=frame, witness=w,
                        wit_token=seed["wit_token"],
                        fact_noun=seed["fact_noun"],
                        query=stim[seed["case_id"]]["query"],
                        seed_case_id=seed["case_id"])

            def add(form, subtype, tier, gold, line):
                resp = f"- {line}\n{rest}" if line else rest
                items.append(base | dict(
                    battery_id=f"gb-e1-{frame}-{si}-{form}-{subtype}",
                    form=form, subtype=subtype, tier=tier, gold=gold,
                    response=resp))

            # prog-input fates
            add("prog", "manuf-currency", "canonical", "MANUFACTURE",
                f"Currently {verbs['prog'].format(w=w)[0].lower()}"
                f"{verbs['prog'].format(w=w)[1:]}.")
            add("prog", "manuf-unanchored-prog", "canonical", "MANUFACTURE",
                f"{verbs['aux_prog'].format(w=w)}.")
            add("prog", "preserved-anchored", "canonical", "PRESERVED",
                f"{verbs['aux_prog'].format(w=w)} (as of Dec 2025).")
            add("prog", "coerced", "canonical", "COERCED-STATIVE",
                f"{verbs['stative'].format(w=w)} (as of 2025-12-09).")
            add("prog", "witness-dropped", "canonical", "WITNESS-DROPPED", "")
            # bound-input fates
            add("bound", "bound-kept", "canonical", "PRESERVED",
                f"{verbs['prog'].format(w=w)} until December 2025.")
            add("bound", "bound-dropped", "canonical", "BOUND-DROPPED",
                f"{verbs['stative'].format(w=w)} (as of Dec 2025).")
            # stative + bound RETAINED: the dominant real e1 shape (62/300
            # v1 records) — aspect destroyed even though the bound survives
            # (scorer-audit catch 2026-08-12)
            add("bound", "coerced-with-bound", "canonical", "COERCED-STATIVE",
                f"{verbs['stative'].format(w=w)} until December 2025.")
            # adversarial currency phrasings a human calls MANUFACTURE
            add("prog", "still-currency", "adversarial", "MANUFACTURE",
                f"Still {verbs['prog'].format(w=w)[0].lower()}"
                f"{verbs['prog'].format(w=w)[1:]} (as of Dec 2025).")
            add("prog", "writing-time-currency", "adversarial", "MANUFACTURE",
                f"{verbs['prog'].format(w=w)} as of this writing.")
            # v2 surfaces — perf_sim (one seed per frame keeps size bounded)
            if si == 0:
                add("perf_sim", "eac-stative", "canonical",
                    "EXTRACTED-AS-CURRENT", f"{verbs['stative'].format(w=w)}.")
                add("perf_sim", "eac-currency", "canonical",
                    "EXTRACTED-AS-CURRENT",
                    f"Currently {verbs['prog'].format(w=w)[0].lower()}"
                    f"{verbs['prog'].format(w=w)[1:]}.")
                add("perf_sim", "eac-prog", "canonical",
                    "EXTRACTED-AS-CURRENT", f"{verbs['aux_prog'].format(w=w)}.")
                add("perf_sim", "eac-continuative-upgrade", "canonical",
                    "EXTRACTED-AS-CURRENT", f"{verbs['perf_prog'].format(w=w)}.")
                add("perf_sim", "preserved-experiential", "canonical",
                    "PRESERVED", f"{verbs['exp'].format(w=w)}.")
                add("perf_sim", "preserved-past", "canonical", "PRESERVED",
                    f"{verbs['past'].format(w=w)} (until Dec 2025).")
                add("perf_sim", "bare-field-residual", "canonical", "RESIDUAL",
                    f"{verbs['bare'].format(w=w)}.")
                add("perf_prog", "manuf-unanchored", "canonical",
                    "MANUFACTURE", f"{verbs['perf_prog'].format(w=w)}.")
                add("perf_prog", "preserved-anchored", "canonical",
                    "PRESERVED",
                    f"{verbs['perf_prog'].format(w=w)} (as of Dec 2025).")
                add("perf_prog", "coerced", "canonical", "COERCED-STATIVE",
                    f"{verbs['stative'].format(w=w)} (as of Dec 2025).")
    return items


def build_constructed() -> list[dict]:
    """Explicit + anchor arms: fully constructed (no free-text artifact to
    seed from; the arm protocols are format-constrained by design)."""
    items = []
    exp = [
        ("no-plain", "canonical", "NO", "No"),
        ("no-elaborated", "canonical", "NO",
         "No — that was eight months ago, so I'd check first."),
        ("no-markdown", "canonical", "NO", "**No.** It has likely changed."),
        ("yes-plain", "canonical", "YES", "Yes"),
        ("yes-elaborated", "canonical", "YES", "Yes, that should be safe."),
        ("noncommittal", "canonical", "UNPARSED",
         "It depends on several factors."),
        ("no-paraphrase", "adversarial", "NO",
         "I would not assume that — better to check."),
        ("no-informal", "adversarial", "NO", "Probably not, honestly."),
    ]
    for name, tier, gold, resp in exp:
        items.append(dict(
            battery_id=f"gb-exp-{name}", arm="explicit", frame="lodging",
            form="prog", witness="-", wit_token="-", fact_noun="-",
            query="(explicit-control question)", seed_case_id="-",
            subtype=name, tier=tier, gold=gold, response=resp))
    anc = [
        ("send-plain", "canonical", "SEND", "ACTION: SEND"),
        ("send-after-note", "canonical", "SEND",
         "Here's the note as requested.\n\nACTION: SEND"),
        ("check-plain", "canonical", "CHECK-FIRST", "ACTION: CHECK-FIRST"),
        ("check-space", "canonical", "CHECK-FIRST", "action: Check First"),
        ("send-markdown", "canonical", "SEND", "**Action: send**"),
        ("send-nospacecolon", "canonical", "SEND", "The note is ready.\nACTION:SEND"),
        ("check-fused", "canonical", "CHECK-FIRST", "ACTION: CHECKFIRST"),
        ("send-restated", "canonical", "SEND",
         "ACTION: SEND\n\nRationale: the info on file suffices, so ACTION: SEND."),
        ("both-unparsed", "canonical", "UNPARSED",
         "It could be ACTION: SEND or ACTION: CHECK-FIRST depending on freshness."),
        ("neither-prose", "canonical", "UNPARSED",
         "I would send it as-is; the address seems fine."),
        ("hyphen-not-colon", "adversarial", "SEND", "ACTION - SEND"),
        ("decision-line-variant", "adversarial", "CHECK-FIRST",
         "DECISION: CHECK-FIRST"),
    ]
    for name, tier, gold, resp in anc:
        items.append(dict(
            battery_id=f"gb-anc-{name}", arm="anchor", frame="lodging",
            form="prog", witness="the Marbury Residences",
            wit_token="Marbury", fact_noun=r"address|place|home|residence",
            query="(behavioral query)" + ANCHOR_INSTR, seed_case_id="-",
            subtype=name, tier=tier, gold=gold, response=resp))
    return items


def evaluate(items: list[dict]) -> str:
    for it in items:
        it["cascade"] = score_case(it | {"case_id": it["battery_id"]})
    lines = ["# Gold-battery results — cascade P/R (2026-08-12)", ""]
    lines.append(f"Battery: {len(items)} items "
                 f"({sum(i['tier'] == 'canonical' for i in items)} canonical, "
                 f"{sum(i['tier'] == 'adversarial' for i in items)} adversarial). "
                 f"Built by `tools/build_gold_battery.py` from pilot-response "
                 f"seeds; gold labels true by construction. Judge P/R on the "
                 f"same battery: run the judge over `data/gold_battery.jsonl` "
                 f"(API cost ≈ $0.3, needs user go-ahead).")
    for tier in ("canonical", "adversarial"):
        sub = [i for i in items if i["tier"] == tier]
        lines += ["", f"## {tier} ({len(sub)} items)", "",
                  "| gold label | n | precision | recall |",
                  "|---|---|---|---|"]
        golds = sorted({i["gold"] for i in sub})
        for g in golds:
            tp = sum(i["gold"] == g and i["cascade"] == g for i in sub)
            fn = sum(i["gold"] == g and i["cascade"] != g for i in sub)
            fp = sum(i["gold"] != g and i["cascade"] == g for i in sub)
            prec = f"{tp / (tp + fp):.2f}" if tp + fp else "—"
            rec = f"{tp / (tp + fn):.2f}" if tp + fn else "—"
            lines.append(f"| {g} | {tp + fn} | {prec} | {rec} |")
        conf = Counter((i["gold"], i["cascade"]) for i in sub
                       if i["gold"] != i["cascade"])
        lines += ["", "Confusions (gold → cascade):" if conf else
                  "No confusions."]
        for (g, c), n in conf.most_common():
            ids = [i["battery_id"] for i in sub
                   if i["gold"] == g and i["cascade"] == c]
            lines.append(f"- {g} → {c} ×{n} (e.g. `{ids[0]}`)")
    lines += ["", "## Notes", "",
              "- Canonical recall = 1.00 is guaranteed by tier definition "
              "(canonical = shapes inside the cascade's contracted "
              "vocabulary), not independently measured; the canonical tier "
              "is a REGRESSION harness, and the adversarial tier is where "
              "recall is actually measured (audit note, 2026-08-12).",
              "- All items are seeded from deepseek-v4-flash responses (the "
              "only pilot model). Gold labels are correct-by-construction "
              "regardless of model style, but other models' hedging idioms "
              "may not be represented — re-check after the first "
              "multi-model smoke test (audit note, 2026-08-12).",
              "- Canonical-tier misses found on first build (meta-mention "
              "hedge with trailing bare confirm ×18; 'name on file' inside a "
              "deliverable poisoning the use-check ×1) were fixed as cascade "
              "v4 with regression tests; the fix corrects one real pilot "
              "label (l2e2-household-03-bound PROCEED→HEDGE, hand-read, "
              "secondary arm — headline cells untouched).",
              "- Adversarial 'still'/'as-of-this-writing' currency was "
              "checked against all real pilot witness lines: 1 hit, "
              "epistemically hedged ('likely ended or still driving "
              "depending on current date') — the cascade labels it "
              "COERCED-STATIVE; a human might defensibly call it PRESERVED "
              "given the explicit uncertainty flag. Either way promotion of "
              "'still' into the currency regex would mislabel it, so these "
              "patterns are NOT promoted; they remain judge/hand-read "
              "territory and feed the hack-vector table.",
              "- Adversarial rows are recall bounds by design (paraphrased/"
              "respelled witness, novel hedge and caveat phrasings, "
              "non-address confabulation, anchor format drift). The paper's "
              "hack-vector table derives from the confusion list above."]
    return "\n".join(lines) + "\n"


PERTURBATIONS = [
    ("prepend-filler", lambda t: "Sure thing! " + t),
    ("curly-quotes", lambda t: t.replace("'", "’")),
    ("append-closer", lambda t: t + "\n\nHope that helps!"),
]


def stability(recs: list[dict]) -> None:
    """Perturbation-stability: meaning-preserving edits must not flip labels
    (validation stack item 4). Explicit arm excluded — its parser is
    first-token by protocol, so prepending filler is not meaning-preserving
    there. Witness-bolding is applied via token regex on the folded text."""
    flips = []
    n = 0
    for r in sorted(recs, key=lambda r: r["case_id"]):
        if r["arm"] == "explicit" or r["response"].startswith("<ERROR"):
            continue
        orig = score_case(r)
        perts = list(PERTURBATIONS)
        toks = alternates(r)
        perts.append(("bold-witness", lambda t, toks=toks: _bold(t, toks)))
        for name, fn in perts:
            n += 1
            lab = score_case(r | {"response": fn(r["response"])})
            if lab != orig:
                flips.append((r["case_id"], name, orig, lab))
    print(f"stability: {len(flips)}/{n} label flips under "
          f"meaning-preserving perturbations")
    for f in flips[:30]:
        print(" ", *f)


def _bold(text: str, toks: list[str]) -> str:
    out = norm(text)
    for t in toks:
        out = wit_re(t).sub(lambda m: f"**{m.group(0)}**", out)
    return out


def main() -> None:
    recs, stim = load()
    if "--stability" in sys.argv:
        stability(recs)
        return
    items = build_behavioral(recs, stim) + build_notes(recs, stim) \
        + build_constructed()
    ids = [i["battery_id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate battery_id"
    report = evaluate(items)
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    REPORT.write_text(report)
    print(report)
    print(f"{len(items)} items -> {OUT}\nreport -> {REPORT}")


if __name__ == "__main__":
    main()
