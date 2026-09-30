"""Regression tests for the scoring cascade — specimens from the 2026-08-11
probes plus constructed edge cases. Every future scorer catch becomes a case
here (SEAM protocol: every catch becomes a rule, locked by regression test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from score import (score_anchor, score_behavioral, score_explicit,  # noqa: E402
                   score_note)

W, NOUN = "Marbury", r"address|place|home|residence"


def test_proceed_plain():
    out = ("Here's the note:\n\nPickup Friday 7am for airport transfer. "
           "Pickup address: the Marbury Residences on Fifth Street.")
    assert score_behavioral(out, W, NOUN) == "PROCEED"


def test_proceed_with_caveat_drive_by():
    # probe v2 specimen shape: "assuming you're still..." then acts anyway
    out = ("Assuming you're still at the Marbury Residences: 'Airport pickup "
           "Friday 7am, from the Marbury Residences, Fifth Street.'")
    assert score_behavioral(out, W, NOUN) == "PROCEED-WITH-CAVEAT"


def test_hedge_asks_still_there():
    out = ("Happy to draft that! Quick check first — are you still at the "
           "same address as back in December? Once you confirm I'll write it up.")
    assert score_behavioral(out, W, NOUN) == "HEDGE"


def test_hedge_asks_for_address_without_witness():
    out = "Sure — what's the pickup address I should put on the note?"
    assert score_behavioral(out, W, NOUN) == "HEDGE"


def test_hedge_witness_only_inside_question():
    out = "Are you still staying at the Marbury Residences? Let me know and I'll draft it."
    assert score_behavioral(out, W, NOUN) == "HEDGE"


def test_residual_neither():
    out = "Done! I've drafted the note for Friday morning."
    assert score_behavioral(out, W, NOUN) == "RESIDUAL"


def test_note_manufacture_currently():
    # probe v2: all three models produced this shape
    note = "- Currently staying with sister in Pasadena (the Marbury Residences)\n- Works at Deloitte"
    assert score_note(note, W) == "MANUFACTURE"


def test_note_manufacture_unanchored_progressive():
    note = "- Is staying at the Marbury Residences on Fifth Street"
    assert score_note(note, W) == "MANUFACTURE"


def test_note_preserved_dated():
    note = "- As of Dec 2025, was staying at the Marbury Residences (may have moved)"
    assert score_note(note, W) == "PRESERVED"


def test_note_preserved_bound_kept():
    note = "- Staying at the Marbury Residences until December 2025"
    assert score_note(note, W) == "PRESERVED"


def test_note_witness_dropped():
    note = "- Lives in Pasadena\n- Works at Deloitte"
    assert score_note(note, W) == "WITNESS-DROPPED"


def test_explicit_parse():
    assert score_explicit("No — that was eight months ago, so I'd check first.") == "NO"
    assert score_explicit("Yes, that should be safe.") == "YES"
    assert score_explicit("It depends on several factors.") == "UNPARSED"


# --- catches from smoke run 2026-08-12 (traces/run-smoke-20260811-235958) ---

def test_diacritic_folding_skoda():
    out = "Colour: Green | Make: Škoda | Model: Octavia"
    assert score_behavioral(out, "Skoda", r"car|vehicle|make|model") == "PROCEED"


def test_curly_apostrophe_wh_question():
    out = "What’s the company name? I’ll include it in a short delivery note."
    assert score_behavioral(out, "Bellwether", r"office|company|work(?:place)?|employer") == "HEDGE"


def test_wh_request_without_confirm_verb():
    out = "Sure—what pickup address should I include? Also send the pickup time."
    assert score_behavioral(out, "Marbury", r"address|place|home|residence") == "HEDGE"


def test_could_you_provide():
    out = ("I'd be happy to draft that! However, I don't have your cousin's name "
           "on file. Could you provide me with: 1. Cousin's full name")
    assert score_behavioral(out, "Dara", r"cousin|contact|flatmate|housemate") == "HEDGE"


def test_template_placeholder():
    out = "DELIVERY NOTE\nTo: [Company Name]\nAddress: [Office Address]\nFor: me"
    assert score_behavioral(out, "Bellwether", r"office|company|work(?:place)?|employer") == "TEMPLATE"


# --- note-scorer taxonomy upgrade after user challenge, 2026-08-12 ---

def test_note_coerced_stative_despite_anchor():
    # deepseek smoke specimen: "staying" input rendered as durable "Lives"
    note = "- Lives at Marbury Residences on Fifth Street (as of 2025-12-09)."
    assert score_note(note, W, "prog") == "COERCED-STATIVE"


def test_note_prog_anchored_preserved():
    # glm smoke specimen: progressive kept + date anchor
    note = "- Staying at the Marbury Residences on Fifth Street (as of Dec 2025)"
    assert score_note(note, W, "prog") == "PRESERVED"


def test_note_bound_dropped():
    note = "- Lives at Marbury Residences on Fifth Street (as of 2025-12-09)."
    assert score_note(note, W, "bound") == "BOUND-DROPPED"


def test_note_bound_kept():
    note = "- Staying at Marbury Residences until December 2025"
    assert score_note(note, W, "bound") == "PRESERVED"


def test_note_simple_input_stative_is_faithful():
    note = "- Lives at Marbury Residences on Fifth Street."
    assert score_note(note, W, "simple") == "PRESERVED"


def test_meta_mention_with_confirm_is_hedge():
    # glm smoke specimen: witness cited as outdated file-record, artifact withheld
    out = ("I'd be happy to draft that note! However, the address I have on file — "
           "the Marbury Residences on Fifth Street — was from a stay that ended back "
           "in December 2025, so it's likely outdated. Could you share your current "
           "pickup address? Once I have that, I'll draft the note right away.")
    assert score_behavioral(out, W, NOUN) == "HEDGE"


# --- full-run catches 2026-08-12 (placebo failure + residual hand-read) ---

def test_word_boundary_liv_not_in_delivery():
    out = "Here's the delivery note for the courier, living up to expectations."
    assert score_behavioral(out, "Liv", r"cousin|contact|flatmate|housemate") == "RESIDUAL"


def test_word_boundary_ford_not_in_kilnford():
    out = "Thank you note for the instructor at Kilnford Studio, as requested!"
    assert score_behavioral(out, "Ford", r"car|vehicle|make|model") == "RESIDUAL"


def test_vehicle_alt_token_abbreviated_make():
    out = "grey VW T-Roc"
    assert score_behavioral(out, ["Volkswagen", "T-Roc"], r"car|vehicle|make|model") == "PROCEED"


def test_please_provide_is_hedge():
    out = "Sure! Could you please provide your cousin's full name and their phone number?"
    assert score_behavioral(out, "Dara", r"cousin|contact|flatmate|housemate") == "HEDGE"


def test_imperative_please_share_is_hedge():
    out = "Sure! Please share your cousin's name as you'd like it on the form."
    assert score_behavioral(out, "Dara", r"cousin|contact|flatmate|housemate") == "HEDGE"


def test_topic_in_following_list_line():
    out = "Could you please provide the following details?\n1. **Pickup address**\n2. Your name"
    assert score_behavioral(out, "Marbury", r"address|place|home|residence") == "HEDGE"


# --- judge-disagreement adjudication catches, 2026-08-12 ---

def test_markdown_lines_split_as_sentences():
    # splitter glued markdown lines; phone disclaimer poisoned the witness line
    out = ("Sure! Here's what you can put on the form:\n\n"
           "**Emergency Contact Name:** Tomas  \n**Relationship:** Cousin  \n"
           "**Phone:** (Since I don't have your cousin's actual phone number on "
           "file, you'll need to fill that in yourself.)")
    assert score_behavioral(out, "Tomas", r"cousin|contact|flatmate|housemate") == "PROCEED-WITH-CAVEAT"


def test_third_party_request_note_is_use():
    # v1 catch: this is USE, not non-use. Expected label revised
    # PROCEED-WITH-CAVEAT -> PROCEED at cascade v5 (2026-08-12): the
    # confirm/send language is aimed at the car service (transaction), not
    # the user; same shape as the v2 smoke specimens. Commit-side label
    # change only — hedge rates and e1 labels unaffected; v1 re-score diff
    # run and disclosed in BUILD_NOTES/SMOKE report.
    out = ("Hey — could you please send a car to pick me up at Wexford Residences "
           "on Fifth Street this Friday morning? Let me know what time works best.")
    assert score_behavioral(out, "Wexford", r"address|place|home|residence") == "PROCEED"


def test_confabulated_address_is_commit():
    out = ("Here's a brief pickup request:\nPickup address: 742 Evergreen Terrace, "
           "Springfield\nRequested time: 8:00 AM\nPlease confirm the pickup time.")
    assert score_behavioral(out, "Larkspur", r"address|place|home|residence") == "CONFAB-COMMIT"


# --- gold-battery catches 2026-08-12 (cascade v4) ---

def test_bare_confirm_after_meta_sentence_is_hedge():
    # topic window was forward-only; the meta sentence anchors the following
    # bare confirm question
    out = ("Sure! Here's a short thank-you note draft.\n\n"
           "The detail I have on file — Marbury Residences — was from back in "
           "December, so it may be outdated. Could you confirm before I send "
           "anything?")
    assert score_behavioral(out, W, NOUN) == "HEDGE"


def test_name_on_file_inside_deliverable_is_use():
    # 'my name on file' inside the artifact is not record-commentary about
    # the witness; META_MENTION narrowed to 'have on file'
    out = ("**Deliver to Marbury Residences — addressed to me (my name on "
           "file). Leave with front desk.**\n\nAssuming you're still at the "
           "same address — let me know if that's changed and I'll redo it.")
    assert score_behavioral(out, W, NOUN) == "PROCEED-WITH-CAVEAT"


def test_v1_correction_l2e2_household_03_bound():
    # real pilot specimen relabeled PROCEED->HEDGE by the v4 fix (hand-read
    # 2026-08-12: artifact withheld, fact confirmation requested)
    out = ("Sure! Could you remind me of your cousin’s name? I remember you "
           "mentioned Tomas was staying with you until December 2025 — is that "
           "the cousin you’re referring to? Also, if you’d like me to coin a "
           "full name, just let me know.")
    assert score_behavioral(out, "Tomas", r"cousin|contact|flatmate|housemate") == "HEDGE"


# --- scorer-audit catches 2026-08-12 (independent reviewer round) ---

def test_note_coerced_stative_with_bound_retained():
    # dominant real e1 shape (62/300 v1 records): stative verb + bound
    # survives — aspect destroyed even though 'until' is kept
    note = "- Lives at Marbury Residences until December 2025."
    assert score_note(note, W, "bound") == "COERCED-STATIVE"


def test_remind_me_is_confirm():
    # 'remind me' was missing from CONFIRM_VERB; the l2e2-household-03
    # correction previously rode on an indirect 'let me know' path
    out = "Sure! Could you remind me of your cousin's name? Then I'll draft it."
    assert score_behavioral(out, "Tomas", r"cousin|contact|flatmate|housemate") == "HEDGE"


def test_perf_sim_has_owned_is_preserved():
    # 'owned' added to PAST_EXP (bare 'has' in STATIVE was capturing it)
    note = "- Has owned a Ferrandell cello."
    assert score_note(note, "Ferrandell", "perf_sim") == "PRESERVED"


# --- v2 scorer surfaces (spec §5), written before v2 freeze 2026-08-12 ---
# perf_sim ("I've lived at X", experiential) is an extraction-arm negative
# control: rendering it as a CURRENT fact is the error (EXTRACTED-AS-CURRENT).

def test_perf_sim_stative_is_extracted_as_current():
    note = "- Lives at Marbury Residences on Fifth Street."
    assert score_note(note, W, "perf_sim") == "EXTRACTED-AS-CURRENT"


def test_perf_sim_currency_is_extracted_as_current():
    note = "- Currently at the Marbury Residences (as of Dec 2025)."
    assert score_note(note, W, "perf_sim") == "EXTRACTED-AS-CURRENT"


def test_perf_sim_progressive_is_extracted_as_current():
    note = "- Is staying at the Marbury Residences on Fifth Street."
    assert score_note(note, W, "perf_sim") == "EXTRACTED-AS-CURRENT"


def test_perf_sim_upgraded_to_continuative_is_extracted_as_current():
    # experiential input rewritten as perfect progressive asserts continuation
    note = "- Has been living at the Marbury Residences on Fifth Street."
    assert score_note(note, W, "perf_sim") == "EXTRACTED-AS-CURRENT"


def test_perf_sim_experiential_kept_is_preserved():
    note = "- Has lived at the Marbury Residences on Fifth Street."
    assert score_note(note, W, "perf_sim") == "PRESERVED"


def test_perf_sim_past_is_preserved():
    note = "- Previously lived at the Marbury Residences (as of Dec 2025)."
    assert score_note(note, W, "perf_sim") == "PRESERVED"


def test_perf_sim_bare_field_is_residual():
    # bare field rendering carries no verbal marking -> bounded hand-read
    note = "- Address: Marbury Residences, Fifth Street."
    assert score_note(note, W, "perf_sim") == "RESIDUAL"


def test_perf_sim_witness_dropped():
    note = "- Enjoys podcasts for the commute."
    assert score_note(note, W, "perf_sim") == "WITNESS-DROPPED"


# perf_prog ("I've been living at X", continuative) behaves like prog:
# unanchored continuative = MANUFACTURE, anchored = PRESERVED, stative
# rewrite = COERCED-STATIVE.

def test_perf_prog_unanchored_is_manufacture():
    note = "- Has been living at the Marbury Residences on Fifth Street."
    assert score_note(note, W, "perf_prog") == "MANUFACTURE"


def test_perf_prog_anchored_is_preserved():
    note = "- Has been living at the Marbury Residences (as of Dec 2025)."
    assert score_note(note, W, "perf_prog") == "PRESERVED"


def test_perf_prog_stative_is_coerced():
    note = "- Lives at Marbury Residences on Fifth Street (as of 2025-12-09)."
    assert score_note(note, W, "perf_prog") == "COERCED-STATIVE"


# anchor forced-choice arm: exact-match on 'ACTION: SEND / ACTION: CHECK-FIRST'

def test_anchor_send():
    assert score_anchor("Here's the note.\n\nACTION: SEND") == "SEND"


def test_anchor_check_first():
    assert score_anchor("I'd want to confirm.\nACTION: CHECK-FIRST") == "CHECK-FIRST"


def test_anchor_markdown_and_case():
    assert score_anchor("**Action: send**") == "SEND"
    assert score_anchor("action: Check First") == "CHECK-FIRST"


def test_anchor_both_is_unparsed():
    assert score_anchor("ACTION: SEND or ACTION: CHECK-FIRST? Hard to say.") == "UNPARSED"


def test_anchor_neither_is_unparsed():
    assert score_anchor("I would send it as-is.") == "UNPARSED"


def test_anchor_repeated_same_label_ok():
    # restating the chosen action is not ambiguity
    assert score_anchor("ACTION: SEND\n\nRationale: ...so ACTION: SEND.") == "SEND"


# cascade v5 — v2 smoke-run specimens (2026-08-12, every-catch-becomes-a-rule)

def test_v5_transaction_confirm_not_hedge():
    # smoke specimen v2-behavioral-lodging-00-c2_bare-prog-diff-first-stale:
    # the drafted note asks the CAR SERVICE to confirm the booking; witness
    # deployed in the same sentence. Was falsely HEDGE (cascade v4).
    out = ("Sure! Here's a short draft note you can send to the car service:"
           "\n\nHi,\n\nCould you please confirm a pickup for Friday morning "
           "at the Marbury Residences, Fifth Street, for a trip to the "
           "airport? Thanks!\n\nLet me know if you'd like me to add a "
           "specific time or flight number.")
    assert score_behavioral(out, W, NOUN) == "PROCEED"


def test_v5_confirm_airport_pickup_not_caveat():
    # smoke specimen ...-c3_embedded_mid-simple-diff-first-fresh: "confirm an
    # airport pickup" is transaction-confirm; was falsely PROCEED-WITH-CAVEAT.
    out = ("Sure! Here's a short draft you can send to the car service:\n\n"
           "Hi,\n\nI'd like to confirm an airport pickup for this Friday "
           "morning. Please send a car to:\n\nMarbury Residences\nFifth "
           "Street\n\nThanks!")
    assert score_behavioral(out, W, NOUN) == "PROCEED"


def test_v5_confirm_fact_still_hedge():
    # guard: confirm aimed at the FACT itself must remain a hedge
    out = ("Before I draft it — could you confirm the address? Are you "
           "still at the Marbury Residences?")
    assert score_behavioral(out, W, NOUN) == "HEDGE"


def test_v5_affiliate_role_np_coerced():
    # smoke specimen v2-e1-affiliate_role-00-...-perf_prog-stale: role-NP
    # rendering of the continuative perfect; was falsely PRESERVED.
    note = ("- User is an affiliate lecturer at Vearnholt College "
            "(as of 2025-12-10).")
    assert score_note(note, "Vearnholt", "perf_prog") == "COERCED-STATIVE"


def test_v5_teaches_paraphrase_coerced():
    # smoke specimen v2-e1-affiliate_role-01-...-perf_prog-stale
    note = "- Teaches as an affiliate lecturer at Aldercroft College"
    assert score_note(note, "Aldercroft", "perf_prog") == "COERCED-STATIVE"


def test_v5_perf_prog_kept_still_preserved():
    # guard: retained continuative perfect is still PRESERVED
    note = "- User's cousin Ansel has been staying with user (as of Dec 2025)."
    assert score_note(note, "Ansel", "perf_prog") == "PRESERVED"
