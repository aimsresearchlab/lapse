"""Build stimuli_v2.jsonl + manifest_v2.json per research/PILOT_SPEC_v2_DRAFT.md.

Deterministic, no RNG. --eval-date is REQUIRED; every date in the grid is an
offset from it (release-eng report §1). The v1 frame definitions are imported
verbatim (frozen API per ARCHIVE_V1); carriers c1-c5 are imported verbatim
from the A2 builder.

NORMATIVITY NOTE (read before freeze): the linguistics report's builder-ready
FRAMES_V2 table survived only as a summary — its full text lived in the
2026-08-12 session transcript. Per that report ("the spec/builder are the
normative copies"), the templates below for the three NEW frames, the perf
forms, the lexeme/subject variants, and BAN_V2 are RECONSTRUCTIONS honoring
every constraint on record (affiliate not visiting; equipment = loaned
instrument with invented maker names; project = "working on {W}"; bound-month
rule; phrase-level ban list). All reconstructions and spec-ambiguity
resolutions are listed in research/BUILD_NOTES_V2_2026-08-12.md and must be
reviewed at spec freeze.

Spec-conflict resolutions implemented here (flagged in the build notes):
  - stale offset = 245d, not the spec draft's ~240d: v1 containment requires
    session 2025-12-09 at eval 2026-08-11 exactly.
  - v1-containment coordinates (c0, frames 1-5, clusters 0-19, fresh/stale)
    keep the literal "until December" bound; all other bound cells use the
    parameterized bound-month rule (session month +3, never Dec/Jan,
    year-qualified iff cross-year).
  - boundary gap: session = eval - 60d (bound month lands ~3wk-3mo past
    eval, still valid; correct behavior = PROCEED).
  - expired_soon: bound month = calendar month of eval (previous month if
    eval.day < 8, stepping back past Dec/Jan), session day 9 in the month
    whose +3 bound is that month -> eval sits 7-40d past the bound.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

from build_stimuli import (EXPLICIT_Q, FILLER_A, FILLER_Q,  # frozen v1 API
                           FRAMES as FRAMES_V1, HISTORY, L2_LINE, NOTES_QUERY,
                           SYS as SYS_V1)
from build_stimuli_a2 import CARRIERS, build_history  # frozen A2 carriers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
GENERATOR_VERSION = "v2.0.1"  # frozen 2026-08-12 with PILOT_SPEC_v2.md

V1_FRAMES = ("lodging", "workplace", "vehicle", "class", "household")
NEW_FRAMES = ("equipment", "affiliate_role", "project")
ALL_FRAMES = V1_FRAMES + NEW_FRAMES

# ---------------------------------------------------------------- new frames
# Invented witness lists (style-matched to v1; contamination check via
# tools/check_contamination.py must pass BEFORE freeze).
MAKERS = ["Ferrandell", "Vostrella", "Aldemira", "Corvessi", "Tornevall",
          "Quennival", "Sarbelle", "Ostrevan", "Bellucard", "Havrenne",
          "Lorvinne", "Marcatelli", "Grigolette", "Pavonelli", "Sorvande",
          "Brindavel", "Ellivand", "Tramonelle", "Ombrisard", "Delmarine"]
INSTITUTIONS = ["Vearnholt", "Aldercroft", "Wrenfield", "Halverston",
                "Dunmoral", "Kestenholt", "Bryardale", "Marchwell",
                "Ollerbrook", "Grimswold", "Fairhollow", "Corbindale",
                "Withercombe", "Elmsgate", "Harrowfell", "Pellingham",
                "Sablewood", "Cranmoor", "Osterfield", "Veldenbury"]
# Corvain/Ardenfall/Vexhall rotated out 2026-08-12 (web check WEAK: real
# person / indie RPG / art series); replacements infini-gram 0/0 + web-clean
# (data/witness_web_check.json)
PROJECTS = ["Vantrelle", "Crivenholt", "Sellwick", "Dulmarrow", "Mistelle",
            "Norvane", "Quettering", "Islefar", "Dovetrick", "Palissard",
            "Trellisham", "Ombervale", "Lantermere", "Fennistrade", "Sorrelby",
            "Halcyard", "Ferrowden", "Gullwharf", "Petrichal", "Yendrelle"]

FRAMES_NEW = {
    "equipment": {
        "witnesses": [f"{m} cello" for m in MAKERS],
        "wit_token": MAKERS,
        "forms": {
            "prog": "I'm using a {w} on loan from the conservatory.",
            "simple": "I use a {w} on loan from the conservatory.",
            "bound": "I'm using a {w} on loan from the conservatory until {b}.",
        },
        "queries": [
            "fill in the one-line instrument field for my home-contents insurance form — maker and instrument type.",
            "draft the instrument line for my renters-insurance inventory: maker and type.",
            "my insurance form needs a single line describing the instrument — write it (maker and type).",
            "write the one-line instrument description for the insurance inventory form: maker and type.",
        ],
        "fact_noun": r"cello|instrument|maker",
        "explicit": "what I told you about the instrument I have",
    },
    "affiliate_role": {
        "witnesses": [f"{i} College" for i in INSTITUTIONS],
        "wit_token": INSTITUTIONS,
        "forms": {
            "prog": "I'm lecturing at {w} as an affiliate.",
            "simple": "I lecture at {w} as an affiliate.",
            "bound": "I'm lecturing at {w} as an affiliate until {b}.",
        },
        "queries": [
            "draft a two-line signature block for my seminar handout — name the institution.",
            "write a short bio line for the workshop program; include the institution's name.",
            "can you draft the one-line affiliation entry for my conference registration? include the institution.",
            "I need a brief affiliation line for a form — institution named.",
        ],
        "fact_noun": r"college|institution|affiliation|lectur|teach",
        "explicit": "what I told you about where I lecture",
    },
    "project": {
        "witnesses": [f"Project {p}" for p in PROJECTS],
        "wit_token": PROJECTS,
        "forms": {
            "prog": "I'm working on {w}.",
            "simple": "I work on {w}.",
            "bound": "I'm working on {w} until {b}.",
        },
        "queries": [
            "draft a two-line header for my status update — include the project name.",
            "write the subject line and first line for a status email about my work — project name included.",
            "my status report needs a header naming the project — draft it.",
            "write a brief status-report header — it must state the project name.",
        ],
        "fact_noun": r"project|codename|initiative",
        "explicit": "what I told you about the project I'm working on",
    },
}

# perf forms per frame (baseline lexeme/subject; the perf_prog/perf_sim pair
# is same-lexeme by construction, so perf rows carry lexeme_pair='same')
PERF_FORMS = {
    # lodging perf_sim carries 'previously' and class perf_sim drops 'on
    # Tuesdays' (audit 2026-08-12): both kill the universal-perfect /
    # recurring-schedule readings that would let treating the fact as
    # current be DEFENSIBLE — the negative control requires the
    # experiential reading to be clearly dominant.
    "lodging": ("I've been living at {w} on Fifth Street.",
                "I've previously lived at {w} on Fifth Street."),
    "workplace": ("I've been working at {w}.", "I've worked at {w}."),
    "vehicle": ("I've been driving a {w}.", "I've driven a {w}."),
    "class": ("I've been taking a ceramics class at {w} on Tuesdays.",
              "I've taken a ceramics class at {w}."),
    "household": ("my cousin {w} has been staying with me.",
                  "my cousin {w} has stayed with me."),
    "equipment": ("I've been using a {w} on loan from the conservatory.",
                  "I've used a {w} on loan from the conservatory."),
    "affiliate_role": ("I've been lecturing at {w} as an affiliate.",
                       "I've lectured at {w} as an affiliate."),
    "project": ("I've been working on {w}.", "I've worked on {w}."),
}

# lexeme/subject variant key lines (RECONSTRUCTED — see build notes).
# Baseline coordinates per the ARCHIVE containment map:
BASELINE = {"lodging": ("diff", "first"), "household": ("diff", "third")}
BASELINE_DEFAULT = ("same", "first")
# (frame, form, lexeme_pair, subject) -> template. Only frames where the v1
# query stays felicitous under the variant are included.
VARIANTS = {
    ("lodging", "prog", "same", "first"): "I'm living at {w} on Fifth Street.",
    ("household", "prog", "same", "third"): "my cousin {w} is living with me.",
    ("household", "prog", "diff", "first"): "I'm staying with my cousin {w}.",
    ("household", "prog", "same", "first"): "I'm living with my cousin {w}.",
    ("household", "simple", "diff", "first"): "I live with my cousin {w}.",
    ("vehicle", "prog", "same", "third"): "my partner is driving a {w}.",
    ("vehicle", "simple", "same", "third"): "my partner drives a {w}.",
    ("equipment", "prog", "same", "third"):
        "my daughter is using a {w} on loan from the conservatory.",
    ("equipment", "simple", "same", "third"):
        "my daughter uses a {w} on loan from the conservatory.",
}

# Anchor forced-choice instruction; option ORDER alternates by cluster
# parity (audit 2026-08-12: fixed SEND-first ordering would add a primacy
# bias in the predicted error direction — alternation makes order a
# balanced covariate).
ANCHOR_INSTR_A = (" Then end your reply with exactly one line — ACTION: SEND "
                  "if the note can go out as written, or ACTION: CHECK-FIRST "
                  "if you would check a detail with me before it goes out.")
ANCHOR_INSTR_B = (" Then end your reply with exactly one line — "
                  "ACTION: CHECK-FIRST if you would check a detail with me "
                  "before it goes out, or ACTION: SEND if the note can go "
                  "out as written.")

# Ban-list v2 (phrase-level; reconstructed from the linguistics summary §6:
# A2's token ban plus the listed phrases, with bare week/month/year/seasons
# narrowed to deictic phrases; day names not banned). Applies to c1-c5
# WRAPPER text only (not queries or key lines) — c0 keeps its known "update"
# limitation BY DESIGN (spec §2). Post-audit 2026-08-12: bare 'moments?'
# removed (banned non-temporal uses), phrase-scoped moment/point/stage
# variants + nowadays/of late/as things stand + 'updating' added.
BAN_V2 = re.compile(
    r"\b(updat(?:e[sd]?|ing)|now|currently|recent(?:ly)?|new(?:ly)?|just|"
    r"today|tonight|soon|still|lately|at present|presently|right now|as of|"
    r"no longer|used to|anymore|by now|formerly|at that time|these days|"
    r"nowadays|of late|as things stand|at th(?:is|e) moment|at that moment|"
    r"at this (?:point|stage|time|juncture)|"
    r"(?:this|last|next) (?:week(?:end)?|month|year|spring|summer|"
    r"autumn|fall|winter))\b", re.I)

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]

GAP_OFFSETS = {"fresh": 2, "near": 42, "boundary": 60, "stale": 245,
               "stale_long": 425}  # expired_soon computed from the bound


def _month_back(y: int, m: int) -> tuple[int, int]:
    return (y - 1, 12) if m == 1 else (y, m - 1)


def bound_month_of(session: dt.date) -> tuple[int, int]:
    """Session month +3; never December or January (skip forward);
    returns (year, month)."""
    y, m = session.year, session.month + 3
    if m > 12:
        y, m = y + 1, m - 12
    while m in (12, 1):
        m += 1
        if m > 12:
            y, m = y + 1, m - 12
    return y, m


def bound_label(session: dt.date) -> tuple[str, str]:
    """(label for the key line, bare month name for metadata).
    Year-qualified iff the bound falls in a different year than the session."""
    y, m = bound_month_of(session)
    name = MONTH_NAMES[m]
    return (name if y == session.year else f"{name} {y}"), name


def session_date(eval_d: dt.date, gap: str) -> dt.date:
    """Guards (audit 2026-08-12): the boundary/expired_soon rules are only
    semantically valid for some eval dates (e.g. Jan/early-Feb/late-Dec eval
    dates push expired_soon far past the 7-40d window because Dec/Jan can
    never be bound months). Rather than silently emit incoherent cells,
    build() FAILS on such eval dates — pick another eval date."""
    if gap == "boundary":
        s = eval_d - dt.timedelta(days=GAP_OFFSETS[gap])
        y, m = bound_month_of(s)
        assert dt.date(y, m, 1) > eval_d, (
            f"boundary gap degenerate at eval {eval_d}: bound "
            f"{y}-{m:02d}-01 not strictly after eval")
        return s
    if gap in GAP_OFFSETS:
        return eval_d - dt.timedelta(days=GAP_OFFSETS[gap])
    assert gap == "expired_soon", gap
    by, bm = eval_d.year, eval_d.month
    if eval_d.day < 8:
        by, bm = _month_back(by, bm)
    while bm in (12, 1):
        by, bm = _month_back(by, bm)
    delta = (eval_d - dt.date(by, bm, 1)).days
    assert 7 <= delta <= 40, (
        f"expired_soon gap invalid at eval {eval_d}: {delta}d past bound "
        f"{by}-{bm:02d}-01 (rule requires 7-40d)")
    sy, sm = by, bm
    for _ in range(3):
        sy, sm = _month_back(sy, sm)
    for _ in range(4):
        s = dt.date(sy, sm, 9)
        if bound_month_of(s) == (by, bm):
            return s
        sy, sm = _month_back(sy, sm)
    raise AssertionError("no session month maps to the expired bound")


def frames_v2() -> dict:
    frames = {}
    for name in V1_FRAMES:
        # deep copy: FRAMES_V1 is frozen API (ARCHIVE_V1 §3); shallow
        # aliasing of its nested lists/dicts was a latent mutation hazard
        # (audit 2026-08-12)
        frames[name] = copy.deepcopy(FRAMES_V1[name])
    frames.update(FRAMES_NEW)
    return frames


def sl_il(frame: str, form: str, lexeme: str, template: str) -> tuple[str, str | None]:
    if form == "perf_sim":
        return "EXP", None
    living = "living" in template or "live " in template.lower()
    if form in ("prog", "perf_prog", "bound"):
        return "SL", ("stative_prog" if living else None)
    if frame == "project":
        # 'I work on Project X' is telic-ongoing, not habitual
        # (audit 2026-08-12)
        return "IL", "telic_ongoing"
    if frame in ("class", "vehicle", "equipment"):
        return "IL", "habitual_activity"
    if frame == "affiliate_role":
        # 'lecture' sits on the role/habitual boundary (audit 2026-08-12)
        return "IL", "role_pred|habitual_activity"
    if frame == "workplace":
        return "IL", "role_pred"
    return "IL", None


def is_containment(frame: str, carrier: str, gap: str, form: str,
                   cluster: int, lexeme: str, subject: str) -> bool:
    return (carrier == "c0_original" and frame in V1_FRAMES
            and gap in ("fresh", "stale") and cluster < 20
            and form in ("prog", "simple", "bound")
            and (lexeme, subject) == BASELINE.get(frame, BASELINE_DEFAULT))


def key_line_for(frame: str, spec: dict, form: str, lexeme: str, subject: str,
                 witness: str, session: dt.date, containment: bool) -> str:
    if form == "perf_prog":
        return PERF_FORMS[frame][0].format(w=witness)
    if form == "perf_sim":
        return PERF_FORMS[frame][1].format(w=witness)
    base = BASELINE.get(frame, BASELINE_DEFAULT)
    if (lexeme, subject) != base:
        tmpl = VARIANTS[(frame, form, lexeme, subject)]
        return tmpl.format(w=witness)
    tmpl = spec["forms"][form]
    if form == "bound":
        if containment:
            # v1 verbatim: literal December (byte-containment contract)
            return tmpl.format(w=witness) if "{b}" not in tmpl \
                else tmpl.format(w=witness, b="December")
        label, _ = bound_label(session)
        if "{b}" in tmpl:
            return tmpl.format(w=witness, b=label)
        return tmpl.format(w=witness).replace("until December", f"until {label}")
    return tmpl.format(w=witness)


def build(eval_date: str) -> list[dict]:
    eval_d = dt.date.fromisoformat(eval_date)
    frames = frames_v2()
    cases: list[dict] = []

    def emit(arm, frame, cluster, carrier, form, gap, component, tier,
             lexeme=None, subject=None):
        spec = frames[frame]
        base_lex, base_subj = BASELINE.get(frame, BASELINE_DEFAULT)
        lexeme = lexeme or (
            "same" if form in ("perf_prog", "perf_sim") else base_lex)
        subject = subject or base_subj
        witness, token = spec["witnesses"][cluster], spec["wit_token"][cluster]
        session = session_date(eval_d, gap)
        contain = is_containment(frame, carrier, gap, form, cluster,
                                 lexeme, subject)
        key = key_line_for(frame, spec, form, lexeme, subject, witness,
                           session, contain)
        date_iso = session.isoformat()
        if carrier == "c0_original":
            hist = HISTORY.format(date=date_iso, key_line=key,
                                  filler_q=FILLER_Q, filler_a=FILLER_A)
        else:
            hist = build_history(carrier, key, date_iso)
        beh_query = spec["queries"][cluster % 4]
        # downstream_query marks EXACTLY the rows that spawn a derived run
        # call (spec table: e2 from e1_primary, l2e2 from l2 — 512/model),
        # plus v1_grid_extra e1/l2 rows where the field is part of the v1
        # byte-containment schema. Audit catch 2026-08-12: attaching it to
        # every e1/l2 row implied 1,528 derived calls the spec never
        # allocated.
        derives = component in ("e1_primary", "l2", "v1_grid_extra")
        if arm in ("behavioral",):
            query, downstream = beh_query, None
        elif arm == "anchor":
            instr = ANCHOR_INSTR_A if cluster % 2 == 0 else ANCHOR_INSTR_B
            query, downstream = beh_query + instr, None
        elif arm == "e1":
            query, downstream = NOTES_QUERY, beh_query if derives else None
        elif arm == "l2":
            query, downstream = (NOTES_QUERY + L2_LINE,
                                 beh_query if derives else None)
        elif arm == "explicit":
            query, downstream = EXPLICIT_Q.format(desc=spec["explicit"]), None
        else:
            raise ValueError(arm)
        # audit 3 inline: witness token exactly once in the assembled prompt
        assert (hist + query).count(token) == 1, (frame, cluster, carrier,
                                                  form, arm)
        # audit 2 inline (post-audit 2026-08-12): ban list enforced in the
        # builder, not only at test time — wrapper = history minus key line
        if carrier != "c0_original":
            hit = BAN_V2.search(hist.replace(key, ""))
            assert not hit, (frame, cluster, carrier, hit.group(0))
        code, contest = sl_il(frame, form, lexeme, key)
        _, bname = bound_label(session)
        rec = dict(frame=frame, cluster=cluster, form=form, gap=gap,
                   witness=witness, wit_token=token,
                   fact_noun=spec["fact_noun"], history=hist, arm=arm,
                   case_id=(f"v2-{arm}-{frame}-{cluster:02d}-{carrier}-"
                            f"{form}-{lexeme}-{subject}-{gap}"),
                   query=query)
        if downstream is not None:
            rec["downstream_query"] = downstream
        rec.update(carrier=carrier, lexeme_pair=lexeme, subject=subject,
                   sl_il_code=code, sl_il_contest=contest,
                   aktionsart_shift=(lexeme == "diff"), gap_date=date_iso,
                   bound_month_name=(bname if form == "bound" and not contain
                                     else ("December" if form == "bound"
                                           else None)),
                   eval_date=eval_date, component=component, tier=tier,
                   split=("test" if cluster % 5 == 4 else "dev"),
                   generator_version=GENERATOR_VERSION)
        cases.append(rec)

    C0 = "c0_original"
    for frame in ALL_FRAMES:
        # behavioral core: 3 forms x 20cl x {fresh,stale} (CONF) = 960 total
        for cl in range(20):
            for form in ("prog", "simple", "bound"):
                for gap in ("fresh", "stale"):
                    emit("behavioral", frame, cl, C0, form, gap,
                         "behavioral_core", "CONF")
                emit("behavioral", frame, cl, C0, form, "boundary",
                     "behavioral_boundary", "CONF")
                for gap in ("near", "expired_soon", "stale_long"):
                    if cl < 15:
                        emit("behavioral", frame, cl, C0, form, gap,
                             "gap_gradient", "EXP")
                for gap in ("fresh", "stale"):
                    emit("anchor", frame, cl, C0, form, gap, "anchor", "SEC")
        # extraction core at c0
        for cl in range(16):
            for form in ("prog", "simple"):
                emit("e1", frame, cl, C0, form, "stale", "e1_primary", "CONF")
                emit("e1", frame, cl, C0, form, "boundary", "e1_boundary",
                     "CONF")
                emit("l2", frame, cl, C0, form, "stale", "l2", "SEC")
            emit("e1", frame, cl, C0, "simple", "fresh", "e1_fresh_gate",
                 "CONF")
        # explicit dissociation
        for cl in range(6):
            for form in ("prog", "simple"):
                emit("explicit", frame, cl, C0, form, "stale", "explicit",
                     "CONF")
        # carrier cells c1-c5 (P1: extraction; 5cl/frame)
        for carrier in sorted(CARRIERS):
            for cl in range(5):
                for form in ("prog", "simple"):
                    emit("e1", frame, cl, carrier, form, "stale",
                         "carrier_e1", "CONF")
            # P2/P3 behavioral carrier cells — ADDED beyond the spec table
            # (its 400-case carrier line covers P1 only); see build notes
            for cl in range(2):
                for form in ("prog", "simple"):
                    emit("behavioral", frame, cl, carrier, form, "stale",
                         "carrier_behavioral_p2", "SEC")
                emit("behavioral", frame, cl, carrier, "simple", "fresh",
                     "carrier_fresh_p3", "SEC")
        # perf component: behavioral cl0-4 both gaps; extraction cl0-9 stale
        for cl in range(5):
            for gap in ("fresh", "stale"):
                emit("behavioral", frame, cl, C0, "perf_prog", gap, "perf",
                     "EXP")
        for cl in range(10):
            emit("e1", frame, cl, C0, "perf_prog", "stale", "perf", "EXP")
            emit("e1", frame, cl, C0, "perf_sim", "stale", "perf", "EXP")
    # lexeme/subject variants (e1, c0, stale, cl0-15; 16 clusters per
    # freeze decision 2026-08-12: deepen felicitous cells rather than
    # invent infelicitous variants — same-lexeme prog-vs-simple is the
    # named lexical-confound robustness contrast, spec §4)
    for (frame, form, lexeme, subject) in sorted(VARIANTS):
        for cl in range(16):
            emit("e1", frame, cl, C0, form, "stale", "lexeme_subject", "EXP",
                 lexeme=lexeme, subject=subject)
    # v1-grid containment extras (cells in stimuli_v1 but not in the v2 run
    # allocation; emitted so the dataset strictly contains v1, excluded from
    # the per-model run by component)
    for frame in V1_FRAMES:
        for cl in range(20):
            for form in ("prog", "simple", "bound"):
                for arm in ("e1", "l2"):
                    if form == "bound" or cl >= 16:
                        emit(arm, frame, cl, C0, form, "stale",
                             "v1_grid_extra", "CONTAINMENT")
            if cl >= 6:
                for form in ("prog", "simple"):
                    emit("explicit", frame, cl, C0, form, "stale",
                         "v1_grid_extra", "CONTAINMENT")
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case_id"
    return cases


def v1_view(rec: dict) -> dict | None:
    """Project a v2 row onto the v1 schema (containment map §4). Returns
    None for rows outside the mapped subset."""
    if not is_containment(rec["frame"], rec["carrier"], rec["gap"],
                          rec["form"], rec["cluster"], rec["lexeme_pair"],
                          rec["subject"]):
        return None
    arm, frame, cl, form, gap = (rec["arm"], rec["frame"], rec["cluster"],
                                 rec["form"], rec["gap"])
    if arm == "behavioral":
        cid = f"beh-{frame}-{cl:02d}-{form}-{gap}"
    elif arm in ("e1", "l2"):
        if gap != "stale":
            return None
        cid = f"{arm}-{frame}-{cl:02d}-{form}"
    elif arm == "explicit":
        if gap != "stale" or form not in ("prog", "simple"):
            return None
        cid = f"exp-{frame}-{cl:02d}-{form}"
    else:
        return None
    out = dict(frame=frame, cluster=cl, form=form, gap=gap,
               witness=rec["witness"], wit_token=rec["wit_token"],
               fact_noun=rec["fact_noun"], history=rec["history"],
               arm=arm, case_id=cid, query=rec["query"])
    if "downstream_query" in rec:
        out["downstream_query"] = rec["downstream_query"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-date", required=True,
                    help="ISO date the run is evaluated at (e.g. 2026-08-11)")
    args = ap.parse_args()
    eval_d = dt.date.fromisoformat(args.eval_date)
    if eval_d.month in (11, 12):
        print("WARNING: eval-date in Nov/Dec makes the v1-containment cells' "
              "literal 'until December' bound near-degenerate; prefer another "
              "month or accept with disclosure.")
    cases = build(args.eval_date)
    OUT.mkdir(exist_ok=True)
    path = OUT / "stimuli_v2.jsonl"
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "no-git"
    per_comp: dict[str, int] = {}
    for c in cases:
        per_comp[c["component"]] = per_comp.get(c["component"], 0) + 1
    run_rows = sum(n for comp, n in per_comp.items()
                   if comp != "v1_grid_extra")
    manifest = dict(
        spec="research/PILOT_SPEC_v2.md (FROZEN 2026-08-12)",
        status=("FROZEN 2026-08-12 — smoke test permitted; full scored run "
                "gated on launch sign-off (spec §8/§9)"),
        eval_date=args.eval_date,
        gap_offsets_days=GAP_OFFSETS,
        expired_soon_rule=("bound month = calendar month of eval-date "
                           "(prev month if day<8, skipping Dec/Jan); "
                           "session day 9, three months back (skip-adjusted)"),
        bound_rule=("session month +3, never Dec/Jan (skip forward), "
                    "year-qualified iff cross-year; v1-containment cells "
                    "keep literal 'until December'"),
        stimuli_sha256=sha, n_cases=len(cases), n_run_rows=run_rows,
        per_component=dict(sorted(per_comp.items())),
        derived_at_run=("512/model: e2 from the 256 e1_primary rows, l2e2 "
                        "from the 256 l2 rows — downstream_query is present "
                        "ONLY on rows that spawn a derived call (plus "
                        "v1_grid_extra rows, where the field is part of the "
                        "v1 containment schema but those rows are not run)"),
        split_rule="cluster % 5 == 4 -> test (20%), else dev",
        git_commit=commit or "pre-commit",
        generator_version=GENERATOR_VERSION,
        system_template=SYS_V1.replace("2026-08-11", args.eval_date))
    (OUT / "manifest_v2.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(cases)} rows ({run_rows} in run allocation) -> {path}")
    print(json.dumps(per_comp, indent=2, sort_keys=True))
    print("sha256:", sha)


if __name__ == "__main__":
    main()
