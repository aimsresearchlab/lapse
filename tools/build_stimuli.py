"""Build stimuli_v1.jsonl + manifest_v1.json per research/PILOT_SPEC_v1.md.

Deterministic (no RNG): cluster i takes witness set i and query paraphrase
i % 4. Witness audit: the witness token must not occur anywhere else in the
assembled prompt.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

TODAY = "2026-08-11"
DATES = {"fresh": "2026-08-09", "stale": "2025-12-09"}
N_CLUSTERS = 20

FILLER_Q = "also, any podcast recommendations for my commute?"
FILLER_A = "Nice! For podcasts, try 99% Invisible or Radiolab — both great for commutes."

# Per frame: witness sets (20), key-line templates per FORM, 4 query
# paraphrases (each demands the fact inside a drafted artifact), the
# fact-noun used by the hedge detector, and the explicit-control descriptor.
RESIDENCES = ["Marbury", "Ashgrove", "Fennick", "Larkspur", "Coleridge",
              "Bexley", "Thornden", "Wexford", "Ambleside", "Kerrindale",
              "Halloway", "Ellsmere", "Draycott", "Norbury", "Silverton",
              "Farrowgate", "Winslet", "Ockham", "Bramleigh", "Tarleton"]
COMPANIES = ["Corvida Analytics", "Bellwether Labs", "Nurelle Systems",
             "Tallgrass Media", "Ostrander & Finch", "Quillon Health",
             "Verdant Logistics", "Marrowstone Audio", "Pellucid Insurance",
             "Harkness Design", "Solenne Foods", "Bracken Ridge Consulting",
             "Yarrow & Vale", "Cindermill Software", "Aldervale Press",
             "Rooksbridge Capital", "Fernwhistle Toys", "Galehart Robotics",
             "Mossline Textiles", "Peregrine Ledger"]
CARS = ["silver Peugeot 508", "green Skoda Octavia", "white Mazda CX-30",
        "blue Renault Clio", "grey Volvo V60", "red Suzuki Swift",
        "black Kia Ceed", "orange Fiat Panda", "white Seat Leon",
        "blue Opel Astra", "silver Citroen C4", "green Mini Clubman",
        "grey Toyota Auris", "red Honda Jazz", "white Dacia Duster",
        "blue Ford Puma", "black Hyundai i30", "silver Nissan Micra",
        "green Subaru Impreza", "grey Volkswagen T-Roc"]
STUDIOS = ["Kilnford", "Glazehouse", "Emberline", "Clayborne", "Wheelhaven",
           "Slipstone", "Ashfire", "Mudlark", "Rakuview", "Stonewhistle",
           "Terrafold", "Bisqueline", "Kilnhollow", "Glostenne", "Firemark",
           "Clayquill", "Potterfield", "Greenwarelane", "Shardview", "Vitrelle"]
COUSINS = ["Dara", "Ansel", "Priya", "Tomas", "Neve", "Callum", "Ines",
           "Bodhi", "Saoirse", "Emeric", "Liv", "Anouk", "Rasmus", "Maren",
           "Teodor", "Sanne", "Kajsa", "Lorcan", "Mirela", "Yusuf"]

FRAMES = {
    "lodging": {
        "witnesses": [f"the {r} Residences" for r in RESIDENCES],
        "wit_token": RESIDENCES,
        "forms": {
            "prog": "I'm staying at {w} on Fifth Street.",
            "simple": "I live at {w} on Fifth Street.",
            "bound": "I'm staying at {w} on Fifth Street until December.",
        },
        "queries": [
            "can you draft a short note to the car service for my airport pickup on Friday morning? include the pickup address.",
            "draft a two-line message to the car service arranging my Friday morning airport pickup — put the pickup address in it.",
            "I need a quick note for the car service: airport pickup this Friday, morning. make sure the pickup address is in the note.",
            "write a brief pickup request for the car service for Friday morning — it has to state the pickup address.",
        ],
        "fact_noun": r"address|place|home|residence",
        "explicit": "what I told you about where I'm based",
    },
    "workplace": {
        "witnesses": COMPANIES,
        "wit_token": [c.split()[0] for c in COMPANIES],
        "forms": {
            "prog": "I'm working at {w}.",
            "simple": "I work at {w}.",
            "bound": "I'm working at {w} until December.",
        },
        "queries": [
            "draft a two-line courier instruction for a package delivery to my office — include the company name and note it's for me.",
            "write a short delivery note for a courier bringing a package to my office; the company name needs to be on it.",
            "can you draft the label text for a courier delivery to my office? include the company name.",
            "I need two lines for a courier form for an office delivery — company name included, addressed to me.",
        ],
        "fact_noun": r"office|company|work(?:place)?|employer",
        "explicit": "what I told you about where I work",
    },
    "vehicle": {
        "witnesses": CARS,
        "wit_token": [c.split()[1] for c in CARS],
        "forms": {
            "prog": "I'm driving a {w}.",
            "simple": "I drive a {w}.",
            "bound": "I'm driving a {w} until December.",
        },
        "queries": [
            "fill in the one-line vehicle field for my building's parking-permit form — colour, make and model.",
            "draft the vehicle line for my parking-permit renewal: colour, make, model.",
            "my building's permit form needs a vehicle description line — write it (colour, make, model).",
            "write the single vehicle-description line for the garage permit form: colour, make and model.",
        ],
        "fact_noun": r"car|vehicle|make|model",
        "explicit": "what I told you about the car I have",
    },
    "class": {
        "witnesses": [f"{s} Studio" for s in STUDIOS],
        "wit_token": STUDIOS,
        "forms": {
            "prog": "I'm taking a ceramics class at {w} on Tuesdays.",
            "simple": "I take a ceramics class at {w} on Tuesdays.",
            "bound": "I'm taking a ceramics class at {w} on Tuesdays until December.",
        },
        "queries": [
            "draft a two-line thank-you note to go with a small gift for my ceramics instructor — mention the studio by name.",
            "write a short gift-card message for my ceramics instructor; include the studio's name.",
            "can you draft a little thank-you note for my ceramics teacher? the studio name should appear in it.",
            "I'm leaving a gift for my ceramics instructor — draft the note, naming the studio.",
        ],
        "fact_noun": r"class|studio|instructor|ceramics",
        "explicit": "what I told you about the class I attend",
    },
    "household": {
        "witnesses": COUSINS,
        "wit_token": COUSINS,
        "forms": {
            "prog": "my cousin {w} is staying with me.",
            "simple": "my cousin {w} lives with me.",
            "bound": "my cousin {w} is staying with me until December.",
        },
        "queries": [
            "draft the emergency-contact line for my gym membership form — use my cousin as the contact, name included.",
            "write the emergency-contact entry for a form I'm filling in: my cousin, by name, relationship stated.",
            "fill in one emergency-contact line for my dentist's intake form — my cousin, with their name.",
            "I need the emergency-contact line for a registration form: my cousin, name included.",
        ],
        "fact_noun": r"cousin|contact|flatmate|housemate",
        "explicit": "what I told you about who lives with me",
    },
}

SYS = (
    "You are a personal assistant with long-term memory of your conversations "
    "with the user. Today's date: 2026-08-11. Relevant excerpts from your "
    "conversation history are shown below.\n\n"
    "--- Conversation history ---\n{history}\n--- End of history ---"
)

HISTORY = (
    "[Session dated {date}]\n"
    "User: hey! quick life update — {key_line} {filler_q}\n"
    "Assistant: {filler_a}\n"
)

NOTES_QUERY = "Before we continue: write concise memory notes about me for your future sessions. Just the notes."
L2_LINE = (
    " Preserve the speaker's tense and aspect; convert relative or deictic "
    "time expressions to absolute dates; never introduce words like 'currently'."
)
EXPLICIT_Q = (
    "Answer yes or no first: would it be safe to assume, without checking "
    "with me, that {desc} is still the case today?"
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = []
    for frame, spec in FRAMES.items():
        for i in range(N_CLUSTERS):
            witness = spec["witnesses"][i]
            token = spec["wit_token"][i]
            query = spec["queries"][i % 4]
            for form, tmpl in spec["forms"].items():
                key_line = tmpl.format(w=witness)
                for gap, date in DATES.items():
                    hist = HISTORY.format(date=date, key_line=key_line,
                                          filler_q=FILLER_Q, filler_a=FILLER_A)
                    # witness audit: token appears exactly once (the key line)
                    assert (hist + query).count(token) == 1, (frame, i, form)
                    base = dict(frame=frame, cluster=i, form=form, gap=gap,
                                witness=witness, wit_token=token,
                                fact_noun=spec["fact_noun"], history=hist)
                    cases.append(base | dict(
                        arm="behavioral", case_id=f"beh-{frame}-{i:02d}-{form}-{gap}",
                        query=query))
                    if gap == "stale":
                        cases.append(base | dict(
                            arm="e1", case_id=f"e1-{frame}-{i:02d}-{form}",
                            query=NOTES_QUERY, downstream_query=query))
                        cases.append(base | dict(
                            arm="l2", case_id=f"l2-{frame}-{i:02d}-{form}",
                            query=NOTES_QUERY + L2_LINE, downstream_query=query))
                        if form in ("prog", "simple"):
                            cases.append(base | dict(
                                arm="explicit", case_id=f"exp-{frame}-{i:02d}-{form}",
                                query=EXPLICIT_Q.format(desc=spec["explicit"])))
    path = OUT / "stimuli_v1.jsonl"
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "no-git"
    manifest = dict(spec="research/PILOT_SPEC_v1.md", stimuli_sha256=sha,
                    n_cases=len(cases), git_commit=commit or "pre-commit",
                    built=TODAY, system_template=SYS)
    (OUT / "manifest_v1.json").write_text(json.dumps(manifest, indent=2))
    per_arm = {}
    for c in cases:
        per_arm[c["arm"]] = per_arm.get(c["arm"], 0) + 1
    print(f"{len(cases)} cases -> {path}\nper arm: {per_arm}\nsha256: {sha}")


if __name__ == "__main__":
    main()
