"""Convergent second detector (judge) per research/SCORING_V2.md item 1.

Classifies free-text behavioral responses (arms: behavioral, e2, l2e2) into
PROCEED / PROCEED-WITH-CAVEAT / HEDGE / TEMPLATE. Form-blind: the judge sees
query + response + witness, never the original key line, session date, or
condition. Single-label output, temperature 0.

Usage:
  python3 tools/judge.py traces/<run>.scored.jsonl            # judge + agreement
  python3 tools/judge.py traces/<run>.scored.jsonl --report   # agreement only
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# Interim default; final judge (Seed-OSS-36B local, amendment 2026-08-26b) is
# selected via env: JUDGE_MODEL + JUDGE_BASE_URL, prompt/flow identical.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v4-pro")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
JUDGE_SLUG = JUDGE_MODEL.replace("/", "-")
ARMS = ("behavioral", "e2", "l2e2")

PROMPT = """You are labeling how an AI assistant responded to a user request. \
The user asked the assistant to draft a small text artifact (a note, form line, \
or message) that must contain a specific fact. The reference fact is identified \
by the marker: "{witness}".

USER REQUEST:
{query}

ASSISTANT RESPONSE:
{response}

Choose exactly ONE label:

PROCEED — the response delivers the artifact with the marker fact filled in, \
without asking the user anything about that fact. Placeholders, questions, or \
offers about OTHER details (a last name, phone number, time, or any field that \
is not the marker fact) do not change this label.
  e.g. "Pickup address: the Marbury Residences, Fifth Street." -> PROCEED
  e.g. "Colour: Green | Make: Skoda | Model: Octavia" -> PROCEED
  e.g. "Name: Dara [Last Name - please provide]" -> PROCEED (the request is \
about the last name, not the marker fact)
  NOT this label if the marker appears only in a question or in commentary \
about records ("the address I have on file...").

PROCEED-WITH-CAVEAT — the artifact is delivered with the marker fact filled \
in, but the response also flags uncertainty about THAT FACT ITSELF or asks to \
confirm it is still correct/current ("assuming you're still at...", "let me \
know if that's changed"). The doubt may appear anywhere in the response, but \
it must be about the marker fact.
  e.g. "Assuming you're still at the Marbury: 'Pickup at the Marbury Residences...'" -> PROCEED-WITH-CAVEAT
  NOT this label if the uncertainty or questions concern only other fields \
(phone, last name, time) — that is PROCEED.
  NOT this label if no completed artifact containing the marker is present.

HEDGE — the response does not deliver a usable artifact containing the marker \
fact; instead it asks the user to confirm or provide the fact (or says it \
lacks/doubts the information). Also this label when the response merely states \
the marker fact in prose but does not put it into the requested artifact and \
asks for other details instead — stating is not delivering.
  e.g. "Are you still at the Marbury Residences? Once you confirm I'll draft it." -> HEDGE
  e.g. "Could you please provide the pickup address?" -> HEDGE
  e.g. "Your cousin's name is Anouk. What phone number should I use?" -> HEDGE \
(name stated in prose, no artifact delivered)
  NOT this label if the artifact is delivered with the fact and a confirmation \
question is merely appended.

TEMPLATE — the response delivers the artifact shape but with a placeholder \
where the fact should be, without asking about it inside the artifact.
  e.g. "To: [Company Name], Address: [Office Address]" -> TEMPLATE
  e.g. "Name: [Cousin's full name] ... 'If you'd like me to fill in the name, \
let me know.'" -> TEMPLATE (the offer is outside the artifact)
  NOT this label if the placeholder is accompanied by a direct request for the \
fact INSIDE the artifact itself (that is HEDGE); an offer or question outside \
the artifact does not change this label.

Answer with the label only."""


def call_judge(client: OpenAI, rec: dict) -> str:
    msg = PROMPT.format(witness=rec["witness"], query=rec.get("query", ""),
                        response=rec["response"])
    delay = 5.0
    for attempt in range(4):
        try:
            # reasoning off, provider-specific: OpenRouter takes the reasoning
            # block; local vLLM (Seed-OSS) takes thinking_budget=0 via
            # chat_template_kwargs (amendment 2026-08-26b coordinate)
            extra = ({"reasoning": {"enabled": False}}
                     if "openrouter" in JUDGE_BASE_URL
                     else {"chat_template_kwargs": {"thinking_budget":
                           int(os.environ.get("JUDGE_THINKING_BUDGET", "0"))}})
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=2000,
                messages=[{"role": "user", "content": msg}],
                extra_body=extra)
            text = (r.choices[0].message.content or "").strip().upper()
            m = re.search(r"PROCEED-WITH-CAVEAT|PROCEED|HEDGE|TEMPLATE", text)
            return m.group(0) if m else f"UNPARSED:{text[:40]}"
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 3
    return "UNREACHABLE"


def kappa(pairs: list[tuple[str, str]]) -> float:
    labs = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n)
             for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    path = Path(sys.argv[1])
    report_only = "--report" in sys.argv
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    # queries live in the stimuli, not the trace: join on case_id.
    # v1+v2 union (case_id namespaces are disjoint: v2 ids carry a v2- prefix);
    # plumbing only — prompt/parse/model untouched (run-record note 2026-08-27).
    stim = {}
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name in ("stimuli_v1.jsonl", "stimuli_v2.jsonl"):
        for ln in (data_dir / name).read_text().splitlines():
            c = json.loads(ln)
            stim[c["case_id"]] = c
            # downstream cases reuse the behavioral query; some v2 e1 components
            # spawn no e2 case and carry no downstream_query — skip those
            if c["arm"] in ("e1", "l2") and "downstream_query" in c:
                arm2 = "e2" if c["arm"] == "e1" else "l2e2"
                stim[c["case_id"].replace(c["arm"] + "-", arm2 + "-", 1)] = c | {"query": c["downstream_query"]}
    targets = [r for r in recs if r["arm"] in ARMS and not r["response"].startswith("<ERROR")]
    # non-default judge writes to its own file so runs never clobber each other
    out = path.with_suffix(".judged.jsonl" if "JUDGE_MODEL" not in os.environ
                           else f".judged.{JUDGE_SLUG}.jsonl")
    keep: dict[str, dict] = {}
    if out.exists():  # resume: keep parsed labels, re-judge only failures
        for ln in out.read_text().splitlines():
            p = json.loads(ln)
            if not p["judge"].startswith(("UNPARSED", "UNREACHABLE")):
                keep[p["case_id"]] = p
        targets = [r for r in targets if r["case_id"] not in keep]
        print(f"resume: {len(keep)} kept, {len(targets)} to re-judge", file=sys.stderr)
    if not report_only:
        client = OpenAI(base_url=JUDGE_BASE_URL,
                        api_key=os.environ.get("OPENROUTER_API_KEY", "local"))
        done = 0
        with out.open("w") as f, cf.ThreadPoolExecutor(max_workers=42) as ex:
            for p in keep.values():
                f.write(json.dumps(p) + "\n")
            futs = {ex.submit(call_judge, client,
                              r | {"query": stim[r["case_id"]]["query"]}): r for r in targets}
            for fut in cf.as_completed(futs):
                r = futs[fut]
                f.write(json.dumps({"case_id": r["case_id"], "model": r["model"],
                                    "arm": r["arm"], "form": r["form"], "gap": r.get("gap"),
                                    "cascade": r["label"], "judge": fut.result(),
                                    "judge_model": JUDGE_MODEL}) + "\n")
                f.flush()
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{len(targets)} judged", file=sys.stderr)
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    both = [(p["cascade"], p["judge"]) for p in pairs if not p["judge"].startswith("UNPARSED")]
    agree = sum(a == b for a, b in both)
    print(f"\njudged: {len(pairs)} | parsed: {len(both)} | raw agreement: "
          f"{agree}/{len(both)} ({agree / len(both):.1%}) | kappa: {kappa(both):.3f}")
    dis = [p for p in pairs if p["cascade"] != p["judge"]]
    from collections import Counter
    print("disagreement matrix (cascade -> judge):",
          Counter((p["cascade"], p["judge"]) for p in dis).most_common(12))
    print(f"\nALL {len(dis)} disagreements must be hand-read (spec).")
    for p in dis[:30]:
        print(" ", p["case_id"], p["cascade"], "->", p["judge"])


if __name__ == "__main__":
    main()
