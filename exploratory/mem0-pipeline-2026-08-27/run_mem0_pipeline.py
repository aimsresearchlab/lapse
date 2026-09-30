"""Real-pipeline ecological arm: LAPSE minimal pairs through installed mem0.

EXPLORATORY (not the frozen grid). mem0 2.0.19 OSS, as-deployed write
path: Memory.add() = extraction LLM call + update/merge LLM call +
vector store. Backend deepseek/deepseek-v4-flash via OpenRouter, temp 0
(same model as confirmatory column 1). Embedder: fastembed (local).
Store: qdrant embedded (local path). No monkeypatching — the pipeline
runs exactly as shipped; write time = actual run time (deployment-
faithful; the prompt's "Today's date" is the real today).

Design: 8 frames x 4 clusters x {prog, simple} = 64 users. Phase 1:
add() the raw grid user utterance (extracted from stimulus history).
Phase 2: same user, a no-new-information same-topic ping — probes
whether the UPDATE/merge phase rewrites or flattens the stored
temporal form when re-consolidating.

Deterministic post-checks per stored memory (descriptive; full dump
retained for hand-reading): witness survival, progressive-morphology
survival, temporal-adverbial survival, date-anchor presence.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

from mem0 import Memory  # noqa: E402

N_CLUSTERS = 4
FORMS = ("prog", "simple")

cfg = {
    "llm": {"provider": "openai", "config": {
        "model": "deepseek/deepseek-v4-flash",
        "openai_base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "temperature": 0.0,
    }},
    "embedder": {"provider": "fastembed", "config": {
        "model": "BAAI/bge-small-en-v1.5",  # 384-dim, local
    }},
    "vector_store": {"provider": "qdrant", "config": {
        "path": str(HERE / "qdrant_local"),
        "embedding_model_dims": 384,
        "on_disk": True,
    }},
}


def user_turn(history: str) -> str:
    for ln in history.splitlines():
        if ln.startswith("User: "):
            return ln[len("User: "):]
    raise ValueError("no user turn")


def main() -> None:
    stim = [json.loads(ln) for ln in
            (REPO / "data" / "stimuli_v2.jsonl").read_text().splitlines()]
    cases = [c for c in stim
             if c["component"] == "e1_primary" and c["gap"] == "stale"
             and c["carrier"] == "c0_original" and c["form"] in FORMS
             and c["cluster"] < N_CLUSTERS]
    cases.sort(key=lambda c: (c["frame"], c["cluster"], c["form"]))
    print(f"{len(cases)} cases", file=sys.stderr)

    m = Memory.from_config(cfg)
    out = (HERE / "mem0_pipeline_trace.jsonl").open("w")

    for i, c in enumerate(cases):
        uid = f"lapse-{c['frame']}-{c['cluster']}-{c['form']}"
        utt = user_turn(c["history"])
        topic = c["fact_noun"].split("|")[0]

        r1 = m.add([{"role": "user", "content": utt}], user_id=uid)
        mem1 = [x["memory"]
                for x in m.get_all(filters={"user_id": uid})["results"]]

        ping = (f"quick follow-up on my {topic} situation — nothing has "
                f"changed, everything is still exactly as I told you.")
        r2 = m.add([{"role": "user", "content": ping}], user_id=uid)
        mem2 = [x["memory"]
                for x in m.get_all(filters={"user_id": uid})["results"]]

        out.write(json.dumps(dict(
            case_id=c["case_id"], frame=c["frame"], cluster=c["cluster"],
            form=c["form"], witness=c["witness"], wit_token=c["wit_token"],
            utterance=utt, ping=ping,
            add1=r1, memories_after_add=mem1,
            add2=r2, memories_after_ping=mem2,
        ), ensure_ascii=False) + "\n")
        out.flush()
        print(f"{i + 1}/{len(cases)} {uid}", file=sys.stderr)

    out.close()

    # deterministic post-checks
    prog_morph = re.compile(r"\b\w+ing\b")
    advb = re.compile(r"\b(temporar\w+|for now|for a bit|for the time being"
                      r"|while|until|currently|at the moment|these days"
                      r"|right now)\b", re.I)
    anchor = re.compile(r"\b(20\d\d|january|february|march|april|may|june"
                        r"|july|august|september|october|november"
                        r"|december|as of|since)\b", re.I)
    rows = [json.loads(ln) for ln in
            (HERE / "mem0_pipeline_trace.jsonl").read_text().splitlines()]
    print("\nphase form  witness  prog-morph  adverbial  date-anchor  n")
    for phase, key in (("add", "memories_after_add"),
                       ("ping", "memories_after_ping")):
        for form in FORMS:
            rs = [r for r in rows if r["form"] == form]
            wit = sum(any(r["wit_token"].lower() in mm.lower()
                          for mm in r[key]) for r in rs)
            pm = sum(any(prog_morph.search(mm) for mm in r[key]) for r in rs)
            ad = sum(any(advb.search(mm) for mm in r[key]) for r in rs)
            an = sum(any(anchor.search(mm) for mm in r[key]) for r in rs)
            print(f"{phase:5s} {form:6s} {wit:3d}      {pm:3d}         "
                  f"{ad:3d}        {an:3d}          {len(rs)}")


if __name__ == "__main__":
    main()
