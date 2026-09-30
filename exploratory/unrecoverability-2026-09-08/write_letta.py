"""Letta writer for the unrecoverability arm (Part G). Copied from Part D
run_letta.py; changes: inputs = writer_inputs.jsonl (288 grid items), model
handle passed on argv (openai-proxy/deepseek-v4-flash = official DeepSeek
endpoint through Letta's OpenAI provider with OPENAI_API_BASE), output
writer_trace_letta_<tag>.jsonl. Letta has no reference-time input; the
reader header supplies the stale write date.

Original docstring follows.

Part D of research/READ_TIME_LADDER_SPEC_2026-09-02.md — Letta.

letta 0.16.8 server (local, embedded Postgres via pgserver, no code changes),
letta-client 1.12.1. Default agent (server-default agent_type, default
persona/human memory blocks, base tools incl. core-memory edits and
archival insert), one fresh agent per item, one user message, no follow-up.
Memory = whatever the agent itself chose to write: memory blocks after the
turn + archival passages + the tool calls it made.

    python run_letta.py openrouter/deepseek/deepseek-v4-flash [limit]

Inputs: inputs.jsonl (162 items). Output: trace_letta_<writer>.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from letta_client import Letta

HERE = Path(__file__).resolve().parent
WRITER = sys.argv[1]
TAG = re.sub(r"[^a-z0-9]+", "-", WRITER.lower())
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
EMBEDDING = "letta/letta-free"
# Letta's own defaults (constants.DEFAULT_HUMAN="basic", DEFAULT_PERSONA="sam_pov"), copied verbatim from the package.
MEMORY_BLOCKS = [
    {"label": "human", "value": (HERE / "default_human.txt").read_text().strip()},
    {"label": "persona", "value": (HERE / "default_persona.txt").read_text().strip()},
]


def dump_messages(msgs):
    out = []
    for m in msgs:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        keep = {k: d.get(k) for k in ("message_type", "content", "reasoning", "tool_call", "tool_calls", "tool_return", "name", "status") if d.get(k) is not None}
        out.append(keep)
    return out


def main():
    items = [json.loads(l) for l in (HERE / "writer_inputs.jsonl").read_text().splitlines()]
    if LIMIT:
        items = items[:LIMIT]
    out_path = HERE / f"writer_trace_letta_{TAG}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()}
    c = Letta(base_url="http://localhost:8283", timeout=300)
    out = out_path.open("a")
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        t0 = time.time()
        err, blocks_before, blocks_after, passages, msgs, agent_id = None, [], [], [], [], None
        try:
            a = c.agents.create(name=f"g-{it['id']}"[:100], model=WRITER, embedding=EMBEDDING, memory_blocks=MEMORY_BLOCKS)
            agent_id = a.id
            blocks_before = [dict(label=b.label, value=b.value) for b in c.agents.blocks.list(agent_id)]
            r = c.agents.messages.create(agent_id, messages=[{"role": "user", "content": it["text"]}])
            msgs = dump_messages(r.messages)
            blocks_after = [dict(label=b.label, value=b.value) for b in c.agents.blocks.list(agent_id)]
            passages = [p.text for p in c.agents.passages.list(agent_id)]
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:500]
        out.write(json.dumps(dict(it, writer=WRITER, system="letta 0.16.8", agent_id=agent_id,
                                  blocks_before=blocks_before, blocks_after=blocks_after, passages=passages,
                                  messages=msgs, error=err, secs=round(time.time() - t0, 1)),
                             ensure_ascii=False) + "\n")
        out.flush()
        changed = [b["label"] for b in blocks_after if b not in blocks_before]
        print(f"{i + 1}/{len(items)} {it['form']} {it['id']} -> blocks changed {changed} passages {len(passages)}"
              + (f" ERR {err}" if err else ""), file=sys.stderr)
    out.close()


if __name__ == "__main__":
    main()
