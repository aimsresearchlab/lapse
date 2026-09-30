"""Part D of research/READ_TIME_LADDER_SPEC_2026-09-02.md — Graphiti.

graphiti-core 0.30.1, unmodified add_episode path (extract nodes -> dedupe
-> extract edges + valid_at/invalid_at -> dedupe edges -> node attributes),
embedded Kuzu driver (no server), fastembed bge-small embedder, no-op
reranker (only used at search time). One isolated group_id per item, so no
cross-item dedupe. reference_time = wall clock at run.

    python run_graphiti.py deepseek/deepseek-v4-flash

Inputs: inputs.jsonl (162 items: 32 grid prog + 32 grid simple (08-27
pairs), 32 grid perfect-progressive + duration, 9 LongMemEval + 57 WildChat
real utterances). Output: trace_graphiti_<writer>.jsonl with every entity
node and entity edge (name, fact, valid_at, invalid_at, expired_at).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from openai import AsyncOpenAI
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType

HERE = Path(__file__).resolve().parent
WRITER = sys.argv[1]
TAG = re.sub(r"[^a-z0-9]+", "-", WRITER.lower())
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None


class FastEmbedder(EmbedderClient):
    def __init__(self):
        from fastembed import TextEmbedding
        self.m = TextEmbedding("BAAI/bge-small-en-v1.5")

    async def create(self, input_data):
        if isinstance(input_data, str):
            input_data = [input_data]
        return [float(x) for x in next(iter(self.m.embed(list(input_data))))]

    async def create_batch(self, input_data_list):
        return [[float(x) for x in v] for v in self.m.embed(list(input_data_list))]


class NoopReranker(CrossEncoderClient):
    async def rank(self, query, passages):
        return [(p, 1.0) for p in passages]


def edge_row(e):
    f = lambda d: d.isoformat() if d else None
    return dict(name=e.name, fact=e.fact, valid_at=f(e.valid_at), invalid_at=f(e.invalid_at),
                expired_at=f(e.expired_at), attributes=e.attributes or {})


async def main():
    items = [json.loads(l) for l in (HERE / "inputs.jsonl").read_text().splitlines()]
    if LIMIT:
        items = items[:LIMIT]
    out_path = HERE / f"trace_graphiti_{TAG}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()}
    llm = OpenAIGenericClient(
        LLMConfig(api_key=os.environ["OPENROUTER_API_KEY"], model=WRITER, small_model=WRITER,
                  base_url="https://openrouter.ai/api/v1", temperature=0.0),
        structured_output_mode="json_object",
        client=AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1",
                           timeout=120.0, max_retries=3))
    g = Graphiti(graph_driver=KuzuDriver(db=str(HERE / f"kuzu_{TAG}")), llm_client=llm,
                 embedder=FastEmbedder(), cross_encoder=NoopReranker())
    await g.build_indices_and_constraints()  # no-op for Kuzu in 0.30.1: FTS indices are never built
    import kuzu
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.graph_queries import get_fulltext_indices
    conn = kuzu.Connection(g.driver.db)
    for q in ["INSTALL FTS;", "LOAD EXTENSION FTS;"] + get_fulltext_indices(GraphProvider.KUZU):
        try:
            conn.execute(q)
        except Exception as e:  # noqa: BLE001  (index already exists on resume)
            print("index setup:", q[:60], repr(e)[:120], file=sys.stderr)
    conn.close()
    out = out_path.open("a")
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        gid = re.sub(r"[^A-Za-z0-9_]", "_", f"d_{TAG}_{it['id']}")
        t0 = time.time()
        ref = datetime.now(timezone.utc)
        err, nodes, edges = None, [], []
        try:
            g.driver._database = gid  # KuzuDriver lacks _database; graphiti compares it to group_id
            r = await g.add_episode(name=it["id"], episode_body="user: " + it["text"], source=EpisodeType.message,
                                    source_description="user chat message", reference_time=ref, group_id=gid)
            nodes = [dict(name=n.name, labels=n.labels, summary=n.summary, attributes=n.attributes or {}) for n in r.nodes]
            edges = [edge_row(e) for e in r.edges]
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        out.write(json.dumps(dict(it, writer=WRITER, system="graphiti-core 0.30.1", reference_time=ref.isoformat(),
                                  nodes=nodes, edges=edges, error=err, secs=round(time.time() - t0, 1)),
                             ensure_ascii=False) + "\n")
        out.flush()
        print(f"{i + 1}/{len(items)} {it['set']} {it['id']} -> {len(nodes)} nodes {len(edges)} edges"
              + (f" ERR {err}" if err else ""), file=sys.stderr)
    out.close()
    await g.close()


if __name__ == "__main__":
    asyncio.run(main())
