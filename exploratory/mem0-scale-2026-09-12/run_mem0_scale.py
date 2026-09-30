#!/usr/bin/env python3
"""Installed mem0 write path, sharded. Usage: python run_mem0_scale.py <shard> <nshards>. Append-only, resumable."""
import json, os, sys, time
from pathlib import Path
from mem0 import Memory
HERE = Path(__file__).resolve().parent
SHARD, N = int(sys.argv[1]), int(sys.argv[2])
WRITER = "deepseek/deepseek-v4-flash"
cfg = {"llm": {"provider": "openai", "config": {"model": WRITER, "openai_base_url": "https://openrouter.ai/api/v1", "api_key": os.environ["OPENROUTER_API_KEY"], "temperature": 0.0}},
       "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5"}},
       "vector_store": {"provider": "qdrant", "config": {"path": str(HERE / f"qdrant_shard{SHARD}"), "embedding_model_dims": 384, "on_disk": True}}}
items = [json.loads(l) for l in (HERE / "inputs.jsonl").read_text().splitlines()]
items = [it for i, it in enumerate(items) if i % N == SHARD]
out_path = HERE / f"trace_shard{SHARD}.jsonl"
done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
m = Memory.from_config(cfg)
with out_path.open("a") as out:
    for i, it in enumerate(items):
        if it["id"] in done: continue
        uid = f"s-{it['id']}"; t0 = time.time(); err = None
        try:
            r = m.add([{"role": "user", "content": it["text"]}], user_id=uid)
            mems = [x["memory"] for x in m.get_all(filters={"user_id": uid})["results"]]
        except Exception as e:  # noqa: BLE001
            r, mems, err = None, [], repr(e)
        out.write(json.dumps(dict(it, writer=WRITER, add=r, memories=mems, error=err, secs=round(time.time() - t0, 1), ts=time.time()), ensure_ascii=False) + "\n"); out.flush()
        print(f"shard{SHARD} {i+1}/{len(items)} {it['id']} -> {len(mems)} mem" + (f" ERR {err}" if err else ""), file=sys.stderr)
