#!/usr/bin/env python3
"""Inputs for the installed-mem0 scale-up: the user turn of every qualifier-ablation stimulus (128 clusters x 4 forms)."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "qualifier-ablation-2026-09-12" / "stimuli.jsonl"
rows = [json.loads(l) for l in SRC.read_text().splitlines()]
out = []
for r in rows:
    utt = next(ln[len("User: "):] for ln in r["history"].splitlines() if ln.startswith("User: "))
    out.append({"id": r["case_id"], "frame": r["frame"], "cluster": r["cluster"], "form": r["form"], "witness": r["witness"],
                "wit_token": r["wit_token"], "marker": r["marker"], "text": utt})
assert len(out) == 512
p = HERE / "inputs.jsonl"; p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in out))
man = {"source": str(SRC), "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "n": len(out), "inputs_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
       "writer": "deepseek/deepseek-v4-flash via OpenRouter (mem0 openai provider; host not pinnable through mem0 config, as in the 2026-08-27 run)",
       "mem0": "2.0.19 OSS, unmodified write path, fastembed bge-small, qdrant on disk, one isolated user per item, no ping phase", "temperature": 0}
(HERE / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n"); print(json.dumps(man))
