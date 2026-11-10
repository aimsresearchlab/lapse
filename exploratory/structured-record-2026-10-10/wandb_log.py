#!/usr/bin/env python3
"""Log the stage results to W&B (project lapse, group structured-record)."""
import json
import subprocess
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
reader, age = sys.argv[1], sys.argv[2]
res = json.loads((HERE / f"results-{reader}-{age}.json").read_text())
commit = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
run = wandb.init(project="lapse", group="structured-record", name=f"structured-record-{reader}-{age}-2026-10-10",
                 config={"reader": reader, "age": age, "writer": "deepseek/deepseek-v4-flash (OpenRouter, Baidu fp8)",
                         "spec": "research/STRUCTURED_RECORD_SPEC_2026-10-10.md", "git_commit": commit,
                         "freeze": json.loads((HERE / "freeze_confirmatory.json").read_text()),
                         "writer_summary": json.loads((HERE / "writer_summary.json").read_text())})
flat = {f"{arm}/{k}": v for arm, d in res["arms"].items() for k, v in d.items()}
flat.update({f"H1/{k}": v for k, v in res["H1"].items()})
flat.update({f"H2/{k}": v for k, v in res["H2"].items()})
run.log(flat)
art = wandb.Artifact(f"structured-record-{reader}-{age}", type="results")
for p in (f"results-{reader}-{age}.json", "writer_summary.json", "notes.jsonl", f"traces/target-{reader}-{age}.jsonl"):
    art.add_file(str(HERE / p))
run.log_artifact(art)
run.finish()
