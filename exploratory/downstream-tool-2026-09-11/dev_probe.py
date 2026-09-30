#!/usr/bin/env python3
"""Compile the disclosed R2 dev-probe slice (6 contents x 3 controls x 3 readers). Never a freeze or a gate."""
import argparse, json
from pathlib import Path
import build_inputs, freeze_inputs

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); config = json.loads(a.config.read_text()); freeze_inputs.validate_config(config)
    source, _ = build_inputs.extract(Path(build_inputs.__file__).resolve().parents[2] / "exploratory/unrecoverability-2026-09-08/reader_g_inputs.jsonl")
    items = build_inputs.dev_probe({x["witness"].casefold() for x in source})
    rows = [freeze_inputs.compiled_item(item, reader) for item in items for reader in config["reader_configs"]]
    for i, row in enumerate(rows, 1): row["call_order"] = i
    a.output.parent.mkdir(parents=True, exist_ok=True); freeze_inputs.write_jsonl(a.output, rows); print(json.dumps({"rows": len(rows), "output": str(a.output)}))

if __name__ == "__main__": main()
