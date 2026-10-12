#!/usr/bin/env python3
"""Added-reader runner (research/DOWNSTREAM_TOOL_READER_ADDENDUM_2026-10-10.md).

Runs the powered run's runner unchanged (../downstream-tool-powered-2026-09-22/run_powered.py,
same generated items, prompt, tools, parser, scoring). Only the config, compiled inputs, traces,
gates and freeze file are redirected to this folder.

    python run_addendum.py compile|run --stage STAGE --reader haiku55-anthropic [--age A]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "downstream-tool-powered-2026-09-22"))
import run_powered as P  # noqa: E402

P.CONFIG = json.loads((HERE / "config_addendum.json").read_text())
P.HERE = HERE            # freeze_confirmatory.json and gates/ live here
P.COMPILED = HERE / "compiled"
P.TRACES = HERE / "traces"
P.TRACES.mkdir(exist_ok=True)

if __name__ == "__main__":
    P.main()
