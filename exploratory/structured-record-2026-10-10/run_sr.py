#!/usr/bin/env python3
"""Reader runner: the powered run's run_powered.py unchanged, redirected to this folder.

    python run_sr.py compile|run --stage cellgate|target --reader R --age A
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "downstream-tool-powered-2026-09-22"))
import run_powered as P  # noqa: E402

P.CONFIG = json.loads((HERE / "config_sr.json").read_text())
P.HERE = HERE            # freeze_confirmatory.json and gates/ live here
P.COMPILED = HERE / "compiled"
P.TRACES = HERE / "traces"
P.TRACES.mkdir(exist_ok=True)


def stage_rows(stage, age):
    if stage not in ("cellgate", "target"):
        raise SystemExit(f"stage {stage} is not part of this experiment")
    return P.load(HERE / "generated" / f"{stage}-{age}.jsonl")


P.stage_rows = stage_rows

if __name__ == "__main__":
    P.main()
