#!/usr/bin/env python3
"""Cell gate on record-format controls: the powered run's analyze_powered.py gate, redirected here.

    python analyze_gate.py gate --kind cellgate --reader R --age A
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "downstream-tool-powered-2026-09-22"))
import analyze_powered as A  # noqa: E402

A.HERE = HERE
A.TRACES = HERE / "traces"
A.GATES = HERE / "gates"

if __name__ == "__main__":
    A.main()
