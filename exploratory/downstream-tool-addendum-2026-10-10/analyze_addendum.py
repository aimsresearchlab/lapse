#!/usr/bin/env python3
"""Added-reader analysis: the powered run's analyze_powered.py unchanged, redirected to this folder.

    python analyze_addendum.py gate --kind admission|cellgate --reader R [--age A]
    python analyze_addendum.py sweep --reader R
    python analyze_addendum.py final      # writes results_powered.json here (this cell is its own family)
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
