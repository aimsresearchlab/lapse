"""Run a frozen analysis script against the relabeled traces, writing outputs here.

    python3 run_frozen.py analysis_v2 | frame_jackknife | frame_sign_test
"""
import importlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = HERE / "root"
which = sys.argv[1]
if which == "analysis_v2":
    sys.path.insert(0, str(REPO / "tools"))
    m = importlib.import_module("analysis_v2"); m.ROOT = ROOT; m.main()
elif which == "frame_jackknife":
    sys.path.insert(0, str(REPO / "exploratory/reanalysis-2026-09-12"))
    m = importlib.import_module("frame_jackknife"); m.ROOT = ROOT; m.OUT = HERE / "out_frame_jackknife"
    m.OUT.mkdir(exist_ok=True); m.main()
elif which == "frame_sign_test":
    sys.path.insert(0, str(REPO / "exploratory/reanalysis-2026-09-12"))
    sys.path.insert(0, str(REPO / "exploratory/frame-sign-test-2026-10-09"))
    fj = importlib.import_module("frame_jackknife"); fj.ROOT = ROOT
    m = importlib.import_module("frame_sign_test")
    for k in ("ROOT",):
        if hasattr(m, k): setattr(m, k, ROOT)
    for k in ("OUT", "HERE"):
        if hasattr(m, k): setattr(m, k, HERE / "out_frame_sign_test")
    (HERE / "out_frame_sign_test").mkdir(exist_ok=True); m.main()
