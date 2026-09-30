# Powered verification-tool run (2026-09-22)

Spec, deviations, freeze blocks, and results:
`research/DOWNSTREAM_TOOL_POWERED_SPEC_2026-09-22.md` (results record:
`research/DOWNSTREAM_TOOL_POWERED_2026-09-22.md`).

Pipeline (no step edits another step's output):

1. `build_powered.py` writes `generated/` (targets, sweep, controls, end-to-end
   pairs, manifest). Imports R2's `build_inputs.py` wrapper and prompt template.
2. `run_powered.py compile|run --stage dev|admission|sweep|cellgate|target|e2e`
   compiles reader-specific inputs into `compiled/` and appends attempts to
   `traces/`. Imports R2's `run_tool.py` parser and scoring. `target` and `e2e`
   refuse to run without `freeze_confirmatory.json` and a passed cell gate.
3. `analyze_powered.py gate|sweep|final` writes `gates/*.json` and
   `results_powered.json`.
4. `power.py` -> `power.json` (spec section 8). `wandb_log.py` logs to W&B
   project `lapse`, run `downstream-tool-powered-2026-09-22`.

Tests: `python3 -m pytest -q .` (offline).
