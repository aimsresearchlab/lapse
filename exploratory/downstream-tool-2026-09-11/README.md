# Downstream verification-tool task (2026-09-11 spec; run 2026-09-12)

Design, calibration history, hashes, and results: `research/DOWNSTREAM_TOOL_SPEC_2026-09-11.md`
(§12 freeze, §13 R0, §14 R1 + R2 repair + R2 gate, §15 result).

## Pipeline

1. `build_inputs.py` (v4, revision R2) reads the frozen Graphiti read-time
   inputs, keeps the 69 clean same-lexeme progressive/simple-present source
   facts, emits both arms per pair plus 30 synthetic witness-disjoint
   calibration contents x three controls. Deterministic; no API calls.
2. `freeze_inputs.py --config run_config_r2_2026-09-12.json --output-dir frozen-r2`
   compiles the reader cross-product (three pinned readers), prices, and the
   hashed freeze manifest.
3. `run_tool.py --stage control` runs calibration; `analyze_tool.py calibration`
   writes the gate. `run_tool.py --stage target` refuses to run unless the
   manifest hash, gate, and control trace all bind; `analyze_tool.py target`
   computes the frozen paired statistics.
4. `dev_probe.py` compiles the disclosed 54-call R2 dev slice (never a gate).

Tests: `python3 -m pytest -q .` (24 offline tests).

## Layout

| Path | Revision | Content |
|---|---|---|
| `generated-r0/`, `frozen-r0/`, `trace_control.jsonl`, `calibration_gate_r0.json`, `run_config_2026-09-12.json` | R0 | first calibration; gate FAILED (API + parser) |
| `generated-r1/`, `frozen-r1/`, `trace_control_r1.jsonl`, `calibration_gate_r1.json`, `run_config_r1_2026-09-12.json` | R1 | second calibration; gate FAILED (fresh-execute floor) |
| `dev/` | R2 dev | 54-call instrument probe, 54/54 |
| `generated/`, `frozen-r2/`, `trace_control_r2.jsonl`, `calibration_gate_r2.json`, `run_config_r2_2026-09-12.json` | R2 | gate PASSED 270/270 |
| `trace_target_r2.jsonl`, `target_result_r2.json` | R2 | 414 target calls; frozen analysis; success rule NOT MET |
| `frozen/` | R0 | duplicate of `frozen-r0/` left from the first freeze |

Target notes are byte-identical across R0, R1, and R2; only the prompt,
tools, and calibration slices changed, and every change is recorded in the
spec before the corresponding calls.
