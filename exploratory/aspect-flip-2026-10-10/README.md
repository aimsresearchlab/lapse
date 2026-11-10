# Aspect-flip outcome: relabel and recount (2026-10-10, $0)

Why: the memory-write human check (`research/MEMWRITE_ANNOTATION_SPEC_2026-10-09.md`, one annotator
so far) supports the cascade's COERCED-STATIVE label (kappa .83) but not MANUFACTURE on kept
progressives (kappa .38 for the planned outcome). Record: `../stative-only-recount-2026-10-10/RESULTS.md`.
The paper now leads with the aspect-flip outcome; the planned outcome is reported in Appendix B.

- `relabel.py` writes `root/traces/*.scored.jsonl`: memory-write rows (arms e1, l2; forms prog,
  perf_prog, simple) relabeled so that the frozen scripts' DESTROYED set computes the aspect flip
  (progressive source destroyed = COERCED-STATIVE; simple source destroyed = witness line gains a
  progressive). Frozen label kept in `label_frozen`. Exclusions unchanged.
- `run_frozen.py` runs `tools/analysis_v2.py`, `frame_jackknife.py` and `frame_sign_test.py` unchanged
  against `root/` (outputs `out_*`); `analysis_v2_secondary.py` likewise (`out_analysis_v2_secondary.txt`).
- `numbers.py` -> `numbers.json`: main-test tables under both outcomes; asserts the frozen block
  reproduces the paper's previous tables (244:0, 807:2, rates).
- `tables_extra.py` -> `tables_extra.json`: guided-writing, perfect-progressive and same-lexeme numbers
  under both outcomes (frozen block reproduces the previous tables).
- Figures: `../../../figures/aspect-persistence/recompute.py --definition aspect-flip`
  (`data/figure_data_aspect_flip.json`; `--definition frozen` still reproduces `figure_data.json` byte
  for byte); Makefile default `DEFINITION=aspect-flip`.

Key numbers: confirmatory 210:0 (87, 71, 52; Holm 1.9e-26, 8.5e-22, 2.2e-16); all 11 722:0; same-lexeme
194:0; without vehicle and workplace 122:0; frame sign test Holm .012/.016/.016, no negative cell.
Real-utterance and mem0 counts already score loss of the progressive itself and are unchanged.
