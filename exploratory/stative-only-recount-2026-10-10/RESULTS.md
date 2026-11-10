# Aspect-flip recount and memory-write label check (2026-10-10)

Status: **two annotators (2026-10-10): human-human kappa .75; see `data/memwrite_annotations/agreement.json` and the deviation section of the annotation spec.** Earlier status: provisional, one annotator; the paper now reports both annotators. Cost $0 for the recount; about $1 for the Opus 5.5 labels.

## Why

The headline counts (244:0 confirmatory, 807:2 across 11 configurations) use the frozen
cascade definition: destroyed = `COERCED-STATIVE` or `MANUFACTURE` (`tools/score.py`).
`MANUFACTURE` fires on a currency word ("currently", "right now", "at present",
"presently", "as of now") or on an undated present progressive, so it can fire when the
progressive survives. The human check below tested whether people read those notes as
destroyed.

## Human and LLM check (64 blinded notes, `research/MEMWRITE_ANNOTATION_SPEC_2026-10-09.md`)

Labels: one human (first export plus a Q1-only relabel of 16 notes after the Q1 wording
problem, merged in `data/memwrite_annotations/memwrite_annotations_annotator1_merged.jsonl`,
local only), Seed-OSS-36B-Instruct (the paper's judge; W&B `lapse/8q0gfmal`), Claude Opus
5.5 (OpenRouter, reasoning effort low; W&B `lapse/a469matg`).

Protocol deviation: the unblinding key was opened before the co-author's export, for
headline numbers only (no per-note scorer labels shown to the human annotator). The
co-author remains fully blind.

| Comparison (Q2, same kind of situation) | Agreement | kappa |
|---|---:|---:|
| human vs Opus 5.5 | 59/64 | .78 |
| human vs Seed | 59/64 | .80 |
| Opus 5.5 vs Seed | 60/64 | .82 |
| human, Opus, Seed (Fleiss) | 57/64 all agree | .80 |
| human vs cascade, frozen (stative + manufacture) | 44/64 | .38 |
| human vs cascade, stative only | 60/64 | .83 |
| Opus 5.5 vs cascade, stative only | 61/64 | .85 |
| Seed vs cascade, stative only | 59/64 | .78 |

Q1 (form in notes) after the relabel: human vs Opus 58/64, kappa .80; human vs Seed
51/64, kappa .60.

Stratum counts (human Q2 = CHANGED): simple `PRESERVED` 0/20 (no hidden reversals);
simple `MANUFACTURE` 0/8; progressive `COERCED-STATIVE` 13/13; progressive `MANUFACTURE`
2/12; progressive `PRESERVED` 2/11 (cascade misses on the progressive side). The sample
over-represents `MANUFACTURE` (20/64 vs about 50/758 in the population), so the frozen
kappa understates population-level agreement.

Decision rule from the spec (full 762-note LLM pass only if hidden reversals appear or
stative precision is below about 90%): not triggered.

## Recount (`recount.py`, all 11 configurations, main test)

Pairs and exclusions as in `exploratory/reanalysis-2026-09-12/frame_jackknife.py`. The
frozen column reproduces `tab:allmodels` exactly.

- **frozen**: destroyed = `COERCED-STATIVE` or `MANUFACTURE`.
- **stative only**: destroyed = `COERCED-STATIVE`. The cascade never gives a simple source
  this label, so c = 0 by construction.
- **aspect flip** (symmetric, can show reversals): progressive source destroyed =
  `COERCED-STATIVE`; simple source destroyed = its witness line gains a progressive
  (`PROGRESSIVE` or `PERF_PROG` in `tools/score.py`), dated or not.

| Model | n | frozen b:c | aspect flip b:c |
|---|---:|---:|---:|
| DeepSeek V4 Flash | 128 | 104:0 | 87:0 |
| GPT-5.6 Luna | 128 | 79:0 | 71:0 |
| GLM-5.2 | 125 | 61:0 | 52:0 |
| Claude Sonnet 5 | 128 | 64:2 | 66:0 |
| Qwen3-8B | 128 | 68:0 | 60:0 |
| Mistral Small 3.2 | 127 | 72:0 | 72:0 |
| GPT-OSS-20B | 128 | 85:0 | 50:0 |
| OLMo-2-32B | 128 | 76:0 | 76:0 |
| Gemini 3.1 Pro | 122 | 84:0 | 78:0 |
| Grok 4.6 | 128 | 49:0 | 47:0 |
| Qwen3.8-Max | 127 | 65:0 | 63:0 |
| **Confirmatory three** | 381 | 244:0 | **210:0** |
| **All 11** | 1,397 | 807:2 | **722:0** |

- No writer added the progressive to a simple-present note in any of the 1,397 pairs.
- Both Sonnet reversals and GLM's 8 simple-side destructions (equipment frame) are
  `MANUFACTURE`: "currently" added to a simple-present note. The human annotator marked
  0/8 of the GLM notes as changed.
- Every configuration stays at one-sided p < 1e-14 (unadjusted).
- The count drops about 11% overall; GPT-OSS-20B drops most (85 to 50) because many of its
  notes add "currently".

Caveat: the simple-side check uses the cascade's progressive pattern, which covers the
frame verbs (staying, living, working, taking, driving, using, lecturing) only.

## Planned paper change (after the co-author's labels)

Make the aspect-flip count the headline (210:0 confirmatory, 722:0 across 11; "no
configuration ever added the progressive"). Keep 244:0 and 807:2 in the appendix as the
pre-specified metric, with the human check as the reason. Touches: abstract,
introduction, section 4 text and Table 1, Figure 2a, `tab:allmodels`, conclusion, the
GLM-equipment sentence and Figure 2 dagger note. The frame figure, the instruction table
and the mem0 rates still use the frozen definition: recount them or label the definition.

Already in the paper (2026-10-10, uncommitted): Appendix B `manufacture` definition fixed
to match the code; section 4 sentence "Counting only rewrites to the simple present ...
gave 210:0".
