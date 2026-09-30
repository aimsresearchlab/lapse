# tools/ — reader's guide

These scripts built the LAPSE benchmark, ran the models, scored the outputs,
and computed the paper's main statistics. Their file names and internal
names are kept as they were when they produced the results. Dated records
in `research/`, the specs, and `claims.md` refer to them by these names, so
they are not renamed. This guide gives each script a plain description.

Naming used below:

- **v2** is the dataset and pipeline behind the paper (public name LAPSE).
  **v1** and **A2** are earlier pilot versions; v2 imports their frame
  definitions and carrier templates, so those builders stay in place.
- Task names follow `GLOSSARY.md`: memory-write task (`e1`), memory-use task
  (`e2`), guided memory-write task (`l2`), guided memory-use task (`l2e2`),
  direct-use task (`behavioral`), validity question (`explicit`),
  send-or-check choice (`anchor`). The short IDs appear in code and trace
  fields.
- A **witness** is the unique token that marks the target user fact in a
  stimulus, so a scorer can tell whether a response used that fact.

## Pipeline, in reading order

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1. Build stimuli | `build_stimuli_v2.py` | Builds the LAPSE stimuli: matched user statements that differ only in temporal form, placed in conversation histories. Deterministic; requires `--eval-date`. | `data/stimuli_v2.jsonl`, `data/manifest_v2.json` |
| 1a. | `build_stimuli.py` | v1 pilot builder. Frozen; v2 imports its frame definitions. | `data/stimuli_v1.jsonl` |
| 1b. | `build_stimuli_a2.py` | A2 pilot builder (carrier paraphrases). Frozen; v2 imports its carriers. | `data/stimuli_a2.jsonl` |
| 1c. | `check_contamination.py` | Counts each witness string in pretraining corpora (RedPajama, Dolma) through the infini-gram API. | `data/witness_contamination_check.json` |
| 2. Run models | `run_pilot_v2.py` | Sends v2 stimuli to each model (OpenRouter or a local vLLM server via `--base-url`). `--dry` estimates cost; `--full` needs `--i-have-user-approval`. | `traces/run-v2-*.jsonl` |
| 2a. | `run_pilot.py` | v1 runner. Superseded by `run_pilot_v2.py`. | `traces/run-*.jsonl` |
| 3. Score | `score.py` | Rule-based scorer (the "cascade"). Labels memory notes (did the note drop the temporal cue?) and responses (did the model act on the fact or check first?). | `traces/*.scored.jsonl` |
| 3a. | `judge.py` | Second, model-based scorer for free-text responses. Sees the query, response, and witness only, never the condition. | `traces/*.scored.judged.*.jsonl` |
| 4. Validate scorers | `build_gold_battery.py` | Builds synthetic responses with known labels and measures the cascade's precision and recall on them. | `data/gold_battery.jsonl` |
| 4a. | `judge_battery.py` | Same check for the judge. | report to stdout |
| 4b. | `sample_gold.py` | Draws the 250-item human annotation sample and blinds it for the web annotator. | `web/data/gold_items.js`, `data/gold_key_v1.json` |
| 4c. | `judge_gold.py` | Runs the judge on the 250 blinded items. | `data/gold_items.judged.*.jsonl` |
| 4d. | `gold_agreement.py` | Human-human and human-scorer agreement (Cohen's kappa) from annotator exports. | report, `data/gold_disagreements.jsonl` |
| 5. Analyze | `analysis_v2.py` | Preregistered tests: paired McNemar on progressive vs simple-present notes per model, Holm-corrected; robustness checks. | `research/ANALYSIS_V2_*.txt` |
| 5a. | `analysis_v2_secondary.py` | Descriptive secondary analyses (guided-writing lever, send-or-check choice, gap gradient). Not part of the confirmatory tests. | `research/ANALYSIS_V2_SECONDARY_*.txt` |
| 6. Natural-text screens | `screen_ecological.py` | Finds naturally occurring temporally marked user facts in LongMemEval and LoCoMo. No API calls. | `data/ecological_screen_v1.json` |
| 6a. | `screen_wildchat.py` | Same pattern screen over WildChat-1M user turns. | `data/wildchat_screen_v1.json` |

Each script's top docstring has its usage line and design details.

## Tests

`tests/test_scorer.py` (scorer regressions), `tests/test_stimuli_v2.py` and
`tests/test_stimuli_a2.py` (stimulus audits). Run `python3 -m pytest -q tests`.

## Later experiments

Experiments after the main run (installed mem0/Graphiti/Letta pipelines,
reader tasks, the verification-tool task) keep their own scripts inside
`exploratory/<name>-<date>/`, next to their inputs and traces. The matching
write-up is in `research/`; `research/README.md` is the index.
