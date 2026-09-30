# LAPSE: aspectual flattening in LLM memory consolidation

Code, data, model outputs, and labels for *Memory Consolidation Flattens the
Temporal Shape of User Facts* ([arXiv:2609.36457](https://arxiv.org/abs/2609.36457)).

A user says "I am driving a Peugeot." The memory writer stores "The user drives
a Peugeot." The progressive told a later reader the fact might not last; the
stored note asserts it as a standing fact. LAPSE (Linguistic Aspect Persistence
and Stability Evaluation) measures this with matched user statements that
differ only in temporal form.

On the three confirmatory writers, the progressive was flattened while its
simple-present match was kept in 244 of 381 matched pairs, and never the
reverse (DeepSeek V4 Flash 104/0 of 128, GPT-5.6 Luna 79/0 of 128, GLM-5.2
61/0 of 125). The installed mem0, Graphiti, and Letta pipelines flatten too.

## Contents

```
data/          LAPSE stimuli (5,916 rows: 4,791 dev, 1,125 test), build manifests,
               scorer-validation gold sets, witness contamination checks
tools/         stimulus builder, model runner, rule-based scorer, judge,
               preregistered and secondary analyses (see tools/README.md)
tests/         scorer regressions and stimulus audits
traces/        11 model columns: raw outputs, scored labels, judge labels
results/       frozen analysis outputs the paper reports
exploratory/   follow-up experiments: installed pipelines, reader tasks,
               verification tool, qualifier ablation, reanalyses
MANIFEST.sha256  SHA-256 of every data file, results file, and script
```

## Install

Python 3.10 or newer. The scorer and analyses use only the standard library;
`openai` is needed to query models and `pytest` to run the tests.

```bash
git clone https://github.com/aimsresearchlab/lapse.git
cd lapse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The follow-up experiments under `exploratory/` have their own dependencies
(mem0, graphiti-core, letta, scipy); their scripts name them at the top.

## Reproduce the paper's numbers

No model calls. Each command prints nothing on success except the test count:

```bash
python3 tools/analysis_v2.py | cmp - results/ANALYSIS_V2_2026-08-27.txt
python3 tools/analysis_v2_secondary.py | cmp - results/ANALYSIS_V2_SECONDARY_2026-08-27.txt
python3 -m pytest -q tests                 # 85 passed
shasum -a 256 -c --quiet MANIFEST.sha256
```

## Use the stimuli

Each row of `data/stimuli_v2.jsonl` is one model call. Fill the manifest's
system template with the row's conversation history and send the row's query
as the user turn:

```python
import json

manifest = json.load(open("data/manifest_v2.json"))
rows = [json.loads(line) for line in open("data/stimuli_v2.jsonl")]

row = next(r for r in rows if r["component"] == "e1_primary")
messages = [
    {"role": "system", "content": manifest["system_template"].format(history=row["history"])},
    {"role": "user", "content": row["query"]},
]
```

Skip rows with `component == "v1_grid_extra"`; they exist only so the tests
can check v2 against the earlier build. Field meanings are in
`data/README.md`.

## Evaluate a new model

The committed stimuli are dated for an evaluation on 2026-08-11. A new run
should rebuild them for the day it runs, so that "today" in the system prompt
matches the session dates. The builder writes into `data/` next to `tools/`,
so work in a copy of `tools/` and keep the reference build intact:

```bash
mkdir -p ~/lapse-run && cp -r tools ~/lapse-run/ && cd ~/lapse-run

# 1. Build stimuli dated today (5,916 rows, deterministic).
python3 tools/build_stimuli_v2.py --eval-date $(date +%F)

# 2. Check the call count (and, on OpenRouter, the projected cost).
python3 tools/run_pilot_v2.py --stimuli data/stimuli_v2.jsonl \
    --manifest data/manifest_v2.json --models <model-id> --dry

# 3. Run. --smoke sends a 289-call dev subset; --full sends all 6,008 calls
#    and requires --i-have-user-approval as a guard against accidental spend.
export OPENROUTER_API_KEY=...
python3 tools/run_pilot_v2.py --stimuli data/stimuli_v2.jsonl \
    --manifest data/manifest_v2.json --models <model-id> \
    --full --i-have-user-approval          # writes traces/run-v2-full-<time>.jsonl

# 4. Score, then report the paper's per-model statistics.
python3 tools/score.py traces/run-v2-full-<time>.jsonl     # writes .scored.jsonl
python3 tools/evaluate.py traces/run-v2-full-<time>.scored.jsonl
```

For a local OpenAI-compatible server (for example `vllm serve`), add
`--base-url http://localhost:8000/v1`; set `LOCAL_API_KEY` if the server
needs one, and add `--disable-thinking` for Qwen3-style chat templates. A run
that stops partway resumes with `--resume traces/<run>.jsonl`.

The first line of `evaluate.py` output is the main result: how many matched
progressive/simple pairs had only the progressive note flattened (`b`) versus
only the simple one (`c`). `score.py` also lists responses its rules could not
label (`RESIDUAL`); these are excluded from the tests. The paper adds a
second, model-based detector (`tools/judge.py`); it is optional for a new
model.

Rerunning a paper model is not expected to reproduce byte-identical
responses. Rescoring a committed raw trace with `tools/score.py` reproduces
the committed `.scored.jsonl` exactly.

## Model columns

| Tier | Model | Rows |
|---|---|---|
| Confirmatory | deepseek/deepseek-v4-flash, openai/gpt-5.6-luna, z-ai/glm-5.2 | 6,008 calls each |
| Open weights | Qwen3-8B, Mistral Small 3.2 24B | 6,008 calls each |
| Extension (estimation only) | gpt-oss-20b, OLMo-2-32B-Instruct | 6,008 calls each |
| Frontier subset | Claude Sonnet 5, Gemini 3.1 Pro, Grok 4.6, Qwen3.8-Max | 736 calls each |

All runs used temperature 0 and the frozen generator v2.0.1.

## Exploratory experiments

| Directory | What it tests |
|---|---|
| `mem0-pipeline-2026-08-27` | Installed mem0 full stack, 64 records |
| `mem0-scale-2026-09-12` | Installed mem0, 128 statements x 4 forms, DeepSeek writer |
| `mem0-scale-gemini-2026-09-22` | Same inputs, Gemini 3 Flash writer |
| `partd-systems-2026-09-02` | Installed Graphiti and Letta, 162 records each |
| `read-time-ladder-2026-09-08` | Dated-note reader task, 1,152 items per reader |
| `unrecoverability-2026-09-08` | Whether readers recover the cue after mem0, Graphiti, or Letta writes |
| `downstream-tool-2026-09-11` | Verification-tool task, rounds R0 to R2 |
| `downstream-tool-age-2026-09-12` | Verification-tool task by note age |
| `downstream-tool-powered-2026-09-22` | Powered verification-tool run with preregistered cells |
| `qualifier-ablation-2026-09-12` | Writer behavior with temporal qualifiers added or removed |
| `reanalysis-2026-09-12` | Zero-cost reanalyses: gap gradient, replicates, jackknife, multiplicity |

Some exploratory manifests and traces record absolute paths from the machine
that produced them. They are left unedited so their hashes still match the
paper's audit ledger.

## Not included

- The sealed answer key for the 250-item human gold sample
  (`data/gold_key_v1.json`). `tools/gold_agreement.py` needs it to recompute
  human-scorer agreement.
- WildChat, LoCoMo, and LongMemEval excerpts used in the natural-text screens
  and the real-utterance writer test. See `DATA_LICENSES.md`.

## Citation

```bibtex
@misc{panthi2026lapse,
  title         = {Memory Consolidation Flattens the Temporal Shape of User Facts},
  author        = {Panthi, Sugam and Yeamin, Muhaiminul and Luo, Siyan and Abdelfattah, Rabab},
  year          = {2026},
  eprint        = {2609.36457},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2609.36457},
  url           = {https://arxiv.org/abs/2609.36457}
}
```
