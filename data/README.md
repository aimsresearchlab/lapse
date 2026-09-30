# data/

| File | What it is |
|---|---|
| `stimuli_v2.jsonl` | The LAPSE stimulus set (5,916 rows; `split` is `dev` or `test`). `v2` is internal versioning. |
| `manifest_v2.json` | Build manifest for `stimuli_v2.jsonl` (generator version, eval-date, sha). |
| `gold_battery.jsonl` | Synthetic responses with known labels, used to validate the scorer. |
| `gold_battery.judged.local-seed-oss-36b-instruct-pv2.jsonl` | Final judge (Seed-OSS-36B-Instruct, prompt v2) on the battery. |
| `gold_manifest_v1.json` | Sampling manifest for the 250-item human gold sample. |
| `gold_annotations_annotator1.jsonl`, `gold_annotations_annotator2.jsonl` | Blind human labels, 250 items each. |
| `gold_disagreements.jsonl` | The 25 items the two annotators disagreed on. |
| `gold_adjudicated.jsonl` | Final adjudicated labels. |
| `gold_items.judged.local-seed-oss-36b-instruct-pv2.jsonl` | Final judge on the 250 gold items. |
| `witness_contamination_check.json` | Infini-gram counts of each witness string in pretraining corpora. |
| `witness_web_check.json` | Web collision check for witness names. |
| `stimuli_v1.jsonl`, `stimuli_a2.jsonl`, `manifest_v1.json`, `manifest_a2.json` | Earlier pilot builds. The v2 builder imports their definitions and the tests check v2 against them. |

## Stimulus fields

| Field | Meaning |
|---|---|
| `case_id` | Unique row id; also the join key for traces. |
| `history` | Dated conversation excerpt containing the user's statement; goes into the manifest's `system_template`. |
| `query` | The user turn sent to the model. |
| `downstream_query` | Present on memory-writing rows that spawn a second call: the runner feeds the model's own notes back as history and asks this. |
| `arm` | Task. `e1`: write memory notes. `l2`: write memory notes with an instruction to preserve tense and aspect. `behavioral`: act on the fact directly. `explicit`: ask whether the fact still holds. `anchor`: choose to send or to check first. The derived calls are `e2` (use the notes) and `l2e2` (use the guided notes). |
| `form` | Temporal form of the statement: `prog` ("I'm driving"), `simple` ("I drive"), `bound` (explicit end date), `perf_prog` / `perf_sim` (perfect forms, "I've been living"). |
| `frame` | Content domain: lodging, workplace, vehicle, class, household, equipment, affiliate_role, project. |
| `cluster` | Content instance within a frame. Rows sharing `frame` and `cluster` differ only in the manipulated factor. |
| `gap`, `gap_date` | Time between the statement and evaluation: `fresh`, `near`, `boundary`, `expired_soon`, `stale`, `stale_long`; `gap_date` is the session date. |
| `witness`, `wit_token` | The invented entity that marks the target fact (e.g. "the Marbury Residences") and the token the scorer searches for. |
| `fact_noun` | Words the scorer accepts as referring to the fact type. |
| `carrier` | Sentence frame around the statement: `c0_original` or paraphrases `c1`–`c5`. |
| `lexeme_pair` | `same`: progressive and simple use the same verb. `diff`: different verbs (e.g. staying / live). |
| `subject` | `first` ("I'm…") or `third` ("my cousin Dara is…"). |
| `sl_il_code`, `sl_il_contest`, `aktionsart_shift` | Linguistic annotations: stage-level vs individual-level predicate, contested cases, and whether the verb's aspectual class changes across the pair. |
| `bound_month_name` | Month named in `bound` statements. |
| `component` | Which design cell the row belongs to (`e1_primary` is the confirmatory test). Counts are in `manifest_v2.json`. |
| `tier` | `CONF` confirmatory, `SEC` secondary, `EXP` exploratory, `CONTAINMENT` (earlier-build rows, not run). |
| `split` | `dev` or `test` (`cluster % 5 == 4`). |
| `eval_date`, `generator_version` | Build date and builder version. |
