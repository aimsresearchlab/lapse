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
