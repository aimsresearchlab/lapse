import wandb, json
run = wandb.init(
    entity="sugam-panthi-university-of-southern-mississippi",
    project="lapse",
    id="mem0-scale-gemini-3-flash-2026-09-22",
    resume="must",
)
results = json.load(open("results.json"))
metrics = {}
for form, d in results["forms"].items():
    metrics[f"{form}/n"] = d["n"]
    metrics[f"{form}/witness_stored"] = d["witness_stored"]
    metrics[f"{form}/date_anchor"] = d["date_anchor"]
    for k in ("flattened", "flattened_rate", "manufacture", "manufacture_rate", "qual_lost", "qual_lost_rate"):
        if k in d: metrics[f"{form}/{k}"] = d[k]
metrics["n_rows"] = results["n_rows"]
metrics["errors"] = results["errors"]
metrics["overlap_with_0827_n"] = results["overlap_with_0827"]["n"]
metrics["overlap_with_0827_agree"] = results["overlap_with_0827"]["agree"]
metrics["openrouter_usage_total_before"] = 662.278668907
metrics["openrouter_usage_total_after"] = 663.217879207
metrics["openrouter_usage_delta_partial_window"] = 663.217879207 - 662.278668907
metrics["hand_read_n"] = 36
metrics["hand_read_agreement"] = 36
run.log(metrics)
art = wandb.Artifact("mem0-scale-gemini-3-flash-2026-09-22-results", type="results")
art.add_file("results.json")
art.add_file("manifest.json")
art.add_file("handread_sample.json")
run.log_artifact(art)
run.summary.update(metrics)
run.finish()
print("done, url:", run.url)
