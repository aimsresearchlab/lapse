#!/usr/bin/env python3
"""W&B logging for the powered verification-tool run (online; key from WANDB_API_KEY, never printed).

    python wandb_log.py init
    python wandb_log.py stage <json-file>      # log a gate/sweep JSON as metrics
    python wandb_log.py finish                 # log results_powered.json metrics + artifact, finish
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ID = "downstream-tool-powered-2026-09-22"
KW = dict(entity="sugam-panthi-university-of-southern-mississippi", project="lapse", id=RUN_ID)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flat(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(f"{prefix}/{k}" if prefix else str(k), v, out)
    elif isinstance(obj, bool):
        out[prefix] = int(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = obj
    return out


def init():
    cfg = json.loads((HERE / "config_powered.json").read_text())
    man = json.loads((HERE / "generated/manifest.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    run = wandb.init(**KW, group="downstream-tool-powered", name=RUN_ID, resume="allow", config={
        "spec": "research/DOWNSTREAM_TOOL_POWERED_SPEC_2026-09-22.md", "git_commit": commit, "git_dirty": True,
        "readers": cfg["readers"], "price_snapshot": cfg["price_snapshot"], "temperature": 0, "max_output_tokens": 1024,
        "tool_choice": "required", "seed": None, "bootstrap_seed": 20260922, "ages": man["ages"], "today_in_prompt": man["today"],
        "counts": man["counts"], "input_hashes": man["hashes"], "sources": man["sources"], "hard_cost_cap_usd": cfg["hard_cost_cap_usd"],
        "builder_sha256": sha(HERE / "build_powered.py"), "runner_sha256": sha(HERE / "run_powered.py"),
        "analysis_sha256": sha(HERE / "analyze_powered.py"), "r2_run_tool_sha256": sha(ROOT / "exploratory/downstream-tool-2026-09-11/run_tool.py")})
    run.log({"status/launched": 1})
    print("run url:", run.url)


def stage(path):
    d = json.loads(Path(path).read_text())
    run = wandb.init(**KW, resume="must")
    name = Path(path).stem
    m = flat(name, {k: v for k, v in d.items() if k not in ("trace",)}, {})
    run.log(m); run.summary.update(m)
    print("logged", len(m), "metrics from", name)


def finish():
    run = wandb.init(**KW, resume="must")
    res = json.loads((HERE / "results_powered.json").read_text())
    keep = ("n_pairs", "complete_pairs", "execute", "rate", "flat_only", "prog_only", "p_item_one_sided", "p_item_two_sided",
            "holm_item", "p_cluster_one_sided", "holm_cluster", "contents_net_positive", "contents_net_negative",
            "risk_difference", "rd_ci95_cluster_bootstrap", "rd_missingness_bounds", "holm_e2e_family", "unparsed_or_error", "error_attempts")
    m = {}
    for fam in ("cells", "e2e"):
        for k, v in res[fam].items():
            sub = {kk: vv for kk, vv in v.items() if kk in keep}
            if isinstance(sub.get("rd_ci95_cluster_bootstrap"), list):
                lo, hi = sub.pop("rd_ci95_cluster_bootstrap"); sub["rd_ci_lo"], sub["rd_ci_hi"] = lo, hi
            if isinstance(sub.get("rd_missingness_bounds"), list):
                lo, hi = sub.pop("rd_missingness_bounds"); sub["rd_miss_lo"], sub["rd_miss_hi"] = lo, hi
            flat(f"{fam}/{k.replace('|', '/')}", sub, m)
    m["success"] = int(res["success"]); m["holm_family_size"] = res["holm_family_size"]
    spend = json.loads((HERE / "spend.json").read_text()) if (HERE / "spend.json").exists() else {}
    flat("spend", spend, m)
    run.log(m); run.summary.update(m)
    art = wandb.Artifact("downstream-tool-powered-2026-09-22-results", type="results")
    for f in ("results_powered.json", "freeze_confirmatory.json", "generated/manifest.json", "spend.json"):
        if (HERE / f).exists():
            art.add_file(str(HERE / f), name=f.replace("/", "_"))
    for g in sorted((HERE / "gates").glob("*.json")):
        art.add_file(str(g), name=f"gates_{g.name}")
    run.log_artifact(art)
    run.finish()
    print("finished:", run.url)


if __name__ == "__main__":
    {"init": init, "finish": finish}.get(sys.argv[1], lambda: stage(sys.argv[2]))() if sys.argv[1] != "stage" else stage(sys.argv[2])
