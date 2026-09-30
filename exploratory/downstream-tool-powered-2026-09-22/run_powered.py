#!/usr/bin/env python3
"""Append-only runner for the powered verification-tool run.

Parser, scoring, trace format, and the per-call request are the R2 runner's
(exploratory/downstream-tool-2026-09-11/run_tool.py), imported unchanged. The only
difference in the request is the provider-routing object, which is taken from
the reader config so that a quantization filter can be pinned (R2 pinned by
provider name only).

    python run_powered.py compile --stage STAGE --reader R [--age A]
    python run_powered.py run --stage STAGE --reader R [--age A] [--dry-run]

Stages: dev (R2 dev slice), admission (fresh + expired-bound@M8 + no-memory),
sweep (calibration-sweep pairs at M1/M3/M8), cellgate (fresh + expired-bound at
the cell age), target, e2e. target and e2e refuse to run unless the confirmatory
freeze file exists, names this reader-age cell, matches the input hashes, and the
cell gate passed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2DIR = HERE.parent / "downstream-tool-2026-09-11"
sys.path.insert(0, str(R2DIR))
import run_tool as R2R  # noqa: E402  (R2 parser/scoring/trace helpers, unchanged)

CONFIG = json.loads((HERE / "config_powered.json").read_text())
GEN = HERE / "generated"
COMPILED = HERE / "compiled"
TRACES = HERE / "traces"


def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def stage_rows(stage, age):
    if stage == "dev":
        rows = [r for r in load(R2DIR / "dev/probe_r2.jsonl") if r["reader_config_id"] == "deepseek-v41-firstparty-nonthinking"]
        keep = ("model", "provider", "reader_config_id", "served_model_expectation", "reasoning", "temperature",
                "max_output_tokens", "input_tokens_estimate", "frozen_call_order", "call_order")
        return [{k: v for k, v in r.items() if k not in keep} for r in rows]
    if stage == "admission":
        return [r for r in load(GEN / "controls.jsonl") if (r["control"], r["age_tag"]) in
                {("fresh", "TODAY"), ("expired_bounded", "M8"), ("no_memory", "M8")}]
    if stage == "sweep":
        return load(GEN / "sweep.jsonl")
    if stage == "cellgate":
        return [r for r in load(GEN / "controls.jsonl") if (r["control"], r["age_tag"]) in
                {("fresh", "TODAY"), ("expired_bounded", age)}]
    if stage == "target":
        return [r for r in load(GEN / "targets.jsonl") if r["age_tag"] == age]
    if stage == "e2e":
        return [r for r in load(GEN / "e2e.jsonl") if r["age_tag"] == age]
    raise ValueError(stage)


def tag(stage, reader, age): return f"{stage}-{reader}" + (f"-{age}" if age and stage in ("cellgate", "target", "e2e") else "")


def compile_stage(stage, reader, age):
    cfg = CONFIG["readers"][reader]
    out = []
    for r in stage_rows(stage, age):
        row = dict(r)
        row.update({"stage": "control" if stage in ("dev", "admission", "cellgate") else "target", "powered_stage": stage,
                    "reader_config_id": reader, "model": cfg["model"], "provider": cfg["provider"]["only"][0],
                    "provider_routing": cfg["provider"], "temperature": CONFIG["temperature"],
                    "max_output_tokens": CONFIG["max_output_tokens"],
                    "input_tokens_estimate": math.ceil(len(R2R.canonical_json({"messages": r["messages"], "tools": r["tools"]})) / 3) + 64})
        if "reasoning" in cfg:
            row["reasoning"] = cfg["reasoning"]
        out.append(row)
    out.sort(key=lambda x: hashlib.sha256(f"powered-v1/call-order:{reader}:{x['id']}".encode()).hexdigest())
    COMPILED.mkdir(exist_ok=True)
    path = COMPILED / f"{tag(stage, reader, age)}.jsonl"
    path.write_text("".join(R2R.canonical_json(x) + "\n" for x in out))
    return path, out


def recorded_spend():
    total = 0.0
    for p in TRACES.glob("*.jsonl"):
        for r in load(p):
            c = (r.get("usage") or {}).get("cost")
            if isinstance(c, (int, float)):
                total += c
            elif isinstance(r.get("cost_usd"), (int, float)):
                total += r["cost_usd"]
    return total


def check_confirmatory(stage, reader, age, input_hash):
    frz_path = HERE / "freeze_confirmatory.json"
    if not frz_path.is_file():
        raise SystemExit("refused: freeze_confirmatory.json missing")
    frz = json.loads(frz_path.read_text())
    cell = f"{reader}|{age}"
    if cell not in frz["cells"]:
        raise SystemExit(f"refused: {cell} is not a frozen confirmatory cell")
    if frz["compiled_hashes"].get(f"{stage}|{cell}") != input_hash:
        raise SystemExit("refused: compiled input hash differs from the freeze")
    gate = json.loads((HERE / "gates" / f"cellgate-{reader}-{age}.json").read_text())
    if gate.get("passed") is not True:
        raise SystemExit(f"refused: cell gate not passed for {cell}")
    return {"freeze_file": str(frz_path), "freeze_hash": R2R.sha256_file(frz_path), "cell": cell,
            "gate_file": f"gates/cellgate-{reader}-{age}.json", "gate_hash": R2R.sha256_file(HERE / "gates" / f"cellgate-{reader}-{age}.json")}


RATE_LIMIT_ATTEMPTS = 6


def allowed(item_id, prior, max_attempts):
    """Deviation D2: attempts that failed with HTTP 429 (no model output) may be retried up to 6 times in total;
    any other error keeps the R2 rule of one retry (2 attempts)."""
    errs = [r["error"] for r in prior if r["item_id"] == item_id and r.get("error")]
    if errs and all("429" in e for e in errs):
        return RATE_LIMIT_ATTEMPTS
    return max_attempts


def run_stage(stage, reader, age, dry_run, max_attempts=2):
    path = COMPILED / f"{tag(stage, reader, age)}.jsonl"
    if not path.is_file():
        raise SystemExit(f"compile first: {path}")
    items = [R2R.validate_item(x) for x in load(path)]
    input_hash = R2R.sha256_file(path)
    freeze = check_confirmatory(stage, reader, age, input_hash) if stage in ("target", "e2e") else {}
    trace = TRACES / f"{tag(stage, reader, age)}.jsonl"
    prior = load(trace) if trace.exists() else []
    done = {r["item_id"] for r in prior if r.get("error") is None}
    attempts = {}
    for r in prior:
        attempts[r["item_id"]] = attempts.get(r["item_id"], 0) + 1
    todo = [x for x in items if x["id"] not in done and attempts.get(x["id"], 0) < allowed(x["id"], prior, max_attempts)]
    prices = {m: {"input_per_million": p["input_per_million"], "output_per_million": p["output_per_million"]}
              for m, p in CONFIG["price_snapshot"]["prices"].items()}
    # expected cost: mean billed cost per call of this reader so far (any stage), else worst case/4
    per_call = [((r.get("usage") or {}).get("cost")) for p in TRACES.glob(f"*-{reader}*.jsonl") for r in load(p)]
    per_call = [c for c in per_call if isinstance(c, (int, float))]
    worst = R2R.estimate_cost(todo, prices)["estimated_cost_usd"] if todo else 0.0
    expected = (1.5 * sum(per_call) / len(per_call) * len(todo)) if per_call else worst / 4
    spent = recorded_spend()
    summary = {"stage": stage, "reader": reader, "age": age, "input": str(path), "input_hash": input_hash, "items": len(items),
               "todo": len(todo), "recorded_spend_usd": round(spent, 6), "expected_stage_usd": round(expected, 6),
               "worst_case_stage_usd": round(worst, 6), "cap_usd": CONFIG["hard_cost_cap_usd"]}
    print(json.dumps(summary))
    if spent + expected > CONFIG["hard_cost_cap_usd"]:
        raise SystemExit("refused: recorded spend + expected stage cost exceeds the hard cap")
    if dry_run or not todo:
        return
    from openai import OpenAI
    client = OpenAI(base_url=CONFIG["base_url"], api_key=os.environ[CONFIG["api_key_env"]])
    stop = {"flag": False}

    def one(item):
        if stop["flag"]:
            return None
        started = time.time()
        rec = {"attempt_id": str(uuid.uuid4()), "timestamp_unix": started, "stage": item["stage"], "powered_stage": stage,
               "item_id": item["id"], "item_sha256": hashlib.sha256(R2R.canonical_json(item).encode()).hexdigest(), "item": item,
               "run_identity": {"input_hash": input_hash, "freeze_hash": freeze.get("freeze_hash")}, "freeze": freeze,
               "reader_config_id": reader, "configured_model": item["model"], "provider": item["provider"],
               "served_model": None, "generation_id": None, "usage": {}, "cost_usd": None, "error": None, "score": "UNPARSED",
               "first_tool_name": None, "tool_call_count": 0, "parsed_arguments": None, "parse_error": None}
        try:
            extra = {"provider": item["provider_routing"]}
            if "reasoning" in item:
                extra["reasoning"] = item["reasoning"]
            resp = client.chat.completions.create(model=item["model"], messages=item["messages"], tools=item["tools"],
                                                  tool_choice="required", temperature=item["temperature"],
                                                  max_tokens=item["max_output_tokens"], extra_body=extra)
            rec["raw_response"] = R2R.raw_response(resp)
            rec.update(R2R.response_metadata(resp, item["model"], item["provider"]))
            rec["served_provider"] = rec["raw_response"].get("provider")
            rec.update(R2R.usage_and_cost(resp, prices.get(item["model"])))
            rec.update(R2R.parse_first_tool_call(resp, item))
        except Exception as exc:  # retained as a failed attempt
            rec["error"] = repr(exc)
        rec["elapsed_seconds"] = round(time.time() - started, 6)
        R2R.append_trace(trace, rec)
        return rec

    for attempt in range(RATE_LIMIT_ATTEMPTS):
        prior = load(trace) if trace.exists() else []
        done = {r["item_id"] for r in prior if r.get("error") is None}
        counts = {}
        for r in prior:
            counts[r["item_id"]] = counts.get(r["item_id"], 0) + 1
        todo = [x for x in items if x["id"] not in done and counts.get(x["id"], 0) < allowed(x["id"], prior, max_attempts)]
        if not todo:
            break
        if attempt:
            time.sleep(min(60, 10 * attempt))
        with cf.ThreadPoolExecutor(CONFIG["readers"][reader].get("workers", CONFIG["workers"])) as ex:
            recs = [r for r in ex.map(one, todo) if r]
        if recorded_spend() > CONFIG["hard_cost_cap_usd"]:
            stop["flag"] = True
            raise SystemExit("hard cap reached; stopping")
    final = load(trace)
    ok = {r["item_id"]: r for r in final if r.get("error") is None}
    errs = [r for r in final if r.get("error")]
    from collections import Counter
    print(json.dumps({"stage": stage, "reader": reader, "age": age, "completed": len(ok), "of": len(items),
                      "error_attempts": len(errs), "scores": dict(Counter(r["score"] for r in ok.values())),
                      "served_providers": dict(Counter(str(r.get("served_provider")) for r in ok.values())),
                      "served_models": dict(Counter(str(r.get("served_model")) for r in ok.values())),
                      "recorded_spend_usd": round(recorded_spend(), 6)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("compile", "run"))
    ap.add_argument("--stage", required=True, choices=("dev", "admission", "sweep", "cellgate", "target", "e2e"))
    ap.add_argument("--reader", required=True, choices=sorted(CONFIG["readers"]))
    ap.add_argument("--age", choices=("M1", "M3", "M8"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage in ("cellgate", "target", "e2e") and not a.age:
        ap.error("--age required for this stage")
    if a.cmd == "compile":
        path, rows = compile_stage(a.stage, a.reader, a.age)
        print(json.dumps({"compiled": str(path), "rows": len(rows), "sha256": R2R.sha256_file(path)}))
    else:
        run_stage(a.stage, a.reader, a.age, a.dry_run)


if __name__ == "__main__":
    main()
