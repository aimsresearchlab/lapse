#!/usr/bin/env python3
"""Compile draft inputs into a reproducible, config-pinned freeze package; never calls an API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(row) + "\n" for row in rows))


def require(config, *keys):
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError("config missing: " + ", ".join(missing))


def validate_config(config):
    require(config, "reader_configs", "thresholds", "success_rule", "saturation_rule", "retry_rule",
            "api_settings", "price_snapshot", "hard_cost_cap_usd", "user_authorization", "no_target_before_gate")
    readers = config["reader_configs"]
    if not isinstance(readers, list) or len(readers) != 3:
        raise ValueError("config must contain exactly three reader_configs")
    ids = set()
    for reader in readers:
        require(reader, "reader_config_id", "model", "provider", "served_model_expectation", "temperature", "max_output_tokens")
        if reader["reader_config_id"] in ids:
            raise ValueError("reader_config_id values must be unique")
        ids.add(reader["reader_config_id"])
        if not isinstance(reader["max_output_tokens"], int) or reader["max_output_tokens"] <= 0:
            raise ValueError("reader max_output_tokens must be positive integers")
    prices = config["price_snapshot"]
    require(prices, "catalog_timestamp", "prices")
    for reader in readers:
        price = prices["prices"].get(reader["model"])
        if not price or not all(isinstance(price.get(key), (int, float)) for key in ("input_per_million", "output_per_million")):
            raise ValueError(f"missing price for {reader['model']}")
    if not isinstance(config["hard_cost_cap_usd"], (int, float)) or config["hard_cost_cap_usd"] <= 0:
        raise ValueError("hard_cost_cap_usd must be positive")
    if config["no_target_before_gate"] is not True:
        raise ValueError("no_target_before_gate must be true")


def conservative_input_tokens(item):
    """Round up a character estimate and add a fixed tool-call envelope."""
    payload = canonical({"messages": item["messages"], "tools": item["tools"]})
    return math.ceil(len(payload) / 3) + 64


def compiled_item(item, reader):
    row = dict(item)
    row.update({"stage": "control" if item["kind"] == "calibration" else "target",
                "reader_config_id": reader["reader_config_id"], "model": reader["model"],
                "provider": reader["provider"], "served_model_expectation": reader["served_model_expectation"],
                "temperature": reader["temperature"], "max_output_tokens": reader["max_output_tokens"],
                "input_tokens_estimate": conservative_input_tokens(item)})
    if "reasoning" in reader:
        row["reasoning"] = reader["reasoning"]
    return row


def cost(items, config):
    prices = config["price_snapshot"]["prices"]
    return sum((row["input_tokens_estimate"] * prices[row["model"]]["input_per_million"] +
                row["max_output_tokens"] * prices[row["model"]]["output_per_million"]) / 1_000_000 for row in items)


def schedule(rows):
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"downstream-tool/frozen-call-order/v1:{row['reader_config_id']}:{row['id']}".encode()).hexdigest())
    for rank, row in enumerate(ordered, 1):
        row["frozen_call_order"] = rank
    return ordered


def compile_freeze(generated_inputs, config_path, output_dir):
    generated_inputs, config_path, output_dir = Path(generated_inputs), Path(config_path), Path(output_dir)
    config = json.loads(config_path.read_text())
    validate_config(config)
    items = load_jsonl(generated_inputs)
    if {item["kind"] for item in items} != {"target", "calibration"}:
        raise ValueError("generated inputs must contain target and calibration items")
    frozen = schedule([compiled_item(item, reader) for item in items for reader in config["reader_configs"]])
    control = [item for item in frozen if item["stage"] == "control"]
    target = [item for item in frozen if item["stage"] == "target"]
    if len(control) != 90 * 3 or len(target) != 138 * 3:
        raise ValueError("unexpected cross-product size")
    estimate = cost(frozen, config)
    if estimate > config["hard_cost_cap_usd"]:
        raise ValueError(f"estimated cost ${estimate:.8f} exceeds hard cap ${config['hard_cost_cap_usd']:.8f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path, target_path, prices_path = output_dir / "control.jsonl", output_dir / "target.jsonl", output_dir / "prices.json"
    write_jsonl(control_path, control)
    write_jsonl(target_path, target)
    prices_path.write_text(json.dumps(config["price_snapshot"]["prices"], indent=2, sort_keys=True) + "\n")
    generated_manifest = generated_inputs.with_name("manifest.json")
    source_path = Path(json.loads(generated_manifest.read_text())["source"])
    manifest = {"freeze_compiler_version": 1, "generated_inputs": str(generated_inputs),
                "generated_manifest": str(generated_manifest), "source_inputs": str(source_path),
                "hashes": {"source_inputs": sha256(source_path), "generated_inputs": sha256(generated_inputs),
                           "generated_manifest": sha256(generated_manifest), "run_config": sha256(config_path),
                           "frozen_control": sha256(control_path), "frozen_target": sha256(target_path),
                           "runner_prices": sha256(prices_path)},
                "reader_configs": config["reader_configs"], "thresholds": config["thresholds"],
                "success_rule": config["success_rule"], "saturation_rule": config["saturation_rule"],
                "retry_rule": config["retry_rule"], "api_settings": config["api_settings"],
                "price_snapshot": config["price_snapshot"], "hard_cost_cap_usd": config["hard_cost_cap_usd"],
                "estimated_cost_usd": estimate, "user_authorization": config["user_authorization"],
                "no_target_before_gate": True, "counts": {"control": len(control), "target": len(target)}}
    (output_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-inputs", type=Path, default=Path(__file__).resolve().parent / "generated/inputs.jsonl")
    parser.add_argument("--config", type=Path, required=True, help="freeze-time JSON config; not supplied by this repository")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_freeze(args.generated_inputs, args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
