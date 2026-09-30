#!/usr/bin/env python3
"""Fail-closed offline analysis for the verification-tool task.

This module never calls a model.  It joins a frozen input manifest to an
append-only JSONL trace and refuses to analyse incomplete or ambiguous runs.

Required frozen-manifest fragment (the rest of the manifest is unconstrained)::

    {
      "reader_configurations": [
        {"id": "reader-a", "model": "...", "provider": "..."},
        {"id": "reader-b", "model": "...", "provider": "..."},
        {"id": "reader-c", "model": "...", "provider": "..."}
      ],
      "retry_resolution": {
        "reader-a/DC-workplace-00-FRESH": {
          "selected_attempt_id": "...", "reason": "transient provider error"
        }
      }
    }

There must be exactly three configurations.  A trace record identifies its
configuration with ``configuration_id``/``config_id`` or, only when unique,
by its configured_model/provider pair.  A duplicated (configuration, item)
attempt is rejected unless ``retry_resolution`` explicitly selects one
attempt.  The resolution is itself bound by the manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
VALID_SCORES = {"EXECUTE", "VERIFY"}
CONTROL_EXPECTED = {
    "fresh": ("EXECUTE", 22),
    "expired_bounded": ("VERIFY", 24),
    "no_memory": ("VERIFY", 24),
}
CONFIG_KEYS = ("reader_configurations", "configurations", "reader_configs")


class AnalysisError(ValueError):
    """Raised when a trace cannot support the frozen analysis."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AnalysisError(f"missing file: {path}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid JSONL {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AnalysisError(f"JSON object required at {path}:{number}")
        rows.append(row)
    if not rows:
        raise AnalysisError(f"empty JSONL: {path}")
    return rows


def frozen_configurations(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    supplied = next((manifest[key] for key in CONFIG_KEYS if key in manifest), None)
    if not isinstance(supplied, list) or len(supplied) != 3:
        raise AnalysisError("freeze manifest must contain exactly three reader_configurations")
    configs: dict[str, dict[str, Any]] = {}
    for entry in supplied:
        if not isinstance(entry, dict):
            raise AnalysisError("each frozen reader configuration must be an object")
        legacy_id, compiler_id = entry.get("id"), entry.get("reader_config_id")
        if legacy_id is not None and (not isinstance(legacy_id, str) or not legacy_id):
            raise AnalysisError("frozen reader configuration id must be a non-empty string")
        if compiler_id is not None and (not isinstance(compiler_id, str) or not compiler_id):
            raise AnalysisError("frozen reader configuration reader_config_id must be a non-empty string")
        if legacy_id is not None and compiler_id is not None and legacy_id != compiler_id:
            raise AnalysisError("frozen reader configuration id and reader_config_id must agree")
        config_id = compiler_id if compiler_id is not None else legacy_id
        if not config_id:
            raise AnalysisError("each frozen reader configuration needs id or reader_config_id")
        if config_id in configs:
            raise AnalysisError(f"duplicate frozen configuration id: {config_id}")
        configs[config_id] = {**entry, "id": config_id}
    return configs


def record_config_id(record: dict[str, Any], configs: dict[str, dict[str, Any]]) -> str:
    embedded = record.get("item") if isinstance(record.get("item"), dict) else {}
    explicit = record.get("configuration_id", record.get("config_id",
                          record.get("reader_config_id", embedded.get("reader_config_id"))))
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in configs:
            raise AnalysisError(f"trace has unknown configuration_id: {explicit!r}")
        config = configs[explicit]
        for record_key, config_keys in (("configured_model", ("model", "configured_model")),
                                        ("provider", ("provider",))):
            expected = next((config[key] for key in config_keys if key in config), None)
            actual = record.get(record_key)
            if expected is not None and actual != expected:
                raise AnalysisError(
                    f"configuration {explicit} has {record_key}={actual!r}, expected {expected!r}"
                )
        return explicit

    model, provider = record.get("configured_model"), record.get("provider")
    matches = [identifier for identifier, config in configs.items()
               if config.get("model", config.get("configured_model")) == model
               and config.get("provider") == provider]
    if len(matches) != 1:
        raise AnalysisError(
            "trace record needs configuration_id/config_id when model/provider do not identify exactly one frozen configuration"
        )
    return matches[0]


def _selected_attempt_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        selected = value.get("selected_attempt_id")
        if isinstance(selected, str) and selected:
            return selected
    return None


def select_final_attempts(
    trace: Iterable[dict[str, Any]], expected_item_ids: set[str], stage: str,
    configs: dict[str, dict[str, Any]], manifest: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Select one frozen final attempt per (configuration, item), or fail."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attempt_ids: set[str] = set()
    for record in trace:
        if record.get("stage") != stage:
            raise AnalysisError(f"trace stage mismatch: expected {stage!r}, got {record.get('stage')!r}")
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or item_id not in expected_item_ids:
            raise AnalysisError(f"trace contains unknown or missing item_id: {item_id!r}")
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise AnalysisError(f"trace attempt for {item_id} lacks attempt_id")
        if attempt_id in attempt_ids:
            raise AnalysisError(f"duplicate attempt_id in append-only trace: {attempt_id}")
        attempt_ids.add(attempt_id)
        config_id = record_config_id(record, configs)
        grouped[(config_id, item_id)].append(record)

    retry_resolution = manifest.get("retry_resolution", {})
    if not isinstance(retry_resolution, dict):
        raise AnalysisError("retry_resolution must be an object when supplied")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    used_resolution_keys: set[str] = set()
    for config_id in configs:
        for item_id in expected_item_ids:
            key = (config_id, item_id)
            attempts = grouped.get(key, [])
            display_key = f"{config_id}/{item_id}"
            if not attempts:
                raise AnalysisError(f"missing final attempt: {display_key}")
            if len(attempts) == 1:
                selected[key] = attempts[0]
                continue
            resolution = retry_resolution.get(display_key)
            attempt_id = _selected_attempt_id(resolution)
            if not attempt_id:
                raise AnalysisError(f"duplicate final attempts for {display_key}; frozen retry_resolution is required")
            matching = [row for row in attempts if row["attempt_id"] == attempt_id]
            if len(matching) != 1:
                raise AnalysisError(f"retry_resolution for {display_key} does not select one recorded attempt")
            selected[key] = matching[0]
            used_resolution_keys.add(display_key)
    unused = set(retry_resolution) - used_resolution_keys
    if unused:
        raise AnalysisError(f"retry_resolution names non-duplicate or unknown attempts: {sorted(unused)}")
    return selected


def is_valid(record: dict[str, Any]) -> bool:
    return record.get("error") in (None, "") and record.get("score") in VALID_SCORES


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """Exact binomial McNemar p-value; b/c are the two discordant directions."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = b + c
    if not discordant:
        return 1.0
    lower = min(b, c)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm--Bonferroni adjusted p-values, with deterministic tie handling."""
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, p_values[key] * (count - index)))
        adjusted[key] = running
    return adjusted


def _binding(input_path: Path, trace_path: Path, manifest_path: Path,
             configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_sha256": sha256_file(input_path),
        "trace_sha256": sha256_file(trace_path),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "configuration_ids": list(configs),
    }


def _logical_items(rows: Iterable[dict[str, Any]], kind: str,
                   configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse compiler-expanded rows only after proving the full cross-product."""
    selected = [row for row in rows if row.get("kind") == kind]
    if not selected:
        raise AnalysisError(f"no frozen {kind} items")
    expanded = ["reader_config_id" in row for row in selected]
    if any(expanded) and not all(expanded):
        raise AnalysisError(f"{kind} input mixes compiler-expanded and generic rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise AnalysisError(f"{kind} item lacks a non-empty id")
        grouped[item_id].append(row)
    if all(expanded):
        expected_ids = set(configs)
        expected_stage = "control" if kind == "calibration" else "target"
        for item_id, copies in grouped.items():
            if {copy.get("reader_config_id") for copy in copies} != expected_ids or len(copies) != len(configs):
                raise AnalysisError(f"compiler-expanded {kind} item lacks exactly one row per frozen reader: {item_id}")
            if {copy.get("stage") for copy in copies} != {expected_stage}:
                raise AnalysisError(f"compiler-expanded {kind} item has wrong stage: {item_id}")
    elif any(len(copies) != 1 for copies in grouped.values()):
        raise AnalysisError(f"generic {kind} input has duplicate ids")
    return {item_id: copies[0] for item_id, copies in grouped.items()}


def _calibration_items(rows: Iterable[dict[str, Any]],
                       configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = _logical_items(rows, "calibration", configs)
    if None in items or len(items) != 90:
        raise AnalysisError("calibration requires exactly 90 unique frozen items")
    observed = Counter(row.get("control") for row in items.values())
    if observed != Counter({"fresh": 30, "expired_bounded": 30, "no_memory": 30}):
        raise AnalysisError("calibration requires exactly 30 fresh, 30 expired_bounded, and 30 no_memory items")
    frames = {row.get("frame") for row in items.values()}
    if len(frames) != 6 or None in frames:
        raise AnalysisError("calibration requires exactly six named frames")
    cells = Counter((row["frame"], row["control"]) for row in items.values())
    if any(cells[(frame, control)] != 5 for frame in frames for control in CONTROL_EXPECTED):
        raise AnalysisError("calibration requires exactly five items in every frame x control cell")
    for row in items.values():
        expected, _ = CONTROL_EXPECTED[row["control"]]
        if row.get("expected_first_tool") != expected:
            raise AnalysisError(f"calibration item {row['id']} has a mismatched expected_first_tool")
    return items


def analyze_calibration(input_path: Path, trace_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, configs = read_json(manifest_path), frozen_configurations(read_json(manifest_path))
    items = _calibration_items(read_jsonl(input_path), configs)
    final = select_final_attempts(read_jsonl(trace_path), set(items), "control", configs, manifest)
    results: dict[str, Any] = {}
    all_failures: list[str] = []
    for config_id in configs:
        rows = [(item, final[(config_id, item_id)]) for item_id, item in sorted(items.items())]
        parsed = sum(is_valid(record) for _, record in rows)
        controls: dict[str, Any] = {}
        failures: list[str] = []
        for control, (expected, threshold) in CONTROL_EXPECTED.items():
            cell_rows = [(item, record) for item, record in rows if item["control"] == control]
            correct = sum(record.get("score") == expected and record.get("error") in (None, "")
                          for _, record in cell_rows)
            controls[control] = {"expected": expected, "correct": correct, "n": len(cell_rows),
                                 "threshold": threshold, "passed": correct >= threshold}
            if correct < threshold:
                failures.append(f"{control} {correct}/30 < {threshold}/30")
        frame_cells: dict[str, Any] = {}
        for frame in sorted({item["frame"] for item in items.values()}):
            for control, (expected, _) in CONTROL_EXPECTED.items():
                cell_rows = [(item, record) for item, record in rows
                             if item["frame"] == frame and item["control"] == control]
                correct = sum(record.get("score") == expected and record.get("error") in (None, "")
                              for _, record in cell_rows)
                key = f"{frame}/{control}"
                frame_cells[key] = {"correct": correct, "n": len(cell_rows), "threshold": 3,
                                    "passed": correct >= 3}
                if correct < 3:
                    failures.append(f"{key} {correct}/5 < 3/5")
        if parsed < 86:
            failures.append(f"valid parsed {parsed}/90 < 86/90")
        fresh_execute_pass = controls["fresh"]["passed"]
        expired_bound_verify_pass = controls["expired_bounded"]["passed"]
        no_memory_verify_pass = controls["no_memory"]["passed"]
        tool_parse_pass = parsed >= 86
        # The runner's compatibility field is deliberately stricter than its
        # name: an otherwise-passing aggregate cannot hide a failed 3/5 frame
        # cell in the frozen calibration matrix.
        all_three_pass = (fresh_execute_pass and expired_bound_verify_pass
                          and no_memory_verify_pass and tool_parse_pass
                          and all(cell["passed"] for cell in frame_cells.values()))
        results[config_id] = {
            "valid_parsed": parsed, "n": 90, "valid_parsed_threshold": 86,
            "controls": controls, "frame_control_cells": frame_cells,
            "fresh_execute_pass": fresh_execute_pass,
            "expired_bound_verify_pass": expired_bound_verify_pass,
            "no_memory_verify_pass": no_memory_verify_pass,
            "tool_parse_pass": tool_parse_pass,
            "all_three_pass": all_three_pass,
            "passed": not failures, "failures": failures,
        }
        all_failures.extend(f"{config_id}: {failure}" for failure in failures)
    binding = _binding(input_path, trace_path, manifest_path, configs)
    return {
        "analysis": "verification-tool-calibration-gate", "schema_version": SCHEMA_VERSION,
        "stage": "control",
        "binding": binding, "reader_results": results, "passed": not all_failures,
        # Flat compatibility fields consumed by run_tool.require_target_gate.
        # The duplicate binding values keep the richer audit self-describing.
        "freeze_manifest_hash": binding["freeze_manifest_sha256"],
        "calibration_trace_hash": binding["trace_sha256"],
        "failures": all_failures,
    }


def _target_items(rows: Iterable[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = _logical_items(rows, "target", configs)
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items.values():
        if row.get("form") not in {"progressive", "simple"} or not isinstance(row.get("pair_id"), str):
            raise AnalysisError(f"target item {row.get('id')} needs pair_id and progressive/simple form")
        pairs[row["pair_id"]].append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row["form"] for row in pair} != {"progressive", "simple"}:
            raise AnalysisError(f"target pair {pair_id} must contain exactly one progressive and one simple item")
    return items


def _arm_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    execute = sum(record.get("score") == "EXECUTE" and record.get("error") in (None, "") for record in rows)
    valid = sum(is_valid(record) for record in rows)
    return {
        "n": total,
        "execute": execute,
        "verify": sum(record.get("score") == "VERIFY" and record.get("error") in (None, "") for record in rows),
        "errors": sum(record.get("error") not in (None, "") for record in rows),
        "unparsed": sum(record.get("score") == "UNPARSED" for record in rows),
        "execute_missing_or_invented_witness": sum(
            record.get("parse_error") == "execute_missing_or_invented_witness" for record in rows
        ),
        "valid_parsed": valid,
        "execute_rate_all_final": execute / total,
        "execute_rate_valid_parsed": (execute / valid if valid else None),
        "saturation_flag": execute / total < 0.10 or execute / total > 0.90,
    }


def _binary_execute(record: dict[str, Any]) -> int:
    return int(is_valid(record) and record.get("score") == "EXECUTE")


def _target_reader_result(config_id: str, pairs: dict[str, dict[str, str]],
                          final: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    flat_records = [final[(config_id, forms["simple"])] for forms in pairs.values()]
    progressive_records = [final[(config_id, forms["progressive"])] for forms in pairs.values()]
    complete: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (final[(config_id, forms["simple"])], final[(config_id, forms["progressive"])])
        for forms in pairs.values()
        if is_valid(final[(config_id, forms["simple"])]) and is_valid(final[(config_id, forms["progressive"])])
    ]
    b = sum(_binary_execute(flat) == 1 and _binary_execute(progressive) == 0 for flat, progressive in complete)
    c = sum(_binary_execute(flat) == 0 and _binary_execute(progressive) == 1 for flat, progressive in complete)
    risk_difference = (sum(_binary_execute(flat) - _binary_execute(progressive) for flat, progressive in complete)
                       / len(complete) if complete else None)
    all_pairs = [(final[(config_id, forms["simple"])], final[(config_id, forms["progressive"])])
                 for forms in pairs.values()]
    # Worst case for the proposed positive effect: unknown simple=0, unknown progressive=1.
    lower = sum(
        (_binary_execute(flat) if is_valid(flat) else 0)
        - (_binary_execute(progressive) if is_valid(progressive) else 1)
        for flat, progressive in all_pairs
    ) / len(all_pairs)
    upper = sum(
        (_binary_execute(flat) if is_valid(flat) else 1)
        - (_binary_execute(progressive) if is_valid(progressive) else 0)
        for flat, progressive in all_pairs
    ) / len(all_pairs)
    return {
        "configuration_id": config_id,
        "pair_count": len(all_pairs), "complete_pair_count": len(complete),
        "flat_minus_progressive_execute_risk_difference": risk_difference,
        "discordant_flat_execute_progressive_not": b,
        "discordant_flat_not_progressive_execute": c,
        "discordant_total": b + c,
        "mcnemar_two_sided_p": exact_mcnemar_two_sided(b, c),
        "conservative_missingness_risk_difference": {"lower": lower, "upper": upper},
        "arms": {"flat": _arm_summary(flat_records), "progressive": _arm_summary(progressive_records)},
    }


def _validate_calibration_gate(gate_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    gate = read_json(gate_path)
    if (gate.get("analysis") != "verification-tool-calibration-gate" or gate.get("stage") != "control"
            or gate.get("passed") is not True):
        raise AnalysisError("target analysis requires a passed verification-tool calibration gate")
    gate_binding = gate.get("binding")
    if not isinstance(gate_binding, dict):
        raise AnalysisError("calibration gate lacks a binding object")
    for key in ("freeze_manifest_sha256", "configuration_ids"):
        if gate_binding.get(key) != binding[key]:
            raise AnalysisError(f"calibration gate {key} does not match the frozen target analysis")
    return gate


def analyze_target(input_path: Path, trace_path: Path, manifest_path: Path,
                   calibration_gate_path: Path) -> dict[str, Any]:
    manifest, configs = read_json(manifest_path), frozen_configurations(read_json(manifest_path))
    binding = _binding(input_path, trace_path, manifest_path, configs)
    gate = _validate_calibration_gate(calibration_gate_path, binding)
    items = _target_items(read_jsonl(input_path), configs)
    final = select_final_attempts(read_jsonl(trace_path), set(items), "target", configs, manifest)
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for item in items.values():
        pairs[item["pair_id"]][item["form"]] = item["id"]
    results = {config_id: _target_reader_result(config_id, pairs, final) for config_id in configs}
    raw = {config_id: result["mcnemar_two_sided_p"] for config_id, result in results.items()}
    adjusted = holm_adjust(raw)
    positive = reverse = 0
    for config_id, result in results.items():
        difference = result["flat_minus_progressive_execute_risk_difference"]
        result["holm_adjusted_p"] = adjusted[config_id]
        result["holm_significant"] = adjusted[config_id] < 0.05
        result["positive_holm_significant"] = bool(result["holm_significant"] and difference is not None and difference > 0)
        result["reverse_holm_significant"] = bool(result["holm_significant"] and difference is not None and difference < 0)
        positive += result["positive_holm_significant"]
        reverse += result["reverse_holm_significant"]
    saturation = [
        f"{config_id}/{form}" for config_id, result in results.items()
        for form, arm in result["arms"].items() if arm["saturation_flag"]
    ]
    return {
        "analysis": "verification-tool-target-analysis", "schema_version": SCHEMA_VERSION,
        "binding": {**binding, "calibration_gate_sha256": sha256_file(calibration_gate_path),
                    "calibration_trace_sha256": gate["binding"].get("trace_sha256")},
        "reader_results": results,
        "success": positive >= 2 and reverse == 0,
        "positive_holm_significant_readers": positive,
        "significant_reverse_readers": reverse,
        "target_saturation_flags": saturation,
        "note": "No pooled p-value is calculated; the three fixed reader configurations are the multiplicity family.",
    }


def emit(result: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise AnalysisError(f"refusing to overwrite existing analysis output: {output}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibration", "target"))
    parser.add_argument("--input", type=Path, required=True, help="frozen JSONL containing calibration and/or target items")
    parser.add_argument("--trace", type=Path, required=True, help="append-only JSONL trace for this stage")
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path, help="required for target analysis")
    parser.add_argument("--output", type=Path, help="new JSON result path; refuses to overwrite")
    args = parser.parse_args()
    try:
        if args.stage == "calibration":
            result = analyze_calibration(args.input, args.trace, args.freeze_manifest)
        else:
            if args.calibration_gate is None:
                raise AnalysisError("target analysis requires --calibration-gate")
            result = analyze_target(args.input, args.trace, args.freeze_manifest, args.calibration_gate)
        emit(result, args.output)
    except AnalysisError as exc:
        print(f"analysis refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
