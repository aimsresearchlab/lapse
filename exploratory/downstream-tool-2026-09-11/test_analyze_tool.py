import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import build_inputs
import freeze_inputs


MODULE = Path(__file__).with_name("analyze_tool.py")
SPEC = importlib.util.spec_from_file_location("analyze_tool", MODULE)
analyze_tool = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(analyze_tool)

RUNNER_MODULE = Path(__file__).with_name("run_tool.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("run_tool_for_analysis_test", RUNNER_MODULE)
run_tool = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(run_tool)


CONFIGS = [
    {"id": "reader-a", "model": "model-a", "provider": "provider-a"},
    {"id": "reader-b", "model": "model-b", "provider": "provider-b"},
    {"id": "reader-c", "model": "model-c", "provider": "provider-c"},
]
FRAMES = ["workplace", "vehicle", "class", "equipment", "affiliate_role", "project"]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def manifest(tmp_path, retry_resolution=None):
    path = tmp_path / "freeze.json"
    write_json(path, {"reader_configurations": CONFIGS, "retry_resolution": retry_resolution or {}})
    return path


def calibration_items():
    rows = []
    for frame in FRAMES:
        for cluster in range(5):
            for control, expected in (("fresh", "EXECUTE"), ("expired_bounded", "VERIFY"), ("no_memory", "VERIFY")):
                rows.append({"id": f"DC-{frame}-{cluster}-{control}", "kind": "calibration", "frame": frame,
                             "control": control, "expected_first_tool": expected})
    return rows


def trace_for(items, stage, config, scorer):
    rows = []
    for number, item in enumerate(items):
        score, error, parse_error = scorer(item, config["id"])
        rows.append({"attempt_id": f"{config['id']}-{number}", "configuration_id": config["id"],
                     "configured_model": config["model"], "provider": config["provider"], "stage": stage,
                     "item_id": item["id"], "score": score, "error": error, "parse_error": parse_error})
    return rows


def good_control_scorer(item, _):
    return item["expected_first_tool"], None, None


def test_calibration_gate_passes_and_binds_exact_artifacts(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    trace = tmp_path / "control-trace.jsonl"
    frozen = manifest(tmp_path)
    rows = calibration_items()
    write_jsonl(inputs, rows)
    write_jsonl(trace, [record for config in CONFIGS for record in trace_for(rows, "control", config, good_control_scorer)])

    result = analyze_tool.analyze_calibration(inputs, trace, frozen)
    assert result["passed"] is True
    assert result["binding"]["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert result["binding"]["freeze_manifest_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert result["binding"]["configuration_ids"] == [config["id"] for config in CONFIGS]
    assert result["stage"] == "control"
    assert result["freeze_manifest_hash"] == result["binding"]["freeze_manifest_sha256"]
    assert result["calibration_trace_hash"] == result["binding"]["trace_sha256"]
    assert all(reader["valid_parsed"] == 90 for reader in result["reader_results"].values())
    assert all(reader["all_three_pass"] is True for reader in result["reader_results"].values())

    gate = tmp_path / "real-gate.json"
    write_json(gate, result)
    accepted = run_tool.require_target_gate(
        "target", frozen, result["freeze_manifest_hash"], gate, trace,
        {config["id"] for config in CONFIGS},
    )
    assert accepted["calibration_trace_hash"] == result["calibration_trace_hash"]


def test_calibration_gate_fails_for_frame_cell_and_parsing_threshold(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    trace = tmp_path / "control-trace.jsonl"
    frozen = manifest(tmp_path)
    rows = calibration_items()
    write_jsonl(inputs, rows)

    def bad(item, config_id):
        if config_id == "reader-a" and item["frame"] == "vehicle" and item["control"] == "fresh":
            return "VERIFY", None, None
        if config_id == "reader-a" and item["frame"] == "workplace" and item["control"] == "fresh":
            return "UNPARSED", None, "no_tool_call"
        return good_control_scorer(item, config_id)

    write_jsonl(trace, [record for config in CONFIGS for record in trace_for(rows, "control", config, bad)])
    result = analyze_tool.analyze_calibration(inputs, trace, frozen)
    reader = result["reader_results"]["reader-a"]
    assert result["passed"] is False
    assert reader["frame_control_cells"]["vehicle/fresh"]["passed"] is False
    assert reader["controls"]["fresh"]["passed"] is False
    assert reader["fresh_execute_pass"] is False
    assert reader["all_three_pass"] is False


def test_calibration_accepts_freeze_compiler_reader_configs_schema_end_to_end(tmp_path):
    """The analysis consumes the compiler's actual manifest and 270-row control file."""
    source = Path(__file__).resolve().parents[2] / "exploratory/unrecoverability-2026-09-08/reader_g_inputs.jsonl"
    generated = tmp_path / "generated"
    build_inputs.build(source, generated)
    config = {
        "reader_configs": [
            {"reader_config_id": f"reader-{n}", "model": f"model-{n}", "provider": "pinned-provider",
             "served_model_expectation": f"model-{n}", "temperature": 0, "max_output_tokens": 24}
            for n in range(1, 4)
        ],
        "thresholds": {"fresh_execute": 0.7, "verify": 0.7, "parse": 0.95},
        "success_rule": "predeclared", "saturation_rule": "predeclared", "retry_rule": "one logged retry",
        "api_settings": {"endpoint": "freeze-time", "tool_choice": "required"},
        "price_snapshot": {"catalog_timestamp": "2026-09-12T00:00:00Z", "prices": {
            f"model-{n}": {"input_per_million": 1, "output_per_million": 2} for n in range(1, 4)}},
        "hard_cost_cap_usd": 10, "user_authorization": {"authorized": True, "date": "2026-09-12"},
        "no_target_before_gate": True,
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    frozen_dir = tmp_path / "frozen"
    freeze_inputs.compile_freeze(generated / "inputs.jsonl", config_path, frozen_dir)
    control_path, manifest_path = frozen_dir / "control.jsonl", frozen_dir / "freeze_manifest.json"
    control_rows = [json.loads(line) for line in control_path.read_text().splitlines()]
    assert len(control_rows) == 270
    trace = tmp_path / "control-trace.jsonl"
    trace_rows = []
    for number, item in enumerate(control_rows):
        trace_rows.append({"attempt_id": f"compiler-{number}", "stage": "control", "item_id": item["id"],
                           "item": item, "configured_model": item["model"], "provider": item["provider"],
                           "score": item["expected_first_tool"], "error": None, "parse_error": None})
    write_jsonl(trace, trace_rows)
    result = analyze_tool.analyze_calibration(control_path, trace, manifest_path)
    assert result["passed"] is True
    assert result["binding"]["configuration_ids"] == ["reader-1", "reader-2", "reader-3"]


def test_reader_config_id_must_agree_with_legacy_id_when_both_are_present():
    with pytest.raises(analyze_tool.AnalysisError, match="must agree"):
        analyze_tool.frozen_configurations({"reader_configs": [
            {"id": "legacy-a", "reader_config_id": "compiler-a"},
            {"reader_config_id": "reader-b"},
            {"reader_config_id": "reader-c"},
        ]})


def test_duplicate_trace_is_refused_without_frozen_resolution(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    trace = tmp_path / "control-trace.jsonl"
    frozen = manifest(tmp_path)
    rows = calibration_items()
    records = [record for config in CONFIGS for record in trace_for(rows, "control", config, good_control_scorer)]
    duplicate = dict(records[0])
    duplicate["attempt_id"] = "retry-attempt"
    records.append(duplicate)
    write_jsonl(inputs, rows)
    write_jsonl(trace, records)
    with pytest.raises(analyze_tool.AnalysisError, match="duplicate final attempts"):
        analyze_tool.analyze_calibration(inputs, trace, frozen)


def test_explicit_frozen_retry_resolution_selects_one_attempt(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    trace = tmp_path / "control-trace.jsonl"
    rows = calibration_items()
    first = trace_for(rows, "control", CONFIGS[0], good_control_scorer)
    retry = dict(first[0])
    retry["attempt_id"] = "retry-selected"
    retry["score"] = "EXECUTE"
    records = first + [retry]
    records += trace_for(rows, "control", CONFIGS[1], good_control_scorer)
    records += trace_for(rows, "control", CONFIGS[2], good_control_scorer)
    frozen = manifest(tmp_path, {"reader-a/DC-workplace-0-fresh": {"selected_attempt_id": "retry-selected", "reason": "network retry"}})
    write_jsonl(inputs, rows)
    write_jsonl(trace, records)
    assert analyze_tool.analyze_calibration(inputs, trace, frozen)["passed"] is True


def target_items(pair_count=12):
    rows = []
    for number in range(pair_count):
        pair = f"DT-workplace-{number:02d}"
        for form in ("progressive", "simple"):
            rows.append({"id": f"{pair}-{form}", "kind": "target", "pair_id": pair, "form": form})
    return rows


def test_target_reports_paired_mcnemar_holm_and_missingness(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    control_trace = tmp_path / "control-trace.jsonl"
    target_trace = tmp_path / "target-trace.jsonl"
    frozen = manifest(tmp_path)
    controls = calibration_items()
    targets = target_items()
    write_jsonl(inputs, controls + targets)
    write_jsonl(control_trace, [record for config in CONFIGS for record in trace_for(controls, "control", config, good_control_scorer)])
    gate = analyze_tool.analyze_calibration(inputs, control_trace, frozen)
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate)

    def target_scorer(item, config_id):
        if config_id in {"reader-a", "reader-b"}:
            return ("EXECUTE" if item["form"] == "simple" else "VERIFY"), None, None
        if item["id"] == "DT-workplace-00-simple":
            return "UNPARSED", None, "execute_missing_or_invented_witness"
        return "VERIFY", None, None

    write_jsonl(target_trace, [record for config in CONFIGS for record in trace_for(targets, "target", config, target_scorer)])
    result = analyze_tool.analyze_target(inputs, target_trace, frozen, gate_path)
    assert result["success"] is True
    assert "pooled p-value" in result["note"]
    a = result["reader_results"]["reader-a"]
    assert a["discordant_flat_execute_progressive_not"] == 12
    assert a["mcnemar_two_sided_p"] < 0.001
    assert a["holm_significant"] is True
    c = result["reader_results"]["reader-c"]
    assert c["complete_pair_count"] == 11
    assert c["arms"]["flat"]["execute_missing_or_invented_witness"] == 1
    assert c["conservative_missingness_risk_difference"]["lower"] == 0
    assert c["conservative_missingness_risk_difference"]["upper"] > 0


def test_target_refuses_missing_final_attempt_and_gate_manifest_mismatch(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    control_trace = tmp_path / "control-trace.jsonl"
    target_trace = tmp_path / "target-trace.jsonl"
    frozen = manifest(tmp_path)
    controls, targets = calibration_items(), target_items(1)
    write_jsonl(inputs, controls + targets)
    write_jsonl(control_trace, [record for config in CONFIGS for record in trace_for(controls, "control", config, good_control_scorer)])
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, analyze_tool.analyze_calibration(inputs, control_trace, frozen))
    records = [record for config in CONFIGS for record in trace_for(targets, "target", config, lambda *_: ("VERIFY", None, None))]
    write_jsonl(target_trace, records[:-1])
    with pytest.raises(analyze_tool.AnalysisError, match="missing final attempt"):
        analyze_tool.analyze_target(inputs, target_trace, frozen, gate_path)
