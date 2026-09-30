import hashlib
import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).with_name("run_tool.py")
SPEC = importlib.util.spec_from_file_location("run_tool", MODULE)
run_tool = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(run_tool)


ITEM = {
    "id": "control-workplace-01", "stage": "control", "model": "reader-x", "provider": "test-provider",
    "messages": [{"role": "user", "content": "test"}], "tools": [
        {"type": "function", "function": {"name": "submit_reimbursement", "description": "Submit a reimbursement.",
         "parameters": {"type": "object", "properties": {"employer": {"type": "string"}, "amount": {"type": "number"}}, "required": ["employer", "amount"]}}},
        {"type": "function", "function": {"name": "request_confirmation", "description": "Ask for a current value.",
         "parameters": {"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"]}}},
    ], "max_output_tokens": 20,
    "input_tokens_estimate": 100, "execute_tool": "submit_reimbursement",
    "verify_tool": "request_confirmation", "witness": "Corvida Analytics", "target_field": "employer",
}


def response(calls, usage=None):
    return {"id": "gen-123", "model": "served-reader-x", "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"message": {"tool_calls": calls}}]}


def call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def test_parse_execute_requires_planted_witness():
    got = run_tool.parse_first_tool_call(response([call("submit_reimbursement", '{"employer":"Corvida Analytics","amount":25}')]), ITEM)
    assert got["score"] == "EXECUTE"
    assert got["first_tool_name"] == "submit_reimbursement"
    wrong = run_tool.parse_first_tool_call(response([call("submit_reimbursement", '{"employer":"Other Co"}')]), ITEM)
    assert wrong["score"] == "UNPARSED"
    assert wrong["parse_error"] == "execute_missing_or_invented_witness"


def test_parse_verify_and_bad_calls_are_visible():
    assert run_tool.parse_first_tool_call(response([call("request_confirmation", '{"field":"employer"}')]), ITEM)["score"] == "VERIFY"
    assert run_tool.parse_first_tool_call(response([call("request_confirmation", '{"field":"car"}')]), ITEM)["parse_error"] == "verify_wrong_field"
    assert run_tool.parse_first_tool_call(response([call("submit_reimbursement", "{")]), ITEM)["parse_error"] == "malformed_arguments"
    assert run_tool.parse_first_tool_call(response([]), ITEM)["parse_error"] == "no_tool_call"
    assert run_tool.parse_first_tool_call(response([call("request_confirmation", '{"field":"employer"}'), call("submit_reimbursement", '{}')]), ITEM)["parse_error"] == "multiple_tool_calls"


def test_item_requires_actual_execute_and_verify_schemas():
    run_tool.validate_item(ITEM)
    invalid = {**ITEM, "tools": ITEM["tools"][:1]}
    try:
        run_tool.validate_item(invalid)
    except ValueError as exc:
        assert "execute and verify" in str(exc)
    else:
        raise AssertionError("missing verify schema was accepted")


def test_builder_system_user_request_derives_messages_and_standard_tools():
    builder_item = {key: value for key, value in ITEM.items() if key not in {"messages", "tools"}}
    builder_item.update(system="memory context", user_request="please act")
    normalized = run_tool.validate_item(builder_item)
    assert normalized["messages"] == [{"role": "system", "content": "memory context"},
                                      {"role": "user", "content": "please act"}]
    assert {tool["function"]["name"] for tool in normalized["tools"]} == {
        "submit_reimbursement", "request_confirmation"}


def test_target_gate_requires_hash_and_passed_calibration(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"frozen":true}')
    gate = tmp_path / "gate.json"
    calibration_trace = tmp_path / "calibration.jsonl"
    calibration_trace.write_text('{"item_id":"control-1"}\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    trace_digest = hashlib.sha256(calibration_trace.read_bytes()).hexdigest()
    passed_reader = {"all_three_pass": True, "fresh_execute_pass": True,
                     "expired_bound_verify_pass": True, "no_memory_verify_pass": True,
                     "tool_parse_pass": True}
    gate.write_text(json.dumps({"freeze_manifest_hash": digest, "calibration_trace_hash": trace_digest,
                                "reader_results": {"reader-x": passed_reader}}))
    result = run_tool.require_target_gate("target", manifest, digest, gate, calibration_trace, {"reader-x"})
    assert result["freeze_hash"] == digest
    try:
        run_tool.require_target_gate("target", manifest, "0" * 64, gate, calibration_trace, {"reader-x"})
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")
    gate.write_text(json.dumps({"freeze_manifest_hash": digest, "calibration_trace_hash": trace_digest,
                                "reader_results": {"reader-x": {**passed_reader, "all_three_pass": False}}}))
    try:
        run_tool.require_target_gate("target", manifest, digest, gate, calibration_trace, {"reader-x"})
    except ValueError as exc:
        assert "all-three" in str(exc)
    else:
        raise AssertionError("failed calibration was accepted")


def test_library_runner_also_refuses_ungated_target(tmp_path):
    try:
        run_tool.run_items([{**ITEM, "stage": "target"}], "target", tmp_path / "trace.jsonl", object())
    except ValueError as exc:
        assert "require_target_gate" in str(exc)
    else:
        raise AssertionError("ungated target run was accepted")


def test_estimate_and_append_only_trace_with_fake_client(tmp_path):
    prices = {"reader-x": {"input_per_million": 2.0, "output_per_million": 4.0}}
    estimate = run_tool.estimate_cost([ITEM], prices)
    assert estimate["calls"] == 1
    assert estimate["estimated_cost_usd"] == 0.00028

    class FakeCreate:
        def create(self, **kwargs):
            assert kwargs["tool_choice"] == "required"
            assert kwargs["extra_body"] == {"provider": {"only": ["test-provider"], "allow_fallbacks": False}}
            return response([call("request_confirmation", '{"field":"employer"}')])
    class FakeClient:
        class chat:
            class completions:
                create = FakeCreate().create
    trace = tmp_path / "trace.jsonl"
    first = run_tool.run_items([ITEM], "control", trace, FakeClient(), prices, input_hash="input-hash", workers=2)
    second = run_tool.run_items([ITEM], "control", trace, FakeClient(), prices, input_hash="input-hash", workers=2)
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(lines) == 2 and len({line["attempt_id"] for line in lines}) == 2
    assert first[0]["score"] == second[0]["score"] == "VERIFY"
    assert lines[0]["served_model"] == "served-reader-x"
    assert lines[0]["generation_id"] == "gen-123"
    assert lines[0]["usage"]["prompt_tokens"] == 10
    assert lines[0]["cost_usd"] == 0.00004
    assert lines[0]["raw_response"]["id"] == "gen-123"
    assert lines[0]["item_sha256"] == hashlib.sha256(run_tool.canonical_json(ITEM).encode()).hexdigest()
    assert lines[0]["item"]["messages"] == ITEM["messages"]


def test_frozen_reasoning_field_is_forwarded_with_provider_pin(tmp_path):
    seen = {}
    class FakeCreate:
        def create(self, **kwargs):
            seen.update(kwargs)
            return response([call("request_confirmation", '{"field":"employer"}')])
    class FakeClient:
        class chat:
            class completions:
                create = FakeCreate().create
    frozen = {**ITEM, "reasoning": {"enabled": False}}
    records = run_tool.run_items([frozen], "control", tmp_path / "trace.jsonl", FakeClient(), input_hash="input-hash")
    assert seen["extra_body"] == {"provider": {"only": ["test-provider"], "allow_fallbacks": False},
                                  "reasoning": {"enabled": False}}
    assert records[0]["item"]["reasoning"] == {"enabled": False}


def test_resume_skips_success_but_requires_explicit_failed_retry_and_cap(tmp_path):
    trace = tmp_path / "trace.jsonl"
    identity = {"input_hash": "frozen-input", "freeze_hash": "frozen-manifest"}
    run_tool.append_trace(trace, {"item_id": ITEM["id"], "error": None, "run_identity": identity, "cost_usd": 0.1})
    failed = {**ITEM, "id": "control-workplace-02"}
    run_tool.append_trace(trace, {"item_id": failed["id"], "error": "Timeout", "run_identity": identity, "cost_usd": 0.2})
    todo, summary = run_tool.resume_items([ITEM, failed], trace, identity, retry_failed=False)
    assert todo == []
    assert summary == {"skipped_success": 1, "skipped_failed": 1}
    todo, summary = run_tool.resume_items([ITEM, failed], trace, identity, retry_failed=True)
    assert [item["id"] for item in todo] == [failed["id"]]
    assert summary == {"skipped_success": 1, "skipped_failed": 0}
    try:
        run_tool.enforce_cost_cap(todo, {"reader-x": {"input_per_million": 2, "output_per_million": 4}}, 0.2,
                                  run_tool.recorded_attempts(trace, identity))
    except ValueError as exc:
        assert "cost cap exceeded" in str(exc)
    else:
        raise AssertionError("cap-exceeding retry was accepted")
