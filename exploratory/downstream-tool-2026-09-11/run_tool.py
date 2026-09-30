#!/usr/bin/env python3
"""Append-only runner for the proposed verification-tool task.

This program deliberately contains no experiment items, prices, or provider
identities.  Those are freeze-time inputs.  It accepts OpenAI-compatible
function tools, requires a tool call, and scores only the first (and only)
tool call as EXECUTE, VERIFY, or UNPARSED.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


LABELS = {"EXECUTE", "VERIFY", "UNPARSED"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reader_id(item: dict[str, Any]) -> str:
    return str(item.get("reader_config_id", item["model"]))


def require_target_gate(stage: str, manifest: Path | None, manifest_hash: str | None,
                        calibration: Path | None, calibration_trace: Path | None,
                        readers: set[str]) -> dict[str, str | None]:
    """Reject targets unless the exact freeze and a passed calibration exist."""
    if stage != "target":
        return {"freeze_manifest": None, "freeze_hash": None, "calibration_gate": None,
                "calibration_trace_hash": None}
    if not manifest or not manifest_hash or not calibration or not calibration_trace:
        raise ValueError("target stage requires manifest/hash, calibration gate, and calibration trace")
    if not manifest.is_file():
        raise ValueError(f"freeze manifest does not exist: {manifest}")
    actual = sha256_file(manifest)
    if actual.lower() != manifest_hash.lower():
        raise ValueError(f"freeze hash mismatch: expected {manifest_hash}, got {actual}")
    try:
        gate = json.loads(calibration.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"calibration gate does not exist: {calibration}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration gate is not JSON: {calibration}") from exc
    if gate.get("freeze_manifest_hash") != actual:
        raise ValueError("target stage refused: calibration gate is not bound to this freeze manifest hash")
    if not calibration_trace.is_file():
        raise ValueError(f"calibration trace does not exist: {calibration_trace}")
    trace_hash = sha256_file(calibration_trace)
    if gate.get("calibration_trace_hash") != trace_hash:
        raise ValueError("target stage refused: calibration gate is not bound to this calibration trace hash")
    reader_results = gate.get("reader_results")
    if not isinstance(reader_results, dict):
        raise ValueError("target stage refused: calibration gate lacks reader_results")
    required_control_passes = ("fresh_execute_pass", "expired_bound_verify_pass",
                               "no_memory_verify_pass", "tool_parse_pass")
    failed = sorted(reader for reader in readers
                    if (reader_results.get(reader, {}).get("all_three_pass") is not True or
                        any(reader_results.get(reader, {}).get(field) is not True
                            for field in required_control_passes)))
    if failed:
        raise ValueError("target stage refused: all-three calibration pass missing for " + ", ".join(failed))
    return {"freeze_manifest": str(manifest), "freeze_hash": actual,
            "calibration_gate": str(calibration), "calibration_trace": str(calibration_trace),
            "calibration_trace_hash": trace_hash}


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function", tool)
        return function.get("name") if isinstance(function, dict) else None
    function = getattr(tool, "function", None)
    return getattr(function, "name", None)


def standard_tool_schemas(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the frozen standard execute/verify schemas when builder omits tools."""
    return [
        {"type": "function", "function": {"name": item["execute_tool"],
         "description": "Execute the requested task using the supplied current value.",
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"value": {"type": "string"}}, "required": ["value"]}}},
        {"type": "function", "function": {"name": item["verify_tool"],
         "description": "Request confirmation of the current value for a field.",
         "parameters": {"type": "object", "additionalProperties": False,
                        "properties": {"field": {"type": "string"}}, "required": ["field"]}}},
    ]


def materialize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize builder records without mutating the frozen input object."""
    normalized = dict(item)
    if "messages" not in normalized:
        if not isinstance(normalized.get("system"), str) or not isinstance(normalized.get("user_request"), str):
            raise ValueError(f"{normalized.get('id', '<unknown>')}: need messages or system + user_request")
        normalized["messages"] = [{"role": "system", "content": normalized["system"]},
                                  {"role": "user", "content": normalized["user_request"]}]
    if "tools" not in normalized:
        normalized["tools"] = standard_tool_schemas(normalized)
    return normalized


def validate_item(item: dict[str, Any]) -> dict[str, Any]:
    """Ensure the frozen request contains the two callable function schemas."""
    item = materialize_item(item)
    required = ("id", "model", "provider", "messages", "tools", "max_output_tokens",
                "execute_tool", "verify_tool", "witness", "target_field")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"{item.get('id', '<unknown>')}: missing required fields: {', '.join(missing)}")
    if not isinstance(item["tools"], list):
        raise ValueError(f"{item['id']}: tools must be a list of OpenAI function tools")
    names = {_tool_name(tool) for tool in item["tools"]}
    if item["execute_tool"] not in names or item["verify_tool"] not in names:
        raise ValueError(f"{item['id']}: tools must include named execute and verify functions")
    return item


def _tool_arguments(tool: Any) -> Any:
    if isinstance(tool, dict):
        function = tool.get("function", tool)
        return function.get("arguments") if isinstance(function, dict) else None
    function = getattr(tool, "function", None)
    return getattr(function, "arguments", None)


def _message_tool_calls(response: Any) -> list[Any]:
    """Read standard OpenAI chat-completions objects or model_dump dictionaries."""
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, dict):
        return []
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    return message.get("tool_calls") or []


def _contains_witness(arguments: dict[str, Any], witness: str) -> bool:
    target = witness.casefold()
    return any(isinstance(value, str) and value.casefold() == target
               for value in arguments.values())


def parse_first_tool_call(response: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Classify a strict, single, valid first tool call.

    Multiple calls are UNPARSED even if their first call is otherwise valid;
    this makes the frozen multiple-call exclusion visible rather than silently
    treating a later verification as irrelevant.
    """
    calls = _message_tool_calls(response)
    first_name = _tool_name(calls[0]) if calls else None
    result: dict[str, Any] = {"score": "UNPARSED", "first_tool_name": first_name,
                              "tool_call_count": len(calls), "parsed_arguments": None,
                              "parse_error": None}
    if len(calls) != 1:
        result["parse_error"] = "no_tool_call" if not calls else "multiple_tool_calls"
        return result
    raw_args = _tool_arguments(calls[0])
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        result["parse_error"] = "malformed_arguments"
        return result
    if not isinstance(arguments, dict):
        result["parse_error"] = "arguments_not_object"
        return result
    result["parsed_arguments"] = arguments
    execute_name, verify_name = item.get("execute_tool"), item.get("verify_tool")
    witness, field = item.get("witness"), item.get("target_field")
    if not all(isinstance(x, str) and x for x in (execute_name, verify_name, witness, field)):
        result["parse_error"] = "item_missing_scoring_fields"
    elif first_name == execute_name and _contains_witness(arguments, witness):
        result["score"] = "EXECUTE"
    elif first_name == execute_name:
        result["parse_error"] = "execute_missing_or_invented_witness"
    elif first_name == verify_name and arguments.get("field") == field:
        result["score"] = "VERIFY"
    elif first_name == verify_name:
        result["parse_error"] = "verify_wrong_field"
    else:
        result["parse_error"] = "unexpected_tool"
    return result


def usage_and_cost(response: Any, price: dict[str, Any] | None) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    usage = response.get("usage") or {} if isinstance(response, dict) else {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    cost = None
    if price and isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
        cost = (prompt * float(price["input_per_million"]) +
                completion * float(price["output_per_million"])) / 1_000_000
    return {"usage": usage, "cost_usd": cost}


def estimate_cost(items: Iterable[dict[str, Any]], prices: dict[str, Any]) -> dict[str, Any]:
    rows, total = [], 0.0
    for item in items:
        item = validate_item(item)
        model = item["model"]
        price = prices.get(model)
        if not price:
            raise ValueError(f"no price entry for model {model}")
        prompt = item.get("input_tokens_estimate")
        completion = item.get("max_output_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            raise ValueError(f"{item.get('id')}: integer input_tokens_estimate and max_output_tokens required")
        cost = (prompt * float(price["input_per_million"]) +
                completion * float(price["output_per_million"])) / 1_000_000
        total += cost
        rows.append({"id": item.get("id"), "model": model, "estimated_cost_usd": cost})
    return {"calls": len(rows), "estimated_cost_usd": total, "items": rows}


TRACE_LOCK = threading.Lock()


def append_trace(path: Path, record: dict[str, Any]) -> None:
    """Durably append one attempt.  This function never rewrites a trace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(record) + "\n").encode("utf-8")
    with TRACE_LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)


def response_metadata(response: Any, configured_model: str, provider: str) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    response = response if isinstance(response, dict) else {}
    return {"configured_model": configured_model, "provider": provider,
            "served_model": response.get("model"), "generation_id": response.get("id")}


def raw_response(response: Any) -> dict[str, Any]:
    dumped = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(dumped, dict):
        return {"unserializable_response": repr(dumped)}
    return json.loads(canonical_json(dumped))


def recorded_attempts(trace: Path, run_identity: dict[str, str | None]) -> list[dict[str, Any]]:
    if not trace.exists():
        return []
    attempts = []
    for line_number, line in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid existing trace JSON at line {line_number}") from exc
        if row.get("run_identity") == run_identity:
            attempts.append(row)
    return attempts


def resume_items(items: Iterable[dict[str, Any]], trace: Path, run_identity: dict[str, str | None],
                 retry_failed: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Skip completed calls; failed attempts require explicit retry authorization."""
    attempts = recorded_attempts(trace, run_identity)
    successful = {row.get("item_id") for row in attempts if row.get("error") is None}
    failed = {row.get("item_id") for row in attempts if row.get("error") is not None} - successful
    todo, skipped_success, skipped_failed = [], 0, 0
    for source_item in items:
        item = validate_item(source_item)
        if item["id"] in successful:
            skipped_success += 1
        elif item["id"] in failed and not retry_failed:
            skipped_failed += 1
        else:
            todo.append(item)
    return todo, {"skipped_success": skipped_success, "skipped_failed": skipped_failed}


def enforce_cost_cap(items: Iterable[dict[str, Any]], prices: dict[str, Any], cap_usd: float,
                     previous_attempts: Iterable[dict[str, Any]] = ()) -> dict[str, float]:
    if cap_usd < 0:
        raise ValueError("cost cap must be nonnegative")
    prior = sum(float(row["cost_usd"]) for row in previous_attempts
                if isinstance(row.get("cost_usd"), (int, float)))
    estimate = estimate_cost(items, prices)["estimated_cost_usd"]
    if prior + estimate > cap_usd:
        raise ValueError(f"cost cap exceeded: recorded ${prior:.6f} + estimated ${estimate:.6f} > cap ${cap_usd:.6f}")
    return {"recorded_cost_usd": prior, "estimated_cost_usd": estimate, "cost_cap_usd": cap_usd}


def run_items(items: Iterable[dict[str, Any]], stage: str, trace: Path, client: Any,
              prices: dict[str, Any] | None = None, gate: dict[str, str | None] | None = None,
              input_hash: str | None = None, workers: int = 1) -> list[dict[str, Any]]:
    """Call each item once; errors are records, not reasons to overwrite history."""
    if stage == "target" and not gate:
        raise ValueError("target stage refused: call require_target_gate before run_items")
    if workers < 1:
        raise ValueError("workers must be at least one")
    normalized_items = [validate_item(item) for item in items]
    def one(item: dict[str, Any]) -> dict[str, Any]:
        if item.get("stage") and item["stage"] != stage:
            raise ValueError(f"{item.get('id')}: item stage {item['stage']} differs from requested {stage}")
        started = time.time()
        record = {"attempt_id": str(uuid.uuid4()), "timestamp_unix": started, "stage": stage,
                  "item_id": item.get("id"), "item_sha256": hashlib.sha256(canonical_json(item).encode()).hexdigest(),
                  "item": item, "run_identity": {"input_hash": input_hash, "freeze_hash": (gate or {}).get("freeze_hash")},
                  "freeze": gate or {},
                  "configured_model": item.get("model"), "provider": item.get("provider"),
                  "served_model": None, "generation_id": None, "usage": {}, "cost_usd": None,
                  "error": None, "score": "UNPARSED", "first_tool_name": None,
                  "tool_call_count": 0, "parsed_arguments": None, "parse_error": None}
        try:
            extra_body = {"provider": {"only": [item["provider"]], "allow_fallbacks": False}}
            if "reasoning" in item:
                extra_body["reasoning"] = item["reasoning"]
            response = client.chat.completions.create(
                model=item["model"], messages=item["messages"], tools=item["tools"],
                tool_choice="required", temperature=item.get("temperature", 0),
                max_tokens=item["max_output_tokens"],
                extra_body=extra_body,
            )
            record["raw_response"] = raw_response(response)
            record.update(response_metadata(response, item["model"], item["provider"]))
            record.update(usage_and_cost(response, (prices or {}).get(item["model"])))
            record.update(parse_first_tool_call(response, item))
        except Exception as exc:  # retained as a traceable failed attempt
            record["error"] = repr(exc)
        record["elapsed_seconds"] = round(time.time() - started, 6)
        append_trace(trace, record)
        return record
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(one, normalized_items))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frozen JSONL item file")
    parser.add_argument("--stage", choices=("control", "target"), required=True)
    parser.add_argument("--trace", type=Path, default=Path("trace.jsonl"))
    parser.add_argument("--prices", type=Path, help="JSON: model -> per-million input/output prices")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; never creates a client or calls a provider")
    parser.add_argument("--cost-cap-usd", type=float, required=True,
                        help="hard preflight ceiling for recorded plus reserved estimated spend")
    parser.add_argument("--workers", type=int, default=6, help="bounded concurrent provider requests (default: 6)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="explicitly retry failed matching attempts; successful attempts always stay skipped")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--freeze-hash")
    parser.add_argument("--calibration-gate", type=Path)
    parser.add_argument("--calibration-trace", type=Path)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint; required for live calls")
    parser.add_argument("--api-key-env", help="environment variable holding API key; required for live calls")
    args = parser.parse_args()
    source_items = load_jsonl(args.input)
    items = [validate_item(item) for item in source_items]
    gate = require_target_gate(args.stage, args.freeze_manifest, args.freeze_hash,
                               args.calibration_gate, args.calibration_trace,
                               {reader_id(item) for item in items})
    if not args.prices:
        parser.error("--prices is required to enforce the cost cap")
    prices = json.loads(args.prices.read_text(encoding="utf-8"))
    input_hash = sha256_file(args.input)
    run_identity = {"input_hash": input_hash, "freeze_hash": gate.get("freeze_hash")}
    todo, resume_summary = resume_items(items, args.trace, run_identity, args.retry_failed)
    previous = recorded_attempts(args.trace, run_identity)
    cap = enforce_cost_cap(todo, prices, args.cost_cap_usd, previous)
    if args.dry_run:
        print(canonical_json({"stage": args.stage, "freeze": gate, "run_identity": run_identity,
                              "resume": resume_summary, **cap, **estimate_cost(todo, prices)}))
        return
    if not args.base_url or not args.api_key_env:
        parser.error("live calls require --base-url and --api-key-env")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable is unset: {args.api_key_env}")
    from openai import OpenAI  # imported only when a paid call was explicitly requested
    records = run_items(todo, args.stage, args.trace, OpenAI(base_url=args.base_url, api_key=api_key), prices, gate,
                        input_hash=input_hash, workers=args.workers)
    print(canonical_json({"attempts": len(records), "trace": str(args.trace), "resume": resume_summary, "cap": cap,
                          "scores": {label: sum(r["score"] == label for r in records) for label in LABELS}}))


if __name__ == "__main__":
    main()
