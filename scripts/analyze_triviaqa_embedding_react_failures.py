#!/usr/bin/env python3
"""Offline diagnostic report for the TriviaQA embedding/ReAct AgentGraph run.

The analyzer is deliberately independent from the evaluation runtime: it reads
the persisted selected-task, trajectory, evaluator, ReAct trace, and Tool
receipt fields and writes a JSON report plus a Chinese Markdown rendering.  It
does not import model/provider code and never performs a network or model call.

The input trajectory file may still be growing.  Reads are bounded to the file
size observed when the script starts, and an incomplete run is explicitly
reported as ``partial`` instead of being presented as a 128-example result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts/triviaqa_embedding_react_v1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/triviaqa_embedding_react_v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing file: {_display_path(path)}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {_display_path(path)}: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, f"expected JSON object in {_display_path(path)}"
    return value, None


def _read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only bytes present at entry so an active append cannot extend the read."""

    diagnostics: dict[str, Any] = {
        "path": _display_path(path),
        "snapshot_bytes": 0,
        "valid_records": 0,
        "malformed_records": [],
    }
    if not path.exists():
        diagnostics["missing"] = True
        return [], diagnostics

    snapshot_bytes = path.stat().st_size
    diagnostics["snapshot_bytes"] = snapshot_bytes
    with path.open("rb") as handle:
        payload = handle.read(snapshot_bytes)

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            diagnostics["malformed_records"].append(
                {
                    "line_number": line_number,
                    "error": f"{type(exc).__name__}: {exc}",
                    "possibly_active_trailing_write": line_number
                    == len(payload.splitlines())
                    and not payload.endswith(b"\n"),
                }
            )
            continue
        if not isinstance(value, dict):
            diagnostics["malformed_records"].append(
                {
                    "line_number": line_number,
                    "error": "record is not a JSON object",
                    "possibly_active_trailing_write": False,
                }
            )
            continue
        records.append(value)
    diagnostics["valid_records"] = len(records)
    return records, diagnostics


def _task_id_from_trajectory(trajectory: Mapping[str, Any]) -> str | None:
    task_id = _mapping(trajectory.get("task")).get("task_id")
    return task_id if isinstance(task_id, str) and task_id else None


def _deduplicate_trajectories(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    missing_task_id_records = 0
    for row in rows:
        task_id = _task_id_from_trajectory(row)
        if task_id is None:
            missing_task_id_records += 1
            continue
        if task_id in by_task:
            duplicates[task_id] += 1
        by_task[task_id] = row
    return by_task, {
        "duplicate_task_record_counts": dict(sorted(duplicates.items())),
        "missing_task_id_records": missing_task_id_records,
        "deduplicated_task_count": len(by_task),
        "duplicate_resolution": "last complete JSONL record for a task_id wins",
    }


def _iter_executions(
    trajectory: Mapping[str, Any],
) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
    for turn_position, turn in enumerate(_list(trajectory.get("turns"))):
        if not isinstance(turn, Mapping):
            continue
        round_index = turn.get("round_index", turn_position)
        if not isinstance(round_index, int):
            round_index = turn_position
        for execution_position, execution in enumerate(_list(turn.get("executions"))):
            if isinstance(execution, Mapping):
                yield round_index, execution_position, execution


def _receipt_signature(receipt: Mapping[str, Any]) -> str:
    """Collapse a receipt copied into downstream metadata without merging reruns."""

    ended = receipt.get("ended_at_monotonic")
    started = receipt.get("started_at_monotonic")
    if ended is not None or started is not None:
        return _canonical(
            {
                "tool_id": receipt.get("tool_id"),
                "tool_version": receipt.get("tool_version"),
                "started_at_monotonic": started,
                "ended_at_monotonic": ended,
                "request": receipt.get("request"),
            }
        )
    return _canonical(receipt)


def _normalize_query(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).casefold()
    return normalized or None


def _trace_status_and_code(entry: Mapping[str, Any]) -> tuple[str | None, str | None]:
    observation = _mapping(entry.get("observation"))
    status = observation.get("observation_status", entry.get("observation_status"))
    code = observation.get("public_error_code", entry.get("public_error_code"))
    return (
        str(status) if status is not None else None,
        str(code) if code is not None else None,
    )


def _trace_action(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    action = entry.get("structured_action")
    if isinstance(action, Mapping):
        return action
    observation = _mapping(entry.get("observation"))
    action = observation.get("executed_action", entry.get("executed_action"))
    return action if isinstance(action, Mapping) else {}


def _tool_summary(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    unique_receipts: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    receipt_action_counts: Counter[str] = Counter()
    attempted_action_counts: Counter[str] = Counter()
    trace_status_counts: Counter[str] = Counter()
    react_error_code_counts: Counter[str] = Counter()
    duplicate_suppressions = 0
    tool_error_observations = 0

    for round_index, execution_position, execution in _iter_executions(trajectory):
        metadata = _mapping(execution.get("metadata"))
        response = _mapping(metadata.get("response"))
        for receipt_position, raw_receipt in enumerate(_list(response.get("tool_receipts"))):
            if not isinstance(raw_receipt, Mapping):
                continue
            signature = _receipt_signature(raw_receipt)
            if signature in seen_receipts:
                continue
            seen_receipts.add(signature)
            receipt = dict(raw_receipt)
            request = _mapping(receipt.get("request"))
            action = request.get("action")
            action_name = str(action).casefold() if action is not None else "unknown"
            receipt_action_counts[action_name] += 1
            unique_receipts.append(
                {
                    "round_index": round_index,
                    "execution_position": execution_position,
                    "receipt_position": receipt_position,
                    "agent_id": execution.get("agent_id"),
                    "receipt": receipt,
                }
            )

        for raw_entry in _list(response.get("react_trace")):
            if not isinstance(raw_entry, Mapping):
                continue
            action = _trace_action(raw_entry)
            name = action.get("name", action.get("action"))
            attempted_action_counts[
                str(name).casefold() if name is not None else "unknown"
            ] += 1
            status, code = _trace_status_and_code(raw_entry)
            trace_status_counts[status or "missing"] += 1
            if code:
                react_error_code_counts[code] += 1
            if code == "duplicate_tool_request":
                duplicate_suppressions += 1
            if status == "tool_error":
                tool_error_observations += 1

    successful_receipts = 0
    tool_error_receipts = 0
    search_queries: list[str] = []
    search_query_values: list[str] = []
    read_document_ids: list[str] = []
    for item in unique_receipts:
        receipt = _mapping(item.get("receipt"))
        result = receipt.get("result")
        result_mapping = _mapping(result)
        failed = (
            receipt.get("error_type") is not None
            or not isinstance(result, Mapping)
            or result_mapping.get("completed") is False
        )
        if failed:
            tool_error_receipts += 1
        else:
            successful_receipts += 1
        request = _mapping(receipt.get("request"))
        action = str(request.get("action", "")).casefold()
        arguments = _mapping(request.get("arguments"))
        if action == "search":
            raw_query = arguments.get("query")
            normalized_query = _normalize_query(raw_query)
            if normalized_query:
                search_queries.append(normalized_query)
                search_query_values.append(str(raw_query))
        elif action == "read":
            document_id = arguments.get("passage_id", arguments.get("doc_id"))
            if isinstance(document_id, str) and document_id:
                read_document_ids.append(document_id)

    distinct_queries: list[str] = []
    distinct_query_values: list[str] = []
    seen_queries: set[str] = set()
    for normalized, original in zip(search_queries, search_query_values):
        if normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        distinct_queries.append(normalized)
        distinct_query_values.append(original)

    return {
        "tool_receipt_count": len(unique_receipts),
        "receipt_action_counts": dict(sorted(receipt_action_counts.items())),
        "successful_receipt_count": successful_receipts,
        "tool_error_count": tool_error_receipts,
        "search_count": receipt_action_counts.get("search", 0),
        "read_count": receipt_action_counts.get("read", 0),
        "search_queries": search_query_values,
        "distinct_search_queries": distinct_query_values,
        "second_or_later_distinct_query_count": max(len(distinct_queries) - 1, 0),
        "has_second_distinct_query": len(distinct_queries) >= 2,
        "read_document_ids": read_document_ids,
        "duplicate_suppression_count": duplicate_suppressions,
        "tool_error_observation_count": tool_error_observations,
        "react_attempted_action_counts": dict(sorted(attempted_action_counts.items())),
        "react_trace_status_counts": dict(sorted(trace_status_counts.items())),
        "react_error_code_counts": dict(sorted(react_error_code_counts.items())),
    }


def _metrics(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(trajectory.get("evaluation"))
    metrics = _mapping(evaluation.get("metrics"))
    return {
        "evaluator_version": evaluation.get("evaluator_version"),
        "valid": evaluation.get("valid"),
        "reward": evaluation.get("reward"),
        "reason": evaluation.get("reason"),
        "exact_match": metrics.get("exact_match"),
        "token_f1": metrics.get("token_f1"),
    }


def _is_evaluated(trajectory: Mapping[str, Any]) -> bool:
    evaluation = trajectory.get("evaluation")
    return isinstance(evaluation, Mapping) and bool(evaluation)


def _is_wrong(trajectory: Mapping[str, Any]) -> bool:
    if not _is_evaluated(trajectory):
        return False
    metric = _metrics(trajectory)
    if metric.get("valid") is not True:
        return True
    exact_match = metric.get("exact_match")
    return not isinstance(exact_match, (int, float)) or float(exact_match) < 1.0


def _answer_producer(trajectory: Mapping[str, Any]) -> dict[str, Any] | None:
    final_answer = trajectory.get("final_answer")
    turns = [turn for turn in _list(trajectory.get("turns")) if isinstance(turn, Mapping)]
    for fallback_index, turn in reversed(list(enumerate(turns))):
        runtime = _mapping(turn.get("runtime_summary"))
        runtime_answer = runtime.get("final_answer")
        if final_answer is not None and runtime_answer != final_answer:
            continue
        output_agent_id = runtime.get("output_agent_id")
        for execution in reversed(_list(turn.get("executions"))):
            if not isinstance(execution, Mapping):
                continue
            if output_agent_id is not None and execution.get("agent_id") != output_agent_id:
                continue
            return {
                "round_index": turn.get("round_index", fallback_index),
                "step": "agent_output",
                "agent_id": execution.get("agent_id"),
                "execution_id": execution.get("execution_id"),
                "observed_output": execution.get("output"),
            }
    return None


def _explicit_failure_signals(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for fallback_index, turn in enumerate(_list(trajectory.get("turns"))):
        if not isinstance(turn, Mapping):
            continue
        round_index = turn.get("round_index", fallback_index)
        for execution in _list(turn.get("executions")):
            if not isinstance(execution, Mapping):
                continue
            response = _mapping(_mapping(execution.get("metadata")).get("response"))
            for trace_position, entry in enumerate(_list(response.get("react_trace"))):
                if not isinstance(entry, Mapping):
                    continue
                status, code = _trace_status_and_code(entry)
                if status in {"success", "completed", None}:
                    continue
                if code == "duplicate_tool_request":
                    category = "duplicate_suppression"
                elif status == "tool_error":
                    category = "tool_execution"
                else:
                    category = "react_validation"
                signals.append(
                    {
                        "round_index": round_index,
                        "step": "react_action_observation",
                        "agent_id": execution.get("agent_id"),
                        "trace_position": trace_position,
                        "category": category,
                        "observation_status": status,
                        "public_error_code": code,
                        "action": _trace_action(entry),
                    }
                )

        feedback = turn.get("canvas_feedback")
        if not isinstance(feedback, str) or not feedback:
            continue
        lowered = feedback.casefold()
        category: str | None = None
        if any(
            marker in lowered
            for marker in (
                "cannot declare allowed_tools",
                "unknown tool",
                "unregistered execution adapter",
                "requires unregistered",
                "tool_not_allowed",
                "tool action",
            )
        ):
            category = "orchestration_tool_contract"
        elif any(
            marker in lowered
            for marker in (
                "must consume exactly one upstream",
                "cannot finish",
                "topology",
                "relation",
                "output agent",
                "all agents",
            )
        ) and not feedback.startswith("accepted"):
            category = "orchestration_relationship"
        elif any(
            marker in lowered
            for marker in (
                "invalid action",
                "invalid json",
                "unable to parse",
                "must contain exactly",
                "at most 3 agents",
                "unsupported action",
            )
        ):
            category = "director_action_schema"
        elif "execution_error=" in lowered:
            category = "agent_execution"
        if category:
            signals.append(
                {
                    "round_index": round_index,
                    "step": "canvas_feedback",
                    "agent_id": None,
                    "category": category,
                    "observed_feedback": feedback,
                    "director_action": turn.get("action"),
                }
            )
    return signals


def _first_causal_failure(
    trajectory: Mapping[str, Any], tool_summary: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = _mapping(trajectory.get("evaluation"))
    details = _mapping(evaluation.get("details"))
    termination = trajectory.get("termination_reason")
    answer_producer = _answer_producer(trajectory)

    if evaluation.get("valid") is not True:
        return {
            "category": "evaluator_or_canonicalization",
            "round_index": None,
            "step": "evaluator_receipt",
            "agent_id": None,
            "evidence": dict(evaluation),
            "basis": "the persisted evaluator receipt is invalid",
        }

    if termination == "finish":
        if details.get("structured_answer_extracted") is False:
            return {
                "category": "formatting",
                **(answer_producer or {"round_index": None, "step": "final_answer", "agent_id": None}),
                "evidence": {
                    "final_answer": trajectory.get("final_answer"),
                    "evaluator_details": dict(details),
                },
                "basis": "the evaluator explicitly recorded that no structured answer was extracted",
            }
        if int(tool_summary.get("read_count", 0)) > 0:
            return {
                "category": "agent_reasoning_or_answer_selection",
                **(answer_producer or {"round_index": None, "step": "final_answer", "agent_id": None}),
                "evidence": {
                    "final_answer": trajectory.get("final_answer"),
                    "scored_prediction": details.get("scored_prediction"),
                    "successful_search_count": tool_summary.get("search_count"),
                    "successful_read_count": tool_summary.get("read_count"),
                },
                "basis": "search/read receipts exist, but the terminal answer received exact_match < 1",
            }
        if int(tool_summary.get("search_count", 0)) > 0:
            return {
                "category": "retrieval_read_incomplete",
                **(answer_producer or {"round_index": None, "step": "final_answer", "agent_id": None}),
                "evidence": {
                    "final_answer": trajectory.get("final_answer"),
                    "search_count": tool_summary.get("search_count"),
                    "read_count": tool_summary.get("read_count"),
                },
                "basis": "a search receipt exists, no read receipt exists, and the terminal answer is wrong",
            }
        return {
            "category": "retrieval_not_invoked",
            **(answer_producer or {"round_index": None, "step": "final_answer", "agent_id": None}),
            "evidence": {
                "final_answer": trajectory.get("final_answer"),
                "search_count": 0,
                "read_count": 0,
            },
            "basis": "the run finished with a wrong answer and no persisted retrieval receipt",
        }

    signals = _explicit_failure_signals(trajectory)
    if signals:
        first = dict(signals[0])
        first["basis"] = (
            "earliest persisted failing Canvas/ReAct observation on a trajectory "
            "that did not recover to an evaluator-correct terminal result"
        )
        return first
    return {
        "category": "terminal_max_rounds" if termination == "max_rounds" else "terminal_other",
        "round_index": None,
        "step": "terminal_receipt",
        "agent_id": None,
        "evidence": {
            "termination_reason": termination,
            "terminal_failure": trajectory.get("terminal_failure"),
            "final_answer": trajectory.get("final_answer"),
        },
        "basis": "no earlier explicit persisted failure signal was available",
    }


def _propagation(
    trajectory: Mapping[str, Any], first_failure: Mapping[str, Any]
) -> list[dict[str, Any]]:
    first_round = first_failure.get("round_index")
    if not isinstance(first_round, int):
        first_round = -1
    events: list[dict[str, Any]] = []
    for fallback_index, turn in enumerate(_list(trajectory.get("turns"))):
        if not isinstance(turn, Mapping):
            continue
        round_index = turn.get("round_index", fallback_index)
        if not isinstance(round_index, int) or round_index < first_round:
            continue
        feedback = turn.get("canvas_feedback")
        if isinstance(feedback, str) and any(
            marker in feedback.casefold()
            for marker in ("cannot", "error", "invalid", "rejected", "must ")
        ):
            events.append(
                {
                    "round_index": round_index,
                    "source": "canvas_feedback",
                    "observed": feedback,
                }
            )
        runtime = _mapping(turn.get("runtime_summary"))
        if "final_answer" in runtime:
            events.append(
                {
                    "round_index": round_index,
                    "source": "runtime_summary.final_answer",
                    "observed": runtime.get("final_answer"),
                }
            )
    events.append(
        {
            "round_index": None,
            "source": "terminal_receipt",
            "observed": {
                "termination_reason": trajectory.get("termination_reason"),
                "explicit_finish": trajectory.get("explicit_finish"),
                "terminal_failure": trajectory.get("terminal_failure"),
                "final_answer": trajectory.get("final_answer"),
            },
        }
    )
    events.append(
        {
            "round_index": None,
            "source": "evaluator_receipt",
            "observed": trajectory.get("evaluation"),
        }
    )
    return events


def _agent_execution_chain(execution: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(execution.get("metadata"))
    request = _mapping(metadata.get("request"))
    response = _mapping(metadata.get("response"))
    return {
        "agent_id": execution.get("agent_id"),
        "execution_id": execution.get("execution_id"),
        "error_type": execution.get("error_type"),
        "agent_input": {
            key: request.get(key)
            for key in (
                "request_id",
                "run_id",
                "graph_revision",
                "phase",
                "execution_role",
                "is_output_agent",
                "is_format_agent",
                "provider_id",
                "problem",
                "agent",
                "model",
                "rendered_messages",
            )
            if key in request
        },
        "agent_communication": {
            key: request.get(key)
            for key in (
                "communication_condition",
                "upstream",
                "own_draft",
                "peer_draft",
            )
            if key in request
        },
        "agent_output": execution.get("output"),
        "provider_response_metadata": {
            key: value
            for key, value in response.items()
            if key not in {"react_trace", "tool_receipts"}
        },
        "react_action_observation_trace": response.get("react_trace", []),
        "tool_receipts": response.get("tool_receipts", []),
    }


def _demo(
    selected_task: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    tool_summary: Mapping[str, Any],
) -> dict[str, Any]:
    task = _mapping(trajectory.get("task")) or selected_task
    evaluator_payload = _mapping(_mapping(selected_task.get("metadata")).get("evaluator_payload"))
    turns: list[dict[str, Any]] = []
    for fallback_index, raw_turn in enumerate(_list(trajectory.get("turns"))):
        if not isinstance(raw_turn, Mapping):
            continue
        turns.append(
            {
                "round_index": raw_turn.get("round_index", fallback_index),
                "director": {
                    "request_id": raw_turn.get("director_request_id"),
                    "policy_response": raw_turn.get("policy_response"),
                    "parsed_action": raw_turn.get("action"),
                    "receipt_verified": raw_turn.get("receipt_verified"),
                },
                "canvas_edit_and_feedback": {
                    "graph_revision": raw_turn.get("graph_revision"),
                    "previous_graph_snapshot_id": raw_turn.get("previous_graph_snapshot_id"),
                    "graph_snapshot_id": raw_turn.get("graph_snapshot_id"),
                    "graph_snapshot": raw_turn.get("graph_snapshot"),
                    "feedback": raw_turn.get("canvas_feedback"),
                },
                "agent_executions": [
                    _agent_execution_chain(execution)
                    for execution in _list(raw_turn.get("executions"))
                    if isinstance(execution, Mapping)
                ],
                "runtime_summary": raw_turn.get("runtime_summary"),
            }
        )

    first_failure = _first_causal_failure(trajectory, tool_summary)
    return {
        "task_id": task.get("task_id", selected_task.get("task_id")),
        "input": {"question": task.get("question", selected_task.get("question"))},
        "reference_target": {
            "ground_truth": selected_task.get("ground_truth", task.get("ground_truth")),
            "accepted_answers": evaluator_payload.get("accepted_answers"),
        },
        "system_final_output_and_metrics": {
            "final_answer": trajectory.get("final_answer"),
            "metrics": _metrics(trajectory),
        },
        "tool_usage": dict(tool_summary),
        "actual_execution_chain": turns,
        "first_causal_failure": first_failure,
        "downstream_error_propagation": _propagation(trajectory, first_failure),
        "terminal_receipt": {
            key: trajectory.get(key)
            for key in (
                "termination_reason",
                "explicit_finish",
                "terminal_failure",
                "condition_satisfied",
                "final_answer",
                "trajectory_id",
                "rollout_id",
            )
            if key in trajectory
        },
        "evaluator_receipt": trajectory.get("evaluation"),
    }


def _aggregate(
    selected_tasks: Sequence[dict[str, Any]],
    trajectories: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    selected_ids = [
        task_id
        for task in selected_tasks
        if isinstance((task_id := task.get("task_id")), str) and task_id
    ]
    selected_set = set(selected_ids)
    per_task: dict[str, dict[str, Any]] = {}
    aggregate_receipt_actions: Counter[str] = Counter()
    aggregate_attempted_actions: Counter[str] = Counter()
    aggregate_trace_statuses: Counter[str] = Counter()
    aggregate_error_codes: Counter[str] = Counter()
    terminal_statuses: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    total_receipts = successful_receipts = tool_errors = 0
    duplicate_suppressions = tool_error_observations = 0
    secondary_distinct_query_count = 0
    tasks_with_search = tasks_with_read = tasks_with_second_query = 0
    evaluated_count = valid_evaluator_count = correct_count = 0
    em_sum = f1_sum = 0.0
    wrong_ids: list[str] = []

    for task_id in selected_ids:
        trajectory = trajectories.get(task_id)
        if trajectory is None:
            per_task[task_id] = {"task_id": task_id, "trajectory_present": False}
            continue
        tool = _tool_summary(trajectory)
        metric = _metrics(trajectory)
        evaluated = _is_evaluated(trajectory)
        wrong = _is_wrong(trajectory)
        if evaluated:
            evaluated_count += 1
            terminal_statuses[str(trajectory.get("termination_reason", "missing"))] += 1
            if metric.get("valid") is True:
                valid_evaluator_count += 1
            exact_match = metric.get("exact_match")
            token_f1 = metric.get("token_f1")
            if isinstance(exact_match, (int, float)):
                em_sum += float(exact_match)
                if float(exact_match) >= 1.0:
                    correct_count += 1
            if isinstance(token_f1, (int, float)):
                f1_sum += float(token_f1)
        if wrong:
            wrong_ids.append(task_id)
            category_counts[_first_causal_failure(trajectory, tool)["category"]] += 1

        total_receipts += int(tool["tool_receipt_count"])
        successful_receipts += int(tool["successful_receipt_count"])
        tool_errors += int(tool["tool_error_count"])
        duplicate_suppressions += int(tool["duplicate_suppression_count"])
        tool_error_observations += int(tool["tool_error_observation_count"])
        secondary_distinct_query_count += int(tool["second_or_later_distinct_query_count"])
        if int(tool["search_count"]) > 0:
            tasks_with_search += 1
        if int(tool["read_count"]) > 0:
            tasks_with_read += 1
        if bool(tool["has_second_distinct_query"]):
            tasks_with_second_query += 1
        aggregate_receipt_actions.update(tool["receipt_action_counts"])
        aggregate_attempted_actions.update(tool["react_attempted_action_counts"])
        aggregate_trace_statuses.update(tool["react_trace_status_counts"])
        aggregate_error_codes.update(tool["react_error_code_counts"])
        per_task[task_id] = {
            "task_id": task_id,
            "trajectory_present": True,
            "termination_reason": trajectory.get("termination_reason"),
            "terminal_failure": trajectory.get("terminal_failure"),
            "final_answer": trajectory.get("final_answer"),
            "metrics": metric,
            "wrong": wrong,
            "tool_usage": tool,
        }

    wrong_count = len(wrong_ids)
    taxonomy = [
        {
            "category": category,
            "count": count,
            "percentage_of_wrong": round(100.0 * count / wrong_count, 2)
            if wrong_count
            else 0.0,
        }
        for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return (
        {
            "selected_task_count": len(selected_ids),
            "trajectory_count_for_selected_tasks": sum(
                1 for task_id in selected_ids if task_id in trajectories
            ),
            "evaluated_count": evaluated_count,
            "valid_evaluator_count": valid_evaluator_count,
            "wrong_count": wrong_count,
            "correct_exact_match_count": correct_count,
            "observed_metrics": {
                "exact_match_percentage": round(100.0 * em_sum / evaluated_count, 2)
                if evaluated_count
                else None,
                "token_f1_percentage": round(100.0 * f1_sum / evaluated_count, 2)
                if evaluated_count
                else None,
                "denominator": evaluated_count,
                "scope": "observed snapshot; official only when run_status is complete",
            },
            "tool_usage": {
                "tool_receipt_count": total_receipts,
                "receipt_action_counts": dict(sorted(aggregate_receipt_actions.items())),
                "search_count": aggregate_receipt_actions.get("search", 0),
                "read_count": aggregate_receipt_actions.get("read", 0),
                "successful_receipt_count": successful_receipts,
                "tool_error_count": tool_errors,
                "tasks_with_search": tasks_with_search,
                "tasks_with_read": tasks_with_read,
                "tasks_with_second_distinct_query": tasks_with_second_query,
                "second_or_later_distinct_query_count": secondary_distinct_query_count,
                "duplicate_suppression_count": duplicate_suppressions,
                "tool_error_observation_count": tool_error_observations,
                "react_attempted_action_counts": dict(sorted(aggregate_attempted_actions.items())),
                "react_trace_status_counts": dict(sorted(aggregate_trace_statuses.items())),
                "react_error_code_counts": dict(sorted(aggregate_error_codes.items())),
            },
            "terminal_status_counts": dict(sorted(terminal_statuses.items())),
            "wrong_failure_taxonomy": taxonomy,
            "unexpected_trajectory_task_ids": sorted(set(trajectories) - selected_set),
        },
        per_task,
        wrong_ids,
    )


def _select_demo_ids(
    wrong_ids: Sequence[str],
    trajectories: Mapping[str, dict[str, Any]],
    per_task: Mapping[str, Mapping[str, Any]],
    count: int,
) -> list[str]:
    """Choose one richly evidenced example per observed category, then fill in order."""

    category_candidates: dict[str, list[tuple[int, int, str]]] = {}
    for order, task_id in enumerate(wrong_ids):
        trajectory = trajectories[task_id]
        tool = _mapping(per_task[task_id].get("tool_usage"))
        category = str(_first_causal_failure(trajectory, tool).get("category"))
        richness = (
            100 * int(tool.get("tool_receipt_count", 0))
            + 10 * int(tool.get("duplicate_suppression_count", 0))
            + len(_list(trajectory.get("turns")))
        )
        category_candidates.setdefault(category, []).append((-richness, order, task_id))

    category_order = sorted(
        category_candidates,
        key=lambda category: (-len(category_candidates[category]), category),
    )
    selected: list[str] = []
    for category in category_order:
        candidate = sorted(category_candidates[category])[0][2]
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= count:
            return selected
    for task_id in wrong_ids:
        if task_id not in selected:
            selected.append(task_id)
        if len(selected) >= count:
            break
    return selected


def _json_block(value: object) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_markdown(report: Mapping[str, Any]) -> str:
    run = _mapping(report.get("run"))
    summary = _mapping(report.get("summary"))
    tool = _mapping(summary.get("tool_usage"))
    metrics = _mapping(summary.get("observed_metrics"))
    lines = [
        "# TriviaQA embedding/ReAct Tool 失败分析",
        "",
        f"生成时间：`{report.get('generated_at')}`。本报告完全由已落盘 trajectory、Agent metadata、ReAct trace、Tool receipt 与 evaluator receipt 离线生成。",
        "",
    ]
    if run.get("status") != "complete":
        lines.extend(
            [
                "> **状态：partial。** 当前只分析已落盘且可解析的样本；下述 EM/F1 不是完整 128 条正式结果。全量结束后重跑同一命令即可覆盖为完整报告。",
                "",
            ]
        )
    else:
        lines.extend(["> **状态：complete。** 选定 128 条均已有 trajectory 与 evaluator receipt。", ""])

    lines.extend(
        [
            "## 汇总",
            "",
            "| 项目 | 数值 |",
            "| --- | ---: |",
            f"| 选定任务 | {summary.get('selected_task_count')} |",
            f"| 已落盘 trajectory | {summary.get('trajectory_count_for_selected_tasks')} |",
            f"| 已评测 | {summary.get('evaluated_count')} |",
            f"| 错误样本（EM < 1 或 evaluator invalid） | {summary.get('wrong_count')} |",
            f"| observed EM | {metrics.get('exact_match_percentage')}% |",
            f"| observed F1 | {metrics.get('token_f1_percentage')}% |",
            "",
            "## Tool 与 ReAct 计数",
            "",
            "`search/read` 以去重后的实际 Tool receipt 为准；“二次 query”指同一 task 内第二个及之后的不同规范化 search query；duplicate suppression 以 ReAct trace 的 `duplicate_tool_request` 为准。",
            "",
            "| 项目 | 数量 |",
            "| --- | ---: |",
            f"| Tool receipts | {tool.get('tool_receipt_count')} |",
            f"| search | {tool.get('search_count')} |",
            f"| read | {tool.get('read_count')} |",
            f"| 使用 search 的任务 | {tool.get('tasks_with_search')} |",
            f"| 使用 read 的任务 | {tool.get('tasks_with_read')} |",
            f"| 存在第二个不同 query 的任务 | {tool.get('tasks_with_second_distinct_query')} |",
            f"| 第二个及之后的不同 query 总数 | {tool.get('second_or_later_distinct_query_count')} |",
            f"| duplicate suppression | {tool.get('duplicate_suppression_count')} |",
            f"| Tool error receipts | {tool.get('tool_error_count')} |",
            f"| ReAct tool_error observations | {tool.get('tool_error_observation_count')} |",
            "",
            "### ReAct public error code",
            "",
            _json_block(tool.get("react_error_code_counts", {})),
            "",
            "### Terminal 状态",
            "",
            _json_block(summary.get("terminal_status_counts", {})),
            "",
            "### 错误样本首个因果失败点分类",
            "",
            "占比的分母是当前 snapshot 中的错误样本数；分类只使用已落盘字段。",
            "",
            "| 类别 | 数量 | 错误样本占比 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in _list(summary.get("wrong_failure_taxonomy")):
        if isinstance(row, Mapping):
            lines.append(
                f"| {_md_cell(row.get('category'))} | {row.get('count')} | {row.get('percentage_of_wrong')}% |"
            )

    demos = _list(report.get("wrong_demos"))
    lines.extend(
        [
            "",
            "## 典型错误 demo",
            "",
            f"选择规则：先按当前错误分类频数取每类一条，再在该类中优先选择 receipt/trace 更完整的真实样本；不足时按 selected task 顺序补齐。实际生成 {len(demos)} 条，未生成的类别不会虚构案例。",
            "",
        ]
    )
    for demo_index, demo in enumerate(demos, start=1):
        if not isinstance(demo, Mapping):
            continue
        input_value = _mapping(demo.get("input"))
        target = _mapping(demo.get("reference_target"))
        output_metrics = _mapping(demo.get("system_final_output_and_metrics"))
        lines.extend(
            [
                f"### Demo {demo_index}: `{demo.get('task_id')}`",
                "",
                f"- 输入/问题：{input_value.get('question')}",
                f"- 参考答案/目标：`{target.get('ground_truth')}`；accepted answers：`{target.get('accepted_answers')}`",
                f"- 系统最终输出：`{output_metrics.get('final_answer')}`",
                f"- evaluator 指标：`{_canonical(output_metrics.get('metrics'))}`",
                "",
                "#### 首个因果失败点",
                "",
                _json_block(demo.get("first_causal_failure")),
                "",
                "#### 后续错误传播",
                "",
                _json_block(demo.get("downstream_error_propagation")),
                "",
                "#### Terminal / evaluator receipt",
                "",
                _json_block(
                    {
                        "terminal_receipt": demo.get("terminal_receipt"),
                        "evaluator_receipt": demo.get("evaluator_receipt"),
                    }
                ),
                "",
                "#### 实际 Director → Canvas → Agent/communication → ReAct Tool 链路",
                "",
            ]
        )
        for turn in _list(demo.get("actual_execution_chain")):
            if not isinstance(turn, Mapping):
                continue
            lines.extend(
                [
                    f"<details><summary>Round {turn.get('round_index')}</summary>",
                    "",
                    _json_block(
                        {
                            "director": turn.get("director"),
                            "canvas_edit_and_feedback": turn.get("canvas_edit_and_feedback"),
                            "agent_executions": turn.get("agent_executions"),
                            "runtime_summary": turn.get("runtime_summary"),
                        }
                    ),
                    "",
                    "</details>",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = _resolve(args.selected_tasks)
    trajectories_path = _resolve(args.trajectories)
    manifest_path = _resolve(args.run_manifest)
    selected_tasks, selected_diagnostics = _read_jsonl_snapshot(selected_path)
    trajectory_rows, trajectory_diagnostics = _read_jsonl_snapshot(trajectories_path)
    trajectories, dedup_diagnostics = _deduplicate_trajectories(trajectory_rows)
    manifest, manifest_error = _read_json(manifest_path)

    selected_by_id = {
        task_id: task
        for task in selected_tasks
        if isinstance((task_id := task.get("task_id")), str) and task_id
    }
    summary, per_task, wrong_ids = _aggregate(selected_tasks, trajectories)
    selected_ids = list(selected_by_id)
    missing_ids = [task_id for task_id in selected_ids if task_id not in trajectories]
    unevaluated_ids = [
        task_id
        for task_id in selected_ids
        if task_id in trajectories and not _is_evaluated(trajectories[task_id])
    ]
    malformed_count = len(trajectory_diagnostics["malformed_records"])
    complete = not missing_ids and not unevaluated_ids and malformed_count == 0 and bool(selected_ids)
    run_status = "complete" if complete else "partial"

    demo_ids = _select_demo_ids(
        wrong_ids,
        trajectories,
        per_task,
        max(args.demo_count, 0),
    )
    demos = [
        _demo(selected_by_id[task_id], trajectories[task_id], per_task[task_id]["tool_usage"])
        for task_id in demo_ids
    ]

    return {
        "schema_version": "flowsteer.triviaqa.embedding-react.tool-failure-analysis.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "offline_persisted_fields_only",
        "run": {
            "status": run_status,
            "complete_definition": (
                "every selected task has one parseable trajectory with an evaluator receipt"
            ),
            "missing_trajectory_task_ids": missing_ids,
            "unevaluated_trajectory_task_ids": unevaluated_ids,
            "manifest_reported_status": manifest.get("status"),
            "manifest_agentgraph_progress": manifest.get("agentgraph_progress"),
            "manifest_read_error": manifest_error,
            "official_full_validation_metrics": summary["observed_metrics"]
            if complete
            else None,
        },
        "inputs": {
            "selected_tasks": selected_diagnostics,
            "trajectories": trajectory_diagnostics,
            "run_manifest": _display_path(manifest_path),
            "trajectory_deduplication": dedup_diagnostics,
        },
        "count_definitions": {
            "search_read": "deduplicated persisted Tool receipts grouped by request.action",
            "second_query": (
                "second and later distinct case-folded whitespace-normalized search query within one task"
            ),
            "duplicate_suppression": (
                "ReAct trace entries whose persisted public_error_code is duplicate_tool_request"
            ),
            "tool_error": (
                "unique Tool receipt with error_type, missing result, or result.completed=false"
            ),
            "wrong": "evaluator invalid, missing exact_match, or exact_match < 1",
            "first_causal_failure": (
                "finish+wrong uses the terminal answer producer and persisted retrieval/evaluator state; "
                "non-finish uses the earliest explicit failing ReAct/Canvas observation"
            ),
        },
        "summary": summary,
        "samples": [per_task[task_id] for task_id in selected_ids],
        "wrong_demo_selection": {
            "requested_count": args.demo_count,
            "actual_count": len(demos),
            "selected_task_ids": demo_ids,
            "shortfall": max(args.demo_count - len(demos), 0),
            "no_fabrication_policy": "only evaluator-observed wrong trajectories are eligible",
        },
        "wrong_demos": demos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks",
        default=str(DEFAULT_ARTIFACT_DIR / "selected_tasks.jsonl"),
    )
    parser.add_argument(
        "--trajectories",
        default=str(DEFAULT_ARTIFACT_DIR / "agentgraph_trajectories.jsonl"),
    )
    parser.add_argument(
        "--run-manifest",
        default=str(DEFAULT_ARTIFACT_DIR / "run_manifest.json"),
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_REPORT_DIR / "tool_failure_analysis.json"),
    )
    parser.add_argument(
        "--output-markdown",
        default=str(DEFAULT_REPORT_DIR / "tool_failure_analysis.md"),
    )
    parser.add_argument("--demo-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo_count < 0:
        raise SystemExit("--demo-count must be non-negative")
    report = build_report(args)
    output_json = _resolve(args.output_json)
    output_markdown = _resolve(args.output_markdown)
    _atomic_write_text(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    _atomic_write_text(output_markdown, _render_markdown(report))
    print(
        json.dumps(
            {
                "status": report["run"]["status"],
                "selected": report["summary"]["selected_task_count"],
                "evaluated": report["summary"]["evaluated_count"],
                "wrong_demos": report["wrong_demo_selection"]["actual_count"],
                "output_json": _display_path(output_json),
                "output_markdown": _display_path(output_markdown),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
