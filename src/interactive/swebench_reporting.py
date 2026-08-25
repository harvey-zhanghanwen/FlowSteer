"""Offline SWE-bench telemetry and first-failure reporting.

This module consumes only persisted Direct/AgentGraph execution receipts and
official SWE-bench evaluator receipts.  It does not execute a repository,
invoke a model, inspect a gold patch, or infer success from model prose.  In
particular, local test accounting reads the structured ``run_tests``
``ToolResult`` fields, while task resolution remains the official harness'
boolean ``resolved`` result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Optional


SWEBENCH_TOOL_ACTION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "search": ("search_code",),
    "view": ("view_file",),
    "edit": ("apply_patch", "exact_edit", "str_replace_editor"),
    "test": ("run_tests",),
    "command": ("bash",),
}

_INVALID_TOOL_ACTION_CODES = frozenset(
    {
        "duplicate_tool_request",
        "tool_action_not_registered",
        "tool_arguments_not_object",
        "tool_arguments_schema_invalid",
        "tool_not_allowed",
    }
)

_PATCH_APPLY_STATUSES = frozenset({"patch_apply_failed"})
_TEST_FAILURE_STATUSES = frozenset({"test_timeout", "unresolved"})
_ENVIRONMENT_STATUSES = frozenset(
    {
        "env_not_ready",
        "image_build_failed",
        "image_missing_or_pull_failed",
        "test_patch_apply_failed",
        "worktree_create_failed",
    }
)
_ENVIRONMENT_PHASES = frozenset(
    {
        "container",
        "docker",
        "environment",
        "image",
        "repository_setup",
        "worktree",
    }
)
_PROVIDER_PHASES = frozenset(
    {"executor_provider", "model_provider", "provider", "provider_request"}
)


def _mapping(value: object) -> Optional[Mapping[str, Any]]:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _task_id_from_trajectory(trajectory: Mapping[str, Any]) -> str:
    task = _mapping(trajectory.get("task"))
    task_id = task.get("task_id") if task is not None else trajectory.get("task_id")
    return task_id if isinstance(task_id, str) else ""


def _trajectory_index(
    trajectories: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    values: Sequence[Mapping[str, Any]]
    if isinstance(trajectories, Mapping):
        values = tuple(
            value for value in trajectories.values() if isinstance(value, Mapping)
        )
    elif isinstance(trajectories, Sequence) and not isinstance(
        trajectories, (str, bytes)
    ):
        values = tuple(value for value in trajectories if isinstance(value, Mapping))
    else:
        values = ()
    return {
        task_id: trajectory
        for trajectory in values
        if (task_id := _task_id_from_trajectory(trajectory))
    }


def _execution_response(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _mapping(execution.get("metadata"))
    if metadata is None:
        return {}
    response = _mapping(metadata.get("response"))
    return response if response is not None else metadata


def _new_receipt_sequence(
    response: Mapping[str, Any],
    *,
    field_name: str,
    continued_count_name: str,
) -> tuple[Mapping[str, Any], ...]:
    values = _sequence(response.get(field_name))
    continued = response.get(continued_count_name, 0)
    if (
        isinstance(continued, int)
        and not isinstance(continued, bool)
        and 0 <= continued <= len(values)
    ):
        return values[continued:]
    return values


def _tool_receipt_action(receipt: Mapping[str, Any]) -> Optional[str]:
    request = _mapping(receipt.get("request"))
    action = request.get("action") if request is not None else None
    return action if isinstance(action, str) and action else None


def _tool_receipt_value(receipt: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    result = _mapping(receipt.get("result"))
    return _mapping(result.get("value")) if result is not None else None


def _trace_observation(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = _mapping(entry.get("observation"))
    return observation if observation is not None else entry


def _trace_action(entry: Mapping[str, Any]) -> Optional[str]:
    for field_name in ("structured_action", "executed_action"):
        action = _mapping(entry.get(field_name))
        name = action.get("name") if action is not None else None
        if isinstance(name, str) and name:
            return name
    observation = _trace_observation(entry)
    executed = _mapping(observation.get("executed_action"))
    name = executed.get("name") if executed is not None else None
    return name if isinstance(name, str) and name else None


def _provider_failure_from_metadata(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("request_status") == "failed":
        return True
    return any(
        call.get("request_status") == "failed"
        for call in _sequence(metadata.get("model_calls"))
    )


def _normalized_token(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _structured_failure_phase(value: Mapping[str, Any]) -> str:
    """Classify only explicit receipt fields, never free-form error prose."""

    fields = (
        value.get("failure_phase"),
        value.get("phase"),
        value.get("failure_layer"),
        value.get("stage"),
        value.get("classification"),
        value.get("failure_type"),
    )
    tokens = tuple(token for raw in fields if (token := _normalized_token(raw)))
    if any(
        token in _PATCH_APPLY_STATUSES or token == "patch_apply"
        for token in tokens
    ):
        return "patch_apply_failure"
    if any(
        token in _TEST_FAILURE_STATUSES or token in {"test", "grading"}
        for token in tokens
    ):
        return "test_failure"
    if any(
        token in _ENVIRONMENT_PHASES or token in _ENVIRONMENT_STATUSES
        for token in tokens
    ):
        return "environment_failure"
    if any(token in _PROVIDER_PHASES for token in tokens):
        return "provider_failure"
    if value.get("provider_id") or value.get("http_status") is not None:
        if value.get("request_status") == "failed":
            return "provider_failure"
    return "unclassified_failure"


def classify_swebench_evaluation(
    evaluation: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Classify one official SWE-bench evaluation receipt.

    The upstream SkillFlow evaluator exposes a small exact status vocabulary in
    ``harness_details``.  No model output, log text, or ``PASSED`` substring is
    consulted here.
    """

    if not isinstance(evaluation, Mapping):
        return {
            "outcome": "evaluation_missing",
            "failure_phase": "evaluator_failure",
            "resolved": None,
        }
    details = _mapping(evaluation.get("details")) or {}
    metrics = _mapping(evaluation.get("metrics")) or {}
    resolved_detail = details.get("resolved")
    resolved_metric = metrics.get("resolved")
    resolved = (
        resolved_detail
        if isinstance(resolved_detail, bool)
        else resolved_metric == 1.0
        if isinstance(resolved_metric, (int, float))
        and not isinstance(resolved_metric, bool)
        else None
    )
    if evaluation.get("valid") is not True:
        structured = _structured_failure_phase(details)
        return {
            "outcome": (
                structured
                if structured != "unclassified_failure"
                else "evaluator_failure"
            ),
            "failure_phase": (
                structured
                if structured != "unclassified_failure"
                else "evaluator_failure"
            ),
            "resolved": None,
            "evaluator_reason": evaluation.get("reason"),
        }

    status = _normalized_token(details.get("harness_details"))
    if resolved is True or status == "resolved":
        return {"outcome": "resolved", "failure_phase": None, "resolved": True}
    if status == "empty_patch":
        return {
            "outcome": "workspace_patch_failure",
            "failure_phase": "workspace_patch_failure",
            "resolved": False,
        }
    if status in _PATCH_APPLY_STATUSES or status.startswith("patch_apply_failed("):
        return {
            "outcome": "patch_apply_failure",
            "failure_phase": "patch_apply_failure",
            "resolved": False,
        }
    if status in _ENVIRONMENT_STATUSES:
        return {
            "outcome": "environment_failure",
            "failure_phase": "environment_failure",
            "resolved": False,
        }
    if status in _TEST_FAILURE_STATUSES:
        return {
            "outcome": "test_failure",
            "failure_phase": "test_failure",
            "resolved": False,
        }
    structured = _structured_failure_phase(details)
    if structured != "unclassified_failure":
        return {
            "outcome": structured,
            "failure_phase": structured,
            "resolved": False,
        }
    if resolved is False:
        return {
            "outcome": "official_harness_unresolved",
            "failure_phase": "official_harness_unresolved",
            "resolved": False,
        }
    return {
        "outcome": "evaluation_result_invalid",
        "failure_phase": "evaluator_failure",
        "resolved": None,
    }


class _ArmAccumulator:
    def __init__(self) -> None:
        self.raw_tool_actions: Counter[str] = Counter()
        self.grouped_tool_actions: Counter[str] = Counter(
            {name: 0 for name in SWEBENCH_TOOL_ACTION_GROUPS}
        )
        self.invalid_structured_action_statuses: Counter[str] = Counter()
        self.invalid_structured_action_codes: Counter[str] = Counter()
        self.invalid_tool_action_codes: Counter[str] = Counter()
        self.budget_exhausted_action_codes: Counter[str] = Counter()
        self.tool_dispatch_error_types: Counter[str] = Counter()
        self.tool_backend_error_types: Counter[str] = Counter()
        self.tool_result_failure_actions: Counter[str] = Counter()
        self.local_test: Counter[str] = Counter(
            {"passed": 0, "failed": 0, "timed_out": 0, "backend_failure": 0}
        )
        self.provider_failure_types: Counter[str] = Counter()
        self.executor_failure_types: Counter[str] = Counter()
        self.evaluation_outcomes: Counter[str] = Counter()
        self.collection_failure_phases: Counter[str] = Counter()

    def add_response(self, response: Mapping[str, Any]) -> None:
        traces = _new_receipt_sequence(
            response,
            field_name="react_trace",
            continued_count_name="continued_action_history_count",
        )
        receipts = _new_receipt_sequence(
            response,
            field_name="tool_receipts",
            continued_count_name="continued_tool_receipt_count",
        )
        for entry in traces:
            observation = _trace_observation(entry)
            status = observation.get("observation_status")
            code = observation.get("public_error_code")
            if status in {"parse_error", "schema_invalid"}:
                self.invalid_structured_action_statuses[str(status)] += 1
                if isinstance(code, str) and code:
                    self.invalid_structured_action_codes[code] += 1
                    if code in _INVALID_TOOL_ACTION_CODES:
                        self.invalid_tool_action_codes[code] += 1
            elif status == "budget_exhausted":
                if isinstance(code, str) and code:
                    self.budget_exhausted_action_codes[code] += 1
                else:
                    self.budget_exhausted_action_codes[
                        "untyped_action_budget_exhaustion"
                    ] += 1
            elif status == "tool_error":
                error_type = observation.get("error_type")
                self.tool_dispatch_error_types[
                    str(error_type or "untyped_tool_dispatch_error")
                ] += 1

        for receipt in receipts:
            action = _tool_receipt_action(receipt)
            if action is not None:
                self.raw_tool_actions[action] += 1
                for group_name, actions in SWEBENCH_TOOL_ACTION_GROUPS.items():
                    if action in actions:
                        self.grouped_tool_actions[group_name] += 1
            error_type = receipt.get("error_type")
            if isinstance(error_type, str) and error_type:
                self.tool_backend_error_types[error_type] += 1
                if action == "run_tests":
                    self.local_test["backend_failure"] += 1
                continue
            value = _tool_receipt_value(receipt)
            if value is None:
                continue
            if value.get("ok") is False:
                self.tool_result_failure_actions[action or "unknown"] += 1
                if action == "run_tests":
                    self.local_test["backend_failure"] += 1
                continue
            if action != "run_tests":
                continue
            passed = value.get("passed")
            if value.get("timed_out") is True:
                self.local_test["timed_out"] += 1
            elif passed is True:
                self.local_test["passed"] += 1
            elif passed is False:
                self.local_test["failed"] += 1

    def add_execution(self, execution: Mapping[str, Any]) -> None:
        response = _execution_response(execution)
        self.add_response(response)
        failed_calls = [
            call
            for call in _sequence(response.get("model_calls"))
            if call.get("request_status") == "failed"
        ]
        for call in failed_calls:
            error_type = call.get("error_type")
            self.provider_failure_types[
                str(error_type or "untyped_provider_failure")
            ] += 1
        error_type = execution.get("error_type")
        if isinstance(error_type, str) and error_type and not failed_calls:
            self.executor_failure_types[error_type] += 1

    def add_failure_record(self, failure: Mapping[str, Any]) -> None:
        metadata = _mapping(failure.get("metadata")) or {}
        self.add_response(metadata)
        failed_calls = [
            call
            for call in _sequence(metadata.get("model_calls"))
            if call.get("request_status") == "failed"
        ]
        error_type = failure.get("error_type")
        if failed_calls:
            for call in failed_calls:
                self.provider_failure_types[
                    str(call.get("error_type") or "untyped_provider_failure")
                ] += 1
        elif _provider_failure_from_metadata(metadata):
            self.provider_failure_types[
                str(error_type or "untyped_provider_failure")
            ] += 1
        elif isinstance(error_type, str) and error_type:
            self.executor_failure_types[error_type] += 1

    def to_dict(self) -> dict[str, Any]:
        invalid_count = sum(self.invalid_structured_action_statuses.values())
        return {
            "raw_tool_action_counts": dict(sorted(self.raw_tool_actions.items())),
            "tool_action_group_counts": {
                name: int(self.grouped_tool_actions[name])
                for name in SWEBENCH_TOOL_ACTION_GROUPS
            },
            "invalid_structured_action_count": invalid_count,
            "invalid_structured_action_status_counts": dict(
                sorted(self.invalid_structured_action_statuses.items())
            ),
            "invalid_structured_action_code_counts": dict(
                sorted(self.invalid_structured_action_codes.items())
            ),
            "invalid_tool_action_count": sum(self.invalid_tool_action_codes.values()),
            "invalid_tool_action_code_counts": dict(
                sorted(self.invalid_tool_action_codes.items())
            ),
            "structured_action_budget_exhausted_count": sum(
                self.budget_exhausted_action_codes.values()
            ),
            "structured_action_budget_exhausted_code_counts": dict(
                sorted(self.budget_exhausted_action_codes.items())
            ),
            "tool_dispatch_error_count": sum(self.tool_dispatch_error_types.values()),
            "tool_dispatch_error_type_counts": dict(
                sorted(self.tool_dispatch_error_types.items())
            ),
            "tool_backend_error_count": sum(self.tool_backend_error_types.values()),
            "tool_backend_error_type_counts": dict(
                sorted(self.tool_backend_error_types.items())
            ),
            "tool_result_failure_count": sum(
                self.tool_result_failure_actions.values()
            ),
            "tool_result_failure_action_counts": dict(
                sorted(self.tool_result_failure_actions.items())
            ),
            "local_test_receipts": {
                "observed_count": int(
                    self.local_test["passed"]
                    + self.local_test["failed"]
                    + self.local_test["timed_out"]
                ),
                "passed_count": int(self.local_test["passed"]),
                "failed_count": int(self.local_test["failed"]),
                "timeout_count": int(self.local_test["timed_out"]),
                "backend_failure_count": int(self.local_test["backend_failure"]),
            },
            "provider_failure_count": sum(self.provider_failure_types.values()),
            "provider_failure_type_counts": dict(
                sorted(self.provider_failure_types.items())
            ),
            "executor_failure_count": sum(self.executor_failure_types.values()),
            "executor_failure_type_counts": dict(
                sorted(self.executor_failure_types.items())
            ),
            "evaluation_outcome_counts": dict(
                sorted(self.evaluation_outcomes.items())
            ),
            "collection_failure_phase_counts": dict(
                sorted(self.collection_failure_phases.items())
            ),
        }


def _direct_executions(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    direct = _mapping(row.get("direct")) or {}
    executions = _sequence(direct.get("executions"))
    if executions:
        return executions
    execution = _mapping(direct.get("execution"))
    return (execution,) if execution is not None else ()


def _trajectory_executions(
    trajectory: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        execution
        for turn in _sequence(trajectory.get("turns"))
        for execution in _sequence(turn.get("executions"))
    )


def _trajectory_failure_records(
    trajectory: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for turn in _sequence(trajectory.get("turns")):
        runtime = _mapping(turn.get("runtime_summary")) or {}
        result.extend(_sequence(runtime.get("failure_records")))
    return tuple(result)


def _collection_failure_arm(failure: Mapping[str, Any]) -> str:
    condition = failure.get("condition")
    if isinstance(condition, str) and "direct" in condition.casefold():
        return "direct"
    return "agentgraph"


def _row_is_agentgraph_wrong(row: Mapping[str, Any]) -> bool:
    graph = _mapping(row.get("agentgraph")) or {}
    resolved = graph.get("resolved")
    return graph.get("valid") is not True or resolved != 1.0


def _diagnosis(
    *,
    task_id: str,
    failure_layer: str,
    failure_type: str,
    receipt_source: str,
    turn_index: object = None,
    inner_turn: object = None,
    agent_id: object = None,
    action: object = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "diagnosis_scope": "first_observable_failure",
        "causal_attribution": False,
        "interpretation": (
            "earliest persisted failure receipt; not proof of causal attribution"
        ),
        "failure_layer": failure_layer,
        "failure_type": failure_type,
        "turn_index": turn_index,
        "inner_turn": inner_turn,
        "agent_id": agent_id,
        "action": action,
        "receipt_source": receipt_source,
    }


def _trace_failure_event(entry: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    observation = _trace_observation(entry)
    status = observation.get("observation_status")
    code = observation.get("public_error_code")
    if status in {"parse_error", "schema_invalid", "budget_exhausted"}:
        return "structured_tool_action", str(code or status)
    if status == "tool_error":
        return "tool_dispatch", str(
            observation.get("error_type") or "untyped_tool_dispatch_error"
        )
    result = _mapping(observation.get("result"))
    action = _trace_action(entry)
    if result is not None and result.get("ok") is False:
        return "tool_result", f"{action or 'tool'}_result_failure"
    if action == "run_tests" and result is not None:
        if result.get("timed_out") is True:
            return "tool_result", "local_test_timeout"
        if result.get("passed") is False:
            return "tool_result", "local_test_failed"
    return None


def _first_execution_failure(
    execution: Mapping[str, Any],
) -> Optional[tuple[str, str, object, object]]:
    response = _execution_response(execution)
    traces = _new_receipt_sequence(
        response,
        field_name="react_trace",
        continued_count_name="continued_action_history_count",
    )
    model_calls = _sequence(response.get("model_calls"))
    events: list[tuple[int, int, str, str, object]] = []
    for index, call in enumerate(model_calls):
        if call.get("request_status") != "failed":
            continue
        turn = call.get("turn")
        order = turn if isinstance(turn, int) and not isinstance(turn, bool) else index
        events.append(
            (
                order,
                0,
                "provider_execution",
                str(call.get("error_type") or "provider_request_failed"),
                None,
            )
        )
    for index, entry in enumerate(traces):
        failure = _trace_failure_event(entry)
        if failure is None:
            continue
        turn = entry.get("turn")
        order = turn if isinstance(turn, int) and not isinstance(turn, bool) else index
        events.append((order, 1, failure[0], failure[1], _trace_action(entry)))
    if events:
        first = min(events, key=lambda item: (item[0], item[1]))
        return first[2], first[3], first[0], first[4]

    for receipt in _new_receipt_sequence(
        response,
        field_name="tool_receipts",
        continued_count_name="continued_tool_receipt_count",
    ):
        action = _tool_receipt_action(receipt)
        error_type = receipt.get("error_type")
        if isinstance(error_type, str) and error_type:
            return "tool_dispatch", error_type, None, action
        value = _tool_receipt_value(receipt)
        if value is not None and value.get("ok") is False:
            return "tool_result", f"{action or 'tool'}_result_failure", None, action
        if action == "run_tests" and value is not None:
            if value.get("timed_out") is True:
                return "tool_result", "local_test_timeout", None, action
            if value.get("passed") is False:
                return "tool_result", "local_test_failed", None, action

    error_type = execution.get("error_type")
    if isinstance(error_type, str) and error_type:
        layer = (
            "provider_execution"
            if _provider_failure_from_metadata(response)
            else "executor_runtime"
        )
        return layer, error_type, None, None
    return None


def _evaluation_diagnosis(
    task_id: str,
    evaluation: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    classified = classify_swebench_evaluation(evaluation)
    outcome = str(classified["outcome"])
    layer = {
        "patch_apply_failure": "official_harness_patch_apply",
        "test_failure": "official_harness_test",
        "official_harness_unresolved": "official_harness_test",
        "environment_failure": "official_harness_environment",
        "provider_failure": "provider_execution",
        "workspace_patch_failure": "workspace_patch",
    }.get(outcome, "official_harness_evaluator")
    return _diagnosis(
        task_id=task_id,
        failure_layer=layer,
        failure_type=outcome,
        receipt_source="official_swebench_evaluation",
    )


def diagnose_swebench_wrong_demo(
    row: Mapping[str, Any],
    *,
    trajectory: Optional[Mapping[str, Any]] = None,
    collection_failures: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Return the first observable AgentGraph failure for one SWE-bench row."""

    task_id = row.get("task_id")
    task_id = task_id if isinstance(task_id, str) else ""
    graph = _mapping(row.get("agentgraph")) or {}
    matching_collection = [
        failure
        for failure in collection_failures
        if isinstance(failure, Mapping)
        and failure.get("task_id") == task_id
        and _collection_failure_arm(failure) == "agentgraph"
    ]
    if trajectory is None:
        if matching_collection:
            phase = _structured_failure_phase(matching_collection[0])
            layer = {
                "provider_failure": "provider_execution",
                "environment_failure": "repository_environment",
                "patch_apply_failure": "official_harness_patch_apply",
                "test_failure": "official_harness_test",
            }.get(phase, "collection_runtime")
            return _diagnosis(
                task_id=task_id,
                failure_layer=layer,
                failure_type=phase,
                receipt_source="collection_failure",
            )
        return _diagnosis(
            task_id=task_id,
            failure_layer="collection_runtime",
            failure_type="trajectory_missing",
            receipt_source="paired_result",
        )

    for turn_position, turn in enumerate(_sequence(trajectory.get("turns"))):
        round_index = turn.get("round_index", turn_position)
        action = _mapping(turn.get("action"))
        action_name = action.get("action") if action is not None else None
        if not isinstance(action_name, str) or not action_name:
            return _diagnosis(
                task_id=task_id,
                failure_layer="director_action_parsing",
                failure_type="invalid_or_unparsed_director_action",
                receipt_source="canvas_turn",
                turn_index=round_index,
            )
        feedback = turn.get("canvas_feedback")
        accepted = isinstance(feedback, str) and (
            feedback.startswith("accepted ") or feedback == "workflow finished"
        )
        if not accepted:
            return _diagnosis(
                task_id=task_id,
                failure_layer="canvas_action_validation",
                failure_type="canvas_action_rejected",
                receipt_source="canvas_turn",
                turn_index=round_index,
                action=action_name,
            )

        executions = _sequence(turn.get("executions"))
        for execution in executions:
            failure = _first_execution_failure(execution)
            if failure is None:
                continue
            return _diagnosis(
                task_id=task_id,
                failure_layer=failure[0],
                failure_type=failure[1],
                receipt_source="execution_receipt",
                turn_index=round_index,
                inner_turn=failure[2],
                agent_id=execution.get("agent_id"),
                action=failure[3],
            )
        runtime = _mapping(turn.get("runtime_summary")) or {}
        for failure_record in _sequence(runtime.get("failure_records")):
            synthetic_execution = {
                "agent_id": failure_record.get("agent_id"),
                "error_type": failure_record.get("error_type"),
                "metadata": {
                    "response": _mapping(failure_record.get("metadata")) or {}
                },
            }
            failure = _first_execution_failure(synthetic_execution)
            if failure is None:
                continue
            return _diagnosis(
                task_id=task_id,
                failure_layer=failure[0],
                failure_type=failure[1],
                receipt_source="runtime_failure_record",
                turn_index=round_index,
                inner_turn=failure[2],
                agent_id=failure_record.get("agent_id"),
                action=failure[3],
            )
        if runtime.get("execution_status") == "failed":
            return _diagnosis(
                task_id=task_id,
                failure_layer="executor_runtime",
                failure_type="runtime_failed_without_structured_failure_record",
                receipt_source="runtime_summary",
                turn_index=round_index,
            )

    final_answer = trajectory.get("final_answer", graph.get("final_answer"))
    if not isinstance(final_answer, str) or not final_answer.strip():
        return _diagnosis(
            task_id=task_id,
            failure_layer="workspace_patch",
            failure_type="workspace_patch_missing",
            receipt_source="trajectory_terminal",
        )
    if trajectory.get("explicit_finish", graph.get("explicit_finish")) is not True:
        return _diagnosis(
            task_id=task_id,
            failure_layer="finish",
            failure_type="explicit_finish_missing",
            receipt_source="trajectory_terminal",
        )
    evaluation = _mapping(graph.get("evaluation"))
    if evaluation is None:
        evaluation = _mapping(trajectory.get("evaluation"))
    return _evaluation_diagnosis(task_id, evaluation)


def aggregate_swebench_receipts(
    paired_rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]] = (),
    collection_failures: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Aggregate persisted Direct/AgentGraph SWE-bench receipts offline."""

    rows = tuple(row for row in paired_rows if isinstance(row, Mapping))
    trajectory_by_task = _trajectory_index(trajectories)
    accumulators = {"direct": _ArmAccumulator(), "agentgraph": _ArmAccumulator()}

    for row in rows:
        for execution in _direct_executions(row):
            accumulators["direct"].add_execution(execution)
        task_id = row.get("task_id")
        trajectory = (
            trajectory_by_task.get(task_id)
            if isinstance(task_id, str)
            else None
        )
        if trajectory is not None:
            for execution in _trajectory_executions(trajectory):
                accumulators["agentgraph"].add_execution(execution)
            for failure in _trajectory_failure_records(trajectory):
                accumulators["agentgraph"].add_failure_record(failure)
        direct = _mapping(row.get("direct")) or {}
        graph = _mapping(row.get("agentgraph")) or {}
        for arm, value in (("direct", direct), ("agentgraph", graph)):
            classified = classify_swebench_evaluation(
                _mapping(value.get("evaluation"))
            )
            accumulators[arm].evaluation_outcomes[str(classified["outcome"])] += 1

    failures = tuple(
        failure for failure in collection_failures if isinstance(failure, Mapping)
    )
    for failure in failures:
        arm = _collection_failure_arm(failure)
        accumulators[arm].collection_failure_phases[
            _structured_failure_phase(failure)
        ] += 1

    diagnoses = [
        diagnose_swebench_wrong_demo(
            row,
            trajectory=(
                trajectory_by_task.get(row.get("task_id"))
                if isinstance(row.get("task_id"), str)
                else None
            ),
            collection_failures=failures,
        )
        for row in rows
        if _row_is_agentgraph_wrong(row)
    ]
    return {
        "schema_version": "flowsteer.swebench.offline-reporting.v1",
        "receipt_policy": {
            "source": "persisted_structured_receipts_only",
            "resolution_metric": "official_swebench_harness_resolved",
            "llm_judge_used": False,
            "passed_string_parsing_used": False,
            "wrong_demo_diagnosis": "first_observable_failure_not_causal_attribution",
        },
        "tool_action_groups": {
            name: list(actions)
            for name, actions in SWEBENCH_TOOL_ACTION_GROUPS.items()
        },
        "arms": {
            arm: accumulator.to_dict()
            for arm, accumulator in accumulators.items()
        },
        "wrong_demo_count": len(diagnoses),
        "wrong_demo_diagnoses": diagnoses,
    }


__all__ = [
    "SWEBENCH_TOOL_ACTION_GROUPS",
    "aggregate_swebench_receipts",
    "classify_swebench_evaluation",
    "diagnose_swebench_wrong_demo",
]
