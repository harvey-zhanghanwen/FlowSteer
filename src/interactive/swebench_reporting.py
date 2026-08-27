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
    "edit": ("apply_patch", "edit_file", "exact_edit", "str_replace_editor"),
    "test": ("run_tests",),
    "command": ("bash",),
}

# These categories follow the persisted boundaries that exist in the reused
# SkillFlow SWE-bench repository/evaluator path and the FlowSteer
# Director/Canvas/runtime path.  They are task-attempt categories, not run
# preflight categories: missing global environments or Docker access must be
# reported separately as run blockers until a task execution has started.
SWEBENCH_FAILURE_TAXONOMY: tuple[Mapping[str, str], ...] = (
    {
        "code": "collection_receipt_failure",
        "label": "collection / receipt failure",
    },
    {
        "code": "provider_failure",
        "label": "model provider failure",
    },
    {
        "code": "repository_environment_failure",
        "label": "task-scoped repository environment failure",
    },
    {
        "code": "orchestration_failure",
        "label": "Director / Canvas orchestration failure",
    },
    {
        "code": "agent_communication_failure",
        "label": "Agent communication / artifact routing failure",
    },
    {
        "code": "repository_tool_failure",
        "label": "repository Tool protocol / execution failure",
    },
    {
        "code": "local_validation_failure",
        "label": "local test validation failure / timeout",
    },
    {
        "code": "terminal_budget_failure",
        "label": "FINISH / round or Tool budget failure",
    },
    {
        "code": "patch_publication_application_failure",
        "label": "workspace patch publication / application failure",
    },
    {
        "code": "official_target_test_failure",
        "label": "official FAIL_TO_PASS target-test failure",
    },
    {
        "code": "official_regression_failure",
        "label": "official PASS_TO_PASS regression",
    },
    {
        "code": "official_test_failure_unclassified",
        "label": "official unresolved test outcome without F2P/P2P detail",
    },
    {
        "code": "evaluator_runtime_failure",
        "label": "official evaluator / harness runtime failure",
    },
    {
        "code": "unclassified_receipt_failure",
        "label": "unclassified structured receipt failure",
    },
)

_TAXONOMY_CODES = frozenset(
    str(category["code"]) for category in SWEBENCH_FAILURE_TAXONOMY
)

_COMMUNICATION_FAILURE_CODES = frozenset(
    {
        "artifact_routing_failed",
        "blocked_by_upstream",
        "communication_contract_violation",
        "dependency_artifact_missing",
        "missing_upstream_artifact",
        "upstream_artifact_missing",
    }
)

_EVALUATOR_ONLY_REPORT_FIELDS = frozenset(
    {
        "evaluator_payload",
        "fail_to_pass",
        "gold_patch",
        "ground_truth",
        "pass_to_pass",
        "test_patch",
    }
)

_SAFE_HARNESS_DETAIL_STATUSES = frozenset(
    {
        "empty_patch",
        "env_not_ready",
        "image_build_failed",
        "image_missing_or_pull_failed",
        "patch_apply_failed",
        "resolved",
        "test_patch_apply_failed",
        "test_timeout",
        "unresolved",
        "worktree_create_failed",
    }
)

_TASK_EXECUTION_COLLECTION_STAGES = frozenset(
    {
        "agent_execution",
        "agentgraph_collection",
        "director_canvas_runtime_or_evaluator",
        "generation_or_evaluator",
        "model_request",
        "official_evaluator",
        "provider_request",
        "task_execution",
    }
)

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


def _public_receipt_value(value: object) -> object:
    """Project a persisted receipt without evaluator-only target material."""

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized_key = _normalized_token(key)
            if normalized_key in _EVALUATOR_ONLY_REPORT_FIELDS:
                projected[key] = None
                projected[f"{key}_role"] = "evaluator_only_redacted"
                continue
            if normalized_key == "harness_details":
                if isinstance(nested, Mapping):
                    projected[key] = _public_receipt_value(nested)
                elif _normalized_token(nested) in _SAFE_HARNESS_DETAIL_STATUSES:
                    projected[key] = nested
                else:
                    projected[key] = None
                    projected[f"{key}_role"] = "evaluator_only_redacted"
                continue
            if normalized_key == "tests_status" and isinstance(nested, Mapping):
                summary: dict[str, Any] = {}
                for suite_name in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                    suite = _mapping(nested.get(suite_name))
                    success = suite.get("success") if suite is not None else None
                    summary[suite_name] = {
                        "success_count": (
                            len(success)
                            if isinstance(success, Sequence)
                            and not isinstance(success, (str, bytes))
                            else None
                        ),
                        "failure_count": _structured_test_failure_count(suite),
                        "test_ids": None,
                        "test_ids_role": "evaluator_only_redacted",
                    }
                projected["tests_status_summary"] = summary
                continue
            projected[key] = _public_receipt_value(nested)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_public_receipt_value(item) for item in value]
    return value


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


def _evaluation_for_wrong_demo(
    row: Mapping[str, Any],
    trajectory: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    graph = _mapping(row.get("agentgraph")) or {}
    evaluation = _mapping(graph.get("evaluation"))
    if evaluation is None and trajectory is not None:
        evaluation = _mapping(trajectory.get("evaluation"))
    return evaluation


def _structured_test_failure_count(value: object) -> Optional[int]:
    status = _mapping(value)
    if status is None:
        return None
    for field_name in ("failure", "failures", "failed"):
        failures = status.get(field_name)
        if isinstance(failures, Sequence) and not isinstance(
            failures, (str, bytes)
        ):
            return len(failures)
        if isinstance(failures, int) and not isinstance(failures, bool):
            return max(0, failures)
    return None


def _official_test_failure_assignment(
    evaluation: Optional[Mapping[str, Any]],
) -> tuple[str, str]:
    """Use structured official test status when it was actually persisted.

    SWE-bench's report distinguishes FAIL_TO_PASS target tests from
    PASS_TO_PASS regression tests.  The current reused SkillFlow wrapper often
    persists only ``harness_details='unresolved'``; that lossy receipt must stay
    explicitly unclassified rather than being guessed from model prose or log
    text.
    """

    details = _mapping(evaluation.get("details")) if evaluation is not None else None
    details = details or {}
    status = _mapping(details.get("tests_status"))
    if status is None:
        harness = _mapping(details.get("harness_details"))
        status = _mapping(harness.get("tests_status")) if harness is not None else None
    if status is None:
        return (
            "official_test_failure_unclassified",
            "official_unresolved_without_structured_f2p_p2p_status",
        )
    f2p_failed = _structured_test_failure_count(status.get("FAIL_TO_PASS"))
    p2p_failed = _structured_test_failure_count(status.get("PASS_TO_PASS"))
    if isinstance(p2p_failed, int) and p2p_failed > 0:
        return (
            "official_regression_failure",
            (
                "target_and_regression_tests_failed"
                if isinstance(f2p_failed, int) and f2p_failed > 0
                else "pass_to_pass_regression"
            ),
        )
    if isinstance(f2p_failed, int) and f2p_failed > 0:
        return "official_target_test_failure", "fail_to_pass_target_not_fixed"
    return (
        "official_test_failure_unclassified",
        "official_unresolved_without_failed_test_breakdown",
    )


def _diagnosis_category(diagnosis: Mapping[str, Any]) -> tuple[str, str]:
    layer = _normalized_token(diagnosis.get("failure_layer"))
    failure_type = _normalized_token(diagnosis.get("failure_type"))
    action = _normalized_token(diagnosis.get("action"))
    if failure_type in _COMMUNICATION_FAILURE_CODES or layer in {
        "agent_communication",
        "artifact_routing",
    }:
        return "agent_communication_failure", failure_type or layer
    if layer in {"collection_runtime"}:
        return "collection_receipt_failure", failure_type or layer
    if layer in {"provider_execution"}:
        return "provider_failure", failure_type or layer
    if layer in {
        "official_harness_environment",
        "repository_environment",
    }:
        return "repository_environment_failure", failure_type or layer
    if layer in {
        "canvas_action_validation",
        "director_action_parsing",
        "graph_scheduler",
        "orchestration",
    }:
        return "orchestration_failure", failure_type or layer
    if failure_type in {
        "canvas_action_domain_exhausted",
        "explicit_finish_missing",
        "max_rounds",
        "max_rounds_exhausted",
        "tool_call_budget_exhausted",
    } or layer in {"finish", "terminal_budget"}:
        return "terminal_budget_failure", failure_type or layer
    if failure_type in {"local_test_failed", "local_test_timeout"}:
        return "local_validation_failure", failure_type
    if layer in {"structured_tool_action", "tool_dispatch", "tool_result"}:
        if failure_type.endswith("budget_exhausted"):
            return "terminal_budget_failure", failure_type
        return "repository_tool_failure", failure_type or action or layer
    if layer in {"official_harness_patch_apply", "workspace_patch"}:
        return "patch_publication_application_failure", failure_type or layer
    if layer == "official_harness_test":
        return (
            "official_test_failure_unclassified",
            failure_type or "official_unresolved_without_structured_f2p_p2p_status",
        )
    if layer in {"official_harness_evaluator"}:
        return "evaluator_runtime_failure", failure_type or layer
    if layer in {"executor_runtime"}:
        return "collection_receipt_failure", failure_type or layer
    return "unclassified_receipt_failure", failure_type or layer or "unknown"


def _wrong_demo_taxonomy_assignment(
    row: Mapping[str, Any],
    trajectory: Optional[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
) -> Mapping[str, str]:
    """Assign one mutually exclusive primary observable category.

    This is an outcome-oriented taxonomy: a valid official terminal receipt
    takes precedence over intermediate observations, without claiming that an
    intermediate failure recovered or caused the terminal outcome.  Terminal
    non-completion takes precedence when the evaluator was not admissible.
    When neither exists, the first saved failure receipt is used.  None of
    these rules is presented as root-cause proof.
    """

    graph = _mapping(row.get("agentgraph")) or {}
    termination_reason = _normalized_token(
        (
            trajectory.get("termination_reason")
            if trajectory is not None
            else graph.get("termination_reason")
        )
    )
    diagnosis_layer = _normalized_token(diagnosis.get("failure_layer"))
    if diagnosis_layer in {"finish", "workspace_patch"}:
        category, subtype = _diagnosis_category(diagnosis)
        basis = "terminal_completion_receipt"
    elif termination_reason in {
        "canvas_action_domain_exhausted",
        "max_rounds",
        "max_rounds_exhausted",
        "tool_call_budget_exhausted",
    }:
        category, subtype = "terminal_budget_failure", termination_reason
        basis = "terminal_completion_receipt"
    else:
        category = ""
        subtype = ""
        basis = ""
    evaluation = _evaluation_for_wrong_demo(row, trajectory)
    classified = classify_swebench_evaluation(evaluation)
    outcome = str(classified["outcome"])
    if category:
        pass
    elif outcome == "patch_apply_failure" or outcome == "workspace_patch_failure":
        category, subtype = (
            "patch_publication_application_failure",
            outcome,
        )
        basis = "official_terminal_evaluator_receipt"
    elif outcome in {"test_failure", "official_harness_unresolved"}:
        category, subtype = _official_test_failure_assignment(evaluation)
        basis = "official_terminal_evaluator_receipt"
    elif outcome == "environment_failure":
        category, subtype = "repository_environment_failure", outcome
        basis = "task_scoped_evaluator_receipt"
    elif outcome in {"evaluator_failure", "evaluation_result_invalid"}:
        category, subtype = "evaluator_runtime_failure", outcome
        basis = "official_evaluator_receipt"
    elif outcome == "provider_failure":
        category, subtype = "provider_failure", outcome
        basis = "structured_provider_receipt"
    else:
        category, subtype = _diagnosis_category(diagnosis)
        basis = "first_observable_failure_receipt"
    if category not in _TAXONOMY_CODES:
        category = "unclassified_receipt_failure"
    return {
        "category": category,
        "subtype": subtype,
        "assignment_basis": basis,
        "causal_attribution": "not_established",
    }


def _agent_execution_report(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _mapping(execution.get("metadata"))
    request = _mapping(metadata.get("request")) if metadata is not None else None
    response = _mapping(metadata.get("response")) if metadata is not None else None
    if response is None and metadata is not None and any(
        field_name in metadata
        for field_name in ("model_calls", "react_trace", "tool_receipts")
    ):
        response = metadata
    agent = _mapping(request.get("agent")) if request is not None else None

    def receipt_sequence(field_name: str) -> Optional[list[object]]:
        if response is None or field_name not in response:
            return None
        return list(_sequence(response.get(field_name)))

    return {
        "execution_id": execution.get("execution_id"),
        "agent_id": execution.get("agent_id"),
        "model_id": execution.get("model_id"),
        "provider": execution.get("provider"),
        "contract": agent.get("contract") if agent is not None else None,
        "agent_input": (
            _public_receipt_value(dict(request)) if request is not None else None
        ),
        "agent_output": execution.get("output"),
        "agent_communication": (
            {
                "phase": request.get("phase"),
                "upstream": _public_receipt_value(request.get("upstream")),
                "own_draft": _public_receipt_value(request.get("own_draft")),
                "peer_draft": _public_receipt_value(request.get("peer_draft")),
            }
            if request is not None
            else None
        ),
        "react_tool_execution": (
            {
                "model_calls": _public_receipt_value(
                    receipt_sequence("model_calls")
                ),
                "action_observation_trace": _public_receipt_value(
                    receipt_sequence("react_trace")
                ),
                "tool_receipts": _public_receipt_value(
                    receipt_sequence("tool_receipts")
                ),
                "continued_action_history_count": response.get(
                    "continued_action_history_count"
                ),
                "continued_tool_receipt_count": response.get(
                    "continued_tool_receipt_count"
                ),
            }
            if response is not None
            else None
        ),
        "execution_receipt": {
            "error_type": execution.get("error_type"),
            "latency_ms": execution.get("latency_ms"),
            "input_tokens": execution.get("input_tokens"),
            "output_tokens": execution.get("output_tokens"),
            "model_fingerprint": execution.get("model_fingerprint"),
        },
    }


def _director_canvas_turn_report(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    executions = (
        _sequence(turn.get("executions"))
        if "executions" in turn
        else None
    )
    return {
        "round_index": turn.get("round_index"),
        "director": {
            "input_prompt": _public_receipt_value(turn.get("prompt")),
            "raw_output": _public_receipt_value(turn.get("policy_response")),
            "parsed_action": _public_receipt_value(turn.get("action")),
            "policy_version": turn.get("policy_version"),
            "policy_adapter": turn.get("policy_adapter"),
            "request_id": turn.get("director_request_id"),
            "attempt_count": turn.get("director_attempt_count"),
            "latency_ms": turn.get("director_latency_ms"),
        },
        "canvas_edit": {
            "action": _public_receipt_value(turn.get("action")),
            "feedback": _public_receipt_value(turn.get("canvas_feedback")),
            "graph_revision": turn.get("graph_revision"),
            "previous_graph_snapshot_id": turn.get("previous_graph_snapshot_id"),
            "graph_snapshot_id": turn.get("graph_snapshot_id"),
            "graph_snapshot": _public_receipt_value(turn.get("graph_snapshot")),
        },
        "agent_executions": (
            [_agent_execution_report(execution) for execution in executions]
            if executions is not None
            else None
        ),
        "runtime_receipt": (
            _public_receipt_value(turn.get("runtime_summary"))
            if "runtime_summary" in turn
            else None
        ),
    }


def _observed_post_failure_receipts(
    trajectory: Optional[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    evaluation: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    turns = _sequence(trajectory.get("turns")) if trajectory is not None else ()
    first_turn = diagnosis.get("turn_index")
    later_turn_count: Optional[int]
    later_turns: Optional[list[Mapping[str, Any]]]
    if isinstance(first_turn, int) and not isinstance(first_turn, bool):
        later_turns = [
            turn
            for turn in turns
            if isinstance(turn.get("round_index"), int)
            and turn.get("round_index") > first_turn
        ]
        later_turn_count = len(later_turns)
    elif trajectory is not None:
        later_turns = []
        later_turn_count = 0
    else:
        later_turns = None
        later_turn_count = None
    return {
        "interpretation": (
            "receipts observed after the first failure; temporal order is not "
            "proof of causal propagation"
        ),
        "later_turn_count": later_turn_count,
        "subsequent_turn_receipts": (
            [_director_canvas_turn_report(turn) for turn in later_turns]
            if later_turns is not None
            else None
        ),
        "terminal_receipt": (
            {
                "explicit_finish": trajectory.get("explicit_finish"),
                "termination_reason": trajectory.get("termination_reason"),
                "terminal_failure": trajectory.get("terminal_failure"),
                "final_patch_present": bool(
                    isinstance(trajectory.get("final_answer"), str)
                    and trajectory.get("final_answer", "").strip()
                ),
            }
            if trajectory is not None
            else None
        ),
        "evaluator_receipt": _public_receipt_value(evaluation),
    }


def _representative_wrong_demo(
    row: Mapping[str, Any],
    trajectory: Optional[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    assignment: Mapping[str, str],
    collection_failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    task_id = row.get("task_id")
    graph = _mapping(row.get("agentgraph")) or {}
    repository_state = _mapping(row.get("repository_state")) or {}
    evaluation = _evaluation_for_wrong_demo(row, trajectory)
    matching_collection = [
        _public_receipt_value(dict(failure))
        for failure in collection_failures
        if failure.get("task_id") == task_id
        and _collection_failure_arm(failure) == "agentgraph"
    ]
    terminal_receipt = (
        {
            "final_patch": trajectory.get("final_answer"),
            "explicit_finish": trajectory.get("explicit_finish"),
            "termination_reason": trajectory.get("termination_reason"),
            "terminal_failure": trajectory.get("terminal_failure"),
            "valid_lineage_fallback_used": trajectory.get(
                "valid_lineage_fallback_used"
            ),
            "valid_lineage_fallback_receipt": _public_receipt_value(
                trajectory.get("valid_lineage_fallback_receipt")
            ),
        }
        if trajectory is not None
        else None
    )
    return {
        "schema_version": "flowsteer.swebench.wrong-demo.v2",
        "demo_id": f"{task_id}:{assignment['category']}",
        "task_id": task_id,
        "sample_id": repository_state.get("instance_id", task_id),
        "arm": "agentgraph",
        "input": {
            "question": row.get("question"),
            "repository_state": dict(repository_state),
        },
        "reference_target": {
            "metric": "official_swebench_harness_resolved",
            "expected_resolved": True,
            "gold_patch": None,
            "fail_to_pass": None,
            "pass_to_pass": None,
            "redaction_role": "evaluator_only_redacted",
        },
        "system_result": {
            "final_patch": graph.get("final_answer"),
            "metric_name": row.get("primary_metric", "resolved"),
            "metric_value": graph.get("resolved"),
            "evaluator_valid": graph.get("valid"),
            "explicit_finish": graph.get("explicit_finish"),
            "termination_reason": graph.get("termination_reason"),
        },
        "execution_chain": {
            "trajectory_id": (
                trajectory.get("trajectory_id")
                if trajectory is not None
                else graph.get("trajectory_id")
            ),
            "director_canvas_turns": (
                [
                    _director_canvas_turn_report(turn)
                    for turn in _sequence(trajectory.get("turns"))
                ]
                if trajectory is not None
                and "turns" in trajectory
                else None
            ),
            "output_agent_inbox": _public_receipt_value(
                graph.get("output_agent_inbox")
            ),
            "terminal_receipt": terminal_receipt,
            "evaluator_receipt": _public_receipt_value(evaluation),
        },
        "failure_analysis": {
            "primary_observable_category": assignment["category"],
            "subtype": assignment["subtype"],
            "assignment_basis": assignment["assignment_basis"],
            "first_observable_failure": dict(diagnosis),
            "first_causal_failure": None,
            "causality_status": "not_established",
            "causal_error_propagation": None,
            "subsequent_observed_receipts": _observed_post_failure_receipts(
                trajectory,
                diagnosis,
                evaluation,
            ),
        },
        "source_receipts": {
            "join_key": task_id,
            "trajectory_present": trajectory is not None,
            "collection_failures": matching_collection,
        },
    }


def _agentgraph_collection_failures_for_task(
    task_id: str,
    collection_failures: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        failure
        for failure in collection_failures
        if failure.get("task_id") == task_id
        and _collection_failure_arm(failure) == "agentgraph"
    )


def _task_attempt_started(
    row: Mapping[str, Any],
    trajectory: Optional[Mapping[str, Any]],
    collection_failures: Sequence[Mapping[str, Any]],
) -> bool:
    """Exclude run-level preflight placeholders from task failure counts."""

    if trajectory is not None:
        return True
    graph = _mapping(row.get("agentgraph")) or {}
    if graph.get("available") is True or graph.get("trajectory_id") is not None:
        return True
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return False
    for failure in _agentgraph_collection_failures_for_task(
        task_id,
        collection_failures,
    ):
        if failure.get("execution_started") is True:
            return True
        if _normalized_token(failure.get("stage")) in _TASK_EXECUTION_COLLECTION_STAGES:
            return True
    return False


def build_swebench_wrong_demo_taxonomy(
    paired_rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]] = (),
    collection_failures: Sequence[Mapping[str, Any]] = (),
    *,
    representative_demo_limit: int = 1,
) -> Mapping[str, Any]:
    """Build a receipt-only SWE-bench failure taxonomy and reproducible demos.

    Counts use one mutually exclusive primary observable category per failed
    AgentGraph task.  Shares use the number of evaluated/collected wrong tasks
    as denominator.  With no wrong task, every count is explicitly zero while
    every share is ``None``/N/A; run preflight blockers remain outside this
    task-level schema.
    """

    if not isinstance(representative_demo_limit, int) or isinstance(
        representative_demo_limit, bool
    ) or not 1 <= representative_demo_limit <= 3:
        raise ValueError("representative_demo_limit must be an integer in [1, 3]")
    candidate_rows = tuple(
        row
        for row in paired_rows
        if isinstance(row, Mapping) and _row_is_agentgraph_wrong(row)
    )
    trajectory_by_task = _trajectory_index(trajectories)
    failures = tuple(
        failure for failure in collection_failures if isinstance(failure, Mapping)
    )
    assignments: list[
        tuple[
            Mapping[str, Any],
            Optional[Mapping[str, Any]],
            Mapping[str, Any],
            Mapping[str, str],
        ]
    ] = []
    seen_task_ids: set[str] = set()
    for row in candidate_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("every SWE-bench paired row must have a non-empty task_id")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate SWE-bench paired task_id: {task_id}")
        seen_task_ids.add(task_id)
        trajectory = (
            trajectory_by_task.get(task_id)
        )
        if not _task_attempt_started(row, trajectory, failures):
            continue
        diagnosis = diagnose_swebench_wrong_demo(
            row,
            trajectory=trajectory,
            collection_failures=failures,
        )
        assignment = _wrong_demo_taxonomy_assignment(row, trajectory, diagnosis)
        assignments.append((row, trajectory, diagnosis, assignment))

    denominator = len(assignments)
    counts = Counter(
        assignment["category"] for _, _, _, assignment in assignments
    )
    categories: list[dict[str, Any]] = []
    all_demos: list[Mapping[str, Any]] = []
    for definition in SWEBENCH_FAILURE_TAXONOMY:
        code = str(definition["code"])
        selected = [
            value for value in assignments if value[3]["category"] == code
        ]
        demos = [
            _representative_wrong_demo(
                row,
                trajectory,
                diagnosis,
                assignment,
                failures,
            )
            for row, trajectory, diagnosis, assignment in selected[
                :representative_demo_limit
            ]
        ]
        all_demos.extend(demos)
        count = int(counts[code])
        categories.append(
            {
                **dict(definition),
                "count": count,
                "share": count / denominator if denominator else None,
                "share_status": (
                    "computed_over_wrong_task_denominator"
                    if denominator
                    else "not_applicable_no_evaluated_wrong_demo"
                ),
                "task_ids": [
                    row.get("task_id") for row, _, _, _ in selected
                ],
                "representative_demo_ids": [demo["demo_id"] for demo in demos],
            }
        )
    return {
        "schema_version": "flowsteer.swebench.failure-taxonomy.v2",
        "classification_scope": "task_level_agentgraph_wrong_attempts",
        "counting_rule": (
            "one mutually exclusive primary observable category per wrong task; "
            "official terminal receipts override recoverable intermediate observations"
        ),
        "denominator": denominator,
        "denominator_status": (
            "available" if denominator else "no_evaluated_wrong_demo"
        ),
        "share_unit": "fraction_of_wrong_tasks",
        "causal_attribution_policy": (
            "first observable receipt is reported; causal failure and causal "
            "propagation remain null without explicit causal or intervention evidence"
        ),
        "run_preflight_blockers_included": False,
        "categories": categories,
        "representative_demos": all_demos,
        "not_applicable": {
            "answer_string_canonicalization": (
                "SWE-bench is scored by patch application and official tests, not "
                "answer-string canonicalization"
            ),
            "external_information_retrieval": (
                "the initial adapter exposes repository search/view Tools; no external "
                "retrieval service is part of this protocol"
            ),
            "llm_judge": "not used by the official Resolved evaluator",
        },
    }


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

    diagnoses: list[Mapping[str, Any]] = []
    seen_wrong_task_ids: set[str] = set()
    for row in rows:
        if not _row_is_agentgraph_wrong(row):
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("every SWE-bench paired row must have a non-empty task_id")
        if task_id in seen_wrong_task_ids:
            raise ValueError(f"duplicate SWE-bench paired task_id: {task_id}")
        seen_wrong_task_ids.add(task_id)
        trajectory = trajectory_by_task.get(task_id)
        if not _task_attempt_started(row, trajectory, failures):
            continue
        diagnoses.append(
            diagnose_swebench_wrong_demo(
                row,
                trajectory=trajectory,
                collection_failures=failures,
            )
        )
    failure_taxonomy = build_swebench_wrong_demo_taxonomy(
        rows,
        trajectories,
        failures,
    )
    return {
        "schema_version": "flowsteer.swebench.offline-reporting.v2",
        "receipt_policy": {
            "source": "persisted_structured_receipts_only",
            "resolution_metric": "official_swebench_harness_resolved",
            "llm_judge_used": False,
            "passed_string_parsing_used": False,
            "wrong_demo_diagnosis": "first_observable_failure_not_causal_attribution",
            "failure_taxonomy": (
                "mutually_exclusive_primary_observable_category_not_root_cause"
            ),
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
        "wrong_demo_failure_taxonomy": failure_taxonomy,
    }


__all__ = [
    "SWEBENCH_FAILURE_TAXONOMY",
    "SWEBENCH_TOOL_ACTION_GROUPS",
    "aggregate_swebench_receipts",
    "build_swebench_wrong_demo_taxonomy",
    "classify_swebench_evaluation",
    "diagnose_swebench_wrong_demo",
]
