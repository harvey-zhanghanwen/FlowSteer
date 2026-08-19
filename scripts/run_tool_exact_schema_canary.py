#!/usr/bin/env python3
"""Run one diagnostic-only exact-schema Tool/ReAct canary.

This is a deliberately separate diagnostic entry point.  It reuses
``LiveSmokeBackend._runtime_for_task`` to obtain the task-scoped runtime and
then executes one ordinary ``AgentRuntime`` node through the existing
``ToolReactExecutionAdapter``.  The model, rather than this script, produces
every ``StructuredAction``.  The script only verifies the resulting public
trace and therefore fails closed when the requested sequence is not selected.

No evaluator, benchmark metric, Director rollout, Skill evidence, optimizer,
or training path is invoked here.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_completion_benchmark_round import (  # noqa: E402
    _benchmark_slice,
    _evaluation_section,
    validate_completion_benchmark_config,
)
from train_agentgraph_smoke import (  # noqa: E402
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
)
from src.interactive.agent_graph import AgentGraph, AgentNode  # noqa: E402
from src.interactive.computation_tools import (  # noqa: E402
    AIME_CALCULATOR_TOOL_ID,
    AIME_PYTHON_EXEC_TOOL_ID,
)
from src.interactive.healthbench_tool_adapter import (  # noqa: E402
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
)
from src.interactive.qa_tool_adapter import (  # noqa: E402
    QA_RETRIEVAL_READ_TOOL_ID,
    QA_RETRIEVAL_SEARCH_TOOL_ID,
)
from src.interactive.react_execution import ReactExecutionError  # noqa: E402
from src.interactive.records import TaskRecord  # noqa: E402
from src.interactive.task_dataset import iter_task_records  # noqa: E402
from src.interactive.tool_runtime import ToolRegistry  # noqa: E402


SCHEMA_VERSION = "flowsteer.tool_exact_schema_canary.v1"
SUPPORTED_DATASETS = (
    "hotpotqa",
    "triviaqa",
    "aime_2026",
    "healthbench_professional",
)
DEFAULT_CONFIGS: Mapping[str, str] = {
    "hotpotqa": "config/evaluation_hotpotqa_tool_react_stable_zero.yaml",
    "triviaqa": "config/evaluation_triviaqa_tool_react_stable_zero.yaml",
    "aime_2026": "config/development_aime2026_computation_tool_stable_zero.yaml",
    "healthbench_professional": (
        "config/evaluation_healthbench_professional_medrag_tool_stable_zero.yaml"
    ),
}


class ToolCanaryError(RuntimeError):
    """The diagnostic could not establish exact Tool/ReAct compliance."""


@dataclass(frozen=True, slots=True)
class ActionStep:
    kind: str
    name: str
    resource_id: Optional[str]

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "kind": self.kind,
            "name": self.name,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True, slots=True)
class CanarySpecification:
    dataset_key: str
    required_resource_ids: tuple[str, ...]
    admitted_sequences: tuple[tuple[ActionStep, ...], ...]
    contract: str
    completion_condition: str


def _specification(dataset_key: str) -> CanarySpecification:
    complete = ActionStep("complete", "complete", None)
    if dataset_key in {"hotpotqa", "triviaqa"}:
        return CanarySpecification(
            dataset_key=dataset_key,
            required_resource_ids=(
                QA_RETRIEVAL_READ_TOOL_ID,
                QA_RETRIEVAL_SEARCH_TOOL_ID,
            ),
            admitted_sequences=(
                (
                    ActionStep(
                        "tool", "search", QA_RETRIEVAL_SEARCH_TOOL_ID
                    ),
                    ActionStep("tool", "read", QA_RETRIEVAL_READ_TOOL_ID),
                    complete,
                ),
            ),
            contract=(
                "Use the public retrieval Tools for this forced diagnostic. "
                "First select search with a query derived only from the task. "
                "After receiving that public observation, select read with one "
                "returned passage_id. Then complete with a short artifact based "
                "only on the task and public Tool observations."
            ),
            completion_condition=(
                "Exactly one successful search followed by exactly one successful "
                "read, then one complete action; do not skip, repeat, or reorder "
                "these actions."
            ),
        )
    if dataset_key == "aime_2026":
        return CanarySpecification(
            dataset_key=dataset_key,
            required_resource_ids=(
                AIME_CALCULATOR_TOOL_ID,
                AIME_PYTHON_EXEC_TOOL_ID,
            ),
            admitted_sequences=(
                (
                    ActionStep(
                        "tool", "calculator", AIME_CALCULATOR_TOOL_ID
                    ),
                    complete,
                ),
                (
                    ActionStep(
                        "tool", "python_exec", AIME_PYTHON_EXEC_TOOL_ID
                    ),
                    complete,
                ),
            ),
            contract=(
                "Use one public computation Tool for this forced diagnostic. "
                "Select either calculator or python_exec with an expression or "
                "program derived only from the task. After receiving its public "
                "observation, complete with a short diagnostic artifact."
            ),
            completion_condition=(
                "Exactly one successful calculator or python_exec action followed "
                "by exactly one complete action; do not skip or repeat the Tool."
            ),
        )
    if dataset_key == "healthbench_professional":
        return CanarySpecification(
            dataset_key=dataset_key,
            required_resource_ids=(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,),
            admitted_sequences=(
                (
                    ActionStep(
                        "tool", "search", HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID
                    ),
                    complete,
                ),
            ),
            contract=(
                "Use the public MedRAG textbook search Tool for this forced "
                "diagnostic. Select search with a query derived only from the "
                "healthcare conversation. After receiving its public observation, "
                "complete with a short artifact based only on public information."
            ),
            completion_condition=(
                "Exactly one successful search followed by exactly one complete "
                "action; do not skip or repeat the Tool."
            ),
        )
    raise ToolCanaryError(f"unsupported Tool canary dataset: {dataset_key}")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _public_runtime_task(task: TaskRecord) -> TaskRecord:
    """Remove evaluator and upstream compatibility payloads before execution."""

    dataset_key = _dataset_key(task)
    public_metadata = {
        "dataset_key": dataset_key,
        "task_family": str(task.metadata.get("task_family", dataset_key)),
    }
    benchmark_slice = _benchmark_slice(task)
    if benchmark_slice:
        public_metadata["benchmark_slice"] = benchmark_slice
    return TaskRecord(
        task_id=task.task_id,
        question=task.question,
        ground_truth=None,
        split=task.split,
        metadata=public_metadata,
    )


def load_first_development_task(
    config: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
) -> TaskRecord:
    """Return the first aligned validation task as a gold-free runtime record."""

    _, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    if dataset_key not in SUPPORTED_DATASETS:
        raise ToolCanaryError(f"dataset {dataset_key!r} has no Tool canary")
    if bounded.get("split") != "validation":
        raise ToolCanaryError(
            "diagnostic Tool canary requires the development validation split"
        )
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ToolCanaryError("config.data must be a mapping")
    raw_path = data.get("validation_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ToolCanaryError("config.data.validation_path must be non-empty")
    requested_slice = bounded.get("benchmark_slice")
    for task in iter_task_records(
        _resolve_from_root(root, raw_path), expected_split="validation"
    ):
        if _dataset_key(task) != dataset_key:
            continue
        if requested_slice and _benchmark_slice(task) != str(requested_slice):
            continue
        return _public_runtime_task(task)
    raise ToolCanaryError(
        f"no aligned development task is available for {dataset_key}"
    )


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _json_type_matches(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if expected == "boolean":
        return type(value) is bool
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return False


def _argument_schema_issues(
    arguments: object,
    schema: Mapping[str, object],
    *,
    prefix: str,
) -> list[str]:
    """Check the object-schema subset published by the existing Tool catalog."""

    issues: list[str] = []
    if schema.get("type") != "object" or not isinstance(arguments, Mapping):
        return [f"{prefix}: arguments are not an object allowed by the schema"]
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    if not isinstance(properties, Mapping) or not isinstance(
        required, (list, tuple)
    ):
        return [f"{prefix}: published argument schema is incompatible"]
    required_names = {
        str(name) for name in required if isinstance(name, str) and name
    }
    missing = sorted(required_names.difference(arguments))
    if missing:
        issues.append(f"{prefix}: missing required arguments {missing}")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(arguments).difference(properties))
        if extra:
            issues.append(f"{prefix}: unpublished arguments {extra}")
    for name, value in arguments.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, Mapping):
            continue
        expected_type = field_schema.get("type")
        if expected_type is not None and not _json_type_matches(
            value, expected_type
        ):
            issues.append(f"{prefix}: argument {name!r} has the wrong type")
            continue
        if isinstance(value, str) and isinstance(
            field_schema.get("minLength"), int
        ) and len(value) < int(field_schema["minLength"]):
            issues.append(f"{prefix}: argument {name!r} is too short")
        if (
            type(value) is int
            and isinstance(field_schema.get("minimum"), (int, float))
            and value < field_schema["minimum"]
        ):
            issues.append(f"{prefix}: argument {name!r} is below minimum")
    return issues


def _observed_actions(
    trace: Sequence[Mapping[str, object]],
) -> tuple[ActionStep, ...]:
    observed: list[ActionStep] = []
    for entry in trace:
        action = entry.get("structured_action")
        if not isinstance(action, Mapping):
            continue
        kind = action.get("kind")
        name = action.get("name")
        resource_id = action.get("resource_id")
        if not isinstance(kind, str) or not isinstance(name, str):
            continue
        if resource_id is not None and not isinstance(resource_id, str):
            continue
        observed.append(ActionStep(kind, name, resource_id))
    return tuple(observed)


def _compliance(
    *,
    specification: CanarySpecification,
    tool_registry: ToolRegistry,
    trace: Sequence[Mapping[str, object]],
    tool_receipts: Sequence[Mapping[str, object]],
    completed_artifact: Optional[str],
) -> dict[str, object]:
    schema_issues: list[str] = []
    dispatched_tool_actions: list[Mapping[str, object]] = []
    for index, entry in enumerate(trace, start=1):
        action = entry.get("structured_action")
        if not isinstance(action, Mapping):
            schema_issues.append(
                f"turn {index}: model output was not an admitted StructuredAction"
            )
            continue
        if set(action) != {
            "arguments",
            "kind",
            "name",
            "resource_id",
            "skill_id",
        }:
            schema_issues.append(f"turn {index}: StructuredAction field set differs")
            continue
        kind = action.get("kind")
        name = action.get("name")
        arguments = action.get("arguments")
        resource_id = action.get("resource_id")
        if action.get("skill_id") is not None:
            schema_issues.append(f"turn {index}: skill_id is not allowed")
        if kind == "complete":
            if name != "complete" or resource_id is not None:
                schema_issues.append(
                    f"turn {index}: completion name/resource_id differs"
                )
            if not isinstance(arguments, Mapping) or set(arguments) != {"value"}:
                schema_issues.append(
                    f"turn {index}: completion arguments differ from schema"
                )
            continue
        if kind != "tool" or not isinstance(resource_id, str):
            schema_issues.append(f"turn {index}: action kind/resource_id differs")
            continue
        try:
            capability = tool_registry.require_capability(resource_id)
        except KeyError:
            schema_issues.append(f"turn {index}: resource_id is not registered")
            continue
        if not isinstance(name, str) or name not in capability.action_schemas:
            schema_issues.append(f"turn {index}: action name is not published")
            continue
        schema_issues.extend(
            _argument_schema_issues(
                arguments,
                capability.action_schemas[name],
                prefix=f"turn {index}",
            )
        )
        # A ToolReceipt exists only after the bounded runtime admits and
        # dispatches an action.  Budget-exhausted/schema-invalid model turns
        # remain model-compliance evidence, not missing backend receipts.
        if isinstance(entry.get("observation"), Mapping):
            dispatched_tool_actions.append(action)

    receipt_issues: list[str] = []
    if len(tool_receipts) != len(dispatched_tool_actions):
        receipt_issues.append("Tool action and backend receipt counts differ")
    for index, (action, receipt) in enumerate(
        zip(dispatched_tool_actions, tool_receipts), start=1
    ):
        request = receipt.get("request")
        result = receipt.get("result")
        if (
            receipt.get("tool_id") != action.get("resource_id")
            or not isinstance(request, Mapping)
            or request.get("action") != action.get("name")
            or request.get("arguments") != action.get("arguments")
        ):
            receipt_issues.append(
                f"Tool receipt {index} does not match the model action"
            )
        if receipt.get("error_type") is not None or not isinstance(
            result, Mapping
        ):
            receipt_issues.append(f"Tool receipt {index} reports backend failure")
        elif result.get("completed") is not True:
            receipt_issues.append(
                f"Tool receipt {index} is not a completed public observation"
            )

    observed = _observed_actions(trace)
    sequence_matches = observed in specification.admitted_sequences
    model_issues: list[str] = []
    if not sequence_matches:
        model_issues.append("model-selected action sequence is not admitted")
    if len(observed) != len(trace):
        model_issues.append("one or more model turns was not a StructuredAction")
    if not isinstance(completed_artifact, str) or not completed_artifact.strip():
        model_issues.append("model did not produce a non-empty completion artifact")
    if not trace or trace[-1].get("observation_status") != "completed":
        model_issues.append("ReAct trace did not terminate with completion")

    schema_passed = not schema_issues
    backend_passed = not receipt_issues and bool(tool_receipts)
    model_passed = not model_issues
    return {
        "passed": schema_passed and backend_passed and model_passed,
        "schema_compliance": {
            "passed": schema_passed,
            "issues": schema_issues,
            "authority": "ToolCapability.action_schemas",
        },
        "backend_compliance": {
            "passed": backend_passed,
            "issues": receipt_issues,
            "successful_receipts": sum(
                1
                for receipt in tool_receipts
                if receipt.get("error_type") is None
                and isinstance(receipt.get("result"), Mapping)
                and receipt["result"].get("completed") is True
            ),
        },
        "model_compliance": {
            "passed": model_passed,
            "issues": model_issues,
            "action_selection": "model_generated_not_injected",
        },
        "observed_sequence": [step.to_dict() for step in observed],
    }


def _base_receipt(
    *,
    task: TaskRecord,
    specification: CanarySpecification,
    condition_id: str,
    model_id: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "diagnostic_only": True,
        "forced_probe": True,
        "grpo_eligible": False,
        "skill_evidence_eligible": False,
        "training_enabled": False,
        "controls": {
            "diagnostic_only": True,
            "forced_probe": True,
            "grpo_eligible": False,
            "skill_evidence_eligible": False,
            "training_enabled": False,
            "director_invoked": False,
            "evaluator_invoked": False,
            "benchmark_metrics_included": False,
            "structured_actions_injected": False,
        },
        "condition_id": condition_id,
        "dataset_key": specification.dataset_key,
        "task": {
            "task_id": task.task_id,
            "split": task.split,
            "question": task.question,
            "public_metadata": dict(task.metadata),
            "ground_truth_supplied_to_runtime": False,
        },
        "model": {
            "model_id": model_id,
            "selection_source": "configured_development_direct_model_id",
        },
        "required_resources": list(specification.required_resource_ids),
        "admitted_sequences": [
            [step.to_dict() for step in sequence]
            for sequence in specification.admitted_sequences
        ],
        "benchmark_metrics": None,
    }


async def execute_tool_canary(
    *,
    backend: LiveSmokeBackend,
    task: TaskRecord,
    model_id: str,
    condition_id: str,
    run_id: str,
) -> dict[str, object]:
    """Execute one model-generated Tool sequence and return its receipt."""

    if task.ground_truth is not None:
        raise ToolCanaryError("runtime task must not carry ground truth")
    specification = _specification(_dataset_key(task))
    receipt = _base_receipt(
        task=task,
        specification=specification,
        condition_id=condition_id,
        model_id=model_id,
    )
    close_runtime = lambda: None
    tool_registry: Optional[ToolRegistry] = None
    try:
        task_runtime, tool_registry, close_runtime = backend._runtime_for_task(
            task,
            condition_id=condition_id,
        )
        if not isinstance(tool_registry, ToolRegistry):
            raise ToolCanaryError("task-scoped runtime returned no ToolRegistry")
        if tuple(tool_registry.resource_ids) != tuple(
            sorted(specification.required_resource_ids)
        ):
            raise ToolCanaryError(
                "task-scoped ToolRegistry differs from the canary resource set"
            )
        backend.registry.require_model(model_id)
        node = AgentNode(
            "tool_contract_canary",
            model_id,
            specification.contract,
            role_family="tool_use",
            allowed_tools=tool_registry.resource_ids,
            execution_mode="react",
            artifact_type="diagnostic_artifact",
            completion_condition=specification.completion_condition,
        )
        graph = AgentGraph(nodes=(node,), output_agent_id=node.id)
        result = await task_runtime.execute(graph, task.question, run_id=run_id)
        if len(result.calls) != 1:
            raise ToolCanaryError(
                "single ReAct Agent produced an incompatible outer call count"
            )
        metadata = dict(result.calls[0].response.metadata)
        raw_trace = metadata.get("react_trace", ())
        raw_tool_receipts = metadata.get("tool_receipts", ())
        if not isinstance(raw_trace, (list, tuple)) or not all(
            isinstance(item, Mapping) for item in raw_trace
        ):
            raise ToolCanaryError("runtime emitted no compatible ReAct trace")
        if not isinstance(raw_tool_receipts, (list, tuple)) or not all(
            isinstance(item, Mapping) for item in raw_tool_receipts
        ):
            raise ToolCanaryError("runtime emitted no compatible Tool receipts")
        trace = tuple(dict(item) for item in raw_trace)
        tool_receipts = tuple(dict(item) for item in raw_tool_receipts)
        compliance = _compliance(
            specification=specification,
            tool_registry=tool_registry,
            trace=trace,
            tool_receipts=tool_receipts,
            completed_artifact=result.final_answer,
        )
        model = backend.registry.require_model(model_id)
        provider = backend.registry.provider_for(model_id)
        receipt["model"] = {
            **dict(receipt["model"]),
            "provider_id": provider.provider_id,
            "provider_model": model.model_name,
        }
        receipt.update(
            {
                "status": "passed" if compliance["passed"] else "failed",
                "runtime": {
                    "run_id": result.run_id,
                    "output_agent_id": result.output_agent_id,
                    "executed_agent_ids": list(result.executed_agent_ids),
                    "react_turns_used": metadata.get("react_turns_used"),
                    "tool_calls": metadata.get("tool_calls"),
                    "model_calls": metadata.get("model_calls", []),
                    "completed_artifact": result.final_answer,
                },
                "react_trace": list(trace),
                "tool_receipts": list(tool_receipts),
                "compliance": compliance,
            }
        )
        if not compliance["passed"]:
            receipt["failure"] = (
                "Model-selected actions did not establish exact-schema, backend, "
                "and model compliance; no action was injected to repair the trace."
            )
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        raise
    except BaseException as exc:
        current: Optional[BaseException] = exc
        react_failure: Optional[ReactExecutionError] = None
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ReactExecutionError):
                react_failure = current
                break
            current = current.__cause__ or current.__context__
        if react_failure is not None and tool_registry is not None:
            trace = tuple(dict(item) for item in react_failure.react_trace)
            tool_receipts = tuple(
                dict(item) for item in react_failure.tool_receipts
            )
            compliance = _compliance(
                specification=specification,
                tool_registry=tool_registry,
                trace=trace,
                tool_receipts=tool_receipts,
                completed_artifact=None,
            )
            receipt.update(
                {
                    "status": "failed",
                    "failure": _safe_error(exc),
                    "runtime": {
                        "model_calls": list(react_failure.model_calls),
                        "completed_artifact": None,
                    },
                    "react_trace": list(trace),
                    "tool_receipts": list(tool_receipts),
                    "compliance": compliance,
                }
            )
        else:
            receipt.update(
                {
                    "status": "failed",
                    "failure": _safe_error(exc),
                    "compliance": {
                        "passed": False,
                        "schema_compliance": {
                            "passed": False,
                            "issues": ["runtime did not return a complete trace"],
                        },
                        "backend_compliance": {
                            "passed": False,
                            "issues": [
                                "runtime did not return complete Tool receipts"
                            ],
                        },
                        "model_compliance": {
                            "passed": False,
                            "issues": [
                                "required action sequence was not established "
                                "without injecting model actions"
                            ],
                            "action_selection": "model_generated_not_injected",
                        },
                        "observed_sequence": [],
                    },
                }
            )
    finally:
        try:
            close_runtime()
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["runtime_close_failure"] = _safe_error(exc)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument(
        "--config",
        help="Evaluation-only Tool condition; defaults to the dataset condition.",
    )
    parser.add_argument(
        "--output",
        help=(
            "New JSON receipt path; defaults to "
            "artifacts/tool_exact_schema_canary/<dataset>.json"
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    config_path = _resolve(args.config or DEFAULT_CONFIGS[args.dataset])
    from src.interactive.config_loader import load_yaml

    config = load_yaml(config_path)
    validate_completion_benchmark_config(config)
    _, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    if dataset_key != args.dataset:
        raise ToolCanaryError(
            f"config dataset {dataset_key!r} does not match --dataset {args.dataset!r}"
        )
    output = _resolve(
        args.output
        or f"artifacts/tool_exact_schema_canary/{dataset_key}.json"
    )
    if output.exists():
        raise ToolCanaryError(
            f"receipt already exists; refusing a duplicate model call: {output}"
        )
    task = load_first_development_task(config)
    model_id = str(bounded["direct_model_id"])
    condition_id = str(config["experiment"]["condition_id"])
    backend = LiveSmokeBackend.from_config(
        config,
        PROJECT_ROOT,
        evaluation_only=True,
    )
    receipt = await execute_tool_canary(
        backend=backend,
        task=task,
        model_id=model_id,
        condition_id=condition_id,
        run_id=f"tool-exact-schema-canary:{dataset_key}:{task.task_id}",
    )
    _write_json(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "dataset_key": dataset_key,
                "task_id": task.task_id,
                "output": str(output),
                "diagnostic_only": True,
                "benchmark_metrics": None,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if receipt["status"] == "passed" else 2


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
