"""MBPP+ public-example Tool/ReAct execution.

This module is a thin dataset adapter over SkillFlow's bounded Python
executor and :class:`ToolReactExecutionAdapter`.  It exposes only the public
assertion already present in the task prompt.  EvalPlus Base/Plus inputs,
expected outputs, and statuses remain evaluator-only terminal data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .computation_tools import AIMEComputationToolBackend
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


MBPPPLUS_PYTHON_EXEC_TOOL_ID = "mbpp-plus.python_exec"
MBPPPLUS_PYTHON_EXEC_ACTION = "run_public_test"
MBPPPLUS_PYTHON_EXEC_TOOL_VERSION = "skillflow.training-tools.python-exec.v1"
MBPPPLUS_DATASET_SCOPE = ("mbpp_plus",)


def extract_mbppplus_public_assertions(problem: str) -> tuple[str, ...]:
    """Return the public assertions verbatim from one MBPP+ task prompt."""

    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("MBPP+ problem must be non-empty text")
    assertions = tuple(
        line.strip()
        for line in problem.splitlines()
        if re.match(r"^\s*assert\s+", line)
    )
    if not assertions:
        raise ValueError("MBPP+ problem has no public assertion")
    return assertions


def mbppplus_public_test_program(source: str, assertions: Sequence[str]) -> str:
    """Compose one candidate source with the exact public assertions."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("candidate source must be non-empty text")
    if (
        isinstance(assertions, (str, bytes))
        or not assertions
        or any(not isinstance(item, str) or not item.strip() for item in assertions)
    ):
        raise ValueError("public assertions must contain non-empty strings")
    return source.rstrip() + "\n\n" + "\n".join(item.strip() for item in assertions)


@dataclass(frozen=True, slots=True)
class MBPPPlusPublicTestBackend:
    """Execute candidate source followed by the task's public assertions."""

    public_assertions: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.public_assertions or any(
            not isinstance(item, str) or not item.strip()
            for item in self.public_assertions
        ):
            raise ValueError("public_assertions must contain non-empty text")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != MBPPPLUS_PYTHON_EXEC_ACTION:
            raise ValueError("MBPP+ backend received an incompatible action")
        if set(request.arguments) != {"code"}:
            raise ValueError("run_public_test arguments must contain exactly code")
        source = request.arguments.get("code")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("code must be non-empty text")

        # DIRECT_REUSE: AIMEComputationToolBackend is the existing direct port
        # of SkillFlow training/tools.py::_python_exec.  The MBPP+ adaptation
        # changes only the public program supplied to that executor.
        backend = AIMEComputationToolBackend(
            "python_exec",
            self.timeout_seconds,
        )
        result = backend.invoke(
            ToolRequest(
                "python_exec",
                {
                    "code": mbppplus_public_test_program(
                        source,
                        self.public_assertions,
                    )
                },
            )
        )
        value = result.value
        observation = (
            value.get("observation")
            if isinstance(value, Mapping)
            else None
        )
        ok = value.get("ok") if isinstance(value, Mapping) else False
        return ToolResult(
            {
                "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                "ok": ok is True,
                "observation": (
                    observation
                    if isinstance(observation, str)
                    else "[ERROR] Public test executor returned no observation"
                ),
                "public_assertion_count": len(self.public_assertions),
                "public_assertions": list(self.public_assertions),
                "interface_requirement": (
                    "Preserve the entry point and positional argument order "
                    "shown in the public assertions."
                ),
            }
        )


def create_mbppplus_public_test_registry(
    problem: str,
    *,
    timeout_seconds: float,
) -> ToolRegistry:
    """Register the task-scoped MBPP+ public-test Python Tool."""

    assertions = extract_mbppplus_public_assertions(problem)
    input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["code"],
        "properties": {
            "code": {
                "type": "string",
                "minLength": 1,
                # Complete Python source must preserve at least one physical
                # line boundary through the JSON wire.  Without this schema
                # constraint the local policy flattens ``def`` bodies and the
                # public executor can only report IndentationError.
                "pattern": r"^[\s\S]*\n[\s\S]*$",
            }
        },
    }
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "ok",
            "observation",
            "public_assertion_count",
            "public_assertions",
            "interface_requirement",
        ],
        "properties": {
            "action": {"const": MBPPPLUS_PYTHON_EXEC_ACTION},
            "ok": {"type": "boolean"},
            "observation": {"type": "string"},
            "public_assertion_count": {"type": "integer", "minimum": 1},
            "public_assertions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "interface_requirement": {"type": "string", "minLength": 1},
        },
    }
    capability = ToolCapability(
        tool_id=MBPPPLUS_PYTHON_EXEC_TOOL_ID,
        dataset_scope=MBPPPLUS_DATASET_SCOPE,
        action_schemas={MBPPPLUS_PYTHON_EXEC_ACTION: input_schema},
        input_schema=input_schema,
        output_schema=output_schema,
        side_effect="isolated_child_process",
        timeout_seconds=timeout_seconds + 6.0,
        version=MBPPPLUS_PYTHON_EXEC_TOOL_VERSION,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                MBPPPlusPublicTestBackend(assertions, timeout_seconds),
                capability,
            ),
        )
    )


def validate_mbppplus_tested_completion(execution: object) -> Optional[str]:
    """Require the terminal source to match a successful public Tool receipt."""

    source = getattr(execution, "final_answer", None)
    output_metadata = getattr(execution, "output_metadata", None)
    if not isinstance(source, str) or not source.strip():
        return "MBPP+ terminal artifact is empty"
    if not isinstance(output_metadata, Mapping):
        return "MBPP+ terminal execution has no output metadata"
    for metadata in output_metadata.values():
        if not isinstance(metadata, Mapping):
            continue
        raw_receipts = metadata.get("tool_receipts", ())
        if not isinstance(raw_receipts, (list, tuple)):
            continue
        for receipt in raw_receipts:
            if not isinstance(receipt, Mapping):
                continue
            request = receipt.get("request")
            result = receipt.get("result")
            arguments = (
                request.get("arguments") if isinstance(request, Mapping) else None
            )
            value = result.get("value") if isinstance(result, Mapping) else None
            if (
                receipt.get("tool_id") == MBPPPLUS_PYTHON_EXEC_TOOL_ID
                and isinstance(request, Mapping)
                and request.get("action") == MBPPPLUS_PYTHON_EXEC_ACTION
                and isinstance(arguments, Mapping)
                and arguments.get("code") == source
                and isinstance(value, Mapping)
                and value.get("ok") is True
            ):
                return None
    return (
        "MBPP+ terminal source must exactly match a successful "
        "run_public_test Tool receipt from the current graph revision"
    )
class MBPPPlusReactExecutionAdapter(ToolReactExecutionAdapter):
    """Require completion to match a successful public-test Tool receipt."""

    # DIRECT_REUSE: the Canvas ``continue`` boundary follows the already
    # deployed WebShop one-action/one-observation execution contract.  The
    # generic bounded adapter still owns parsing, Tool dispatch, receipts and
    # continuation; this dataset adapter only declares that each invocation is
    # intentionally one ReAct turn.
    stepwise_director = True
    execution_semantics = "one_action_one_observation"
    # MBPP+ ReAct exists to execute the task-scoped public-test Tool.  A
    # completion-only Agent remains available through the ordinary reasoning
    # adapter; publishing react+no-tools as a correlated Canvas profile makes
    # the model generate Tool actions which that Agent is forbidden to run.
    requires_allowed_tools = True

    def _contract(
        self,
        request,
        observations: list[Mapping[str, object]],
    ) -> str:
        # NECESSARY_ADAPTATION: StructuredAction carries source inside a JSON
        # string.  Preserve Python line structure at that wire boundary; the
        # public executor must receive complete source, not whitespace-flattened
        # pseudocode.  This constrains Tool syntax only and does not prescribe
        # an algorithm, Agent role, relation, or topology.
        return (
            super()._contract(request, observations)
            + "\nFor run_public_test, arguments.code must be syntactically valid "
            "complete Python source. Encode Python line breaks as JSON \\n "
            "escapes and preserve indentation after decoding. Do not flatten "
            "multiple Python statements into one line."
            " Preserve the exact positional argument order shown by the "
            "task's public assertions."
        )

    def _state_conditioned_action_domain(
        self,
        request,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        del request
        tested_source = self._latest_tested_source(observations)
        return (
            frozenset()
            if tested_source is not None
            else frozenset(
                {
                    (
                        MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                        MBPPPLUS_PYTHON_EXEC_ACTION,
                    )
                }
            ),
            tested_source is not None,
        )

    @staticmethod
    def _latest_tested_source(
        observations: list[Mapping[str, object]],
    ) -> Optional[str]:
        for observation in reversed(observations):
            action = observation.get("executed_action")
            result = observation.get("result")
            if (
                observation.get("observation_status") == "success"
                and isinstance(action, Mapping)
                and action.get("resource_id") == MBPPPLUS_PYTHON_EXEC_TOOL_ID
                and action.get("name") == MBPPPLUS_PYTHON_EXEC_ACTION
                and isinstance(action.get("arguments"), Mapping)
                and isinstance(action["arguments"].get("code"), str)
                and isinstance(result, Mapping)
                and result.get("ok") is True
            ):
                return str(action["arguments"]["code"])
        return None

    def _state_conditioned_response_schema(
        self,
        request,
        observations: list[Mapping[str, object]],
    ) -> Optional[dict[str, object]]:
        tested_source = self._latest_tested_source(observations)
        if tested_source is None:
            return super()._state_conditioned_response_schema(
                request,
                observations,
            )
        # The successful public Tool receipt already fixes the semantic
        # artifact.  Constrain the following completion to copy that exact
        # source so a later formatting/generation step cannot silently alter a
        # tested program.  This exposes no hidden evaluator data.
        return {
            "type": "object",
            "required": [
                "arguments",
                "kind",
                "name",
                "resource_id",
                "skill_id",
            ],
            "properties": {
                "arguments": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"const": tested_source}},
                    "additionalProperties": False,
                },
                "kind": {"const": "complete"},
                "name": {"const": "complete"},
                "resource_id": {"const": None},
                "skill_id": {"const": None},
            },
            "additionalProperties": False,
        }

    def _completion_error(
        self,
        *,
        action,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> Optional[str]:
        del action
        for receipt in reversed(tool_receipts):
            request = receipt.get("request")
            result = receipt.get("result")
            value = result.get("value") if isinstance(result, Mapping) else None
            arguments = (
                request.get("arguments") if isinstance(request, Mapping) else None
            )
            if (
                isinstance(request, Mapping)
                and request.get("action") == MBPPPLUS_PYTHON_EXEC_ACTION
                and isinstance(arguments, Mapping)
                and arguments.get("code") == artifact
                and isinstance(value, Mapping)
                and value.get("ok") is True
            ):
                return None
        return "mbppplus_completion_requires_matching_public_test"

    async def execute(self, request):
        try:
            return await super().execute(request)
        except ReactExecutionError as exc:
            trace = tuple(exc.react_trace)
            last = trace[-1] if trace else None
            if (
                isinstance(last, Mapping)
                and last.get("observation_status") == "budget_exhausted"
                and last.get("public_error_code")
                == "tool_call_budget_exhausted"
            ):
                exc.tool_plan_exhausted = True
            raise


__all__ = [
    "MBPPPLUS_DATASET_SCOPE",
    "MBPPPLUS_PYTHON_EXEC_ACTION",
    "MBPPPLUS_PYTHON_EXEC_TOOL_ID",
    "MBPPPLUS_PYTHON_EXEC_TOOL_VERSION",
    "MBPPPlusPublicTestBackend",
    "MBPPPlusReactExecutionAdapter",
    "create_mbppplus_public_test_registry",
    "extract_mbppplus_public_assertions",
    "mbppplus_public_test_program",
    "validate_mbppplus_tested_completion",
]
