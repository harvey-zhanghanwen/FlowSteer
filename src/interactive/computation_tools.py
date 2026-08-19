"""AIME computation Tools backed by SkillFlow's bounded local executors.

The execution path in this module is a dependency-light port of SkillFlow
``training/tools.py``::``execute_tool`` -> ``_calculator`` / ``_python_exec``
-> ``_exec_code_in_process``.  In particular, Python code runs in the same
separate ``multiprocessing.Process`` with a hard join/terminate/kill timeout;
it is never dispatched through a shell.

Only the observation-producing execution path is retained.  SkillFlow's tool
success rewards and failure penalties belong to its training loop and are
deliberately not part of these Tool results or capabilities.
"""

from __future__ import annotations

import ast
import io
import math
import multiprocessing
import sys
import traceback
from dataclasses import dataclass
from typing import Mapping

from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


AIME_CALCULATOR_TOOL_ID = "aime-computation.calculator"
AIME_PYTHON_EXEC_TOOL_ID = "aime-computation.python_exec"
AIME_COMPUTATION_TOOL_VERSION = "skillflow.training-tools.v1"
AIME_DATASET_SCOPE = ("aime_2026",)
DEFAULT_PYTHON_EXEC_TIMEOUT_SECONDS = 30.0


# Direct port of SkillFlow training/tools.py::_CALC_NAMESPACE.
_CALC_NAMESPACE = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_CALC_NAMESPACE.update(
    {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "int": int,
        "float": float,
        "sum": sum,
        "pow": pow,
        "True": True,
        "False": False,
        "len": len,
    }
)


def _exec_code_in_process(
    code: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Direct port of SkillFlow's isolated child-process entry point."""

    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        sys.stdout = captured_out
        sys.stderr = captured_err
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(code, namespace)
        stdout = captured_out.getvalue()
        stderr = captured_err.getvalue()
        result_parts = []
        if stdout:
            result_parts.append(f"[STDOUT]\n{stdout}")
        if stderr:
            result_parts.append(f"[STDERR]\n{stderr}")
        if not result_parts:
            result_parts.append("[OK] Code executed successfully (no output)")
        result_queue.put("\n".join(result_parts))
    except Exception:
        stderr = captured_err.getvalue()
        trace = traceback.format_exc()
        parts = []
        if stderr:
            parts.append(f"[STDERR]\n{stderr}")
        parts.append(f"[EXCEPTION]\n{trace}")
        result_queue.put("\n".join(parts))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _calculator(arguments: Mapping[str, object]) -> str:
    """Port SkillFlow's calculator observation without training reward logic."""

    expression = arguments.get("expression", "")
    if not isinstance(expression, str) or not expression.strip():
        return "[ERROR] No expression provided"
    try:
        tree = ast.parse(expression, mode="eval")
        result = eval(
            compile(tree, "<calc>", "eval"),
            {"__builtins__": {}},
            _CALC_NAMESPACE,
        )
        return f"[RESULT] {expression} = {result}"
    except Exception as exc:
        return (
            f"[ERROR] Cannot evaluate '{expression}': "
            f"{type(exc).__name__}: {exc}"
        )


def _python_exec(
    arguments: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> str:
    """Port SkillFlow's process-isolated Python executor and hard timeout."""

    code = arguments.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "[ERROR] No code provided"

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_exec_code_in_process,
        args=(code, result_queue),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        return (
            f"[ERROR] Code execution timed out after {timeout_seconds:g}s. "
            "Simplify your code or use a different approach."
        )

    if not result_queue.empty():
        return result_queue.get()
    return "[ERROR] Code execution produced no result"


def _is_success(observation: str) -> bool:
    """Direct port of SkillFlow ``is_tool_success`` classification."""

    return not any(
        tag in observation
        for tag in ("[ERROR]", "[TOOL_ERROR]", "[FAIL]", "[EXCEPTION]")
    )


@dataclass(frozen=True, slots=True)
class AIMEComputationToolBackend:
    """One action-bound backend for an AIME calculator or Python Tool."""

    action: str
    python_timeout_seconds: float = DEFAULT_PYTHON_EXEC_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.action not in {"calculator", "python_exec"}:
            raise ValueError("unsupported AIME computation action")
        if self.python_timeout_seconds <= 0:
            raise ValueError("python_timeout_seconds must be positive")

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != self.action:
            raise ValueError(
                f"{self.action} backend received incompatible action "
                f"{request.action!r}"
            )
        expected_fields = {"expression"} if self.action == "calculator" else {"code"}
        if set(request.arguments) != expected_fields:
            field = next(iter(expected_fields))
            raise ValueError(
                f"{self.action} arguments must contain exactly {field}"
            )
        value = request.arguments[next(iter(expected_fields))]
        if not isinstance(value, str) or not value.strip():
            field = next(iter(expected_fields))
            raise ValueError(f"{field} must be non-empty text")

        observation = (
            _calculator(request.arguments)
            if self.action == "calculator"
            else _python_exec(
                request.arguments,
                timeout_seconds=self.python_timeout_seconds,
            )
        )
        return ToolResult(
            {
                "action": self.action,
                "ok": _is_success(observation),
                "observation": observation,
            }
        )


def create_aime_computation_registry(
    *,
    python_timeout_seconds: float = DEFAULT_PYTHON_EXEC_TIMEOUT_SECONDS,
    calculator_timeout_seconds: float = 2.0,
) -> ToolRegistry:
    """Register AIME-only calculator and process-isolated Python Tools."""

    if python_timeout_seconds <= 0:
        raise ValueError("python_timeout_seconds must be positive")
    if calculator_timeout_seconds <= 0:
        raise ValueError("calculator_timeout_seconds must be positive")

    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "ok", "observation"],
        "properties": {
            "action": {"type": "string"},
            "ok": {"type": "boolean"},
            "observation": {"type": "string"},
        },
    }
    calculator_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["expression"],
        "properties": {
            "expression": {"type": "string", "minLength": 1},
        },
    }
    calculator_capability = ToolCapability(
        tool_id=AIME_CALCULATOR_TOOL_ID,
        dataset_scope=AIME_DATASET_SCOPE,
        action_schemas={"calculator": calculator_input_schema},
        input_schema=calculator_input_schema,
        output_schema=output_schema,
        side_effect="none",
        timeout_seconds=calculator_timeout_seconds,
        version=AIME_COMPUTATION_TOOL_VERSION,
    )
    python_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "minLength": 1},
        },
    }
    python_capability = ToolCapability(
        tool_id=AIME_PYTHON_EXEC_TOOL_ID,
        dataset_scope=AIME_DATASET_SCOPE,
        action_schemas={"python_exec": python_input_schema},
        input_schema=python_input_schema,
        output_schema=output_schema,
        side_effect="isolated_child_process",
        # Keep the registry timeout outside SkillFlow's internal process
        # timeout/termination boundary so the child is reaped first.
        timeout_seconds=python_timeout_seconds + 6.0,
        version=AIME_COMPUTATION_TOOL_VERSION,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                AIME_CALCULATOR_TOOL_ID,
                AIMEComputationToolBackend("calculator", python_timeout_seconds),
                calculator_capability,
            ),
            ToolRegistration(
                AIME_PYTHON_EXEC_TOOL_ID,
                AIMEComputationToolBackend("python_exec", python_timeout_seconds),
                python_capability,
            ),
        )
    )


__all__ = [
    "AIME_CALCULATOR_TOOL_ID",
    "AIME_COMPUTATION_TOOL_VERSION",
    "AIME_DATASET_SCOPE",
    "AIME_PYTHON_EXEC_TOOL_ID",
    "AIMEComputationToolBackend",
    "DEFAULT_PYTHON_EXEC_TIMEOUT_SECONDS",
    "create_aime_computation_registry",
]
