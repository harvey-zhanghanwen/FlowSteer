"""Bounded iterative Coding Agent execution inside AgentGraph.

The model/tool/observation loop reuses :class:`ToolReactExecutionAdapter` and
the SkillFlow-derived repository tools.  The additional completion admission
requires an actual repository edit, an inspected workspace diff, and a test
execution before a patch artifact may leave the coding node.  Official
SWE-bench resolution remains exclusively in ``swebench_adapter.py``.
"""

from __future__ import annotations

from typing import Callable, Optional

from .agent_runtime import AgentGateway
from .react_execution import ToolReactExecutionAdapter
from .tool_runtime import StructuredAction, ToolRegistry


def _receipt_action(receipt: dict[str, object]) -> Optional[str]:
    request = receipt.get("request")
    if not isinstance(request, dict):
        return None
    action = request.get("action")
    return action if isinstance(action, str) else None


def _receipt_value(receipt: dict[str, object]) -> Optional[dict[str, object]]:
    result = receipt.get("result")
    if not isinstance(result, dict):
        return None
    value = result.get("value")
    return value if isinstance(value, dict) else None


def _changed_edit(receipt: dict[str, object]) -> bool:
    action = _receipt_action(receipt)
    if action not in {"apply_patch", "exact_edit", "str_replace_editor"}:
        return False
    value = _receipt_value(receipt)
    if action == "str_replace_editor" and value is not None:
        if value.get("command") not in {
            "create",
            "insert",
            "str_replace",
            "undo_edit",
        }:
            return False
    return (
        value is not None
        and value.get("ok") is True
        and value.get("changed") is True
    )


def _observed_test(receipt: dict[str, object]) -> bool:
    """Return whether run_tests produced a public pass/fail observation.

    SkillFlow admits a code-generation terminal diff without requiring a
    passing test.  Preserve that upstream semantic here: both ``passed=True``
    and ``passed=False`` are observations, while an invalid request or Tool
    dispatch error is not evidence that a test ran.
    """

    if _receipt_action(receipt) != "run_tests":
        return False
    value = _receipt_value(receipt)
    return (
        value is not None
        and value.get("ok") is True
        and type(value.get("passed")) is bool
    )


def _changed_diff(receipt: dict[str, object]) -> bool:
    if _receipt_action(receipt) != "diff":
        return False
    value = _receipt_value(receipt)
    return value is not None and value.get("changed") is True


def _last_receipt_index(
    tool_receipts: list[dict[str, object]],
    predicate: Callable[[dict[str, object]], bool],
    *,
    after_index: int = -1,
) -> Optional[int]:
    for index in range(len(tool_receipts) - 1, after_index, -1):
        if predicate(tool_receipts[index]):
            return index
    return None


def _submitted_workspace_diff(
    tool_receipts: list[dict[str, object]],
    *,
    after_index: int = -1,
) -> Optional[str]:
    """Return the newest non-empty diff observed after ``after_index``."""

    for receipt in reversed(tool_receipts[after_index + 1 :]):
        if not _changed_diff(receipt):
            continue
        value = _receipt_value(receipt)
        if value is None:
            continue
        diff = value.get("diff")
        if isinstance(diff, str) and diff.strip():
            return diff
    return None


class CodingExecutionAdapter(ToolReactExecutionAdapter):
    """Coding-mode adapter for inspect/edit/test/revision/finalize loops."""

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        max_turns: int,
        max_tool_calls: int,
        max_action_tokens: int = 512,
    ) -> None:
        super().__init__(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_action_tokens=max_action_tokens,
            execution_mode="coding",
        )

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> Optional[str]:
        del action
        last_edit = _last_receipt_index(tool_receipts, _changed_edit)
        if last_edit is None:
            return "coding_completion_requires_edit"
        last_test = _last_receipt_index(
            tool_receipts,
            _observed_test,
            after_index=last_edit,
        )
        if last_test is None:
            return "coding_completion_requires_test"
        if (
            _submitted_workspace_diff(
                tool_receipts,
                after_index=last_test,
            )
            is None
        ):
            return "coding_completion_requires_changed_diff"
        return None

    def _completion_artifact(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str:
        # SkillFlow submits repository state rather than model prose.  This
        # adapter additionally requires an inspected diff receipt; select only
        # one observed after the latest changed edit and subsequent test so a
        # pre-revision receipt cannot submit a stale patch.
        del action, artifact
        last_edit = _last_receipt_index(tool_receipts, _changed_edit)
        last_test = (
            None
            if last_edit is None
            else _last_receipt_index(
                tool_receipts,
                _observed_test,
                after_index=last_edit,
            )
        )
        workspace_diff = (
            None
            if last_test is None
            else _submitted_workspace_diff(
                tool_receipts,
                after_index=last_test,
            )
        )
        if workspace_diff is None:  # pragma: no cover - guarded by admission
            raise RuntimeError("coding completion has no workspace diff")
        return workspace_diff


__all__ = ["CodingExecutionAdapter"]
