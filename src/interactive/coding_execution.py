"""Bounded iterative Coding Agent execution inside AgentGraph.

The model/tool/observation loop reuses :class:`ToolReactExecutionAdapter` and
the SkillFlow-derived repository tools.  The additional completion admission
requires an actual repository edit, an inspected workspace diff, and a test
execution before a patch artifact may leave the coding node.  Official
SWE-bench resolution remains exclusively in ``swebench_adapter.py``.
"""

from __future__ import annotations

from typing import Optional

from .agent_runtime import AgentGateway
from .react_execution import ToolReactExecutionAdapter
from .tool_runtime import StructuredAction, ToolRegistry


def _receipt_action(receipt: dict[str, object]) -> Optional[str]:
    request = receipt.get("request")
    if not isinstance(request, dict):
        return None
    action = request.get("action")
    return action if isinstance(action, str) else None


def _changed_diff(receipt: dict[str, object]) -> bool:
    if _receipt_action(receipt) != "diff":
        return False
    result = receipt.get("result")
    if not isinstance(result, dict):
        return False
    value = result.get("value")
    return isinstance(value, dict) and value.get("changed") is True


def _submitted_workspace_diff(
    tool_receipts: list[dict[str, object]],
) -> Optional[str]:
    """Return the most recent non-empty workspace diff observation."""

    for receipt in reversed(tool_receipts):
        if not _changed_diff(receipt):
            continue
        result = receipt.get("result")
        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if not isinstance(value, dict):
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
    ) -> None:
        super().__init__(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
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
        actions = [_receipt_action(receipt) for receipt in tool_receipts]
        if "exact_edit" not in actions:
            return "coding_completion_requires_edit"
        if "run_tests" not in actions:
            return "coding_completion_requires_test"
        if _submitted_workspace_diff(tool_receipts) is None:
            return "coding_completion_requires_changed_diff"
        return None

    def _completion_artifact(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str:
        # SkillFlow submits ``_generate_workspace_diff()`` at terminal time;
        # prose or a model-reconstructed patch is never substituted for the
        # actual prepared worktree state.  The changed-diff admission above
        # guarantees this lookup succeeds.
        del action, artifact
        workspace_diff = _submitted_workspace_diff(tool_receipts)
        if workspace_diff is None:  # pragma: no cover - guarded by admission
            raise RuntimeError("coding completion has no workspace diff")
        return workspace_diff


__all__ = ["CodingExecutionAdapter"]
