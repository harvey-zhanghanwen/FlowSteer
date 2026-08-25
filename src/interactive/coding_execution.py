"""Bounded iterative Coding Agent execution inside AgentGraph.

The model/tool/observation loop reuses :class:`ToolReactExecutionAdapter` and
the SkillFlow-derived repository tools.  The additional completion admission
requires an actual repository edit, an inspected workspace diff, and a test
execution before a patch artifact may leave the coding node.  Official
SWE-bench resolution remains exclusively in ``swebench_adapter.py``.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Mapping, Optional

from .agent_runtime import AgentGateway, AgentRequest, AgentResponse
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
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

    TESTED_DIFF_COMPLETION = "tested_diff_receipt"
    WORKSPACE_DIFF_COMPLETION = "workspace_diff"

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        max_turns: int,
        max_tool_calls: int,
        max_action_tokens: int = 512,
        completion_policy: str = TESTED_DIFF_COMPLETION,
        workspace_diff: Optional[Callable[[], str]] = None,
        repository_runtime_receipt: Optional[
            Callable[[], Mapping[str, object]]
        ] = None,
        task_max_turns: Optional[int] = None,
        task_max_tool_calls: Optional[int] = None,
    ) -> None:
        if completion_policy not in {
            self.TESTED_DIFF_COMPLETION,
            self.WORKSPACE_DIFF_COMPLETION,
        }:
            raise ValueError("unsupported Coding completion policy")
        if (
            completion_policy == self.WORKSPACE_DIFF_COMPLETION
            and workspace_diff is None
        ):
            raise ValueError(
                "workspace_diff completion requires a patch materializer"
            )
        for field_name, value in (
            ("task_max_turns", task_max_turns),
            ("task_max_tool_calls", task_max_tool_calls),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        super().__init__(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_action_tokens=max_action_tokens,
            execution_mode="coding",
        )
        self._completion_policy = completion_policy
        self._workspace_diff = workspace_diff
        self._repository_runtime_receipt = repository_runtime_receipt
        self._per_call_max_turns = max_turns
        self._per_call_max_tool_calls = max_tool_calls
        self._task_max_turns = task_max_turns or max_turns
        self._task_max_tool_calls = task_max_tool_calls or max_tool_calls
        self._task_turns_used = 0
        self._task_tool_calls_used = 0
        # One task owns one mutable repository state.  Reciprocal AgentGraph
        # blocks may schedule Agents concurrently, but SkillFlow's repository
        # episode and its global budget are sequential.  Serialize complete
        # coding executions so repository mutations and budget admission keep
        # one deterministic receipt order without constraining graph topology.
        self._task_execution_lock = asyncio.Lock()

    @property
    def task_budget_receipt(self) -> dict[str, int]:
        return {
            "max_turns": self._task_max_turns,
            "turns_used": self._task_turns_used,
            "max_tool_calls": self._task_max_tool_calls,
            "tool_calls_used": self._task_tool_calls_used,
        }

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute against one budget shared by every coding call in the task.

        NECESSARY_ADAPTATION: SkillFlow has one bounded Agent episode, while a
        progressive FlowSteer Canvas may re-execute or replace its repository
        Agent.  Keeping the counters on the task-scoped adapter prevents a new
        Agent ID or Canvas revision from resetting the execution budget.  The
        Direct and AgentGraph arms each receive an identically configured
        task-scoped adapter and therefore the same repository Tool/turn cap.
        """

        async with self._task_execution_lock:
            return await self._execute_with_task_budget(request)

    async def _execute_with_task_budget(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        remaining_turns = self._task_max_turns - self._task_turns_used
        remaining_tools = self._task_max_tool_calls - self._task_tool_calls_used
        if remaining_turns <= 0:
            raise ReactExecutionError(
                "coding task-global turn budget is exhausted",
                react_trace=tuple(request.action_history),
                tool_receipts=tuple(request.prior_tool_receipts),
                tool_plan_exhausted=True,
            )

        original_turn_limit = self._max_turns
        original_tool_limit = self._max_tool_calls
        prior_receipt_count = len(request.prior_tool_receipts)
        self._max_turns = min(self._per_call_max_turns, remaining_turns)
        self._max_tool_calls = min(
            self._per_call_max_tool_calls,
            prior_receipt_count + max(remaining_tools, 0),
        )
        response: Optional[AgentResponse] = None
        error: Optional[BaseException] = None
        try:
            generated = await super().execute(request)
            if not isinstance(generated, AgentResponse):  # pragma: no cover
                raise TypeError("CodingExecutionAdapter requires AgentResponse")
            response = generated
        except BaseException as exc:
            error = exc
        finally:
            self._max_turns = original_turn_limit
            self._max_tool_calls = original_tool_limit

        if response is not None:
            metadata = dict(response.metadata)
            current_model_calls = metadata.get("model_calls", ())
            current_receipts = metadata.get("tool_receipts", ())
        else:
            assert error is not None
            current_model_calls = getattr(error, "model_calls", ())
            current_receipts = getattr(error, "tool_receipts", ())
        new_turns = (
            len(current_model_calls)
            if isinstance(current_model_calls, (list, tuple))
            else 0
        )
        total_receipts = (
            len(current_receipts)
            if isinstance(current_receipts, (list, tuple))
            else prior_receipt_count
        )
        new_tool_calls = max(total_receipts - prior_receipt_count, 0)
        self._task_turns_used = min(
            self._task_turns_used + new_turns,
            self._task_max_turns,
        )
        self._task_tool_calls_used = min(
            self._task_tool_calls_used + new_tool_calls,
            self._task_max_tool_calls,
        )
        if error is not None:
            raise error
        assert response is not None
        metadata["task_global_budget"] = self.task_budget_receipt
        if self._repository_runtime_receipt is not None:
            metadata["repository_runtime"] = dict(
                self._repository_runtime_receipt()
            )
        return AgentResponse(response.text, metadata)

    def _materialized_workspace_diff(self) -> Optional[str]:
        if self._workspace_diff is None:
            return None
        patch = self._workspace_diff()
        return patch if isinstance(patch, str) and patch.strip() else None

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> Optional[str]:
        del action
        if self._completion_policy == self.WORKSPACE_DIFF_COMPLETION:
            return (
                None
                if self._materialized_workspace_diff() is not None
                else "coding_completion_requires_workspace_diff"
            )
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
        if self._completion_policy == self.WORKSPACE_DIFF_COMPLETION:
            # DIRECT_REUSE: SkillFlow code_generation termination ignores
            # model prose and submits ``_generate_workspace_diff()`` from the
            # current detached worktree.  This also permits a Director to use
            # a search-only or test-optional functional subgraph without
            # imposing one fixed Tool order; a non-empty repository patch is
            # the only Agent-level completion invariant.
            del action, artifact, tool_receipts
            workspace_diff = self._materialized_workspace_diff()
            if workspace_diff is None:  # pragma: no cover - admission guard
                raise RuntimeError("coding completion has no workspace diff")
            return workspace_diff
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
