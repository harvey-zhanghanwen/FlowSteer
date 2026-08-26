"""Bounded iterative Coding Agent execution inside AgentGraph.

The model/tool/observation loop reuses :class:`ToolReactExecutionAdapter` and
the SkillFlow-derived repository tools.  The additional completion admission
requires an actual repository edit, an inspected workspace diff, and a test
execution before a patch artifact may leave the coding node.  Official
SWE-bench resolution remains exclusively in ``swebench_adapter.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from typing import Callable, Mapping, Optional

from .agent_runtime import AgentGateway, AgentRequest, AgentResponse
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
from .scientific_sampling import ScientificSamplingCoordinate
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
    if action not in {
        "apply_patch",
        "edit_file",
        "exact_edit",
        "str_replace_editor",
    }:
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

    # The task-scoped lock below serializes every complete coding execution
    # against the one mutable SWE-bench worktree.  AgentRuntime may therefore
    # route the same registered repository resource to multiple graph Agents
    # without concurrent repository mutation or task-budget races.
    serializes_stateful_resource = True

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
        sampling_base_seed: Optional[int] = None,
        sampling_coordinate: Optional[ScientificSamplingCoordinate] = None,
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
            sampling_base_seed=sampling_base_seed,
            sampling_coordinate=sampling_coordinate,
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
        task_scoped_tool_binding: tuple[str, ...] = ()
        if not request.agent.allowed_tools:
            # NECESSARY_ADAPTATION: SkillFlow's SWE-bench Coding Agent enters
            # an environment whose repository actions are always available;
            # they are not a semantic role or a Director Canvas action.  A
            # free AgentGraph coding node may therefore omit ``allowed_tools``
            # without losing the task-scoped repository environment.  Keep an
            # explicit Director selection unchanged and bind only an empty
            # declaration to the resources already registered for this task.
            task_scoped_tool_binding = self._tool_registry.resource_ids
            if task_scoped_tool_binding:
                request = replace(
                    request,
                    agent=replace(
                        request.agent,
                        allowed_tools=task_scoped_tool_binding,
                    ),
                )
        remaining_turns = self._task_max_turns - self._task_turns_used
        remaining_tools = self._task_max_tool_calls - self._task_tool_calls_used
        if remaining_turns <= 0 or remaining_tools <= 0:
            # DIRECT_REUSE: SkillFlow force-terminates a code-generation
            # episode at either global bound and submits the exact current
            # workspace diff, including the empty string.  Do not resample a
            # coding Agent after the task-scoped repository budget is spent.
            response = self._force_terminated_response(
                request=request,
                current_model_calls=(),
                current_receipts=tuple(request.prior_tool_receipts),
                react_trace=tuple(request.action_history),
                new_turns=0,
                termination_reason=(
                    "task_global_turn_budget"
                    if remaining_turns <= 0
                    else "task_global_tool_budget"
                ),
            )
            return self._attach_task_runtime_metadata(
                response,
                task_scoped_tool_binding=task_scoped_tool_binding,
            )

        original_turn_limit = self._max_turns
        original_tool_limit = self._max_tool_calls
        prior_receipt_count = len(request.prior_tool_receipts)
        effective_turn_limit = min(self._per_call_max_turns, remaining_turns)
        self._max_turns = effective_turn_limit
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
        if (
            error is not None
            and isinstance(error, ReactExecutionError)
            and self._completion_policy == self.WORKSPACE_DIFF_COMPLETION
            and new_turns >= effective_turn_limit
            and str(error).endswith("without a valid completion")
        ):
            # DIRECT_REUSE: SkillFlow's code-generation environment calls
            # ``_force_terminate`` at the episode bound and submits the
            # current repository diff when one exists.  Preserve the same
            # termination semantic here instead of discarding a legal patch
            # merely because the model did not emit a separate completion
            # action on its final bounded turn.
            response = self._force_terminated_response(
                request=request,
                current_model_calls=tuple(current_model_calls),
                current_receipts=tuple(current_receipts),
                react_trace=tuple(getattr(error, "react_trace", ())),
                new_turns=new_turns,
                termination_reason="max_turns",
            )
            error = None
        if error is not None:
            raise error
        assert response is not None
        return self._attach_task_runtime_metadata(
            response,
            task_scoped_tool_binding=task_scoped_tool_binding,
        )

    def _attach_task_runtime_metadata(
        self,
        response: AgentResponse,
        *,
        task_scoped_tool_binding: tuple[str, ...],
    ) -> AgentResponse:
        metadata = dict(response.metadata)
        metadata["task_global_budget"] = self.task_budget_receipt
        metadata["task_scoped_tool_binding"] = {
            "source": "ToolRegistry.resource_ids",
            "applied": bool(task_scoped_tool_binding),
            "resource_ids": list(task_scoped_tool_binding),
        }
        if self._repository_runtime_receipt is not None:
            metadata["repository_runtime"] = dict(
                self._repository_runtime_receipt()
            )
        return AgentResponse(response.text, metadata)

    def _force_terminated_response(
        self,
        *,
        request: AgentRequest,
        current_model_calls: tuple[object, ...],
        current_receipts: tuple[object, ...],
        react_trace: tuple[object, ...],
        new_turns: int,
        termination_reason: str,
    ) -> AgentResponse:
        """Return SkillFlow's exact bounded-episode repository artifact."""

        workspace_diff = self.materialize_workspace_diff()
        last_turn = (
            react_trace[-1].get("turn")
            if react_trace and isinstance(react_trace[-1], Mapping)
            else None
        )
        return AgentResponse(
            workspace_diff,
            {
                "execution_mode": "coding",
                "react_turns_used": (
                    last_turn
                    if isinstance(last_turn, int)
                    else len(request.action_history) + new_turns
                ),
                "new_react_turns_used": new_turns,
                "continued_action_history_count": len(request.action_history),
                "continued_tool_receipt_count": len(
                    request.prior_tool_receipts
                ),
                "continuation_source_agent_id": (
                    request.continuation_source_agent_id
                ),
                "tool_calls": len(current_receipts),
                "tool_receipts": tuple(
                    dict(item) for item in current_receipts
                    if isinstance(item, Mapping)
                ),
                "react_trace": tuple(
                    dict(item) for item in react_trace
                    if isinstance(item, Mapping)
                ),
                "model_calls": tuple(
                    dict(item) for item in current_model_calls
                    if isinstance(item, Mapping)
                ),
                "truncated": True,
                "termination_reason": termination_reason,
                "workspace_diff_submitted": bool(workspace_diff.strip()),
                "termination_source": (
                    "SkillFlow training.environment._force_terminate"
                ),
            },
        )

    def _materialized_workspace_diff(self) -> Optional[str]:
        if self._workspace_diff is None:
            return None
        patch = self._workspace_diff()
        return patch if isinstance(patch, str) and patch.strip() else None

    def materialize_workspace_diff(self) -> str:
        """Return the task worktree diff used by SkillFlow termination/evaluation."""

        patch = self._materialized_workspace_diff()
        return patch if patch is not None else ""

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Expose completion only after SkillFlow has a submit-ready diff.

        DIRECT_REUSE: SkillFlow's ``code_generation`` episode has repository
        actions during the episode and submits ``_generate_workspace_diff`` at
        termination.  A prose-only completion before the workspace changes is
        therefore not a legal code-generation action.  Repository actions stay
        fully open; this is only the measured completion admission boundary.
        """

        if self._completion_policy == self.WORKSPACE_DIFF_COMPLETION:
            del request, observations
            return None, self._materialized_workspace_diff() is not None
        return super()._state_conditioned_action_domain(request, observations)

    @staticmethod
    def _shorten_one_line(value: object, limit: int) -> str:
        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    @classmethod
    def _skillflow_observation_summary(
        cls,
        observation: Mapping[str, object],
    ) -> dict[str, object]:
        """Port SkillFlow's bounded SWE_MEMORY Tool-result summaries.

        DIRECT_REUSE: ``training/environment.py::_swe_memory_summary`` keeps
        at most six recent repository observations and summarizes search,
        view, edit, command, and test results.  SkillFlow stores observations
        as text; this runtime stores the same public receipt as a mapping, so
        this is the minimal representation adapter for that upstream policy.
        """

        action = observation.get("executed_action")
        action_value = action if isinstance(action, Mapping) else {}
        arguments = action_value.get("arguments")
        arguments_value = arguments if isinstance(arguments, Mapping) else {}
        name = str(action_value.get("name") or "")
        result = observation.get("result")
        result_value = result if isinstance(result, Mapping) else {}
        summary: dict[str, object] = {
            "action": name,
            "observation_status": observation.get("observation_status"),
        }
        if observation.get("public_error_code") is not None:
            summary["public_error_code"] = observation.get("public_error_code")
        if name == "search_code":
            summary["query"] = cls._shorten_one_line(
                arguments_value.get("query"), 90
            )
            matches = result_value.get("matches")
            if isinstance(matches, list):
                summary["matches"] = [
                    {
                        "path": item.get("path"),
                        "line": item.get("line"),
                        "text": cls._shorten_one_line(item.get("text"), 100),
                    }
                    for item in matches[:3]
                    if isinstance(item, Mapping)
                ]
            summary["match_count"] = result_value.get("match_count")
        elif name == "view_file":
            summary.update(
                path=result_value.get("path") or arguments_value.get("path"),
                start_line=result_value.get("start_line"),
                end_line=result_value.get("end_line"),
            )
            lines = result_value.get("lines")
            if isinstance(lines, list):
                selected = lines if len(lines) <= 6 else lines[:3] + lines[-3:]
                summary["lines"] = [
                    {
                        "line": item.get("line"),
                        "text": cls._shorten_one_line(item.get("text"), 110),
                    }
                    for item in selected
                    if isinstance(item, Mapping)
                ]
        elif name == "edit_file":
            summary.update(
                path=result_value.get("path") or arguments_value.get("path"),
                instruction=cls._shorten_one_line(
                    arguments_value.get("instruction"), 140
                ),
                ok=result_value.get("ok"),
                changed=result_value.get("changed"),
                workspace_diff_nonempty=result_value.get(
                    "workspace_diff_nonempty"
                ),
            )
        elif name in {"bash", "run_tests"}:
            command = arguments_value.get("command")
            if command is None:
                command = arguments_value.get("test_cmd")
            summary.update(
                command=cls._shorten_one_line(command, 180),
                ok=result_value.get("ok"),
                returncode=result_value.get("returncode"),
                passed=result_value.get("passed"),
                timed_out=result_value.get("timed_out"),
                output=cls._shorten_one_line(
                    result_value.get("output")
                    or result_value.get("stdout")
                    or result_value.get("stderr"),
                    240,
                ),
            )
        elif name == "list_files":
            files = result_value.get("files")
            if isinstance(files, list):
                summary["files"] = [str(item) for item in files[:20]]
                summary["file_count"] = len(files)
        else:
            summary["result"] = cls._shorten_one_line(
                json.dumps(result_value, ensure_ascii=False, sort_keys=True),
                280,
            )
        return summary

    @classmethod
    def _model_visible_observations(
        cls,
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Expose SkillFlow's bounded current observation plus SWE_MEMORY."""

        canonical = ToolReactExecutionAdapter._model_visible_observations(
            observations
        )
        if not canonical:
            return []
        prior = [
            cls._skillflow_observation_summary(item)
            for item in canonical[:-1]
        ]
        compressed: list[dict[str, object]] = []
        for item in prior:
            if compressed and {
                key: value
                for key, value in compressed[-1].items()
                if key != "repeated"
            } == item:
                compressed[-1]["repeated"] = int(
                    compressed[-1].get("repeated", 1)
                ) + 1
            else:
                compressed.append(dict(item))
        memory = compressed[-6:]
        current = dict(canonical[-1])
        current_summary = cls._skillflow_observation_summary(current)
        result = current.get("result")
        result_value = result if isinstance(result, Mapping) else {}
        action_name = str(current_summary.get("action") or "")
        if action_name == "view_file":
            lines = result_value.get("lines")
            if isinstance(lines, list):
                current_summary["current_lines"] = [
                    {
                        "line": item.get("line"),
                        "text": str(item.get("text", "")),
                    }
                    for item in lines[:60]
                    if isinstance(item, Mapping)
                ]
        elif action_name == "search_code":
            matches = result_value.get("matches")
            if isinstance(matches, list):
                current_summary["current_matches"] = [
                    dict(item) for item in matches[:20] if isinstance(item, Mapping)
                ]
        elif action_name == "edit_file":
            current_summary["updated_snippet"] = str(
                result_value.get("updated_snippet") or ""
            )[:2000]
        elif action_name in {"bash", "run_tests"}:
            output = str(
                result_value.get("output")
                or result_value.get("stdout")
                or result_value.get("stderr")
                or ""
            )
            if len(output) > 6000:
                output = (
                    output[:4000]
                    + "\n[OBSERVATION_TRUNCATED]\n"
                    + output[-1800:]
                )
            current_summary["current_output"] = output
        visible: list[dict[str, object]] = []
        if memory:
            visible.append(
                {
                    "SWE_MEMORY": memory,
                    "omitted_prior_observations": max(
                        0, len(compressed) - len(memory)
                    ),
                }
            )
        visible.append({"current_observation": current_summary})
        return visible

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        """Append SkillFlow's code-generation episode guidance.

        DIRECT_REUSE: the behavioral rules come from
        ``training/task_prompts.py::CODE_GENERATION``.  The one necessary
        adaptation binds SkillFlow's private ``M_exec`` natural-language
        editor to this task-scoped ToolRegistry.  It does not prescribe an
        AgentGraph role or topology.
        """

        return (
            super()._contract(request, observations)
            + "\n\nSkillFlow code-generation episode guidance: Fix the bug in "
            "the real repository using the provided source tools. The evaluator "
            "submits the workspace diff at the end. Never edit test files. Make "
            "minimal changes. The repository root is already the current working "
            "directory: every path passed to a source tool must be repository-relative; "
            "never use /testbed, /workspace, environment paths, or cd to an absolute "
            "path. Do not repeat an identical search, view, edit, or command after "
            "either success or failure. Keep track of the current "
            "best source candidate, the evidence observed for it, and remaining "
            "uncertainty. When the source location and expected behavior are clear, "
            "use edit_file with a repository-relative path and a precise natural-"
            "language instruction grounded in the viewed source. Tool strings are "
            "plain source/shell text, "
            "not HTML entities. Inspect real source files; do not use "
            "synthetic reproductions or example datasets as evidence. A non-empty "
            "workspace diff is the submitted artifact, not explanatory prose."
        )

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> Optional[dict[str, object]]:
        """Bind every SkillFlow repository action to its exact argument schema.

        DIRECT_REUSE: SkillFlow publishes each repository action as one native
        function schema.  The local text-generation boundary represents that
        same function choice as mutually exclusive five-field
        ``StructuredAction`` branches.  This prevents parameters for one Tool
        from being nested into or mixed with another Tool's arguments while
        leaving the action choice to the Agent.
        """

        exact = super()._state_conditioned_response_schema(
            request,
            observations,
        )
        if exact is not None:
            return exact
        admitted_tool_actions, completion_admitted = (
            self._state_conditioned_action_domain(request, observations)
        )
        if admitted_tool_actions is None:
            action_pairs = {
                (capability.tool_id, action_name)
                for tool_id in request.agent.allowed_tools
                for capability in (
                    self._tool_registry.require_capability(tool_id),
                )
                for action_name in capability.action_names
            }
        else:
            action_pairs = set(admitted_tool_actions)
        if not action_pairs and not completion_admitted:
            return None
        branches: list[dict[str, object]] = []
        for resource_id, action_name in sorted(action_pairs):
            capability = self._tool_registry.require_capability(resource_id)
            argument_schema = capability.action_schemas.get(action_name)
            if not isinstance(argument_schema, Mapping):  # pragma: no cover
                continue
            branches.append(
                {
                    "type": "object",
                    "required": [
                        "arguments",
                        "kind",
                        "name",
                        "resource_id",
                        "skill_id",
                    ],
                    "properties": {
                        "arguments": dict(argument_schema),
                        "kind": {"const": "tool"},
                        "name": {"const": action_name},
                        "resource_id": {"const": resource_id},
                        "skill_id": {"const": None},
                    },
                    "additionalProperties": False,
                }
            )
        if completion_admitted:
            branches.append(
                {
                    "type": "object",
                    "required": [
                        "arguments",
                        "kind",
                        "name",
                        "resource_id",
                        "skill_id",
                    ],
                    "properties": {
                        "arguments": dict(
                            self._completion_arguments_schema(request)
                        ),
                        "kind": {"const": "complete"},
                        "name": {"const": "complete"},
                        "resource_id": {"const": None},
                        "skill_id": {"const": None},
                    },
                    "additionalProperties": False,
                }
            )
        if len(branches) == 1:
            return branches[0]
        return {"oneOf": branches}

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
