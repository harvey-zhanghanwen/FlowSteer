"""Optional HealthBench tools on the existing SkillFlow bounded ReAct loop.

Direct reuse: ToolReactExecutionAdapter owns parsing, Action--Observation,
continuation, measured shared budgets, recovery, receipts and completion.
HealthBenchAuthoritativeReactExecutionAdapter owns artifact quality and exact
evidence receipt binding. Necessary adaptation: its search-only action domain
cannot express optional source reading, label lookup or calculation. This
subclass widens only that per-node declared domain; it adds no roles, workflow,
rubric access, patient simulator or clinical scoring rule.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .agent_runtime import AgentRequest
from .healthbench_evidence_adapter import (
    HealthBenchAuthoritativeReactExecutionAdapter,
    _evidence_preserves_query_anchors,
    _query_preserves_task_surface,
)
from .openai_gateway import (
    _healthbench_medrag_evidence,
    _healthbench_search_candidates,
)
from .tool_runtime import StructuredAction


_SEARCH_TOOLS = frozenset({
    "healthbench-authoritative.search", "healthbench-medrag.search",
    "healthbench-literature.search", "healthbench-trials.search",
    "healthbench-bookshelf.search", "healthbench-pdq.search",
    "healthbench-ahrq.search", "healthbench-terminology.search",
})


class HealthBenchClinicalReactExecutionAdapter(
    HealthBenchAuthoritativeReactExecutionAdapter
):
    """Admit optional declared tools, never require a medical role or search."""

    def __init__(self, **kwargs: Any) -> None:
        for flag in (
            "require_initial_search", "require_refinement_on_insufficient_evidence",
        ):
            if kwargs.get(flag, False):
                raise ValueError(f"optional clinical tools require {flag}=False")
        super().__init__(**kwargs)

    def _contract(self, request, observations):
        value = super()._contract(request, observations)
        if "healthbench-bookshelf.search" in request.agent.allowed_tools:
            # Necessary task adaptation on the existing ReAct contract, not a
            # role or a question-specific answer/routing rule. Older profiles
            # retain their original text.
            value += (
                "\nInterpret named studies and ambiguous terms using the original conversation "
                "and source context, without changing the requested subject or relation. "
                "An empty search is specific to that query/source, not proof that no study or evidence exists. "
                "Read metadata-only hits before attributing clinical findings; another available source "
                "may resolve uncertainty within the shared budget. Do not assume the contract's "
                "proposed conclusion is established by the sources."
            )
        return value

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        # DIRECT_REUSE: execute() starts its shared counter at the number of
        # retained ToolReceipts. Do not replenish it on a Canvas continuation,
        # or double-count restored Action history beside those same receipts.
        continued_count = len(self._continuation_observations(request.action_history))
        dispatched = len(request.prior_tool_receipts) + sum(
            observation.get("observation_status") in {"success", "tool_error"}
            and isinstance(observation.get("executed_action"), Mapping)
            and observation["executed_action"].get("kind") == "tool"
            for observation in observations[continued_count:]
        )
        if dispatched >= self._max_tool_calls:
            return frozenset(), True
        successful_searches = 0
        for observation in observations:
            action = observation.get("executed_action")
            result = observation.get("result")
            if not (
                observation.get("observation_status") == "success"
                and isinstance(action, Mapping)
                and action.get("resource_id") in _SEARCH_TOOLS
                and action.get("name") == "search"
                and isinstance(result, Mapping)
                and bool(result.get("evidence") or result.get("ranked_chunks"))
            ):
                continue
            relevant_observation = observation
            if action.get("resource_id") == "healthbench-medrag.search" and "ranked_chunks" in result:
                relevant_observation = {
                    **observation,
                    "result": {**result, "evidence": list(_healthbench_medrag_evidence(result))},
                }
            # Direct reuse of the authoritative adapter's opted-in public
            # anchor check. A non-empty but unrelated hit must not consume
            # one of the relevant-search slots merely because a new Tool
            # subclass widened the action domain. All dispatches still spend
            # the original total Tool budget, and completion stays optional.
            if self._require_relevant_evidence and not _evidence_preserves_query_anchors(
                request, relevant_observation,
            ):
                continue
            successful_searches += 1
        admitted = frozenset(
            (tool_id, action_name)
            for tool_id in request.agent.allowed_tools
            for action_name in self._tool_registry.require_capability(tool_id).action_names
            if not (
                tool_id in _SEARCH_TOOLS
                and action_name == "search"
                and successful_searches >= self._max_successful_searches
            )
        )
        return admitted, True

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> dict[str, object]:
        # Reuse the authoritative adapter's strict mutually exclusive action
        # schema; generalize the number of branches, not the wire contract.
        admitted, _ = self._state_conditioned_action_domain(request, observations)
        branches = [
            self._action_schema(
                arguments_schema=self._tool_registry.require_capability(tool_id).action_schemas[name],
                kind="tool", name=name, resource_id=tool_id,
            )
            for tool_id, name in sorted(admitted)
        ]
        branches.append(self._action_schema(
            arguments_schema=self._completion_arguments_schema(request),
            kind="complete", name="complete", resource_id=None,
        ))
        return branches[0] if len(branches) == 1 else {"oneOf": branches}

    def _completion_arguments_schema(self, request: AgentRequest) -> Mapping[str, object]:
        schema = deepcopy(dict(super()._completion_arguments_schema(request)))
        # The exact same evidence shape also binds source-read and drug-label
        # receipts. Do not suggest that calculation is a literature source.
        value = schema["properties"]["value"]
        evidence_value = next(
            (branch for branch in value.get("anyOf", [value]) if branch.get("type") == "object"),
            None,
        )
        if evidence_value is not None:
            if not any(
                tool_id in _SEARCH_TOOLS
                or tool_id in {"healthbench-source.read", "healthbench-drug.lookup", "healthbench-knowledge.search"}
                for tool_id in request.agent.allowed_tools
            ):
                # Availability of a calculator does not turn a numeric
                # contract artifact into a literature-grounded finding.
                schema["properties"]["value"] = {
                    "type": "string", "minLength": 1,
                    "description": "The completed artifact required by the Agent contract.",
                    **(
                        {"maxLength": self._max_completion_artifact_characters}
                        if self._max_completion_artifact_characters is not None else {}
                    ),
                }
                return schema
            fields = evidence_value["properties"]["evidence_items"]["items"]["properties"]
            for name in ("document_id", "source", "title", "date", "url"):
                fields[name]["description"] = (
                    f"Copy {name} exactly from the same successful retrieval "
                    "Observation evidence item; calculation is not literature evidence."
                )
        return schema

    @staticmethod
    def _model_visible_observations(
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        visible = HealthBenchAuthoritativeReactExecutionAdapter._model_visible_observations(observations)
        for observation in visible:
            action = observation.get("executed_action")
            result = observation.get("result")
            if (
                isinstance(action, Mapping)
                and action.get("resource_id") == "healthbench-medrag.search"
                and observation.get("observation_status") == "success"
                and isinstance(result, Mapping)
            ):
                # Convert the unchanged SkillFlow public ranked chunks once,
                # avoiding both duplicated context and lost source metadata.
                observation["result"] = {
                    **{key: value for key, value in result.items() if key != "ranked_chunks"},
                    "evidence": list(_healthbench_medrag_evidence(result)),
                }
        return visible

    @staticmethod
    def _successful_search_evidence(
        tool_receipts: list[dict[str, object]],
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(item for _, _, item in _healthbench_search_candidates(tool_receipts))

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        inherited = super()._tool_action_error(
            request=request, action=action, observations=observations,
        )
        if inherited is not None:
            return inherited
        # The parent's check is scoped to authoritative.search. Reuse only
        # its existing lexical task-anchor boundary for the optional local
        # MedRAG and external literature/registry searches; do not add a medical vocabulary, rewrite entities,
        # tighten local query length, or merge duplicate requests across
        # distinct sources. Drug names/source IDs/calculations are unaffected.
        if (
            self._require_task_query_anchor
            and action.resource_id in _SEARCH_TOOLS - {"healthbench-authoritative.search"}
            and action.name == "search"
            and not _query_preserves_task_surface(request.problem, action.arguments.get("query"))
        ):
            return "query_does_not_preserve_public_task_anchor"
        # All optional capabilities are read-only or pure calculation. Unlike
        # an edit/test environment, another tool cannot change their input
        # state. Reuse prior observations instead of redispatching an exact
        # request anywhere in the same bounded attempt/continuation. Offset
        # changes remain legal for a subsequent source page.
        for observation in observations:
            prior = observation.get("executed_action")
            if (
                observation.get("observation_status") in {"success", "tool_error"}
                and isinstance(prior, Mapping)
                and prior.get("kind") == "tool"
                and prior.get("resource_id") == action.resource_id
                and prior.get("name") == action.name
                and prior.get("arguments") == action.arguments
            ):
                return "duplicate_tool_request"
        return None
