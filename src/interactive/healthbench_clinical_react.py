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
import json
import re
from typing import Any, Mapping

from .agent_runtime import AgentRequest, CommunicationCondition
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
from .healthbench_professional_adapter import parse_model_visible_conversation


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

    def __init__(self, *, public_task_validation: bool = False, **kwargs: Any) -> None:
        if type(public_task_validation) is not bool:
            raise ValueError("public_task_validation must be boolean")
        self._public_task_validation = public_task_validation
        for flag in (
            "require_initial_search", "require_refinement_on_insufficient_evidence",
        ):
            if kwargs.get(flag, False):
                raise ValueError(f"optional clinical tools require {flag}=False")
        super().__init__(**kwargs)

    def _contract(self, request, observations):
        value = super()._contract(request, observations)
        if self._public_task_validation:
            value += (
                "\nAn Agent contract is a proposed assignment, not verified evidence. "
                "Check the original conversation for contradictory patient attributes "
                "and unsupported premises before accepting an upstream conclusion. "
                "A status=insufficient artifact is not a verified finding. Preserve "
                "its uncertainty and distinguish the task product from a status report. "
                "A completed answer must contain the requested material, not just a "
                "title, patient background, promise, or statement of completion. "
                "Limited search results cannot prove that evidence does not exist."
            )
            literal = self._literal_initial_query(request, observations)
            if literal is not None:
                value += (
                    " First identify the short study request without an inferred "
                    "expansion. If searching, use this literal query first: "
                    + json.dumps(literal, ensure_ascii=False)
                    + ". Subsequent queries may refine it using the observations."
                )
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

    def _literal_initial_query(self, request, observations):
        """Public-name lookup on the existing admission boundary, not aliases.

        Only a short standalone study request receives this first-query rule.
        Clinical conversations, source reads and calculations are unaffected.
        A completed literal lookup, even empty, unlocks ordinary refinement.
        """
        if not self._public_task_validation:
            return None
        try:
            messages = parse_model_visible_conversation(
                request.problem, include_rubric_context=False,
            )
        except ValueError:
            return None
        users = [message["content"].strip() for message in messages if message["role"] == "user"]
        if len(users) != 1:
            return None
        query = users[0]
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not (2 <= len(words) <= min(12, self._max_query_content_tokens)) or len(query) > 160 or "\n" in query:
            return None
        if not re.search(r"\b(?:trial|study)\b", query, flags=re.IGNORECASE):
            return None
        normalize = lambda text: " ".join(str(text).casefold().split())
        search_tools = _SEARCH_TOOLS | {"healthbench-knowledge.search"}
        for observation in observations:
            action = observation.get("executed_action", {})
            if (observation.get("observation_status") == "success"
                    and isinstance(action, Mapping) and action.get("name") == "search"
                    and action.get("resource_id") in search_tools
                    and action.get("arguments", {}).get("database") != "conversation"
                    and normalize(action.get("arguments", {}).get("query", "")) == normalize(query)):
                return None
        # Reuse an actual upstream lookup; graph ancestry or a status message
        # alone cannot demonstrate that the literal query was executed.
        receipts = list(request.prior_tool_receipts)
        for message in (() if request.communication_condition is CommunicationCondition.UPSTREAM_MASKED else request.upstream):
            receipts.extend(message.tool_receipts)
        for receipt in receipts:
            result = receipt.get("result", {})
            arguments = receipt.get("request", {}).get("arguments", {})
            if (receipt.get("tool_id") in search_tools and not receipt.get("error_type")
                    and isinstance(result, Mapping) and result.get("completed") is True
                    and arguments.get("database") != "conversation"
                    and normalize(arguments.get("query", "")) == normalize(query)):
                return None
        return query

    def _public_search_action_error(self, request, action, observations):
        search_tools = _SEARCH_TOOLS | {"healthbench-knowledge.search"}
        if action.name != "search" or action.resource_id not in search_tools:
            return None
        if action.arguments.get("database") == "conversation":
            return None
        literal = self._literal_initial_query(request, observations)
        if literal is not None and (
            " ".join(str(action.arguments.get("query", "")).casefold().split())
            != " ".join(literal.casefold().split())
        ):
            return "initial_study_lookup_must_preserve_literal_public_request"
        if (self._public_task_validation and self._require_task_query_anchor
                and not _query_preserves_task_surface(request.problem, action.arguments.get("query"))):
            return "query_does_not_preserve_public_task_anchor"
        return None

    def _completion_error(self, *, action, artifact, tool_receipts):
        inherited = super()._completion_error(
            action=action, artifact=artifact, tool_receipts=tool_receipts,
        )
        if inherited is not None or not self._public_task_validation:
            return inherited
        value = action.arguments.get("value")
        text = value.get("summary", "") if isinstance(value, Mapping) else artifact
        normalized = " ".join(str(text).split())
        # Conservative surface checks, not a semantic grader. No medical
        # answer dictionary or minimum response length is introduced.
        if re.fullmatch(
            r"(?:translation|summary|review|verification|assessment|task|work|output)"
            r"(?:\s+[\w-]+){0,7}\s+(?:complete[d]?|finished|ready)"
            r"(?:\s*[-:—]\s*(?:[\w-]+\s+){0,5}(?:required|pending)"
            r"(?:\s+[\w-]+){0,6})?[.!]?", normalized, flags=re.IGNORECASE,
        ):
            return "completion_artifact_is_status_only_provide_actual_task_product"
        unqualified_absence = re.search(
            r"\b(?:no\s+(?:(?:randomized|controlled|clinical|relevant|published)\s+){0,3}"
            r"(?:trials?|studies)\s+(?:exists?|is\s+available|was\s+found|were\s+found|was\s+identified|were\s+identified))\b",
            normalized, flags=re.IGNORECASE,
        )
        bounded_scope = re.search(
            r"\b(?:(?:our|my|this|these|the\s+limited)\s+(?:search|searches|retrieval)"
            r"|(?:searched|consulted|accessible|retrieved)\s+(?:sources|records|results|databases)"
            r"|(?:does\s+not|cannot)\s+(?:prove|establish|rule\s+out))\b",
            normalized, flags=re.IGNORECASE,
        )
        if unqualified_absence and not bounded_scope:
            return "limited_search_cannot_establish_absence_state_retrieval_scope_or_supported_finding"
        return None

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
        literal = self._literal_initial_query(request, observations)
        branches = [
            self._action_schema(
                arguments_schema=self._tool_registry.require_capability(tool_id).action_schemas[name],
                kind="tool", name=name, resource_id=tool_id,
            )
            for tool_id, name in sorted(admitted)
        ]
        if literal is not None:
            conversation_branches = []
            for branch in branches:
                properties = branch["properties"]
                if (properties["name"].get("const") == "search"
                        and properties["resource_id"].get("const")
                        in _SEARCH_TOOLS | {"healthbench-knowledge.search"}):
                    arguments = deepcopy(properties["arguments"])
                    if properties["resource_id"].get("const") == "healthbench-knowledge.search":
                        conversation_branch = deepcopy(branch)
                        conversation_branch["properties"]["arguments"]["properties"]["database"] = {"const": "conversation"}
                        conversation_branches.append(conversation_branch)
                        arguments["properties"]["database"] = {"enum": ["medical_references", "drug_labels"]}
                    # The provider sees the same constraint admission enforces;
                    # do not spend a model turn discovering a hidden rule.
                    arguments["properties"]["query"]["const"] = literal
                    properties["arguments"] = arguments
            branches.extend(conversation_branches)
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
        public_error = self._public_search_action_error(request, action, observations)
        if public_error is not None:
            return public_error
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
