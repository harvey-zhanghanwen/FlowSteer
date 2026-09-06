"""Bounded model-driven Tool/ReAct execution for AgentGraph nodes.

The loop follows SkillFlow's ``BoundedAgent`` public action/observation
contract: one generated ``StructuredAction`` is parsed per turn, executable
actions are admitted against a frozen resource registry, one measured result
is returned as the next public observation, and completion is explicit.  It
does not persist or request hidden chain-of-thought.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import re
from types import MappingProxyType
from typing import Mapping, Optional

from .aime2026_adapter import (
    extract_aime2026_artifact_assessments,
    extract_aime2026_candidate,
)
from .agent_runtime import (
    AgentGateway,
    AgentRequest,
    AgentResponse,
    GatewayResponse,
)
from .openai_gateway import supports_local_sglang_top_k
from .scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
)
from .tool_runtime import (
    ActionKind,
    StructuredAction,
    ToolRegistry,
    ToolRequest,
)


class ReactExecutionError(RuntimeError):
    """A bounded Tool/ReAct node did not produce a valid completion."""

    def __init__(
        self,
        message: str,
        *,
        react_trace: tuple[Mapping[str, object], ...] = (),
        tool_receipts: tuple[Mapping[str, object], ...] = (),
        model_calls: tuple[Mapping[str, object], ...] = (),
        tool_plan_exhausted: bool = False,
        failure_category: str | None = None,
        failure_reason: str | None = None,
        bounded_regeneration_attempt_count: int | None = None,
        regeneration_exhausted: bool | None = None,
    ) -> None:
        super().__init__(message)
        if type(tool_plan_exhausted) is not bool:
            raise TypeError("tool_plan_exhausted must be bool")
        # DIRECT_REUSE: SkillFlow persists every bounded-agent action and
        # observation even when the turn budget is exhausted.  Carry the same
        # public failure receipt through AgentRuntime's exception chain so a
        # diagnostic failure is not misreported as "no Tool call".
        self.react_trace = tuple(dict(item) for item in react_trace)
        self.tool_receipts = tuple(dict(item) for item in tool_receipts)
        self.model_calls = tuple(dict(item) for item in model_calls)
        self.tool_plan_exhausted = tool_plan_exhausted
        if failure_category is not None:
            if not isinstance(failure_category, str) or not failure_category.strip():
                raise ValueError("failure_category must be non-empty text")
            self.failure_category = failure_category.strip()
        if failure_reason is not None:
            if not isinstance(failure_reason, str) or not failure_reason.strip():
                raise ValueError("failure_reason must be non-empty text")
            self.failure_reason = failure_reason.strip()
        if bounded_regeneration_attempt_count is not None:
            if (
                type(bounded_regeneration_attempt_count) is not int
                or bounded_regeneration_attempt_count < 0
            ):
                raise ValueError(
                    "bounded_regeneration_attempt_count must be a non-negative integer"
                )
            self.bounded_regeneration_attempt_count = (
                bounded_regeneration_attempt_count
            )
        if regeneration_exhausted is not None:
            if type(regeneration_exhausted) is not bool:
                raise TypeError("regeneration_exhausted must be bool")
            self.regeneration_exhausted = regeneration_exhausted


class ReactGenerationError(RuntimeError):
    """One ReAct policy-generation request failed before producing an Action."""

    def __init__(
        self,
        message: str,
        *,
        cause_error_type: str,
        react_trace: tuple[Mapping[str, object], ...] = (),
        tool_receipts: tuple[Mapping[str, object], ...] = (),
        model_calls: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        if not isinstance(cause_error_type, str) or not cause_error_type.strip():
            raise ValueError("cause_error_type must be non-empty text")
        self.cause_error_type = cause_error_type.strip()
        self.react_trace = tuple(dict(item) for item in react_trace)
        self.tool_receipts = tuple(dict(item) for item in tool_receipts)
        self.model_calls = tuple(dict(item) for item in model_calls)


def _continuation_step_offset(
    action_history: tuple[Mapping[str, object], ...],
) -> int:
    """Return the last persisted Action turn after strict monotonic validation.

    A cross-Agent continuation may retain only Tool-bearing turns, so gaps are
    valid.  List length is not a valid generation coordinate in that case.
    """

    last_turn = 0
    for item in action_history:
        turn = item.get("turn")
        if type(turn) is not int or turn < 1:
            raise ValueError(
                "continued ReAct action_history turns must be positive integers"
            )
        if turn <= last_turn:
            raise ValueError(
                "continued ReAct action_history turns must be strictly increasing"
            )
        last_turn = turn
    return last_turn


def _parse_structured_action(text: str) -> StructuredAction:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("action text is empty")
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("action text is not one JSON object") from exc
    return StructuredAction.from_value(value)


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


_PROVENANCE_BOUND_ASSESSMENT_PROTOCOLS = frozenset(
    {
        "provenance_bound_candidate_assessment_v1",
        "provenance_bound_candidate_assessment_v2",
    }
)
_ARTIFACT_ASSESSMENT_BLOCK = re.compile(
    r"<artifact_assessments>\s*.*?\s*</artifact_assessments>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _provenance_bound_candidate_sources(
    request: AgentRequest,
) -> tuple[dict[str, object], ...]:
    """Project model-visible candidate sources without consulting a target."""

    if (
        request.artifact_assessment_protocol
        not in _PROVENANCE_BOUND_ASSESSMENT_PROTOCOLS
        or getattr(
            request.communication_condition,
            "value",
            request.communication_condition,
        )
        != "normal"
    ):
        return ()
    messages = [*request.upstream]
    if request.peer_draft is not None:
        messages.append(request.peer_draft)
    result: list[dict[str, object]] = []
    for message in messages:
        candidate, _, parsing_failure = extract_aime2026_candidate(
            message.artifact
        )
        if candidate is None:
            continue
        result.append(
            {
                "source_agent": message.source_agent_id,
                "artifact_id": message.artifact_id,
                "candidate": candidate,
                "artifact_complete": message.artifact_complete,
                "execution_diagnostic_codes": list(
                    message.execution_diagnostic_codes
                ),
                "upstream_tool_receipt_count": len(message.tool_receipts),
                "candidate_parsing_failure_reason": parsing_failure,
            }
        )
    return tuple(result)


def _has_public_derivation(
    artifact: str,
    *,
    candidates: frozenset[str],
) -> bool:
    """Reject only an assessment completion collapsed to answer markers.

    This is a structural completeness check, not a correctness judgment.  It
    deliberately does not inspect the benchmark target or recompute a result.
    """

    without_assessments = _ARTIFACT_ASSESSMENT_BLOCK.sub("", artifact)
    retained_lines: list[str] = []
    for raw_line in without_assessments.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        compact = (
            re.sub(r"\s+", "", line)
            .replace("$", "")
            .casefold()
            .rstrip(".!")
        )
        marker_forms = {
            candidate.casefold()
            for candidate in candidates
        }
        marker_forms.update(
            form
            for candidate in candidates
            for form in {
                f"finalanswer:{candidate}".casefold(),
                f"answer:{candidate}".casefold(),
                f"answer={candidate}".casefold(),
                f"theansweris{candidate}".casefold(),
                f"theansweris:{candidate}".casefold(),
                f"answeris{candidate}".casefold(),
                f"answeris:{candidate}".casefold(),
                f"\\boxed{{{candidate}}}".casefold(),
            }
        )
        if compact in marker_forms:
            continue
        if compact in {"artifactassessment", "artifactassessments"}:
            continue
        retained_lines.append(line)
    return bool(" ".join(retained_lines).strip())


class ToolReactExecutionAdapter:
    """Execute an Agent-selected bounded sequence of registered Tool calls."""

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        max_turns: int,
        max_tool_calls: int,
        max_action_tokens: int = 512,
        thinking_budget_tokens: int | None = None,
        execution_mode: str = "react",
        sampling_base_seed: int | None = None,
        sampling_coordinate: ScientificSamplingCoordinate | None = None,
    ) -> None:
        if not hasattr(gateway, "generate"):
            raise TypeError("gateway must implement generate")
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if type(max_tool_calls) is not int or max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a non-negative integer")
        if type(max_action_tokens) is not int or max_action_tokens < 1:
            raise ValueError("max_action_tokens must be a positive integer")
        if thinking_budget_tokens is not None and (
            type(thinking_budget_tokens) is not int
            or thinking_budget_tokens < 1
        ):
            raise ValueError(
                "thinking_budget_tokens must be a positive integer or None"
            )
        if execution_mode not in {"react", "coding"}:
            raise ValueError("execution_mode must be react or coding")
        if (sampling_base_seed is None) != (sampling_coordinate is None):
            raise ValueError(
                "sampling_base_seed and sampling_coordinate must be supplied together"
            )
        if sampling_base_seed is not None and (
            type(sampling_base_seed) is not int
            or not 0 <= sampling_base_seed < 2**64
        ):
            raise ValueError("sampling_base_seed must be an unsigned 64-bit integer")
        if sampling_coordinate is not None and not isinstance(
            sampling_coordinate,
            ScientificSamplingCoordinate,
        ):
            raise TypeError(
                "sampling_coordinate must be a ScientificSamplingCoordinate"
            )
        if (
            sampling_base_seed is not None
            and sampling_coordinate is not None
            and sampling_coordinate.sampling_schedule_hash
            != scientific_sampling_schedule_hash(base_seed=sampling_base_seed)
        ):
            raise ValueError(
                "sampling_coordinate schedule hash does not match sampling_base_seed"
            )
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        # DIRECT_REUSE: SkillFlow RolloutDecoding.max_action_tokens bounds one
        # action generation independently from the outer Agent completion
        # budget.  ReAct observations grow over turns, so reusing the generic
        # 4096-token completion allowance can exceed an 8K context window.
        self._max_action_tokens = max_action_tokens
        self._thinking_budget_tokens = thinking_budget_tokens
        self._execution_mode = execution_mode
        self._sampling_base_seed = sampling_base_seed
        self._sampling_coordinate = sampling_coordinate
        # AgentRuntime's timeout boundary normalizes CancelledError.  Preserve
        # the already-materialized public request receipt by request ID, using
        # the same handoff contract as EnvironmentExecutionAdapter.
        self._cancelled_prefixes: dict[str, Mapping[str, object]] = {}

    def take_cancelled_failure_metadata(
        self,
        request_id: str,
    ) -> Mapping[str, object]:
        """Consume one cancelled Action-generation receipt."""

        return self._cancelled_prefixes.pop(request_id, MappingProxyType({}))

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Return the model-visible action domain for the current public state.

        The generic SkillFlow-style bounded executor exposes every registered
        Tool action plus explicit completion.  Dataset adapters may narrow
        that public domain when their environment contract has a measured
        state transition; ``None`` preserves the complete Tool domain.
        """

        del request, observations
        return None, True

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        """Return the JSON Schema for an admitted completion's arguments."""

        candidate_sources = _provenance_bound_candidate_sources(request)
        value_schema: dict[str, object] = {
            "description": (
                "The completed artifact required by the Agent contract"
            )
        }
        if candidate_sources:
            # PROJECT_NECESSARY_ADAPTATION: SkillFlow keeps completion in the
            # five-field StructuredAction ``arguments.value``.  For this
            # project's provenance-bound AIME protocol, constrain that same
            # field to the complete textual artifact so a scalar candidate is
            # not structurally interchangeable with a downstream assessment.
            # Runtime admission below remains authoritative for exact source
            # bindings and never consults the evaluator target.
            value_schema.update(
                {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The complete public derivation followed by the exact "
                        "provenance-bound <artifact_assessments> JSON block; "
                        "a bare candidate is not a complete artifact"
                    ),
                }
            )
        return {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": value_schema,
            },
            "additionalProperties": False,
        }

    @staticmethod
    def _provenance_bound_completion_guidance(
        request: AgentRequest,
    ) -> str:
        """Expose exact source identities without prescribing a workflow."""

        candidate_sources = _provenance_bound_candidate_sources(request)
        if not candidate_sources:
            return ""
        public_bindings = [
            {
                "source_agent": source["source_agent"],
                "artifact_id": source["artifact_id"],
                "candidate": source["candidate"],
                "artifact_complete": source["artifact_complete"],
                "execution_diagnostic_codes": source[
                    "execution_diagnostic_codes"
                ],
            }
            for source in candidate_sources
        ]
        return (
            "\nProvenance-bound COMPLETE wire contract: arguments.value must "
            "preserve the contract-relevant public derivation and any public "
            "Tool observation used; a bare scalar candidate is incomplete. "
            "Append exactly one <artifact_assessments> JSON-array block and "
            "assess every candidate-bearing source below. Copy each "
            "artifact_id and candidate exactly. source_agent identifies the "
            "binding but is not an extra assessment-object field. The block "
            "must be strict JSON: use plain-text mathematical basis or escape "
            "every literal backslash inside JSON strings. Do not "
            "consult or infer a benchmark target. Required source bindings: "
            + json.dumps(
                public_bindings,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> Optional[dict[str, object]]:
        """Build SkillFlow's strict five-field StructuredAction schema.

        SkillFlow's OpenAI provider sends ``ModelRequest.response_schema`` as
        ``response_format.json_schema``.  A generic ReAct state admits every
        declared Tool action plus explicit completion, so represent that
        choice as ``oneOf`` rather than dropping structured decoding exactly
        when Tool use is enabled.  Dataset adapters with their own multi-branch
        conditioning keep their existing override; this base method still
        supplies their singleton schemas.  The strict parser remains
        authoritative after generation.
        """

        admitted_tool_actions, completion_admitted = (
            self._state_conditioned_action_domain(request, observations)
        )
        if admitted_tool_actions is None:
            admitted_tool_actions = frozenset(
                (capability.tool_id, action_name)
                for tool_id in request.agent.allowed_tools
                for capability in (self._tool_registry.require_capability(tool_id),)
                for action_name in capability.action_names
            )
        elif (
            type(self)._state_conditioned_response_schema
            is not ToolReactExecutionAdapter._state_conditioned_response_schema
            and (
                len(admitted_tool_actions) > 1
                or (admitted_tool_actions and completion_admitted)
            )
        ):
            # Preserve existing dataset-specific multi-branch schema builders.
            return None
        return self._response_schema_for_domain(
            request=request,
            admitted_tool_actions=admitted_tool_actions,
            completion_admitted=completion_admitted,
        )

    def _response_schema_for_domain(
        self,
        *,
        request: AgentRequest,
        admitted_tool_actions: frozenset[tuple[str, str]],
        completion_admitted: bool,
    ) -> Optional[dict[str, object]]:
        """Serialize one measured action domain without changing its choices."""

        branches: list[dict[str, object]] = []
        for resource_id, name in sorted(admitted_tool_actions):
            capability = self._tool_registry.require_capability(resource_id)
            arguments_schema = capability.action_schemas.get(name)
            if not isinstance(arguments_schema, Mapping):
                continue
            branches.append(
                self._structured_action_response_branch(
                    arguments_schema=arguments_schema,
                    kind="tool",
                    name=name,
                    resource_id=resource_id,
                )
            )
        if completion_admitted:
            branches.append(
                self._structured_action_response_branch(
                    arguments_schema=self._completion_arguments_schema(request),
                    kind="complete",
                    name="complete",
                    resource_id=None,
                )
            )
        if not branches:
            return None
        if len(branches) == 1:
            return branches[0]
        return {"oneOf": branches}

    @staticmethod
    def _structured_action_response_branch(
        *,
        arguments_schema: Mapping[str, object],
        kind: str,
        name: str,
        resource_id: Optional[str],
    ) -> dict[str, object]:
        """Return SkillFlow's exact five-field StructuredAction branch."""

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
                "arguments": dict(arguments_schema),
                "kind": {"const": kind},
                "name": {"const": name},
                "resource_id": {"const": resource_id},
                "skill_id": {"const": None},
            },
            "additionalProperties": False,
        }

    @staticmethod
    def _model_visible_observations(
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Render canonical public observations without replaying invalid actions.

        SkillFlow persists the complete sampled Action in the trajectory while
        presenting an invalid next-turn Observation as status/error feedback.
        Keep successful Action--Observation state intact, but do not feed a
        malformed action body back to the model as an imitation target.
        """

        visible: list[dict[str, object]] = []
        invalid_fields = (
            "observation_status",
            "public_error_code",
            "tool_id",
            "action_name",
            "argument_validation",
            "allowed_action_names",
            "expected_top_level_fields",
            "forbidden_wrapper_fields",
            "repair_instruction",
            "repeat_count",
            "cached_observation",
        )
        for observation in observations:
            if observation.get("observation_status") in {
                "parse_error",
                "schema_invalid",
            }:
                visible.append(
                    {
                        key: observation[key]
                        for key in invalid_fields
                        if key in observation
                    }
                )
            else:
                visible.append(dict(observation))
        return visible

    @staticmethod
    def _continuation_observations(
        action_history: tuple[Mapping[str, object], ...],
    ) -> list[Mapping[str, object]]:
        """Restore SkillFlow public Action--Observation continuation state.

        Successful Tool turns persist their Observation under ``observation``;
        parse/schema failures publish the canonical error fields directly on
        the trace entry.  Only those public fields are restored, never model
        hidden state or an unvalidated semantic artifact.
        """

        result: list[Mapping[str, object]] = []
        for entry in action_history:
            nested = entry.get("observation")
            if isinstance(nested, Mapping):
                result.append(MappingProxyType(dict(nested)))
                continue
            status = entry.get("observation_status")
            if not isinstance(status, str):
                continue
            public = {
                key: entry[key]
                for key in (
                    "observation_status",
                    "public_error_code",
                    "tool_id",
                    "action_name",
                    "argument_validation",
                    "allowed_action_names",
                    "expected_top_level_fields",
                    "forbidden_wrapper_fields",
                    "repair_instruction",
                    "repeat_count",
                    "cached_observation",
                    "executed_action",
                    "error_type",
                )
                if key in entry
            }
            result.append(MappingProxyType(public))
        return result

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        capabilities = [
            self._tool_registry.require_capability(tool_id)
            for tool_id in request.agent.allowed_tools
        ]
        without_fixed_actions = [
            capability.tool_id
            for capability in capabilities
            if not capability.action_schemas
        ]
        if without_fixed_actions:
            raise ReactExecutionError(
                "generic Tool/ReAct execution requires fixed action schemas for "
                + ", ".join(without_fixed_actions)
            )
        tool_ids = [capability.tool_id for capability in capabilities]
        admitted_tool_actions, completion_admitted = (
            self._state_conditioned_action_domain(request, observations)
        )
        action_contracts = [
            (
                capability.tool_id,
                action_name,
                dict(argument_schema),
            )
            for capability in capabilities
            for action_name, argument_schema in capability.action_schemas.items()
            if admitted_tool_actions is None
            or (capability.tool_id, action_name) in admitted_tool_actions
        ]
        action_contract_text = "\n".join(
            "- kind is \"tool\"; name is "
            + json.dumps(action_name, ensure_ascii=False)
            + "; resource_id is "
            + json.dumps(tool_id, ensure_ascii=False)
            + "; skill_id is null; Arguments JSON Schema is "
            + json.dumps(
                argument_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for tool_id, action_name, argument_schema in action_contracts
        )
        coding_wire_guidance = ""
        if self._execution_mode == "coding" and any(
            isinstance(argument_schema.get("properties"), Mapping)
            and "code" in argument_schema["properties"]
            for _, _, argument_schema in action_contracts
        ):
            # NECESSARY_ADAPTATION: the StructuredAction boundary is reused
            # directly from SkillFlow, while a coding Tool additionally needs
            # its source program to survive JSON transport intact.  These are
            # neutral wire constraints only; they do not prescribe a task
            # method, Agent role, topology, or mathematical workflow.
            coding_wire_guidance = (
                "\nCoding Tool wire contract: arguments.code must contain the "
                "complete executable source program. Encode source line breaks "
                "as JSON \\n escapes so the decoded arguments.code contains real "
                "newlines. Include an explicit print(...) statement for the Tool "
                "observation. Do not put natural-language reasoning, prose, "
                "Markdown fences, or an unfinished source prefix in arguments.code."
            )
        provenance_completion_guidance = (
            self._provenance_bound_completion_guidance(request)
        )
        # DIRECT_REUSE: SkillFlow rollout/context.py::_ACTION_GUIDANCE uses
        # ``arguments={"value": ...}`` for completion.  Do not place a
        # concrete placeholder such as ``"final artifact"`` in the public
        # action example: generation models can copy it verbatim and thereby
        # sever the semantic artifact routed to the next AgentGraph node.
        completion_schema = {
            "kind": {"const": "complete"},
            "name": {"const": "complete"},
            "arguments": dict(self._completion_arguments_schema(request)),
            "resource_id": {"const": None},
            "skill_id": {"const": None},
        }
        # DIRECT_REUSE: SkillFlow's ReAct prompt carries action_history and
        # the latest public observation into the next turn.  State explicitly
        # that this is continuation state; otherwise the local policy can
        # restart the first action on every turn even though the observation
        # is present.  This does not choose an action or encode a workflow.
        continuation_guidance = (
            "\nContinue from the newest public observation; do not restart "
            "the first step. Do not repeat an identical Tool request after "
            "either success or failure; after a failure, change the request "
            "or complete with an explicit insufficient-evidence artifact."
            if observations
            else ""
        )
        return (
            request.agent.contract
            + f"\n\nExecution mode: {self._execution_mode}. Return exactly one JSON StructuredAction "
            "and no other text. Use a tool action only from allowed_tools, or "
            "complete when the declared completion condition is met.\n"
            # DIRECT_REUSE: this is the exact public action guidance and
            # five-field wire contract enforced by SkillFlow
            # rollout/context.py::_ACTION_GUIDANCE and
            # runtime/contracts.py::StructuredAction.from_value.  The local
            # Qwen service otherwise sometimes emits the legacy
            # {"action": ..., "arguments": ...} ToolRequest shape, which is
            # not an admitted StructuredAction.
            + "Choose exactly one action from the currently admissible schemas "
            "below. Every action object must contain exactly these five fields: "
            "arguments, kind, name, resource_id, skill_id. Do not use an "
            "action field. For a tool action use kind=tool, the exact name and "
            "resource_id below, and skill_id=null. "
            + (
                "For completion use kind=complete, name=complete, "
                "arguments={\"value\": ...}, resource_id=null, and skill_id=null.\n"
                if completion_admitted
                else "A completion action is not currently admissible.\n"
            )
            + "Currently admissible Tool action contracts follow. These are field "
            "constraints, not wrapper fields and not response objects. Put "
            "arguments, kind, name, resource_id, and skill_id directly in the "
            "single top-level JSON object. The arguments value must be an "
            "instance of the stated JSON Schema; never return the schema itself. "
            "Never emit action_envelope or argument_json_schema fields.\n"
            + (action_contract_text or "- none")
            + coding_wire_guidance
            + provenance_completion_guidance
            + (
                "\nCurrently admissible completion schema: "
                + json.dumps(
                    completion_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if completion_admitted
                else "\nCompletion is not admissible in the current public state."
            )
            + "\nAllowed tool resource IDs: "
            + json.dumps(
                tool_ids,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\nCompletion condition: "
            + (request.agent.completion_condition or "produce the declared artifact")
            + "\nPublic observations: "
            + json.dumps(
                self._model_visible_observations(observations),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + continuation_guidance
        )

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        mode = getattr(request.agent.execution_mode, "value", request.agent.execution_mode)
        if mode != self._execution_mode:
            raise ReactExecutionError(
                "execution adapter mode does not match the Agent contract"
            )
        observations = self._continuation_observations(request.action_history)
        continuation_source = request.continuation_source_agent_id
        trace: list[dict[str, object]] = [
            {
                **dict(item),
                "continued_from_prior_revision": True,
                **(
                    {"continuation_source_agent_id": continuation_source}
                    if continuation_source is not None
                    else {}
                ),
            }
            for item in request.action_history
        ]
        tool_receipts: list[dict[str, object]] = [
            dict(item) for item in request.prior_tool_receipts
        ]
        model_calls: list[dict[str, object]] = []
        tool_calls = len(tool_receipts)
        continuation_turn_count = len(trace)
        truncation_regeneration_pending = False
        try:
            continuation_step_offset = _continuation_step_offset(
                request.action_history
            )
        except ValueError as exc:
            raise ReactExecutionError(
                str(exc),
                react_trace=tuple(trace),
                tool_receipts=tuple(tool_receipts),
            ) from exc
        last_dispatched_tool_action_key: Optional[str] = None
        duplicate_tool_request_count = 0
        for observation in reversed(observations):
            executed_action = observation.get("executed_action")
            if (
                isinstance(executed_action, Mapping)
                and executed_action.get("kind") == "tool"
            ):
                last_dispatched_tool_action_key = json.dumps(
                    dict(executed_action),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                break

        for turn in range(1, self._max_turns + 1):
            admitted_tool_actions, completion_admitted = (
                self._state_conditioned_action_domain(request, observations)
            )
            effective_tool_actions = admitted_tool_actions
            action_domain_narrowed = False
            remaining_turns = self._max_turns - turn + 1
            if completion_admitted and (
                remaining_turns == 1 or tool_calls >= self._max_tool_calls
            ):
                # NECESSARY_ADAPTATION: the Tool call and COMPLETE are separate
                # SkillFlow StructuredActions.  Do not spend the last available
                # Action turn, or a post-Tool-budget repair turn, on another
                # Tool request and then fail without giving the policy its
                # explicit completion transition.
                effective_tool_actions = frozenset()
                action_domain_narrowed = True
            elif not completion_admitted and remaining_turns == 1:
                effective_tool_actions = frozenset()
                action_domain_narrowed = True
            if (
                effective_tool_actions is not None
                and not effective_tool_actions
                and not completion_admitted
            ):
                # NECESSARY_ADAPTATION: SkillFlow bounds one Agent rollout but
                # does not publish this project's explicit empty action mask.
                # Stop at that measured no-transition state instead of spending
                # later turns on actions the current Tool contract must reject;
                # dataset adapters attach the typed terminal diagnosis while
                # preserving the exact public prefix.
                raise ReactExecutionError(
                    "bounded ReAct action domain is exhausted",
                    react_trace=tuple(trace),
                    tool_receipts=tuple(tool_receipts),
                    model_calls=tuple(model_calls),
                    tool_plan_exhausted=True,
                )
            if action_domain_narrowed:
                response_schema = self._response_schema_for_domain(
                    request=request,
                    admitted_tool_actions=effective_tool_actions,
                    completion_admitted=completion_admitted,
                )
            else:
                response_schema = self._state_conditioned_response_schema(
                    request,
                    observations,
                )
            truncation_regeneration = truncation_regeneration_pending
            turn_max_action_tokens = (
                min(self._max_action_tokens, 512)
                if truncation_regeneration
                else self._max_action_tokens
            )
            turn_thinking_budget_tokens = (
                None if truncation_regeneration else self._thinking_budget_tokens
            )
            # DIRECT_REUSE: SkillFlow keeps max_reasoning_tokens and
            # max_action_tokens as independent decoding limits.  Its
            # thinking-enabled OpenAI path adds the reasoning allowance to the
            # provider max_tokens value, so reasoning cannot consume the
            # StructuredAction serialization budget.
            turn_provider_max_tokens = turn_max_action_tokens + (
                turn_thinking_budget_tokens or 0
            )
            model_metadata = {
                **dict(request.model.metadata),
                "max_tokens": str(turn_provider_max_tokens),
            }
            if truncation_regeneration:
                # DIRECT_REUSE: SkillFlow
                # training/batch_inference.py::supervisor_call performs one
                # short same-model retry after an otherwise unparseable
                # finish_reason=length response. Its recovery request uses a
                # 512-token bound and disables thinking. The malformed Action
                # remains in the public receipt but is not replayed as an
                # imitation target.
                model_metadata["chat_template_enable_thinking"] = "false"
                model_metadata["require_reasoning_trace"] = "false"
                model_metadata.pop("thinking_budget_tokens", None)
            elif turn_thinking_budget_tokens is not None:
                model_metadata["thinking_budget_tokens"] = str(
                    turn_thinking_budget_tokens
                )
            absolute_step_index = continuation_step_offset + turn
            scientific_sampling_receipt: dict[str, object] | None = None
            requested_sampling: dict[str, object] = {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_tokens": turn_provider_max_tokens,
                "seed": None,
                "thinking_budget": turn_thinking_budget_tokens,
            }
            if (
                self._sampling_base_seed is not None
                and self._sampling_coordinate is not None
            ):
                generation_seed = derive_generation_seed(
                    base_seed=self._sampling_base_seed,
                    coordinate=self._sampling_coordinate,
                    step_index=absolute_step_index,
                    phase=GenerationPhase.ACTION,
                )
                # DIRECT_REUSE: SkillFlow fixes temperature=1, top_p=1 and one
                # step-specific scientific seed.  Its SGLang-native top_k=-1
                # is forwarded only through an explicitly declared compatible
                # local provider; portable remote OpenAI providers remain
                # untouched and the receipt records top_k=None.
                model_metadata["temperature"] = "1.0"
                model_metadata["top_p"] = "1.0"
                model_metadata["generation_seed"] = str(generation_seed)
                top_k = None
                if supports_local_sglang_top_k(request):
                    top_k = -1
                    model_metadata["top_k"] = "-1"
                requested_sampling = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": top_k,
                    "max_tokens": turn_provider_max_tokens,
                    "seed": generation_seed,
                    "thinking_budget": turn_thinking_budget_tokens,
                }
                scientific_sampling_receipt = {
                    "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
                    "base_seed": self._sampling_base_seed,
                    "coordinate": self._sampling_coordinate.to_value(),
                    "phase": GenerationPhase.ACTION.value,
                    "step_index": absolute_step_index,
                    "generation_seed": generation_seed,
                    "requested_sampling": dict(requested_sampling),
                }
            if response_schema is not None:
                model_metadata["response_json_schema"] = json.dumps(
                    response_schema,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            contract = self._contract(request, observations)
            if truncation_regeneration:
                contract += (
                    "\nYour previous response was too long and got truncated. "
                    "Return exactly one short StructuredAction immediately. "
                    "If the final artifact is ready, return the complete "
                    "StructuredAction."
                )
            if action_domain_narrowed and completion_admitted:
                contract += (
                    "\nThe remaining Action budget now admits only the explicit "
                    "complete StructuredAction. Do not issue another Tool action."
                )
            agent = replace(
                request.agent,
                contract=contract,
            )
            turn_request = replace(
                request,
                request_id=f"{request.request_id}:react:{turn}",
                agent=agent,
                model=replace(
                    request.model,
                    metadata=model_metadata,
                ),
            )
            model_call: dict[str, object] = {
                "turn": absolute_step_index,
                "request_id": turn_request.request_id,
                "requested_sampling": dict(requested_sampling),
                "max_action_tokens": turn_max_action_tokens,
                "max_reasoning_tokens": turn_thinking_budget_tokens,
                "request_status": "requested",
                "generation_mode": (
                    "structured_action_truncation_regeneration"
                    if truncation_regeneration
                    else "action"
                ),
                **(
                    {
                        "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
                        "scientific_sampling": scientific_sampling_receipt,
                    }
                    if scientific_sampling_receipt is not None
                    else {}
                ),
            }
            try:
                generated = await self._gateway.generate(turn_request)
                response = (
                    generated
                    if isinstance(generated, AgentResponse)
                    else AgentResponse(generated)
                )
            except asyncio.CancelledError as exc:
                model_call["request_status"] = "cancelled"
                model_call["error_type"] = type(exc).__name__
                model_calls.append(model_call)
                cancellation_metadata = MappingProxyType(
                    {
                        "react_trace": tuple(dict(item) for item in trace),
                        "tool_receipts": tuple(
                            dict(item) for item in tool_receipts
                        ),
                        "model_calls": tuple(
                            dict(item) for item in model_calls
                        ),
                        "cause_error_type": type(exc).__name__,
                    }
                )
                exc.react_trace = cancellation_metadata["react_trace"]
                exc.tool_receipts = cancellation_metadata["tool_receipts"]
                exc.model_calls = cancellation_metadata["model_calls"]
                exc.cause_error_type = type(exc).__name__
                self._cancelled_prefixes[request.request_id] = (
                    cancellation_metadata
                )
                raise
            except Exception as exc:
                failure_sampling = getattr(exc, "requested_sampling", None)
                if isinstance(failure_sampling, Mapping):
                    model_call["requested_sampling"] = dict(failure_sampling)
                model_call["request_status"] = "failed"
                model_call["error_type"] = type(exc).__name__
                model_calls.append(model_call)
                error = ReactGenerationError(
                    f"{type(exc).__name__}: "
                    + (" ".join(str(exc).split()) or "generation failed"),
                    cause_error_type=type(exc).__name__,
                    react_trace=tuple(trace),
                    tool_receipts=tuple(tool_receipts),
                    model_calls=tuple(model_calls),
                )
                for field_name in (
                    "provider_id",
                    "model_id",
                    "http_status",
                    "request_status",
                ):
                    value = getattr(exc, field_name, None)
                    if value is not None:
                        setattr(error, field_name, value)
                raise error from exc
            response_sampling = response.metadata.get("requested_sampling")
            if isinstance(response_sampling, Mapping):
                model_call["requested_sampling"] = dict(response_sampling)
            model_call["request_status"] = "completed"
            model_call["metadata"] = dict(response.metadata)
            model_calls.append(model_call)
            entry: dict[str, object] = {
                "turn": absolute_step_index,
                "action_text": response.text,
            }
            try:
                action = _parse_structured_action(response.text)
            except (TypeError, ValueError) as exc:
                finish_reason = response.metadata.get("finish_reason")
                if finish_reason == "length" or truncation_regeneration:
                    public_error_code = (
                        "structured_action_regeneration_failed"
                        if truncation_regeneration
                        else "structured_action_truncated"
                    )
                    observation = MappingProxyType(
                        {
                            "observation_status": "parse_error",
                            "public_error_code": public_error_code,
                            "finish_reason": finish_reason,
                            "expected_top_level_fields": [
                                "arguments",
                                "kind",
                                "name",
                                "resource_id",
                                "skill_id",
                            ],
                            "forbidden_wrapper_fields": [
                                "action_envelope",
                                "argument_json_schema",
                            ],
                            "repair_instruction": (
                                "Your response was too long and got truncated. "
                                "Return exactly one short StructuredAction "
                                "immediately. If the final artifact is ready, "
                                "return the complete StructuredAction."
                            ),
                            "bounded_regeneration_attempted": (
                                truncation_regeneration
                            ),
                            "action_text": response.text,
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    if truncation_regeneration or turn >= self._max_turns:
                        raise ReactExecutionError(
                            "StructuredAction serialization remained invalid "
                            "after one bounded same-model regeneration",
                            react_trace=tuple(trace),
                            tool_receipts=tuple(tool_receipts),
                            model_calls=tuple(model_calls),
                            tool_plan_exhausted=False,
                            failure_category=(
                                "structured_action_serialization_failure"
                            ),
                            failure_reason="output_truncation",
                            bounded_regeneration_attempt_count=(
                                1 if truncation_regeneration else 0
                            ),
                            regeneration_exhausted=True,
                        ) from exc
                    truncation_regeneration_pending = True
                    continue
                observation = MappingProxyType(
                    {
                        "observation_status": "parse_error",
                        "public_error_code": type(exc).__name__,
                        "expected_top_level_fields": [
                            "arguments",
                            "kind",
                            "name",
                            "resource_id",
                            "skill_id",
                        ],
                        "forbidden_wrapper_fields": [
                            "action_envelope",
                            "argument_json_schema",
                        ],
                        "repair_instruction": (
                            "Return exactly one StructuredAction JSON object and "
                            "place the five expected fields directly at its top "
                            "level; do not wrap them in action_envelope."
                        ),
                        # Persist the sampled public Action in the trajectory.
                        # ``_model_visible_observations`` sends only canonical
                        # error feedback into the next turn, so malformed JSON
                        # is not replayed as an imitation target.
                        "action_text": response.text,
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            truncation_regeneration_pending = False
            entry["structured_action"] = action.to_value()
            if action.kind is ActionKind.COMPLETE:
                if not completion_admitted:
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_action_not_admitted",
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                if not isinstance(action.arguments, dict) or "value" not in action.arguments:
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_schema_invalid",
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                artifact = _completion_text(action.arguments["value"])
                if not artifact.strip():
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_empty",
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                completion_error = self._completion_error(
                    action=action,
                    artifact=artifact,
                    tool_receipts=tool_receipts,
                )
                if completion_error is not None:
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": completion_error,
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                artifact = self._completion_artifact(
                    action=action,
                    artifact=artifact,
                    tool_receipts=tool_receipts,
                )
                if not isinstance(artifact, str) or not artifact.strip():
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_artifact_empty",
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                (
                    provenance_completion_error,
                    provenance_completion_receipt,
                ) = self._provenance_bound_completion_admission(
                    request=request,
                    artifact=artifact,
                    tool_receipts=tool_receipts,
                )
                if provenance_completion_error is not None:
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": (
                                provenance_completion_error
                            ),
                            "repair_instruction": (
                                self._provenance_bound_completion_repair_instruction(
                                    request
                                )
                            ),
                            "executed_action": action.to_value(),
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                if provenance_completion_receipt is not None:
                    entry["artifact_assessment_completion_receipt"] = dict(
                        provenance_completion_receipt
                    )
                entry["observation_status"] = "completed"
                trace.append(entry)
                return AgentResponse(
                    artifact,
                    {
                        "execution_mode": self._execution_mode,
                        "react_turns_used": absolute_step_index,
                        "new_react_turns_used": turn,
                        "continued_action_history_count": continuation_turn_count,
                        "continued_tool_receipt_count": len(
                            request.prior_tool_receipts
                        ),
                        "continuation_source_agent_id": continuation_source,
                        "tool_calls": tool_calls,
                        "tool_receipts": tool_receipts,
                        "react_trace": trace,
                        "model_calls": model_calls,
                        **(
                            {
                                "artifact_assessment_completion_receipt": dict(
                                    provenance_completion_receipt
                                )
                            }
                            if provenance_completion_receipt is not None
                            else {}
                        ),
                    },
                )

            if action.kind is not ActionKind.TOOL:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "skill_action_not_admitted",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            if (
                effective_tool_actions is not None
                and (action.resource_id, action.name) not in effective_tool_actions
            ):
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_action_not_admitted",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            if action.resource_id not in request.agent.allowed_tools:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_not_allowed",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            capability = self._tool_registry.require_capability(action.resource_id)
            # NECESSARY_ADAPTATION: SkillFlow publishes the fixed action domain
            # in model-visible task context and leaves semantic validation to
            # the environment.  AgentGraph carries that same domain in
            # ToolCapability, so reject an unpublished name and arguments that
            # violate its Draft 2020-12 schema before dispatch.  The concrete
            # backend remains authoritative for operation-specific semantics.
            if action.name not in capability.action_names:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_action_not_registered",
                        "tool_id": action.resource_id,
                        "allowed_action_names": list(capability.action_names),
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            admission_error = self._tool_action_error(
                request=request,
                action=action,
                observations=observations,
            )
            if admission_error is not None:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": admission_error,
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            if tool_calls >= self._max_tool_calls:
                observation = MappingProxyType(
                    {
                        "observation_status": "budget_exhausted",
                        "public_error_code": "tool_call_budget_exhausted",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            if not isinstance(action.arguments, dict):
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_arguments_not_object",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            argument_validation_error = capability.argument_validation_error(
                action.name,
                action.arguments,
            )
            if argument_validation_error is not None:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_arguments_schema_invalid",
                        "tool_id": action.resource_id,
                        "action_name": action.name,
                        "argument_validation": argument_validation_error,
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            action_key = json.dumps(
                action.to_value(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if action_key == last_dispatched_tool_action_key:
                # SkillFlow exposes each Action--Observation transition to the
                # policy.  Suppress only an immediately repeated executable
                # action in the same Tool-interaction state.  A different Tool
                # action may change that state (for example, an edit before
                # rerunning the same test command), so the earlier request must
                # then be admissible again.
                duplicate_tool_request_count += 1
                cached_observation = next(
                    (
                        dict(previous)
                        for previous in reversed(observations)
                        if previous.get("observation_status")
                        in {"success", "tool_error"}
                    ),
                    {},
                )
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "duplicate_tool_request",
                        "repeat_count": duplicate_tool_request_count,
                        "cached_observation": cached_observation,
                        "repair_instruction": (
                            "The identical Tool action was already dispatched. "
                            "Use its cached observation and COMPLETE when the "
                            "declared artifact is ready, or choose a different "
                            "admissible Tool action that can add new evidence."
                        ),
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            last_dispatched_tool_action_key = action_key
            duplicate_tool_request_count = 0
            tool_calls += 1
            result, receipt = await self._tool_registry.ainvoke_with_receipt(
                action.resource_id,
                ToolRequest(action.name, action.arguments),
            )
            serialized_receipt = receipt.to_value()
            tool_receipts.append(serialized_receipt)
            if result is None:
                observation = MappingProxyType(
                    {
                        "observation_status": "tool_error",
                        "tool_id": action.resource_id,
                        # SkillFlow's next-turn action_history includes the
                        # already generated action beside its observation.
                        # Preserve that public continuation state so the
                        # policy can distinguish a retry from the next step.
                        "executed_action": action.to_value(),
                        "error_type": receipt.error_type,
                    }
                )
            elif isinstance(result.value, Mapping) and result.value.get("ok") is False:
                # The backend invocation completed and therefore has a normal
                # receipt, but computation/coding Tools use payload ``ok=false``
                # for an operation-level failure.  Expose both facts instead of
                # mislabelling the observation as a successful Tool result.
                observation = MappingProxyType(
                    {
                        "observation_status": "tool_error",
                        "public_error_code": "tool_result_not_ok",
                        "tool_id": action.resource_id,
                        "executed_action": action.to_value(),
                        "tool_version": receipt.tool_version,
                        "result": result.value,
                        "completed": result.completed,
                        "tool_invocation_status": "completed",
                    }
                )
            else:
                observation = MappingProxyType(
                    {
                        "observation_status": "success",
                        "tool_id": action.resource_id,
                        "executed_action": action.to_value(),
                        "tool_version": receipt.tool_version,
                        "result": result.value,
                        "completed": result.completed,
                    }
                )
            entry["observation"] = dict(observation)
            trace.append(entry)
            observations.append(observation)

        raise ReactExecutionError(
            f"{self._execution_mode} agent {request.agent.id!r} exhausted "
            f"{self._max_turns} turns "
            "without a valid completion",
            react_trace=tuple(trace),
            tool_receipts=tuple(tool_receipts),
            model_calls=tuple(model_calls),
        )

    @staticmethod
    def _provenance_bound_completion_repair_instruction(
        request: AgentRequest,
    ) -> str:
        sources = _provenance_bound_candidate_sources(request)
        exact_bindings = [
            {
                "source_agent": source["source_agent"],
                "assessed_artifact_id": source["artifact_id"],
                "candidate": source["candidate"],
            }
            for source in sources
        ]
        return (
            "Return one complete StructuredAction whose arguments.value "
            "preserves the public derivation and ends with exactly one "
            "<artifact_assessments> JSON-array block. Include one assessment "
            "for every exact source binding below; copy assessed_artifact_id "
            "and candidate without alteration. A bare scalar candidate is "
            "not a complete assessment artifact. Use strict JSON and escape "
            "every literal backslash inside JSON strings. Required bindings: "
            + json.dumps(
                exact_bindings,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def _provenance_bound_completion_admission(
        *,
        request: AgentRequest,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> tuple[Optional[str], Optional[dict[str, object]]]:
        """Validate one public assessment artifact against routed provenance.

        This keeps SkillFlow's COMPLETE action unchanged and adds only the
        project protocol admission needed by a downstream AgentGraph node. It
        compares public artifact identities and candidates, never correctness
        and never the hidden benchmark target.
        """

        candidate_sources = _provenance_bound_candidate_sources(request)
        if not candidate_sources:
            return None, None
        if any(
            not isinstance(source.get("artifact_id"), str)
            or not str(source["artifact_id"]).strip()
            for source in candidate_sources
        ):
            return "provenance_completion_source_artifact_id_missing", None
        expected_by_artifact_id: dict[str, dict[str, object]] = {}
        for source in candidate_sources:
            artifact_id = str(source["artifact_id"])
            if artifact_id in expected_by_artifact_id:
                return "provenance_completion_duplicate_source_artifact_id", None
            expected_by_artifact_id[artifact_id] = source
        source_candidates = frozenset(
            str(source["candidate"])
            for source in candidate_sources
        )
        if not _has_public_derivation(
            artifact,
            candidates=source_candidates,
        ):
            return "provenance_completion_public_derivation_missing", None
        assessments, parsing_failure = (
            extract_aime2026_artifact_assessments(artifact)
        )
        if parsing_failure is not None:
            return (
                "provenance_completion_assessment_invalid:"
                + parsing_failure,
                None,
            )
        assessment_by_artifact_id = {
            str(assessment["assessed_artifact_id"]): dict(assessment)
            for assessment in assessments
        }
        if set(assessment_by_artifact_id) != set(expected_by_artifact_id):
            return "provenance_completion_assessed_artifacts_mismatch", None
        assessment_bindings: list[dict[str, object]] = []
        for artifact_id, source in expected_by_artifact_id.items():
            assessment = assessment_by_artifact_id[artifact_id]
            if str(assessment.get("candidate")) != str(source["candidate"]):
                return "provenance_completion_candidate_binding_mismatch", None
            assessment_bindings.append(
                {
                    "source_agent": source["source_agent"],
                    "artifact_id": artifact_id,
                    **assessment,
                }
            )
        output_candidate, _, output_parsing_failure = (
            extract_aime2026_candidate(artifact)
        )
        return (
            None,
            {
                "protocol": request.artifact_assessment_protocol,
                "admission_status": "admitted",
                "public_derivation_preserved": True,
                "candidate_sources": [
                    dict(source) for source in candidate_sources
                ],
                "assessment_bindings": assessment_bindings,
                "output_candidate": output_candidate,
                "output_candidate_parsing_failure_reason": (
                    output_parsing_failure
                ),
                # Full Tool receipts remain in the adjacent canonical
                # ``tool_receipts`` metadata field; count them here to bind
                # this completion admission to that same public execution.
                "local_tool_receipt_count": len(tool_receipts),
            },
        )

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> Optional[str]:
        """Dataset adapters may add public completion admission checks."""

        del action, artifact, tool_receipts
        return None

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> Optional[str]:
        """Dataset adapters may add public action-admission checks."""

        del request, action, observations
        return None

    def _completion_artifact(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str:
        """Return the admitted public artifact after completion validation."""

        del action, tool_receipts
        return artifact


__all__ = [
    "ReactExecutionError",
    "ToolReactExecutionAdapter",
]
