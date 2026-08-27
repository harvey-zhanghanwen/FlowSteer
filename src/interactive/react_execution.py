"""Bounded model-driven Tool/ReAct execution for AgentGraph nodes.

The loop follows SkillFlow's ``BoundedAgent`` public action/observation
contract: one generated ``StructuredAction`` is parsed per turn, executable
actions are admitted against a frozen resource registry, one measured result
is returned as the next public observation, and completion is explicit.  It
does not persist or request hidden chain-of-thought.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType
from typing import Mapping, Optional

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
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.metadata = MappingProxyType(dict(metadata or {}))


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
        self._max_action_tokens = max_action_tokens
        self._execution_mode = execution_mode
        self._sampling_base_seed = sampling_base_seed
        self._sampling_coordinate = sampling_coordinate

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        tools = [
            self._tool_registry.require_capability(tool_id).to_value()
            for tool_id in request.agent.allowed_tools
        ]
        action_schema = {
            "kind": "tool",
            "name": "tool action",
            "arguments": "object matching the selected Tool action schema",
            "resource_id": "one allowed tool_id",
            "skill_id": None,
        }
        completion_schema = {
            "kind": "complete",
            "name": "complete",
            "arguments": {"value": "final artifact"},
            "resource_id": None,
            "skill_id": None,
        }
        return (
            request.agent.contract
            + f"\n\nExecution mode: {self._execution_mode}. Return exactly one JSON StructuredAction "
            "and no other text. Use a tool action only from allowed_tools, or "
            "complete when the declared completion condition is met.\n"
            + "Tool action schema: "
            + json.dumps(action_schema, sort_keys=True, separators=(",", ":"))
            + "\nNever omit an action's required arguments."
            + "\nCompletion schema: "
            + json.dumps(completion_schema, sort_keys=True, separators=(",", ":"))
            + "\nAllowed tools: "
            + json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\nCompletion condition: "
            + (request.agent.completion_condition or "produce the declared artifact")
            + "\nPublic observations: "
            + json.dumps(
                [dict(observation) for observation in observations],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _allowed_tools_for_turn(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[str, ...]:
        """Return the public action mask for this continuation turn."""

        del observations
        return tuple(request.agent.allowed_tools)

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Return the public Tool/completion domain for this continuation.

        DIRECT_REUSE: SkillFlow publishes the legal bounded action set after
        every Observation.  Generic Tool/ReAct execution leaves every action
        and explicit completion available; dataset adapters may narrow that
        domain from measured public state without changing AgentGraph roles or
        topology.
        """

        del request, observations
        return None, True

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        """Return the JSON Schema for an admitted completion's arguments."""

        del request
        return {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {
                    "description": (
                        "The completed artifact required by the Agent contract"
                    )
                }
            },
            "additionalProperties": False,
        }

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> Optional[dict[str, object]]:
        """Build SkillFlow's strict schema when exactly one action is legal.

        SkillFlow sends the request-scoped response schema through
        ``response_format.json_schema``.  A generic Tool domain may contain
        mutually exclusive actions, so this thin adapter constrains a measured
        state with exactly one legal Tool action or completion.  Missing schema
        metadata for such a narrowed state is an execution error rather than an
        unconstrained fallback.
        """

        admitted_tool_actions, completion_admitted = (
            self._state_conditioned_action_domain(request, observations)
        )
        arguments_schema: Optional[Mapping[str, object]] = None
        kind: Optional[str] = None
        name: Optional[str] = None
        resource_id: Optional[str] = None
        if admitted_tool_actions is not None and len(admitted_tool_actions) == 1:
            if completion_admitted:
                return None
            resource_id, name = next(iter(admitted_tool_actions))
            capability = self._tool_registry.require_capability(resource_id)
            arguments_schema = capability.action_schemas.get(name)
            if arguments_schema is None:
                raise ReactExecutionError(
                    "admitted Tool action has no registered argument schema"
                )
            kind = "tool"
        elif (
            admitted_tool_actions is not None
            and not admitted_tool_actions
            and completion_admitted
        ):
            arguments_schema = self._completion_arguments_schema(request)
            kind = "complete"
            name = "complete"
        if arguments_schema is None or kind is None or name is None:
            return None
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

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        mode = getattr(request.agent.execution_mode, "value", request.agent.execution_mode)
        if mode != self._execution_mode:
            raise ReactExecutionError(
                "execution adapter mode does not match the Agent contract"
            )
        observations: list[Mapping[str, object]] = []
        trace: list[dict[str, object]] = []
        tool_receipts: list[dict[str, object]] = []
        model_calls: list[dict[str, object]] = []
        tool_calls = 0

        for turn in range(1, self._max_turns + 1):
            allowed_tools = self._allowed_tools_for_turn(request, observations)
            masked_request = replace(
                request,
                agent=replace(request.agent, allowed_tools=allowed_tools),
            )
            response_schema = self._state_conditioned_response_schema(
                masked_request,
                observations,
            )
            admitted_tool_actions, _ = self._state_conditioned_action_domain(
                masked_request,
                observations,
            )
            if admitted_tool_actions is not None and response_schema is None:
                raise ReactExecutionError(
                    "state-conditioned StructuredAction schema is unavailable"
                )
            model_metadata = dict(masked_request.model.metadata)
            model_metadata["max_tokens"] = str(self._max_action_tokens)
            # The schema belongs to this exact public state.  Never carry a
            # prior turn's schema into the next request.
            model_metadata.pop("response_json_schema", None)
            scientific_sampling_receipt: dict[str, object] | None = None
            requested_sampling: dict[str, object] = {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_tokens": self._max_action_tokens,
                "seed": None,
            }
            if (
                self._sampling_base_seed is not None
                and self._sampling_coordinate is not None
            ):
                generation_seed = derive_generation_seed(
                    base_seed=self._sampling_base_seed,
                    coordinate=self._sampling_coordinate,
                    step_index=turn,
                    phase=GenerationPhase.ACTION,
                )
                # DIRECT_REUSE: SkillFlow fixes temperature=1, top_p=1 and one
                # step-specific seed. Native top_k=-1 remains gated by an
                # explicitly declared local SGLang capability.
                model_metadata["temperature"] = "1.0"
                model_metadata["top_p"] = "1.0"
                model_metadata["generation_seed"] = str(generation_seed)
                top_k = None
                if supports_local_sglang_top_k(masked_request):
                    top_k = -1
                    model_metadata["top_k"] = "-1"
                requested_sampling = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": top_k,
                    "max_tokens": self._max_action_tokens,
                    "seed": generation_seed,
                }
                scientific_sampling_receipt = {
                    "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
                    "base_seed": self._sampling_base_seed,
                    "coordinate": self._sampling_coordinate.to_value(),
                    "phase": GenerationPhase.ACTION.value,
                    "step_index": turn,
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
            agent = replace(
                masked_request.agent,
                contract=self._contract(masked_request, observations),
            )
            turn_request = replace(
                masked_request,
                request_id=f"{request.request_id}:react:{turn}",
                agent=agent,
                model=replace(
                    masked_request.model,
                    metadata=model_metadata,
                ),
            )
            generated = await self._gateway.generate(turn_request)
            response = (
                generated if isinstance(generated, AgentResponse) else AgentResponse(generated)
            )
            model_calls.append(
                {
                    "turn": turn,
                    "request_id": turn_request.request_id,
                    "requested_sampling": dict(requested_sampling),
                    **(
                        {
                            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
                            "scientific_sampling": scientific_sampling_receipt,
                        }
                        if scientific_sampling_receipt is not None
                        else {}
                    ),
                    "metadata": dict(response.metadata),
                }
            )
            entry: dict[str, object] = {
                "turn": turn,
                "action_text": response.text,
            }
            try:
                action = _parse_structured_action(response.text)
            except (TypeError, ValueError) as exc:
                observation = MappingProxyType(
                    {
                        "observation_status": "parse_error",
                        "public_error_code": type(exc).__name__,
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            entry["structured_action"] = action.to_value()
            if action.kind is ActionKind.COMPLETE:
                _, completion_admitted = self._state_conditioned_action_domain(
                    request,
                    observations,
                )
                if not completion_admitted:
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_not_admitted",
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                if (
                    not isinstance(action.arguments, dict)
                    or set(action.arguments) != {"value"}
                ):
                    observation = MappingProxyType(
                        {
                            "observation_status": "schema_invalid",
                            "public_error_code": "completion_schema_invalid",
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
                        }
                    )
                    entry.update(observation)
                    trace.append(entry)
                    observations.append(observation)
                    continue
                entry["observation_status"] = "completed"
                trace.append(entry)
                return AgentResponse(
                    artifact,
                    {
                        "execution_mode": self._execution_mode,
                        "react_turns_used": turn,
                        "tool_calls": tool_calls,
                        "tool_receipts": tool_receipts,
                        "react_trace": trace,
                        "model_calls": model_calls,
                    },
                )

            if action.kind is not ActionKind.TOOL:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "skill_action_not_admitted",
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            if action.resource_id not in allowed_tools:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_not_allowed",
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            capability = self._tool_registry.require_capability(action.resource_id)
            if action.name not in capability.action_names:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_action_not_registered",
                        "tool_id": action.resource_id,
                        "allowed_action_names": list(capability.action_names),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            admitted_tool_actions, _ = self._state_conditioned_action_domain(
                request,
                observations,
            )
            if (
                admitted_tool_actions is not None
                and (action.resource_id, action.name) not in admitted_tool_actions
            ):
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "tool_action_not_admitted",
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue
            action_error = self._tool_action_error(
                request=request,
                action=action,
                observations=observations,
            )
            if action_error is not None:
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": action_error,
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
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

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
                        "error_type": receipt.error_type,
                    }
                )
            else:
                observation = MappingProxyType(
                    {
                        "observation_status": "success",
                        "tool_id": action.resource_id,
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
            metadata={
                "execution_mode": self._execution_mode,
                "react_turns_used": self._max_turns,
                "tool_calls": tool_calls,
                "tool_receipts": tool_receipts,
                "react_trace": trace,
                "model_calls": model_calls,
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
        """Dataset adapters may reject a Tool action from public state."""

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
