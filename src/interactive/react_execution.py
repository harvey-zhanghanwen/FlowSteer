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
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        # DIRECT_REUSE: SkillFlow RolloutDecoding.max_action_tokens bounds one
        # action generation independently from the outer Agent completion
        # budget.  ReAct observations grow over turns, so reusing the generic
        # 4096-token completion allowance can exceed an 8K context window.
        self._max_action_tokens = max_action_tokens
        self._execution_mode = execution_mode

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

        SkillFlow's OpenAI provider sends ``ModelRequest.response_schema`` as
        ``response_format.json_schema``.  Generic Tool domains may contain
        several mutually exclusive actions, so this thin adapter constrains
        only a measured state with exactly one legal Tool action or completion.
        The strict parser remains authoritative after generation.
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
        trace: list[dict[str, object]] = [
            {**dict(item), "continued_from_prior_revision": True}
            for item in request.action_history
        ]
        tool_receipts: list[dict[str, object]] = [
            dict(item) for item in request.prior_tool_receipts
        ]
        model_calls: list[dict[str, object]] = []
        tool_calls = len(tool_receipts)
        continuation_turn_count = len(trace)
        last_dispatched_tool_action_key: Optional[str] = None
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
            response_schema = self._state_conditioned_response_schema(
                request,
                observations,
            )
            model_metadata = {
                **dict(request.model.metadata),
                "max_tokens": str(self._max_action_tokens),
            }
            if response_schema is not None:
                model_metadata["response_json_schema"] = json.dumps(
                    response_schema,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            agent = replace(
                request.agent,
                contract=self._contract(request, observations),
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
            generated = await self._gateway.generate(turn_request)
            response = (
                generated if isinstance(generated, AgentResponse) else AgentResponse(generated)
            )
            model_calls.append(
                {
                    "turn": turn,
                    "request_id": turn_request.request_id,
                    "metadata": dict(response.metadata),
                }
            )
            entry: dict[str, object] = {
                "turn": continuation_turn_count + turn,
                "action_text": response.text,
            }
            try:
                action = _parse_structured_action(response.text)
            except (TypeError, ValueError) as exc:
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

            entry["structured_action"] = action.to_value()
            if action.kind is ActionKind.COMPLETE:
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
                entry["observation_status"] = "completed"
                trace.append(entry)
                return AgentResponse(
                    artifact,
                    {
                        "execution_mode": self._execution_mode,
                        "react_turns_used": continuation_turn_count + turn,
                        "new_react_turns_used": turn,
                        "continued_action_history_count": continuation_turn_count,
                        "continued_tool_receipt_count": len(
                            request.prior_tool_receipts
                        ),
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
                observation = MappingProxyType(
                    {
                        "observation_status": "schema_invalid",
                        "public_error_code": "duplicate_tool_request",
                        "executed_action": action.to_value(),
                    }
                )
                entry.update(observation)
                trace.append(entry)
                observations.append(observation)
                continue

            last_dispatched_tool_action_key = action_key
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
