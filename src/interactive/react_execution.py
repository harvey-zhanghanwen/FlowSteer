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
        if execution_mode not in {"react", "coding"}:
            raise ValueError("execution_mode must be react or coding")
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        self._execution_mode = execution_mode

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
            agent = replace(
                masked_request.agent,
                contract=self._contract(masked_request, observations),
            )
            turn_request = replace(
                masked_request,
                request_id=f"{request.request_id}:react:{turn}",
                agent=agent,
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
                if not isinstance(action.arguments, dict) or "value" not in action.arguments:
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
