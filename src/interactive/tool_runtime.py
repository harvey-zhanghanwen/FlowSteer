"""Typed tool registry for the unified AgentRuntime.

The request/result/backend/registration/registry contracts are a dependency-
light port of SkillFlow ``skillev.runtime.tools``.  ``ToolCapability`` and
``ToolReceipt`` are the minimal project extensions required for asynchronous
multi-Agent execution, dataset scoping, and replayable public trajectories.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import time
from types import MappingProxyType
from typing import Any, Optional, Protocol, Union, cast


JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]


class ActionKind(str, Enum):
    """Direct port of SkillFlow's executable structured-action kinds."""

    TOOL = "tool"
    SKILL = "skill"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class StructuredAction:
    """Direct port of SkillFlow's serializable atomic action contract."""

    kind: ActionKind
    name: str
    arguments: JsonValue
    resource_id: Optional[str] = None
    skill_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise TypeError("Structured action kind is incompatible")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Structured action name cannot be empty")
        normalized = _normalize_json(self.arguments)
        if normalized != self.arguments:
            raise ValueError("Structured action arguments must be normalized JSON")
        if self.kind in {ActionKind.TOOL, ActionKind.SKILL} and not self.resource_id:
            raise ValueError("Executable actions require a resource ID")
        if self.kind is ActionKind.SKILL and not self.skill_id:
            raise ValueError("Skill actions require a skill ID")
        if self.kind is not ActionKind.SKILL and self.skill_id is not None:
            raise ValueError("Only Skill actions may carry a skill ID")
        if self.kind is ActionKind.COMPLETE and self.resource_id is not None:
            raise ValueError("Completion is not dispatched to a resource")
        object.__setattr__(self, "name", self.name.strip())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "arguments": self.arguments,
            "kind": self.kind.value,
            "name": self.name,
            "resource_id": self.resource_id,
            "skill_id": self.skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> "StructuredAction":
        normalized = _normalize_json(value)
        if not isinstance(normalized, dict) or set(normalized) != {
            "arguments",
            "kind",
            "name",
            "resource_id",
            "skill_id",
        }:
            raise ValueError("Structured action has an incompatible field set")
        raw_kind = normalized["kind"]
        raw_name = normalized["name"]
        resource_id = normalized["resource_id"]
        skill_id = normalized["skill_id"]
        if not isinstance(raw_kind, str) or not isinstance(raw_name, str):
            raise TypeError("Structured action kind/name must be text")
        if resource_id is not None and not isinstance(resource_id, str):
            raise TypeError("Structured action resource_id must be text or null")
        if skill_id is not None and not isinstance(skill_id, str):
            raise TypeError("Structured action skill_id must be text or null")
        return cls(
            kind=ActionKind(raw_kind),
            name=raw_name,
            arguments=normalized["arguments"],
            resource_id=resource_id,
            skill_id=skill_id,
        )


def _normalize_json(value: object) -> JsonValue:
    """Local compatibility adapter for SkillFlow's canonical JSON helper."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cast(JsonValue, json.loads(encoded))
    except (TypeError, ValueError) as exc:
        raise TypeError("value must be normalized JSON") from exc


@dataclass(frozen=True, slots=True)
class ToolRequest:
    action: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("Tool action cannot be empty")
        normalized = _normalize_json(dict(self.arguments))
        if not isinstance(normalized, dict):
            raise TypeError("Tool arguments must form a JSON object")
        object.__setattr__(self, "action", self.action.strip())
        object.__setattr__(self, "arguments", MappingProxyType(normalized))

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action": self.action,
            "arguments": cast(JsonValue, dict(self.arguments)),
        }

    def to_structured_action(
        self,
        *,
        resource_id: str,
        skill_id: Optional[str] = None,
    ) -> StructuredAction:
        return StructuredAction(
            kind=ActionKind.SKILL if skill_id is not None else ActionKind.TOOL,
            name=self.action,
            arguments=cast(JsonValue, dict(self.arguments)),
            resource_id=resource_id,
            skill_id=skill_id,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    value: JsonValue
    completed: bool = True

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise TypeError("Tool completion status must be boolean")
        normalized = _normalize_json(self.value)
        if normalized != self.value:
            raise ValueError("Tool result must be normalized JSON")

    def to_value(self) -> dict[str, JsonValue]:
        return {"value": self.value, "completed": self.completed}


class ToolBackend(Protocol):
    def invoke(self, request: ToolRequest) -> Union[ToolResult, Awaitable[ToolResult]]:
        ...


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """NEW_PROJECT_EXTENSION: immutable Director-visible tool metadata."""

    tool_id: str
    dataset_scope: tuple[str, ...]
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    side_effect: str
    timeout_seconds: Optional[float]
    version: str
    availability: bool = True

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.tool_id, "tool_id"),
            (self.side_effect, "side_effect"),
            (self.version, "version"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if (
            not isinstance(self.dataset_scope, tuple)
            or not self.dataset_scope
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.dataset_scope
            )
            or len(set(self.dataset_scope)) != len(self.dataset_scope)
        ):
            raise ValueError("dataset_scope must contain unique non-empty strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when supplied")
        if type(self.availability) is not bool:
            raise TypeError("availability must be boolean")
        input_schema = _normalize_json(dict(self.input_schema))
        output_schema = _normalize_json(dict(self.output_schema))
        if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
            raise TypeError("tool schemas must be JSON objects")
        object.__setattr__(self, "tool_id", self.tool_id.strip())
        object.__setattr__(
            self,
            "dataset_scope",
            tuple(item.strip() for item in self.dataset_scope),
        )
        object.__setattr__(self, "side_effect", self.side_effect.strip())
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "input_schema", MappingProxyType(input_schema))
        object.__setattr__(self, "output_schema", MappingProxyType(output_schema))

    def supports_dataset(self, dataset_id: str) -> bool:
        return "*" in self.dataset_scope or dataset_id in self.dataset_scope

    def to_value(self) -> dict[str, object]:
        return {
            "tool_id": self.tool_id,
            "dataset_scope": list(self.dataset_scope),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "side_effect": self.side_effect,
            "timeout_seconds": self.timeout_seconds,
            "version": self.version,
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class ToolReceipt:
    """NEW_PROJECT_EXTENSION: one measured, public tool invocation receipt."""

    tool_id: str
    tool_version: str
    request: ToolRequest
    result: Optional[ToolResult]
    started_at_monotonic: float
    ended_at_monotonic: float
    latency_ms: float
    error_type: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.tool_id or not self.tool_version:
            raise ValueError("tool receipt identity fields cannot be empty")
        if self.ended_at_monotonic < self.started_at_monotonic:
            raise ValueError("tool receipt end time precedes start time")
        if self.latency_ms < 0:
            raise ValueError("tool receipt latency cannot be negative")
        if (self.result is None) == (self.error_type is None):
            raise ValueError("tool receipt requires exactly one result or error_type")

    def to_value(self) -> dict[str, object]:
        return {
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
            "request": self.request.to_value(),
            "result": None if self.result is None else self.result.to_value(),
            "started_at_monotonic": self.started_at_monotonic,
            "ended_at_monotonic": self.ended_at_monotonic,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    resource_id: str
    backend: ToolBackend
    capability: Optional[ToolCapability] = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("Tool registration requires a resource ID")
        object.__setattr__(self, "resource_id", self.resource_id.strip())
        if self.capability is not None and self.capability.tool_id != self.resource_id:
            raise ValueError("Tool capability ID must match registration resource ID")


class ToolRegistry:
    """SkillFlow-compatible immutable-at-execution resource registry."""

    def __init__(self, registrations: tuple[ToolRegistration, ...]) -> None:
        resources: dict[str, ToolBackend] = {}
        capabilities: dict[str, ToolCapability] = {}
        for registration in registrations:
            if registration.resource_id in resources:
                raise ValueError("Tool resource IDs must be unique")
            resources[registration.resource_id] = registration.backend
            if registration.capability is not None:
                capabilities[registration.resource_id] = registration.capability
        self._resources: Mapping[str, ToolBackend] = MappingProxyType(resources)
        self._capabilities: Mapping[str, ToolCapability] = MappingProxyType(
            capabilities
        )

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._resources))

    @property
    def capabilities(self) -> tuple[ToolCapability, ...]:
        return tuple(self._capabilities[key] for key in sorted(self._capabilities))

    def require_capability(self, resource_id: str) -> ToolCapability:
        try:
            return self._capabilities[resource_id]
        except KeyError as exc:
            raise KeyError("Tool resource has no registered capability") from exc

    def _backend(self, resource_id: str) -> ToolBackend:
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise KeyError("Unknown tool resource") from exc

    def invoke(self, resource_id: str, request: ToolRequest) -> ToolResult:
        result = self._backend(resource_id).invoke(request)
        if inspect.isawaitable(result):
            raise TypeError("async tool backend requires ToolRegistry.ainvoke")
        if not isinstance(result, ToolResult):
            raise TypeError("tool backend returned an incompatible result")
        return result

    async def ainvoke(self, resource_id: str, request: ToolRequest) -> ToolResult:
        capability = self.require_capability(resource_id)
        if not capability.availability:
            raise RuntimeError("Tool resource is unavailable")
        backend = self._backend(resource_id)

        async def execute() -> ToolResult:
            method = backend.invoke
            if inspect.iscoroutinefunction(method):
                result = await method(request)
            else:
                candidate = await asyncio.to_thread(method, request)
                result = await candidate if inspect.isawaitable(candidate) else candidate
            if not isinstance(result, ToolResult):
                raise TypeError("tool backend returned an incompatible result")
            return result

        if capability.timeout_seconds is None:
            return await execute()
        return await asyncio.wait_for(execute(), timeout=capability.timeout_seconds)

    async def ainvoke_with_receipt(
        self,
        resource_id: str,
        request: ToolRequest,
    ) -> tuple[Optional[ToolResult], ToolReceipt]:
        capability = self.require_capability(resource_id)
        started = time.monotonic()
        try:
            result = await self.ainvoke(resource_id, request)
        except Exception as exc:
            ended = time.monotonic()
            receipt = ToolReceipt(
                tool_id=resource_id,
                tool_version=capability.version,
                request=request,
                result=None,
                started_at_monotonic=started,
                ended_at_monotonic=ended,
                latency_ms=max((ended - started) * 1000.0, 0.0),
                error_type=type(exc).__name__,
            )
            return None, receipt
        ended = time.monotonic()
        receipt = ToolReceipt(
            tool_id=resource_id,
            tool_version=capability.version,
            request=request,
            result=result,
            started_at_monotonic=started,
            ended_at_monotonic=ended,
            latency_ms=max((ended - started) * 1000.0, 0.0),
        )
        return result, receipt


@dataclass(slots=True)
class FakeTool:
    """SkillFlow-compatible deterministic backend for CPU behavior tests."""

    handlers: Mapping[str, Callable[[Mapping[str, object]], object]]
    calls: list[ToolRequest] = field(default_factory=list)

    def invoke(self, request: ToolRequest) -> ToolResult:
        try:
            handler = self.handlers[request.action]
        except KeyError as exc:
            raise KeyError("Fake tool has no handler for the requested action") from exc
        value = _normalize_json(handler(request.arguments))
        self.calls.append(request)
        return ToolResult(value)


__all__ = [
    "ActionKind",
    "FakeTool",
    "JsonValue",
    "StructuredAction",
    "ToolBackend",
    "ToolCapability",
    "ToolReceipt",
    "ToolRegistration",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
]
