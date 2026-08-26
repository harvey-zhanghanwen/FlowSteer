"""Strict parser for the first JSON AgentGraph action in a Director response."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


AGENT_EXECUTION_MODES = frozenset({"reasoning", "react", "coding"})


class AgentActionType(str, Enum):
    ADD_SUBGRAPH = "add_subgraph"
    ADD_AGENT = "add_agent"
    MODIFY_AGENT = "modify_agent"
    DELETE_AGENT = "delete_agent"
    SET_RELATION = "set_relation"
    SET_OUTPUT = "set_output"
    FINISH = "finish"


class AgentActionParseError(ValueError):
    """Raised when the first JSON object is not one strict Canvas action."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    model_id: str
    contract: str
    role_family: Optional[str] = None
    allowed_tools: Optional[Tuple[str, ...]] = None
    execution_mode: Optional[str] = None
    artifact_type: Optional[str] = None
    completion_condition: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "contract": self.contract,
        }
        if self.role_family is not None:
            result["role_family"] = self.role_family
        if self.allowed_tools is not None:
            result["allowed_tools"] = list(self.allowed_tools)
        if self.execution_mode is not None:
            result["execution_mode"] = self.execution_mode
        if self.artifact_type is not None:
            result["artifact_type"] = self.artifact_type
        if self.completion_condition is not None:
            result["completion_condition"] = self.completion_condition
        return result


@dataclass(frozen=True, slots=True)
class RelationSpec:
    source_id: str
    target_id: str
    source_to_target: bool
    target_to_source: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_to_target": self.source_to_target,
            "target_to_source": self.target_to_source,
        }


@dataclass(frozen=True, slots=True)
class AgentAction:
    action_type: AgentActionType
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    contract: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    source_to_target: Optional[bool] = None
    target_to_source: Optional[bool] = None
    raw_json: str = ""
    consumed_start: int = 0
    consumed_end: int = 0
    # Optional free-text analysis metadata; it does not select an Operator.
    role_family: Optional[str] = None
    agents: Tuple[AgentSpec, ...] = ()
    relations: Tuple[RelationSpec, ...] = ()
    output_agent_id: Optional[str] = None
    allowed_tools: Optional[Tuple[str, ...]] = None
    execution_mode: Optional[str] = None
    artifact_type: Optional[str] = None
    completion_condition: Optional[str] = None

    @property
    def prompt(self) -> Optional[str]:
        return self.contract

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {"action": self.action_type.value}
        if self.action_type is AgentActionType.ADD_SUBGRAPH:
            result["agents"] = [item.to_dict() for item in self.agents]
            result["relations"] = [item.to_dict() for item in self.relations]
            if self.output_agent_id is not None:
                result["output_agent_id"] = self.output_agent_id
            return result
        for key in (
            "agent_id",
            "model_id",
            "contract",
            "role_family",
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
            "execution_mode",
            "artifact_type",
            "completion_condition",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.allowed_tools is not None:
            result["allowed_tools"] = list(self.allowed_tools)
        return result


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentActionParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AgentActionParseError(f"non-finite JSON number is not allowed: {value}")


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentActionParseError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(data: Mapping[str, Any], key: str) -> Optional[str]:
    if key not in data:
        return None
    return _required_string(data, key)


def _optional_string_array(
    data: Mapping[str, Any], key: str
) -> Optional[Tuple[str, ...]]:
    if key not in data:
        return None
    value = data.get(key)
    if not isinstance(value, list):
        raise AgentActionParseError(f"{key} must be a JSON array")
    normalized: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AgentActionParseError(
                f"{key} must contain only non-empty strings"
            )
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized):
        raise AgentActionParseError(f"{key} values must be unique")
    return tuple(normalized)


def _optional_execution_mode(
    data: Mapping[str, Any], key: str = "execution_mode"
) -> Optional[str]:
    value = _optional_string(data, key)
    if value is not None and value not in AGENT_EXECUTION_MODES:
        raise AgentActionParseError(
            f"{key} must be one of: {', '.join(sorted(AGENT_EXECUTION_MODES))}"
        )
    return value


def _strict_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise AgentActionParseError(f"{key} must be a JSON boolean")
    return value


def _agent_spec(data: Any) -> AgentSpec:
    if not isinstance(data, Mapping):
        raise AgentActionParseError("each add_subgraph agent must be an object")
    allowed_contracts = {"contract", "prompt"} & set(data)
    if len(allowed_contracts) != 1:
        raise AgentActionParseError(
            "each add_subgraph agent requires exactly one of contract or prompt"
        )
    contract_key = next(iter(allowed_contracts))
    _check_keys(
        data,
        {"agent_id", "model_id", contract_key},
        {
            "role_family",
            "allowed_tools",
            "execution_mode",
            "artifact_type",
            "completion_condition",
        },
    )
    return AgentSpec(
        agent_id=_required_string(data, "agent_id"),
        model_id=_required_string(data, "model_id"),
        contract=_required_string(data, contract_key),
        role_family=_optional_string(data, "role_family"),
        allowed_tools=_optional_string_array(data, "allowed_tools"),
        execution_mode=_optional_execution_mode(data),
        artifact_type=_optional_string(data, "artifact_type"),
        completion_condition=_optional_string(data, "completion_condition"),
    )


def _relation_spec(data: Any) -> RelationSpec:
    if not isinstance(data, Mapping):
        raise AgentActionParseError("each add_subgraph relation must be an object")
    _check_keys(
        data,
        {"source_id", "target_id", "source_to_target", "target_to_source"},
    )
    source_id = _required_string(data, "source_id")
    target_id = _required_string(data, "target_id")
    if source_id == target_id:
        raise AgentActionParseError("add_subgraph relation endpoints must be different")
    relation = RelationSpec(
        source_id=source_id,
        target_id=target_id,
        source_to_target=_strict_bool(data, "source_to_target"),
        target_to_source=_strict_bool(data, "target_to_source"),
    )
    if not relation.source_to_target and not relation.target_to_source:
        raise AgentActionParseError(
            "add_subgraph relation must contain at least one directed edge"
        )
    return relation


def _check_keys(data: Mapping[str, Any], required: Set[str], optional: Set[str] = set()) -> None:
    keys = set(data)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise AgentActionParseError(f"missing action fields: {', '.join(missing)}")
    if unknown:
        raise AgentActionParseError(f"unknown action fields: {', '.join(unknown)}")


class AgentActionParser:
    """Parse exactly the earliest JSON object; never salvage a later object."""

    def __init__(self) -> None:
        self._decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )

    def parse(self, text: str) -> AgentAction:
        if not isinstance(text, str) or not text:
            raise AgentActionParseError("response must be a non-empty string")
        stripped_start = len(text) - len(text.lstrip())
        start = stripped_start
        try:
            leading_value, leading_end = self._decoder.raw_decode(text[start:])
        except (AgentActionParseError, json.JSONDecodeError, TypeError, ValueError):
            object_start = text.find("{", stripped_start)
            array_start = text.find("[", stripped_start)
            candidates = [position for position in (object_start, array_start) if position >= 0]
            if not candidates:
                raise AgentActionParseError("response does not contain a JSON object")
            start = min(candidates)
        else:
            if not isinstance(leading_value, dict):
                raise AgentActionParseError("the first JSON value must be an object")
            end = start + leading_end
            raw_json = text[start:end]
            return self._build_action(leading_value, raw_json=raw_json, start=start, end=end)
        try:
            value, relative_end = self._decoder.raw_decode(text[start:])
        except AgentActionParseError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AgentActionParseError(
                f"the first JSON object is malformed at character {start}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise AgentActionParseError("the first JSON value must be an object")
        end = start + relative_end
        raw_json = text[start:end]
        return self._build_action(value, raw_json=raw_json, start=start, end=end)

    def _build_action(
        self,
        data: Mapping[str, Any],
        *,
        raw_json: str,
        start: int,
        end: int,
    ) -> AgentAction:
        action_name = _required_string(data, "action")
        try:
            action_type = AgentActionType(action_name)
        except ValueError as exc:
            allowed = ", ".join(action.value for action in AgentActionType)
            raise AgentActionParseError(
                f"unsupported action {action_name!r}; expected one of: {allowed}"
            ) from exc

        common = {
            "action_type": action_type,
            "raw_json": raw_json,
            "consumed_start": start,
            "consumed_end": end,
        }
        if action_type is AgentActionType.ADD_SUBGRAPH:
            _check_keys(
                data,
                {"action", "agents", "relations"},
                {"output_agent_id"},
            )
            raw_agents = data.get("agents")
            raw_relations = data.get("relations")
            if not isinstance(raw_agents, list) or not raw_agents:
                raise AgentActionParseError("add_subgraph agents must be a non-empty array")
            if not isinstance(raw_relations, list):
                raise AgentActionParseError("add_subgraph relations must be an array")
            agents = tuple(_agent_spec(item) for item in raw_agents)
            if len(agents) > 3:
                raise AgentActionParseError(
                    "add_subgraph supports at most three Agents"
                )
            if len({item.agent_id for item in agents}) != len(agents):
                raise AgentActionParseError("add_subgraph agent_id values must be unique")
            relations = tuple(_relation_spec(item) for item in raw_relations)
            relation_pairs = {
                tuple(sorted((item.source_id, item.target_id))) for item in relations
            }
            if len(relation_pairs) != len(relations):
                raise AgentActionParseError(
                    "add_subgraph may contain at most one relation per endpoint pair"
                )
            # Qwen may serialize an omitted subgraph Output as JSON null.  This
            # normalization is local to ADD_SUBGRAPH; other optional strings
            # retain the strict non-empty-string contract.
            output_agent_id = None
            if data.get("output_agent_id") is not None:
                output_agent_id = _required_string(data, "output_agent_id")
            return AgentAction(
                agents=agents,
                relations=relations,
                output_agent_id=output_agent_id,
                **common,
            )

        if action_type is AgentActionType.ADD_AGENT:
            allowed_contracts = {"contract", "prompt"} & set(data)
            if len(allowed_contracts) != 1:
                raise AgentActionParseError("add_agent requires exactly one of contract or prompt")
            contract_key = next(iter(allowed_contracts))
            _check_keys(
                data,
                {"action", "agent_id", "model_id", contract_key},
                {
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                    "artifact_type",
                    "completion_condition",
                },
            )
            return AgentAction(
                agent_id=_required_string(data, "agent_id"),
                model_id=_required_string(data, "model_id"),
                contract=_required_string(data, contract_key),
                role_family=_optional_string(data, "role_family"),
                allowed_tools=_optional_string_array(data, "allowed_tools"),
                execution_mode=_optional_execution_mode(data),
                artifact_type=_optional_string(data, "artifact_type"),
                completion_condition=_optional_string(
                    data, "completion_condition"
                ),
                **common,
            )

        if action_type is AgentActionType.MODIFY_AGENT:
            _check_keys(
                data,
                {"action", "agent_id"},
                {
                    "model_id",
                    "contract",
                    "prompt",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                    "artifact_type",
                    "completion_condition",
                },
            )
            if "contract" in data and "prompt" in data:
                raise AgentActionParseError("modify_agent accepts contract or prompt, not both")
            model_id = _optional_string(data, "model_id")
            contract_key = "contract" if "contract" in data else "prompt"
            contract = _optional_string(data, contract_key) if contract_key in data else None
            role_family = _optional_string(data, "role_family")
            allowed_tools = _optional_string_array(data, "allowed_tools")
            execution_mode = _optional_execution_mode(data)
            artifact_type = _optional_string(data, "artifact_type")
            completion_condition = _optional_string(data, "completion_condition")
            if all(
                value is None
                for value in (
                    model_id,
                    contract,
                    role_family,
                    allowed_tools,
                    execution_mode,
                    artifact_type,
                    completion_condition,
                )
            ):
                raise AgentActionParseError(
                    "modify_agent requires at least one mutable Agent field"
                )
            return AgentAction(
                agent_id=_required_string(data, "agent_id"),
                model_id=model_id,
                contract=contract,
                role_family=role_family,
                allowed_tools=allowed_tools,
                execution_mode=execution_mode,
                artifact_type=artifact_type,
                completion_condition=completion_condition,
                **common,
            )

        if action_type is AgentActionType.DELETE_AGENT:
            _check_keys(data, {"action", "agent_id"})
            return AgentAction(agent_id=_required_string(data, "agent_id"), **common)

        if action_type is AgentActionType.SET_RELATION:
            _check_keys(
                data,
                {
                    "action",
                    "source_id",
                    "target_id",
                    "source_to_target",
                    "target_to_source",
                },
            )
            source_id = _required_string(data, "source_id")
            target_id = _required_string(data, "target_id")
            if source_id == target_id:
                raise AgentActionParseError("set_relation endpoints must be different")
            return AgentAction(
                source_id=source_id,
                target_id=target_id,
                source_to_target=_strict_bool(data, "source_to_target"),
                target_to_source=_strict_bool(data, "target_to_source"),
                **common,
            )

        if action_type is AgentActionType.SET_OUTPUT:
            _check_keys(data, {"action", "agent_id"})
            return AgentAction(agent_id=_required_string(data, "agent_id"), **common)

        _check_keys(data, {"action"})
        return AgentAction(**common)


def parse_first_agent_action(text: str) -> AgentAction:
    return AgentActionParser().parse(text)


parse_agent_action = parse_first_agent_action


__all__ = [
    "AgentAction",
    "AgentActionParseError",
    "AgentActionParser",
    "AgentActionType",
    "AGENT_EXECUTION_MODES",
    "AgentSpec",
    "RelationSpec",
    "parse_agent_action",
    "parse_first_agent_action",
]
