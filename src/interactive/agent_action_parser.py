"""Strict parser for the first JSON AgentGraph action in a Director response."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


class AgentActionType(str, Enum):
    ADD_AGENT = "add_agent"
    MODIFY_AGENT = "modify_agent"
    DELETE_AGENT = "delete_agent"
    SET_RELATION = "set_relation"
    SET_OUTPUT = "set_output"
    FINISH = "finish"


class AgentActionParseError(ValueError):
    """Raised when the first JSON object is not one strict atomic action."""


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

    @property
    def prompt(self) -> Optional[str]:
        return self.contract

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {"action": self.action_type.value}
        for key in (
            "agent_id",
            "model_id",
            "contract",
            "role_family",
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
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


def _strict_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise AgentActionParseError(f"{key} must be a JSON boolean")
    return value


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
        if action_type is AgentActionType.ADD_AGENT:
            allowed_contracts = {"contract", "prompt"} & set(data)
            if len(allowed_contracts) != 1:
                raise AgentActionParseError("add_agent requires exactly one of contract or prompt")
            contract_key = next(iter(allowed_contracts))
            _check_keys(
                data,
                {"action", "agent_id", "model_id", contract_key},
                {"role_family"},
            )
            return AgentAction(
                agent_id=_required_string(data, "agent_id"),
                model_id=_required_string(data, "model_id"),
                contract=_required_string(data, contract_key),
                role_family=_optional_string(data, "role_family"),
                **common,
            )

        if action_type is AgentActionType.MODIFY_AGENT:
            _check_keys(
                data,
                {"action", "agent_id"},
                {"model_id", "contract", "prompt", "role_family"},
            )
            if "contract" in data and "prompt" in data:
                raise AgentActionParseError("modify_agent accepts contract or prompt, not both")
            model_id = _optional_string(data, "model_id")
            contract_key = "contract" if "contract" in data else "prompt"
            contract = _optional_string(data, contract_key) if contract_key in data else None
            role_family = _optional_string(data, "role_family")
            if model_id is None and contract is None and role_family is None:
                raise AgentActionParseError(
                    "modify_agent requires model_id, contract, prompt, or role_family"
                )
            return AgentAction(
                agent_id=_required_string(data, "agent_id"),
                model_id=model_id,
                contract=contract,
                role_family=role_family,
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
    "parse_agent_action",
    "parse_first_agent_action",
]
