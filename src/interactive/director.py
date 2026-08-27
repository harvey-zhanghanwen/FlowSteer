"""Inference-time Qwen Flow-Director loop over the strict AgentGraph Canvas."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStepResult
from .model_registry import ModelRegistry
from .tool_runtime import ToolRegistry


DIRECTOR_SYSTEM_PROMPT = """You are the Flow-Director. Incrementally edit and execute an AgentGraph. Return exactly one JSON Canvas action each turn.

Use only the latest admissible_actions and action_target_domains. Agent declarations use agent_id, model_id, contract, optional role_family, execution_mode, allowed_tools, artifact_type, and completion_condition. add_subgraph may add one to three Agents plus directed or bidirectional relations and an optional output_agent_id. Other actions are add_agent, modify_agent, delete_agent, set_relation, set_output, and finish.

One accepted Canvas edit is one execution boundary. ReAct is an Agent execution mode, not a role. Preserve useful artifacts, diagnose failures, repair or augment before deletion, and do not assume a fixed role sequence or topology."""


# DIRECT_REUSE: these Canvas action schemas and the two-stage SGLang
# discriminator are the model-admissible constrained-decoding boundary used by
# the current FlowSteer implementation.  They mirror AgentActionParser; the
# parser remains authoritative after generation.
_NON_EMPTY_STRING_SCHEMA = {"type": "string", "minLength": 1}
_AGENT_SPEC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agent_id", "model_id", "contract"],
    "properties": {
        "agent_id": _NON_EMPTY_STRING_SCHEMA,
        "model_id": _NON_EMPTY_STRING_SCHEMA,
        "contract": _NON_EMPTY_STRING_SCHEMA,
        "role_family": _NON_EMPTY_STRING_SCHEMA,
        "allowed_tools": {
            "type": "array",
            "items": _NON_EMPTY_STRING_SCHEMA,
            "uniqueItems": True,
        },
        "execution_mode": {"enum": ["reasoning", "react", "coding"]},
        "artifact_type": _NON_EMPTY_STRING_SCHEMA,
        "completion_condition": _NON_EMPTY_STRING_SCHEMA,
    },
}
_RELATION_SPEC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_id",
        "target_id",
        "source_to_target",
        "target_to_source",
    ],
    "properties": {
        "source_id": _NON_EMPTY_STRING_SCHEMA,
        "target_id": _NON_EMPTY_STRING_SCHEMA,
        "source_to_target": {"type": "boolean"},
        "target_to_source": {"type": "boolean"},
    },
    "anyOf": [
        {"properties": {"source_to_target": {"const": True}}},
        {"properties": {"target_to_source": {"const": True}}},
    ],
}
_MUTABLE_AGENT_PROPERTIES = {
    "model_id": _NON_EMPTY_STRING_SCHEMA,
    "contract": _NON_EMPTY_STRING_SCHEMA,
    "role_family": _NON_EMPTY_STRING_SCHEMA,
    "allowed_tools": {
        "type": "array",
        "items": _NON_EMPTY_STRING_SCHEMA,
        "uniqueItems": True,
    },
    "execution_mode": {"enum": ["reasoning", "react", "coding"]},
    "artifact_type": _NON_EMPTY_STRING_SCHEMA,
    "completion_condition": _NON_EMPTY_STRING_SCHEMA,
}
DIRECTOR_ACTION_JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "additionalProperties": False,
            "required": ["action", "agents", "relations"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": _AGENT_SPEC_JSON_SCHEMA,
                },
                "relations": {
                    "type": "array",
                    "items": _RELATION_SPEC_JSON_SCHEMA,
                },
                "output_agent_id": {
                    "anyOf": [_NON_EMPTY_STRING_SCHEMA, {"type": "null"}]
                },
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id", "model_id", "contract"],
            "properties": {
                "action": {"const": "add_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
                "model_id": _NON_EMPTY_STRING_SCHEMA,
                "contract": _NON_EMPTY_STRING_SCHEMA,
                "role_family": _NON_EMPTY_STRING_SCHEMA,
                "allowed_tools": _MUTABLE_AGENT_PROPERTIES["allowed_tools"],
                "execution_mode": _MUTABLE_AGENT_PROPERTIES["execution_mode"],
                "artifact_type": _NON_EMPTY_STRING_SCHEMA,
                "completion_condition": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "modify_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
                **_MUTABLE_AGENT_PROPERTIES,
            },
            "anyOf": [
                {"required": [field_name]}
                for field_name in _MUTABLE_AGENT_PROPERTIES
            ],
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "delete_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": [
                "action",
                "source_id",
                "target_id",
                "source_to_target",
                "target_to_source",
            ],
            "properties": {
                "action": {"const": "set_relation"},
                **_RELATION_SPEC_JSON_SCHEMA["properties"],
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "set_output"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"const": "finish"}},
        },
    ],
}
DIRECTOR_ACTION_SCHEMA_VERSION = "agentgraph.canvas-action-json-schema.v1"
DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE = (
    "model_admissible_canvas_actions"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION = (
    "agentgraph.model-admissible-action-mask.v2"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3 = (
    "agentgraph.model-admissible-action-mask.v3"
)
DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION = (
    "agentgraph.live-action-target-domains.v1"
)


def director_action_json_schema_text(actions: Sequence[str]) -> str:
    """Render the strict parser schema for one configured Canvas profile."""

    if isinstance(actions, (str, bytes)) or not actions:
        raise ValueError("Canvas actions must be a non-empty sequence")
    by_name = {
        branch["properties"]["action"]["const"]: branch
        for branch in DIRECTOR_ACTION_JSON_SCHEMA["oneOf"]
    }
    normalized = tuple(actions)
    if (
        any(
            not isinstance(action, str) or action not in by_name
            for action in normalized
        )
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("Canvas actions contain an unknown or duplicate action")
    return json.dumps(
        {
            "type": "object",
            "oneOf": [by_name[action] for action in normalized],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_state_conditioned_sampling_json_schema_text(action: str) -> str:
    """Render one exact action branch for SGLang constrained decoding."""

    by_name = {
        branch["properties"]["action"]["const"]: branch
        for branch in DIRECTOR_ACTION_JSON_SCHEMA["oneOf"]
    }
    if action not in by_name:
        raise ValueError("state-conditioned sampling received an unknown action")
    branch = json.loads(json.dumps(by_name[action]))
    if action == "add_subgraph":
        relation_schema = branch["properties"]["relations"]["items"]
        relation_properties = relation_schema["properties"]
        relation_required = relation_schema["required"]
        branch["properties"]["relations"]["items"] = {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": relation_required,
                    "properties": {
                        **relation_properties,
                        "source_to_target": {"const": True},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": relation_required,
                    "properties": {
                        **relation_properties,
                        "target_to_source": {"const": True},
                    },
                },
            ]
        }
    return json.dumps(
        branch,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_model_admissible_sampling_json_schema_text(
    actions: Sequence[str],
) -> str:
    """Render FlowSteer's v2 action-discriminator schema for SGLang."""

    normalized = tuple(actions)
    director_action_json_schema_text(normalized)
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"enum": list(normalized)}},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_model_admissible_schema_branch(actions: Sequence[str]) -> str:
    """Bind one canonical v2 schema request to its exact action domain."""

    normalized = tuple(actions)
    director_model_admissible_sampling_json_schema_text(normalized)
    return "admissible-v2:" + ",".join(normalized)


def director_actions_from_admissible_schema_branch(
    branch: str,
) -> Tuple[str, ...]:
    """Recover the exact action domain from a canonical v2 branch receipt."""

    prefixes = ("admissible-v2:", "admissible-v3:")
    prefix = next(
        (
            candidate
            for candidate in prefixes
            if isinstance(branch, str) and branch.startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        raise ValueError("model-admissible schema branch has an invalid prefix")
    actions = tuple(branch[len(prefix) :].split(","))
    director_model_admissible_sampling_json_schema_text(actions)
    return actions


def director_model_admissible_schema_branch_v3(
    actions: Sequence[str],
) -> str:
    """Bind a v3 discriminator to its exact live Canvas action domain."""

    normalized = tuple(actions)
    director_model_admissible_sampling_json_schema_text(normalized)
    return "admissible-v3:" + ",".join(normalized)


def director_live_action_target_domains_json(
    actions: Sequence[str],
    action_target_domains: Mapping[str, Any],
) -> str:
    """Canonicalize one request's public live action target domains."""

    normalized_actions = tuple(actions)
    director_model_admissible_sampling_json_schema_text(normalized_actions)
    if not isinstance(action_target_domains, Mapping):
        raise ValueError("live action target domains must be an object")
    for action in normalized_actions:
        if not isinstance(action_target_domains.get(action), Mapping):
            raise ValueError(f"live target domain for {action} must be an object")
    try:
        normalized = json.loads(
            json.dumps(
                dict(action_target_domains),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("live action target domains must be JSON-serializable") from exc
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _live_execution_domain(
    action_domain: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    model_ids = [
        str(value)
        for value in action_domain.get("model_ids", ())
        if isinstance(value, str) and value
    ]
    profiles = action_domain.get("execution_profiles", ())
    execution_modes: list[str] = []
    tool_ids: list[str] = []
    if isinstance(profiles, (list, tuple)):
        for profile in profiles:
            if not isinstance(profile, Mapping):
                continue
            mode = profile.get("execution_mode")
            if isinstance(mode, str) and mode and mode not in execution_modes:
                execution_modes.append(mode)
            tools = profile.get("allowed_tools", ())
            if isinstance(tools, (list, tuple)):
                for tool_id in tools:
                    if (
                        isinstance(tool_id, str)
                        and tool_id
                        and tool_id not in tool_ids
                    ):
                        tool_ids.append(tool_id)
    if not model_ids or not execution_modes:
        raise ValueError("live Agent declaration domain is incomplete")
    return model_ids, execution_modes, tool_ids


def _live_execution_profiles(
    action_domain: Mapping[str, Any],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the exact Runtime execution-profile domain.

    This follows FlowSteer's live execution-profile validation: execution mode
    and Tool capability are one admissible pair, not independent marginals.
    """

    raw_profiles = action_domain.get("execution_profiles", ())
    if not isinstance(raw_profiles, (list, tuple)) or not raw_profiles:
        raise ValueError("live Agent declaration has no execution profiles")
    profiles: list[tuple[str, tuple[str, ...]]] = []
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("live execution profile is malformed")
        mode = raw_profile.get("execution_mode")
        tools = raw_profile.get("allowed_tools")
        if (
            mode not in {"reasoning", "react", "coding"}
            or not isinstance(tools, (list, tuple))
            or any(not isinstance(tool, str) or not tool for tool in tools)
        ):
            raise ValueError("live execution profile is invalid")
        profile = (str(mode), tuple(str(tool) for tool in tools))
        if profile in profiles:
            raise ValueError("live execution profiles contain a duplicate")
        profiles.append(profile)
    return tuple(profiles)


def _live_execution_profile_for_mode(
    action_domain: Mapping[str, Any],
    execution_mode: str,
) -> tuple[str, tuple[str, ...]]:
    matches = [
        profile
        for profile in _live_execution_profiles(action_domain)
        if profile[0] == execution_mode
    ]
    if len(matches) != 1:
        raise ValueError(
            "live execution mode must identify exactly one Runtime profile"
        )
    return matches[0]


def _live_agent_schema(
    action_domain: Mapping[str, Any],
    *,
    declaration_phase: bool = False,
    execution_profile: Optional[tuple[str, tuple[str, ...]]] = None,
) -> Mapping[str, Any]:
    model_ids, execution_modes, tool_ids = _live_execution_domain(action_domain)
    required = ["agent_id", "model_id", "contract"]
    properties: dict[str, Any] = {
        "agent_id": _NON_EMPTY_STRING_SCHEMA,
        "model_id": {"enum": model_ids},
        "contract": _NON_EMPTY_STRING_SCHEMA,
        "role_family": _NON_EMPTY_STRING_SCHEMA,
        "artifact_type": _NON_EMPTY_STRING_SCHEMA,
        "completion_condition": _NON_EMPTY_STRING_SCHEMA,
    }
    if declaration_phase:
        required.append("execution_mode")
        properties["execution_mode"] = {"enum": execution_modes}
    elif execution_profile is not None:
        mode, tools = execution_profile
        if (mode, tuple(tools)) not in _live_execution_profiles(action_domain):
            raise ValueError("selected execution profile is outside the live domain")
        required.extend(["execution_mode", "allowed_tools"])
        properties["execution_mode"] = {"const": mode}
        properties["allowed_tools"] = {"const": list(tools)}
    else:
        properties["execution_mode"] = {"enum": execution_modes}
        properties["allowed_tools"] = {
            "type": "array",
            "items": {"enum": tool_ids},
            "uniqueItems": True,
            "maxItems": len(tool_ids),
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def director_live_add_subgraph_agent_declarations_json_schema_text(
    action_target_domains: Mapping[str, Any],
) -> str:
    """Render a topology-neutral ADD declaration phase under live domains."""

    domain = action_target_domains.get("add_subgraph")
    if not isinstance(domain, Mapping):
        raise ValueError("add_subgraph has no live target domain")
    max_new_agents = domain.get("max_new_agents", 3)
    if type(max_new_agents) is not int or not 1 <= max_new_agents <= 3:
        raise ValueError("add_subgraph live Agent limit is invalid")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "agents"],
        "properties": {
            "action": {"const": "add_subgraph"},
            "agents": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_new_agents,
                "items": _live_agent_schema(domain, declaration_phase=True),
            },
        },
    }
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def director_live_add_subgraph_agent_declarations_from_text(
    text: str,
    action_target_domains: Optional[Mapping[str, Any]] = None,
) -> Tuple[Mapping[str, Any], ...]:
    """Parse an exact constrained ADD declaration without rewriting it."""

    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (TypeError, ValueError) as exc:
        raise ValueError("ADD declaration phase is not JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"action", "agents"}:
        raise ValueError("ADD declaration phase has incompatible fields")
    if value.get("action") != "add_subgraph":
        raise ValueError("ADD declaration phase changed its action")
    agents = value.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("ADD declaration phase has no Agents")
    normalized = tuple(dict(agent) for agent in agents if isinstance(agent, Mapping))
    if len(normalized) != len(agents):
        raise ValueError("ADD declaration phase contains a non-object Agent")
    if action_target_domains is not None:
        domain = action_target_domains.get("add_subgraph")
        if not isinstance(domain, Mapping):
            raise ValueError("add_subgraph has no live target domain")
        bound: list[Mapping[str, Any]] = []
        for declaration in normalized:
            mode = declaration.get("execution_mode")
            if not isinstance(mode, str):
                raise ValueError("ADD declaration has no execution mode")
            _, tools = _live_execution_profile_for_mode(domain, mode)
            value = dict(declaration)
            existing_tools = value.get("allowed_tools")
            if existing_tools is not None and existing_tools != list(tools):
                raise ValueError("ADD declaration changed its Runtime profile")
            value["allowed_tools"] = list(tools)
            bound.append(value)
        normalized = tuple(bound)
    return normalized


def director_live_execution_mode_selector_json_schema_text(
    action: str,
    action_target_domains: Mapping[str, Any],
) -> str:
    """Select one live Runtime profile by its unique execution mode."""

    domain = action_target_domains.get(action)
    if not isinstance(domain, Mapping):
        raise ValueError(f"{action} has no live target domain")
    profiles = _live_execution_profiles(domain)
    modes = [mode for mode, _ in profiles]
    if len(modes) != len(set(modes)):
        raise ValueError(
            "live Runtime profiles require unique execution modes for sampling"
        )
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "execution_mode"],
            "properties": {
                "action": {"const": action},
                "execution_mode": {"enum": modes},
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_execution_profile_from_text(
    text: str,
    action: str,
    action_target_domains: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Bind a constrained execution-mode selection to its Tool capability."""

    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (TypeError, ValueError) as exc:
        raise ValueError("execution-profile phase is not JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"action", "execution_mode"}
        or value.get("action") != action
        or not isinstance(value.get("execution_mode"), str)
    ):
        raise ValueError("execution-profile phase has incompatible fields")
    domain = action_target_domains.get(action)
    if not isinstance(domain, Mapping):
        raise ValueError(f"{action} has no live target domain")
    return _live_execution_profile_for_mode(
        domain,
        str(value["execution_mode"]),
    )


def director_live_modify_agent_field_selector_json_schema_text() -> str:
    """Render the existing FlowSteer MODIFY-field discriminator."""

    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "field"],
            "properties": {
                "action": {"const": "modify_agent"},
                "field": {"enum": list(_MUTABLE_AGENT_PROPERTIES)},
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_action_parameter_json_schema_text(
    action: str,
    action_target_domains: Mapping[str, Any],
    *,
    add_agents: Optional[Sequence[Mapping[str, Any]]] = None,
    modify_field: Optional[str] = None,
    execution_profile: Optional[tuple[str, tuple[str, ...]]] = None,
) -> str:
    """Render SGLang-compatible parameters from the current live domain."""

    domain = action_target_domains.get(action)
    if not isinstance(domain, Mapping):
        raise ValueError(f"{action} has no live target domain")
    if action == "add_subgraph":
        if not add_agents:
            raise ValueError("add_subgraph parameters require committed Agents")
        agents = [dict(agent) for agent in add_agents]
        existing = [
            str(value)
            for value in domain.get("existing_agent_ids", ())
            if isinstance(value, str) and value
        ]
        agent_ids = [str(agent["agent_id"]) for agent in agents]
        endpoint_ids = list(dict.fromkeys([*existing, *agent_ids]))
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents", "relations"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {"const": agents},
                "relations": {
                    "type": "array",
                    "maxItems": len(endpoint_ids) * max(len(endpoint_ids) - 1, 0),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "source_id",
                            "target_id",
                            "source_to_target",
                            "target_to_source",
                        ],
                        "properties": {
                            "source_id": {"enum": endpoint_ids},
                            "target_id": {"enum": endpoint_ids},
                            "source_to_target": {"type": "boolean"},
                            "target_to_source": {"type": "boolean"},
                        },
                    },
                },
                "output_agent_id": {"enum": [*endpoint_ids, None]},
            },
        }
    elif action == "add_agent":
        if execution_profile is None:
            raise ValueError("add_agent parameters require an execution profile")
        agent_schema = dict(
            _live_agent_schema(domain, execution_profile=execution_profile)
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", *agent_schema["required"]],
            "properties": {
                "action": {"const": "add_agent"},
                **agent_schema["properties"],
            },
        }
    elif action == "modify_agent":
        if modify_field not in _MUTABLE_AGENT_PROPERTIES:
            raise ValueError("modify_agent parameters require one mutable field")
        agent_ids = [
            str(value)
            for value in domain.get("agent_ids", ())
            if isinstance(value, str) and value
        ]
        if not agent_ids:
            raise ValueError("modify_agent has no live Agent target")
        if modify_field == "model_id":
            values = [
                str(value)
                for value in domain.get("model_ids", ())
                if isinstance(value, str) and value
            ]
            field_schema: Mapping[str, Any] = {"enum": values}
        elif modify_field in {"execution_mode", "allowed_tools"}:
            _, execution_modes, tool_ids = _live_execution_domain(domain)
            field_schema = (
                {"enum": execution_modes}
                if modify_field == "execution_mode"
                else {
                    "type": "array",
                    "items": {"enum": tool_ids},
                    "uniqueItems": True,
                    "maxItems": len(tool_ids),
                }
            )
        else:
            field_schema = _NON_EMPTY_STRING_SCHEMA
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agent_id", modify_field],
            "properties": {
                "action": {"const": "modify_agent"},
                "agent_id": {"enum": agent_ids},
                modify_field: field_schema,
            },
        }
    elif action in {"delete_agent", "set_output"}:
        agent_ids = [
            str(value)
            for value in domain.get("agent_ids", ())
            if isinstance(value, str) and value
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": action},
                "agent_id": {"enum": agent_ids},
            },
        }
    elif action == "set_relation":
        agent_ids = [
            str(value)
            for value in domain.get("agent_ids", ())
            if isinstance(value, str) and value
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action",
                "source_id",
                "target_id",
                "source_to_target",
                "target_to_source",
            ],
            "properties": {
                "action": {"const": "set_relation"},
                "source_id": {"enum": agent_ids},
                "target_id": {"enum": agent_ids},
                "source_to_target": {"type": "boolean"},
                "target_to_source": {"type": "boolean"},
            },
        }
    elif action == "finish":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"const": "finish"}},
        }
    else:
        raise ValueError("live parameter schema received an unknown action")
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


class DirectorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectorResponse:
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DirectorClient(Protocol):
    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> DirectorResponse:
        ...


class OpenAIDirectorClient:
    """OpenAI-compatible chat client for the local Qwen3.5-9B endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8015/v1",
        model: str = "supervisor_theta",
        api_key_env: Optional[str] = None,
        policy_version: str = "qwen3.5-9b-sglang-unversioned",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 768,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be absolute HTTP(S)")
        if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Flow-Director must use the local Qwen3.5-9B endpoint")
        if model != "supervisor_theta":
            raise ValueError("Flow-Director model must be supervisor_theta")
        if not model.strip() or not policy_version.strip():
            raise ValueError("model and policy_version must be non-empty")
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Director temperature/top_p are invalid")
        if max_tokens <= 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Director token, timeout, and retry limits are invalid")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.policy_version = policy_version
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> DirectorResponse:
        if any(
            value is not None
            for value in (
                action_json_schema,
                action_json_schema_version,
                action_schema_branch,
                action_target_domains_json,
                action_target_domain_version,
            )
        ):
            raise DirectorError(
                "state-conditioned action schemas require the native SGLang client"
            )
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("Director seed must be a non-negative integer or None")
        api_key = "EMPTY"
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env, "")
            if not api_key:
                raise DirectorError(f"missing Director credential environment variable: {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            # SkillFlow sends the generation seed through the provider payload.
            payload["seed"] = seed
        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                value = await asyncio.to_thread(self._post, api_key, payload)
                parsed = self._parse(value)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                        "attempt_count": attempt + 1,
                        "generation_seed": seed,
                    }
                )
                return DirectorResponse(parsed.text, metadata)
            except HTTPError as exc:
                last_error = exc
                if not (exc.code in {408, 409, 425, 429} or exc.code >= 500):
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))
        detail = f"HTTP {last_error.code}" if isinstance(last_error, HTTPError) else type(last_error).__name__
        raise DirectorError(f"Director request failed: {detail}") from last_error

    def _post(self, api_key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-Director/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise DirectorError("Director returned a non-object response")
        return value

    def _parse(self, value: Mapping[str, Any]) -> DirectorResponse:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise DirectorError("Director response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise DirectorError("Director response has no text content")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return DirectorResponse(
            text=message["content"],
            metadata={
                "policy_version": self.policy_version,
                "model": value.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "request_id": value.get("id"),
            },
        )


@dataclass(frozen=True, slots=True)
class DirectorTurn:
    turn_index: int
    prompt: str
    response: DirectorResponse
    canvas_result: AgentWorkflowStepResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    final_answer: Optional[str]
    turns: Tuple[DirectorTurn, ...]
    final_graph: Mapping[str, Any]
    termination_reason: str
    explicit_finish: bool


class AgentGraphOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        client: DirectorClient,
        *,
        max_rounds: int = 20,
        seed: int = 42,
        history_window: int = 4,
        tool_registry: Optional[ToolRegistry] = None,
        sampling_action_profile: Optional[str] = None,
        sampling_action_schema_version: str = (
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
        ),
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if isinstance(history_window, bool) or not isinstance(history_window, int) or history_window < 1:
            raise ValueError("history_window must be a positive integer")
        self.registry = registry
        self.client = client
        self.max_rounds = max_rounds
        self.seed = seed
        self.history_window = history_window
        self.tool_registry = tool_registry
        if sampling_action_profile not in {
            None,
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
        }:
            raise ValueError("unsupported Director sampling action profile")
        self.sampling_action_profile = sampling_action_profile
        if sampling_action_schema_version not in {
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
        }:
            raise ValueError("unsupported Director sampling action schema version")
        self.sampling_action_schema_version = sampling_action_schema_version

    def action_schema_request(
        self,
        env: AgentWorkflowEnv,
    ) -> Mapping[str, str]:
        """Return FlowSteer's evaluation-only live action discriminator."""

        if self.sampling_action_profile is None:
            return {}
        actions = env.model_admissible_action_types()
        if (
            self.sampling_action_schema_version
            == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
        ):
            targets = env.model_admissible_action_targets()
            return {
                "action_json_schema": (
                    director_model_admissible_sampling_json_schema_text(actions)
                ),
                "action_json_schema_version": (
                    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
                ),
                "action_schema_branch": (
                    director_model_admissible_schema_branch_v3(actions)
                ),
                "action_target_domains_json": (
                    director_live_action_target_domains_json(actions, targets)
                ),
                "action_target_domain_version": (
                    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
                ),
            }
        return {
            "action_json_schema": (
                director_model_admissible_sampling_json_schema_text(actions)
            ),
            "action_json_schema_version": (
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            "action_schema_branch": director_model_admissible_schema_branch(
                actions
            ),
        }

    def _tool_catalog(self, env: AgentWorkflowEnv) -> list[dict[str, object]]:
        if self.tool_registry is None:
            return []
        if env.runtime.tool_registry is not self.tool_registry:
            raise DirectorError(
                "Director and AgentRuntime must share the same ToolRegistry"
            )
        dataset_id = env.runtime.dataset_id
        return [
            capability.to_value()
            for capability in self.tool_registry.capabilities
            if dataset_id is None or capability.supports_dataset(dataset_id)
        ]

    def build_prompt(
        self,
        env: AgentWorkflowEnv,
        turn_index: int,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        preferred = self.registry.select_weighted(
            seed=self.seed + turn_index,
            cheap_bias=1.0,
            fast_bias=1.0,
        )
        catalog = [
            {
                "model_id": model_id,
                "selection_weight": self.registry.require_model(model_id).selection_weight,
                "cheap_weight": self.registry.require_model(model_id).cheap_weight,
                "fast_weight": self.registry.require_model(model_id).fast_weight,
            }
            for model_id in self.registry.model_ids
        ]
        complete_validation = env.graph.validate(
            self.registry,
            require_complete=True,
        )
        snapshot = env.snapshot()
        admissible_actions = env.model_admissible_action_types()
        action_target_domains = env.model_admissible_action_targets()
        payload = {
            "task": env.problem,
            "turn": turn_index,
            "max_rounds": self.max_rounds,
            "remaining_rounds": max(self.max_rounds - env.turn_count, 0),
            "current_graph": env.graph.to_dict(),
            "canvas_feedback": snapshot.last_feedback,
            # SkillFlow presents a bounded visible action-history tail to its
            # ReAct policy; keep the same boundary without adding role recipes.
            "recent_canvas_history": [
                entry.to_dict() for entry in snapshot.history[-self.history_window :]
            ],
            "complete_validation": {
                "valid": complete_validation.valid,
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in complete_validation.issues
                ],
            },
            "model_catalog": catalog,
            "weighted_preferred_model": preferred.model_id,
            "admissible_actions": list(admissible_actions),
            "action_target_domains": action_target_domains,
            "recovery_state": env.recovery_state(),
        }
        if env.max_agents is not None:
            payload["max_agents"] = env.max_agents
        tool_catalog = self._tool_catalog(env)
        if tool_catalog:
            payload["tool_catalog"] = tool_catalog
        if skills:
            payload["available_skills"] = list(skills)
        return (
            "Choose one admissible next edit. The preferred model is only a "
            "cheap/fast suggestion.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    async def run(
        self,
        env: AgentWorkflowEnv,
        problem: str,
        *,
        skills: Sequence[Mapping[str, Any]] = (),
    ) -> OrchestrationResult:
        env.reset(problem)
        turns: list[DirectorTurn] = []
        for index in range(self.max_rounds):
            prompt = self.build_prompt(env, index, skills)
            schema_request = self.action_schema_request(env)
            response = await self.client.propose(
                prompt,
                seed=self.seed + index,
                **schema_request,
            )
            canvas = await env.step(response.text)
            turns.append(DirectorTurn(index, prompt, response, canvas))
            if canvas.done and canvas.final_answer is not None:
                return OrchestrationResult(
                    final_answer=canvas.final_answer,
                    turns=tuple(turns),
                    final_graph=env.graph.to_dict(),
                    termination_reason="finish",
                    explicit_finish=True,
                )
        return OrchestrationResult(
            final_answer=None,
            turns=tuple(turns),
            final_graph=env.graph.to_dict(),
            termination_reason="max_rounds",
            explicit_finish=False,
        )


__all__ = [
    "AgentGraphOrchestrator",
    "DIRECTOR_ACTION_JSON_SCHEMA",
    "DIRECTOR_ACTION_SCHEMA_VERSION",
    "DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3",
    "DIRECTOR_SYSTEM_PROMPT",
    "DirectorClient",
    "DirectorError",
    "DirectorResponse",
    "DirectorTurn",
    "OpenAIDirectorClient",
    "OrchestrationResult",
    "director_action_json_schema_text",
    "director_actions_from_admissible_schema_branch",
    "director_model_admissible_sampling_json_schema_text",
    "director_model_admissible_schema_branch",
    "director_model_admissible_schema_branch_v3",
    "director_live_action_parameter_json_schema_text",
    "director_live_action_target_domains_json",
    "director_live_add_subgraph_agent_declarations_from_text",
    "director_live_add_subgraph_agent_declarations_json_schema_text",
    "director_live_execution_mode_selector_json_schema_text",
    "director_live_execution_profile_from_text",
    "director_live_modify_agent_field_selector_json_schema_text",
    "director_state_conditioned_sampling_json_schema_text",
]
