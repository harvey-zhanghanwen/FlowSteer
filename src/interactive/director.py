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
_QWEN_JSON_EOS_TEXT = "<|endoftext|>"
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


def _live_role_constraints(
    action_domain: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    raw = action_domain.get("role_constraints")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("live role_constraints must be a non-empty object")
    for role_family, constraint in raw.items():
        if (
            not isinstance(role_family, str)
            or not role_family
            or not isinstance(constraint, Mapping)
        ):
            raise ValueError("live role constraint is malformed")
        profiles = constraint.get("execution_profiles")
        if not isinstance(profiles, (list, tuple)) or not profiles:
            raise ValueError("live role constraint has no execution profiles")
    return raw


def _live_admitted_role_families(
    action_domain: Mapping[str, Any],
    role_constraints: Mapping[str, Any],
) -> tuple[str, ...]:
    raw = action_domain.get("admitted_new_role_families")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("live admitted_new_role_families is missing")
    roles = tuple(raw)
    if (
        len(roles) != len(set(roles))
        or any(
            not isinstance(role, str) or role not in role_constraints
            for role in roles
        )
    ):
        raise ValueError("live admitted_new_role_families is invalid")
    return roles


def _live_role_execution_profiles(
    role_family: str,
    constraint: Mapping[str, Any],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    profiles = _live_execution_profiles(
        {"execution_profiles": constraint.get("execution_profiles")}
    )
    if not profiles:
        raise ValueError(f"live role {role_family!r} has no execution profile")
    return profiles


def _live_role_profile_for_mode(
    action_domain: Mapping[str, Any],
    role_family: str,
    execution_mode: str,
) -> tuple[str, tuple[str, ...]]:
    constraints = _live_role_constraints(action_domain)
    if constraints is None or role_family not in constraints:
        raise ValueError("live Agent role is outside the typed domain")
    profiles = [
        profile
        for profile in _live_role_execution_profiles(
            role_family,
            constraints[role_family],
        )
        if profile[0] == execution_mode
    ]
    if len(profiles) != 1:
        raise ValueError(
            "live role/execution_mode must identify exactly one Runtime profile"
        )
    if profiles[0] not in _live_execution_profiles(action_domain):
        raise ValueError("live role exposes an unregistered Runtime profile")
    return profiles[0]


def _live_agent_schema(
    action_domain: Mapping[str, Any],
    *,
    declaration_phase: bool = False,
    execution_profile: Optional[tuple[str, tuple[str, ...]]] = None,
) -> Mapping[str, Any]:
    model_ids, execution_modes, tool_ids = _live_execution_domain(action_domain)
    role_constraints = _live_role_constraints(action_domain)
    admitted_roles = (
        ()
        if role_constraints is None
        else _live_admitted_role_families(action_domain, role_constraints)
    )
    required = ["agent_id", "model_id", "contract"]
    properties: dict[str, Any] = {
        "agent_id": _NON_EMPTY_STRING_SCHEMA,
        "model_id": {"enum": model_ids},
        "contract": _NON_EMPTY_STRING_SCHEMA,
        "role_family": (
            _NON_EMPTY_STRING_SCHEMA
            if role_constraints is None
            else {"enum": list(admitted_roles)}
        ),
        "artifact_type": _NON_EMPTY_STRING_SCHEMA,
        "completion_condition": _NON_EMPTY_STRING_SCHEMA,
    }
    if declaration_phase:
        if role_constraints is not None:
            role_branches: list[Mapping[str, Any]] = []
            for role_family in admitted_roles:
                for mode, _ in _live_role_execution_profiles(
                    role_family,
                    role_constraints[role_family],
                ):
                    role_branches.append(
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "agent_id",
                                "model_id",
                                "contract",
                                "role_family",
                                "execution_mode",
                            ],
                            "properties": {
                                **properties,
                                "role_family": {"enum": [role_family]},
                                "execution_mode": {"const": mode},
                            },
                        }
                    )
            if not role_branches:
                raise ValueError("live typed Agent declaration domain is empty")
            return {"anyOf": role_branches}
        required.append("execution_mode")
        properties["execution_mode"] = {"enum": execution_modes}
    elif execution_profile is not None:
        mode, tools = execution_profile
        if (mode, tuple(tools)) not in _live_execution_profiles(action_domain):
            raise ValueError("selected execution profile is outside the live domain")
        required.extend(["execution_mode", "allowed_tools"])
        properties["execution_mode"] = {"const": mode}
        properties["allowed_tools"] = {"const": list(tools)}
        if role_constraints is not None:
            compatible_roles = [
                role_family
                for role_family in admitted_roles
                if (mode, tuple(tools))
                in _live_role_execution_profiles(
                    role_family,
                    role_constraints[role_family],
                )
            ]
            if not compatible_roles:
                raise ValueError(
                    "selected execution profile has no admitted semantic role"
                )
            required.append("role_family")
            properties["role_family"] = {"enum": compatible_roles}
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


def _live_new_agent_ids(
    existing_agent_ids: Sequence[str],
    max_agents: int,
) -> Tuple[str, ...]:
    """Assign FlowSteer-style neutral node_N IDs for one ADD phase."""

    if (
        isinstance(existing_agent_ids, (str, bytes))
        or any(
            not isinstance(agent_id, str) or not agent_id
            for agent_id in existing_agent_ids
        )
        or len(existing_agent_ids) != len(set(existing_agent_ids))
    ):
        raise ValueError("existing Agent IDs are invalid")
    if type(max_agents) is not int or not 1 <= max_agents <= 3:
        raise ValueError("new Agent ID count must be between one and three")
    used = set(existing_agent_ids)
    result: list[str] = []
    index = 1
    while len(result) < max_agents:
        candidate = f"node_{index}"
        index += 1
        if candidate in used:
            continue
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def _typed_add_agent_schema(
    domain: Mapping[str, Any],
    *,
    agent_id: str,
    role_family: str,
) -> Mapping[str, Any]:
    constraints = _live_role_constraints(domain)
    if constraints is None or role_family not in constraints:
        raise ValueError("typed ADD role is outside the live domain")
    model_ids, _, _ = _live_execution_domain(domain)
    required = [
        "agent_id",
        "model_id",
        "contract",
        "role_family",
        "execution_mode",
        "allowed_tools",
    ]
    branches: list[Mapping[str, Any]] = []
    for execution_mode, allowed_tools in _live_role_execution_profiles(
        role_family,
        constraints[role_family],
    ):
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": {
                    "agent_id": {"const": agent_id},
                    "model_id": {"enum": model_ids},
                    "contract": _NON_EMPTY_STRING_SCHEMA,
                    "role_family": {"const": role_family},
                    "execution_mode": {"const": execution_mode},
                    "allowed_tools": {"const": list(allowed_tools)},
                    "artifact_type": _NON_EMPTY_STRING_SCHEMA,
                    "completion_condition": _NON_EMPTY_STRING_SCHEMA,
                },
            }
        )
    if not branches:
        raise ValueError("typed ADD role has no Runtime execution profile")
    return {"anyOf": branches}


def director_live_add_subgraph_agent_declarations_json_schema_text(
    action_target_domains: Mapping[str, Any],
    *,
    selected_agent_roles: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Render positional ADD declarations with Canvas-assigned Agent IDs."""

    domain = action_target_domains.get("add_subgraph")
    if not isinstance(domain, Mapping):
        raise ValueError("add_subgraph has no live target domain")
    max_agents = domain.get("max_new_agents", 3)
    min_agents = domain.get("min_new_agents", 1)
    if (
        type(min_agents) is not int
        or type(max_agents) is not int
        or not 1 <= min_agents <= max_agents <= 3
    ):
        raise ValueError("add_subgraph live Agent-count domain is invalid")
    role_constraints = _live_role_constraints(domain)
    if role_constraints is None:
        if selected_agent_roles is not None:
            raise ValueError("untyped ADD has no role-selection phase")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "type": "array",
                    "minItems": min_agents,
                    "maxItems": max_agents,
                    "items": _live_agent_schema(domain, declaration_phase=True),
                },
            },
        }
        return json.dumps(schema, sort_keys=True, separators=(",", ":"))
    existing_ids = domain.get("existing_agent_ids", ())
    if not isinstance(existing_ids, (list, tuple)):
        raise ValueError("typed ADD existing Agent IDs are invalid")
    new_ids = _live_new_agent_ids(existing_ids, max_agents)
    admitted_roles = _live_admitted_role_families(domain, role_constraints)
    selected_roles: Optional[Tuple[str, ...]] = None
    if selected_agent_roles is not None:
        if not min_agents <= len(selected_agent_roles) <= max_agents:
            raise ValueError("add_subgraph selected Agent roles have invalid count")
        values: list[str] = []
        for position, item in enumerate(selected_agent_roles):
            if not isinstance(item, Mapping) or set(item) != {
                "agent_id",
                "role_family",
            }:
                raise ValueError("add_subgraph selected Agent role is malformed")
            if item.get("agent_id") != new_ids[position]:
                raise ValueError(
                    "add_subgraph selected Agent role changed its Canvas node ID"
                )
            role = item.get("role_family")
            if not isinstance(role, str) or role not in admitted_roles:
                raise ValueError(
                    "add_subgraph selected Agent role is outside the live domain"
                )
            values.append(role)
        selected_roles = tuple(values)
    positional_schemas: list[Mapping[str, Any]] = []
    positional_count = len(selected_roles) if selected_roles is not None else max_agents
    for position, agent_id in enumerate(new_ids[:positional_count]):
        roles = (
            (selected_roles[position],)
            if selected_roles is not None
            else admitted_roles
        )
        branches = [
            branch
            for role in roles
            for branch in _typed_add_agent_schema(
                domain,
                agent_id=agent_id,
                role_family=role,
            )["anyOf"]
        ]
        positional_schemas.append({"anyOf": branches})
    counts = (
        (len(selected_roles),)
        if selected_roles is not None
        else tuple(range(min_agents, max_agents + 1))
    )
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "oneOf": [
                        {
                            "type": "array",
                            "minItems": count,
                            "maxItems": count,
                            "prefixItems": positional_schemas[:count],
                            "items": False,
                        }
                        for count in counts
                    ]
                },
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_add_subgraph_role_selection_json_schema_text(
    action_target_domains: Mapping[str, Any],
) -> str:
    """Select typed ADD count and role families before free contracts."""

    director_live_add_subgraph_agent_declarations_json_schema_text(
        action_target_domains
    )
    domain = action_target_domains["add_subgraph"]
    role_constraints = _live_role_constraints(domain)
    if role_constraints is None:
        raise ValueError("untyped ADD has no role-selection phase")
    min_agents = domain.get("min_new_agents", 1)
    max_agents = domain.get("max_new_agents", 3)
    roles = _live_admitted_role_families(domain, role_constraints)
    new_ids = _live_new_agent_ids(domain.get("existing_agent_ids", ()), max_agents)
    positional_roles = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["agent_id", "role_family"],
            "properties": {
                "agent_id": {"const": agent_id},
                "role_family": {"enum": list(roles)},
            },
        }
        for agent_id in new_ids
    ]
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "oneOf": [
                        {
                            "type": "array",
                            "minItems": count,
                            "maxItems": count,
                            "prefixItems": positional_roles[:count],
                            "items": False,
                        }
                        for count in range(min_agents, max_agents + 1)
                    ]
                },
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_add_subgraph_role_selection_from_text(
    text: str,
    action_target_domains: Mapping[str, Any],
) -> Tuple[Mapping[str, str], ...]:
    """Parse the exact typed ADD count/role selection."""

    try:
        value, end = json.JSONDecoder().raw_decode(text.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("ADD role-selection phase is not JSON") from exc
    trailing = text.strip()[end:].strip()
    if trailing and trailing != _QWEN_JSON_EOS_TEXT:
        raise ValueError(
            "ADD role-selection phase contains trailing text: "
            f"{trailing[:80]!r}"
        )
    if not isinstance(value, Mapping) or set(value) != {"action", "agents"}:
        raise ValueError("ADD role-selection phase has incompatible fields")
    if value.get("action") != "add_subgraph":
        raise ValueError("ADD role-selection phase changed its action")
    director_live_add_subgraph_role_selection_json_schema_text(
        action_target_domains
    )
    domain = action_target_domains["add_subgraph"]
    raw_agents = value.get("agents")
    min_agents = domain.get("min_new_agents", 1)
    max_agents = domain.get("max_new_agents", 3)
    if (
        not isinstance(raw_agents, list)
        or not min_agents <= len(raw_agents) <= max_agents
    ):
        raise ValueError("ADD selected Agent roles have invalid count")
    constraints = _live_role_constraints(domain)
    assert constraints is not None
    roles = _live_admitted_role_families(domain, constraints)
    expected_ids = _live_new_agent_ids(
        domain.get("existing_agent_ids", ()),
        max_agents,
    )
    normalized: list[Mapping[str, str]] = []
    for position, item in enumerate(raw_agents):
        if not isinstance(item, Mapping) or set(item) != {
            "agent_id",
            "role_family",
        }:
            raise ValueError("ADD selected Agent role is malformed")
        if item.get("agent_id") != expected_ids[position]:
            raise ValueError("ADD selected Agent reused or changed a Canvas node ID")
        role = item.get("role_family")
        if not isinstance(role, str) or role not in roles:
            raise ValueError("ADD selected Agent role is outside the live domain")
        normalized.append(
            {"agent_id": expected_ids[position], "role_family": role}
        )
    return tuple(normalized)


def director_live_add_subgraph_agent_declarations_from_text(
    text: str,
    action_target_domains: Optional[Mapping[str, Any]] = None,
    *,
    selected_agent_roles: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[Mapping[str, Any], ...]:
    """Parse an exact constrained ADD declaration without rewriting it."""

    stripped = text.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except (TypeError, ValueError) as exc:
        raise ValueError("ADD declaration phase is not JSON") from exc
    trailing = stripped[end:].strip()
    if trailing and trailing != _QWEN_JSON_EOS_TEXT:
        raise ValueError(
            "ADD declaration phase contains trailing text: "
            f"{trailing[:80]!r}"
        )
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
        role_constraints = _live_role_constraints(domain)
        if role_constraints is not None:
            min_agents = domain.get("min_new_agents", 1)
            max_agents = domain.get("max_new_agents", 3)
            if not min_agents <= len(normalized) <= max_agents:
                raise ValueError("ADD declaration has invalid Agent count")
            expected_ids = _live_new_agent_ids(
                domain.get("existing_agent_ids", ()),
                max_agents,
            )
            admitted_roles = _live_admitted_role_families(
                domain,
                role_constraints,
            )
            expected_roles = (
                None
                if selected_agent_roles is None
                else tuple(
                    (item.get("agent_id"), item.get("role_family"))
                    for item in selected_agent_roles
                )
            )
            if expected_roles is not None and len(expected_roles) != len(normalized):
                raise ValueError("ADD declaration changed its selected Agent count")
            bound: list[Mapping[str, Any]] = []
            seen_ids: set[str] = set()
            for position, declaration in enumerate(normalized):
                agent_id = declaration.get("agent_id")
                role_family = declaration.get("role_family")
                model_id = declaration.get("model_id")
                contract = declaration.get("contract")
                mode = declaration.get("execution_mode")
                if (
                    agent_id != expected_ids[position]
                    or agent_id in seen_ids
                    or not isinstance(role_family, str)
                    or role_family not in admitted_roles
                    or not isinstance(model_id, str)
                    or model_id not in domain.get("model_ids", ())
                    or not isinstance(contract, str)
                    or not contract.strip()
                    or not isinstance(mode, str)
                ):
                    raise ValueError(
                        "typed ADD Agent declaration is outside the live domain"
                    )
                _, tools = _live_role_profile_for_mode(
                    domain,
                    role_family,
                    mode,
                )
                existing_tools = declaration.get("allowed_tools")
                if existing_tools != list(tools):
                    raise ValueError("ADD declaration changed its Runtime profile")
                if expected_roles is not None and (
                    agent_id,
                    role_family,
                ) != expected_roles[position]:
                    raise ValueError("ADD declaration changed its selected role")
                seen_ids.add(agent_id)
                bound.append(dict(declaration))
            return tuple(bound)
        bound: list[Mapping[str, Any]] = []
        for declaration in normalized:
            mode = declaration.get("execution_mode")
            if not isinstance(mode, str):
                raise ValueError("ADD declaration has no execution mode")
            role_family = declaration.get("role_family")
            if _live_role_constraints(domain) is not None:
                if not isinstance(role_family, str):
                    raise ValueError("ADD declaration has no semantic role")
                _, tools = _live_role_profile_for_mode(
                    domain,
                    role_family,
                    mode,
                )
            else:
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


def director_live_modify_agent_field_selector_json_schema_text(
    action_target_domains: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render the FlowSteer MODIFY-field discriminator from live deltas."""

    fields = list(_MUTABLE_AGENT_PROPERTIES)
    if action_target_domains is not None:
        domain = action_target_domains.get("modify_agent")
        if not isinstance(domain, Mapping):
            raise ValueError("modify_agent has no live target domain")
        raw_fields = domain.get("mutable_fields")
        if (
            not isinstance(raw_fields, (list, tuple))
            or not raw_fields
            or any(field not in _MUTABLE_AGENT_PROPERTIES for field in raw_fields)
            or len(raw_fields) != len(set(raw_fields))
        ):
            raise ValueError("modify_agent live mutable field domain is invalid")
        fields = list(raw_fields)

    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "field"],
            "properties": {
                "action": {"const": "modify_agent"},
                "field": {"enum": fields},
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _live_existing_agent_roles(
    domain: Mapping[str, Any],
    role_constraints: Mapping[str, Any],
) -> dict[str, str]:
    existing_ids = tuple(domain.get("existing_agent_ids", ()))
    raw_agents = domain.get("existing_agents")
    if not isinstance(raw_agents, (list, tuple)) or len(raw_agents) != len(
        existing_ids
    ):
        raise ValueError("typed ADD existing-Agent role domain is incomplete")
    roles: dict[str, str] = {}
    ordered_ids: list[str] = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, Mapping):
            raise ValueError("typed ADD existing-Agent role entry is malformed")
        agent_id = raw_agent.get("agent_id")
        role_family = raw_agent.get("role_family")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or not isinstance(role_family, str)
            or role_family not in role_constraints
            or agent_id in roles
        ):
            raise ValueError("typed ADD existing-Agent role entry is invalid")
        ordered_ids.append(agent_id)
        roles[agent_id] = role_family
    if tuple(ordered_ids) != existing_ids:
        raise ValueError("typed ADD existing-Agent roles changed Canvas order")
    return roles


def _hotpotqa_directed_role_relation_allowed(
    source_role: str,
    target_role: str,
) -> bool:
    """Mirror the incremental HotpotQA semantic-edge validator."""

    if source_role == "format":
        return False
    if target_role == "verifier":
        return source_role == "reasoner"
    if target_role == "format":
        return source_role == "verifier"
    return True


def director_live_add_subgraph_relation_candidates(
    action_target_domains: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    """Project exact role-valid relations for one sampled ADD unit.

    DIRECT_REUSE: this is the strict unified-QA ADD relation projection.  ADD
    admits at most one relation incident to a newly declared Agent; later
    state-conditioned SET_RELATION edits grow the graph after execution
    feedback.  This keeps topology sampled while removing open endpoint and
    direction fields that the Canvas would reject.
    """

    domain = action_target_domains.get("add_subgraph")
    if not isinstance(domain, Mapping):
        raise ValueError("add_subgraph has no live target domain")
    if domain.get("semantic_protocol") != "hotpotqa.qa_memory.worker_lineage.v1":
        return ()
    agents = director_live_add_subgraph_agent_declarations_from_text(
        json.dumps(
            {"action": "add_subgraph", "agents": [dict(agent) for agent in agents]},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        action_target_domains,
    )
    role_constraints = _live_role_constraints(domain)
    if role_constraints is None:
        raise ValueError("typed ADD role constraints are missing")
    roles = _live_existing_agent_roles(domain, role_constraints)
    existing_ids = tuple(domain.get("existing_agent_ids", ()))
    same_action_ids: set[str] = set()
    for agent in agents:
        if not isinstance(agent, Mapping):
            raise ValueError("typed ADD Agent declaration is malformed")
        agent_id = agent.get("agent_id")
        role_family = agent.get("role_family")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or agent_id in roles
            or agent_id in same_action_ids
            or not isinstance(role_family, str)
            or role_family not in role_constraints
        ):
            raise ValueError("typed ADD Agent declaration is outside the live domain")
        same_action_ids.add(agent_id)
        roles[agent_id] = role_family
    endpoint_ids = [*existing_ids, *[str(agent["agent_id"]) for agent in agents]]
    semantic_pairs = {
        ("evidence_retriever", "reasoner"),
        ("repair", "reasoner"),
        ("reasoner", "verifier"),
        ("verifier", "format"),
    }
    candidates: list[Mapping[str, Any]] = []
    for source_index, source_id in enumerate(endpoint_ids):
        for target_id in endpoint_ids[source_index + 1 :]:
            if source_id not in same_action_ids and target_id not in same_action_ids:
                continue
            source_role = roles[source_id]
            target_role = roles[target_id]
            source_to_target = _hotpotqa_directed_role_relation_allowed(
                source_role,
                target_role,
            )
            target_to_source = _hotpotqa_directed_role_relation_allowed(
                target_role,
                source_role,
            )
            forward = (source_role, target_role) in semantic_pairs
            reverse = (target_role, source_role) in semantic_pairs
            if forward or reverse:
                if forward and source_to_target:
                    sender_id, receiver_id = source_id, target_id
                elif reverse and target_to_source:
                    sender_id, receiver_id = target_id, source_id
                else:
                    sender_id = receiver_id = None
                if sender_id is not None and receiver_id is not None:
                    candidates.append(
                        {
                            "source_id": sender_id,
                            "target_id": receiver_id,
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    )
                if (
                    source_id in same_action_ids
                    and target_id in same_action_ids
                    and source_to_target
                    and target_to_source
                ):
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_to_target": True,
                            "target_to_source": True,
                        }
                    )
                continue
            if source_to_target:
                candidates.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                )
            if target_to_source:
                candidates.append(
                    {
                        "source_id": target_id,
                        "target_id": source_id,
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                )
            if (
                source_id in same_action_ids
                and target_id in same_action_ids
                and source_to_target
                and target_to_source
            ):
                candidates.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_to_target": True,
                        "target_to_source": True,
                    }
                )
    return tuple(candidates)


def _live_modify_agent_candidates(
    action_target_domains: Mapping[str, Any],
    field_name: str,
) -> Tuple[Mapping[str, Any], ...]:
    domain = action_target_domains.get("modify_agent")
    if not isinstance(domain, Mapping):
        raise ValueError("modify_agent has no live target domain")
    raw_fields = domain.get("mutable_fields")
    raw_candidates = domain.get("per_agent_candidates")
    if (
        not isinstance(raw_fields, (list, tuple))
        or field_name not in raw_fields
        or not isinstance(raw_candidates, (list, tuple))
    ):
        raise ValueError("modify_agent field is outside the live domain")
    admitted: list[Mapping[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("modify_agent candidate is malformed")
        agent_id = candidate.get("agent_id")
        fields = candidate.get("mutable_fields")
        current_values = candidate.get("current_values")
        discrete_domains = candidate.get("discrete_value_domains")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or not isinstance(fields, (list, tuple))
            or not isinstance(current_values, Mapping)
            or not isinstance(discrete_domains, Mapping)
        ):
            raise ValueError("modify_agent candidate is malformed")
        if field_name in fields:
            admitted.append(candidate)
    if not admitted:
        raise ValueError("modify_agent field has no live Agent target")
    return tuple(admitted)


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
        output_agent_ids: list[str] = endpoint_ids
        role_constraints = _live_role_constraints(domain)
        if role_constraints is not None:
            roles: dict[str, str] = {}
            raw_existing_agents = domain.get("existing_agents", ())
            if not isinstance(raw_existing_agents, (list, tuple)):
                raise ValueError("typed ADD domain has no existing Agent roles")
            for item in raw_existing_agents:
                if not isinstance(item, Mapping):
                    raise ValueError("typed existing Agent role is malformed")
                agent_id = item.get("agent_id")
                role_family = item.get("role_family")
                if not isinstance(agent_id, str) or not isinstance(
                    role_family,
                    str,
                ):
                    raise ValueError("typed existing Agent role is invalid")
                roles[agent_id] = role_family
            for agent in agents:
                roles[str(agent["agent_id"])] = str(agent["role_family"])
            output_role = domain.get("output_role_family")
            if not isinstance(output_role, str) or output_role not in role_constraints:
                raise ValueError("typed ADD output role domain is invalid")
            output_agent_ids = [
                agent_id
                for agent_id in endpoint_ids
                if roles.get(agent_id) == output_role
            ]
            relation_candidates = director_live_add_subgraph_relation_candidates(
                action_target_domains,
                agents,
            )
        else:
            relation_candidates = ()
        if role_constraints is not None:
            relations_schema: Mapping[str, Any] = {
                "type": "array",
                "maxItems": 1 if relation_candidates else 0,
                **(
                    {
                        "uniqueItems": True,
                        "items": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "source_id",
                                        "target_id",
                                        "source_to_target",
                                        "target_to_source",
                                    ],
                                    "properties": {
                                        key: {"const": value}
                                        for key, value in candidate.items()
                                    },
                                }
                                for candidate in relation_candidates
                            ]
                        },
                    }
                    if relation_candidates
                    else {}
                ),
            }
        else:
            relations_schema = {
                "type": "array",
                "maxItems": len(endpoint_ids)
                * max(len(endpoint_ids) - 1, 0),
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
            }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents", "relations"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {"const": agents},
                "relations": relations_schema,
                # A typed QA Output is selected only by the subsequent live
                # SET_OUTPUT boundary after the complete relation lineage has
                # passed authoritative Canvas validation.  This prevents an
                # ADD declaration which omits Verifier from claiming Format.
                "output_agent_id": (
                    {"const": None}
                    if role_constraints is not None
                    else {"enum": [*output_agent_ids, None]}
                ),
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
        if "per_agent_candidates" in domain:
            candidates = _live_modify_agent_candidates(
                action_target_domains,
                modify_field,
            )
            branches: list[Mapping[str, Any]] = []
            for candidate in candidates:
                raw_discrete = candidate["discrete_value_domains"]
                discrete_values = raw_discrete.get(modify_field)
                if discrete_values is not None:
                    if not isinstance(discrete_values, (list, tuple)) or not discrete_values:
                        raise ValueError(
                            "modify_agent discrete value domain must be non-empty"
                        )
                    field_schema: Mapping[str, Any] = {
                        "enum": list(discrete_values)
                    }
                else:
                    field_schema = _MUTABLE_AGENT_PROPERTIES[modify_field]
                branches.append(
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action", "agent_id", modify_field],
                        "properties": {
                            "action": {"const": "modify_agent"},
                            "agent_id": {"const": candidate["agent_id"]},
                            modify_field: field_schema,
                        },
                    }
                )
            schema = {"type": "object", "oneOf": branches}
        else:
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
                field_schema = {"enum": values}
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
        candidates = domain.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("set_relation exact live candidates are missing")
        branches: list[Mapping[str, Any]] = []
        required = (
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or set(candidate) != set(required):
                raise ValueError("set_relation live candidate is malformed")
            source_id = candidate.get("source_id")
            target_id = candidate.get("target_id")
            source_to_target = candidate.get("source_to_target")
            target_to_source = candidate.get("target_to_source")
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(target_id, str)
                or not target_id
                or source_id == target_id
                or type(source_to_target) is not bool
                or type(target_to_source) is not bool
            ):
                raise ValueError("set_relation live candidate is invalid")
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action", *required],
                    "properties": {
                        "action": {"const": "set_relation"},
                        **{
                            key: {"const": candidate[key]}
                            for key in required
                        },
                    },
                }
            )
        schema = {"type": "object", "oneOf": branches}
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

    def terminal_canvas_diagnosis(
        self,
        env: AgentWorkflowEnv,
    ) -> Optional[Mapping[str, Any]]:
        """Return FlowSteer's natural terminal for exhausted QA-memory edits.

        DIRECT_REUSE + NECESSARY ADAPTATION: unified QA stops a bounded
        progressive Canvas when its live action domain is empty.  Do not ask
        the Director to sample outside that domain and do not synthesize a
        FINISH action.  This projection contains only control-plane state.
        """

        if env.required_evidence_tool_id != "hotpotqa.qa_memory":
            return None
        if env.model_admissible_action_types():
            return None
        return {
            "public_error_code": "canvas_action_domain_exhausted",
            "graph_revision": env.graph.revision,
            "finish_admissibility": dict(env.finish_admissibility()),
            "recovery_state": dict(env.recovery_state()),
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
    "director_live_add_subgraph_relation_candidates",
    "director_live_add_subgraph_role_selection_from_text",
    "director_live_add_subgraph_role_selection_json_schema_text",
    "director_live_execution_mode_selector_json_schema_text",
    "director_live_execution_profile_from_text",
    "director_live_modify_agent_field_selector_json_schema_text",
    "director_state_conditioned_sampling_json_schema_text",
]
