from __future__ import annotations

import asyncio
from itertools import product
import json
from collections.abc import Mapping, Sequence

import pytest

from src.interactive.director import (
    director_live_action_parameter_json_schema_text,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_relation_candidates,
)
from src.interactive.agent_runtime import AgentRequest
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def _domains(
    *,
    enabled: bool,
    existing_agent_ids: Sequence[str] = (),
    max_new_agents: int = 3,
    min_relations: int = 0,
    max_relations: int = 3,
    output_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    domain: dict[str, object] = {
        "declaration_mode": "free_contract_execution_profile",
        "min_new_agents": 1,
        "max_new_agents": max_new_agents,
        "existing_agent_ids": list(existing_agent_ids),
        "existing_agents": [
            {
                "agent_id": agent_id,
                "model_id": "model",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
            for agent_id in existing_agent_ids
        ],
        "required_agent_fields": [
            "agent_id",
            "model_id",
            "contract",
            "execution_mode",
            "allowed_tools",
        ],
        "contract_type": "free_text",
        "model_ids": ["model"],
        "execution_profiles": [
            {"execution_mode": "reasoning", "allowed_tools": []}
        ],
        "model_execution_profiles": [
            {
                "model_id": "model",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
        ],
        "required_tool_id": None,
        "require_declared_dependency_relations": enabled,
        "min_relations": min_relations,
        "max_relations": max_relations,
        "endpoint_scope": {
            "relation_endpoint_sources": [
                "existing_agent_ids",
                "same_action_agent_ids",
            ],
            "output_agent_id_sources": [
                "existing_agent_ids",
                "same_action_agent_ids",
            ],
        },
    }
    if output_provenance is not None:
        domain["output_provenance"] = dict(output_provenance)
    return {"add_subgraph": domain}


def _agent(agent_id: str, contract: str) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "model_id": "model",
        "contract": contract,
        "execution_mode": "reasoning",
        "allowed_tools": [],
    }


def _relation_from_schema(schema: Mapping[str, object]) -> dict[str, object]:
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    return {
        key: value["const"]
        for key, value in properties.items()
        if isinstance(value, Mapping)
    }


def _relation_array_variants(
    schema: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], ...]:
    branches = schema.get("oneOf", (schema,))
    assert isinstance(branches, Sequence)
    variants: list[tuple[dict[str, object], ...]] = []
    for branch in branches:
        assert isinstance(branch, Mapping)
        if branch.get("maxItems") == 0:
            variants.append(())
            continue
        prefix_items = branch.get("prefixItems")
        assert isinstance(prefix_items, Sequence)
        item_choices: list[tuple[Mapping[str, object], ...]] = []
        for item_schema in prefix_items:
            assert isinstance(item_schema, Mapping)
            raw_choices = item_schema.get("anyOf", (item_schema,))
            assert isinstance(raw_choices, Sequence)
            item_choices.append(tuple(raw_choices))
        for selected in product(*item_choices):
            variants.append(
                tuple(_relation_from_schema(item) for item in selected)
            )
    return tuple(variants)


def _directed_edges(
    relations: Sequence[Mapping[str, object]],
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for relation in relations:
        source_id = relation["source_id"]
        target_id = relation["target_id"]
        assert isinstance(source_id, str)
        assert isinstance(target_id, str)
        if relation["source_to_target"] is True:
            edges.add((source_id, target_id))
        if relation["target_to_source"] is True:
            edges.add((target_id, source_id))
    return edges


class _RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return f"artifact from {request.agent.id}"


def _environment(gateway: _RecordingGateway) -> AgentWorkflowEnv:
    registry = ModelRegistry(
        [ProviderSpec("provider", kind="test")],
        [ModelSpec("model", "provider")],
    )
    return AgentWorkflowEnv(
        registry,
        gateway,
        problem="Answer the current healthcare question.",
        execute_on_edit=True,
        require_declared_dependency_relations=True,
    )


def _dependency_add_action(*, reciprocal: bool, reverse: bool = False) -> str:
    source_id, target_id = (
        ("node_2", "node_1") if reverse else ("node_1", "node_2")
    )
    return json.dumps(
        {
            "action": "add_subgraph",
            "agents": [
                _agent("node_1", "Search authoritative sources and return evidence."),
                _agent(
                    "node_2",
                    "Synthesize the search results from node_1 into an answer.",
                ),
            ],
            "relations": [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_to_target": True,
                    "target_to_source": reciprocal,
                }
            ],
        }
    )


def test_live_candidates_remove_reverse_only_but_keep_reciprocal_and_free_pairs() -> None:
    domains = _domains(enabled=True)
    agents = [
        _agent("node_1", "Search authoritative sources and return evidence."),
        _agent(
            "node_2",
            "Synthesize the search results from node_1 into a complete answer.",
        ),
        _agent("node_3", "Independently check the answer for material conflicts."),
    ]

    candidates = director_live_add_subgraph_relation_candidates(domains, agents)

    forward = {
        "source_id": "node_1",
        "target_id": "node_2",
        "source_to_target": True,
        "target_to_source": False,
    }
    reverse = {
        "source_id": "node_2",
        "target_id": "node_1",
        "source_to_target": True,
        "target_to_source": False,
    }
    reciprocal = {
        "source_id": "node_1",
        "target_id": "node_2",
        "source_to_target": True,
        "target_to_source": True,
    }
    unrelated_reverse = {
        "source_id": "node_3",
        "target_id": "node_1",
        "source_to_target": True,
        "target_to_source": False,
    }
    assert forward in candidates
    assert reverse not in candidates
    assert reciprocal in candidates
    assert unrelated_reverse in candidates


def test_dependency_projection_ignores_outgoing_relation_language() -> None:
    assert AgentWorkflowEnv._declared_contract_dependency_ids(
        "Send the completed evidence artifact to node_2 for synthesis."
    ) == ()
    assert AgentWorkflowEnv._declared_contract_dependency_ids(
        "Synthesize the search results from node_1 into a complete answer."
    ) == ("node_1",)


def test_final_add_schema_requires_all_dependencies_not_unrelated_equal_count() -> None:
    domains = _domains(enabled=True, max_relations=2)
    agents = [
        _agent("node_1", "Produce a complete evidence artifact."),
        _agent("node_2", "Produce an independent safety analysis."),
        _agent(
            "node_3",
            "Synthesize node_1's artifact and node_2's findings into the response.",
        ),
    ]

    schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=agents,
        )
    )
    variants = _relation_array_variants(schema["properties"]["relations"])

    assert variants
    assert all(len(relations) == 2 for relations in variants)
    assert all(
        {("node_1", "node_3"), ("node_2", "node_3")}
        <= _directed_edges(relations)
        for relations in variants
    )
    assert any(
        any(relation["target_to_source"] is True for relation in relations)
        for relations in variants
    ), "a reciprocal relation remains legal when it carries the required direction"


def test_final_add_schema_keeps_optional_topology_after_dependency_coverage() -> None:
    domains = _domains(enabled=True, max_relations=2)
    agents = [
        _agent("node_1", "Produce a complete evidence artifact."),
        _agent("node_2", "Consume the evidence from node_1 and draft an answer."),
        _agent("node_3", "Independently inspect the task for ambiguity."),
    ]

    schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=agents,
        )
    )
    variants = _relation_array_variants(schema["properties"]["relations"])

    assert all(("node_1", "node_2") in _directed_edges(item) for item in variants)
    assert any(len(item) == 1 for item in variants)
    assert any(
        len(item) == 2
        and any(
            {relation["source_id"], relation["target_id"]}
            == {"node_1", "node_3"}
            for relation in item
        )
        for item in variants
    )


def test_disabled_guard_preserves_reverse_and_empty_relation_domains() -> None:
    domains = _domains(enabled=False, max_new_agents=2, max_relations=1)
    agents = [
        _agent("node_1", "Produce a complete evidence artifact."),
        _agent("node_2", "Synthesize the result from node_1 into an answer."),
    ]

    candidates = director_live_add_subgraph_relation_candidates(domains, agents)
    reverse = {
        "source_id": "node_2",
        "target_id": "node_1",
        "source_to_target": True,
        "target_to_source": False,
    }
    assert reverse in candidates
    schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=agents,
        )
    )
    variants = _relation_array_variants(schema["properties"]["relations"])
    assert () in variants
    assert any(relations == (reverse,) for relations in variants)


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        ("Use node_9's artifact to answer the task.", "unknown Agent"),
        ("Use node_1's artifact to answer the task.", "itself as an input"),
    ],
)
def test_declaration_rejects_unknown_or_self_dependency(
    contract: str,
    message: str,
) -> None:
    domains = _domains(enabled=True, max_new_agents=1, max_relations=1)
    declaration = json.dumps(
        {"action": "add_subgraph", "agents": [_agent("node_1", contract)]}
    )

    with pytest.raises(ValueError, match=message):
        director_live_add_subgraph_agent_declarations_from_text(
            declaration,
            domains,
        )


def test_output_closure_binds_the_explicit_member_of_a_sink_component() -> None:
    provenance = {
        "mode": "required_new_terminal_consumer",
        "eligible_existing_agent_ids": [],
        "same_action_agents_eligible": True,
        "remaining_capacity": 1,
        "eligible_input_agent_ids": ["node_1", "node_2"],
        "required_ingress_component_agent_ids": [["node_1", "node_2"]],
        "required_ingress_count": 1,
    }
    domains = _domains(
        enabled=True,
        existing_agent_ids=("node_1", "node_2"),
        max_new_agents=1,
        min_relations=1,
        max_relations=1,
        output_provenance=provenance,
    )
    agents = [
        _agent(
            "node_3",
            "Produce the final response based on node_2's analysis.",
        )
    ]

    schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=agents,
        )
    )
    variants = _relation_array_variants(schema["properties"]["relations"])

    assert variants
    assert all(
        relations
        == (
            {
                "source_id": "node_2",
                "target_id": "node_3",
                "source_to_target": True,
                "target_to_source": False,
            },
        )
        for relations in variants
    )


def test_output_closure_rejects_dependency_outside_live_sink_ingress() -> None:
    provenance = {
        "mode": "required_new_terminal_consumer",
        "eligible_existing_agent_ids": [],
        "same_action_agents_eligible": True,
        "remaining_capacity": 1,
        "eligible_input_agent_ids": ["node_1"],
        "required_ingress_component_agent_ids": [["node_1"]],
        "required_ingress_count": 1,
    }
    domains = _domains(
        enabled=True,
        existing_agent_ids=("node_1", "node_2"),
        max_new_agents=1,
        min_relations=1,
        max_relations=1,
        output_provenance=provenance,
    )
    agents = [
        _agent(
            "node_3",
            "Produce the final response based on node_2's analysis.",
        )
    ]

    with pytest.raises(ValueError, match="cannot directly route"):
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=agents,
        )


def test_canvas_rejects_reverse_only_atomic_add_without_commit_or_execution() -> None:
    gateway = _RecordingGateway()
    env = _environment(gateway)
    revision = env.revision

    result = asyncio.run(
        env.step(_dependency_add_action(reciprocal=False, reverse=True))
    )

    assert result.accepted is False
    assert "does not route 'node_1' -> 'node_2'" in result.feedback
    assert env.revision == revision
    assert env.graph.nodes == ()
    assert gateway.requests == []


@pytest.mark.parametrize("reciprocal", [False, True])
def test_canvas_accepts_forward_or_reciprocal_declared_dependency(
    reciprocal: bool,
) -> None:
    gateway = _RecordingGateway()
    env = _environment(gateway)

    result = asyncio.run(
        env.step(_dependency_add_action(reciprocal=reciprocal))
    )

    assert result.accepted is True, result.feedback
    assert env.revision > 0
    assert gateway.requests


def test_set_relation_live_domain_preserves_existing_declared_path() -> None:
    gateway = _RecordingGateway()
    env = _environment(gateway)
    added = asyncio.run(
        env.step(_dependency_add_action(reciprocal=False))
    )
    assert added.accepted is True, added.feedback

    candidates = env.model_admissible_action_targets()["set_relation"][
        "candidates"
    ]
    pair_candidates = tuple(
        candidate
        for candidate in candidates
        if {candidate["source_id"], candidate["target_id"]}
        == {"node_1", "node_2"}
    )

    assert pair_candidates
    assert all(
        ("node_1", "node_2") in _directed_edges((candidate,))
        for candidate in pair_candidates
    )
