"""Scope admission on a real nonempty Canvas; no models or medical answers."""

import asyncio
import json

import pytest

from src.interactive.agent_workflow_env import AgentWorkflowEnv
from tests.unit.test_scope_neutral_contract_admission import _add_contract, _registry


@pytest.mark.parametrize("action", ["add_subgraph", "modify_agent"])
@pytest.mark.parametrize("contract,accepted", [
    ("Review node 1's evidence about ABC and report uncertainty.", True),
    ("Review node 1’s evidence: (1) identify ABC (2) report uncertainty.", True),
    ("Review node 47's evidence about ABC.", False),
    ("Review node 1's evidence and use dose 47 mg for ABC.", False),
])
def test_scope_admission_uses_ids_of_existing_agent_nodes(action, contract, accepted):
    env = AgentWorkflowEnv(
        _registry(), gateway=object(), problem="Explain ABC findings.",
        require_scope_neutral_contracts=True,
    )
    first = asyncio.run(env.step(_add_contract("Retrieve evidence about ABC.")))
    assert first.accepted, first.feedback
    original = env.graph.get_node("node_1")
    assert original.id == "node_1"  # nodes is not a mapping of ID strings.
    node_id = "node_2" if action == "add_subgraph" else "node_1"
    request = (
        _add_contract(contract, agent_id=node_id)
        if action == "add_subgraph"
        else json.dumps({"action": action, "agent_id": node_id, "contract": contract})
    )
    result = asyncio.run(env.step(request))
    assert result.accepted is accepted, result.feedback
    if accepted:
        assert env.graph.get_node(node_id).contract == contract
    else:
        assert "unsupported answer/clinical literals" in result.feedback
        assert env.graph.nodes == (original,)
