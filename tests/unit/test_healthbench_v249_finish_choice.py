"""Keep FINISH legal but optional; synthetic Canvas and real config validators."""

import asyncio
import json
from pathlib import Path

import pytest

from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
from scripts.healthbench_candidate_skill_profile import (
    build_candidate_prompt_priors, load_candidate_skill_profile,
)
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from tests.unit.test_agent_graph import make_registry


ROOT = Path(__file__).resolve().parents[2]
TASK = (
    "Translate this maintenance plan into French: 1. Disconnect the power. "
    "2. Clean the filter. 3. Reconnect the power."
)
CONTRACT = (
    "Translate every supplied maintenance instruction into French. "
    "Return the complete translated plan as the user-facing response."
)


def config(version, suffix="dev5"):
    return load_yaml(
        ROOT / f"config/evaluation_healthbench_professional_candidate_skill_v2_{version}_{suffix}.yaml",
        expand_env=False,
    )


@pytest.mark.parametrize("suffix", ["dev5", "full525"])
def test_configs_change_only_run_identity_paths_and_finish_choice(suffix):
    old, new = config(48, suffix), config(49, suffix)
    validate_agent_graph_config(new)
    validate_completion_benchmark_config(new)
    old_identity = f"healthbench_professional_candidate_skill_v2_48_{suffix}"
    new_identity = f"healthbench_professional_candidate_skill_v2_49_{suffix}"
    expected = json.loads(json.dumps(old).replace(old_identity, new_identity))
    expected["agent_graph"]["finish_only_when_admissible"] = False
    assert new == expected  # Includes selected IDs, seed, model pool and every timeout.
    assert old["agent_graph"]["finish_only_when_admissible"] is True
    assert new["agent_graph"]["semantic_protocol_by_source"]["healthbench_professional"] == "none"
    assert new["candidate_skill_evaluation"] == old["candidate_skill_evaluation"]
    assert new["candidate_skill_evaluation"]["profile_path"] == "config/healthbench_candidate_skills_v248.yaml"
    profile = load_candidate_skill_profile(
        ROOT / new["candidate_skill_evaluation"]["profile_path"], run_config=new,
    )
    priors = build_candidate_prompt_priors(profile)
    assert len(priors) == 3 and all(prior["rejectable"] for prior in priors)


class SyntheticGateway:
    def __init__(self, text):
        self.text = text
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return self.text


def environment(*, finish_only, text="A synthetic response.", graph=None):
    settings = config(49)["agent_graph"]
    gateway = SyntheticGateway(text)
    env = AgentWorkflowEnv(
        make_registry(), gateway, problem=TASK, graph=graph,
        execute_on_edit=True, max_agents=settings["max_agents"],
        max_agents_per_subgraph=settings["max_agents_per_subgraph"],
        max_relations_per_subgraph=settings["max_relations_per_subgraph"],
        allowed_actions=settings["actions"],
        semantic_protocol=settings["semantic_protocol_by_source"]["healthbench_professional"],
        recovery_policy=settings["recovery_policy"],
        require_output_protocol_artifact_for_set_output=settings["require_output_protocol_artifact_for_set_output"],
        finish_only_when_admissible=finish_only,
    )
    return env, gateway


async def add_output(env):
    result = await env.step(json.dumps({
        "action": "add_subgraph",
        "agents": [{"agent_id": "worker", "model_id": "balanced", "contract": CONTRACT}],
        "relations": [], "output_agent_id": "worker",
    }))
    assert result.accepted, result.feedback
    assert not result.done and not env.finished
    return result


@pytest.mark.parametrize("finish_only", [False, True])
@pytest.mark.parametrize("text", [
    "Voici le contexte du plan de maintenance.",  # Valid text is not proof of task coverage.
    "Débranchez l’alimentation. Nettoyez le filtre. Rebranchez l’alimentation.",
])
def test_admissible_output_keeps_edits_optional_and_old_true_compatible(finish_only, text):
    async def scenario():
        env, gateway = environment(finish_only=finish_only, text=text)
        result = await add_output(env)
        assert result.final_answer == text
        assert env.finish_admissibility()["admissible"] is True
        actions = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()
        if finish_only:
            assert actions == ("finish",)
            assert set(targets) == {"finish"}
        else:
            assert {"finish", "add_subgraph", "modify_agent"}.issubset(actions)
            assert {"finish", "add_subgraph", "modify_agent"}.issubset(targets)
        assert len(gateway.requests) == 1  # Inspecting the action mask performs no execution.
        finished = await env.step('{"action":"finish"}')
        assert finished.accepted and finished.done and env.finished
        assert finished.final_answer == text
        assert len(gateway.requests) == 1  # Optional FINISH does not require another Agent.

    asyncio.run(scenario())


@pytest.mark.parametrize("finish_only", [False, True])
@pytest.mark.parametrize("state", ["empty_graph", "selected_but_unexecuted_output"])
def test_without_a_valid_current_output_finish_is_not_admitted(finish_only, state):
    graph = None if state == "empty_graph" else AgentGraph(
        [AgentNode("worker", "balanced", CONTRACT)], output_agent_id="worker",
    )
    env, gateway = environment(finish_only=finish_only, graph=graph)
    assert env.finish_admissibility()["admissible"] is False
    assert "finish" not in env.model_admissible_action_types()
    assert "finish" not in env.model_admissible_action_targets()
    assert not env.finished and not gateway.requests
