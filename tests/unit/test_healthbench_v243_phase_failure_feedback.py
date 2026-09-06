"""Synthetic phase-rejection regression; no models, evaluator or API calls.

Reuse the project's native-receipt scripted client and declaration-domain
fixtures. The real v2.42 case rejected an Output closure dependency, not a
missing relation field; the capacity case below is a separate synthetic case.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    director_live_action_target_domains_json,
    director_model_admissible_sampling_json_schema_text_v3,
    director_model_admissible_schema_branch_v3,
)
from src.interactive.rollout_collector import (
    AgentGraphRolloutCollector,
    ReceiptValidationError,
    _validate_v3_hierarchical_action_receipt,
)
from tests.unit.test_declared_dependency_relations import _domains
from tests.unit.test_rollout_collector import (
    EVALUATOR_VERSION,
    POLICY_VERSION,
    ScriptedSGLangClient,
    _orchestrator,
    _registry,
    _task,
    _versions,
)


OUTPUT_CLOSURE_ERROR = (
    "add_subgraph Output closure cannot directly route declared dependencies "
    "[('node_1', 'node_4')]"
)
CAPACITY_ERROR = (
    "add_subgraph declared dependencies exceed the live relation capacity"
)


class NoExecutionGateway:
    async def generate(self, request):
        raise AssertionError("a rejected declaration must not execute an Agent")


class SeededEnvironment(AgentWorkflowEnv):
    def reset(self, problem, graph=None):
        graph = AgentGraph()
        for index in range(1, 4):
            graph.add_agent(
                AgentNode(f"node_{index}", "cheap-model", f"Existing contract {index}")
            )
        graph.set_relation("node_1", "node_2", True, False)
        graph.set_relation("node_1", "node_3", True, False)
        return super().reset(problem, graph)


class RecordingClient(ScriptedSGLangClient):
    def __init__(self, actions):
        self.responses = []
        self.prompts = []
        super().__init__(
            actions,
            policy_version=POLICY_VERSION,
            expected_server_weight_version="default",
        )

    async def propose(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = await super().propose(prompt, **kwargs)
        self.responses.append(response)
        return response


def _schema_request():
    domains = _domains(
        enabled=True,
        existing_agent_ids=("node_1", "node_2", "node_3"),
        max_new_agents=1,
        min_relations=2,
        max_relations=2,
        output_provenance={
            "mode": "required_new_terminal_consumer",
            "eligible_existing_agent_ids": [],
            "same_action_agents_eligible": True,
            "remaining_capacity": 5,
            "eligible_input_agent_ids": ["node_2", "node_3"],
            "required_ingress_component_agent_ids": [["node_2"], ["node_3"]],
            "required_ingress_count": 2,
        },
    )
    domain = domains["add_subgraph"]
    domain["model_ids"] = ["cheap-model"]
    for record in domain["existing_agents"] + domain["model_execution_profiles"]:
        record["model_id"] = "cheap-model"
    actions = ("add_subgraph",)
    return {
        "action_json_schema": director_model_admissible_sampling_json_schema_text_v3(
            actions
        ),
        "action_json_schema_version": DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
        "action_schema_branch": director_model_admissible_schema_branch_v3(actions),
        "action_target_domains_json": director_live_action_target_domains_json(
            actions, domains
        ),
        "action_target_domain_version": DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    }


def _declaration(contract):
    return json.dumps(
        {
            "action": "add_subgraph",
            "agents": [
                {
                    "agent_id": "node_4",
                    "model_id": "cheap-model",
                    "execution_mode": "reasoning",
                    "allowed_tools": [],
                    "contract": contract,
                }
            ],
        },
        separators=(",", ":"),
    ) + "<|endoftext|>"


def _collect(raw_declaration):
    profile = json.dumps(
        {
            "action": "add_subgraph",
            "agents": [
                {"agent_id": "node_4", "execution_mode": "reasoning", "allowed_tools": []}
            ],
        },
        separators=(",", ":"),
    )
    client = RecordingClient([profile, raw_declaration, profile, raw_declaration])
    registry = _registry()
    orchestrator = _orchestrator(registry, client, max_rounds=2)
    schema_request = _schema_request()
    orchestrator.action_schema_request = lambda _env: dict(schema_request)
    environment = SeededEnvironment(
        registry, gateway=NoExecutionGateway(), execute_on_edit=True
    )
    collector = AgentGraphRolloutCollector(orchestrator, environment, _versions())

    def evaluator(task, final_answer, final_graph, runtime):
        assert final_answer is None
        assert runtime is None
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 0.0,
            "metrics": {},
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))
    return trajectory, environment, client, schema_request


@pytest.mark.parametrize(
    ("raw_declaration", "expected_error"),
    [
        (
            _declaration("Synthesize findings from node_1, node_2, and node_3."),
            OUTPUT_CLOSURE_ERROR,
        ),
        (
            _declaration("Use findings from node_1; from node_2; from node_3."),
            CAPACITY_ERROR,
        ),
        ("not-json declaration", None),
    ],
    ids=["observed-output-closure", "synthetic-capacity", "malformed-json"],
)
def test_phase_rejection_reaches_next_prompt_without_executing_or_rewriting(
    raw_declaration, expected_error
):
    trajectory, environment, client, _ = _collect(raw_declaration)
    assert len(client.payloads) == 4  # Selection + declaration only, twice.
    assert len(client.actions) == 0
    assert len(trajectory.turns) == environment.turn_count == len(environment.history) == 2
    assert trajectory.termination_reason == "max_rounds"
    assert not trajectory.explicit_finish
    first = trajectory.turns[0]
    for turn, response, history in zip(
        trajectory.turns, client.responses, environment.history
    ):
        error = response.metadata["phase_failure_error"]
        assert error["phase"] == "add_agent_declarations"
        assert error["message"]
        if expected_error is not None:
            assert error == {
                "phase": "add_agent_declarations",
                "error_type": "ValueError",
                "message": expected_error,
            }
        decoding = turn.runtime_summary["director_action_decoding"]
        assert decoding["phase_failure_error"] == error
        assert decoding["parse_failure_phase"] == error["phase"]
        assert decoding["parameter_schema_branch"] is None
        assert decoding["request_count"] == 2
        assert set(decoding["phase_receipts"]) == {
            "add_agent_execution_profile_selection", "add_agent_declarations"
        }
        phase_receipt = decoding["phase_receipts"]["add_agent_declarations"]
        assert phase_receipt["text"] == turn.policy_response == raw_declaration
        assert tuple(phase_receipt["output_token_ids"]) == tuple(map(ord, raw_declaration))
        assert phase_receipt["behavior_log_probs"] == response.metadata["behavior_log_probs"]
        assert turn.receipt_verified and phase_receipt["receipt_verified"]
        assert turn.action == {} and history.action is None and not history.accepted
        assert turn.executed_prefix_tokens == 0 and turn.executions == ()
        assert turn.graph_revision == first.graph_revision == environment.graph.revision
        assert turn.graph_snapshot == first.graph_snapshot
        assert len(turn.graph_snapshot["nodes"]) == 3
        assert history.feedback == turn.canvas_feedback
        assert error["message"] in turn.canvas_feedback
        assert "missing action fields: relations" not in turn.canvas_feedback
    feedback = first.canvas_feedback
    # The actual next native Director request contains this Canvas observation,
    # not merely a private trajectory diagnostic. Newlines/quotes are JSON escaped.
    assert feedback in client.prompts[1] or json.dumps(feedback)[1:-1] in client.prompts[1]
    assert client.prompts[1] == client.responses[1].metadata["base_prompt_text"]


def test_error_receipt_matches_actual_validator_and_legacy_receipts_remain_readable():
    _, _, client, schema_request = _collect(
        _declaration("Synthesize findings from node_1, node_2, and node_3.")
    )
    metadata = dict(client.responses[0].metadata)
    expected_phases = {
        "add_agent_execution_profile_selection", "add_agent_declarations"
    }
    assert _validate_v3_hierarchical_action_receipt(None, metadata, schema_request) == expected_phases
    altered = dict(metadata)
    altered["phase_failure_error"] = {
        **metadata["phase_failure_error"], "message": "missing action fields: relations"
    }
    with pytest.raises(ReceiptValidationError, match="diagnosis differs"):
        _validate_v3_hierarchical_action_receipt(None, altered, schema_request)
    legacy = dict(metadata)
    legacy.pop("phase_failure_error")
    assert _validate_v3_hierarchical_action_receipt(None, legacy, schema_request) == expected_phases
