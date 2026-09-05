"""Synthetic continuation regressions; no model, Tool, or grader calls."""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_PROMPT_VERSION,
    DIRECTOR_PROMPT_VERSION_V18,
    DIRECTOR_PROMPT_VERSION_V19,
    DIRECTOR_SYSTEM_PROMPT,
    DIRECTOR_SYSTEM_PROMPT_V16,
    DIRECTOR_SYSTEM_PROMPT_V18,
    DIRECTOR_SYSTEM_PROMPT_V19,
    DirectorResponse,
    decode_director_transcript,
    director_system_prompt_for_version,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("synthetic", kind="test")],
        [ModelSpec("model-a", "synthetic"), ModelSpec("model-b", "synthetic")],
    )


class _Gateway:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(f"Artifact from {request.agent.id}")


class _ScriptedDirector:
    def __init__(self, gateway: _Gateway, actions=()) -> None:
        self.gateway = gateway
        self.actions = list(actions)
        self.prompts = []
        self.calls_before_action = []

    async def propose(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.calls_before_action.append(len(self.gateway.requests))
        return DirectorResponse(json.dumps(self.actions.pop(0)))


def _agent(agent_id: str, model_id: str, contract: str) -> dict[str, str]:
    # No enumerated role, preselected medical workflow, or role-first codec.
    return {"agent_id": agent_id, "model_id": model_id, "contract": contract}


def _relation(source: str, target: str, reciprocal=False) -> dict[str, object]:
    return {
        "source_id": source,
        "target_id": target,
        "source_to_target": True,
        "target_to_source": reciprocal,
    }


class HealthBenchDirectorV19Tests(unittest.IsolatedAsyncioTestCase):
    def test_version_registration_preserves_historical_prompt_and_default(self):
        # Byte-level legacy text equality, not a digest/integrity audit.
        legacy_v18 = DIRECTOR_SYSTEM_PROMPT_V16 + (
            "\n\nKeep unresolved names, abbreviations, quantities, time points, "
            "and the requested answer slot verbatim in Agent contracts and "
            "Tool tasks. Do not expand or replace them unless the task or an "
            "observed artifact supplies the meaning."
        )
        self.assertEqual(legacy_v18.encode(), DIRECTOR_SYSTEM_PROMPT_V18.encode())
        self.assertEqual(
            legacy_v18.encode(),
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V18).encode(),
        )
        self.assertEqual(
            DIRECTOR_SYSTEM_PROMPT,
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION),
        )
        self.assertEqual(
            DIRECTOR_SYSTEM_PROMPT_V19,
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V19),
        )
        self.assertLess(len(DIRECTOR_SYSTEM_PROMPT_V19), len(legacy_v18))

    def test_v19_is_task_faithful_and_domain_neutral(self):
        prompt = DIRECTOR_SYSTEM_PROMPT_V19
        for required in (
            "do not answer a neighboring question",
            "Distinguish the user's request from unverified premises",
            "without suppressing material contradictions or risks",
            "finds no support does not establish",
            "do not discard valid evidence because another Agent disagrees",
            "Keep Agent roles, count, models and topology open",
            "execution_mode describes execution, not role",
            "finish_admissibility is present and admissible",
            "admissible_action_types and action_target_domains",
        ):
            self.assertIn(required, prompt)
        for prohibited in (
            "healthbench", "rubric", "doctor", "researcher", "verifier",
            "formatter", "must use three", "ground truth",
        ):
            self.assertNotIn(prohibited, prompt.casefold())

    def test_v19_reuses_v18_compact_history_without_losing_current_state(self):
        gateway = _Gateway()
        client = _ScriptedDirector(gateway)
        latest = "Canvas observation.\n\n" + json.dumps({
            "graph": {"agents": [{"id": "current"}]},
            "current_artifact_receipts": [{"content": "Latest source evidence"}],
        })
        messages = [
            {"role": "system", "content": "test system"},
            {"role": "user", "content": "Full original conversation and goal"},
            {"role": "user", "content": "Canvas observation.\n\n" + json.dumps({
                "graph": {"agents": [{"id": "past"}]},
                "current_artifact_receipts": [{"content": "Past evidence"}],
                "execution_feedback": "Measured execution status",
            })},
            {"role": "assistant", "content": '{"action":"finish"}'},
            {"role": "user", "content": latest},
        ]
        old = AgentGraphOrchestrator(
            _registry(), client, prompt_version=DIRECTOR_PROMPT_VERSION_V18,
        )._compact_historical_messages(messages)
        new = AgentGraphOrchestrator(
            _registry(), client, prompt_version=DIRECTOR_PROMPT_VERSION_V19,
        )._compact_historical_messages(messages)
        self.assertEqual(old, new)
        self.assertEqual(messages[1], new[1])
        self.assertEqual(messages[-1], new[-1])
        self.assertNotIn("Past evidence", new[2]["content"])
        self.assertIn("Measured execution status", new[2]["content"])

    async def test_single_free_contract_can_finish_without_named_roles(self):
        registry = _registry()
        gateway = _Gateway()
        client = _ScriptedDirector(gateway, [
            {"action": "add_subgraph", "agents": [
                _agent("response", "model-a", "Answer the original request completely."),
            ], "relations": [], "output_agent_id": "response"},
            {"action": "finish"},
        ])
        env = AgentWorkflowEnv(registry, gateway=gateway, execute_on_edit=True)
        result = await AgentGraphOrchestrator(
            registry, client, prompt_version=DIRECTOR_PROMPT_VERSION_V19,
            semantic_protocol="none", max_rounds=2,
        ).run(env, "Explain which criterion is needed, without inventing context.")
        self.assertTrue(result.explicit_finish)
        self.assertEqual("Artifact from response", result.final_answer)
        self.assertEqual([0, 1], client.calls_before_action)
        transcript = decode_director_transcript(client.prompts[0])
        self.assertEqual(DIRECTOR_SYSTEM_PROMPT_V19, transcript[0]["content"])

    async def test_reciprocal_functional_unit_executes_before_next_director_action(self):
        registry = _registry()
        gateway = _Gateway()
        client = _ScriptedDirector(gateway, [
            {"action": "add_subgraph", "agents": [
                _agent("left", "model-a", "Assess supplied context independently."),
                _agent("right", "model-b", "Assess alternative interpretations independently."),
            ], "relations": [_relation("left", "right", reciprocal=True)]},
            {"action": "add_subgraph", "agents": [
                _agent("response", "model-b", "Use both assessments to answer the original request."),
            ], "relations": [
                _relation("left", "response"), _relation("right", "response"),
            ], "output_agent_id": "response"},
            {"action": "finish"},
        ])
        env = AgentWorkflowEnv(registry, gateway=gateway, execute_on_edit=True)
        result = await AgentGraphOrchestrator(
            registry, client, prompt_version=DIRECTOR_PROMPT_VERSION_V19,
            semantic_protocol="none", max_rounds=3,
        ).run(env, "Which interpretation is supported by the provided context?")
        self.assertTrue(result.explicit_finish)
        self.assertEqual([0, 4, 5], client.calls_before_action)
        self.assertEqual(2, sum(request.peer_draft is not None for request in gateway.requests))
        output_request = gateway.requests[-1]
        self.assertTrue(output_request.is_output_agent)
        self.assertEqual(
            {"left", "right"},
            {message.source_agent_id for message in output_request.upstream},
        )
        self.assertEqual({"model-a", "model-b"}, {request.model.model_id for request in gateway.requests})
        self.assertEqual("Artifact from response", result.final_answer)


if __name__ == "__main__":
    unittest.main()
