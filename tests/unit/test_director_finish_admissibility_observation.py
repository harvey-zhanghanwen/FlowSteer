from __future__ import annotations

import json
import unittest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    DIRECTOR_PROMPT_VERSION_V15,
    decode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


class _ImmediateGateway:
    async def generate(self, request):
        return AgentResponse("Complete assistant response")


class _UnusedDirector:
    async def propose(self, prompt, **kwargs):  # pragma: no cover - never sampled
        raise AssertionError("this observation test must not sample the Director")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", kind="test")],
        [ModelSpec("model", "provider")],
    )


def _latest_observation(prompt: str) -> dict[str, object]:
    messages = decode_director_transcript(prompt)
    if messages is None:
        raise AssertionError("expected a canonical Director transcript")
    content = messages[-1]["content"]
    heading, separator, raw_payload = content.partition("\n\n")
    if not separator or not heading.startswith("Canvas observation."):
        raise AssertionError("latest message is not a Canvas observation")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise AssertionError("Canvas observation payload must be an object")
    return payload


class NeutralDirectorFinishAdmissibilityObservationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_v15_observes_false_gate_then_only_finish_when_admissible(
        self,
    ) -> None:
        registry = _registry()
        env = AgentWorkflowEnv(
            registry,
            _ImmediateGateway(),
            problem="A conversation requiring a complete assistant response",
            execute_on_edit=True,
            finish_only_when_admissible=True,
        )
        orchestrator = AgentGraphOrchestrator(
            registry,
            _UnusedDirector(),
            prompt_version=DIRECTOR_PROMPT_VERSION_V15,
            semantic_protocol="none",
            sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
        )

        initial = _latest_observation(orchestrator.build_prompt(env, 0, ()))
        initial_gate = initial["finish_admissibility"]
        self.assertFalse(initial_gate["admissible"])
        self.assertEqual("graph_validation", initial_gate["stage"])
        self.assertTrue(initial_gate["reason"])

        added = await env.step(
            '{"action":"add_agent","agent_id":"answer","model_id":"model",'
            '"contract":"Produce the complete assistant response."}'
        )
        self.assertTrue(added.accepted, added.feedback)
        selected = await env.step(
            '{"action":"set_output","agent_id":"answer"}'
        )
        self.assertTrue(selected.accepted, selected.feedback)

        final = _latest_observation(orchestrator.build_prompt(env, 0, ()))
        self.assertEqual(
            {
                "admissible": True,
                "graph_revision": env.revision,
                "submission_semantics": "explicit_finish",
            },
            final["finish_admissibility"],
        )
        self.assertEqual(["finish"], final["admissible_action_types"])


if __name__ == "__main__":
    unittest.main()
