from __future__ import annotations

import unittest
from dataclasses import replace

from src.interactive.agent_runtime import UpstreamMessage
from src.interactive.environment_execution import build_environment_execution_resources
from src.interactive.openai_gateway import build_agent_messages
from test_environment_execution import (
    FakeSession,
    SequenceGateway,
    make_request,
)


class CompactEnvironmentFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def _execute(self, compact: bool | None):
        session = FakeSession()
        gateway = SequenceGateway(["look"])
        options = {} if compact is None else {"compact_execution_feedback": compact}
        environment = build_environment_execution_resources(
            gateway=gateway,
            session_factory=lambda _request: session,
            task_family="alfworld",
            max_turns=2,
            stepwise_director=True,
            alfworld_policy_action_profile="skillflow_public_invariants_v2",
            **options,
        )
        request = replace(
            make_request("alfworld", tool_id="alfworld.environment"),
            upstream=(
                UpstreamMessage(
                    "analysis-a",
                    "actor",
                    "A marked object remains beside the open receptacle.",
                    graph_revision=3,
                    environment_revision=0,
                    artifact_version="analysis-a:3",
                    request_or_dependency="Assess the publicly visible object.",
                ),
                UpstreamMessage(
                    "analysis-b",
                    "actor",
                    "The previous action did not establish a new location.",
                    graph_revision=3,
                    environment_revision=0,
                    artifact_version="analysis-b:3",
                    request_or_dependency="Check the current public observation.",
                ),
            ),
        )
        try:
            response = await environment.execution_adapter.execute(request)
        finally:
            environment.close()
        return request, gateway.requests[0], response

    async def test_compact_routes_each_complete_artifact_once_with_provenance(self):
        original, generated, _response = await self._execute(True)
        self.assertEqual(original.upstream, generated.upstream)
        self.assertNotIn("[ROUTED UPSTREAM AGENT ARTIFACTS]", generated.problem)
        rendered = "\n".join(item["content"] for item in build_agent_messages(generated))
        for message in original.upstream:
            self.assertEqual(1, rendered.count(message.content))
            self.assertIn(f"source_agent: {message.source_agent_id}", rendered)
            self.assertIn(f"target_agent: {message.target_agent_id}", rendered)
            self.assertIn(f"artifact_id: {message.artifact_id}", rendered)
            self.assertIn("graph_revision: 3", rendered)
            self.assertIn("environment_revision: 0", rendered)
            self.assertIn(message.request_or_dependency, rendered)

    async def test_default_and_explicit_false_preserve_previous_prompt(self):
        original, default_request, _ = await self._execute(None)
        _, legacy_request, _ = await self._execute(False)
        self.assertEqual(default_request.problem, legacy_request.problem)
        self.assertIn("[ROUTED UPSTREAM AGENT ARTIFACTS]", default_request.problem)
        rendered = "\n".join(
            item["content"] for item in build_agent_messages(default_request)
        )
        for message in original.upstream:
            self.assertEqual(2, rendered.count(message.content))

    async def test_compact_preserves_complete_latest_state_and_action_domain(self):
        _, legacy_request, legacy = await self._execute(False)
        _, compact_request, compact = await self._execute(True)
        self.assertEqual(legacy_request.model.metadata, compact_request.model.metadata)
        state = compact.metadata["environment_current_state"]
        previous_state = legacy.metadata["environment_current_state"]
        for field in (
            "environment_revision",
            "last_action",
            "current_observation",
            "admissible_actions",
            "policy_action_domain",
            "goal_progress",
            "held_objects",
            "remaining_action_budget",
            "total_action_budget",
            "environment_terminal",
            "latest_action_observation",
            "action_observation_history",
        ):
            self.assertEqual(previous_state[field], state[field], field)
        latest = state["latest_action_observation"]
        self.assertEqual("complete the alfworld task", latest["task_instruction"])
        self.assertEqual("look", latest["action"])
        self.assertEqual("room zero", latest["observation"])
        self.assertEqual("room one", latest["next_observation"])
        self.assertEqual("room one", latest["observation_result"])
        self.assertFalse(latest["observation_result_clipped"])
        self.assertEqual(["finish"], latest["next_admissible_actions"])
        self.assertEqual(0, latest["environment_revision_before"])
        self.assertEqual(1, latest["environment_revision_after"])
        self.assertEqual(1, latest["remaining_action_budget"])
        self.assertEqual(2, latest["total_action_budget"])
        self.assertEqual("success", latest["observation_status"])
        self.assertEqual(1, len(compact.metadata["environment_receipts"]))

    def test_compact_feedback_requires_boolean(self):
        with self.assertRaisesRegex(TypeError, "compact_execution_feedback must be bool"):
            build_environment_execution_resources(
                gateway=SequenceGateway(["look"]),
                session_factory=lambda _request: FakeSession(),
                task_family="alfworld",
                max_turns=2,
                compact_execution_feedback="true",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
