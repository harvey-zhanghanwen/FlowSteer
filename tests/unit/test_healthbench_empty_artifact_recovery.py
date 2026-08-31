from __future__ import annotations

import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("m1", "fake"), ModelSpec("m2", "fake")],
    )


def producer_consumer_graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode("source", "m1", "provide producer context"),
            AgentNode("producer", "m1", "produce the intermediate artifact"),
            AgentNode("consumer", "m2", "consume the intermediate artifact"),
        ],
        [
            AgentRelation("source", "producer", True, False),
            AgentRelation("producer", "consumer", True, False),
        ],
        output_agent_id="consumer",
    )


class EmptyArtifactGateway:
    def __init__(self, producer_text: str) -> None:
        self.producer_text = producer_text
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if request.agent.id == "source":
            return AgentResponse("source context", {"finish_reason": "stop"})
        if request.agent.id == "producer":
            return AgentResponse(
                self.producer_text,
                {"finish_reason": "length"},
            )
        return AgentResponse(
            f"consumer received {request.upstream[0].content}",
            {"finish_reason": "stop"},
        )


class RepairableEmptyArtifactGateway(EmptyArtifactGateway):
    """Publish one empty producer completion, then a valid repair Artifact."""

    def __init__(self) -> None:
        super().__init__("")
        self.producer_calls = 0

    async def generate(self, request: AgentRequest) -> AgentResponse:
        if request.agent.id == "producer":
            self.requests.append(request)
            self.producer_calls += 1
            if self.producer_calls == 1:
                return AgentResponse("", {"finish_reason": "length"})
            return AgentResponse(
                "usable producer artifact after repair",
                {"finish_reason": "stop"},
            )
        return await super().generate(request)


class HealthBenchEmptyArtifactRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _failed_environment(
        self,
    ) -> tuple[AgentWorkflowEnv, RepairableEmptyArtifactGateway]:
        catalog = registry()
        gateway = RepairableEmptyArtifactGateway()
        runtime = AgentRuntime(catalog, gateway)
        graph = producer_consumer_graph()
        environment = AgentWorkflowEnv(
            catalog,
            runtime=runtime,
            graph=graph,
            problem="question",
            execute_on_edit=True,
            max_agents=4,
            allowed_actions=(
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            ),
            recovery_policy="preserve_diagnose_repair_augment",
        )

        initial = await environment.step(
            '{"action":"modify_agent","agent_id":"source",'
            '"contract":"provide the producer context artifact"}'
        )
        self.assertTrue(initial.accepted, initial.feedback)
        self.assertEqual(1, gateway.producer_calls)
        self.assertEqual(1, len(initial.execution_failure_records))
        self.assertEqual(
            "CompletionArtifactEmpty",
            initial.execution_failure_records[0].error_type,
        )
        self.assertEqual(("producer",), environment._mandatory_repair_agent_ids())
        return environment, gateway

    async def test_empty_completion_is_a_producer_scoped_typed_failure(self) -> None:
        for empty_text in ("", " \n\t "):
            with self.subTest(empty_text=repr(empty_text)):
                catalog = registry()
                gateway = EmptyArtifactGateway(empty_text)
                runtime = AgentRuntime(catalog, gateway)
                graph = producer_consumer_graph()

                with self.assertRaises(AgentRuntimeError) as raised:
                    await runtime.execute(graph, "question")

                failure = raised.exception
                self.assertEqual(
                    ["source", "producer"],
                    [item.agent.id for item in gateway.requests],
                )
                self.assertEqual(
                    ("producer",),
                    tuple(
                        record.agent_id for record in failure.failure_records
                    ),
                )
                record = failure.failure_records[0]
                self.assertIsInstance(record, AgentFailureRecord)
                self.assertEqual("CompletionArtifactEmpty", record.error_type)
                self.assertEqual(
                    "completion_artifact_empty",
                    record.metadata["public_error_code"],
                )
                self.assertIs(False, record.metadata["artifact_complete"])
                self.assertEqual(0, record.metadata["response_text_characters"])
                self.assertEqual("length", record.metadata["finish_reason"])
                self.assertEqual(
                    {"source"},
                    set(record.metadata["input_artifact_versions"]),
                )
                provenance = record.metadata["input_artifact_provenance"]
                self.assertEqual(1, len(provenance))
                self.assertEqual("source", provenance[0]["source_agent_id"])
                self.assertEqual("producer", provenance[0]["target_agent_id"])
                self.assertEqual("source context", provenance[0]["content"])
                self.assertEqual(
                    record.metadata["input_artifact_versions"]["source"],
                    provenance[0]["artifact_version"],
                )

                partial = failure.partial_result
                self.assertIsNotNone(partial)
                assert partial is not None
                self.assertEqual({"source": "source context"}, dict(partial.outputs))
                self.assertEqual({"source"}, set(partial.output_metadata))
                self.assertEqual(2, len(partial.calls))
                producer_call = next(
                    call
                    for call in partial.calls
                    if call.request.agent.id == "producer"
                )
                self.assertEqual(empty_text, producer_call.response.text)
                self.assertIn("producer", failure.pending_agent_ids)
                self.assertIn("consumer", failure.pending_agent_ids)
                self.assertEqual(("consumer",), failure.blocked_agent_ids)

    async def test_empty_producer_remains_the_mandatory_repair_target(self) -> None:
        catalog = registry()
        gateway = EmptyArtifactGateway("")
        runtime = AgentRuntime(catalog, gateway)
        graph = producer_consumer_graph()

        with self.assertRaises(AgentRuntimeError) as raised:
            await runtime.execute(graph, "question")

        failure = raised.exception
        environment = AgentWorkflowEnv(
            catalog,
            runtime=runtime,
            graph=graph,
            problem="question",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        environment._record_failure_state(
            failure.failure_records,
            current_agent_ids={node.id for node in graph.nodes},
        )

        self.assertEqual(("producer",), environment._mandatory_repair_agent_ids())
        self.assertNotIn("consumer", environment._mandatory_repair_agent_ids())

    async def test_raw_non_modify_actions_cannot_bypass_empty_producer_repair(
        self,
    ) -> None:
        raw_actions = {
            "add_subgraph": (
                '{"action":"add_subgraph","agents":[{'
                '"agent_id":"helper","model_id":"m1",'
                '"contract":"produce a distinct repair artifact",'
                '"execution_mode":"reasoning","allowed_tools":[]}],'
                '"relations":[]}'
            ),
            "delete_agent": '{"action":"delete_agent","agent_id":"source"}',
            "set_relation": (
                '{"action":"set_relation","source_id":"source",'
                '"target_id":"consumer","source_to_target":true,'
                '"target_to_source":false}'
            ),
            "set_output": '{"action":"set_output","agent_id":"source"}',
            "finish": '{"action":"finish"}',
        }
        for action_name, action_text in raw_actions.items():
            with self.subTest(action=action_name):
                environment, gateway = await self._failed_environment()
                revision = environment.revision

                rejected = await environment.step(action_text)

                self.assertFalse(rejected.accepted, rejected.feedback)
                self.assertEqual(revision, environment.revision)
                self.assertEqual(1, gateway.producer_calls)
                self.assertIn("mandatory_repair_agent_ids=['producer']", rejected.feedback)
                self.assertEqual(
                    ("producer",),
                    environment._mandatory_repair_agent_ids(),
                )

    async def test_only_responsible_producer_modify_repairs_then_continues(
        self,
    ) -> None:
        environment, gateway = await self._failed_environment()
        self.assertEqual(("modify_agent",), environment.model_admissible_action_types())
        self.assertEqual(
            ["producer"],
            environment.model_admissible_action_targets()["modify_agent"][
                "agent_ids"
            ],
        )

        blocked_consumer = await environment.step(
            '{"action":"modify_agent","agent_id":"consumer",'
            '"contract":"attempt to replace the blocked consumer"}'
        )
        self.assertFalse(blocked_consumer.accepted, blocked_consumer.feedback)
        self.assertEqual(1, gateway.producer_calls)
        self.assertIn(
            "mandatory_repair_agent_ids=['producer']",
            blocked_consumer.feedback,
        )

        repaired = await environment.step(
            '{"action":"modify_agent","agent_id":"producer",'
            '"contract":"produce one non-empty intermediate artifact"}'
        )
        self.assertTrue(repaired.accepted, repaired.feedback)
        self.assertEqual(2, gateway.producer_calls)
        self.assertEqual((), repaired.execution_failure_records)
        self.assertEqual((), environment._mandatory_repair_agent_ids())
        self.assertEqual(
            "usable producer artifact after repair",
            environment._progressive_outputs["producer"],
        )
        self.assertEqual(
            "consumer received usable producer artifact after repair",
            environment._progressive_outputs["consumer"],
        )

        finished = await environment.step('{"action":"finish"}')
        self.assertTrue(finished.accepted, finished.feedback)
        self.assertTrue(finished.done)

    async def test_nonempty_completion_behavior_is_unchanged(self) -> None:
        catalog = registry()
        gateway = EmptyArtifactGateway("usable producer artifact")

        result = await AgentRuntime(catalog, gateway).execute(
            producer_consumer_graph(),
            "question",
        )

        self.assertEqual(
            {
                "source": "source context",
                "producer": "usable producer artifact",
                "consumer": "consumer received usable producer artifact",
            },
            dict(result.outputs),
        )
        self.assertEqual(
            "consumer received usable producer artifact",
            result.final_answer,
        )
        self.assertEqual(
            ["source", "producer", "consumer"],
            [item.agent.id for item in gateway.requests],
        )
        self.assertEqual(3, len(result.calls))


if __name__ == "__main__":
    unittest.main()
