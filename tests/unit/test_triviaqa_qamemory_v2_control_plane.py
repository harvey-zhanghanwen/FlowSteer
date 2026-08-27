from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import (
    AgentGraphOrchestrator,
    DirectorResponse,
    OpenAIDirectorClient,
    QA_DIRECTOR_PROMPT_VERSION,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    decode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


QA_MEMORY_TOOL_ID = "triviaqa.qa_memory"
RETRIEVAL_QUERY = "private rewritten query about a marine chronometer"
CANONICAL_ANSWER = "John Harrison"
PARAPHRASE_QUESTION = "Which clockmaker built a practical marine chronometer?"
PARAPHRASE_ANSWER = "The clockmaker was John Harrison."
TOOL_OBSERVATION = "John Harrison developed the marine chronometer."
RETRIEVAL_ARTIFACT = "Receipt-grounded evidence says John Harrison."


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("m", "fake")],
    )


def _tool_registry() -> ToolRegistry:
    capability = ToolCapability(
        tool_id=QA_MEMORY_TOOL_ID,
        dataset_scope=("triviaqa",),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        side_effect="read_only_retrieval",
        timeout_seconds=None,
        version="triviaqa-qa-memory-test-v1",
        action_schemas={
            "search": {"type": "object"},
            "read": {"type": "object"},
        },
    )
    backend = FakeTool(
        {
            "search": lambda arguments: dict(arguments),
            "read": lambda arguments: dict(arguments),
        }
    )
    return ToolRegistry(
        (ToolRegistration(QA_MEMORY_TOOL_ID, backend, capability),)
    )


def _tool_receipts() -> tuple[dict[str, object], ...]:
    return (
        {
            "tool_id": QA_MEMORY_TOOL_ID,
            "tool_version": "triviaqa-qa-memory-test-v1",
            "request": {
                "action": "search",
                "arguments": {"query": RETRIEVAL_QUERY, "top_k": 3},
            },
            "result": {
                "completed": True,
                "value": {
                    "hits": [
                        {
                            "memory_id": "memory-1",
                            "rank": 1,
                            "similarity": 0.91,
                            "paraphrase_question": PARAPHRASE_QUESTION,
                            "paraphrase_answer_statement": PARAPHRASE_ANSWER,
                            "canonical_answer": CANONICAL_ANSWER,
                        }
                    ]
                },
            },
            "error_type": None,
        },
        {
            "tool_id": QA_MEMORY_TOOL_ID,
            "tool_version": "triviaqa-qa-memory-test-v1",
            "request": {
                "action": "read",
                "arguments": {"memory_id": "memory-1"},
            },
            "result": {
                "completed": True,
                "value": {
                    "memory": {
                        "memory_id": "memory-1",
                        "paraphrase_question": PARAPHRASE_QUESTION,
                        "paraphrase_answer_statement": PARAPHRASE_ANSWER,
                        "canonical_answer": CANONICAL_ANSWER,
                        "text": TOOL_OBSERVATION,
                    }
                },
            },
            "error_type": None,
        },
    )


class _RetrievalAdapter:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            RETRIEVAL_ARTIFACT,
            {
                "tool_receipts": _tool_receipts(),
                "react_trace": (
                    {"turn": 1, "observation": TOOL_OBSERVATION},
                    {"turn": 2, "observation_status": "completed"},
                ),
            },
        )


class _OutputGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(f"<answer>{CANONICAL_ANSWER}</answer>")


class _UnusedDirector:
    async def propose(self, prompt: str, **_: object) -> DirectorResponse:
        raise AssertionError(f"unexpected Director generation: {prompt[:40]}")


class _CapturingDirectorClient(OpenAIDirectorClient):
    def __init__(self) -> None:
        super().__init__()
        self.provider_payload: dict[str, object] | None = None

    def _post(self, api_key: str, payload: object) -> dict[str, object]:
        del api_key
        assert isinstance(payload, dict)
        self.provider_payload = dict(payload)
        return {
            "id": "director-test",
            "model": "supervisor_theta",
            "choices": [{"message": {"content": '{"action":"finish"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def _graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "retriever",
                "m",
                "retrieve QA-memory evidence",
                role_family="evidence_retriever",
                allowed_tools=(QA_MEMORY_TOOL_ID,),
                execution_mode="react",
                artifact_type="retrieval_evidence",
            ),
            AgentNode(
                "output",
                "m",
                "format one routed answer",
                role_family="format",
                artifact_type="answer_wrapper",
            ),
        ],
        [AgentRelation("retriever", "output", True, False)],
        output_agent_id="output",
    )


def _observation_from_prompt(prompt: str) -> dict[str, object]:
    messages = decode_director_transcript(prompt)
    if messages is None:
        raise AssertionError("expected canonical Director transcript")
    _, separator, encoded = messages[-1]["content"].partition("\n\n")
    if not separator:
        raise AssertionError("missing Canvas observation payload")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise AssertionError("Canvas observation must be an object")
    return value


class TriviaQAQAMemoryV2ControlPlaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_director_is_toolless_and_worker_receipts_remain_lossless(
        self,
    ) -> None:
        registry = _registry()
        tool_registry = _tool_registry()
        output_gateway = _OutputGateway()
        retrieval_adapter = _RetrievalAdapter()
        runtime = AgentRuntime(
            registry,
            output_gateway,
            execution_adapters={"react": retrieval_adapter},
            tool_registry=tool_registry,
            dataset_id="triviaqa",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Who invented the marine chronometer?",
            graph=_graph(),
            execute_on_edit=True,
            required_evidence_tool_id=QA_MEMORY_TOOL_ID,
            director_feedback_mode="control_plane",
        )

        step = await env.step(
            '{"action":"modify_agent","agent_id":"retriever",'
            '"contract":"retrieve public QA-memory evidence"}'
        )
        self.assertTrue(step.accepted)
        self.assertIsNotNone(step.execution)
        assert step.execution is not None

        orchestrator = AgentGraphOrchestrator(
            registry,
            _UnusedDirector(),
            tool_registry=tool_registry,
        )
        prompt = orchestrator.build_prompt(env, 0, ())
        observation = _observation_from_prompt(prompt)
        self.assertEqual(
            {"allowed_tools": [], "tool_calls_enabled": False},
            observation["director_execution_profile"],
        )
        self.assertEqual(
            [
                {
                    "tool_id": QA_MEMORY_TOOL_ID,
                    "action_names": ["read", "search"],
                    "availability": True,
                }
            ],
            observation["tool_catalog"],
        )
        encoded_observation = json.dumps(observation, ensure_ascii=False)
        for private_value in (
            RETRIEVAL_QUERY,
            CANONICAL_ANSWER,
            PARAPHRASE_QUESTION,
            PARAPHRASE_ANSWER,
            TOOL_OBSERVATION,
            RETRIEVAL_ARTIFACT,
        ):
            self.assertNotIn(private_value, encoded_observation)

        # The worker remains the Tool principal and the explicit graph edge
        # carries its receipt-bearing artifact to the Output request.
        self.assertEqual(1, len(retrieval_adapter.requests))
        worker_request = retrieval_adapter.requests[0]
        self.assertEqual("retriever", worker_request.agent.id)
        self.assertEqual((QA_MEMORY_TOOL_ID,), worker_request.agent.allowed_tools)
        self.assertEqual(1, len(output_gateway.requests))
        routed = output_gateway.requests[0].upstream[0]
        self.assertEqual("retriever", routed.source_agent_id)
        self.assertEqual("output", routed.target_agent_id)
        self.assertEqual(RETRIEVAL_ARTIFACT, routed.content)
        self.assertEqual(2, len(routed.tool_receipts))

        runtime_receipts = step.execution.output_metadata["retriever"][
            "tool_receipts"
        ]
        self.assertEqual(
            RETRIEVAL_QUERY,
            runtime_receipts[0]["request"]["arguments"]["query"],
        )
        worker_call = next(
            call
            for call in step.execution.calls
            if call.request.agent.id == "retriever"
        )
        trajectory_record = execution_record_from_call(worker_call)
        self.assertEqual(
            CANONICAL_ANSWER,
            trajectory_record.metadata["response"]["tool_receipts"][1][
                "result"
            ]["value"]["memory"]["canonical_answer"],
        )

        # The OpenAI-compatible local Director request itself has no Tool
        # binding; the empty allowed_tools declaration is observation state.
        client = _CapturingDirectorClient()
        await client.propose(prompt)
        assert client.provider_payload is not None
        self.assertNotIn("tools", client.provider_payload)
        self.assertNotIn("tool_choice", client.provider_payload)

    def test_failure_receipt_is_typed_but_content_free(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            _OutputGateway(),
            execution_adapters={"react": _RetrievalAdapter()},
            tool_registry=_tool_registry(),
            dataset_id="triviaqa",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Who invented the marine chronometer?",
            graph=_graph(),
            required_evidence_tool_id=QA_MEMORY_TOOL_ID,
            director_feedback_mode="control_plane",
        )
        failure = AgentFailureRecord(
            request_id="request-1",
            agent_id="retriever",
            phase=ExecutionPhase.SINGLE,
            graph_revision=env.graph.revision,
            error_type="ReactExecutionError",
            message=f"failed after observing {CANONICAL_ANSWER}",
            metadata={
                "tool_receipts": _tool_receipts(),
                "react_trace": (
                    {
                        "observation_status": "failed",
                        "public_error_code": "tool_plan_exhausted",
                        "observation": TOOL_OBSERVATION,
                        "repair_instruction": PARAPHRASE_ANSWER,
                    },
                ),
            },
        )

        feedback = env._execution_error_feedback(
            AgentRuntimeError(
                f"runtime failed on {RETRIEVAL_QUERY}",
                failure_records=(failure,),
                pending_agent_ids=("retriever",),
            )
        )

        self.assertIn("agent_runtime_execution_failed", feedback)
        self.assertIn(QA_MEMORY_TOOL_ID, feedback)
        self.assertIn('"actions":["read","search"]', feedback)
        for private_value in (
            RETRIEVAL_QUERY,
            CANONICAL_ANSWER,
            PARAPHRASE_ANSWER,
            TOOL_OBSERVATION,
        ):
            self.assertNotIn(private_value, feedback)

    def test_triviaqa_semantic_lineage_admits_dynamic_qamemory_tool(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            _OutputGateway(),
            execution_adapters={"react": _RetrievalAdapter()},
            tool_registry=_tool_registry(),
            dataset_id="triviaqa",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Who invented the marine chronometer?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id=QA_MEMORY_TOOL_ID,
            director_feedback_mode="control_plane",
        )
        orchestrator = AgentGraphOrchestrator(
            registry,
            _UnusedDirector(),
            tool_registry=runtime.tool_registry,
            prompt_version=QA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy="preserve_diagnose_repair_augment",
        )

        observation = _observation_from_prompt(
            orchestrator.build_prompt(env, 0, ())
        )

        self.assertEqual(
            QA_MEMORY_TOOL_ID,
            observation["terminal_constraints"]["required_evidence_tool_id"],
        )
        add_domains = observation["action_target_domains"].get(
            "add_subgraph", {}
        )
        self.assertIn(QA_MEMORY_TOOL_ID, json.dumps(add_domains))
        role_constraints = add_domains["role_constraints"]
        self.assertEqual(
            [{"execution_mode": "reasoning", "allowed_tools": []}],
            role_constraints["reasoner"]["execution_profiles"],
        )
        self.assertEqual(
            ["react"],
            role_constraints["evidence_retriever"]["execution_modes"],
        )
        self.assertEqual(
            [[QA_MEMORY_TOOL_ID]],
            role_constraints["evidence_retriever"]["allowed_tools"],
        )

    def test_config_requires_control_plane_for_triviaqa_qamemory(self) -> None:
        config = load_yaml(
            "config/evaluation_triviaqa_unified_architecture_v2_fixed128.yaml"
        )
        config["agent_graph"]["required_evidence_tool_id"] = QA_MEMORY_TOOL_ID
        config["agent_graph"]["director_feedback_mode"] = "control_plane"
        validate_agent_graph_config(config)

        config["agent_graph"]["director_feedback_mode"] = "artifact_preview"
        with self.assertRaisesRegex(ConfigurationError, "control_plane"):
            validate_agent_graph_config(config)


if __name__ == "__main__":
    unittest.main()
