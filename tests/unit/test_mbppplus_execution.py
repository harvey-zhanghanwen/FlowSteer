from __future__ import annotations

import asyncio
import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentResponse, AgentRuntime, AgentRuntimeError
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import _director_neutral_state_projection
from src.interactive.mbppplus_execution import (
    MBPPPLUS_PYTHON_EXEC_ACTION,
    MBPPPLUS_PYTHON_EXEC_TOOL_ID,
    MBPPPlusReactExecutionAdapter,
    create_mbppplus_public_test_registry,
    extract_mbppplus_public_assertions,
    validate_mbppplus_tested_completion,
)
from src.interactive.react_execution import ReactExecutionError
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.tool_runtime import FakeTool, ToolRegistration, ToolRegistry


def structured_action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "resource_id": resource_id,
            "skill_id": None,
        }
    )


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return AgentResponse(
            self.outputs.pop(0),
            {"provider_request_id": len(self.requests)},
        )


class StepwiseAdapter:
    stepwise_director = True

    async def execute(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("validation must run before execution")


class SynchronizedFailureAdapter:
    def __init__(self) -> None:
        self.entered = 0
        self.ready = asyncio.Event()

    async def execute(self, request):  # type: ignore[no-untyped-def]
        self.entered += 1
        if self.entered == 2:
            self.ready.set()
        await self.ready.wait()
        raise ReactExecutionError(
            f"{request.agent.id} yielded one public action",
            react_trace=(
                {
                    "turn": 1,
                    "action_text": request.agent.id,
                    "observation_status": "schema_invalid",
                    "public_error_code": "test_event",
                },
            ),
        )


class MBPPPlusExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_director_uses_merged_react_state_without_duplicate_outbox(
        self,
    ) -> None:
        objective = "Write add_one and satisfy the public assertion."
        action = {
            "kind": "tool",
            "name": MBPPPLUS_PYTHON_EXEC_ACTION,
            "resource_id": MBPPPLUS_PYTHON_EXEC_TOOL_ID,
            "arguments": {"code": "def add_one(x):\n    return x + 1\n"},
        }
        projected = _director_neutral_state_projection(
            {
                "type": "AgentRuntimeError",
                "react_events": [
                    {
                        "agent_id": "solver",
                        "original_task_objective": objective,
                        "agent_contract": "Implement and test add_one.",
                        "final_objective": objective,
                        "execution_status": "awaiting_next_action",
                        "action": action,
                        "action_text": json.dumps(action),
                        "observation": {
                            "observation_status": "success",
                            "executed_action": action,
                            "result": {"ok": True},
                        },
                    }
                ],
                "react_state": {
                    "graph_revision": 3,
                    "pending_agent_ids": ["solver"],
                    "task_objective": objective,
                    "agents": [
                        {
                            "agent_id": "solver",
                            "graph_revision": 3,
                            "agent_contract": "Implement and test add_one.",
                            "final_objective": objective,
                            "execution_status": "awaiting_next_action",
                            "execution_semantics": "one_action_one_observation",
                            "react_turns_used": 1,
                            "tool_calls_used": 1,
                            "latest_action": action,
                            "latest_action_text": json.dumps(action),
                            "latest_observation": {
                                "observation_status": "success",
                                "executed_action": action,
                                "result": {"ok": True},
                            },
                        }
                    ],
                },
            }
        )
        self.assertNotIn("react_events", projected)
        state = projected["react_state"]
        self.assertEqual(objective, state["task_objective"])
        event = state["agents"][0]
        self.assertEqual(objective, event["final_objective"])
        self.assertEqual(action, event["latest_action"])
        self.assertNotIn("executed_action", event["latest_observation"])
        self.assertNotIn("latest_action_text", event)

    async def test_parallel_react_failures_are_all_awaited_and_recorded(
        self,
    ) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        graph = AgentGraph()
        graph.add_agent(
            AgentNode("left", "m", "Produce one action.", execution_mode="react")
        )
        graph.add_agent(
            AgentNode("right", "m", "Produce one action.", execution_mode="react")
        )
        graph.add_agent(AgentNode("sink", "m", "Combine both artifacts."))
        graph.set_relation("left", "sink", True, False)
        graph.set_relation("right", "sink", True, False)
        graph.set_output("sink")
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            timeout_seconds=5.0,
            execution_adapters={"react": SynchronizedFailureAdapter()},
        )
        with self.assertRaises(AgentRuntimeError) as raised:
            await runtime.execute(graph, "test parallel failure receipts")
        self.assertEqual(
            {"left", "right"},
            {record.agent_id for record in raised.exception.failure_records},
        )

    def test_stepwise_react_is_not_admitted_in_reciprocal_block(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
        )
        graph = AgentGraph()
        graph.add_agent(
            AgentNode(
                "tool_user",
                "m",
                "Use one public Tool action at a time.",
                execution_mode="react",
            )
        )
        graph.add_agent(AgentNode("peer", "m", "Review the artifact."))
        graph.set_relation("tool_user", "peer", True, True)
        with self.assertRaisesRegex(
            AgentRuntimeError,
            "cannot be endpoints of a bidirectional relation",
        ):
            runtime.validate_graph_execution_contracts(graph)

    def test_completed_sibling_does_not_trigger_pending_agent_repair(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
        )
        graph = AgentGraph()
        graph.add_agent(
            AgentNode("pending", "m", "Continue after one successful tool action.", execution_mode="react")
        )
        graph.add_agent(
            AgentNode("completed", "m", "Already produced its final artifact.", execution_mode="react")
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Return complete Python source.",
            execute_on_edit=True,
            allowed_actions=("modify_agent", "continue"),
        )
        canvas.reset("Return complete Python source.", graph)
        action = {
            "kind": "tool",
            "name": "public_test",
            "resource_id": "public-test",
            "arguments": {},
        }
        canvas._failure_continuations["pending"] = {
            "react_trace": [
                {
                    "turn": 1,
                    "structured_action": action,
                    "action_text": json.dumps(action),
                    "observation": {"observation_status": "success"},
                }
            ],
            "tool_receipts": [{}],
        }
        canvas._react_event_outbox.append(
            {
                "agent_id": "completed",
                "agent_contract": "Already produced its final artifact.",
                "final_objective": "Return complete Python source.",
                "execution_status": "completed",
                "react_turn": 2,
                "action": {"kind": "complete"},
                "action_text": '{"kind":"complete"}',
                "observation": {"observation_status": "completed"},
            }
        )
        state = canvas.public_react_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(
            {"pending", "completed"},
            {item["agent_id"] for item in state["agents"]},
        )
        self.assertEqual(("continue",), canvas.model_admissible_action_types())

    def test_new_reciprocal_phase_replaces_stale_active_continuation(self) -> None:
        self.assertTrue(
            AgentWorkflowEnv._replace_active_failure_continuation(
                {
                    "execution_phase": "draft",
                    "react_trace": [{"turn": 1}, {"turn": 2}],
                    "tool_receipts": [{"tool_id": "x"}],
                },
                {
                    "execution_phase": "revision",
                    "react_trace": [{"turn": 1}],
                    "tool_receipts": [{"tool_id": "x"}],
                },
            )
        )

    def test_extracts_only_public_assertions_from_prompt(self) -> None:
        problem = (
            "Write a function add_one(x).\n"
            "Your code should pass these tests:\n"
            "assert add_one(2) == 3\n"
        )
        self.assertEqual(
            ("assert add_one(2) == 3",),
            extract_mbppplus_public_assertions(problem),
        )

    async def test_each_react_action_returns_state_to_director_before_continue(
        self,
    ) -> None:
        problem = (
            "Write a function add_one(x).\n"
            "Your code should pass these tests:\n"
            "assert add_one(2) == 3\n"
        )
        source = "def add_one(x):\n    return x + 1\n"
        gateway = SequenceGateway(
            [
                structured_action(
                    "tool",
                    name=MBPPPLUS_PYTHON_EXEC_ACTION,
                    arguments={"code": source},
                    resource_id=MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                ),
                structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": source},
                    resource_id=None,
                ),
            ]
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        registered = create_mbppplus_public_test_registry(
            problem,
            timeout_seconds=5.0,
        )
        tools = ToolRegistry(
            (
                ToolRegistration(
                    MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                    FakeTool(
                        {
                            MBPPPLUS_PYTHON_EXEC_ACTION: lambda arguments: {
                                "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                                "ok": arguments["code"] == source,
                                "observation": "[OK] public assertion passed",
                                "public_assertion_count": 1,
                                "public_assertions": [
                                    "assert add_one(2) == 3"
                                ],
                                "interface_requirement": (
                                    "Preserve the entry point and positional "
                                    "argument order shown in the public assertions."
                                ),
                            }
                        }
                    ),
                    registered.require_capability(MBPPPLUS_PYTHON_EXEC_TOOL_ID),
                ),
            )
        )
        adapter = MBPPPlusReactExecutionAdapter(
            gateway=gateway,
            tool_registry=tools,
            max_turns=1,
            max_tool_calls=2,
            max_action_tokens=512,
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": adapter},
            tool_registry=tools,
            dataset_id="mbpp_plus",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem=problem,
            execute_on_edit=True,
            allowed_actions=(
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
            terminal_execution_validator=validate_mbppplus_tested_completion,
        )
        added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "solver",
                            "model_id": "m",
                            "contract": "Produce complete Python source for the task.",
                            "allowed_tools": [MBPPPLUS_PYTHON_EXEC_TOOL_ID],
                            "execution_mode": "react",
                            "artifact_type": "python_source",
                            "completion_condition": (
                                "Return the exact complete source that passed the "
                                "public test."
                            ),
                        }
                    ],
                    "relations": [],
                    "output_agent_id": "solver",
                }
            )
        )
        self.assertTrue(added.accepted)
        self.assertIsNone(added.execution)
        self.assertEqual(("continue",), canvas.model_admissible_action_types())
        state = canvas.public_react_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(problem.strip(), state["task_objective"])
        agent_state = state["agents"][0]
        self.assertEqual("awaiting_next_action", agent_state["execution_status"])
        self.assertEqual(
            "Return the exact complete source that passed the public test.",
            agent_state["final_objective"],
        )
        self.assertEqual(
            MBPPPLUS_PYTHON_EXEC_ACTION,
            agent_state["latest_action"]["name"],
        )
        self.assertTrue(
            agent_state["latest_observation"]["result"]["ok"]
        )
        self.assertIn("run_public_test", agent_state["latest_action_text"])
        self.assertIn("react_state", added.feedback)
        self.assertIn("react_events", added.feedback)
        first_event = canvas.public_react_events()[0]
        self.assertEqual(problem.strip(), first_event["original_task_objective"])
        self.assertEqual(
            "Return the exact complete source that passed the public test.",
            first_event["final_objective"],
        )
        self.assertEqual("awaiting_next_action", first_event["execution_status"])
        self.assertEqual(
            MBPPPLUS_PYTHON_EXEC_ACTION,
            first_event["action"]["name"],
        )
        self.assertTrue(first_event["observation"]["result"]["ok"])

        self.assertNotIn(("react", ()), runtime.registered_execution_profiles())
        self.assertIn(
            ("react", (MBPPPLUS_PYTHON_EXEC_TOOL_ID,)),
            runtime.registered_execution_profiles(),
        )

        revision = canvas.revision
        continued = await canvas.step(
            '{"action":"continue","agent_id":"solver"}'
        )
        self.assertTrue(continued.accepted)
        self.assertEqual(revision, canvas.revision)
        self.assertIsNotNone(continued.execution)
        assert continued.execution is not None
        self.assertEqual(source, continued.execution.final_answer)
        self.assertIsNone(
            validate_mbppplus_tested_completion(continued.execution)
        )
        completion_events = canvas.public_react_events()
        self.assertEqual(1, len(completion_events))
        self.assertEqual("completed", completion_events[0]["execution_status"])
        self.assertEqual("complete", completion_events[0]["action"]["kind"])
        self.assertEqual(2, len(gateway.requests))
        self.assertIn("finish", canvas.model_admissible_action_types())
        finished = await canvas.step('{"action":"finish"}')
        self.assertTrue(finished.done)


if __name__ == "__main__":
    unittest.main()
