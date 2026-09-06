from __future__ import annotations

import asyncio
import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_runtime import (
    AgentFailureRecord,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    ExecutionPhase,
)
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


class RecordingSuccessAdapter:
    def __init__(self) -> None:
        self.agent_ids: list[str] = []

    async def execute(self, request):  # type: ignore[no-untyped-def]
        self.agent_ids.append(request.agent.id)
        return AgentResponse(
            f"artifact from {request.agent.id}",
            {"provider_request_id": request.agent.id},
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

    async def test_continue_dirty_closure_does_not_retry_unselected_missing_agent(
        self,
    ) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        graph = AgentGraph(
            [
                AgentNode("selected", "m", "advance one action", execution_mode="react"),
                AgentNode("other", "m", "remain failed", execution_mode="react"),
            ]
        )
        adapter = RecordingSuccessAdapter()
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": adapter},
        )

        result = await runtime.execute(
            graph,
            "advance only the selected stepwise Agent",
            require_complete=False,
            dirty_agents={"selected"},
            include_missing_outputs_in_dirty_closure=False,
        )

        self.assertEqual(["selected"], adapter.agent_ids)
        self.assertEqual(("selected",), result.executed_agent_ids)
        self.assertEqual({"selected"}, set(result.outputs))

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

    def test_cancelled_budget_exhaustion_is_not_resumable(self) -> None:
        record = AgentFailureRecord(
            request_id="cancelled-after-budget",
            agent_id="solver",
            phase=ExecutionPhase.SINGLE,
            graph_revision=1,
            error_type="CancelledError",
            message="cancelled after partial public execution",
            metadata={
                "react_trace": [
                    {
                        "turn": 5,
                        "observation_status": "budget_exhausted",
                        "public_error_code": "tool_call_budget_exhausted",
                    }
                ]
            },
        )

        continuation = AgentWorkflowEnv._failure_continuation_candidate(record)

        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertIs(True, continuation["tool_plan_exhausted"])

    def test_successful_pending_react_agent_is_the_only_continue_target(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
        )
        graph = AgentGraph(
            [
                AgentNode("tested", "m", "complete tested source", execution_mode="react"),
                AgentNode("failed", "m", "repair source", execution_mode="react"),
            ]
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=("modify_agent", "continue"),
        )
        canvas._failure_continuations.update(
            {
                "tested": {
                    "react_trace": [
                        {
                            "turn": 1,
                            "observation": {"observation_status": "success"},
                        }
                    ]
                },
                "failed": {
                    "react_trace": [
                        {
                            "turn": 1,
                            "observation": {"observation_status": "tool_error"},
                        }
                    ]
                },
            }
        )

        self.assertEqual(("continue",), canvas.model_admissible_action_types())
        self.assertEqual(
            ["tested"],
            canvas.model_admissible_action_targets()["continue"]["agent_ids"],
        )

    def test_tested_source_is_selected_and_repair_exhausted_branch_is_deletable(
        self,
    ) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
            dataset_id="mbpp_plus",
        )
        graph = AgentGraph(
            [
                AgentNode("tested", "m", "produce source", execution_mode="react"),
                AgentNode("failed", "m", "check source", execution_mode="react"),
            ]
        )
        graph.set_relation("tested", "failed", True, False)
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=(
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
            recovery_policy="preserve_diagnose_repair_augment",
        )
        source = "def solve(x):\n    return x\n"
        canvas._progressive_outputs["tested"] = source
        canvas._progressive_output_metadata["tested"] = {
            "tool_receipts": [
                {
                    "tool_id": MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                    "request": {
                        "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                        "arguments": {"code": source},
                    },
                    "result": {"value": {"ok": True}},
                }
            ]
        }
        failure = AgentFailureRecord(
            request_id="failed-budget",
            agent_id="failed",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent exhausted 1 turns without a valid completion",
            metadata={"tool_plan_exhausted": True},
        )
        canvas._failed_agent_ids.add("failed")
        canvas._repair_exhausted_agent_ids.add("failed")
        canvas._latest_failure_record_by_agent["failed"] = failure

        self.assertEqual(("set_output",), canvas.model_admissible_action_types())
        self.assertEqual(
            ["tested"],
            canvas.model_admissible_action_targets()["set_output"]["agent_ids"],
        )

        canvas.graph.set_output("tested")
        self.assertEqual(("delete_agent",), canvas.model_admissible_action_types())
        self.assertEqual(
            ["failed"],
            canvas.model_admissible_action_targets()["delete_agent"]["agent_ids"],
        )
        candidate = canvas.graph.fork()
        candidate.set_relation("tested", "failed", False, True)
        self.assertIsNone(canvas._preserved_input_change_issue_for(candidate))

    def test_losslessly_forwarded_tested_source_can_be_terminal_sink(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
            dataset_id="mbpp_plus",
        )
        graph = AgentGraph(
            [
                AgentNode("tested", "m", "produce source", execution_mode="react"),
                AgentNode("sink", "m", "forward source"),
            ]
        )
        graph.set_relation("tested", "sink", True, False)
        graph.set_output("tested")
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=("set_output", "finish"),
            recovery_policy="preserve_diagnose_repair_augment",
        )
        source = "def solve(x):\n    return x\n"
        receipt = {
            "tool_id": MBPPPLUS_PYTHON_EXEC_TOOL_ID,
            "request": {
                "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                "arguments": {"code": source},
            },
            "result": {"value": {"ok": True}},
        }
        canvas._progressive_outputs.update(
            {"tested": source, "sink": source.rstrip()}
        )
        canvas._progressive_output_metadata["tested"] = {
            "tool_receipts": [receipt]
        }
        canvas._progressive_output_metadata["sink"] = {
            "input_artifact_provenance": [
                {
                    "source_agent_id": "tested",
                    "artifact_body": source,
                    "tool_receipts": [receipt],
                }
            ]
        }

        self.assertEqual(
            ("sink", "tested"),
            canvas._mbppplus_tested_agent_ids(),
        )
        self.assertEqual(
            ["sink"],
            canvas.model_admissible_action_targets()["set_output"]["agent_ids"],
        )

    def test_terminal_unreachable_mbppplus_agent_exposes_relation_repair(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
            dataset_id="mbpp_plus",
        )
        graph = AgentGraph(
            [
                AgentNode("producer", "m", "produce source", execution_mode="react"),
                AgentNode("output", "m", "return source", execution_mode="react"),
                AgentNode("isolated", "m", "check source", execution_mode="react"),
            ]
        )
        graph.set_relation("producer", "output", True, False)
        graph.set_output("output")
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=("set_relation", "set_output", "finish"),
            recovery_policy="preserve_diagnose_repair_augment",
        )
        source = "def solve(x):\n    return x\n"
        receipt = {
            "tool_id": MBPPPLUS_PYTHON_EXEC_TOOL_ID,
            "request": {
                "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                "arguments": {"code": source},
            },
            "result": {"value": {"ok": True}},
        }
        for agent_id in ("producer", "output", "isolated"):
            canvas._progressive_outputs[agent_id] = source
            canvas._progressive_output_metadata[agent_id] = {
                "tool_receipts": [receipt]
            }

        self.assertEqual(("set_relation",), canvas.model_admissible_action_types())
        self.assertIn(
            {
                "source_id": "isolated",
                "target_id": "output",
                "source_to_target": True,
                "target_to_source": False,
            },
            canvas.model_admissible_action_targets()["set_relation"]["candidates"],
        )

    def test_relation_edit_executes_missing_source_with_dirty_target(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
            dataset_id="mbpp_plus",
        )
        graph = AgentGraph(
            [
                AgentNode("missing_source", "m", "produce source"),
                AgentNode("tested_target", "m", "return source"),
            ]
        )
        graph.set_output("tested_target")
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=("set_relation", "finish"),
            recovery_policy="preserve_diagnose_repair_augment",
        )
        canvas._progressive_outputs["tested_target"] = "def solve(x):\n    return x\n"
        action = AgentAction(
            action_type=AgentActionType.SET_RELATION,
            source_id="missing_source",
            target_id="tested_target",
            source_to_target=True,
            target_to_source=False,
        )

        self.assertEqual(
            ("missing_source",),
            canvas._mbppplus_missing_relation_endpoint_ids(action),
        )
        reverse = AgentAction(
            action_type=AgentActionType.SET_RELATION,
            source_id="tested_target",
            target_id="missing_source",
            source_to_target=False,
            target_to_source=True,
        )
        self.assertEqual(
            ("missing_source",),
            canvas._mbppplus_missing_relation_endpoint_ids(reverse),
        )
        removed = AgentAction(
            action_type=AgentActionType.SET_RELATION,
            source_id="missing_source",
            target_id="tested_target",
            source_to_target=False,
            target_to_source=False,
        )
        self.assertEqual(
            (),
            canvas._mbppplus_missing_relation_endpoint_ids(removed),
        )

    def test_replaced_failed_mbppplus_branch_deletes_from_leaf_to_root(self) -> None:
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            SequenceGateway([]),
            execution_adapters={"react": StepwiseAdapter()},
            dataset_id="mbpp_plus",
        )
        graph = AgentGraph(
            [
                AgentNode("replacement", "m", "produce source", execution_mode="react"),
                AgentNode("failed_root", "m", "produce source", execution_mode="react"),
                AgentNode("blocked_leaf", "m", "forward source"),
            ]
        )
        graph.set_relation("failed_root", "blocked_leaf", True, False)
        graph.set_output("replacement")
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Return complete Python source.",
            execute_on_edit=False,
            allowed_actions=("delete_agent", "finish"),
            recovery_policy="preserve_diagnose_repair_augment",
        )
        source = "def solve(x):\n    return x\n"
        canvas._progressive_outputs["replacement"] = source
        canvas._progressive_output_metadata["replacement"] = {
            "tool_receipts": [
                {
                    "tool_id": MBPPPLUS_PYTHON_EXEC_TOOL_ID,
                    "request": {
                        "action": MBPPPLUS_PYTHON_EXEC_ACTION,
                        "arguments": {"code": source},
                    },
                    "result": {"value": {"ok": True}},
                }
            ]
        }
        failure = AgentFailureRecord(
            request_id="failed-root-budget",
            agent_id="failed_root",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="react agent exhausted 1 turns without a valid completion",
            metadata={"tool_plan_exhausted": True},
        )
        canvas._failed_agent_ids.add("failed_root")
        canvas._repair_exhausted_agent_ids.add("failed_root")
        canvas._latest_failure_record_by_agent["failed_root"] = failure

        self.assertEqual(
            ("blocked_leaf",),
            canvas._mbppplus_redundant_failed_agent_ids(),
        )
        canvas.graph.delete_agent("blocked_leaf")
        self.assertEqual(
            ("failed_root",),
            canvas._mbppplus_redundant_failed_agent_ids(),
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
        self.assertEqual(("finish",), canvas.model_admissible_action_types())
        finished = await canvas.step('{"action":"finish"}')
        self.assertTrue(finished.done)


if __name__ == "__main__":
    unittest.main()
