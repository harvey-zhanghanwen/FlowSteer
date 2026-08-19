from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    CommunicationCondition,
    ExecutionPhase,
)
from src.interactive.environment_execution import (
    build_environment_execution_resources,
    evaluator_locked_ragen_session_factory,
    RAGENEnvironmentSession,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import evaluate_task
from src.interactive.tool_runtime import ToolRequest


def make_request(task_family: str = "alfworld") -> AgentRequest:
    provider = ProviderSpec("fake", kind="test")
    model = ModelSpec("m", "fake")
    return AgentRequest(
        request_id="run:0:actor:single",
        run_id="run",
        graph_revision=0,
        problem=f"complete the {task_family} task",
        agent=AgentNode(
            "actor",
            "m",
            "select an admissible environment action",
            allowed_tools=(f"{task_family}.environment",),
            execution_mode="react",
            artifact_type="environment_observation",
        ),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        communication_condition=CommunicationCondition.NORMAL,
    )


class FakeSession:
    environment_id = "fake:alfworld"
    task_family = "alfworld"

    def __init__(self) -> None:
        self.reset_count = 0
        self.actions: list[str] = []
        self._available: object = ("look", "finish")

    @property
    def available_actions(self) -> object:
        return self._available

    def reset(self) -> str:
        self.reset_count += 1
        return "room zero"

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        if action == "look":
            self._available = ("finish",)
            return "room one", 0.5, False, {"won": False}
        return "task terminal observation", 1.0, True, {"won": True}


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            self.outputs.pop(0),
            {"provider_request_id": f"provider-{len(self.requests)}"},
        )


def resources(
    *, session: FakeSession, gateway: SequenceGateway, max_turns: int
):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family=session.task_family,
        max_turns=max_turns,
    )


class EnvironmentExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_runtime_admits_and_executes_registered_environment_tool(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["finish"])
        environment = resources(session=session, gateway=gateway, max_turns=1)
        model_registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            model_registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="alfworld",
        )
        graph = AgentGraph(
            [make_request().agent],
            output_agent_id="actor",
        )

        result = await runtime.execute(graph, "complete the alfworld task")

        self.assertEqual("task terminal observation", result.final_answer)
        self.assertEqual(["finish"], session.actions)
        self.assertTrue(result.output_metadata["actor"]["environment_terminal"])

    async def test_capability_backend_is_not_callable_outside_episode(self) -> None:
        session = FakeSession()
        runtime = resources(
            session=session,
            gateway=SequenceGateway(["look"]),
            max_turns=1,
        )

        result, receipt = await runtime.tool_registry.ainvoke_with_receipt(
            runtime.tool_id, ToolRequest("look", {})
        )

        self.assertIsNone(result)
        self.assertEqual("EnvironmentExecutionError", receipt.error_type)
        self.assertEqual([], session.actions)

    async def test_reset_action_step_observation_terminal_sequence(self) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["<action>look</action>", "finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=4)

        response = await runtime.execution_adapter.execute(make_request())

        self.assertIsInstance(response, AgentResponse)
        self.assertEqual("task terminal observation", response.text)
        self.assertEqual(1, session.reset_count)
        self.assertEqual(["look", "finish"], session.actions)
        self.assertEqual(2, response.metadata["environment_revision"])
        self.assertTrue(response.metadata["environment_terminal"])
        self.assertEqual(2, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            "alfworld.environment",
            response.metadata["tool_receipts"][0]["tool_id"],
        )
        self.assertEqual("react", response.metadata["execution_mode"])
        self.assertEqual(
            ["provider-1", "provider-2"],
            [
                call["metadata"]["provider_request_id"]
                for call in response.metadata["model_calls"]
            ],
        )
        capability = runtime.tool_registry.require_capability(runtime.tool_id)
        self.assertTrue(capability.availability)
        self.assertEqual(("alfworld",), capability.dataset_scope)
        receipts = response.metadata["environment_receipts"]
        self.assertEqual([1, 2], [item["environment_revision_after"] for item in receipts])
        self.assertEqual("room one", receipts[0]["next_observation"])
        self.assertIn("room one", gateway.requests[1].problem)
        self.assertEqual("reasoning", gateway.requests[0].agent.execution_mode.value)
        # Reward and the official success predicate remain evaluator-only.
        self.assertNotIn("reward", receipts[0])
        self.assertNotIn("info", receipts[1])
        self.assertNotIn("won", str(receipts))

    async def test_parse_error_consumes_turn_without_advancing_environment(self) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["I would inspect the room", "finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=2)

        response = await runtime.execution_adapter.execute(make_request())

        receipts = response.metadata["environment_receipts"]
        self.assertEqual(2, len(receipts))
        self.assertEqual("parse_error", receipts[0]["observation_status"])
        self.assertFalse(receipts[0]["state_advanced"])
        self.assertEqual(0, receipts[0]["environment_revision_after"])
        self.assertEqual(["finish"], session.actions)
        self.assertEqual(1, response.metadata["environment_revision"])

    async def test_turn_budget_returns_nonterminal_public_observation(self) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        response = await runtime.execution_adapter.execute(make_request())

        self.assertEqual("room one", response.text)
        self.assertFalse(response.metadata["environment_terminal"])
        self.assertEqual(1, response.metadata["environment_steps"])

    async def test_webshop_search_action_uses_public_search_bar_semantics(self) -> None:
        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {
                    "has_search_bar": True,
                    "clickables": ["Back to Search"],
                }

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                return "search results", 0.0, True, {"graded_score": 0.0}

        session = WebShopSession()
        gateway = SequenceGateway(["search[red waterproof shoes]"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        response = await runtime.execution_adapter.execute(make_request("webshop"))

        self.assertEqual(["search[red waterproof shoes]"], session.actions)
        self.assertTrue(response.metadata["environment_terminal"])
        self.assertIn("search[<your query>]", gateway.requests[0].problem)

    async def test_ragen_session_calls_deployed_adapter_signature(self) -> None:
        class FakeRAGEN:
            available_actions = ["look"]

            def __init__(self) -> None:
                self.reset_arguments = None
                self.actions: list[str] = []

            def reset(self, env_type, config, question="", extra=None):  # type: ignore[no-untyped-def]
                self.reset_arguments = (env_type, config, question, extra)
                return "initial observation"

            def step(self, action):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                return "next observation", 0.0, False, {"available_actions": ["look"]}

        bridge = FakeRAGEN()
        session = RAGENEnvironmentSession(
            adapter=bridge,
            env_type="alfworld",
            env_config={"seed": 3},
            question="put the apple in the fridge",
            extra={"task_id": "a-3"},
        )

        self.assertEqual("initial observation", session.reset())
        self.assertEqual(
            (
                "alfworld",
                {"seed": 3},
                "put the apple in the fridge",
                {"task_id": "a-3"},
            ),
            bridge.reset_arguments,
        )
        self.assertEqual("next observation", session.step("look")[0])
        self.assertEqual(["look"], bridge.actions)

    async def test_locked_alfworld_factory_reuses_evaluator_inventory_lock(self) -> None:
        target = "/games/game-b.tw-pddl"

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):  # type: ignore[no-untyped-def]
                self.game_files = ["/games/game-a.tw-pddl", target]

        class Adapter:
            available_actions = ["look"]

            def __init__(self) -> None:
                self.reset_arguments = None

            def reset(self, env_type, config, question="", extra=None):  # type: ignore[no-untyped-def]
                self.reset_arguments = (env_type, config, question, extra)
                return "room"

            def step(self, action):  # type: ignore[no-untyped-def]
                return "done", 1.0, True, {"won": True}

        record = TaskRecord(
            task_id="alfworld:1",
            question="put the apple in the fridge",
            ground_truth="",
            split="validation",
            metadata={
                "source": "ALFWorld",
                "dataset_key": "alfworld",
                "environment": {
                    "env_type": "alfworld",
                    "env_config": {"game_file": target, "seed": 999},
                },
            },
        )
        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=module,
        ):
            factory = evaluator_locked_ragen_session_factory(
                record=record,
                dataset="alfworld",
                ragen_adapter_path=Path("/fake/ragen.py"),
                max_environment_steps=50,
            )
        session = factory(make_request())
        session.reset()

        self.assertEqual(1, session.adapter.reset_arguments[1]["seed"])
        self.assertEqual(
            "put the apple in the fridge", session.adapter.reset_arguments[2]
        )

    async def test_evaluator_replays_adapter_trace_without_model_callback(self) -> None:
        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {"has_search_bar": False, "clickables": ["Buy Now"]}

            def reset(self) -> str:
                self.reset_count += 1
                return "Product page"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                return "Purchased", 1.0, True, {"graded_score": 1.0}

        live_session = WebShopSession()
        runtime = resources(
            session=live_session,
            gateway=SequenceGateway(["click[Buy Now]"]),
            max_turns=1,
        )
        response = await runtime.execution_adapter.execute(make_request("webshop"))

        class ReplayAdapter:
            def __init__(self) -> None:
                self._env = SimpleNamespace(current_goal_index=7)
                self.available_actions = {
                    "has_search_bar": False,
                    "clickables": ["Buy Now"],
                }

            def reset(self, env_type, config, question="", extra=None):  # type: ignore[no-untyped-def]
                return "Product page"

            def step(self, action):  # type: ignore[no-untyped-def]
                return "Purchased", 1.0, True, {"graded_score": 1.0}

        record = TaskRecord(
            task_id="webshop:7",
            question="buy the item",
            ground_truth="",
            split="validation",
            metadata={
                "source": "WebShop",
                "dataset_key": "webshop",
                "environment": {
                    "env_type": "webshop",
                    "env_config": {"goal_index": 7},
                },
            },
        )
        callback_called = False

        async def run_graph(_prompt: str) -> str:
            nonlocal callback_called
            callback_called = True
            return "click[Buy Now]"

        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=ReplayAdapter),
        ):
            outcome = await evaluate_task(
                record,
                response.text,
                run_graph=run_graph,
                max_environment_steps=1,
                environment_replay_trace=response.metadata[
                    "evaluator_environment_trace"
                ],
            )

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.metrics["success"])
        self.assertFalse(callback_called)
        # Evaluator truth is absent from public receipts and model calls.
        self.assertNotIn("graded_score", str(response.metadata["environment_receipts"]))
        self.assertNotIn("graded_score", str(response.metadata["model_calls"]))


if __name__ == "__main__":
    unittest.main()
