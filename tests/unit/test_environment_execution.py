from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    CommunicationCondition,
    ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.environment_execution import (
    _public_transition_summary,
    build_environment_execution_resources,
    EnvironmentExecutionError,
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
    *,
    session: FakeSession,
    gateway: SequenceGateway,
    max_turns: int,
    max_observation_chars: int = 0,
    stepwise_director: bool = False,
    structured_actions: bool = False,
):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family=session.task_family,
        max_turns=max_turns,
        max_observation_chars=max_observation_chars,
        stepwise_director=stepwise_director,
        structured_actions=structured_actions,
    )


class EnvironmentExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_repeated_search_transition_is_public_no_progress(self) -> None:
        receipt = {
            "state_advanced": True,
            "action": "search[blue steel table]",
            "observation": "Search page",
            "next_observation": "Search results for blue steel table",
            "observation_status": "success",
            "terminal": False,
        }

        summary = _public_transition_summary(
            task_family="webshop",
            observation="Search results for blue steel table",
            receipts=(receipt, dict(receipt)),
        )

        latest = summary["latest_transition"]
        no_progress = summary["no_progress"]
        assert isinstance(latest, dict)
        assert isinstance(no_progress, dict)
        self.assertTrue(latest["observation_changed"])
        self.assertTrue(latest["result_is_current_state"])
        self.assertTrue(no_progress["detected"])
        self.assertEqual(
            ["repeated_state_action"],
            no_progress["reasons"],
        )
        self.assertEqual(2, no_progress["repeated_state_action_count"])

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
        self.assertEqual({}, dict(capability.action_schemas))
        self.assertEqual((), capability.action_names)
        receipts = response.metadata["environment_receipts"]
        self.assertEqual([1, 2], [item["environment_revision_after"] for item in receipts])
        self.assertEqual("room one", receipts[0]["next_observation"])
        self.assertIn("room one", gateway.requests[1].problem)
        self.assertIn("action='look'", gateway.requests[1].problem)
        self.assertNotIn("next_observation='room one'", gateway.requests[1].problem)
        self.assertEqual("512", gateway.requests[0].model.metadata["max_tokens"])
        self.assertEqual("512", gateway.requests[1].model.metadata["max_tokens"])
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
        self.assertIn("Format repair", gateway.requests[1].problem)
        self.assertIn("environment state did not change", gateway.requests[1].problem)

    async def test_alfworld_public_state_retains_target_coreference_and_count(self) -> None:
        class CountSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = (
                    "move newspaper 2 to sidetable 1",
                    "finish",
                )

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action.startswith("move "):
                    self._available = ("finish",)
                    return "one newspaper is placed", 0.0, False, {"won": False}
                return "task terminal observation", 1.0, True, {"won": True}

        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Put two newspaper and put them on sidetable.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        session = CountSession()
        gateway = SequenceGateway(["move newspaper 2 to sidetable 1", "finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=2)

        response = await runtime.execution_adapter.execute(request)

        first_prompt = gateway.requests[0].problem
        second_prompt = gateway.requests[1].problem
        self.assertIn("target_class=newspaper", first_prompt)
        self.assertIn("destination_class=sidetable", first_prompt)
        self.assertIn("count=2", first_prompt)
        self.assertIn("`them` refers to the required `newspaper` instances", first_prompt)
        self.assertIn("Visible placement progress: 1/2", second_prompt)
        self.assertEqual(
            "newspaper",
            response.metadata["model_calls"][0]["public_state"]
            .split("target_class=", 1)[1]
            .split(";", 1)[0],
        )

    async def test_alfworld_public_state_binds_it_to_transformed_target(self) -> None:
        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Heat some egg and put it in sidetable.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        session = FakeSession()
        gateway = SequenceGateway(["finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        await runtime.execution_adapter.execute(request)

        prompt = gateway.requests[0].problem
        self.assertIn("required_transform=heat", prompt)
        self.assertIn("`it` refers to `egg`", prompt)

    async def test_webshop_public_state_retains_visible_instruction_constraints(self) -> None:
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

        base = make_request("webshop")
        request = AgentRequest(
            request_id=base.request_id,
            run_id=base.run_id,
            graph_revision=base.graph_revision,
            problem=(
                "Find a steel shelf with color: blue, item shape: rectangular, "
                "and price lower than $40."
            ),
            agent=base.agent,
            model=base.model,
            provider=base.provider,
            phase=base.phase,
            communication_condition=base.communication_condition,
        )
        session = WebShopSession()
        gateway = SequenceGateway(["search[blue rectangular steel shelf]"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        await runtime.execution_adapter.execute(request)

        prompt = gateway.requests[0].problem
        self.assertIn("price_lower_than=40", prompt)
        self.assertIn("color=blue", prompt)
        self.assertIn("item_shape=rectangular", prompt)

    async def test_model_prompt_observation_cap_does_not_truncate_receipt(self) -> None:
        class LongSession(FakeSession):
            def reset(self) -> str:
                self.reset_count += 1
                return "A" * 500

        session = LongSession()
        gateway = SequenceGateway(["finish"])
        runtime = resources(
            session=session,
            gateway=gateway,
            max_turns=1,
            max_observation_chars=80,
        )

        response = await runtime.execution_adapter.execute(make_request())

        self.assertIn("OBSERVATION CLIPPED", gateway.requests[0].problem)
        self.assertTrue(response.metadata["model_calls"][0]["observation_clipped"])
        self.assertEqual("A" * 500, response.metadata["environment_reset_receipt"]["observation"])

    async def test_turn_budget_returns_nonterminal_public_observation(self) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        response = await runtime.execution_adapter.execute(make_request())

        self.assertEqual("room one", response.text)
        self.assertFalse(response.metadata["environment_terminal"])
        self.assertEqual(1, response.metadata["environment_steps"])

    async def test_failure_preserves_completed_environment_prefix_receipts(self) -> None:
        class FailingGateway(SequenceGateway):
            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                if len(self.requests) == 1:
                    return AgentResponse(
                        "look",
                        {"provider_request_id": "provider-1"},
                    )
                raise TimeoutError("provider timeout")

        session = FakeSession()
        runtime = resources(
            session=session,
            gateway=FailingGateway([]),
            max_turns=2,
        )

        with self.assertRaises(EnvironmentExecutionError) as caught:
            await runtime.execution_adapter.execute(make_request())

        error = caught.exception
        self.assertEqual("TimeoutError", error.cause_error_type)
        self.assertEqual(1, error.environment_revision)
        self.assertFalse(error.environment_terminal)
        self.assertEqual(1, len(error.environment_receipts))
        self.assertEqual("look", error.environment_receipts[0]["action"])
        self.assertEqual(1, len(error.evaluator_environment_trace))
        self.assertEqual(1, len(error.tool_receipts))
        self.assertEqual(1, len(error.model_calls))
        self.assertEqual("room zero", error.environment_reset_receipt["observation"])

    async def test_canvas_finish_accepts_terminal_or_budget_truncation(self) -> None:
        for raw_action, expected_terminal in (("look", False), ("finish", True)):
            with self.subTest(raw_action=raw_action):
                session = FakeSession()
                gateway = SequenceGateway([raw_action])
                environment = resources(
                    session=session,
                    gateway=gateway,
                    max_turns=1,
                )
                model_registry = ModelRegistry(
                    [ProviderSpec("fake", kind="test")],
                    [ModelSpec("m", "fake")],
                )
                runtime = AgentRuntime(
                    model_registry,
                    gateway,
                    execution_adapters={
                        "react": environment.execution_adapter,
                    },
                    tool_registry=environment.tool_registry,
                    dataset_id="alfworld",
                )
                canvas = AgentWorkflowEnv(
                    model_registry,
                    runtime=runtime,
                    problem="complete the alfworld task",
                    execute_on_edit=True,
                    required_tool_id=environment.tool_id,
                )
                added = await canvas.step(
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"actor","model_id":"m",'
                    '"contract":"act in the environment",'
                    '"allowed_tools":["alfworld.environment"],'
                    '"execution_mode":"react",'
                    '"artifact_type":"environment_observation",'
                    '"completion_condition":"reach a terminal observation"}'
                    '],"relations":[],"output_agent_id":"actor"}'
                )
                self.assertTrue(added.accepted)
                self.assertIsNotNone(added.execution)

                finished = await canvas.step('{"action":"finish"}')

                self.assertTrue(finished.accepted)
                self.assertTrue(finished.done)
                assert added.execution is not None
                metadata = added.execution.output_metadata["actor"]
                self.assertEqual(
                    not expected_terminal,
                    metadata["environment_truncated"],
                )
                self.assertEqual(1, metadata["environment_max_turns"])

    async def test_fresh_finish_accepts_and_returns_budget_truncation(self) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=1,
        )
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
        canvas = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="complete the alfworld task",
            graph=graph,
            execute_on_edit=False,
            required_tool_id=environment.tool_id,
        )

        first = await canvas.step('{"action":"finish"}')

        self.assertTrue(first.accepted)
        self.assertTrue(first.done)
        self.assertFalse(first.execution_reused)
        self.assertIsNotNone(first.execution)
        self.assertIsNone(first.partial_execution)
        self.assertEqual("room one", first.execution.outputs["actor"])
        actor_metadata = first.execution.output_metadata["actor"]
        self.assertFalse(actor_metadata["environment_terminal"])
        self.assertTrue(actor_metadata["environment_truncated"])
        self.assertEqual(1, actor_metadata["environment_max_turns"])
        self.assertEqual(1, len(actor_metadata["environment_receipts"]))
        self.assertEqual(1, len(actor_metadata["tool_receipts"]))
        self.assertEqual(["look"], session.actions)
        self.assertEqual(1, len(gateway.requests))

    async def test_required_environment_capability_uses_atomic_live_repair_domain(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway([])
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
        allowed_actions = (
            "add_agent",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        )

        empty = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="complete the task",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=allowed_actions,
        )
        self.assertEqual(("add_agent",), empty.model_admissible_action_types())
        self.assertEqual(
            ["agent_id", "model_id", "contract"],
            empty.model_admissible_action_targets()["add_agent"][
                "required_agent_fields"
            ],
        )

        reasoning_graph = AgentGraph(
            [AgentNode("arbitrary", "m", "Use the available interface.")]
        )
        reasoning = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="complete the task",
            graph=reasoning_graph,
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=allowed_actions,
        )
        self.assertEqual(
            ("modify_agent",), reasoning.model_admissible_action_types()
        )
        repair = reasoning.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["execution_mode"], repair["mutable_fields"])
        self.assertEqual(
            {"execution_mode": ["react"]},
            repair["per_agent_candidates"][0]["discrete_value_domains"],
        )

        react_graph = AgentGraph(
            [
                AgentNode(
                    "arbitrary",
                    "m",
                    "Use the available interface.",
                    execution_mode="react",
                )
            ]
        )
        react = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="complete the task",
            graph=react_graph,
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=allowed_actions,
        )
        repair = react.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["allowed_tools"], repair["mutable_fields"])
        self.assertEqual(
            {"allowed_tools": [[environment.tool_id]]},
            repair["per_agent_candidates"][0]["discrete_value_domains"],
        )

        actor_graph = AgentGraph([make_request().agent])
        actor = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="complete the task",
            graph=actor_graph,
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=allowed_actions,
        )
        restored = actor.model_admissible_action_types()
        self.assertIn("add_agent", restored)
        self.assertIn("set_output", restored)
        self.assertNotEqual(("modify_agent",), restored)

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
        self.assertIn("Do not return JSON", gateway.requests[0].problem)
        self.assertIn("Current Agent contract:", gateway.requests[0].problem)
        self.assertIn("select an admissible environment action", gateway.requests[0].problem)
        self.assertEqual(
            "environment_action", gateway.requests[0].agent.artifact_type
        )
        self.assertEqual(
            "Select exactly one native action permitted by the current "
            "admissible-action list.",
            gateway.requests[0].agent.contract,
        )

    async def test_webshop_structured_stepwise_calls_advance_one_shared_episode(
        self,
    ) -> None:
        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {
                    "has_search_bar": True,
                    "clickables": ["Back to Search"],
                }

            def reset(self) -> str:
                self.reset_count += 1
                return "Search for a product."

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action.startswith("search["):
                    self._available = {
                        "has_search_bar": False,
                        "clickables": ["B000ITEM01"],
                    }
                    return "Search result B000ITEM01", 0.0, False, {
                        "graded_score": 0.0
                    }
                return "Product B000ITEM01", 0.0, True, {
                    "graded_score": 0.0
                }

        search = json.dumps(
            {
                "resource_id": "webshop",
                "kind": "tool",
                "name": "search",
                "arguments": {"query": "blue steel shelf"},
                "skill_id": None,
            }
        )
        click = json.dumps(
            {
                "resource_id": "webshop",
                "kind": "tool",
                "name": "click",
                "arguments": {"target": "B000ITEM01"},
                "skill_id": None,
            }
        )
        session = WebShopSession()
        gateway = SequenceGateway([search, click])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
            structured_actions=True,
        )
        request = make_request("webshop")

        first = await environment.execution_adapter.execute(request)
        second = await environment.execution_adapter.execute(request)

        self.assertEqual(1, session.reset_count)
        self.assertEqual(
            ["search[blue steel shelf]", "click[B000ITEM01]"],
            session.actions,
        )
        self.assertEqual(1, first.metadata["environment_revision"])
        self.assertEqual(2, second.metadata["environment_revision"])
        self.assertEqual(1, len(first.metadata["environment_receipts"]))
        self.assertEqual(2, len(second.metadata["environment_receipts"]))
        self.assertEqual(
            first.metadata["environment_episode_id"],
            second.metadata["environment_episode_id"],
        )
        self.assertEqual(
            "one_action_one_observation",
            second.metadata["environment_execution_boundary"],
        )
        self.assertTrue(second.metadata["environment_terminal"])
        self.assertIn(
            "response_json_schema", gateway.requests[0].model.metadata
        )
        self.assertIn("StructuredAction JSON", gateway.requests[0].problem)
        self.assertNotIn("Do not return JSON", gateway.requests[0].problem)
        for request_value in gateway.requests:
            self.assertIn("complete the webshop task", request_value.problem)
            self.assertNotIn("graded_score", request_value.problem)

    async def test_webshop_structured_action_fills_only_fixed_wire_constants(
        self,
    ) -> None:
        """Provider-omitted const fields do not discard a semantic action."""

        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {"has_search_bar": True, "clickables": ["Search"]}

            def reset(self) -> str:
                self.reset_count += 1
                return "Search page"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                return "Results", 0.0, True, {"graded_score": 0.0}

        # ``kind`` and ``skill_id`` are fixed by the WebShop tool resource;
        # the provider supplied every semantic field.
        provider_value = json.dumps(
            {
                "resource_id": "webshop",
                "name": "search",
                "arguments": {"query": "blue steel shelf"},
            }
        )
        session = WebShopSession()
        environment = resources(
            session=session,
            gateway=SequenceGateway([provider_value]),
            max_turns=1,
            stepwise_director=True,
            structured_actions=True,
        )

        response = await environment.execution_adapter.execute(make_request("webshop"))

        self.assertEqual(["search[blue steel shelf]"], session.actions)
        receipt = response.metadata["environment_receipts"][0]
        evaluator_entry = response.metadata["evaluator_environment_trace"][0]
        self.assertEqual("format_normalized", receipt["observation_status"])
        self.assertEqual(
            "format_normalized", evaluator_entry["structured_action_status"]
        )
        self.assertEqual(provider_value, evaluator_entry["structured_action_output"])

    async def test_canvas_continue_preserves_graph_revision_and_returns_public_state(
        self,
    ) -> None:
        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {
                    "has_search_bar": True,
                    "clickables": ["Back to Search"],
                }

            def reset(self) -> str:
                self.reset_count += 1
                return "Search page"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action.startswith("search["):
                    self._available = {
                        "has_search_bar": False,
                        "clickables": ["Buy Now"],
                    }
                    return "Matching product page", 0.0, False, {
                        "graded_score": 0.0
                    }
                return "Thank you for shopping with us!", 1.0, True, {
                    "graded_score": 1.0
                }

        actions = [
            json.dumps(
                {
                    "resource_id": "webshop",
                    "kind": "tool",
                    "name": "search",
                    "arguments": {"query": "requested item"},
                    "skill_id": None,
                }
            ),
            json.dumps(
                {
                    "resource_id": "webshop",
                    "kind": "tool",
                    "name": "click",
                    "arguments": {"target": "Buy Now"},
                    "skill_id": None,
                }
            ),
        ]
        session = WebShopSession()
        gateway = SequenceGateway(actions)
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=2,
            max_observation_chars=12,
            stepwise_director=True,
            structured_actions=True,
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="buy the requested item",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=(
                "add_agent",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
        )
        add = await canvas.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "actor",
                    "model_id": "m",
                    "contract": "Act on the current public shopping state.",
                    "execution_mode": "react",
                    "allowed_tools": [environment.tool_id],
                }
            )
        )
        self.assertTrue(add.accepted)
        await canvas.step('{"action":"set_output","agent_id":"actor"}')
        revision_before_continue = canvas.revision
        self.assertIn("continue", canvas.model_admissible_action_types())
        continued = await canvas.step('{"action":"continue"}')

        self.assertTrue(continued.accepted)
        self.assertEqual(revision_before_continue, canvas.revision)
        self.assertEqual(1, session.reset_count)
        self.assertEqual(2, len(session.actions))
        state = canvas.public_environment_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual("buy the requested item", state["task_instruction"])
        self.assertEqual("click[Buy Now]", state["last_action"])
        self.assertTrue(state["current_observation_clipped"])
        self.assertTrue(
            str(state["current_observation"]).startswith("Thank you fo")
        )
        self.assertIn("[OBSERVATION CLIPPED:", state["current_observation"])
        self.assertEqual(
            len("Thank you for shopping with us!"),
            state["current_observation_original_chars"],
        )
        assert continued.execution is not None
        full_trace = continued.execution.output_metadata["actor"][
            "evaluator_environment_trace"
        ]
        self.assertEqual(
            "Thank you for shopping with us!",
            full_trace[-1]["next_observation"],
        )
        self.assertTrue(state["environment_terminal"])
        self.assertNotIn("graded_score", str(state))
        self.assertNotIn("continue", canvas.model_admissible_action_types())
        self.assertIn("finish", canvas.model_admissible_action_types())
        finished = await canvas.step('{"action":"finish"}')
        self.assertTrue(finished.done)

    async def test_webshop_no_progress_returns_goal_and_repairs_preserved_actor(
        self,
    ) -> None:
        class RepeatingWebShopSession(FakeSession):
            environment_id = "fake:webshop:repeat"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {
                    "has_search_bar": False,
                    "clickables": ["6.6ft", "Buy Now"],
                }

            def reset(self) -> str:
                self.reset_count += 1
                return "Product page with size options 6.6ft and Buy Now"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action == "click[6.6ft]":
                    return (
                        "Product page with size options 6.6ft and Buy Now",
                        0.0,
                        False,
                        {"graded_score": 0.0},
                    )
                return (
                    "Thank you for shopping with us!",
                    1.0,
                    True,
                    {"graded_score": 1.0},
                )

        repeated_option = json.dumps(
            {
                "resource_id": "webshop",
                "kind": "tool",
                "name": "click",
                "arguments": {"target": "6.6ft"},
                "skill_id": None,
            }
        )
        buy_now = json.dumps(
            {
                "resource_id": "webshop",
                "kind": "tool",
                "name": "click",
                "arguments": {"target": "Buy Now"},
                "skill_id": None,
            }
        )
        session = RepeatingWebShopSession()
        gateway = SequenceGateway([repeated_option, repeated_option, buy_now])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
            structured_actions=True,
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        instruction = "buy the exact requested size"
        workflow_problem = (
            instruction + "\n\nExecution interface: use the environment tool."
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem=workflow_problem,
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=(
                "add_agent",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
        )
        await canvas.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "actor",
                    "model_id": "m",
                    "contract": "Select an exact constraint match.",
                    "execution_mode": "react",
                    "allowed_tools": [environment.tool_id],
                }
            )
        )
        await canvas.step('{"action":"set_output","agent_id":"actor"}')
        await canvas.step('{"action":"continue"}')

        stalled_state = canvas.public_environment_state()
        self.assertIsNotNone(stalled_state)
        assert stalled_state is not None
        self.assertEqual(instruction, stalled_state["original_task_instruction"])
        self.assertEqual(workflow_problem, stalled_state["task_instruction"])
        self.assertNotIn("admissible_actions", stalled_state)
        self.assertEqual(2, stalled_state["admissible_action_count"])
        progress = stalled_state["public_progress"]
        assert isinstance(progress, dict)
        self.assertTrue(progress["no_progress"]["detected"])
        self.assertEqual(
            ["repeated_state_action"], progress["no_progress"]["reasons"]
        )
        episode_id = stalled_state["environment_episode_id"]
        self.assertEqual(("modify_agent",), canvas.model_admissible_action_types())
        modify_domain = canvas.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["actor"], modify_domain["agent_ids"])
        self.assertEqual(["contract"], modify_domain["mutable_fields"])

        repaired = await canvas.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "actor",
                    "contract": "Do not repeat an unchanged option; use current evidence and remaining budget.",
                }
            )
        )
        self.assertTrue(repaired.accepted)
        self.assertEqual(episode_id, canvas.public_environment_state()["environment_episode_id"])
        self.assertEqual(
            ["click[6.6ft]", "click[6.6ft]", "click[Buy Now]"],
            session.actions,
        )
        self.assertIn("Current Agent contract:", gateway.requests[-1].problem)
        self.assertIn("Do not repeat an unchanged option", gateway.requests[-1].problem)
        self.assertIn("Apply one ReAct control cycle", gateway.requests[-1].problem)
        self.assertNotIn("Execution interface:", gateway.requests[-1].problem)
        self.assertNotIn("graded_score", str(canvas.public_environment_state()))

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
        raw_terminal_observation = (
            "Thank you for shopping with us! [SEP] Purchased [SEP] asin [SEP] "
            "B000TEST [SEP] Reward [SEP] Your score (min 0.0, max 1.0) "
            "[SEP] 1.0 [SEP] Reward Details [SEP] None"
        )

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
                return raw_terminal_observation, 1.0, True, {"graded_score": 1.0}

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
                return raw_terminal_observation, 1.0, True, {"graded_score": 1.0}

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
        # The upstream terminal page embeds the native score.  Preserve that
        # exact raw page only in the evaluator replay trace; the Agent output,
        # public Action--Observation receipt, and model calls receive the
        # reward-free observation projection.
        self.assertEqual("Thank you for shopping with us!", response.text)
        self.assertNotIn("B000TEST", response.text)
        self.assertNotIn("Your score", response.text)
        self.assertNotIn("Reward Details", response.text)
        self.assertEqual(
            raw_terminal_observation,
            response.metadata["evaluator_environment_trace"][0][
                "next_observation"
            ],
        )
        self.assertEqual(
            1.0,
            response.metadata["evaluator_environment_trace"][0]["reward"],
        )
        self.assertNotIn(
            "Your score", str(response.metadata["environment_receipts"])
        )
        self.assertNotIn("Reward Details", str(response.metadata["model_calls"]))
        self.assertNotIn("graded_score", str(response.metadata["environment_receipts"]))
        self.assertNotIn("graded_score", str(response.metadata["model_calls"]))

    async def test_evaluator_replays_structured_stepwise_trace_without_model_callback(
        self,
    ) -> None:
        """StructuredAction is retained while evaluator receives native wire."""

        raw_terminal_observation = (
            "Thank you for shopping with us! [SEP] Purchased [SEP] asin [SEP] "
            "B000TEST [SEP] Reward [SEP] Your score (min 0.0, max 1.0) "
            "[SEP] 1.0 [SEP] Reward Details [SEP] None"
        )
        structured = json.dumps(
            {
                "arguments": {"target": "Buy Now"},
                "kind": "tool",
                "name": "click",
                "resource_id": "webshop",
                "skill_id": None,
            },
            sort_keys=True,
        )

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
                return raw_terminal_observation, 1.0, True, {"graded_score": 1.0}

        runtime = resources(
            session=WebShopSession(),
            gateway=SequenceGateway([structured]),
            max_turns=1,
            stepwise_director=True,
            structured_actions=True,
        )
        response = await runtime.execution_adapter.execute(make_request("webshop"))
        trace = response.metadata["evaluator_environment_trace"]

        self.assertEqual("<action>click[Buy Now]</action>", trace[0]["raw_graph_output"])
        self.assertEqual(structured, trace[0]["structured_action_output"])
        self.assertEqual("success", trace[0]["structured_action_status"])

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
                return raw_terminal_observation, 1.0, True, {"graded_score": 1.0}

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
                environment_replay_trace=trace,
            )

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.metrics["success"])
        self.assertFalse(callback_called)

    async def test_webshop_terminal_private_fields_do_not_reach_downstream_agent(
        self,
    ) -> None:
        class WebShopSession(FakeSession):
            environment_id = "fake:webshop"
            task_family = "webshop"

            def __init__(self) -> None:
                super().__init__()
                self._available = {"has_search_bar": False, "clickables": ["Buy Now"]}

            def reset(self) -> str:
                return "Product page"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                return (
                    "Thank you for shopping with us! [SEP] Purchased [SEP] asin "
                    "[SEP] B000PRIVATE [SEP] Target [SEP] secret attributes "
                    "[SEP] Reward [SEP] Your score (min 0.0, max 1.0) "
                    "[SEP] 0.75 [SEP] Reward Details [SEP] private details",
                    0.75,
                    True,
                    {"graded_score": 0.75},
                )

        gateway = SequenceGateway(["click[Buy Now]", "public terminal summary"])
        session = WebShopSession()
        environment = resources(session=session, gateway=gateway, max_turns=1)
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        actor = make_request("webshop").agent
        downstream = AgentNode(
            "downstream",
            "m",
            "Use the upstream public environment artifact.",
            execution_mode="reasoning",
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        graph = AgentGraph(
            [actor, downstream],
            [AgentRelation("actor", "downstream", True, False)],
            output_agent_id="downstream",
        )

        result = await runtime.execute(graph, "buy the requested item")

        self.assertEqual("public terminal summary", result.final_answer)
        upstream = str(gateway.requests[1].upstream)
        self.assertIn("Thank you for shopping with us!", upstream)
        for private_value in (
            "B000PRIVATE",
            "secret attributes",
            "Your score",
            "Reward Details",
            "graded_score",
        ):
            self.assertNotIn(private_value, upstream)


if __name__ == "__main__":
    unittest.main()
