from __future__ import annotations

import json
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
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION,
    decode_director_transcript,
    director_system_prompt_for_version,
)
from src.interactive.environment_execution import (
    build_environment_execution_resources,
    EnvironmentExecutionError,
    evaluator_locked_ragen_session_factory,
    RAGENEnvironmentSession,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import evaluate_task
from src.interactive.tool_runtime import ToolRequest


def make_request(
    task_family: str = "alfworld",
    *,
    tool_id: str | None = None,
) -> AgentRequest:
    provider = ProviderSpec("fake", kind="test")
    model = ModelSpec("m", "fake")
    resolved_tool_id = tool_id or (
        "alfworld" if task_family == "alfworld" else f"{task_family}.environment"
    )
    return AgentRequest(
        request_id="run:0:actor:single",
        run_id="run",
        graph_revision=0,
        problem=f"complete the {task_family} task",
        agent=AgentNode(
            "actor",
            "m",
            "select an admissible environment action",
            allowed_tools=(resolved_tool_id,),
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
        self.close_count = 0
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

    def close(self) -> None:
        self.close_count += 1


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
):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family=session.task_family,
        max_turns=max_turns,
        max_observation_chars=max_observation_chars,
        stepwise_director=stepwise_director,
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
            runtime.tool_id, ToolRequest("act", {"command": "look"})
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
            "alfworld",
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
        self.assertEqual(("act",), capability.action_names)
        self.assertEqual(
            {"command"},
            set(capability.action_schemas["act"]["properties"]),
        )
        receipts = response.metadata["environment_receipts"]
        self.assertEqual([1, 2], [item["environment_revision_after"] for item in receipts])
        self.assertEqual("room one", receipts[0]["next_observation"])
        self.assertIn("room one", gateway.requests[1].problem)
        self.assertIn("Action: 'look'", gateway.requests[1].problem)
        self.assertNotIn("next_observation='room one'", gateway.requests[1].problem)
        self.assertEqual("512", gateway.requests[0].model.metadata["max_tokens"])
        self.assertEqual("512", gateway.requests[1].model.metadata["max_tokens"])
        self.assertEqual("reasoning", gateway.requests[0].agent.execution_mode.value)
        # Reward and the official success predicate remain evaluator-only.
        self.assertNotIn("reward", receipts[0])
        self.assertNotIn("info", receipts[1])
        self.assertNotIn("won", str(receipts))

    async def test_provider_failure_retries_same_rollout_episode_and_budget(self) -> None:
        class RetryGateway(SequenceGateway):
            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                if len(self.requests) == 1:
                    return AgentResponse(
                        "look",
                        {"provider_request_id": "provider-1"},
                    )
                if len(self.requests) == 2:
                    raise TimeoutError("provider timeout")
                return AgentResponse(
                    "finish",
                    {"provider_request_id": "provider-3"},
                )

        session = FakeSession()
        factory_requests: list[AgentRequest] = []

        def session_factory(request: AgentRequest) -> FakeSession:
            factory_requests.append(request)
            return session

        gateway = RetryGateway([])
        runtime = build_environment_execution_resources(
            gateway=gateway,
            session_factory=session_factory,
            task_family="alfworld",
            max_turns=3,
        )

        with self.assertRaises(EnvironmentExecutionError) as caught:
            await runtime.execution_adapter.execute(make_request())

        failed_reset = caught.exception.environment_reset_receipt
        self.assertIsNotNone(failed_reset)
        self.assertEqual(1, caught.exception.environment_revision)
        self.assertEqual(["look"], session.actions)

        response = await runtime.execution_adapter.execute(make_request())

        self.assertEqual(1, len(factory_requests))
        self.assertEqual(1, session.reset_count)
        self.assertEqual(["look", "finish"], session.actions)
        self.assertEqual(2, response.metadata["environment_revision"])
        self.assertTrue(response.metadata["environment_terminal"])
        self.assertEqual(
            failed_reset["episode_id"],
            response.metadata["environment_reset_receipt"]["episode_id"],
        )
        self.assertEqual(
            ["look", "finish"],
            [item["action"] for item in response.metadata["environment_receipts"]],
        )
        self.assertEqual(
            [0, 1],
            [item["step"] for item in response.metadata["evaluator_environment_trace"]],
        )
        # The failed provider call is not an environment action; the shared
        # rollout budget is charged only by persisted Action--Observation turns.
        self.assertEqual(2, response.metadata["environment_turns_used"])

    async def test_terminal_and_budget_exhaustion_do_not_restart_episode(self) -> None:
        cases = (
            ("finish", True, "task terminal observation"),
            ("look", False, "room one"),
        )
        for action, terminal, observation in cases:
            with self.subTest(action=action):
                session = FakeSession()
                gateway = SequenceGateway([action])
                runtime = resources(session=session, gateway=gateway, max_turns=1)

                first = await runtime.execution_adapter.execute(make_request())
                second = await runtime.execution_adapter.execute(make_request())

                self.assertEqual(observation, first.text)
                self.assertEqual(observation, second.text)
                self.assertEqual(terminal, second.metadata["environment_terminal"])
                self.assertEqual(1, session.reset_count)
                self.assertEqual([action], session.actions)
                self.assertEqual(1, len(gateway.requests))
                self.assertEqual(1, second.metadata["environment_revision"])
                self.assertEqual(1, len(second.metadata["environment_receipts"]))
                self.assertEqual(
                    first.metadata["environment_reset_receipt"]["episode_id"],
                    second.metadata["environment_reset_receipt"]["episode_id"],
                )

    async def test_resource_instances_isolate_rollout_episode_state(self) -> None:
        first_session = FakeSession()
        second_session = FakeSession()
        first = resources(
            session=first_session,
            gateway=SequenceGateway(["look"]),
            max_turns=1,
        )
        second = resources(
            session=second_session,
            gateway=SequenceGateway(["finish"]),
            max_turns=1,
        )

        first_response = await first.execution_adapter.execute(make_request())
        second_response = await second.execution_adapter.execute(make_request())

        self.assertNotEqual(
            first_response.metadata["environment_reset_receipt"]["episode_id"],
            second_response.metadata["environment_reset_receipt"]["episode_id"],
        )
        self.assertEqual(["look"], first_session.actions)
        self.assertEqual(["finish"], second_session.actions)
        self.assertEqual("room one", first_response.text)
        self.assertEqual("task terminal observation", second_response.text)
        self.assertFalse(first_response.metadata["environment_terminal"])
        self.assertTrue(second_response.metadata["environment_terminal"])

    async def test_resources_close_closes_rollout_session_exactly_once(self) -> None:
        session = FakeSession()
        runtime = resources(
            session=session,
            gateway=SequenceGateway(["finish"]),
            max_turns=1,
        )
        await runtime.execution_adapter.execute(make_request())

        runtime.close()
        runtime.close()

        self.assertEqual(1, session.close_count)

    async def test_native_action_subcall_preserves_director_contract(self) -> None:
        original = make_request()
        director_contract = (
            "Use the shared evidence to complete the task while preserving "
            "the requested object and destination."
        )
        request = AgentRequest(
            request_id=original.request_id,
            run_id=original.run_id,
            graph_revision=original.graph_revision,
            problem=original.problem,
            agent=AgentNode(
                original.agent.id,
                original.agent.model_id,
                director_contract,
                allowed_tools=original.agent.allowed_tools,
                execution_mode=original.agent.execution_mode,
                artifact_type=original.agent.artifact_type,
            ),
            model=original.model,
            provider=original.provider,
            phase=original.phase,
            communication_condition=original.communication_condition,
        )
        session = FakeSession()
        gateway = SequenceGateway(["finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        await runtime.execution_adapter.execute(request)

        provider_request = gateway.requests[0]
        self.assertEqual(director_contract, provider_request.agent.contract)
        self.assertIn(
            "You are an expert agent operating in the ALFRED Embodied Environment.",
            provider_request.problem,
        )
        self.assertIn("Your admissible actions", provider_request.problem)

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
        self.assertIn("Action: '<INVALID>'", gateway.requests[1].problem)
        self.assertIn(
            "Result: '[INVALID] No valid <action> tag found.'",
            gateway.requests[1].problem,
        )

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

        first_state = response.metadata["model_calls"][0]["public_state"]
        second_state = response.metadata["model_calls"][1]["public_state"]
        self.assertIn("target_class=newspaper", first_state)
        self.assertIn("destination_class=sidetable", first_state)
        self.assertIn("count=2", first_state)
        self.assertIn("`them` refers to the required `newspaper` instances", first_state)
        self.assertIn("Visible placement progress: 1/2", second_state)
        self.assertEqual(
            "newspaper",
            response.metadata["model_calls"][0]["public_state"]
            .split("target_class=", 1)[1]
            .split(";", 1)[0],
        )

    async def test_alfworld_public_state_tracks_visited_receptacles(self) -> None:
        class SearchSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = ("go to cabinet 1", "go to cabinet 2")

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action == "go to cabinet 1":
                    self._available = ("go to cabinet 2",)
                    return (
                        "You arrive at cabinet 1. In it, you see nothing.",
                        0.0,
                        False,
                        {"won": False},
                    )
                return "task terminal observation", 1.0, True, {"won": True}

        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Put some soapbar in cabinet.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        session = SearchSession()
        gateway = SequenceGateway(["go to cabinet 1", "go to cabinet 2"])
        runtime = resources(session=session, gateway=gateway, max_turns=2)

        await runtime.execution_adapter.execute(request)

        second_prompt = gateway.requests[1].problem
        self.assertIn(
            "Visited receptacles from public Action--Observation history",
            second_prompt,
        )
        self.assertIn("cabinet 1", second_prompt)
        self.assertIn("you see nothing", second_prompt)
        self.assertIn(
            "Currently admissible unvisited receptacles: cabinet 2.",
            second_prompt,
        )

    async def test_alfworld_visible_memory_retains_visited_destination_action(
        self,
    ) -> None:
        class DestinationSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = ("go to cart 1",)

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action == "go to cart 1":
                    self._available = ("go to shelf 2",)
                    return "You arrive at cart 1.", 0.0, False, {"won": False}
                if action == "go to shelf 2":
                    self._available = ("take soapbar 1 from shelf 2",)
                    return (
                        "You arrive at shelf 2 and see soapbar 1.",
                        0.0,
                        False,
                        {"won": False},
                    )
                if action.startswith("take "):
                    self._available = (
                        "go to cart 1",
                        "move soapbar 1 to shelf 2",
                    )
                    return "You pick up soapbar 1.", 0.0, False, {"won": False}
                return "You return to cart 1.", 0.0, False, {"won": False}

        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Put a soapbar in cart.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        session = DestinationSession()
        gateway = SequenceGateway(
            [
                "go to cart 1",
                "go to shelf 2",
                "take soapbar 1 from shelf 2",
                "go to cart 1",
            ]
        )
        runtime = resources(session=session, gateway=gateway, max_turns=4)

        await runtime.execution_adapter.execute(request)

        held_prompt = gateway.requests[3].problem
        self.assertIn("Current visible go targets: cart 1.", held_prompt)
        self.assertIn(
            "Current admissible strings mentioning the destination class: "
            "go to cart 1.",
            held_prompt,
        )
        self.assertIn("Objects implied as held by current actions: soapbar 1.", held_prompt)

    async def test_alfworld_public_goal_state_joins_transform_and_placement(
        self,
    ) -> None:
        class TransformSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = ("move soapbar 1 to cabinet 1",)

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action.startswith("move "):
                    self._available = ("finish",)
                    return (
                        "soapbar 1 is in cabinet 1",
                        0.0,
                        False,
                        {"won": False},
                    )
                return "still nonterminal", 0.0, False, {"won": False}

        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Put a clean soapbar in cabinet.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        gateway = SequenceGateway(["move soapbar 1 to cabinet 1", "finish"])
        runtime = resources(
            session=TransformSession(),
            gateway=gateway,
            max_turns=2,
        )

        response = await runtime.execution_adapter.execute(request)
        public_state = response.metadata["model_calls"][1]["public_state"]

        self.assertIn("Visible placement progress: 1/1", public_state)
        self.assertIn(
            "Visible required transform progress: no completed clean action",
            public_state,
        )
        self.assertIn("Visible transformed-and-placed progress: 0/1", public_state)

    async def test_alfworld_public_state_reports_visible_entity_binding_conflict(
        self,
    ) -> None:
        session = FakeSession()
        session._available = ("move potato 1 to fridge 1", "finish")
        gateway = SequenceGateway(["finish"])
        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem="Cool an apple and put it in sidetable.",
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        await runtime.execution_adapter.execute(request)

        self.assertIn(
            "Visible entity binding conflict: held object class(es)=potato; "
            "task target_class=apple.",
            gateway.requests[0].problem,
        )

    async def test_alfworld_revisit_does_not_claim_no_progress(self) -> None:
        class RevisitSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = ("go to cart 1",)

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action == "go to cart 1":
                    self._available = ("go to shelf 1",)
                    return "at cart", 0.0, False, {"won": False}
                self._available = ("go to cart 1",)
                return "at shelf", 0.0, False, {"won": False}

        session = RevisitSession()
        gateway = SequenceGateway(
            ["go to cart 1", "go to shelf 1", "go to cart 1", "go to shelf 1"]
        )
        runtime = resources(session=session, gateway=gateway, max_turns=4)

        await runtime.execution_adapter.execute(make_request())

        self.assertNotIn("No-progress signal", gateway.requests[3].problem)

    async def test_alfworld_public_progress_retracts_taken_placement(self) -> None:
        class PlacementSession(FakeSession):
            def __init__(self) -> None:
                super().__init__()
                self._available = ("move newspaper 2 to sidetable 1",)

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.actions.append(action)
                if action.startswith("move "):
                    self._available = (
                        "take newspaper 2 from sidetable 1",
                    )
                    return "newspaper 2 is on sidetable 1", 0.0, False, {
                        "won": False
                    }
                if action.startswith("take "):
                    self._available = ("finish",)
                    return "newspaper 2 is held", 0.0, False, {"won": False}
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
        session = PlacementSession()
        gateway = SequenceGateway(
            [
                "move newspaper 2 to sidetable 1",
                "take newspaper 2 from sidetable 1",
                "finish",
            ]
        )
        runtime = resources(session=session, gateway=gateway, max_turns=3)

        response = await runtime.execution_adapter.execute(request)

        self.assertIn(
            "Visible placement progress: 1/2",
            response.metadata["model_calls"][1]["public_state"],
        )
        self.assertIn(
            "Visible placement progress: 0/2",
            response.metadata["model_calls"][2]["public_state"],
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

        response = await runtime.execution_adapter.execute(request)

        public_state = response.metadata["model_calls"][0]["public_state"]
        self.assertIn("required_transform=heat", public_state)
        self.assertIn("`it` refers to `egg`", public_state)

    async def test_alfworld_public_state_ignores_appended_runtime_contract(self) -> None:
        request = make_request()
        request = AgentRequest(
            request_id=request.request_id,
            run_id=request.run_id,
            graph_revision=request.graph_revision,
            problem=(
                "Put two toiletpaper in toilet.\n\n"
                "Execution interface: use the request-scoped environment Tool."
            ),
            agent=request.agent,
            model=request.model,
            provider=request.provider,
            phase=request.phase,
            communication_condition=request.communication_condition,
        )
        session = FakeSession()
        gateway = SequenceGateway(["finish"])
        runtime = resources(session=session, gateway=gateway, max_turns=1)

        response = await runtime.execution_adapter.execute(request)

        model_state = response.metadata["model_calls"][0]["public_state"]
        self.assertIn("target_class=toiletpaper", model_state)
        self.assertIn("destination_class=toilet", model_state)
        self.assertIn("count=2", model_state)

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

    async def test_canvas_finish_requires_measured_terminal_transition(self) -> None:
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
                    '"allowed_tools":["alfworld"],'
                    '"execution_mode":"react",'
                    '"artifact_type":"environment_observation",'
                    '"completion_condition":"reach a terminal observation"}'
                    '],"relations":[],"output_agent_id":"actor"}'
                )
                self.assertTrue(added.accepted)
                self.assertIsNotNone(added.execution)

                finished = await canvas.step('{"action":"finish"}')

                self.assertEqual(expected_terminal, finished.accepted)
                self.assertEqual(expected_terminal, finished.done)
                if not expected_terminal:
                    self.assertIn(
                        "terminal Action--Observation transition",
                        finished.feedback,
                    )

    async def test_fresh_finish_caches_and_returns_nonterminal_execution(self) -> None:
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

        self.assertFalse(first.accepted)
        self.assertFalse(first.done)
        self.assertFalse(first.execution_reused)
        self.assertIsNotNone(first.execution)
        self.assertIsNone(first.partial_execution)
        self.assertEqual("room one", first.execution.outputs["actor"])
        actor_metadata = first.execution.output_metadata["actor"]
        self.assertFalse(actor_metadata["environment_terminal"])
        self.assertEqual(1, len(actor_metadata["environment_receipts"]))
        self.assertEqual(1, len(actor_metadata["tool_receipts"]))
        self.assertIn(
            "terminal Action--Observation transition",
            first.feedback,
        )

        second = await canvas.step('{"action":"finish"}')

        self.assertFalse(second.accepted)
        self.assertFalse(second.done)
        self.assertFalse(second.execution_reused)
        self.assertIsNone(second.execution)
        self.assertEqual("repeated_rejected_action", second.feedback_code)
        self.assertEqual(["look"], session.actions)
        self.assertEqual(1, len(gateway.requests))

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
        self.assertNotIn("Agent contract:", gateway.requests[0].problem)
        self.assertEqual(
            "environment_observation", gateway.requests[0].agent.artifact_type
        )
        self.assertEqual(
            "select an admissible environment action",
            gateway.requests[0].agent.contract,
        )

    async def test_alfworld_stepwise_calls_advance_one_shared_native_episode(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["<action>look</action>", "finish"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=2,
            stepwise_director=True,
        )
        request = make_request(
            "alfworld",
            tool_id="alfworld.environment",
        )

        first = await environment.execution_adapter.execute(request)
        second = await environment.execution_adapter.execute(request)

        self.assertEqual("alfworld.environment", environment.tool_id)
        self.assertEqual(1, session.reset_count)
        self.assertEqual(["look", "finish"], session.actions)
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
        first_state = first.metadata["environment_current_state"]
        second_state = second.metadata["environment_current_state"]
        self.assertEqual("look", first_state["last_action"])
        self.assertEqual(["finish"], first_state["admissible_actions"])
        self.assertEqual(1, first_state["remaining_action_budget"])
        self.assertFalse(first_state["environment_terminal"])
        self.assertEqual(1, len(first_state["action_observation_history"]))
        self.assertEqual(
            "room one",
            first_state["latest_action_observation"]["observation_result"],
        )
        self.assertEqual("finish", second_state["last_action"])
        self.assertEqual([], second_state["admissible_actions"])
        self.assertEqual(0, second_state["remaining_action_budget"])
        self.assertTrue(second_state["environment_terminal"])
        self.assertEqual(2, len(second_state["action_observation_history"]))
        self.assertEqual(
            ["look", "finish"],
            [
                item["action"]
                for item in second_state["action_observation_history"]
            ],
        )
        self.assertIn(
            "Recent executed actions: look.",
            gateway.requests[1].problem,
        )
        for public_value in (first_state, second_state):
            self.assertNotIn("reward", str(public_value))
            self.assertNotIn("won", str(public_value))
            self.assertNotIn("score", str(public_value))
        self.assertIn(
            "Pick exactly one action from the admissible actions list",
            gateway.requests[0].problem,
        )
        self.assertNotIn("search[", gateway.requests[0].problem)
        self.assertNotIn("click[", gateway.requests[0].problem)

    async def test_alfworld_canvas_continue_preserves_graph_revision(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look", "finish"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=2,
            stepwise_director=True,
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
            dataset_id="alfworld",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="complete the alfworld task",
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
        initial_add_domain = canvas.model_admissible_action_targets()["add_agent"]
        self.assertEqual(
            [
                {
                    "execution_mode": "react",
                    "allowed_tools": ["alfworld.environment"],
                }
            ],
            initial_add_domain["execution_profiles"],
        )
        added = await canvas.step(
            '{"action":"add_agent","agent_id":"actor","model_id":"m",'
            '"contract":"Act once from the current public ALFWorld state.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
        )
        self.assertTrue(added.accepted)
        owner_modify_domain = canvas.model_admissible_action_targets()[
            "modify_agent"
        ]["per_agent_candidates"]
        owner_fields = next(
            item["mutable_fields"]
            for item in owner_modify_domain
            if item["agent_id"] == "actor"
        )
        self.assertNotIn("allowed_tools", owner_fields)
        self.assertNotIn("execution_mode", owner_fields)
        later_add_profiles = canvas.model_admissible_action_targets()["add_agent"][
            "execution_profiles"
        ]
        self.assertNotIn(
            {
                "execution_mode": "react",
                "allowed_tools": ["alfworld.environment"],
            },
            later_add_profiles,
        )
        orchestrator = AgentGraphOrchestrator(
            registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=environment.tool_registry,
            max_rounds=4,
            system_prompt=director_system_prompt_for_version(
                STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION
            ),
            prompt_version=STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION,
            sampling_action_profile=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE
            ),
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(canvas, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        observation = json.loads(
            messages[-1]["content"].rpartition("\n\n")[2]
        )
        director_state = observation["environment_state"]
        self.assertEqual("look", director_state["last_action"])
        self.assertEqual(["finish"], director_state["admissible_actions"])
        self.assertEqual(1, director_state["environment_revision"])
        self.assertEqual(1, director_state["remaining_action_budget"])
        self.assertEqual(
            "complete the alfworld task",
            director_state["task_instruction"],
        )
        self.assertEqual(
            "look",
            director_state["latest_action_observation"]["action"],
        )
        self.assertIn("[PUBLIC OBSERVABLE STATE]", director_state["public_state"])
        self.assertNotIn("reward", str(director_state))
        self.assertNotIn("won", str(director_state))
        self.assertNotIn("score", str(director_state))
        selected = await canvas.step(
            '{"action":"set_output","agent_id":"actor"}'
        )
        self.assertTrue(selected.accepted)
        revision_before_continue = canvas.revision
        self.assertIn("continue", canvas.model_admissible_action_types())

        continued = await canvas.step('{"action":"continue"}')

        self.assertTrue(continued.accepted)
        self.assertEqual(revision_before_continue, canvas.revision)
        self.assertEqual(1, session.reset_count)
        self.assertEqual(["look", "finish"], session.actions)
        state = canvas.public_environment_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(
            "complete the alfworld task",
            state["task_instruction"],
        )
        self.assertEqual("finish", state["last_action"])
        self.assertTrue(state["environment_terminal"])
        self.assertNotIn("won", str(state))
        self.assertNotIn("continue", canvas.model_admissible_action_types())
        self.assertIn("finish", canvas.model_admissible_action_types())
        self.assertIn('"environment_state"', continued.feedback)
        self.assertNotIn('"won"', continued.feedback)
        finished = await canvas.step('{"action":"finish"}')
        self.assertTrue(finished.done)

    async def test_alfworld_repeated_parse_error_requires_contract_repair(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(
            [
                "move soapbar 1 to cart 1",
                "move soapbar 1 to cart 1",
            ]
        )
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
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
            dataset_id="alfworld",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Put a soapbar in cart.",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=(
                "add_agent",
                "modify_agent",
                "continue",
                "finish",
            ),
        )
        added = await canvas.step(
            '{"action":"add_agent","agent_id":"actor","model_id":"m",'
            '"contract":"Act once in ALFWorld.","execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
        )
        self.assertTrue(added.accepted)
        self.assertIn("continue", canvas.model_admissible_action_types())

        continued = await canvas.step('{"action":"continue"}')

        self.assertTrue(continued.accepted)
        self.assertEqual(
            ("modify_agent",),
            canvas.model_admissible_action_types(),
        )
        modify_domain = canvas.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["actor"], modify_domain["agent_ids"])
        self.assertEqual(["contract"], modify_domain["mutable_fields"])
        self.assertEqual(
            ["contract"],
            modify_domain["per_agent_candidates"][0]["mutable_fields"],
        )
        rejected = await canvas.step('{"action":"continue"}')
        self.assertFalse(rejected.accepted)
        self.assertIn("repair the existing environment Agent contract", rejected.feedback)
        self.assertEqual([], session.actions)

    async def test_alfworld_second_tool_owner_is_rejected_before_environment_step(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look", "finish"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=2,
            stepwise_director=True,
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
            dataset_id="alfworld",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="complete the alfworld task",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("add_agent", "modify_agent", "finish"),
        )
        owner = await canvas.step(
            '{"action":"add_agent","agent_id":"actor","model_id":"m",'
            '"contract":"Act once in ALFWorld.","execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
        )
        self.assertTrue(owner.accepted)

        removed_owner = await canvas.step(
            '{"action":"modify_agent","agent_id":"actor",'
            '"allowed_tools":[]}'
        )
        self.assertFalse(removed_owner.accepted)
        self.assertIn("must be preserved", removed_owner.feedback)
        self.assertEqual(["look"], session.actions)
        self.assertEqual(1, len(gateway.requests))

        duplicate = await canvas.step(
            '{"action":"add_agent","agent_id":"other","model_id":"m",'
            '"contract":"Also act in ALFWorld.","execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
        )

        self.assertFalse(duplicate.accepted)
        self.assertIn("at most one", duplicate.feedback)
        self.assertEqual(["look"], session.actions)
        self.assertEqual(1, len(gateway.requests))

    async def test_alfworld_stepwise_budget_closure_can_finish_but_is_not_terminal(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway(["look"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=1,
            stepwise_director=True,
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
            dataset_id="alfworld",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="complete the alfworld task",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("add_agent", "set_output", "continue", "finish"),
        )
        added = await canvas.step(
            '{"action":"add_agent","agent_id":"actor","model_id":"m",'
            '"contract":"Act once in ALFWorld.","execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
        )
        self.assertTrue(added.accepted)
        self.assertEqual(
            ("set_output",),
            canvas.model_admissible_action_types(),
        )
        selected = await canvas.step(
            '{"action":"set_output","agent_id":"actor"}'
        )
        self.assertTrue(selected.accepted)
        state = canvas.public_environment_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state["environment_truncated"])
        self.assertFalse(state["environment_terminal"])
        self.assertEqual(("finish",), canvas.model_admissible_action_types())

        finished = await canvas.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        assert finished.execution is not None
        metadata = finished.execution.output_metadata["actor"]
        self.assertFalse(metadata["environment_terminal"])
        self.assertTrue(metadata["environment_truncated"])
        self.assertFalse(metadata["evaluator_environment_trace"][-1]["done"])
        self.assertNotIn("won", str(canvas.public_environment_state()))

    async def test_alfworld_tool_holder_requires_react_and_exclusive_toolset(
        self,
    ) -> None:
        for execution_mode, allowed_tools, expected in (
            (
                "reasoning",
                '["alfworld.environment"]',
                "execution_mode='react'",
            ),
            (
                "react",
                '["alfworld.environment","unrelated.tool"]',
                "must allow exactly",
            ),
        ):
            with self.subTest(
                execution_mode=execution_mode,
                allowed_tools=allowed_tools,
            ):
                session = FakeSession()
                gateway = SequenceGateway([])
                environment = resources(
                    session=session,
                    gateway=gateway,
                    max_turns=2,
                    stepwise_director=True,
                )
                registry = ModelRegistry(
                    [ProviderSpec("fake", kind="test")],
                    [ModelSpec("m", "fake")],
                )
                runtime = AgentRuntime(
                    registry,
                    gateway,
                    execution_adapters={
                        "react": environment.execution_adapter,
                    },
                    tool_registry=environment.tool_registry,
                    dataset_id="alfworld",
                )
                canvas = AgentWorkflowEnv(
                    registry,
                    runtime=runtime,
                    problem="complete the alfworld task",
                    execute_on_edit=True,
                    required_tool_id=environment.tool_id,
                    allowed_actions=("add_agent", "finish"),
                )

                result = await canvas.step(
                    '{"action":"add_agent","agent_id":"actor",'
                    '"model_id":"m","contract":"Act in ALFWorld.",'
                    f'"execution_mode":"{execution_mode}",'
                    f'"allowed_tools":{allowed_tools}'
                    '}'
                )

                self.assertFalse(result.accepted)
                self.assertIn(expected, result.feedback)
                self.assertEqual([], session.actions)
                self.assertEqual([], gateway.requests)

    async def test_alfworld_owner_reciprocal_relation_is_not_admissible(
        self,
    ) -> None:
        session = FakeSession()
        gateway = SequenceGateway([])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=2,
            stepwise_director=True,
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
            dataset_id="alfworld",
        )
        graph = AgentGraph(
            (
                AgentNode(
                    "actor",
                    "m",
                    "Act once in ALFWorld.",
                    allowed_tools=("alfworld.environment",),
                    execution_mode="react",
                ),
                AgentNode("peer", "m", "Inspect public artifacts."),
            ),
            output_agent_id="actor",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="complete the alfworld task",
            graph=graph,
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("set_relation", "finish"),
        )
        reciprocal_candidates = [
            candidate
            for candidate in canvas.model_admissible_action_targets()[
                "set_relation"
            ]["candidates"]
            if candidate["source_to_target"]
            and candidate["target_to_source"]
        ]
        self.assertEqual([], reciprocal_candidates)

        relation = await canvas.step(
            '{"action":"set_relation","source_id":"actor",'
            '"target_id":"peer","source_to_target":true,'
            '"target_to_source":true}'
        )

        self.assertFalse(relation.accepted)
        self.assertIn("reciprocal Agent block", relation.feedback)
        self.assertEqual([], session.actions)
        self.assertEqual([], gateway.requests)

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
