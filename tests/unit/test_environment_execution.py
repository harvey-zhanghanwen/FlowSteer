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
from src.interactive.agent_workflow_env import AgentWorkflowEnv
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


def make_request(task_family: str = "alfworld") -> AgentRequest:
    provider = ProviderSpec("fake", kind="test")
    model = ModelSpec("m", "fake")
    tool_id = "alfworld" if task_family == "alfworld" else f"{task_family}.environment"
    return AgentRequest(
        request_id="run:0:actor:single",
        run_id="run",
        graph_revision=0,
        problem=f"complete the {task_family} task",
        agent=AgentNode(
            "actor",
            "m",
            "select an admissible environment action",
            allowed_tools=(tool_id,),
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
):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family=session.task_family,
        max_turns=max_turns,
        max_observation_chars=max_observation_chars,
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
