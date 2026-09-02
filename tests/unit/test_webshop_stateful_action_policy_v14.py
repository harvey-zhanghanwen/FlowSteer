from __future__ import annotations

import json
import unittest

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
    _public_transition_summary,
    _webshop_model_visible_actions,
    build_environment_execution_resources,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def _structured_action(name: str, arguments: dict[str, str]) -> str:
    return json.dumps(
        {
            "resource_id": "webshop",
            "kind": "tool",
            "name": name,
            "arguments": arguments,
            "skill_id": None,
        }
    )


class _ScriptedGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("unexpected model call")
        return AgentResponse(
            self.outputs.pop(0),
            {"provider_request_id": f"provider-{len(self.requests)}"},
        )


class _WebShopSession:
    task_family = "webshop"

    def __init__(
        self,
        *,
        environment_id: str,
        observation: str,
        available_actions: dict[str, object],
    ) -> None:
        self.environment_id = environment_id
        self.observation = observation
        self._available = available_actions
        self.actions: list[str] = []
        self.reset_count = 0

    @property
    def available_actions(self) -> object:
        return self._available

    def reset(self) -> str:
        self.reset_count += 1
        return self.observation


class _ZeroResultSession(_WebShopSession):
    def __init__(self) -> None:
        super().__init__(
            environment_id="fake:webshop:zero-result",
            observation="WebShop [SEP] Search",
            available_actions={"has_search_bar": True, "clickables": []},
        )

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        self.observation = "WebShop [SEP] Page 1 (Total results: 0)"
        return self.observation, 0.0, False, {"graded_score": 0.0}


class _OptionProductSession(_WebShopSession):
    def __init__(self) -> None:
        super().__init__(
            environment_id="fake:webshop:required-option",
            observation=(
                "WebShop [SEP] < Prev [SEP] Color [SEP] Red [SEP] Blue "
                "[SEP] Demo Bottle [SEP] Price: $12 [SEP] Buy Now"
            ),
            available_actions={
                "has_search_bar": False,
                "clickables": ["Red", "Blue", "Buy Now", "Back to Search"],
            },
        )

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        if action.casefold() == "click[buy now]":
            return (
                "Thank you for shopping with us!",
                1.0,
                True,
                {"graded_score": 1.0},
            )
        return self.observation, 0.0, False, {"graded_score": 0.0}


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("m", "fake")],
    )


def _request(problem: str, *, run_id: str) -> AgentRequest:
    provider = ProviderSpec("fake", kind="test")
    model = ModelSpec("m", "fake")
    return AgentRequest(
        request_id=f"{run_id}:0:actor:single",
        run_id=run_id,
        graph_revision=0,
        problem=problem,
        agent=AgentNode(
            "actor",
            "m",
            "Act once from the current public WebShop state.",
            allowed_tools=("webshop.environment",),
            execution_mode="react",
            artifact_type="environment_observation",
        ),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        communication_condition=CommunicationCondition.NORMAL,
    )


def _resources(session: _WebShopSession, gateway: _ScriptedGateway, max_turns: int):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family="webshop",
        max_turns=max_turns,
        stepwise_director=True,
        structured_actions=True,
    )


class WebShopStatefulActionPolicyV14Tests(unittest.IsolatedAsyncioTestCase):
    async def test_known_zero_result_exact_search_is_not_executed_again(self) -> None:
        query = "exact uncommon blue bottle"
        same_search = _structured_action("search", {"query": query})
        session = _ZeroResultSession()
        gateway = _ScriptedGateway([same_search, same_search])
        environment = _resources(session, gateway, max_turns=3)
        request = _request(
            "Find a blue bottle, color: blue, and price lower than $20.",
            run_id="zero-result-repeat",
        )

        first = await environment.execution_adapter.execute(request)
        second = await environment.execution_adapter.execute(request)

        self.assertEqual([f"search[{query}]"], session.actions)
        first_state = first.metadata["environment_current_state"]
        self.assertEqual(
            {"query": query, "outcome": "zero_results"},
            first_state["public_progress"]["latest_search_outcome"],
        )
        second_state = second.metadata["environment_current_state"]
        self.assertEqual(1, second_state["environment_revision"])
        rejected = second.metadata["environment_receipts"][-1]
        self.assertFalse(rejected["state_advanced"])
        self.assertEqual("precondition_failed", rejected["observation_status"])
        self.assertEqual(
            "known_zero_result_query",
            rejected["precondition_failure_reason"],
        )
        self.assertEqual(
            "known_zero_result_query",
            second_state["public_progress"]["latest_transition"][
                "precondition_failure_reason"
            ],
        )
        self.assertEqual(request.problem, second_state["original_task_instruction"])
        self.assertIn(request.problem, gateway.requests[1].problem)
        self.assertIn("outcome=zero_results", gateway.requests[1].problem)
        self.assertFalse(second_state["environment_terminal"])

    def test_same_public_state_same_action_is_no_progress(self) -> None:
        transition = {
            "state_advanced": True,
            "action": "click[Features]",
            "observation": "Product page",
            "next_observation": "Product page",
            "observation_status": "success",
            "terminal": False,
        }

        summary = _public_transition_summary(
            task_family="webshop",
            observation="Product page",
            receipts=(transition, dict(transition)),
        )

        no_progress = summary["no_progress"]
        self.assertTrue(no_progress["detected"])
        self.assertIn("repeated_state_action", no_progress["reasons"])
        self.assertEqual(2, no_progress["repeated_state_action_count"])
        self.assertEqual(
            ["click[Features]"],
            summary["previous_actions_from_current_state"],
        )

    async def test_buy_now_is_masked_until_explicit_option_is_selected(self) -> None:
        session = _OptionProductSession()
        gateway = _ScriptedGateway(
            [_structured_action("click", {"target": "Buy Now"})]
        )
        environment = _resources(session, gateway, max_turns=2)
        request = _request(
            "Buy the Demo Bottle, color: blue, and price lower than $20.",
            run_id="missing-required-option",
        )

        response = await environment.execution_adapter.execute(request)

        self.assertEqual([], session.actions)
        state = response.metadata["environment_current_state"]
        self.assertNotIn(
            "click[Buy Now]", state["model_visible_admissible_actions"]
        )
        self.assertEqual(0, state["environment_revision"])
        self.assertFalse(state["environment_terminal"])
        self.assertFalse(response.metadata["environment_receipts"][-1]["state_advanced"])

    def test_incidental_option_word_does_not_narrow_task_scope(self) -> None:
        observation = (
            "WebShop [SEP] < Prev [SEP] Size [SEP] Small [SEP] Large "
            "[SEP] End Table [SEP] Price: $12 [SEP] Buy Now"
        )
        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=(
                "Find a small end table that is easy to assemble, "
                "and price lower than 20 dollars."
            ),
            observation=observation,
            receipts=(),
            native_actions=("click[Small]", "click[Large]", "click[Buy Now]"),
        )

        self.assertIn("click[Buy Now]", visible)

    def test_price_above_public_limit_masks_buy_now(self) -> None:
        observation = (
            "WebShop [SEP] < Prev [SEP] Demo Bottle [SEP] "
            "Price: $12 [SEP] Buy Now"
        )
        visible, _, _ = _webshop_model_visible_actions(
            task_instruction="Buy the Demo Bottle, and price lower than $10.",
            observation=observation,
            receipts=(),
            native_actions=("click[Buy Now]", "click[Back to Search]"),
        )

        self.assertNotIn("click[Buy Now]", visible)

    def test_price_equal_to_public_limit_keeps_buy_now(self) -> None:
        observation = (
            "WebShop [SEP] < Prev [SEP] Demo Bottle [SEP] "
            "Price: $10 [SEP] Buy Now"
        )
        visible, _, _ = _webshop_model_visible_actions(
            task_instruction="Buy the Demo Bottle, and price lower than $10.",
            observation=observation,
            receipts=(),
            native_actions=("click[Buy Now]", "click[Back to Search]"),
        )

        self.assertIn("click[Buy Now]", visible)

    def test_original_task_goal_is_preserved_in_every_director_state(self) -> None:
        session = _ZeroResultSession()
        gateway = _ScriptedGateway([])
        environment = _resources(session, gateway, max_turns=4)
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "actor",
                    "m",
                    "Act from the current public state.",
                    allowed_tools=(environment.tool_id,),
                    execution_mode="react",
                    artifact_type="environment_observation",
                )
            ],
            output_agent_id="actor",
        )
        original_goal = (
            "Buy the Demo Bottle, color: blue, and price lower than $20."
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem=(
                original_goal
                + "\n\nExecution interface: use the stateful environment tool."
            ),
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

        for revision, observation in ((1, "Search page"), (2, "Results page")):
            canvas._progressive_output_metadata["actor"] = {
                "environment_current_state": {
                    "environment_episode_id": "episode-1",
                    "environment_id": session.environment_id,
                    "task_family": "webshop",
                    "environment_revision": revision,
                    "last_action": f"search[q{revision}]",
                    "state_advanced": True,
                    "observation_status": "success",
                    "current_observation": observation,
                    "admissible_action_count": 2,
                    "public_progress": {
                        "latest_transition": {
                            "action": f"search[q{revision}]",
                            "state_advanced": True,
                        },
                        "no_progress": {
                            "detected": False,
                            "reasons": [],
                            "repeated_state_action_count": 1,
                            "action_cycle": False,
                        },
                    },
                    "turns_used": revision,
                    "remaining_action_budget": 4 - revision,
                    "environment_terminal": False,
                    "environment_truncated": False,
                }
            }

            state = canvas.public_environment_state()
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(original_goal, state["original_task_instruction"])
            self.assertEqual(revision, state["environment_revision"])
            self.assertEqual(observation, state["current_observation"])

    def test_no_progress_admits_repair_then_augmentation_not_delete_or_finish(
        self,
    ) -> None:
        session = _ZeroResultSession()
        gateway = _ScriptedGateway([])
        environment = _resources(session, gateway, max_turns=6)
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "actor",
                    "m",
                    "Act from public shopping evidence.",
                    allowed_tools=(environment.tool_id,),
                    execution_mode="react",
                    artifact_type="environment_observation",
                )
            ],
            output_agent_id="actor",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Buy the requested product.",
            max_agents=3,
            required_tool_id=environment.tool_id,
            recovery_policy="preserve_diagnose_repair_augment",
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

        raw_state = {
            "environment_episode_id": "episode-1",
            "environment_id": session.environment_id,
            "task_family": "webshop",
            "environment_revision": 2,
            "last_action": "click[Features]",
            "state_advanced": False,
            "observation_status": "precondition_failed",
            "current_observation": "Product page",
            "admissible_action_count": 3,
            "public_progress": {
                "latest_transition": {
                    "action": "click[Features]",
                    "state_advanced": False,
                },
                "no_progress": {
                    "detected": True,
                    "reasons": ["repeated_state_action"],
                    "repeated_state_action_count": 2,
                    "action_cycle": False,
                },
            },
            "turns_used": 2,
            "remaining_action_budget": 4,
            "environment_terminal": False,
            "environment_truncated": False,
        }
        canvas._progressive_output_metadata["actor"] = {
            "environment_current_state": raw_state
        }

        repair_actions = set(canvas.model_admissible_action_types())
        # Two measured repeats already establish a public no-progress fault.
        # Keep in-place repair available while opening augmentation before the
        # bounded WebShop action budget is almost exhausted.
        self.assertEqual({"modify_agent", "add_agent"}, repair_actions)
        self.assertTrue(canvas.graph.has_node("actor"))
        self.assertNotIn("delete_agent", repair_actions)
        self.assertNotIn("finish", repair_actions)
        self.assertNotIn("continue", repair_actions)

        raw_state["public_progress"]["no_progress"][
            "repeated_state_action_count"
        ] = 3
        augment_actions = set(canvas.model_admissible_action_types())
        self.assertIn("modify_agent", augment_actions)
        self.assertIn("add_agent", augment_actions)
        self.assertNotIn("delete_agent", augment_actions)
        self.assertNotIn("finish", augment_actions)
        self.assertNotIn("continue", augment_actions)
        self.assertTrue(canvas.graph.has_node("actor"))


if __name__ == "__main__":
    unittest.main()
