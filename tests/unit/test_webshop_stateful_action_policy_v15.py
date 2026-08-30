from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    CommunicationCondition,
    ExecutionPhase,
)
from src.interactive.environment_execution import (
    _action_prompt,
    _public_state_feedback,
    _public_transition_summary,
    _webshop_current_product_evidence,
    _webshop_model_visible_actions,
    _webshop_option_targets,
    _webshop_purchase_preconditions,
    build_environment_execution_resources,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec


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


def _transition(
    turn: int,
    action: str,
    observation: str,
    next_observation: str,
) -> dict[str, object]:
    return {
        "receipt_type": "environment_transition",
        "turn": turn,
        "action": action,
        "observation": observation,
        "next_observation": next_observation,
        "observation_status": "success",
        "state_advanced": True,
        "terminal": False,
    }


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


class _SearchSession:
    environment_id = "fake:webshop:search-repeat"
    task_family = "webshop"

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.reset_count = 0
        self.observation = "WebShop [SEP] Search"
        self._available: dict[str, object] = {
            "has_search_bar": True,
            "clickables": ["Search"],
        }

    @property
    def available_actions(self) -> object:
        return self._available

    def reset(self) -> str:
        self.reset_count += 1
        return self.observation

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        if action.casefold() == "click[back to search]":
            self.observation = "WebShop [SEP] Search"
            self._available = {
                "has_search_bar": True,
                "clickables": ["Search"],
            }
        else:
            self.observation = (
                "WebShop [SEP] Page 1 (Total results: 50) "
                "[SEP] Back to Search [SEP] B000000001"
            )
            self._available = {
                "has_search_bar": False,
                "clickables": ["Back to Search", "B000000001"],
            }
        return self.observation, 0.0, False, {"graded_score": 0.0}


def _resources(session: _SearchSession, gateway: _ScriptedGateway, max_turns: int):
    return build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family="webshop",
        max_turns=max_turns,
        stepwise_director=True,
        structured_actions=True,
    )


class WebShopStatefulActionPolicyV15Tests(unittest.IsolatedAsyncioTestCase):
    def test_matching_option_protects_group_but_mismatch_keeps_repair(self) -> None:
        product_page = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Color [SEP] Red [SEP] Blue "
            "[SEP] Demo Bottle [SEP] Price: $12 [SEP] Buy Now"
        )
        results = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 1) [SEP] B000000001"
        )
        goal = "Buy the Demo Bottle, color: Blue, and price lower than $20."
        opened = _transition(
            1, "click[B000000001]", results, product_page
        )
        native_actions = (
            "click[Red]",
            "click[Blue]",
            "click[Buy Now]",
            "click[Back to Search]",
        )

        matching_visible, _, matching_selected = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product_page,
            receipts=(
                opened,
                _transition(2, "click[Blue]", product_page, product_page),
            ),
            native_actions=native_actions,
        )

        self.assertEqual({"color": "Blue"}, matching_selected)
        self.assertNotIn("click[Blue]", matching_visible)
        self.assertNotIn("click[Red]", matching_visible)
        self.assertIn("click[Buy Now]", matching_visible)
        self.assertIn("click[Back to Search]", matching_visible)

        repair_visible, _, mismatched_selected = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product_page,
            receipts=(
                opened,
                _transition(2, "click[Red]", product_page, product_page),
            ),
            native_actions=native_actions,
        )

        self.assertEqual({"color": "Red"}, mismatched_selected)
        self.assertNotIn("click[Red]", repair_visible)
        self.assertIn("click[Blue]", repair_visible)
        self.assertNotIn("click[Buy Now]", repair_visible)

    def test_purchase_lower_bound_and_prompt_budget_on_product_and_detail(
        self,
    ) -> None:
        product_page = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Color [SEP] Red [SEP] Blue "
            "[SEP] Demo Bottle [SEP] Price: $12 [SEP] Description "
            "[SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        results = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 1) [SEP] B000000001"
        )
        features = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Public product feature text"
        )
        goal = "Buy the Demo Bottle, color: Blue, and price lower than $20."
        request = _request(goal, run_id="purchase-lower-bound")
        opened = _transition(
            1, "click[B000000001]", results, product_page
        )
        feature_receipts = (
            opened,
            _transition(2, "click[Features]", product_page, features),
        )

        product_purchase = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=product_page,
            receipts=(opened,),
        )
        detail_purchase = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=features,
            receipts=feature_receipts,
        )

        # Product page: select the one missing option, then Buy Now.
        self.assertEqual(2, product_purchase["minimum_actions_to_purchase"])
        # Detail subpage: < Prev, select the missing option, then Buy Now.
        self.assertEqual(3, detail_purchase["minimum_actions_to_purchase"])

        product_prompt = _action_prompt(
            request,
            task_family="webshop",
            observation=product_page,
            admissible_actions=(
                "click[Red]",
                "click[Blue]",
                "click[Back to Search]",
            ),
            receipts=(opened,),
            turn=2,
            structured_actions=True,
            remaining_action_budget=2,
        )
        detail_prompt = _action_prompt(
            request,
            task_family="webshop",
            observation=features,
            admissible_actions=("click[Back to Search]", "click[< Prev]"),
            receipts=feature_receipts,
            turn=3,
            structured_actions=True,
            remaining_action_budget=3,
        )

        self.assertIn(
            "Minimum environment actions required to purchase the current "
            "candidate from this public state: 2.",
            product_prompt,
        )
        self.assertIn(
            "Environment action budget remaining (including the next Action): 2",
            product_prompt,
        )
        self.assertIn(
            "Minimum environment actions required to purchase the current "
            "candidate from this public state: 3.",
            detail_prompt,
        )
        self.assertIn(
            "Environment action budget remaining (including the next Action): 3",
            detail_prompt,
        )
        for prompt in (product_prompt, detail_prompt):
            lowered = prompt.casefold()
            self.assertNotIn("graded_score", lowered)
            self.assertNotIn("evaluator", lowered)
            self.assertNotIn("reward", lowered)
            self.assertNotIn("'won'", lowered)

    def test_product_subpages_recover_public_product_evidence(self) -> None:
        product_page = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Color [SEP] Red [SEP] Blue "
            "[SEP] Demo Bottle [SEP] Price: $12 [SEP] Description "
            "[SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        result_page = (
            "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 1) [SEP] B000000001"
        )
        goal = "Buy the Demo Bottle, color: Blue, and price lower than $20."

        for tab in ("Description", "Features"):
            with self.subTest(tab=tab):
                subpage = (
                    "Instruction: [SEP] Buy the Demo Bottle [SEP] Back to Search "
                    f"[SEP] < Prev [SEP] {tab} text from the selected product"
                )
                receipts = (
                    _transition(
                        1,
                        "click[B000000001]",
                        result_page,
                        product_page,
                    ),
                    _transition(2, "click[Blue]", product_page, product_page),
                    _transition(3, f"click[{tab}]", product_page, subpage),
                )

                evidence = _webshop_current_product_evidence(subpage, receipts)

                self.assertTrue(evidence["available"])
                self.assertTrue(evidence["in_product_scope"])
                self.assertFalse(evidence["purchase_action_visible"])
                self.assertEqual("b000000001", evidence["asin"])
                self.assertEqual("Demo Bottle", evidence["title"])
                self.assertEqual(12.0, evidence["price"])
                self.assertEqual(
                    {"color": ["Red", "Blue"]},
                    evidence["visible_option_groups"],
                )
                self.assertEqual(
                    {"color": "Blue"}, evidence["selected_options"]
                )
                self.assertIn(tab.casefold(), evidence["inspected_tabs"])

                purchase = _webshop_purchase_preconditions(
                    task_instruction=goal,
                    observation=subpage,
                    receipts=receipts,
                )
                self.assertTrue(purchase["product_context_available"])
                self.assertEqual(12.0, purchase["product_price"])
                self.assertFalse(purchase["price_evidence_missing"])

                feedback = _public_state_feedback(
                    _request(goal, run_id=f"subpage-{tab.casefold()}"),
                    task_family="webshop",
                    observation=subpage,
                    admissible_actions=("click[Back to Search]", "click[< Prev]"),
                    receipts=receipts,
                )
                self.assertIn("Current product title: Demo Bottle.", feedback)
                self.assertNotIn("price_evidence_missing=True", feedback)
                self.assertNotIn("Purchase preconditions not satisfied", feedback)

    def test_start_and_results_pages_do_not_render_purchase_blocker(self) -> None:
        goal = "Buy a blue bottle, and price lower than $20."
        request = _request(goal, run_id="non-product-pages")
        start = "WebShop [SEP] Search"
        results = (
            "WebShop [SEP] Page 1 (Total results: 50) "
            "[SEP] Back to Search [SEP] B000000001"
        )
        search_receipt = _transition(
            1, "search[blue bottle]", start, results
        )

        for observation, receipts, actions in (
            (start, (), ("search[<your query>]", "click[Search]")),
            (
                results,
                (search_receipt,),
                ("click[Back to Search]", "click[B000000001]"),
            ),
        ):
            with self.subTest(observation=observation):
                feedback = _public_state_feedback(
                    request,
                    task_family="webshop",
                    observation=observation,
                    admissible_actions=actions,
                    receipts=receipts,
                )
                summary = _public_transition_summary(
                    task_family="webshop",
                    task_instruction=goal,
                    observation=observation,
                    receipts=receipts,
                )

                self.assertNotIn("Purchase preconditions not satisfied", feedback)
                purchase = summary.get("purchase_preconditions")
                self.assertIsInstance(purchase, dict)
                assert isinstance(purchase, dict)
                self.assertFalse(purchase["product_context_available"])
                self.assertFalse(purchase["price_evidence_missing"])
                self.assertIsNone(summary.get("current_product"))

    async def test_normalized_nonzero_search_repeat_is_typed_rejection(self) -> None:
        first_search = _structured_action(
            "search", {"query": "Blue Bottle"}
        )
        back = _structured_action("click", {"target": "Back to Search"})
        normalized_repeat = _structured_action(
            "search", {"query": "  BLUE---bottle  "}
        )
        session = _SearchSession()
        gateway = _ScriptedGateway([first_search, back, normalized_repeat])
        environment = _resources(session, gateway, max_turns=4)
        goal = "Find a blue bottle, and price lower than $20."
        request = _request(goal, run_id="normalized-query-repeat")

        first = await environment.execution_adapter.execute(request)
        second = await environment.execution_adapter.execute(request)
        third = await environment.execution_adapter.execute(request)

        self.assertEqual(
            ["search[Blue Bottle]", "click[Back to Search]"],
            session.actions,
        )
        self.assertEqual(1, first.metadata["environment_current_state"]["environment_revision"])
        self.assertEqual(2, second.metadata["environment_current_state"]["environment_revision"])
        rejected = third.metadata["environment_receipts"][-1]
        self.assertEqual("precondition_failed", rejected["observation_status"])
        self.assertEqual(
            "repeated_search_query",
            rejected["precondition_failure_reason"],
        )
        self.assertFalse(rejected["state_advanced"])
        self.assertEqual(2, third.metadata["environment_current_state"]["environment_revision"])
        self.assertIn(
            "repeated_search_query",
            third.metadata["environment_current_state"]["public_progress"][
                "no_progress"
            ]["reasons"],
        )

    def test_public_attribute_binding_covers_color_flavor_and_units(self) -> None:
        cases = (
            (
                "color",
                (
                    "Choose curtains in the color plaid red black, "
                    "and price lower than 70 dollars."
                ),
                {"color": ["Blue", "Plaid Red Black"]},
                {"color": ["Plaid Red Black"]},
            ),
            (
                "flavor",
                "Buy ready-to-eat Kashmir Potatoes flavor.",
                {"flavor name": ["Original", "Kashmir Potatoes"]},
                {"flavor name": ["Kashmir Potatoes"]},
            ),
            (
                "unit",
                "Buy the 6 ounce bottle.",
                {"size": ["6 Ounce", "12 Ounce"]},
                {"size": ["6 Ounce"]},
            ),
            (
                "size",
                (
                    "I am looking for a 7.5 no. high heel for women, "
                    "and price lower than 140 dollars."
                ),
                {"size": ["7.5", "8"]},
                {"size": ["7.5"]},
            ),
        )

        for label, task, groups, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    expected,
                    _webshop_option_targets(task, groups),
                )

        self.assertEqual(
            {},
            _webshop_option_targets(
                "Find a small end table that is easy to assemble.",
                {"size": ["Small", "Large"]},
            ),
        )

    async def test_original_goal_is_present_after_every_transition(self) -> None:
        session = _SearchSession()
        gateway = _ScriptedGateway(
            [
                _structured_action("search", {"query": "blue bottle"}),
                _structured_action("click", {"target": "Back to Search"}),
                _structured_action("search", {"query": "blue glass bottle"}),
            ]
        )
        environment = _resources(session, gateway, max_turns=4)
        goal = "Find a blue glass bottle, and price lower than $20."
        request = _request(goal, run_id="goal-every-transition")

        responses = [
            await environment.execution_adapter.execute(request),
            await environment.execution_adapter.execute(request),
            await environment.execution_adapter.execute(request),
        ]

        self.assertEqual(3, len(gateway.requests))
        for response, generated_request in zip(responses, gateway.requests):
            state = response.metadata["environment_current_state"]
            self.assertEqual(goal, state["original_task_instruction"])
            self.assertIn(f"Original task instruction:\n{goal}", generated_request.problem)


if __name__ == "__main__":
    unittest.main()
