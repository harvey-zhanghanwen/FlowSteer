from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.interactive.environment_execution import (
    _action_prompt,
    _public_state_feedback,
    _public_transition_summary,
    _webshop_model_visible_actions,
    _webshop_purchase_preconditions,
    _webshop_clicked_option_assignments,
    RAGENEnvironmentSession,
)
from tests.unit.test_webshop_stateful_action_policy_v15 import (
    _request,
    _resources,
    _ScriptedGateway,
    _SearchSession,
    _structured_action,
    _transition,
)


class WebShopConstraintCoverageV50Tests(unittest.IsolatedAsyncioTestCase):
    """Protect the v16 native action boundary with public-only fixtures."""

    product_page = (
        "WebShop [SEP] Back to Search [SEP] < Prev "
        "[SEP] Travel Bottle [SEP] Price: $12 "
        "[SEP] Description [SEP] Features [SEP] Buy Now"
    )
    product_actions = (
        "click[Back to Search]",
        "click[< Prev]",
        "click[Description]",
        "click[Features]",
        "click[Buy Now]",
    )

    def test_local_purchase_preconditions_do_not_claim_complete_goal_coverage(self) -> None:
        goal = (
            "Find a blue bottle with a ceramic lining and a carrying sleeve "
            "in a two-pack, and price lower than 25 dollars."
        )
        purchase = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=self.product_page,
            receipts=(),
        )

        # A price match and no missing parsed option are not evidence for
        # color, material or bundle contents absent from the public page.
        self.assertTrue(purchase["admissible"])
        self.assertEqual({}, purchase["required_option_targets"])
        self.assertEqual(
            "visible_option_bindings_and_price_only", purchase["validation_scope"]
        )
        self.assertEqual("partial", purchase["task_requirement_coverage"])
        self.assertEqual("unverified", purchase["task_satisfaction"])
        self.assertEqual(goal, purchase["original_task_instruction"])

    def test_public_option_binding_preserves_purchase_and_navigation(self) -> None:
        goal = "Buy a demo bottle, color: Blue, and price lower than 20 dollars."
        product_page = (
            "WebShop [SEP] Back to Search [SEP] < Prev "
            "[SEP] Color [SEP] Red [SEP] Blue "
            "[SEP] Demo Bottle [SEP] Price: $12 "
            "[SEP] Description [SEP] Features [SEP] Buy Now"
        )
        receipts = (_transition(1, "click[Blue]", product_page, product_page),)
        purchase = _webshop_purchase_preconditions(
            task_instruction=goal, observation=product_page, receipts=receipts
        )
        visible, _, selected = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product_page,
            receipts=receipts,
            native_actions=self.product_actions + ("click[Red]", "click[Blue]"),
        )

        self.assertTrue(purchase["admissible"])
        self.assertEqual({"color": "Blue"}, selected)
        self.assertEqual([], purchase["missing_option_groups"])
        self.assertEqual("unverified", purchase["task_satisfaction"])
        for action in self.product_actions:
            self.assertIn(action, visible)

    def test_one_action_remaining_does_not_force_purchase_or_remove_search(self) -> None:
        goal = "Find a bottle, and price lower than 25 dollars."
        request = _request(goal, run_id="budget-is-feedback-only")
        for observation, native_actions in (
            (self.product_page, self.product_actions),
            ("WebShop [SEP] Search", ("search[<your query>]",)),
        ):
            visible, _, _ = _webshop_model_visible_actions(
                task_instruction=goal,
                observation=observation,
                receipts=(),
                native_actions=native_actions,
            )
            self.assertEqual(native_actions, visible)
            for budget in (1, 6):
                with self.subTest(observation=observation, budget=budget):
                    prompt = _action_prompt(
                        request,
                        task_family="webshop",
                        observation=observation,
                        admissible_actions=visible,
                        receipts=(),
                        turn=1,
                        structured_actions=True,
                        remaining_action_budget=budget,
                    )
                    action_block = prompt.split("Admissible actions:\n", 1)[1].split(
                        "\n\n", 1
                    )[0]
                    self.assertEqual("\n".join(native_actions), action_block)
                    self.assertIn(
                        "Environment action budget remaining (including the next "
                        f"Action): {budget}",
                        prompt,
                    )
                    self.assertIn(goal, prompt)

    def test_complete_original_goal_reaches_feedback_and_public_progress(self) -> None:
        goal = (
            "Find a blue bottle with a ceramic lining. The bundle must include "
            "two bottles and carrying sleeves, and price lower than 25 dollars."
        )
        request = _request(goal, run_id="original-scope-retained")
        progress = _public_transition_summary(
            task_family="webshop",
            task_instruction=goal,
            observation=self.product_page,
            receipts=(),
        )
        feedback = _public_state_feedback(
            request,
            task_family="webshop",
            observation=self.product_page,
            admissible_actions=self.product_actions,
            receipts=(),
        )

        self.assertEqual(
            goal, progress["purchase_preconditions"]["original_task_instruction"]
        )
        self.assertEqual(
            "partial", progress["purchase_preconditions"]["task_requirement_coverage"]
        )
        self.assertEqual(
            "unverified", progress["purchase_preconditions"]["task_satisfaction"]
        )
        self.assertIn(f"Original task goal: {goal}", feedback)

    async def test_legal_repeated_search_reaches_native_environment_and_counts_budget(self) -> None:
        query = "blue bottle"
        goal = "Find a blue bottle, and price lower than 25 dollars."
        session = _SearchSession()
        gateway = _ScriptedGateway(
            [
                _structured_action("search", {"query": query}),
                _structured_action("click", {"target": "Back to Search"}),
                _structured_action("search", {"query": query}),
            ]
        )
        environment = _resources(session, gateway, max_turns=4)
        request = _request(goal, run_id="native-search-revisit")

        responses = [
            await environment.execution_adapter.execute(request) for _ in range(3)
        ]

        self.assertEqual(
            ["search[blue bottle]", "click[Back to Search]", "search[blue bottle]"],
            session.actions,
        )
        self.assertEqual(1, session.reset_count)
        for revision, response in enumerate(responses, start=1):
            with self.subTest(revision=revision):
                state = response.metadata["environment_current_state"]
                receipt = response.metadata["environment_receipts"][-1]
                self.assertEqual(revision, state["environment_revision"])
                self.assertEqual(revision, state["turns_used"])
                self.assertEqual(4 - revision, state["remaining_action_budget"])
                self.assertEqual(goal, state["original_task_instruction"])
                self.assertTrue(receipt["state_advanced"])
                self.assertEqual("success", receipt["observation_status"])
                self.assertIsNone(receipt.get("precondition_failure_reason"))
                self.assertNotEqual("<INVALID>", receipt["action"])
                self.assertIn(goal, gateway.requests[revision - 1].problem)
        # The second identical public transition remains observable as
        # no-progress evidence; preserving legality does not erase its history.
        self.assertIn(
            "repeated_state_action",
            responses[-1].metadata["environment_current_state"]["public_progress"][
                "no_progress"
            ]["reasons"],
        )

    def test_public_radio_receipt_binds_only_the_clicked_native_group(self) -> None:
        native = SimpleNamespace(text_to_clickable={
            "regular": {"type": "radio", "name": "Fit", "value": "Regular"}
        })
        adapter = SimpleNamespace(
            _env=SimpleNamespace(env=native),
            reset=lambda *args, **kwargs: self.product_page,
            step=lambda action: (self.product_page, 0, False, {}),
        )
        session = RAGENEnvironmentSession(adapter, "webshop", {}, "Buy a shirt", {})
        _, _, _, info = session.step("click[REGULAR]")
        self.assertEqual({"group": "fit", "value": "Regular"}, info["public_option_assignment"])
        receipt = {
            "state_advanced": True,
            "action": "click[REGULAR]",
            "public_option_assignment": info["public_option_assignment"],
        }
        self.assertEqual(
            {"fit": "Regular"},
            _webshop_clicked_option_assignments(
                (receipt,), {"fit": ("Regular",), "length": ("Regular",)}
            ),
        )

    def test_only_provably_empty_next_page_is_filtered(self) -> None:
        native_actions = (
            "click[Next >]", "click[< Prev]", "click[Back to Search]",
            "click[B012345678]", "search[<your query>]",
        )
        for page in (4, 5):
            visible, _, _ = _webshop_model_visible_actions(
                task_instruction="Find a bottle",
                observation=f"WebShop [SEP] Page {page} (Total results: 50)",
                receipts=(), native_actions=native_actions,
            )
            self.assertEqual(page == 4, "click[Next >]" in visible)
            for action in native_actions[1:]:
                self.assertIn(action, visible)


if __name__ == "__main__":
    unittest.main()
