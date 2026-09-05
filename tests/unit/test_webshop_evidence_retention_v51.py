from __future__ import annotations

import json
import unittest

from src.interactive.environment_execution import (
    _action_prompt,
    _prompt_observation,
    _public_state_feedback,
    _public_transition_summary,
    _webshop_current_product_evidence,
)
from tests.unit import test_webshop_constraint_coverage_v50 as _v50_fixtures
from tests.unit.test_webshop_stateful_action_policy_v15 import _request, _transition


class WebShopEvidenceRetentionV51Tests(unittest.TestCase):
    """Retain SkillFlow-style public ReAct history across Director steps.

    Source: SkillFlow training/environment.py::_build_react_prompt retains
    WebShop observation/action history. The project adaptation bounds product
    evidence to the current native ASIN visit, without reading evaluator info.
    These fixtures are synthetic and never initialize a model or environment.
    """

    product_page = _v50_fixtures.WebShopConstraintCoverageV50Tests.product_page
    product_actions = _v50_fixtures.WebShopConstraintCoverageV50Tests.product_actions
    goal = (
        "Find a blue travel bottle with a ceramic lining and carrying sleeve, "
        "in a two-pack, and price lower than 25 dollars."
    )
    results = (
        "WebShop [SEP] Back to Search [SEP] Page 1 (Total results: 2) "
        "[SEP] B000000001 [SEP] Travel Bottle "
        "[SEP] B000000002 [SEP] Storage Jar"
    )
    features = (
        "WebShop [SEP] Back to Search [SEP] < Prev "
        "[SEP] Ceramic lining; blue finish; two bottles per package."
    )
    description = (
        "WebShop [SEP] Back to Search [SEP] < Prev "
        "[SEP] Each bottle includes a carrying sleeve."
    )

    def _inspected_receipts(self) -> list[dict[str, object]]:
        return [
            _transition(1, "click[B000000001]", self.results, self.product_page),
            _transition(2, "click[Features]", self.product_page, self.features),
            _transition(3, "click[< Prev]", self.features, self.product_page),
            _transition(
                4, "click[Description]", self.product_page, self.description
            ),
            _transition(5, "click[< Prev]", self.description, self.product_page),
        ]

    def test_detail_observations_survive_return_to_the_same_product(self) -> None:
        evidence = _webshop_current_product_evidence(
            self.product_page, self._inspected_receipts()
        )

        self.assertEqual("b000000001", evidence["asin"])
        self.assertTrue(evidence["purchase_action_visible"])
        self.assertEqual(
            {"features": self.features, "description": self.description},
            evidence["inspected_tab_observations"],
        )

    def test_current_detail_page_retains_the_other_read_tab(self) -> None:
        evidence = _webshop_current_product_evidence(
            self.description, self._inspected_receipts()[:-1]
        )

        self.assertEqual("b000000001", evidence["asin"])
        self.assertTrue(evidence["in_product_scope"])
        self.assertFalse(evidence["purchase_action_visible"])
        self.assertEqual(
            {"features": self.features, "description": self.description},
            evidence["inspected_tab_observations"],
        )

    def test_latest_successful_tab_observation_wins_over_failed_attempt(self) -> None:
        revised_features = (
            "WebShop [SEP] Back to Search [SEP] < Prev "
            "[SEP] Revised public Features: ceramic lining; blue two-pack."
        )
        receipts = self._inspected_receipts() + [
            _transition(6, "click[Features]", self.product_page, revised_features),
            _transition(7, "click[< Prev]", revised_features, self.product_page),
            {
                **_transition(
                    8,
                    "click[Features]",
                    self.product_page,
                    "NOT_AN_EXECUTED_OBSERVATION",
                ),
                "state_advanced": False,
                "observation_status": "provider_failure",
            },
        ]
        evidence = _webshop_current_product_evidence(self.product_page, receipts)

        self.assertEqual(
            {"features": revised_features, "description": self.description},
            evidence["inspected_tab_observations"],
        )
        self.assertNotIn("NOT_AN_EXECUTED_OBSERVATION", json.dumps(evidence))

    def test_failed_navigation_does_not_rebind_or_clear_current_evidence(self) -> None:
        for action in (
            "search[storage jar]",
            "click[Back to Search]",
            "click[B000000002]",
        ):
            with self.subTest(action=action):
                rejected = {
                    **_transition(6, action, self.product_page, self.product_page),
                    "state_advanced": False,
                    "observation_status": "invalid_action",
                }
                evidence = _webshop_current_product_evidence(
                    self.product_page, self._inspected_receipts() + [rejected]
                )
                self.assertEqual("b000000001", evidence["asin"])
                self.assertEqual(
                    {"features": self.features, "description": self.description},
                    evidence["inspected_tab_observations"],
                )

    def test_new_asin_never_inherits_previous_product_tab_observations(self) -> None:
        other_product = self.product_page.replace("Travel Bottle", "Storage Jar")
        other_features = (
            "WebShop [SEP] Back to Search [SEP] < Prev "
            "[SEP] Clear glass storage jar; sold individually."
        )
        receipts = self._inspected_receipts() + [
            _transition(6, "click[< Prev]", self.product_page, self.results),
            _transition(7, "click[B000000002]", self.results, other_product),
            _transition(8, "click[Features]", other_product, other_features),
            _transition(9, "click[< Prev]", other_features, other_product),
        ]
        evidence = _webshop_current_product_evidence(other_product, receipts)

        self.assertEqual("b000000002", evidence["asin"])
        self.assertEqual(
            {"features": other_features}, evidence["inspected_tab_observations"]
        )
        self.assertNotIn(self.features, json.dumps(evidence))
        self.assertNotIn(self.description, json.dumps(evidence))

    def test_native_search_or_back_ends_the_product_evidence_scope(self) -> None:
        for action, next_observation in (
            ("search[storage jar]", self.results),
            ("click[Back to Search]", "WebShop [SEP] Search"),
        ):
            with self.subTest(action=action):
                receipts = self._inspected_receipts() + [
                    _transition(6, action, self.product_page, next_observation)
                ]
                evidence = _webshop_current_product_evidence(
                    next_observation, receipts
                )
                self.assertEqual({}, evidence["inspected_tab_observations"])
                self.assertIsNone(evidence["asin"])

                # Reopening the same ASIN is a new native visit; do not silently
                # claim that tabs from the previous visit were inspected now.
                reopened = _webshop_current_product_evidence(
                    self.product_page,
                    receipts
                    + [
                        _transition(
                            7,
                            "click[B000000001]",
                            next_observation,
                            self.product_page,
                        )
                    ],
                )
                self.assertEqual("b000000001", reopened["asin"])
                self.assertEqual({}, reopened["inspected_tab_observations"])

    def test_full_read_evidence_and_original_goal_reach_public_model_inputs(self) -> None:
        receipts = self._inspected_receipts()
        # Move the evidence outside the short recent-action/history window.
        receipts.extend(
            _transition(turn, "click[Blue]", self.product_page, self.product_page)
            for turn in range(6, 18)
        )
        for receipt in receipts:
            receipt["info"] = {"hidden_goal": "PRIVATE_GOAL_SENTINEL"}
            receipt["reward"] = "PRIVATE_REWARD_SENTINEL"
            receipt["evaluator"] = {"answer": "PRIVATE_EVALUATOR_SENTINEL"}

        request = _request(self.goal, run_id="retained-public-evidence")
        summary = _public_transition_summary(
            task_family="webshop",
            task_instruction=self.goal,
            observation=self.product_page,
            receipts=receipts,
        )
        self.assertEqual(
            {"features": self.features, "description": self.description},
            summary["current_product"]["inspected_tab_observations"],
        )
        feedback = _public_state_feedback(
            request,
            task_family="webshop",
            observation=self.product_page,
            admissible_actions=self.product_actions,
            receipts=receipts,
        )
        prompt = _action_prompt(
            request,
            task_family="webshop",
            observation=self.product_page,
            admissible_actions=self.product_actions,
            receipts=receipts,
            turn=18,
            remaining_action_budget=2,
            structured_actions=True,
        )
        for label, projected in (
            ("summary", json.dumps(summary)),
            ("feedback", feedback),
            ("action_prompt", prompt),
        ):
            with self.subTest(projection=label):
                self.assertIn(self.goal, projected)
                self.assertIn(self.features, projected)
                self.assertIn(self.description, projected)
                for private_value in (
                    "PRIVATE_GOAL_SENTINEL",
                    "PRIVATE_REWARD_SENTINEL",
                    "PRIVATE_EVALUATOR_SENTINEL",
                ):
                    self.assertNotIn(private_value, projected)
        self.assertIn(self.product_page, prompt)
        for action in self.product_actions:
            self.assertIn(action, prompt)

    def test_prompt_observation_clipping_is_explicit_and_leaves_source_unchanged(self) -> None:
        original = self.features + " Public detail." * 30
        rendered, clipped = _prompt_observation(original, 80)

        self.assertTrue(clipped)
        self.assertTrue(rendered.startswith(original[:80]))
        self.assertIn("[OBSERVATION CLIPPED:", rendered)
        self.assertIn(f"retained first 80 of {len(original)} characters", rendered)
        self.assertIn("full observation remains in the receipt", rendered)
        self.assertEqual(self.features + " Public detail." * 30, original)

    def test_unclipped_observation_is_preserved_exactly(self) -> None:
        for limit in (0, len(self.description), len(self.description) + 1):
            with self.subTest(limit=limit):
                self.assertEqual(
                    (self.description, False),
                    _prompt_observation(self.description, limit),
                )


if __name__ == "__main__":
    unittest.main()
