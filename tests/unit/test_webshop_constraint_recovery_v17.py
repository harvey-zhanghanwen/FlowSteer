from __future__ import annotations

import unittest

from src.interactive.environment_execution import (
    _public_transition_summary,
    _webshop_action_precondition_failure,
    _webshop_model_visible_actions,
    _webshop_purchase_preconditions,
    _webshop_required_option_constraints,
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


class WebShopConstraintRecoveryV17Tests(unittest.TestCase):
    def test_joint_textual_size_and_color_binding_is_group_aware(self) -> None:
        goal = (
            "i want a medium gray long sleeve hoodie, and price lower than "
            "60 dollars"
        )
        page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] Small "
            "[SEP] Medium [SEP] Large [SEP] color [SEP] Black [SEP] Gray "
            "[SEP] Casual Long Sleeve Hoodie [SEP] Price: $19.28 "
            "[SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        native = (
            "click[Small]",
            "click[Medium]",
            "click[Large]",
            "click[Black]",
            "click[Gray]",
            "click[Buy Now]",
            "click[Back to Search]",
        )

        visible, groups, selected = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=page,
            receipts=(),
            native_actions=native,
        )
        constraints = _webshop_required_option_constraints(goal, groups)

        self.assertEqual({}, selected)
        self.assertEqual(["Medium"], constraints["size"]["acceptable_values"])
        self.assertEqual(["Gray"], constraints["color"]["acceptable_values"])
        self.assertIn("click[Medium]", visible)
        self.assertIn("click[Gray]", visible)
        self.assertNotIn("click[Small]", visible)
        self.assertNotIn("click[Large]", visible)
        self.assertNotIn("click[Black]", visible)
        self.assertNotIn("click[Buy Now]", visible)

        receipts = (
            _transition(1, "click[Medium]", page, page),
            _transition(2, "click[Gray]", page, page),
        )
        complete_visible, _, complete_selected = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=page,
            receipts=receipts,
            native_actions=native,
        )
        self.assertEqual(
            {"size": "Medium", "color": "Gray"}, complete_selected
        )
        self.assertIn("click[Buy Now]", complete_visible)

    def test_incidental_small_end_table_is_not_a_size_binding(self) -> None:
        goal = "find a small end table, and price lower than 70 dollars"
        groups = {"size": ["Small", "Large"]}

        self.assertNotIn(
            "size", _webshop_required_option_constraints(goal, groups)
        )

    def test_absent_same_dimension_requirement_masks_only_conflicts(self) -> None:
        goal = (
            "find sugar free barbecue marinade in the 18 ounce size, and "
            "price lower than 60 dollars"
        )
        page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] size "
            "[SEP] 6 Pack [SEP] 12 Ounce (Pack of 1) "
            "[SEP] Demo Marinade [SEP] Price: $12 [SEP] Buy Now"
        )
        native = (
            "click[6 Pack]",
            "click[12 Ounce (Pack of 1)]",
            "click[Buy Now]",
            "click[Back to Search]",
        )

        visible, groups, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=page,
            receipts=(),
            native_actions=native,
        )
        constraint = _webshop_required_option_constraints(goal, groups)["size"]

        self.assertEqual("no_visible_match", constraint["status"])
        self.assertEqual([], constraint["acceptable_values"])
        self.assertEqual(
            ["12 Ounce (Pack of 1)"], constraint["contradicted_values"]
        )
        self.assertIn("click[6 Pack]", visible)
        self.assertNotIn("click[12 Ounce (Pack of 1)]", visible)
        self.assertNotIn("click[Buy Now]", visible)
        self.assertEqual(
            "option_conflicts_with_instruction",
            _webshop_action_precondition_failure(
                action="click[12 Ounce (Pack of 1)]",
                task_instruction=goal,
                observation=page,
                receipts=(),
            ),
        )

    def test_query_reuse_requires_candidate_progress(self) -> None:
        goal = "find a blue bottle"
        start = "WebShop [SEP] Search"
        results = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 2) [SEP] B000000001 "
            "[SEP] Blue Bottle [SEP] $10 [SEP] B000000002 "
            "[SEP] Other Bottle [SEP] $11"
        )
        product = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Blue Bottle [SEP] Price: $10 [SEP] Buy Now"
        )
        searched = (_transition(1, "search[blue bottle]", start, results),)
        progressed = searched + (
            _transition(2, "click[B000000001]", results, product),
            _transition(3, "click[Back to Search]", product, start),
        )

        self.assertEqual(
            "repeated_search_query",
            _webshop_action_precondition_failure(
                action="search[ BLUE---bottle ]",
                task_instruction=goal,
                observation=start,
                receipts=searched,
            ),
        )
        self.assertIsNone(
            _webshop_action_precondition_failure(
                action="search[blue bottle]",
                task_instruction=goal,
                observation=start,
                receipts=progressed,
            )
        )

    def test_restored_results_mask_opened_candidate_when_alternatives_exist(
        self,
    ) -> None:
        goal = "find a blue bottle"
        start = "WebShop [SEP] Search"
        results = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] Page 1 (Total results: 2) [SEP] B000000001 "
            "[SEP] Blue Bottle [SEP] $10 [SEP] B000000002 "
            "[SEP] Other Bottle [SEP] $11"
        )
        product = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Blue Bottle [SEP] Price: $10 [SEP] Buy Now"
        )
        receipts = (
            _transition(1, "search[blue bottle]", start, results),
            _transition(2, "click[B000000001]", results, product),
            _transition(3, "click[Back to Search]", product, start),
            _transition(4, "search[blue bottle]", start, results),
        )

        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=results,
            receipts=receipts,
            native_actions=(
                "click[Back to Search]",
                "click[B000000001]",
                "click[B000000002]",
            ),
        )

        self.assertNotIn("click[B000000001]", visible)
        self.assertIn("click[B000000002]", visible)

    def test_public_attribute_evidence_is_retained_but_advisory(self) -> None:
        title_goal = (
            "find a rubber sole clog, and price lower than 100 dollars"
        )
        title_page = (
            "Instruction: [SEP] "
            + title_goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] "
            "Mens Clog with Arch Support and Rubber Sole [SEP] Price: $47.99 "
            "[SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        title_purchase = _webshop_purchase_preconditions(
            task_instruction=title_goal,
            observation=title_page,
            receipts=(),
        )
        self.assertEqual(
            "supported",
            title_purchase["attribute_evidence"]["rubber sole"]["status"],
        )
        self.assertTrue(title_purchase["admissible"])

        unverified_goal = (
            "find a hands free speaker, and price lower than 100 dollars"
        )
        unverified_page = (
            "Instruction: [SEP] "
            + unverified_goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Demo Speaker "
            "[SEP] Price: $20 [SEP] Description [SEP] Features [SEP] Buy Now"
        )
        unverified_purchase = _webshop_purchase_preconditions(
            task_instruction=unverified_goal,
            observation=unverified_page,
            receipts=(),
        )
        self.assertEqual(
            "unverified",
            unverified_purchase["attribute_evidence"]["hands free"]["status"],
        )
        self.assertTrue(unverified_purchase["admissible"])

    def test_features_evidence_survives_return_to_product_page(self) -> None:
        goal = "find a steel frame shelf, and price lower than 100 dollars"
        results = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] Page 1 (Total results: 1) "
            "[SEP] B000000001"
        )
        product = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Demo Shelf "
            "[SEP] Price: $20 [SEP] Description [SEP] Features [SEP] Buy Now"
        )
        features = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] reinforced steel frame"
        )
        receipts = (
            _transition(1, "click[B000000001]", results, product),
            _transition(2, "click[Features]", product, features),
            _transition(3, "click[< Prev]", features, product),
        )

        summary = _public_transition_summary(
            task_family="webshop",
            task_instruction=goal,
            observation=product,
            receipts=receipts,
        )
        current = summary["current_product"]
        self.assertIsInstance(current, dict)
        self.assertEqual(
            "supported",
            current["attribute_evidence"]["steel frame"]["status"],
        )
        self.assertEqual(["features"], current["tab_evidence_sources"])


if __name__ == "__main__":
    unittest.main()
