from __future__ import annotations

import unittest

from src.interactive.environment_execution import (
    _public_transition_summary,
    _webshop_action_precondition_failure,
    _webshop_model_visible_actions,
    _webshop_parse_product_options,
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

    def test_native_configuration_label_starts_its_own_option_group(self) -> None:
        goal = (
            "find an intel i5 desktop that needs to be omen 25l and "
            "configured with nvidia rtx 3090"
        )
        page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] style "
            "[SEP] intel i5 [SEP] intel i7 [SEP] size [SEP] omen 25l "
            "[SEP] omen 30l [SEP] configuration [SEP] nvidia rtx 3070 "
            "[SEP] nvidia rtx 3090 [SEP] Demo PC [SEP] Price: $1000 "
            "[SEP] Buy Now"
        )

        groups, _ = _webshop_parse_product_options(page)
        constraints = _webshop_required_option_constraints(goal, groups)

        self.assertEqual(["omen 25l", "omen 30l"], groups["size"])
        self.assertEqual(
            ["nvidia rtx 3070", "nvidia rtx 3090"],
            groups["configuration"],
        )
        self.assertEqual(["omen 25l"], constraints["size"]["acceptable_values"])
        self.assertEqual(
            ["nvidia rtx 3090"],
            constraints["configuration"]["acceptable_values"],
        )

    def test_incidental_small_end_table_is_not_a_size_binding(self) -> None:
        goal = "find a small end table, and price lower than 70 dollars"
        groups = {"size": ["Small", "Large"]}

        self.assertNotIn(
            "size", _webshop_required_option_constraints(goal, groups)
        )

    def test_explicit_task_attributes_survive_absent_candidate_values(self) -> None:
        goal = (
            "find a snack with color: teal, size: medium, fit type: relaxed, "
            "flavor: vanilla, and price lower than 70 dollars"
        )
        groups = {
            "color": ["Red", "Blue"],
            "size": ["Small", "Large"],
            "fit": ["Slim", "Regular"],
            "flavor name": ["Chocolate", "Mint"],
        }

        constraints = _webshop_required_option_constraints(goal, groups)

        for group, surface in (
            ("color", "teal"),
            ("size", "medium"),
            ("fit", "relaxed"),
            ("flavor name", "vanilla"),
        ):
            self.assertEqual("no_visible_match", constraints[group]["status"])
            self.assertEqual([], constraints[group]["acceptable_values"])
            self.assertEqual(
                [surface], constraints[group]["requirement_surfaces"]
            )
            self.assertEqual(groups[group], constraints[group]["contradicted_values"])

    def test_absent_upstream_color_and_joint_size_remain_requirements(self) -> None:
        goal = "find a medium gray long sleeve hoodie under 60 dollars"
        groups = {
            "size": ["Small", "Large"],
            "color": ["Black", "Blue"],
        }

        constraints = _webshop_required_option_constraints(goal, groups)

        self.assertEqual("no_visible_match", constraints["size"]["status"])
        self.assertEqual(["medium"], constraints["size"]["requirement_surfaces"])
        self.assertEqual("no_visible_match", constraints["color"]["status"])
        self.assertEqual(["gray"], constraints["color"]["requirement_surfaces"])

    def test_public_color_modifiers_and_bundle_labels_remain_selectable(
        self,
    ) -> None:
        cases = (
            (
                "the cabinet color should be light brown, and price lower than 190",
                ["espresso", "light brown", "white&brown"],
                ["light brown"],
            ),
            (
                "find a folding mattress with wine red color",
                ["beige", "navy blue", "pink", "wine red"],
                ["wine red"],
            ),
            (
                "find a table and chair set that is white and easy to assemble",
                ["dining table only", "table+white chairs", "table+black chairs"],
                ["table+white chairs"],
            ),
            (
                "find socks in black or semi mint rush green color",
                [
                    "black | semi turbo pink | semi mint rush green",
                    "black | white | cool light heather",
                ],
                ["black | semi turbo pink | semi mint rush green"],
            ),
        )

        for goal, values, expected in cases:
            with self.subTest(goal=goal):
                constraint = _webshop_required_option_constraints(
                    goal,
                    {"color": values},
                )["color"]
                self.assertEqual("visible_match", constraint["status"])
                self.assertEqual(expected, constraint["acceptable_values"])

    def test_naturally_labelled_absent_color_remains_no_visible_match(
        self,
    ) -> None:
        constraint = _webshop_required_option_constraints(
            "find size 7.5 camo colored walking shoes",
            {"color": ["a1-blue", "a1-pink"]},
        )["color"]

        self.assertEqual("no_visible_match", constraint["status"])
        self.assertEqual(["camo"], constraint["requirement_surfaces"])
        self.assertEqual([], constraint["acceptable_values"])

    def test_compound_color_is_atomic_not_token_overlap(self) -> None:
        goal = (
            "i would like a wirefree pink amethyst 36c bra that is machine "
            "washable, and price lower than 70 dollars"
        )
        wrong_groups = {
            "size": ["34c", "36c", "38c"],
            "color": ["nude", "hush pink swirl", "black"],
        }
        wrong = _webshop_required_option_constraints(goal, wrong_groups)["color"]

        self.assertEqual(["pink amethyst"], wrong["requirement_surfaces"])
        self.assertEqual("no_visible_match", wrong["status"])
        self.assertEqual([], wrong["acceptable_values"])
        self.assertIn("hush pink swirl", wrong["contradicted_values"])

        correct_groups = {
            **wrong_groups,
            "color": ["hush pink swirl", "pink amethyst"],
        }
        correct = _webshop_required_option_constraints(goal, correct_groups)[
            "color"
        ]
        self.assertEqual(["pink amethyst"], correct["acceptable_values"])

    def test_compound_measurement_option_requires_dimension_conjunction(
        self,
    ) -> None:
        goal = (
            "find a skin treatment in 1.6 ounce pack of 2, and price lower "
            "than 40 dollars"
        )
        groups = {
            "size": [
                "1.6 Ounce (Pack of 2)",
                "3.4 Ounce (Pack of 2)",
                "1.6 Ounce (Pack of 3)",
            ]
        }

        constraint = _webshop_required_option_constraints(goal, groups)["size"]

        self.assertEqual(
            ["1.6 Ounce (Pack of 2)"], constraint["acceptable_values"]
        )
        self.assertEqual(
            ["3.4 Ounce (Pack of 2)", "1.6 Ounce (Pack of 3)"],
            constraint["contradicted_values"],
        )

    def test_compound_dimension_pair_preserves_both_values(self) -> None:
        constraint = _webshop_required_option_constraints(
            "find a folding mattress that is 75*190cm",
            {"size": ["70*190cm", "75*190cm", "80*190cm"]},
        )["size"]

        self.assertEqual(["75*190cm"], constraint["acceptable_values"])
        self.assertEqual(
            ["70*190cm", "80*190cm"],
            constraint["contradicted_values"],
        )

    def test_labeled_width_height_bind_visible_pair_positions(self) -> None:
        goal = (
            "find white shades that are 2 inches in width and 64 inches in "
            "height, and price lower than 130 dollars"
        )
        page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 24x64 in "
            "[SEP] 30x72 in [SEP] color [SEP] white [SEP] Demo Shade "
            "[SEP] Price: $40 [SEP] Buy Now"
        )
        native = (
            "click[24x64 in]",
            "click[30x72 in]",
            "click[white]",
            "click[Buy Now]",
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
            ["24x64 in", "30x72 in"], constraint["contradicted_values"]
        )
        self.assertNotIn("click[24x64 in]", visible)
        self.assertNotIn("click[30x72 in]", visible)
        self.assertEqual(
            "option_conflicts_with_instruction",
            _webshop_action_precondition_failure(
                action="click[24x64 in]",
                task_instruction=goal,
                observation=page,
                receipts=(),
            ),
        )

        matching = _webshop_required_option_constraints(
            goal,
            {"size": ["24x64 in", "2x64 in", "30x72 in"]},
        )["size"]
        self.assertEqual(["2x64 in"], matching["acceptable_values"])

    def test_native_compound_values_are_exact_public_options(self) -> None:
        goal = (
            "i need heavy duty glass and its size is case+4 protectors with "
            "redblack color, and price lower than 50 dollars"
        )
        groups = {
            "color": ["black", "redblack", "wine red pink"],
            "size": ["case+2 protectors", "case+4 protectors"],
        }

        constraints = _webshop_required_option_constraints(goal, groups)

        self.assertEqual(
            ["redblack"], constraints["color"]["acceptable_values"]
        )
        self.assertEqual(
            ["case+4 protectors"], constraints["size"]["acceptable_values"]
        )

    def test_native_compound_requirements_survive_wrong_candidate_inventory(
        self,
    ) -> None:
        goal = (
            "i need heavy duty glass and its size is case+4 protectors with "
            "redblack color, and price lower than 50 dollars"
        )
        constraints = _webshop_required_option_constraints(
            goal,
            {
                "size": ["iphone 13", "iphone 13 pro max"],
                "color": ["black", "camo", "red"],
            },
        )

        self.assertEqual("no_visible_match", constraints["size"]["status"])
        self.assertEqual(
            ["case+4 protectors"],
            constraints["size"]["requirement_surfaces"],
        )
        self.assertEqual("no_visible_match", constraints["color"]["status"])
        self.assertEqual(
            ["redblack"], constraints["color"]["requirement_surfaces"]
        )

    def test_colored_one_control_words_do_not_enter_color_value(self) -> None:
        constraint = _webshop_required_option_constraints(
            "find a compact item; also, choose black colored one",
            {"color": ["black", "brown"]},
        )["color"]

        self.assertEqual(["black"], constraint["acceptable_values"])
        self.assertEqual(["black"], constraint["requirement_surfaces"])

    def test_color_phrase_does_not_absorb_preceding_product_relation(
        self,
    ) -> None:
        constraint = _webshop_required_option_constraints(
            "i'm shopping for memory foam slippers with a rubber sole in a "
            "moonlight blue colour, and price lower than 50 dollars",
            {"color": ["black", "moonlight blue", "navy"]},
        )["color"]

        self.assertEqual(["moonlight blue"], constraint["requirement_surfaces"])
        self.assertEqual(["moonlight blue"], constraint["acceptable_values"])

    def test_label_before_color_value_does_not_absorb_previous_clause(
        self,
    ) -> None:
        constraint = _webshop_required_option_constraints(
            "i am looking for a professional make up brush with easy use. "
            "also choose color b, and price lower than 370 dollars",
            {"color": ["a", "b", "c"]},
        )["color"]

        self.assertEqual(["b"], constraint["requirement_surfaces"])
        self.assertEqual(["b"], constraint["acceptable_values"])

    def test_public_color_phrases_bind_composite_and_prefixed_options(
        self,
    ) -> None:
        cases = (
            (
                "also choose black or semi mint rush green color",
                ["black | semi turbo pink | semi mint rush green"],
            ),
            ("which has wine red color", ["wine red"]),
            ("i am looking for black color", ["black matte"]),
            ('buy the color option "a."', ["a"]),
            ("i need a black color", ["black"]),
            ("the color khaki", ["nets-khaki"]),
        )
        for instruction, expected in cases:
            with self.subTest(instruction=instruction):
                constraint = _webshop_required_option_constraints(
                    instruction,
                    {"color": expected + ["white"]},
                )["color"]
                self.assertEqual(expected, constraint["acceptable_values"])

    def test_compatibility_measurement_is_not_a_style_name_constraint(
        self,
    ) -> None:
        constraints = _webshop_required_option_constraints(
            "i am looking for an inexpensive tv stand for our 60 inch tv "
            "with huge storage space and that should be in ashland pine "
            "color, and price lower than 320 dollars",
            {
                "style name": ["50-inch", "52-inch", "58-inch"],
                "color": ["ashland pine", "dark walnut"],
            },
        )

        self.assertNotIn("style name", constraints)
        self.assertEqual(
            ["ashland pine"], constraints["color"]["acceptable_values"]
        )

    def test_webshop_style_name_and_pluralized_flavor_are_bound(self) -> None:
        style = _webshop_required_option_constraints(
            "find a porthole in window style",
            {"style name": ["mirror", "window"]},
        )["style name"]
        flavor = _webshop_required_option_constraints(
            "find a container of natural sea salt",
            {"flavor": ["spicy chili peppers", "natural salts"]},
        )["flavor"]

        self.assertEqual(["window"], style["acceptable_values"])
        self.assertEqual(["natural salts"], flavor["acceptable_values"])

    def test_number_word_measurement_preserves_compound_tens(self) -> None:
        constraint = _webshop_required_option_constraints(
            "find a cabinet that is fifty-two inch width",
            {"width": ["2 inch", "52 inch", "62 inch"]},
        )["width"]

        self.assertEqual(["52 inch"], constraint["acceptable_values"])
        self.assertEqual(
            ["2 inch", "62 inch"], constraint["contradicted_values"]
        )

    def test_gender_scoped_numeric_size_binds_visible_range_segment(self) -> None:
        constraint = _webshop_required_option_constraints(
            "find shoes for my wife and choose the 5.5 size",
            {
                "size": [
                    "4.5-5.5 women | 3-4 men",
                    "6-7.5 women | 4.5-5.5 men",
                    "5.5 wide",
                ]
            },
        )["size"]

        self.assertEqual(
            ["4.5-5.5 women | 3-4 men"],
            constraint["acceptable_values"],
        )

    def test_gender_scoped_task_keeps_bare_exact_numeric_size(self) -> None:
        constraint = _webshop_required_option_constraints(
            "find sandals for my wife and choose the 5.5 size",
            {"size": ["4.5", "5.5", "6", "5.5 wide"]},
        )["size"]

        self.assertEqual(["5.5"], constraint["acceptable_values"])

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

    def test_product_boundary_keeps_prev_instead_of_resetting_search(self) -> None:
        goal = "find a blue bottle"
        product = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Blue Bottle [SEP] Price: $10 [SEP] Buy Now"
        )

        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product,
            receipts=(),
            native_actions=(
                "click[Back to Search]",
                "click[< Prev]",
                "click[Buy Now]",
            ),
        )

        self.assertNotIn("click[Back to Search]", visible)
        self.assertIn("click[< Prev]", visible)
        self.assertIn("click[Buy Now]", visible)

    def test_paginated_results_are_not_reported_as_product_scope(self) -> None:
        goal = "find a blue bottle"
        results = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] Page 2 (Total results: 50) [SEP] < Prev [SEP] Next > "
            "[SEP] B000000002 [SEP] Blue Bottle [SEP] $10"
        )
        product = (
            "Instruction: [SEP] find a blue bottle [SEP] Back to Search "
            "[SEP] < Prev [SEP] Blue Bottle [SEP] Price: $10 [SEP] Buy Now"
        )
        receipts = (
            _transition(1, "click[B000000002]", results, product),
            _transition(2, "click[< Prev]", product, results),
        )

        summary = _public_transition_summary(
            task_family="webshop",
            task_instruction=goal,
            observation=results,
            receipts=receipts,
        )

        self.assertIsNone(summary["current_product"])
        self.assertFalse(
            summary["purchase_preconditions"]["in_product_scope"]
        )

    def test_public_transition_retains_unranked_visible_result_rows(self) -> None:
        goal = "find a blue bottle under 30 dollars"
        results = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] Page 1 (Total results: 2) "
            "[SEP] B000000001 [SEP] Blue Bottle [SEP] $10.00 "
            "[SEP] B000000002 [SEP] Other Bottle [SEP] $11.50"
        )

        summary = _public_transition_summary(
            task_family="webshop",
            task_instruction=goal,
            observation=results,
            receipts=(),
        )

        self.assertEqual(
            [
                {"asin": "b000000001", "title": "Blue Bottle", "price": 10.0},
                {"asin": "b000000002", "title": "Other Bottle", "price": 11.5},
            ],
            summary["visible_result_rows"],
        )

    def test_purchase_admissibility_requires_current_buy_action(self) -> None:
        goal = "find a blue bottle under 30 dollars"
        results = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] Page 1 (Total results: 1) "
            "[SEP] B000000001 [SEP] Blue Bottle [SEP] $10.00"
        )
        product = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Blue Bottle "
            "[SEP] Price: $10 [SEP] Description [SEP] Buy Now"
        )
        description = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] blue bottle"
        )
        receipts = (
            _transition(1, "click[B000000001]", results, product),
            _transition(2, "click[Description]", product, description),
        )

        self.assertFalse(
            _webshop_purchase_preconditions(
                task_instruction=goal,
                observation=results,
                receipts=(),
            )["admissible"]
        )
        self.assertFalse(
            _webshop_purchase_preconditions(
                task_instruction=goal,
                observation=description,
                receipts=receipts,
            )["admissible"]
        )
        self.assertTrue(
            _webshop_purchase_preconditions(
                task_instruction=goal,
                observation=product,
                receipts=receipts,
            )["admissible"]
        )

    def test_public_attribute_evidence_requires_one_primary_inspection(self) -> None:
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
        self.assertTrue(unverified_purchase["evidence_inspection_required"])
        self.assertFalse(unverified_purchase["admissible"])
        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=unverified_goal,
            observation=unverified_page,
            receipts=(),
            native_actions=(
                "click[Description]",
                "click[Features]",
                "click[Buy Now]",
            ),
        )
        self.assertNotIn("click[Buy Now]", visible)

    def test_labeled_measurements_are_bound_to_width_and_height(self) -> None:
        goal = (
            "find white shades that are 2 inches in width and 64 inches in "
            "height, and price lower than 130 dollars"
        )
        wrong_page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] White Blackout "
            "Shade, 58-Inch Width by 72-Inch Height [SEP] Price: $95 "
            "[SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        wrong = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=wrong_page,
            receipts=(),
        )
        self.assertEqual(
            "contradicted", wrong["measurement_evidence"]["width"]["status"]
        )
        self.assertEqual(
            "contradicted", wrong["measurement_evidence"]["height"]["status"]
        )
        self.assertEqual(
            ["58"], wrong["measurement_evidence"]["width"]["observed_values"]
        )
        self.assertEqual(
            ["72"], wrong["measurement_evidence"]["height"]["observed_values"]
        )
        self.assertFalse(wrong["admissible"])

        correct_page = wrong_page.replace(
            "58-Inch Width by 72-Inch Height",
            "2-Inch Width by 64-Inch Height",
        )
        correct = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=correct_page,
            receipts=(),
        )
        self.assertEqual(
            "supported", correct["measurement_evidence"]["width"]["status"]
        )
        self.assertEqual(
            "supported", correct["measurement_evidence"]["height"]["status"]
        )
        self.assertTrue(correct["admissible"])

    def test_quote_dimension_pair_supports_labeled_width(self) -> None:
        goal = (
            "find machine washable drapes with a fifty-two inch width, and "
            "price lower than 60 dollars"
        )
        correct_page = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Purple Drapes "
            "52\" x 84\" [SEP] Price: $14 [SEP] Description [SEP] Features "
            "[SEP] Reviews [SEP] Buy Now"
        )
        wrong_page = correct_page.replace('52\" x 84\"', '42\" x 84\"')

        correct = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=correct_page,
            receipts=(),
        )
        wrong = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=wrong_page,
            receipts=(),
        )

        self.assertEqual(
            "supported", correct["measurement_evidence"]["width"]["status"]
        )
        self.assertEqual(
            "contradicted", wrong["measurement_evidence"]["width"]["status"]
        )

    def test_inspected_reviews_are_not_exposed_again(self) -> None:
        goal = "find a blue bottle under 30 dollars"
        product = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Blue Bottle "
            "[SEP] Price: $10 [SEP] Description [SEP] Features [SEP] Reviews "
            "[SEP] Buy Now"
        )
        reviews = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Useful review"
        )
        receipts = (
            _transition(1, "click[Reviews]", product, reviews),
            _transition(2, "click[< Prev]", reviews, product),
        )

        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product,
            receipts=receipts,
            native_actions=(
                "click[Description]",
                "click[Features]",
                "click[Reviews]",
                "click[Buy Now]",
            ),
        )

        self.assertNotIn("click[Reviews]", visible)
        self.assertIn("click[Buy Now]", visible)

    def test_tight_budget_exposes_only_current_candidate_completion_path(
        self,
    ) -> None:
        goal = "find a shirt with size: medium, and price lower than 50 dollars"
        product = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] Small "
            "[SEP] Medium [SEP] Demo Shirt [SEP] Price: $20 [SEP] "
            "Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        )
        native = (
            "click[Small]",
            "click[Medium]",
            "click[Description]",
            "click[Features]",
            "click[Reviews]",
            "click[Buy Now]",
            "click[< Prev]",
        )

        visible, _, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product,
            receipts=(),
            native_actions=native,
            remaining_action_budget=2,
        )
        self.assertEqual(("click[Medium]",), visible)

        selected = (_transition(1, "click[Medium]", product, product),)
        ready, _, _ = _webshop_model_visible_actions(
            task_instruction=goal,
            observation=product,
            receipts=selected,
            native_actions=native,
            remaining_action_budget=1,
        )
        self.assertEqual(("click[Buy Now]",), ready)

    def test_inspected_primary_evidence_removes_only_inspection_gate(self) -> None:
        goal = "find a hands free speaker, and price lower than 100 dollars"
        results = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] Page 1 (Total results: 1) "
            "[SEP] B000000001"
        )
        product = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] Demo Speaker "
            "[SEP] Price: $20 [SEP] Description [SEP] Features [SEP] Buy Now"
        )
        description = (
            "Instruction: [SEP] "
            + goal
            + " [SEP] Back to Search [SEP] < Prev [SEP] compact speaker"
        )
        receipts = (
            _transition(1, "click[B000000001]", results, product),
            _transition(2, "click[Description]", product, description),
            _transition(3, "click[< Prev]", description, product),
        )
        purchase = _webshop_purchase_preconditions(
            task_instruction=goal,
            observation=product,
            receipts=receipts,
        )
        self.assertEqual(
            "unverified",
            purchase["attribute_evidence"]["hands free"]["status"],
        )
        self.assertFalse(purchase["evidence_inspection_required"])
        self.assertTrue(purchase["admissible"])

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
