from __future__ import annotations

import unittest

from src.interactive.qa_retrieval import (
    PublicPassageObservation,
    QARetrievalReceipt,
    build_keyword_query,
    receipt_from_mapping,
)


class QARetrievalReceiptTests(unittest.TestCase):
    def test_keyword_query_removes_question_stop_words_deterministically(self) -> None:
        self.assertEqual(
            "American born Sinclair won Nobel Prize Literature 1930",
            build_keyword_query(
                "Which American-born Sinclair won the Nobel Prize for Literature in 1930?"
            ),
        )

    def test_rendered_problem_contains_only_public_search_read_observations(self) -> None:
        receipt = QARetrievalReceipt(
            query="Who won?",
            search_limit=1,
            passages=(
                PublicPassageObservation(
                    rank=1,
                    passage_id="p1",
                    document_id="d1",
                    title="Public title",
                    text="Public passage text.",
                ),
            ),
        )

        rendered = receipt.render_problem("Who won?")
        value = receipt.to_dict()

        self.assertIn("SkillFlow search/read", rendered)
        self.assertIn("Public passage text.", rendered)
        self.assertNotIn("accepted_answers", rendered)
        self.assertNotIn("ground_truth", rendered)
        self.assertEqual(2, value["tool_calls"])
        self.assertEqual(["search", "read"], [item["name"] for item in value["operations"]])

    def test_cached_receipt_round_trip_preserves_problem_rendering(self) -> None:
        original = QARetrievalReceipt(
            query="Question",
            search_limit=2,
            passages=(
                PublicPassageObservation(1, "p1", "d1", "Title", "Text"),
            ),
        )
        restored = receipt_from_mapping(original.to_dict())
        self.assertEqual(original, restored)
        self.assertEqual(
            original.render_problem("Question"),
            restored.render_problem("Question"),
        )


if __name__ == "__main__":
    unittest.main()
