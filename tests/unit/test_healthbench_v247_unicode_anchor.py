"""Synthetic Unicode task anchors; no benchmark, retrieval or model calls."""

from __future__ import annotations

import unittest

from src.interactive.healthbench_evidence_adapter import (
    _normalized_query_tokens,
    _query_preserves_task_surface,
)


class UnicodeTaskAnchorTests(unittest.TestCase):
    def test_exact_unicode_word_survives_ascii_fragments_being_too_short(self):
        for word in ("naïve", "æøå", "bæø"):
            with self.subTest(word=word):
                self.assertTrue(_query_preserves_task_surface(
                    f"marker: {word}", f'"{word}" search',
                ))

    def test_unicode_case_and_canonical_combining_accents_match(self):
        for source, query in (
            ("NAÏVE", "naïve"),
            ("ÆØÅ", "æøå"),
            ("café", "cafe\u0301"),
            ("cafe\u0301", "café"),
            ("naïve", "nai\u0308ve"),
        ):
            with self.subTest(source=source, query=query):
                self.assertTrue(_query_preserves_task_surface(
                    f"marker {source}", query,
                ))

    def test_unicode_path_does_not_translate_fold_accents_or_match_substrings(self):
        for source, query in (
            ("naïve", "naive"),
            ("café", "coffee"),
            ("æøå", "aeoa"),
            ("æøå", "xæøåx"),
            ("æøå", "åøæ"),
        ):
            with self.subTest(source=source, query=query):
                self.assertFalse(_query_preserves_task_surface(
                    f"marker {source}", query,
                ))

    def test_unicode_word_does_not_admit_changed_codes(self):
        for separator in ("", "_", "-", "–", "—"):
            with self.subTest(separator=separator):
                self.assertFalse(_query_preserves_task_surface(
                    f"marker naïve{separator}17", f"naïve{separator}18",
                ))

    def test_anchor_from_another_task_does_not_carry_over(self):
        self.assertTrue(_query_preserves_task_surface("marker naïve", "naïve"))
        self.assertFalse(_query_preserves_task_surface("other æøå", "naïve"))

    def test_existing_ascii_fuzzy_and_no_ascii_fallback_are_unchanged(self):
        self.assertTrue(_query_preserves_task_surface("XYZ", "xyz"))
        self.assertFalse(_query_preserves_task_surface("XYZ", "XQZ"))
        self.assertTrue(_query_preserves_task_surface("luminarccept", "luminarcept"))
        self.assertFalse(_query_preserves_task_surface("12345678901", "12345678902"))
        self.assertTrue(_query_preserves_task_surface("æøå", "unrelated query"))

    def test_shared_budget_tokenizer_keeps_its_existing_ascii_semantics(self):
        self.assertEqual(("na", "ve", "caf"), _normalized_query_tokens("naïve café æøå"))


if __name__ == "__main__":
    unittest.main()
