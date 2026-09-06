"""Synthetic lexical admission cases; no benchmark terms or medical rules."""

from __future__ import annotations

import unittest

from src.interactive.healthbench_evidence_adapter import _query_preserves_task_surface


class QuerySurfaceTypoTests(unittest.TestCase):
    def test_exact_anchor_keeps_existing_case_and_punctuation_normalization(self):
        self.assertTrue(_query_preserves_task_surface(
            'Clinical patient: "Luminarcept".', 'luminarcept treatment',
        ))

    def test_one_inserted_character_in_long_word_is_admitted(self):
        self.assertTrue(_query_preserves_task_surface(
            'Patient with luminarccept', 'luminarcept management',
        ))

    def test_one_deleted_character_in_long_word_is_admitted(self):
        self.assertTrue(_query_preserves_task_surface(
            'Patient with luminarcept', 'luminarccept management',
        ))

    def test_single_substitution_requires_narrow_similarity_threshold(self):
        self.assertTrue(_query_preserves_task_surface(
            'luminarcept', 'luminarceqt',
        ))
        self.assertFalse(_query_preserves_task_surface('felnor', 'felmor'))

    def test_short_acronym_substitution_remains_rejected(self):
        self.assertFalse(_query_preserves_task_surface(
            'Patient with XYZ', 'XQZ management',
        ))
        self.assertTrue(_query_preserves_task_surface('XYZ', 'xyz'))

    def test_even_similar_long_uppercase_identifier_is_not_fuzzy_matched(self):
        self.assertFalse(_query_preserves_task_surface(
            'LUMINARCEPT', 'luminarceqt',
        ))
        self.assertFalse(_query_preserves_task_surface(
            'luminarcept', 'LUMINARCEQT',
        ))

    def test_numeric_and_mixed_code_substitutions_remain_rejected(self):
        self.assertFalse(_query_preserves_task_surface('12345678901', '12345678902'))
        self.assertFalse(_query_preserves_task_surface('luminarcept7', 'luminarceqt7'))
        self.assertFalse(_query_preserves_task_surface('luminarcept-7', 'luminarceqt-8'))
        self.assertTrue(_query_preserves_task_surface('luminarcept7', 'luminarcept7'))

    def test_two_edits_not_admitted_despite_high_similarity(self):
        self.assertFalse(_query_preserves_task_surface(
            'luminarceptabcdef', 'luminarceqtabcdeg',
        ))

    def test_unrelated_query_still_rejected(self):
        self.assertFalse(_query_preserves_task_surface(
            'luminarcept', 'qavodrensis treatment',
        ))

    def test_generic_scaffolding_is_not_an_anchor(self):
        self.assertFalse(_query_preserves_task_surface(
            '[{"role":"user","content":"luminarcept"}]',
            'clinical patient guidelines management',
        ))

    def test_non_ascii_no_anchor_behavior_is_unchanged(self):
        self.assertTrue(_query_preserves_task_surface(
            '[{"role":"user","content":"示例中文描述"}]', 'example query',
        ))


if __name__ == '__main__':
    unittest.main()
