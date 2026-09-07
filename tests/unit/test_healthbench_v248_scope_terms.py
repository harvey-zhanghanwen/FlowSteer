"""Operational query limits are not supplied clinical measurements."""

import pytest

from tests.unit.test_healthbench_v246_scope_repair import scope_step


@pytest.mark.parametrize("contract", [
    "Search ABC using no more than 12 clinical terms.",
    "Retrieve ABC evidence with at most 12 terms and report limitations.",
    "Search ABC with 2 queries of 12 clinical terms each.",
])
def test_query_term_counts_remain_admitted(contract):
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract


@pytest.mark.parametrize("contract", [
    "Search ABC in children aged 12 years using 12 clinical terms.",
    "Retrieve ABC evidence at a dose of 12 mg using 12 terms.",
    "Search ABC with a score denominator of 12 and 12 clinical terms.",
    "Search ABC for patients with 12 clinical findings.",
    "Search ABC using 12 clinical terms for children aged 12 years.",
    "Search ABC with 12 terms about a dose of 12 mg.",
])
def test_query_budget_does_not_ground_clinical_numbers(contract):
    env, result = scope_step(contract)
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback
    assert not env.graph.nodes
