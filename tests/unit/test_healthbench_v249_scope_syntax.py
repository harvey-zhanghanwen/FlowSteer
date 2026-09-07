"""Synthetic list/reference regressions, unrelated to clinical answers."""

import pytest

from tests.unit.test_healthbench_v246_scope_repair import scope_step


@pytest.mark.parametrize("contract", [
    "Search ABC: (1) retrieve sources (2) read relevant evidence (3) state uncertainty.",
    "Review node 1's unresolved ABC findings and retain source references.",
    "Review node 1’s unresolved ABC findings: (1) retrieve sources (2) state uncertainty.",
])
def test_operational_list_and_existing_node_alias_are_not_clinical_literals(contract):
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    # Only literal extraction changes; actual contract/dependency parsing
    # still receives the original text, not a rewritten role or workflow.
    assert env.graph.get_node("node_1").contract == contract


@pytest.mark.parametrize("contract", [
    "Search ABC: (1) use dose 47 mg (2) read sources (3) state uncertainty.",
    "Review node 1's findings for patients aged 47 years.",
    "Search ABC at measurement (47) units.",
    "Review node 47's findings about ABC.",
])
def test_operational_syntax_does_not_ground_clinical_values_or_unknown_nodes(contract):
    env, result = scope_step(contract)
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback
    assert not env.graph.nodes
