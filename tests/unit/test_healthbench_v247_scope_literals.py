"""Public Canvas scope regressions with synthetic, non-benchmark inputs."""

import pytest

from tests.unit.test_healthbench_v246_scope_repair import scope_step


@pytest.mark.parametrize("status", ["unresolved alternatives", "insufficient evidence"])
def test_evidence_status_is_an_operational_deliverable(status):
    contract = f"Retrieve sources about ABC and output '{status}' if evidence is unavailable."
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract


@pytest.mark.parametrize("contract", [
    "Search sources about ABC using a score range (0-47).",
    "Retrieve the denominator 47 for the supplied ABC score.",
    "Execute a search for ABC in children aged 7-11.",
])
def test_search_verb_cannot_hide_unsupported_numeric_scope(contract):
    env, result = scope_step(contract)
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback
    assert not env.graph.nodes


def test_numerical_values_supplied_in_the_original_task_remain_legal():
    contract = "Search sources on the supplied ABC score range 0-47."
    _, result = scope_step(contract, problem="Interpret an ABC score with range 0-47.")
    assert result.accepted, result.feedback


def test_operational_counts_are_not_clinical_scope():
    _, result = scope_step("Search ABC with 2 queries, and report 3 alternatives.")
    assert result.accepted, result.feedback


def test_status_label_does_not_admit_a_diagnosis_literal():
    _, result = scope_step(
        "Output 'insufficient evidence', then diagnose 'Invented Syndrome'."
    )
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback
