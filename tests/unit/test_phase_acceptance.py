"""Acceptance-report tests for LatentLoss Phases A--D."""

from __future__ import annotations

import pytest

from src.interactive.exploration.phase_acceptance import (
    AcceptanceCheck,
    AcceptanceGateError,
    PhaseAcceptance,
    PhaseAcceptanceReport,
    run_phase_acceptance,
)


@pytest.fixture(scope="module")
def report() -> PhaseAcceptanceReport:
    return run_phase_acceptance()


def test_all_synthetic_phase_acceptance_checks_pass(report: PhaseAcceptanceReport) -> None:
    assert report.synthetic_gates_passed
    assert [phase.status for phase in report.phases[:4]] == ["passed"] * 4
    assert all(
        check.passed is True
        for phase in report.phases[:4]
        for check in phase.checks
    )
    report.assert_ready_for_phase_e()


def test_phase_e_remains_pending_and_fail_closed(report: PhaseAcceptanceReport) -> None:
    phase_e = report.phases[-1]
    assert phase_e.phase == "E"
    assert phase_e.execution_boundary == "real_runner_only"
    assert phase_e.status == "pending"
    assert all(check.passed is None for check in phase_e.checks)
    assert not report.phase_e_completed
    assert not report.long_training_authorized
    with pytest.raises(AcceptanceGateError, match="Phase E is pending"):
        report.assert_ready_for_long_training()


def test_report_is_deterministic_and_serializable(report: PhaseAcceptanceReport) -> None:
    repeated = run_phase_acceptance()
    assert repeated.to_dict() == report.to_dict()
    payload = report.to_dict()
    assert payload["version"] == "latent-loss-phase-acceptance-v1"
    assert payload["long_training_authorized"] is False
    assert payload["phases"][0]["checks"][0]["threshold"]["coverage_gte"] == 0.85


def test_any_synthetic_failure_closes_the_gate() -> None:
    failed = PhaseAcceptanceReport(
        seed=0,
        phases=(
            PhaseAcceptance(
                phase="A",
                execution_boundary="synthetic",
                checks=(
                    AcceptanceCheck(
                        check_id="A1",
                        description="intentional failure",
                        passed=False,
                        value=0.0,
                        threshold={"minimum": 1.0},
                        details={},
                    ),
                ),
            ),
            PhaseAcceptance(
                phase="E",
                execution_boundary="real_runner_only",
                checks=(
                    AcceptanceCheck(
                        check_id="E1",
                        description="pending",
                        passed=None,
                        value="not_run",
                        threshold={"real_runner_receipt_required": True},
                        details={},
                    ),
                ),
            ),
        ),
    )
    assert not failed.synthetic_gates_passed
    assert not failed.long_training_authorized
    with pytest.raises(AcceptanceGateError, match="A1"):
        failed.assert_ready_for_phase_e()
