from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.interactive.step_transaction import StepPhase


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stop_hotpotqa_at_step.py"
SPEC = importlib.util.spec_from_file_location("stop_hotpotqa_at_step", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
should_request_stop = MODULE.should_request_stop


def test_request_during_prepared_target_step() -> None:
    assert should_request_stop(
        committed=249,
        current_step={"step": 250, "phase": StepPhase.PREPARED.value},
        target=250,
    ) is True


def test_wait_until_target_step_has_started() -> None:
    assert should_request_stop(committed=249, current_step=None, target=250) is False


@pytest.mark.parametrize("phase", [phase.value for phase in StepPhase])
def test_every_phase_below_target_keeps_training(phase: str) -> None:
    assert should_request_stop(
        committed=248,
        current_step={"step": 249, "phase": phase},
        target=250,
    ) is False


@pytest.mark.parametrize("committed", [250, 251])
def test_request_when_target_is_already_committed(committed: int) -> None:
    assert should_request_stop(
        committed=committed, current_step=None, target=250
    ) is True


@pytest.mark.parametrize("phase", [phase.value for phase in StepPhase])
@pytest.mark.parametrize("step", [250, 251])
def test_request_after_target_step_is_prepared(step: int, phase: str) -> None:
    assert should_request_stop(
        committed=step - 1,
        current_step={"step": step, "phase": phase},
        target=250,
    ) is True


def test_zero_committed_steps_is_valid() -> None:
    assert should_request_stop(committed=0, current_step=None, target=250) is False


@pytest.mark.parametrize("committed", [-1, True, False, 249.0, "249", None])
def test_reject_invalid_committed_count(committed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(committed=committed, current_step=None, target=250)


@pytest.mark.parametrize("target", [0, -1, True, False, 250.0, "250", None])
def test_reject_invalid_target(target: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(committed=249, current_step=None, target=target)


@pytest.mark.parametrize("current_step", [[], "prepared", 250, True])
def test_reject_non_object_current_step(current_step: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(committed=249, current_step=current_step, target=250)


@pytest.mark.parametrize(
    "current_step",
    [{}, {"step": 250}, {"phase": StepPhase.PREPARED.value}],
)
def test_reject_missing_current_step_fields(current_step: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(committed=249, current_step=current_step, target=250)


@pytest.mark.parametrize("step", [0, -1, True, False, 250.0, "250", None])
def test_reject_invalid_current_step_number(step: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(
            committed=249,
            current_step={"step": step, "phase": StepPhase.PREPARED.value},
            target=250,
        )


@pytest.mark.parametrize("phase", ["", "unknown", None, True, 250, [], {}])
def test_reject_invalid_current_step_phase(phase: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        should_request_stop(
            committed=249,
            current_step={"step": 250, "phase": phase},
            target=250,
        )
