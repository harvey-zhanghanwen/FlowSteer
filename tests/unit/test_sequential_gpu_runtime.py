from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from src.interactive.rollout_collector import RolloutGate
from src.interactive.sequential_gpu_runtime import (
    SequentialGpuRuntime,
    SequentialGpuRuntimeError,
    SequentialGpuState,
)


class FakeSGLangManager:
    def __init__(self, events: list[str], *, gpu_id: int = 5) -> None:
        self.events = events
        self.gpu_id = gpu_id
        self._alive = False
        self._pid = None
        self._next_pid = 4100

    def start(self) -> None:
        if self._alive:
            raise RuntimeError("already running")
        self._next_pid += 1
        self._pid = self._next_pid
        self._alive = True
        self.events.append(f"start:{self._pid}")

    def stop(self, timeout_seconds: float = 30.0) -> None:
        self.events.append(f"stop:{self._pid}")
        self._alive = False
        self._pid = None

    def is_alive(self) -> bool:
        return self._alive

    def pid(self):
        return self._pid


@dataclass(frozen=True)
class CallbackReceipt:
    success: bool


def _runtime(events: list[str], *, drain_timeout_seconds: float = 0.05):
    manager = FakeSGLangManager(events)
    gate = RolloutGate(poll_interval_seconds=0.001)
    runtime = SequentialGpuRuntime(
        manager=manager,
        rollout_gate=gate,
        drain_timeout_seconds=drain_timeout_seconds,
        stop_timeout_seconds=0.05,
    )
    return runtime, manager, gate


def test_successful_cycle_has_strict_order_and_resumes_rollouts() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events)
    initial_pid = runtime.start_rollout_service()

    def train():
        assert not manager.is_alive()
        assert gate.paused and gate.in_flight == 0
        events.append("train")
        return CallbackReceipt(success=True)

    def publish():
        assert manager.is_alive()
        assert gate.paused and gate.in_flight == 0
        events.append("publish")
        return {"success": True}

    def canary():
        assert manager.is_alive()
        assert not gate.paused
        events.append("canary")
        return True

    receipt = runtime.run_update_cycle(
        training_callback=train,
        publication_callback=publish,
        canary_callback=canary,
    )

    assert events == [
        f"start:{initial_pid}",
        f"stop:{initial_pid}",
        "train",
        f"start:{receipt.active_pid}",
        "publish",
        "canary",
    ]
    assert receipt.success
    assert receipt.gpu_id == 5
    assert receipt.initial_pid == initial_pid
    assert receipt.active_pid != initial_pid
    assert receipt.gate_drained
    assert receipt.training_succeeded
    assert receipt.publication_succeeded
    assert receipt.canary_succeeded
    assert receipt.service_running_after
    assert receipt.implementation_source == "project_single_gpu_time_multiplexing"
    assert [stage.stage for stage in receipt.stages] == [
        "rollout_service_running",
        "rollout_gate_paused",
        "rollout_gate_drained",
        "rollout_service_stopped",
        "training_callback",
        "rollout_service_restarted",
        "policy_publication",
        "policy_canary",
        "rollout_gate_resumed",
    ]
    assert all(stage.success for stage in receipt.stages)
    assert not gate.paused
    assert runtime.state is SequentialGpuState.ROLLOUT_RUNNING


def test_canary_can_acquire_normal_rollout_gate_without_deadlock() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events)
    runtime.start_rollout_service()

    async def acquire_and_release() -> None:
        await gate.acquire()
        try:
            assert gate.in_flight == 1
            events.append("canary:admitted")
        finally:
            gate.release()

    def canary() -> bool:
        assert manager.is_alive()
        assert not gate.paused
        asyncio.run(asyncio.wait_for(acquire_and_release(), timeout=0.05))
        return True

    receipt = runtime.run_update_cycle(
        training_callback=lambda: events.append("train"),
        publication_callback=lambda: events.append("publish"),
        canary_callback=canary,
    )

    assert receipt.success
    assert receipt.canary_succeeded
    assert "canary:admitted" in events
    assert gate.in_flight == 0
    assert not gate.paused


def test_runtime_refuses_to_adopt_or_stop_preexisting_process() -> None:
    events: list[str] = []
    runtime, manager, _ = _runtime(events)
    manager.start()

    with pytest.raises(RuntimeError, match="refusing to adopt"):
        runtime.start_rollout_service()
    with pytest.raises(RuntimeError, match="not owned"):
        runtime.stop_rollout_service()

    assert manager.is_alive()
    assert not any(event.startswith("stop:") for event in events)


def test_training_failure_stays_paused_and_does_not_restart() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events)
    initial_pid = runtime.start_rollout_service()

    with pytest.raises(SequentialGpuRuntimeError) as captured:
        runtime.run_update_cycle(
            training_callback=lambda: CallbackReceipt(success=False),
            publication_callback=lambda: events.append("publish"),
            canary_callback=lambda: events.append("canary") or True,
        )

    receipt = captured.value.receipt
    assert not receipt.success
    assert receipt.status == "failed_closed_at_training_callback"
    assert receipt.gate_drained
    assert not receipt.training_succeeded
    assert not receipt.publication_succeeded
    assert not receipt.canary_succeeded
    assert not receipt.service_running_after
    assert gate.paused
    assert not manager.is_alive()
    assert events == [f"start:{initial_pid}", f"stop:{initial_pid}"]
    assert runtime.state is SequentialGpuState.FAILED_CLOSED


def test_failed_canary_stops_restarted_owned_service_and_keeps_gate_closed() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events)
    initial_pid = runtime.start_rollout_service()

    def failed_canary() -> bool:
        assert not gate.paused
        events.append("canary")
        return False

    with pytest.raises(SequentialGpuRuntimeError) as captured:
        runtime.run_update_cycle(
            training_callback=lambda: events.append("train"),
            publication_callback=lambda: events.append("publish"),
            canary_callback=failed_canary,
        )

    receipt = captured.value.receipt
    restarted_pid = 4102
    assert events == [
        f"start:{initial_pid}",
        f"stop:{initial_pid}",
        "train",
        f"start:{restarted_pid}",
        "publish",
        "canary",
        f"stop:{restarted_pid}",
    ]
    assert receipt.training_succeeded
    assert receipt.publication_succeeded
    assert not receipt.canary_succeeded
    assert not receipt.service_running_after
    assert receipt.active_pid is None
    assert gate.paused
    assert runtime.owned_pid is None


def test_drain_timeout_never_stops_service_with_in_flight_request() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events, drain_timeout_seconds=0.005)
    initial_pid = runtime.start_rollout_service()
    asyncio.run(gate.acquire())

    with pytest.raises(SequentialGpuRuntimeError) as captured:
        runtime.run_update_cycle(
            training_callback=lambda: events.append("train"),
            publication_callback=lambda: events.append("publish"),
            canary_callback=lambda: True,
        )

    receipt = captured.value.receipt
    assert receipt.status == "failed_closed_at_rollout_gate_drained"
    assert not receipt.gate_drained
    assert receipt.service_running_after
    assert manager.is_alive()
    assert runtime.owned_pid == initial_pid
    assert gate.paused
    assert events == [f"start:{initial_pid}"]
    gate.release()


def test_clean_shutdown_only_after_drain() -> None:
    events: list[str] = []
    runtime, manager, gate = _runtime(events)
    initial_pid = runtime.start_rollout_service()
    runtime.stop_rollout_service()

    assert events == [f"start:{initial_pid}", f"stop:{initial_pid}"]
    assert not manager.is_alive()
    assert runtime.owned_pid is None
    assert runtime.state is SequentialGpuState.STOPPED
    assert gate.paused
