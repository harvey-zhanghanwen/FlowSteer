"""Single-GPU time-multiplexed rollout/training lifecycle.

This module composes the existing :class:`RolloutGate` and
:class:`SGLangSupervisorManager` boundaries.  It is a resource-constrained
project engineering adaptation for machines where only one physical GPU is
available: it is **not** presented as the GPU layout used by SkillFlow.

The lifecycle is deliberately fail-closed.  A successful update has exactly
this order::

    rollout service running
      -> pause and drain rollout admission
      -> stop the runtime-owned SGLang child
      -> run the training callback
      -> restart the runtime-owned SGLang child
      -> publish the new policy
      -> open one bounded canary admission window
      -> run a canary and close that window
      -> resume rollout admission

No model, API, trainer, or publisher is imported or invoked at module import
time.  Callbacks keep those concerns in their existing modules.  This class
never sends an operating-system signal itself; all process operations go
through the supplied ``SGLangSupervisorManager``, and it refuses to adopt or
stop a process that it did not start through :meth:`start_rollout_service`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import time
from typing import Any, Optional, Protocol

from .rollout_collector import RolloutGate
from .sglang_manager import SGLangSupervisorManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SequentialGpuState(str, Enum):
    """Externally observable state of the time-multiplexed runtime."""

    STOPPED = "stopped"
    ROLLOUT_RUNNING = "rollout_running"
    TRANSITIONING = "transitioning"
    TRAINING = "training"
    RESTARTING = "restarting"
    PUBLISHING = "publishing"
    CANARY = "canary"
    FAILED_CLOSED = "failed_closed"


@dataclass(frozen=True)
class SequentialGpuStageReceipt:
    """Timing and process evidence for one lifecycle stage."""

    stage: str
    success: bool
    started_at: str
    completed_at: str
    duration_seconds: float
    pid_before: Optional[int]
    pid_after: Optional[int]
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequentialGpuCycleReceipt:
    """Immutable receipt for one attempted rollout-policy update cycle."""

    gpu_id: int
    success: bool
    status: str
    initial_pid: Optional[int]
    active_pid: Optional[int]
    gate_paused: bool
    gate_drained: bool
    service_running_after: bool
    training_succeeded: bool
    publication_succeeded: bool
    canary_succeeded: bool
    stages: tuple[SequentialGpuStageReceipt, ...] = field(default_factory=tuple)
    error: str = ""
    started_at: str = field(default_factory=_utc_now)
    completed_at: str = field(default_factory=_utc_now)
    duration_seconds: float = 0.0
    implementation_source: str = "project_single_gpu_time_multiplexing"

    def __post_init__(self) -> None:
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be non-negative")
        if not self.status.strip():
            raise ValueError("status must be non-empty")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.success:
            if not (
                self.gate_drained
                and self.service_running_after
                and self.training_succeeded
                and self.publication_succeeded
                and self.canary_succeeded
            ):
                raise ValueError("successful cycle receipt has incomplete evidence")
            if self.gate_paused:
                raise ValueError("successful cycle must resume rollout admission")
        elif not self.gate_paused:
            raise ValueError("failed cycle must remain fail-closed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequentialGpuRuntimeError(RuntimeError):
    """A lifecycle stage failed and rollout admission remains closed."""

    def __init__(self, message: str, receipt: SequentialGpuCycleReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class _ManagedSGLang(Protocol):
    """Public manager surface used by this module and its pure mock tests."""

    gpu_id: int

    def start(self) -> None: ...

    def stop(self, timeout_seconds: float = 30.0) -> None: ...

    def is_alive(self) -> bool: ...

    def pid(self) -> Optional[int]: ...


def _reported_failure(result: object) -> bool:
    """Recognize an explicit callback failure without constraining result type."""

    if isinstance(result, bool):
        return not result
    if isinstance(result, Mapping) and isinstance(result.get("success"), bool):
        return not bool(result["success"])
    success = getattr(result, "success", None)
    return isinstance(success, bool) and not success


def _explicit_canary_success(result: object) -> bool:
    """Require positive canary evidence; absence of evidence is not success."""

    if isinstance(result, bool):
        return result
    if isinstance(result, Mapping):
        value = result.get("canary_succeeded", result.get("success"))
        return value is True
    value = getattr(result, "canary_succeeded", None)
    if value is None:
        value = getattr(result, "success", None)
    return value is True


class SequentialGpuRuntime:
    """Own one SGLang child and time-multiplex it with a training callback.

    ``start_rollout_service`` must launch the service before an update cycle.
    A process that was already alive before that method is never adopted.  The
    runtime therefore cannot stop an external service that merely happens to
    use the same port or GPU.
    """

    def __init__(
        self,
        *,
        manager: _ManagedSGLang | SGLangSupervisorManager,
        rollout_gate: RolloutGate,
        drain_timeout_seconds: float = 300.0,
        stop_timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(rollout_gate, RolloutGate):
            raise TypeError("rollout_gate must be RolloutGate")
        if drain_timeout_seconds <= 0 or stop_timeout_seconds <= 0:
            raise ValueError("drain and stop timeouts must be positive")
        if not all(
            callable(getattr(manager, name, None))
            for name in ("start", "stop", "is_alive", "pid")
        ):
            raise TypeError("manager does not implement the SGLang manager surface")
        gpu_id = getattr(manager, "gpu_id", None)
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("manager.gpu_id must be a non-negative integer")

        self.manager = manager
        self.rollout_gate = rollout_gate
        self.drain_timeout_seconds = float(drain_timeout_seconds)
        self.stop_timeout_seconds = float(stop_timeout_seconds)
        self._state = SequentialGpuState.STOPPED
        self._owned_pid: Optional[int] = None
        self._cycle_lock = threading.Lock()

    @property
    def state(self) -> SequentialGpuState:
        return self._state

    @property
    def owned_pid(self) -> Optional[int]:
        return self._owned_pid

    def start_rollout_service(self) -> int:
        """Start, but never adopt, the manager-owned SGLang child."""

        if self._owned_pid is not None:
            raise RuntimeError("runtime already owns a rollout service")
        if self.manager.is_alive() or self.manager.pid() is not None:
            raise RuntimeError("refusing to adopt a pre-existing manager process")
        if self.rollout_gate.paused or self.rollout_gate.in_flight:
            raise RuntimeError("rollout gate must be open and idle before service start")

        self.manager.start()
        pid = self.manager.pid()
        if not self.manager.is_alive() or type(pid) is not int or pid <= 0:
            # ``stop`` can only target the child stored by this manager.
            self.manager.stop(timeout_seconds=self.stop_timeout_seconds)
            raise RuntimeError("SGLang manager did not expose a live child after start")
        self._owned_pid = pid
        self._state = SequentialGpuState.ROLLOUT_RUNNING
        return pid

    def stop_rollout_service(self) -> None:
        """Drain and stop only the child launched by this runtime."""

        if self._owned_pid is None:
            if self.manager.is_alive() or self.manager.pid() is not None:
                raise RuntimeError("refusing to stop a process not owned by this runtime")
            self._state = SequentialGpuState.STOPPED
            return
        self._require_owned_live_process()
        self.rollout_gate.pause()
        if not self.rollout_gate.wait_for_drain(self.drain_timeout_seconds):
            self._state = SequentialGpuState.FAILED_CLOSED
            raise TimeoutError("timed out draining rollout requests before shutdown")
        self.manager.stop(timeout_seconds=self.stop_timeout_seconds)
        if self.manager.is_alive():
            self._state = SequentialGpuState.FAILED_CLOSED
            raise RuntimeError("runtime-owned SGLang child remained alive after stop")
        self._owned_pid = None
        self._state = SequentialGpuState.STOPPED

    def run_update_cycle(
        self,
        *,
        training_callback: Callable[[], object],
        publication_callback: Callable[[], object],
        canary_callback: Callable[[], object],
    ) -> SequentialGpuCycleReceipt:
        """Run one serialized update and resume rollout only after the canary.

        The ordinary Director client acquires ``rollout_gate`` for every
        rollout, including the post-publication canary.  Keeping the gate
        paused while invoking that callback would therefore deadlock: the
        callback waits for admission while this method waits for the callback
        before resuming admission.  Open the gate only for the synchronous
        canary callback, close it again before accepting the result, and leave
        every failure path paused.  Callers of this single-update lifecycle
        must not prefetch or otherwise enqueue ordinary rollouts during this
        bounded window.
        """

        for name, callback in {
            "training_callback": training_callback,
            "publication_callback": publication_callback,
            "canary_callback": canary_callback,
        }.items():
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if not self._cycle_lock.acquire(blocking=False):
            raise RuntimeError("another sequential GPU update cycle is active")

        cycle_started_at = _utc_now()
        cycle_started = time.monotonic()
        stages: list[SequentialGpuStageReceipt] = []
        initial_pid = self._owned_pid
        gate_drained = False
        training_succeeded = False
        publication_succeeded = False
        canary_succeeded = False
        current_stage = "rollout_service_running"
        try:
            self._require_owned_live_process()
            self._append_stage(stages, current_stage, True, initial_pid, initial_pid)

            self._state = SequentialGpuState.TRANSITIONING
            current_stage = "rollout_gate_paused"
            before = self.manager.pid()
            started_at, started = _utc_now(), time.monotonic()
            self.rollout_gate.pause()
            self._append_timed_stage(stages, current_stage, True, before, before, started_at, started)

            current_stage = "rollout_gate_drained"
            started_at, started = _utc_now(), time.monotonic()
            gate_drained = self.rollout_gate.wait_for_drain(self.drain_timeout_seconds)
            if not gate_drained:
                raise TimeoutError("timed out waiting for in-flight rollout requests")
            self._append_timed_stage(stages, current_stage, True, before, before, started_at, started)

            current_stage = "rollout_service_stopped"
            self._require_owned_live_process()
            before = self.manager.pid()
            started_at, started = _utc_now(), time.monotonic()
            self.manager.stop(timeout_seconds=self.stop_timeout_seconds)
            if self.manager.is_alive() or self.manager.pid() is not None:
                raise RuntimeError("runtime-owned SGLang child remained present after stop")
            self._owned_pid = None
            self._append_timed_stage(stages, current_stage, True, before, None, started_at, started)

            self._state = SequentialGpuState.TRAINING
            current_stage = "training_callback"
            started_at, started = _utc_now(), time.monotonic()
            training_result = training_callback()
            if _reported_failure(training_result):
                raise RuntimeError("training callback reported failure")
            training_succeeded = True
            self._append_timed_stage(stages, current_stage, True, None, None, started_at, started)

            self._state = SequentialGpuState.RESTARTING
            current_stage = "rollout_service_restarted"
            started_at, started = _utc_now(), time.monotonic()
            self.manager.start()
            restarted_pid = self.manager.pid()
            if (
                not self.manager.is_alive()
                or type(restarted_pid) is not int
                or restarted_pid <= 0
            ):
                raise RuntimeError("SGLang service did not become ready after restart")
            self._owned_pid = restarted_pid
            self._append_timed_stage(stages, current_stage, True, None, restarted_pid, started_at, started)

            self._state = SequentialGpuState.PUBLISHING
            current_stage = "policy_publication"
            started_at, started = _utc_now(), time.monotonic()
            publication_result = publication_callback()
            if _reported_failure(publication_result):
                raise RuntimeError("publication callback reported failure")
            publication_succeeded = True
            self._append_timed_stage(
                stages,
                current_stage,
                True,
                restarted_pid,
                restarted_pid,
                started_at,
                started,
            )

            self._state = SequentialGpuState.CANARY
            current_stage = "policy_canary"
            started_at, started = _utc_now(), time.monotonic()
            self.rollout_gate.resume()
            try:
                canary_result = canary_callback()
            finally:
                # The callback uses the normal rollout path, so admission has
                # to be open while it runs.  Re-close it before interpreting
                # success or entering any failure cleanup.
                self.rollout_gate.pause()
            if not self.rollout_gate.wait_for_drain(self.drain_timeout_seconds):
                raise TimeoutError("timed out draining the bounded canary window")
            if not _explicit_canary_success(canary_result):
                raise RuntimeError("canary callback did not report explicit success")
            canary_succeeded = True
            self._append_timed_stage(
                stages,
                current_stage,
                True,
                restarted_pid,
                restarted_pid,
                started_at,
                started,
            )

            current_stage = "rollout_gate_resumed"
            started_at, started = _utc_now(), time.monotonic()
            self.rollout_gate.resume()
            self._state = SequentialGpuState.ROLLOUT_RUNNING
            self._append_timed_stage(
                stages,
                current_stage,
                True,
                restarted_pid,
                restarted_pid,
                started_at,
                started,
            )
            return SequentialGpuCycleReceipt(
                gpu_id=self.manager.gpu_id,
                success=True,
                status="rollout_resumed_after_verified_policy_update",
                initial_pid=initial_pid,
                active_pid=restarted_pid,
                gate_paused=self.rollout_gate.paused,
                gate_drained=gate_drained,
                service_running_after=self.manager.is_alive(),
                training_succeeded=training_succeeded,
                publication_succeeded=publication_succeeded,
                canary_succeeded=canary_succeeded,
                stages=tuple(stages),
                started_at=cycle_started_at,
                completed_at=_utc_now(),
                duration_seconds=time.monotonic() - cycle_started,
            )
        except BaseException as exc:
            failure_pid_before = self.manager.pid()
            self._append_stage(
                stages,
                current_stage,
                False,
                failure_pid_before,
                failure_pid_before,
                detail=f"{type(exc).__name__}: {exc}",
            )
            self.rollout_gate.pause()
            cleanup_error = ""
            # Do not interrupt in-flight requests after a drain timeout.  At
            # every later stage they are proven drained, so a restarted child
            # can be stopped through its owning manager to prevent an
            # unverified policy from serving requests.
            if gate_drained and self._owned_pid is not None and self.manager.is_alive():
                try:
                    self._require_owned_live_process()
                    cleanup_before = self.manager.pid()
                    cleanup_started_at, cleanup_started = _utc_now(), time.monotonic()
                    self.manager.stop(timeout_seconds=self.stop_timeout_seconds)
                    if self.manager.is_alive() or self.manager.pid() is not None:
                        raise RuntimeError("runtime-owned child survived failure cleanup")
                    self._owned_pid = None
                    self._append_timed_stage(
                        stages,
                        "failure_cleanup_service_stop",
                        True,
                        cleanup_before,
                        None,
                        cleanup_started_at,
                        cleanup_started,
                    )
                except BaseException as cleanup_exc:
                    cleanup_error = f"; cleanup={type(cleanup_exc).__name__}: {cleanup_exc}"
            self._state = SequentialGpuState.FAILED_CLOSED
            error = f"{type(exc).__name__}: {exc}{cleanup_error}"
            receipt = SequentialGpuCycleReceipt(
                gpu_id=self.manager.gpu_id,
                success=False,
                status=f"failed_closed_at_{current_stage}",
                initial_pid=initial_pid,
                active_pid=self.manager.pid(),
                gate_paused=self.rollout_gate.paused,
                gate_drained=gate_drained,
                service_running_after=self.manager.is_alive(),
                training_succeeded=training_succeeded,
                publication_succeeded=publication_succeeded,
                canary_succeeded=canary_succeeded,
                stages=tuple(stages),
                error=error,
                started_at=cycle_started_at,
                completed_at=_utc_now(),
                duration_seconds=time.monotonic() - cycle_started,
            )
            raise SequentialGpuRuntimeError(error, receipt) from exc
        finally:
            self._cycle_lock.release()

    def _require_owned_live_process(self) -> int:
        pid = self.manager.pid()
        if self._owned_pid is None:
            raise RuntimeError("runtime does not own a rollout service")
        if pid != self._owned_pid or not self.manager.is_alive():
            raise RuntimeError("manager process is not the runtime-owned live child")
        return self._owned_pid

    @staticmethod
    def _append_stage(
        stages: list[SequentialGpuStageReceipt],
        stage: str,
        success: bool,
        pid_before: Optional[int],
        pid_after: Optional[int],
        *,
        detail: str = "",
    ) -> None:
        timestamp = _utc_now()
        stages.append(
            SequentialGpuStageReceipt(
                stage=stage,
                success=success,
                started_at=timestamp,
                completed_at=timestamp,
                duration_seconds=0.0,
                pid_before=pid_before,
                pid_after=pid_after,
                detail=detail,
            )
        )

    @staticmethod
    def _append_timed_stage(
        stages: list[SequentialGpuStageReceipt],
        stage: str,
        success: bool,
        pid_before: Optional[int],
        pid_after: Optional[int],
        started_at: str,
        started: float,
        *,
        detail: str = "",
    ) -> None:
        stages.append(
            SequentialGpuStageReceipt(
                stage=stage,
                success=success,
                started_at=started_at,
                completed_at=_utc_now(),
                duration_seconds=time.monotonic() - started,
                pid_before=pid_before,
                pid_after=pid_after,
                detail=detail,
            )
        )


__all__ = [
    "SequentialGpuCycleReceipt",
    "SequentialGpuRuntime",
    "SequentialGpuRuntimeError",
    "SequentialGpuStageReceipt",
    "SequentialGpuState",
]
