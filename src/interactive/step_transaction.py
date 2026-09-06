"""Durable phase journal for one sequential AgentGraph optimizer step.

This is a dependency-light adaptation of SkillFlow's
``training/step_transaction.py``.  It keeps the same formal step phases and
the same append/flush/fsync transition boundary.  AgentGraph trajectories are
already persisted as immutable JSONL by the runner, so the recovery record
stores their paths and the exact behavior-policy route instead of pickling a
second copy of the rollout objects.  This module does not define a learning
objective; the existing action-masked one-pass GRPO trainer remains the sole
owner of loss, backward, and ``optimizer.step()``.
"""

from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid


class StepPhase(str, Enum):
    """SkillFlow formal transaction phases, in commit order."""

    PREPARED = "prepared"
    ROLLOUT_COMPLETE = "rollout_complete"
    GRADIENT_IN_PROGRESS = "gradient_in_progress"
    GRADIENT_COMPLETE = "gradient_complete"
    OPTIMIZER_COMMITTED = "optimizer_committed"
    SGLANG_SYNCED = "sglang_synced"
    COMMITTED = "committed"


class StepTransaction:
    """Append durable transitions and atomically replace run recovery state."""

    def __init__(self, run_directory: str | Path) -> None:
        self.directory = Path(run_directory) / "recovery"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / "step_journal.jsonl"
        self.recovery = self.directory / "current_step.json"

    def transition(self, *, step: int, phase: StepPhase, **metadata: object) -> None:
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        if not isinstance(phase, StepPhase):
            raise TypeError("phase must be StepPhase")
        record = {"phase": phase.value, "step": step, **metadata}
        descriptor = os.open(
            self.journal,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def seal(self, *, step: int, state: Mapping[str, Any]) -> None:
        """Atomically bind an in-flight step to its policy and artifact paths."""

        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        payload = {"step": step, **dict(state)}
        temporary = self.directory / f"current_step_tmp_{uuid.uuid4().hex}.json"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.recovery)

    def load(self) -> dict[str, Any] | None:
        if not self.recovery.is_file():
            return None
        value = json.loads(self.recovery.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or type(value.get("step")) is not int:
            raise ValueError("private recovery state is malformed")
        return value

    def clear(self) -> None:
        self.recovery.unlink(missing_ok=True)


__all__ = ["StepPhase", "StepTransaction"]
