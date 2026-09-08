"""Durable phase journal for one sequential AgentGraph optimizer step.

SkillFlow does not publish a ``training/step_transaction.py`` module.  This is
the project transaction journal required by the MD's strict on-policy update
boundary; its append/flush/fsync persistence is an engineering adaptation, not
an upstream SkillFlow class.  AgentGraph trajectories are
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
    """Project transaction phases for the MD-required commit order."""

    PREPARED = "prepared"
    ROLLOUT_COMPLETE = "rollout_complete"
    GRADIENT_IN_PROGRESS = "gradient_in_progress"
    GRADIENT_COMPLETE = "gradient_complete"
    OPTIMIZER_COMMITTED = "optimizer_committed"
    SGLANG_SYNCED = "sglang_synced"
    VALIDATION_COMPLETE = "validation_complete"
    WANDB_ARTIFACT_LOGGED = "wandb_artifact_logged"
    WANDB_LOGGED = "wandb_logged"
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

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"recovery object is malformed: {path.name}")
        return value

    @staticmethod
    def _write_object(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        StepTransaction._sync_directory(path.parent)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def retry_context(self, step: int) -> dict[str, int]:
        """Return the durable fresh-sampling namespace for this optimizer step."""

        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        path = self.directory / "zero_update_retry_state.json"
        if not path.is_file():
            return {"attempt_index": 0, "rollout_index_offset": 0}
        value = self._read_object(path)
        attempt = value.get("attempt_index")
        if (
            type(value.get("step")) is not int
            or value["step"] < 1
            or type(attempt) is not int
            or attempt < 1
            or type(value.get("rollout_index_offset")) is not int
            or value["rollout_index_offset"] != attempt * 100_000_000
            or value.get("status") != "ready_for_resampling"
        ):
            raise ValueError("zero-update retry state is malformed")
        if value["step"] != step:
            return {"attempt_index": 0, "rollout_index_offset": 0}
        return {
            "attempt_index": attempt,
            "rollout_index_offset": value["rollout_index_offset"],
        }

    def _validate_zero_update(
        self,
        *,
        step: int,
        run_state: Mapping[str, Any],
        expected_absolute_update_step: int,
        step_directory: Path,
        recovery_path: Path,
    ) -> dict[str, Any]:
        """Admit only the trainer's two pre-backward, zero-group returns."""

        current = self._read_object(recovery_path)
        allowed_phases = {
            StepPhase.PREPARED.value,
            StepPhase.ROLLOUT_COMPLETE.value,
            StepPhase.GRADIENT_IN_PROGRESS.value,
        }
        if (
            type(current.get("step")) is not int
            or current["step"] != step
            or current.get("phase") not in allowed_phases
            or current.get("behavior_policy_version")
            != run_state["behavior_policy_version"]
            or current.get("behavior_adapter_name") != run_state["behavior_adapter_name"]
        ):
            raise ValueError("in-flight zero-update policy or phase differs")

        phases = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if not isinstance(item, dict) or type(item.get("step")) is not int:
                raise ValueError("step journal is malformed")
            if item["step"] == step:
                phases.append(item.get("phase"))
        if (
            not phases
            or phases[-1] != StepPhase.GRADIENT_IN_PROGRESS.value
            or any(phase not in allowed_phases for phase in phases)
        ):
            raise ValueError("step journal does not prove a pre-update rejection")

        summary = self._read_object(step_directory / "learner" / "training_summary.json")
        for name in ("optimizer_updates", "trained_groups", "trained_trajectories"):
            if type(summary.get(name)) is not int or summary[name] != 0:
                raise ValueError(f"zero-update recovery requires {name}=0")
        for name in ("grad_norm", "trainable_update_l2", "loss"):
            if type(summary.get(name)) not in (int, float) or summary[name] != 0:
                raise ValueError(f"zero-update recovery requires {name}=0")
        for name in (
            "updated_policy_version", "checkpoint_dir", "optimizer_state_checkpoint",
            "training_state_checkpoint",
        ):
            if summary.get(name) != "":
                raise ValueError(f"zero-update recovery requires empty {name}")
        for name in (
            "checkpoint_recoverable", "optimizer_state_saved", "training_state_saved",
            "scheduler_state_saved", "rng_state_saved",
        ):
            if summary.get(name) is not False:
                raise ValueError(f"zero-update recovery requires {name}=false")
        if (
            type(summary.get("update_step")) is not int
            or summary["update_step"] != expected_absolute_update_step
            or type(summary.get("committed_step")) is not int
            or summary["committed_step"] != expected_absolute_update_step - 1
            or summary.get("behavior_policy_version") != run_state["behavior_policy_version"]
            or summary.get("continuation_adapter_checkpoint")
            != run_state["behavior_adapter_checkpoint"]
        ):
            raise ValueError("zero-update summary differs from committed continuation")
        exact = summary.get("exact_groups")
        exclusions = summary.get("exclusions")
        if (
            type(exact) is not int or exact < 1
            or type(summary.get("excluded_groups")) is not int
            or summary["excluded_groups"] != exact
            or not isinstance(exclusions, dict) or len(exclusions) != exact
            or any(not isinstance(key, str) or not key for key in exclusions)
            or any(reason not in {
                "zero_information_group", "behavior_logprob_tolerance_exceeded",
                # Known pre-backward receipt rejection: discard the whole
                # proven-zero-update batch, never admit malformed tokens.
                "invalid_executed_action_span",
            } for reason in exclusions.values())
        ):
            raise ValueError("not every exact group has an allowed rejection reason")
        for relative in (
            "learner/checkpoint_final", "sync_receipt.json", "post_update_canary.jsonl",
            "validation_monitor.jsonl", "wandb_checkpoint_artifact.json", "step_manifest.json",
        ):
            if (step_directory / relative).exists():
                raise ValueError("zero-update attempt contains post-update artifacts")
        return summary

    def recover_zero_update(
        self,
        run_state: Mapping[str, Any],
        expected_absolute_update_step: int,
    ) -> dict[str, Any] | None:
        """Archive a proven zero-update attempt, without replaying its rollouts.

        This is an operational MD adaptation over the existing journal and
        smoke trainer, not a SkillFlow retry algorithm. The caller must hold
        the existing run-process lock. Ambiguous or post-update failures remain
        blocked; the committed checkpoint, journal and evidence are untouched.
        """

        completed = run_state.get("optimizer_updates_completed")
        if type(completed) is not int or completed < 0:
            raise ValueError("committed optimizer step must be a non-negative integer")
        if type(expected_absolute_update_step) is not int or expected_absolute_update_step < 1:
            raise ValueError("expected absolute update step must be positive")
        if "absolute_update_step" in run_state and (
            type(run_state["absolute_update_step"]) is not int
            or run_state["absolute_update_step"] != expected_absolute_update_step - 1
        ):
            raise ValueError("absolute update step differs from committed state")
        for name in ("behavior_policy_version", "behavior_adapter_name", "behavior_adapter_checkpoint"):
            if not isinstance(run_state.get(name), str) or not run_state[name].strip():
                raise ValueError(f"committed state lacks {name}")
        step = completed + 1
        pending_path = self.directory / "zero_update_archive_pending.json"
        state_path = self.directory / "zero_update_retry_state.json"
        context = self.retry_context(step)
        if context["attempt_index"]:
            previous = self._read_object(state_path)
            if (
                previous.get("expected_absolute_update_step") != expected_absolute_update_step
                or previous.get("behavior_policy_version") != run_state["behavior_policy_version"]
                or previous.get("behavior_adapter_checkpoint") != run_state["behavior_adapter_checkpoint"]
            ):
                raise ValueError("retry namespace differs from committed continuation")
        if not pending_path.exists() and not self.recovery.exists():
            if context["attempt_index"]:
                return self._read_object(state_path)
            return None

        source = self.directory.parent / "steps" / f"step_{step:06d}"
        for name in ("behavior_adapter_checkpoint", "optimizer_state_checkpoint"):
            value = run_state.get(name)
            if isinstance(value, str) and value:
                committed_path = Path(value).resolve()
                if committed_path == source.resolve() or source.resolve() in committed_path.parents:
                    raise ValueError("attempt contains the committed continuation checkpoint")
        if pending_path.exists():
            plan = self._read_object(pending_path)
            attempt = plan.get("attempt_index")
            if (
                type(attempt) is not int or attempt < 1
                or attempt not in {context["attempt_index"], context["attempt_index"] + 1}
                or plan.get("step") != step
                or plan.get("expected_absolute_update_step") != expected_absolute_update_step
                or plan.get("behavior_policy_version") != run_state["behavior_policy_version"]
                or plan.get("behavior_adapter_checkpoint") != run_state["behavior_adapter_checkpoint"]
                or plan.get("rollout_index_offset") != attempt * 100_000_000
            ):
                raise ValueError("pending zero-update archive differs from committed state")
        else:
            summary = self._validate_zero_update(
                step=step, run_state=run_state,
                expected_absolute_update_step=expected_absolute_update_step,
                step_directory=source, recovery_path=self.recovery,
            )
            attempt = context["attempt_index"] + 1
            plan = {
                "schema_version": "flowsteer.zero-update-recovery.v1",
                "step": step,
                "attempt_index": attempt,
                "rejected_attempt_index": context["attempt_index"],
                "rollout_index_offset": attempt * 100_000_000,
                "expected_absolute_update_step": expected_absolute_update_step,
                "behavior_policy_version": run_state["behavior_policy_version"],
                "behavior_adapter_checkpoint": run_state["behavior_adapter_checkpoint"],
                "optimizer_updates": 0,
                "consumed_by_optimizer": False,
                "exclusions": summary["exclusions"],
            }
            self._write_object(pending_path, plan)

        archive = self.directory / "zero_update_attempts" / f"step_{step:06d}" / f"attempt_{attempt:06d}"
        archived_step = archive / "step"
        archived_current = archive / "current_step.json"
        if source.exists() == archived_step.exists() or self.recovery.exists() == archived_current.exists():
            raise ValueError("zero-update archive has absent or conflicting source/target")
        self._validate_zero_update(
            step=step, run_state=run_state,
            expected_absolute_update_step=expected_absolute_update_step,
            step_directory=source if source.exists() else archived_step,
            recovery_path=self.recovery if self.recovery.exists() else archived_current,
        )
        archive.mkdir(parents=True, exist_ok=True)
        if source.exists():
            os.rename(source, archived_step)
            self._sync_directory(source.parent)
            self._sync_directory(archive)
        if self.recovery.exists():
            os.rename(self.recovery, archived_current)
            self._sync_directory(self.directory)
            self._sync_directory(archive)
        receipt = {
            **plan,
            "status": "ready_for_resampling",
            "archive_directory": str(archive),
            "archived_step_directory": str(archived_step),
            "archived_recovery_path": str(archived_current),
        }
        self._write_object(archive / "recovery_receipt.json", receipt)
        self._write_object(state_path, receipt)
        archived_plan = archive / "archive_plan.json"
        if archived_plan.exists():
            raise ValueError("completed zero-update archive plan already exists")
        os.rename(pending_path, archived_plan)
        self._sync_directory(self.directory)
        self._sync_directory(archive)
        return receipt


__all__ = ["StepPhase", "StepTransaction"]
