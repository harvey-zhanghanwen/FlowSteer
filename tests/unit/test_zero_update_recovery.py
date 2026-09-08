from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.interactive.step_transaction import StepPhase, StepTransaction


class ZeroUpdateRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.transaction = StepTransaction(self.root)
        self.state = {
            "optimizer_updates_completed": 3,
            "absolute_update_step": 4,
            "behavior_policy_version": "policy-3",
            "behavior_adapter_name": "theta-4",
            "behavior_adapter_checkpoint": str(self.root / "committed" / "theta"),
        }
        self.checkpoint = Path(self.state["behavior_adapter_checkpoint"])
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "training_state.pt").write_bytes(b"committed state")
        self.write(self.root / "run_state.json", self.state)
        self.evidence = self.root / "evidence.jsonl"
        self.evidence.write_text('"existing evidence"\n', encoding="utf-8")
        self.step = self.root / "steps" / "step_000004"
        self.summary = {
            "optimizer_updates": 0,
            "trained_groups": 0,
            "trained_trajectories": 0,
            "grad_norm": 0.0,
            "trainable_update_l2": 0.0,
            "loss": 0.0,
            "updated_policy_version": "",
            "checkpoint_dir": "",
            "optimizer_state_checkpoint": "",
            "training_state_checkpoint": "",
            "checkpoint_recoverable": False,
            "optimizer_state_saved": False,
            "training_state_saved": False,
            "scheduler_state_saved": False,
            "rng_state_saved": False,
            "update_step": 5,
            "committed_step": 4,
            "behavior_policy_version": self.state["behavior_policy_version"],
            "continuation_adapter_checkpoint": self.state["behavior_adapter_checkpoint"],
            "exact_groups": 2,
            "excluded_groups": 2,
            "exclusions": {
                "group-a": "zero_information_group",
                "group-b": "behavior_logprob_tolerance_exceeded",
            },
        }
        self.make_attempt()

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_attempt(self) -> None:
        self.write(self.step / "learner" / "training_summary.json", self.summary)
        (self.step / "trajectories.jsonl").write_text("old rollout\n", encoding="utf-8")
        self.transaction.transition(step=4, phase=StepPhase.PREPARED)
        self.transaction.transition(step=4, phase=StepPhase.ROLLOUT_COMPLETE)
        self.transaction.transition(step=4, phase=StepPhase.GRADIENT_IN_PROGRESS)
        self.transaction.seal(step=4, state={
            "phase": "rollout_complete",
            "behavior_policy_version": self.state["behavior_policy_version"],
            "behavior_adapter_name": self.state["behavior_adapter_name"],
            "candidate_policy_version": "policy-4",
        })

    def recover(self):
        return self.transaction.recover_zero_update(self.state, 5)

    def test_rejected_attempt_is_archived_without_commit_or_evidence_changes(self):
        preserved = [self.root / "run_state.json", self.evidence,
                     self.transaction.journal, self.checkpoint / "training_state.pt"]
        before = [path.read_bytes() for path in preserved]
        receipt = self.recover()
        self.assertEqual(1, receipt["attempt_index"])
        self.assertEqual(100_000_000, receipt["rollout_index_offset"])
        self.assertFalse(receipt["consumed_by_optimizer"])
        self.assertEqual(0, receipt["optimizer_updates"])
        self.assertFalse(self.step.exists())
        self.assertIsNone(self.transaction.load())
        self.assertEqual(before, [path.read_bytes() for path in preserved])
        archived = Path(receipt["archived_step_directory"])
        self.assertEqual("old rollout\n", (archived / "trajectories.jsonl").read_text())
        self.assertEqual(self.summary, json.loads((archived / "learner" / "training_summary.json").read_text()))
        self.assertTrue(Path(receipt["archived_recovery_path"]).is_file())

    def test_repeat_before_sampling_returns_same_durable_namespace(self):
        receipt = self.recover()
        self.assertEqual(receipt, self.recover())
        self.assertEqual({"attempt_index": 1, "rollout_index_offset": 100_000_000},
                         StepTransaction(self.root).retry_context(4))
        self.assertEqual({"attempt_index": 0, "rollout_index_offset": 0},
                         self.transaction.retry_context(5))

    def test_second_rejection_increments_namespace_and_retains_first_attempt(self):
        first = self.recover()
        original = Path(first["archived_step_directory"]) / "trajectories.jsonl"
        self.make_attempt()
        second = self.recover()
        self.assertEqual(2, second["attempt_index"])
        self.assertEqual(200_000_000, second["rollout_index_offset"])
        self.assertNotEqual(first["archive_directory"], second["archive_directory"])
        self.assertEqual("old rollout\n", original.read_text())

    def test_no_inflight_or_retry_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            transaction = StepTransaction(directory)
            self.assertIsNone(transaction.recover_zero_update(self.state, 5))
            self.assertEqual({"attempt_index": 0, "rollout_index_offset": 0},
                             transaction.retry_context(4))

    def test_rejects_nonzero_or_ambiguous_summary_without_archiving(self):
        changes = {
            "optimizer_updates": 1, "trained_groups": 1, "trained_trajectories": 4,
            "grad_norm": 0.1, "trainable_update_l2": 0.1, "loss": 0.1,
            "updated_policy_version": "policy-4", "checkpoint_dir": "new/theta",
            "optimizer_state_checkpoint": "new/state.pt", "training_state_checkpoint": "state.pt",
            "checkpoint_recoverable": True, "optimizer_state_saved": True,
            "training_state_saved": True, "scheduler_state_saved": True, "rng_state_saved": True,
            "update_step": 4, "committed_step": 3,
            "behavior_policy_version": "different-policy", "continuation_adapter_checkpoint": "other",
            "exact_groups": 3, "excluded_groups": 1,
            "exclusions": {"a": "zero_information_group", "b": "incomplete_exact_group"},
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                self.write(self.step / "learner" / "training_summary.json", {**self.summary, name: value})
                with self.assertRaises(ValueError):
                    self.recover()
                self.assertTrue(self.step.is_dir())
                self.assertTrue(self.transaction.recovery.is_file())
        self.write(self.step / "learner" / "training_summary.json", self.summary)

    def test_rejects_boolean_zero_and_nonfinite_gradient(self):
        for name, value in (("optimizer_updates", False), ("grad_norm", False),
                            ("grad_norm", float("nan")), ("trainable_update_l2", float("inf"))):
            with self.subTest(name=name, value=value):
                self.write(self.step / "learner" / "training_summary.json", {**self.summary, name: value})
                with self.assertRaises(ValueError):
                    self.recover()

    def test_rejects_wrong_inflight_step_route_or_post_update_phase(self):
        current = self.transaction.load()
        for changes in ({"step": 5}, {"step": True}, {"behavior_policy_version": "other"},
                        {"behavior_adapter_name": "other"}, {"phase": "gradient_complete"}):
            with self.subTest(changes=changes):
                self.write(self.transaction.recovery, {**current, **changes})
                with self.assertRaises(ValueError):
                    self.recover()

    def test_any_post_gradient_journal_entry_prevents_zero_update_replay(self):
        self.transaction.transition(step=4, phase=StepPhase.GRADIENT_COMPLETE)
        self.transaction.transition(step=4, phase=StepPhase.GRADIENT_IN_PROGRESS)
        with self.assertRaises(ValueError):
            self.recover()
        self.assertTrue(self.step.is_dir())

    def test_checkpoint_presence_prevents_replay_even_with_zero_summary(self):
        (self.step / "learner" / "checkpoint_final").mkdir()
        with self.assertRaises(ValueError):
            self.recover()

    def test_pending_archive_resumes_after_each_rename_interruption(self):
        import os

        original_rename = os.rename
        for target_name in ("current_step.json", "archive_plan.json"):
            with self.subTest(target_name=target_name):
                def interrupted(source, destination):
                    if Path(destination).name == target_name:
                        raise OSError("simulated interrupted archive")
                    return original_rename(source, destination)

                with patch("src.interactive.step_transaction.os.rename", side_effect=interrupted):
                    with self.assertRaises(OSError):
                        self.recover()
                self.assertTrue((self.transaction.directory / "zero_update_archive_pending.json").is_file())
                receipt = StepTransaction(self.root).recover_zero_update(self.state, 5)
                self.assertEqual("ready_for_resampling", receipt["status"])
                self.assertFalse((self.transaction.directory / "zero_update_archive_pending.json").exists())
                self.make_attempt()

    def test_conflicting_archive_target_fails_without_overwriting(self):
        target = self.transaction.directory / "zero_update_attempts" / "step_000004" / "attempt_000001" / "step"
        target.mkdir(parents=True)
        sentinel = target / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.recover()
        self.assertEqual("keep", sentinel.read_text())
        self.assertTrue(self.step.is_dir())

    def test_invalid_context_is_rejected(self):
        path = self.transaction.directory / "zero_update_retry_state.json"
        self.write(path, {"step": 4, "attempt_index": 1, "rollout_index_offset": 0,
                          "status": "ready_for_resampling"})
        with self.assertRaises(ValueError):
            self.transaction.retry_context(4)

    def test_persisted_retry_namespace_cannot_be_rebound_to_another_policy(self):
        self.recover()
        for change in ({"behavior_policy_version": "other"},
                       {"behavior_adapter_checkpoint": "other/theta"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.transaction.recover_zero_update({**self.state, **change}, 5)

    def test_committed_checkpoint_inside_attempt_cannot_be_archived(self):
        self.state["behavior_adapter_checkpoint"] = str(self.step / "committed-theta")
        with self.assertRaises(ValueError):
            self.recover()
        self.assertTrue(self.step.is_dir())

    def test_absolute_and_optimizer_step_numbers_remain_distinct(self):
        with self.assertRaises(ValueError):
            self.transaction.recover_zero_update(self.state, 4)
        self.assertEqual(5, self.recover()["expected_absolute_update_step"])


if __name__ == "__main__":
    unittest.main()
