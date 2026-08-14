from __future__ import annotations

from dataclasses import replace
import json
import math
import tempfile
from pathlib import Path
import unittest

from src.interactive.persistence import (
    AppendOnlyJsonlStore,
    GraphSnapshotEvent,
    SnapshotReplayError,
    replay_snapshots,
    stable_id,
)
from src.interactive.persistence.trajectory_store import DuplicateEventError, EvidenceStore
from src.interactive.skills import (
    SkillEvidence,
    SkillEvidenceGate,
    SkillGateConfig,
    SkillLifecycleManager,
    SkillQuery,
    SkillRecord,
    SkillRetriever,
    SkillStatus,
    SkillStore,
)
from src.interactive.versioning import VersionBundle


def versions(policy: str = "policy-e1") -> VersionBundle:
    return VersionBundle(
        policy=policy,
        model_catalog="catalog-v1",
        evaluator="eval-v1",
        prompt="prompt-v1",
        tool="tools-v1",
        encoder="encoder-v1",
        feature_schema="features-v1",
        posterior="posterior-v1",
        skill_library="skills-v1",
    )


def candidate_skill(**evidence_overrides: object) -> SkillRecord:
    evidence_values = dict(
        baseline="same-contract-model-a",
        paired_effect_mean=0.12,
        calibrated_lower=0.06,
        calibrated_upper=0.18,
        effective_pairs=4,
        independent_problem_ids=["v1", "v2", "v3", "v4"],
        discovery_problem_ids=["d1", "d2"],
        validation_problem_ids=["v1", "v2", "v3", "v4"],
        validation_splits=["validation"],
        heldout_task_families=["math"],
        empirical_coverage=0.95,
        harm_probability=0.01,
        slice_effects={"algebra": 0.08, "geometry": 0.05},
        evidence_ids=["e1", "e2", "e3", "e4"],
    )
    evidence_values.update(evidence_overrides)
    return SkillRecord(
        skill_id="verify-math-v1",
        version=1,
        status=SkillStatus.CANDIDATE,
        condition={"task_family": "math", "graph_stage": "before_final", "tags": ["long"]},
        action={"model_id": "qwen35", "instruction": "independently verify"},
        evidence=SkillEvidence(**evidence_values),
        versions=versions(),
        failure_scope=["short_arithmetic"],
        created_epoch=1,
        eligible_epoch=2,
    )


class PersistenceTests(unittest.TestCase):
    def test_stable_id_and_idempotent_jsonl_append(self) -> None:
        self.assertEqual(stable_id("event", {"b": 2, "a": 1}), stable_id("event", {"a": 1, "b": 2}))
        with tempfile.TemporaryDirectory() as temp:
            store = AppendOnlyJsonlStore(Path(temp) / "events.jsonl", "event")
            event_id = store.append({"hello": "世界"})
            self.assertEqual(store.append({"hello": "世界"}), event_id)
            self.assertEqual(len(store), 1)
            self.assertEqual(store.get(event_id), {"hello": "世界"})

    def test_explicit_duplicate_id_rejects_conflicting_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = AppendOnlyJsonlStore(Path(temp) / "events.jsonl", "event")
            store.append({"value": 1}, event_id="fixed")
            with self.assertRaises(DuplicateEventError):
                store.append({"value": 2}, event_id="fixed")
            with self.assertRaises(ValueError):
                store.append({"value": 3}, event_id=123)  # type: ignore[arg-type]

    def test_read_verifies_content_hash_and_append_recovers_torn_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            store = AppendOnlyJsonlStore(path, "event")
            store.append({"value": 1})
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["value"] = 999
            path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash"):
                list(store)

            path.unlink()
            store = AppendOnlyJsonlStore(path, "event")
            store.append({"value": "世界"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"torn"')
            store.append({"value": 2})
            self.assertEqual([item["value"] for item in store.payloads()], ["世界", 2])

    def test_probe_split_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvidenceStore(temp)
            for split in (None, "test", "typo"):
                with self.assertRaises(ValueError):
                    store.append_probe({"probe_id": "p", "task_split": split})

    def test_snapshot_hash_chain_replays_and_detects_tampering(self) -> None:
        first = GraphSnapshotEvent.create(0, {"nodes": []})
        second = GraphSnapshotEvent.create(1, {"nodes": [{"id": "a"}]}, first.snapshot_id)
        self.assertEqual(replay_snapshots([first, second]), {"nodes": [{"id": "a"}]})
        tampered = replace(second, graph={"nodes": [{"id": "evil"}]})
        with self.assertRaises(SnapshotReplayError):
            replay_snapshots([first, tampered])


class SkillTests(unittest.TestCase):
    @staticmethod
    def evidence_lookup(evidence_id: str):
        if len(evidence_id) < 2 or not evidence_id[1:].isdigit():
            return None
        index = int(evidence_id[1:])
        return {
            "problem_id": f"v{index}",
            "task_split": "validation",
            "policy_version": "policy-e1",
            "evaluator_version": "eval-v1",
            "feature_schema_version": "features-v1",
            "paired_effect": 0.12,
        }

    def gate(self) -> SkillEvidenceGate:
        return SkillEvidenceGate(
            SkillGateConfig(
                delta_min=0.03,
                max_harm_probability=0.05,
                minimum_independent_problems=4,
                minimum_effective_pairs=4,
                minimum_empirical_coverage=0.90,
                minimum_positive_slice_fraction=0.5,
            ),
            evidence_lookup=self.evidence_lookup,
        )

    def test_skill_gate_and_next_epoch_activation(self) -> None:
        skill = candidate_skill()
        manager = SkillLifecycleManager(self.gate())
        with self.assertRaises(ValueError):
            manager.activate(skill, 1)
        active = manager.activate(skill, 2)
        self.assertEqual(active.status, SkillStatus.ACTIVE)
        self.assertEqual(active.activated_epoch, 2)

    def test_discovery_validation_overlap_and_harm_fail(self) -> None:
        with self.assertRaises(ValueError):
            candidate_skill(discovery_problem_ids=["v1"])
        harmful = candidate_skill(harm_probability=0.4)
        self.assertFalse(self.gate().evaluate(harmful).approved)

    def test_nonfinite_or_fabricated_evidence_cannot_publish(self) -> None:
        with self.assertRaises(ValueError):
            candidate_skill(calibrated_lower=math.nan)
        missing = candidate_skill(evidence_ids=["missing"])
        decision = self.gate().evaluate(missing)
        self.assertFalse(decision.approved)
        self.assertTrue(any("unresolved" in reason for reason in decision.reasons))

    def test_version_drift_suspends_active_skill(self) -> None:
        manager = SkillLifecycleManager(self.gate())
        active = manager.activate(candidate_skill(), 2)
        suspended = manager.audit(active, versions(policy="policy-e2"))
        self.assertEqual(suspended.status, SkillStatus.SUSPENDED)
        self.assertIn("policy", suspended.suspended_reason or "")

    def test_store_roundtrip_and_conservative_retrieval(self) -> None:
        manager = SkillLifecycleManager(self.gate())
        candidate = candidate_skill()
        active = manager.activate(candidate, 2)
        with tempfile.TemporaryDirectory() as temp:
            store = SkillStore(Path(temp) / "skills.json")
            store.upsert(candidate)
            store.upsert(active)
            loaded = store.get(active.skill_id)
            self.assertEqual(loaded, active)
            retrieved = SkillRetriever(top_k=2).retrieve(
                store.list(),
                SkillQuery(
                    task_family="math",
                    graph_stage="before_final",
                    tags=["long"],
                    available_models=["qwen35"],
                    current_epoch=2,
                ),
                versions(),
            )
            self.assertEqual([item.skill_id for item in retrieved], [active.skill_id])
            mismatched = SkillRetriever().retrieve(
                store.list(),
                SkillQuery("math", "before_final", tags=["long"], current_epoch=2),
                versions(policy="policy-e2"),
            )
            self.assertEqual(mismatched, [])
            self.assertEqual(len(store.history(active.skill_id)), 2)

    def test_forged_active_receipt_is_not_retrieved_and_retirement_keeps_reason(self) -> None:
        forged = replace(
            candidate_skill(),
            status=SkillStatus.ACTIVE,
            activated_epoch=2,
            gate_config=self.gate().config.to_dict(),
            gate_receipt="forged",
        )
        found = SkillRetriever().retrieve(
            [forged],
            SkillQuery("math", "before_final", tags=["long"], available_models=["qwen35"], current_epoch=2),
            versions(),
        )
        self.assertEqual(found, [])
        active = SkillLifecycleManager(self.gate()).activate(candidate_skill(), 2)
        retired = SkillLifecycleManager(self.gate()).retire(active, "superseded")
        self.assertEqual(retired.status, SkillStatus.RETIRED)
        self.assertEqual(retired.suspended_reason, "superseded")


if __name__ == "__main__":
    unittest.main()
