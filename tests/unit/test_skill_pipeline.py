from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.interactive.persistence import EvidenceStore
from src.interactive.records import (
    EvaluationReceipt,
    ProbeRecord,
    TaskRecord,
    TrajectoryRecord,
)
from src.interactive.skills import (
    SkillEvidencePipeline,
    SkillGateConfig,
    SkillProbeEvidence,
    SkillQuery,
    SkillStatus,
    SkillStore,
    SkillValidationStatistics,
    StructuredSkillCandidate,
    render_validated_skill,
)
from src.interactive.versioning import VersionBundle


CONDITION = {
    "task_family": "hotpotqa",
    "graph_stage": "after_bridge",
    "graph_prefix": {"unresolved_bridge_dependency": True},
    "role_family": "reasoning",
    "graph_position": "downstream",
}
ACTION = {
    "model_id": "executor-a",
    "relation": {
        "source_role": "evidence",
        "target_role": "reasoning",
        "kind": "source_to_target",
    },
    "instruction": "Consume the upstream bridge artifact before answering.",
}
BASELINE_ACTION = {
    "model_id": "executor-a",
    "instruction": "Answer without the upstream bridge artifact.",
}
EXECUTOR_VERSIONS = {"executor-a": "executor-a@2026-08-16"}


def versions() -> VersionBundle:
    return VersionBundle(
        policy="qwen35-director-step-0",
        model_catalog="hotpot-catalog-v6",
        evaluator="hotpot-evaluator-v1",
        prompt="director-prompt-v6",
        tool="tool-runtime-v1",
        feature_schema="agentgraph-prefix-v1",
        skill_library="skill-library-v1",
    )


def trajectory() -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id="trajectory-discovery-1",
        task=TaskRecord(
            task_id="discovery-problem-1",
            question="Which city is associated with the bridge entity?",
            ground_truth="Example City",
            split="train",
            metadata={"task_family": "hotpotqa"},
        ),
        group_id="discovery-problem-1:exploit:qwen35-director-step-0",
        condition_id="exploit",
        rollout_id="rollout-1",
        versions=versions(),
        turns=(),
        final_answer="Example City",
        evaluation=EvaluationReceipt(
            evaluator_version="hotpot-evaluator-v1",
            valid=True,
            reward=1.0,
            metrics={"exact_match": 1.0, "f1": 1.0},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )


def proposal() -> StructuredSkillCandidate:
    return StructuredSkillCandidate(
        skill_id="hotpot-use-bridge-artifact-v1",
        condition=CONDITION,
        action=ACTION,
        baseline_id="same-prefix-without-bridge-consumption",
        baseline_action=BASELINE_ACTION,
        failure_scope=("no_upstream_artifact",),
    )


def probe(
    probe_id: str,
    problem_id: str,
    split: str,
    *,
    candidate_return: float,
    incumbent_return: float,
    runtime_version: str = "agent-runtime-v6",
) -> SkillProbeEvidence:
    record = ProbeRecord(
        probe_id=probe_id,
        problem_id=problem_id,
        task_split=split,
        snapshot_id=f"snapshot-{problem_id}",
        policy_version="qwen35-director-step-0",
        state_features={
            "graph_prefix": {"revision": 3, "unresolved_bridge_dependency": True}
        },
        incumbent_action=BASELINE_ACTION,
        candidate_action=ACTION,
        sampling_probability=0.5,
        incumbent_returns=(incumbent_return,),
        candidate_returns=(candidate_return,),
        executor_versions=EXECUTOR_VERSIONS,
        evaluator_version="hotpot-evaluator-v1",
        feature_schema_version="agentgraph-prefix-v1",
        branch_order=("incumbent", "candidate"),
    )
    return SkillProbeEvidence(
        probe=record,
        condition=CONDITION,
        runtime_version=runtime_version,
        model_catalog_version="hotpot-catalog-v6",
    )


def statistics(
    *,
    lower: float = 0.20,
    upper: float = 0.70,
) -> SkillValidationStatistics:
    return SkillValidationStatistics(
        calibrated_lower=lower,
        calibrated_upper=upper,
        empirical_coverage=0.95,
        harm_probability=0.01,
        heldout_task_families=("hotpotqa",),
        slice_effects={"bridge": 0.40, "comparison": 0.30},
    )


class SkillPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.pipeline = SkillEvidencePipeline(
            evidence_store=EvidenceStore(root / "evidence"),
            skill_store=SkillStore(root / "skills.json"),
            gate_config=SkillGateConfig(
                delta_min=0.10,
                max_harm_probability=0.05,
                minimum_independent_problems=2,
                minimum_effective_pairs=2,
                minimum_empirical_coverage=0.90,
                minimum_positive_slice_fraction=0.50,
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def discover(self):
        return self.pipeline.discover(
            trajectory(),
            proposal(),
            created_epoch=0,
            runtime_version="agent-runtime-v6",
            executor_versions=EXECUTOR_VERSIONS,
        )

    def test_one_trajectory_is_candidate_only_and_cannot_activate(self) -> None:
        candidate = self.discover()

        self.assertEqual(candidate.status, SkillStatus.CANDIDATE)
        self.assertEqual(candidate.evidence.effective_pairs, 0)
        decision = self.pipeline.gate.evaluate(candidate)
        self.assertFalse(decision.approved)
        self.assertIn("too few independent validation problems", decision.reasons)
        with self.assertRaisesRegex(ValueError, "evidence gate failed"):
            self.pipeline.lifecycle.activate(candidate, current_epoch=1)
        with self.assertRaisesRegex(ValueError, "ACTIVE"):
            render_validated_skill(candidate)

    def test_forced_probe_receipt_is_structurally_excluded_from_grpo(self) -> None:
        evidence = probe(
            "probe-train-1",
            "discovery-problem-2",
            "train",
            candidate_return=0.8,
            incumbent_return=0.3,
        )

        payload = evidence.to_probe_payload()
        self.assertTrue(payload["forced_probe"])
        self.assertFalse(payload["grpo_eligible"])
        self.assertEqual(payload["condition"], CONDITION)
        self.assertEqual(payload["runtime_version"], "agent-runtime-v6")

    def test_end_to_end_active_retrieval_is_rejectable_prompt_prior(self) -> None:
        candidate = self.discover()
        result = self.pipeline.confirm_and_publish(
            candidate,
            discovery_probes=(
                probe(
                    "probe-train-1",
                    "discovery-problem-2",
                    "train",
                    candidate_return=0.8,
                    incumbent_return=0.3,
                ),
            ),
            validation_probes=(
                probe(
                    "probe-validation-1",
                    "validation-problem-1",
                    "validation",
                    candidate_return=0.9,
                    incumbent_return=0.3,
                ),
                probe(
                    "probe-validation-2",
                    "validation-problem-2",
                    "validation",
                    candidate_return=0.7,
                    incumbent_return=0.4,
                ),
            ),
            statistics=statistics(),
            validation_epoch=1,
            activation_epoch=2,
        )

        self.assertTrue(result.gate_decision.approved)
        self.assertTrue(result.active)
        self.assertEqual(result.skill.status, SkillStatus.ACTIVE)
        self.assertEqual(result.skill.evidence.effective_pairs, 2)
        self.assertEqual(len(result.skill.evidence.independent_problem_ids), 2)
        self.assertEqual(result.skill.provenance["runtime_version"], "agent-runtime-v6")
        self.assertEqual(
            result.skill.provenance["executor_versions"],
            EXECUTOR_VERSIONS,
        )
        self.assertEqual(
            self.pipeline.active_skill_ids(versions()),
            ("hotpot-use-bridge-artifact-v1",),
        )
        self.assertEqual(
            self.pipeline.active_skill_ids(
                replace(versions(), policy="different-policy")
            ),
            (),
        )

        priors = self.pipeline.retrieve_prompt_priors(
            SkillQuery(
                task_family="hotpotqa",
                graph_stage="after_bridge",
                available_models=("executor-a",),
                current_epoch=2,
            ),
            versions(),
        )
        self.assertEqual(len(priors), 1)
        prior = priors[0]
        self.assertTrue(prior.rejectable)
        self.assertEqual(prior.application_mode, "rejectable_prompt_prior")
        self.assertIn("may accept, modify, or reject", prior.content)
        self.assertIn("Applicability: task_family=hotpotqa", prior.content)
        self.assertIn("Instruction:", prior.content)
        serialized = prior.to_dict()
        self.assertNotIn("canvas_patch", serialized)
        self.assertNotIn("canvas_action", serialized)
        self.assertNotIn("condition", serialized)
        self.assertNotIn("action", serialized)

        history = self.pipeline.skill_store.history(candidate.skill_id)
        self.assertEqual(
            [item.status for item in history],
            [SkillStatus.CANDIDATE, SkillStatus.CANDIDATE, SkillStatus.ACTIVE],
        )

        frozen_pipeline = SkillEvidencePipeline(
            evidence_store=self.pipeline.evidence_store,
            skill_store=self.pipeline.skill_store,
            retrieval_snapshot=(result.skill,),
        )
        retired = self.pipeline.lifecycle.retire(result.skill, "next epoch update")
        self.pipeline.skill_store.upsert(retired)
        query = SkillQuery(
            task_family="hotpotqa",
            graph_stage="after_bridge",
            available_models=("executor-a",),
            current_epoch=2,
        )
        self.assertEqual(self.pipeline.retrieve_prompt_priors(query, versions()), ())
        self.assertEqual(
            len(frozen_pipeline.retrieve_prompt_priors(query, versions())),
            1,
        )

    def test_discovery_and_validation_problem_overlap_is_rejected(self) -> None:
        candidate = self.discover()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.pipeline.confirm_and_publish(
                candidate,
                discovery_probes=(
                    probe(
                        "probe-train-1",
                        "same-problem",
                        "train",
                        candidate_return=0.8,
                        incumbent_return=0.3,
                    ),
                ),
                validation_probes=(
                    probe(
                        "probe-validation-1",
                        "same-problem",
                        "validation",
                        candidate_return=0.9,
                        incumbent_return=0.3,
                    ),
                    probe(
                        "probe-validation-2",
                        "validation-problem-2",
                        "validation",
                        candidate_return=0.7,
                        incumbent_return=0.4,
                    ),
                ),
                statistics=statistics(),
                validation_epoch=1,
                activation_epoch=2,
            )
        self.assertEqual(len(self.pipeline.evidence_store.probes), 0)

    def test_duplicate_validation_problem_cannot_inflate_effective_count(self) -> None:
        candidate = self.discover()
        with self.assertRaisesRegex(ValueError, "complete problem only once"):
            self.pipeline.confirm_and_publish(
                candidate,
                discovery_probes=(
                    probe(
                        "probe-train-1",
                        "discovery-problem-2",
                        "train",
                        candidate_return=0.8,
                        incumbent_return=0.3,
                    ),
                ),
                validation_probes=(
                    probe(
                        "probe-validation-1",
                        "validation-problem-1",
                        "validation",
                        candidate_return=0.9,
                        incumbent_return=0.3,
                    ),
                    probe(
                        "probe-validation-2",
                        "validation-problem-1",
                        "validation",
                        candidate_return=0.7,
                        incumbent_return=0.4,
                    ),
                ),
                statistics=statistics(),
                validation_epoch=1,
                activation_epoch=2,
            )

    def test_version_or_action_mismatch_fails_closed_before_persistence(self) -> None:
        candidate = self.discover()
        mismatched_runtime = probe(
            "probe-validation-1",
            "validation-problem-1",
            "validation",
            candidate_return=0.9,
            incumbent_return=0.3,
            runtime_version="different-runtime",
        )
        with self.assertRaisesRegex(ValueError, "runtime version"):
            self.pipeline.confirm_and_publish(
                candidate,
                discovery_probes=(
                    probe(
                        "probe-train-1",
                        "discovery-problem-2",
                        "train",
                        candidate_return=0.8,
                        incumbent_return=0.3,
                    ),
                ),
                validation_probes=(
                    mismatched_runtime,
                    probe(
                        "probe-validation-2",
                        "validation-problem-2",
                        "validation",
                        candidate_return=0.7,
                        incumbent_return=0.4,
                    ),
                ),
                statistics=statistics(),
                validation_epoch=1,
                activation_epoch=2,
            )
        self.assertEqual(len(self.pipeline.evidence_store.probes), 0)

        wrong_action = replace(
            probe(
                "probe-validation-action",
                "validation-problem-action",
                "validation",
                candidate_return=0.9,
                incumbent_return=0.3,
            ).probe,
            candidate_action={"model_id": "executor-a", "instruction": "Different"},
        )
        mismatch = SkillProbeEvidence(
            probe=wrong_action,
            condition=CONDITION,
            runtime_version="agent-runtime-v6",
            model_catalog_version="hotpot-catalog-v6",
        )
        with self.assertRaisesRegex(ValueError, "candidate action"):
            self.pipeline._validate_probe(
                candidate,
                mismatch,
                expected_split="validation",
            )

    def test_low_upper_bound_retires_instead_of_activating(self) -> None:
        candidate = self.discover()
        result = self.pipeline.confirm_and_publish(
            candidate,
            discovery_probes=(
                probe(
                    "probe-train-1",
                    "discovery-problem-2",
                    "train",
                    candidate_return=0.51,
                    incumbent_return=0.50,
                ),
            ),
            validation_probes=(
                probe(
                    "probe-validation-1",
                    "validation-problem-1",
                    "validation",
                    candidate_return=0.51,
                    incumbent_return=0.50,
                ),
                probe(
                    "probe-validation-2",
                    "validation-problem-2",
                    "validation",
                    candidate_return=0.52,
                    incumbent_return=0.50,
                ),
            ),
            statistics=statistics(lower=-0.02, upper=0.05),
            validation_epoch=1,
            activation_epoch=2,
        )

        self.assertFalse(result.active)
        self.assertTrue(result.gate_decision.no_practical_value)
        self.assertEqual(result.skill.status, SkillStatus.RETIRED)
        self.assertEqual(
            self.pipeline.retrieve_prompt_priors(
                SkillQuery(
                    "hotpotqa",
                    "after_bridge",
                    available_models=("executor-a",),
                    current_epoch=2,
                ),
                versions(),
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
