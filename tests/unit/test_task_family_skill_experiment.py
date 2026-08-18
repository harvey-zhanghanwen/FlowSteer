"""Tests for the task-family parameterization of Skill experiments."""

from __future__ import annotations

import unittest

import numpy as np

from src.interactive.exploration.task_family_skill_experiment import (
    TaskFamilyPosteriorScheduler,
    TaskFamilySkillFeatureMap,
    calibrate_task_family_skill_validation,
)


TASK_FAMILIES = ("aime2026", "healthbench_professional")
CANDIDATES = ("candidate-a", "candidate-b")
VERSIONS = {
    "encoder_version": "aime-health.skill-condition.dev.v1",
    "feature_schema_version": "aime-health.skill-effect.dev.v1",
    "posterior_version": "aime-health.posterior.dev.v1",
}


class TaskFamilySkillFeatureMapTests(unittest.TestCase):
    def test_task_families_and_versions_are_caller_supplied(self) -> None:
        features = TaskFamilySkillFeatureMap(
            TASK_FAMILIES,
            CANDIDATES,
            encoder_version=VERSIONS["encoder_version"],
            feature_schema_version=VERSIONS["feature_schema_version"],
        )

        aime = features.context("aime2026", "candidate-a")
        health = features.context("healthbench_professional", "candidate-a")
        state = features.to_state_features("healthbench_professional", "candidate-b")

        self.assertEqual(features.dimension, 6)
        np.testing.assert_array_equal(
            aime,
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            health,
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.assertEqual(state["task_family"], "healthbench_professional")
        self.assertEqual(state["encoder_version"], VERSIONS["encoder_version"])
        self.assertEqual(
            state["feature_schema_version"],
            VERSIONS["feature_schema_version"],
        )

    def test_unknown_or_duplicated_task_families_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_family"):
            TaskFamilySkillFeatureMap(
                ("aime2026", " aime2026 "),
                CANDIDATES,
                encoder_version=VERSIONS["encoder_version"],
                feature_schema_version=VERSIONS["feature_schema_version"],
            )

        features = TaskFamilySkillFeatureMap(
            TASK_FAMILIES,
            CANDIDATES,
            encoder_version=VERSIONS["encoder_version"],
            feature_schema_version=VERSIONS["feature_schema_version"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported task_family"):
            features.context("hotpotqa", "candidate-a")


class TaskFamilyPosteriorSchedulerTests(unittest.TestCase):
    def scheduler(self) -> TaskFamilyPosteriorScheduler:
        return TaskFamilyPosteriorScheduler(
            TASK_FAMILIES,
            CANDIDATES,
            **VERSIONS,
            seed=17,
            exploration_alpha=0.5,
            observation_variance=0.25,
        )

    def test_cold_start_is_task_family_specific_and_accepts_scalar_effects(self) -> None:
        scheduler = self.scheduler()

        self.assertEqual(scheduler.select("aime2026").candidate_id, "candidate-a")
        scheduler.update_paired_outcomes(
            "aime2026",
            "candidate-a",
            candidate_outcome=1.0,
            baseline_outcome=0.0,
            observation_id="aime-accuracy-pair",
        )
        self.assertEqual(scheduler.select("aime2026").candidate_id, "candidate-b")
        self.assertEqual(
            scheduler.select("healthbench_professional").candidate_id,
            "candidate-a",
        )

        scheduler.update(
            "healthbench_professional",
            "candidate-a",
            -2.75,
            observation_id="health-raw-score-pair",
        )
        self.assertLess(
            scheduler.predict("healthbench_professional", "candidate-a")[0],
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            scheduler.update(
                "healthbench_professional",
                "candidate-b",
                float("nan"),
                observation_id="invalid-pair",
            )

    def test_ucb_evsi_and_posterior_record_reuse_existing_primitives(self) -> None:
        scheduler = self.scheduler()
        for task_family in TASK_FAMILIES:
            scheduler.update(
                task_family,
                "candidate-a",
                0.4,
                observation_id=f"{task_family}-a",
            )
            scheduler.update(
                task_family,
                "candidate-b",
                -0.2,
                observation_id=f"{task_family}-b",
            )

        self.assertEqual(scheduler.select("aime2026").selection_mode, "posterior_ucb")
        first = scheduler.rank_probes_by_evsi(
            "healthbench_professional",
            seed=31,
            posterior_particles=64,
            observation_samples=128,
        )
        second = scheduler.rank_probes_by_evsi(
            "healthbench_professional",
            seed=31,
            posterior_particles=64,
            observation_samples=128,
        )
        record = scheduler.posterior_record(
            epoch=1,
            policy_version="qwen35-9b-aime-health-dev-v1",
            coverage_metrics={"healthbench_professional": 0.95},
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first.candidate_ids), set(CANDIDATES))
        self.assertEqual(record.encoder_version, VERSIONS["encoder_version"])
        self.assertEqual(
            record.feature_schema_version,
            VERSIONS["feature_schema_version"],
        )
        self.assertEqual(tuple(record.valid_task_slices), TASK_FAMILIES)
        self.assertEqual(scheduler.posterior_version, VERSIONS["posterior_version"])


class TaskFamilyCalibrationTests(unittest.TestCase):
    def test_health_raw_score_effects_use_problem_cluster_bootstrap(self) -> None:
        effects = {
            "health-problem-1": 0.8,
            "health-problem-2": 1.2,
            "health-problem-3": 0.4,
            "health-problem-4": 1.6,
        }

        statistics, metadata = calibrate_task_family_skill_validation(
            effects,
            predicted_mean=1.0,
            predicted_std=0.5,
            task_family="healthbench_professional",
            seed=23,
            bootstrap_draws=1_000,
        )

        self.assertEqual(
            statistics.heldout_task_families,
            ("healthbench_professional",),
        )
        self.assertAlmostEqual(
            statistics.slice_effects["healthbench_professional"],
            1.0,
        )
        self.assertEqual(metadata["problem_cluster_count"], 4)
        self.assertEqual(
            metadata["aggregate_interval_method"],
            "problem_cluster_percentile_bootstrap",
        )

    def test_aime_accuracy_differences_are_not_restricted_to_binary_values(self) -> None:
        statistics, metadata = calibrate_task_family_skill_validation(
            {
                "aime-problem-1": -1.0,
                "aime-problem-2": 0.0,
                "aime-problem-3": 1.0,
            },
            predicted_mean=0.0,
            predicted_std=0.5,
            task_family="aime2026",
            seed=29,
            bootstrap_draws=1_000,
        )

        self.assertEqual(statistics.heldout_task_families, ("aime2026",))
        self.assertAlmostEqual(metadata["effect_mean"], 0.0)
        self.assertLessEqual(statistics.calibrated_lower, 0.0)
        self.assertGreaterEqual(statistics.calibrated_upper, 0.0)


if __name__ == "__main__":
    unittest.main()
