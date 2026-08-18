"""Targeted tests for the joint-QA MACE/Bayesian/Skill experiment helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.interactive.exploration.skill_experiment import (
    DATASETS,
    ENCODER_VERSION,
    FEATURE_SCHEMA_VERSION,
    JointQAPosteriorScheduler,
    JointQASkillFeatureMap,
    calibrate_skill_validation,
)


class JointQASkillFeatureMapTests(unittest.TestCase):
    def test_contexts_have_expected_shape_and_distinct_dataset_interactions(self) -> None:
        features = JointQASkillFeatureMap(("grounding", "verification"))

        hotpot_grounding = features.context("hotpotqa", "grounding")
        trivia_grounding = features.context("triviaqa", "grounding")
        hotpot_verification = features.context("hotpotqa", "verification")

        self.assertEqual(features.dimension, 6)
        self.assertEqual(hotpot_grounding.shape, (features.dimension,))
        np.testing.assert_array_equal(
            hotpot_grounding,
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            trivia_grounding,
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.assertFalse(np.array_equal(hotpot_grounding, trivia_grounding))
        self.assertFalse(np.array_equal(hotpot_grounding, hotpot_verification))

    def test_baseline_is_zero_and_state_features_preserve_versioned_contexts(self) -> None:
        features = JointQASkillFeatureMap(("grounding", "verification"))

        baseline = features.baseline("hotpotqa")
        state = features.to_state_features("hotpotqa", "verification")

        np.testing.assert_array_equal(baseline, np.zeros(features.dimension))
        self.assertEqual(state["encoder_version"], ENCODER_VERSION)
        self.assertEqual(state["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(state["prefix_stage"], "empty_canvas")
        self.assertEqual(
            state["effect_scope"],
            "full_trajectory_total_effect_from_empty_canvas",
        )
        np.testing.assert_array_equal(
            state["candidate_context"],
            features.context("hotpotqa", "verification"),
        )
        np.testing.assert_array_equal(state["baseline_context"], baseline)


class JointQAPosteriorSchedulerTests(unittest.TestCase):
    def test_balanced_cold_start_is_dataset_specific_then_switches_to_ucb(self) -> None:
        scheduler = JointQAPosteriorScheduler(
            ("grounding", "verification"),
            seed=7,
            exploration_alpha=0.5,
        )

        first_hotpot = scheduler.select("hotpotqa")
        self.assertEqual(first_hotpot.candidate_id, "grounding")
        self.assertEqual(first_hotpot.selection_mode, "balanced_cold_start")
        self.assertIsNone(first_hotpot.decision)
        scheduler.update(
            "hotpotqa",
            "grounding",
            0.8,
            observation_id="hotpot-grounding",
        )

        second_hotpot = scheduler.select("hotpotqa")
        self.assertEqual(second_hotpot.candidate_id, "verification")
        self.assertEqual(second_hotpot.selection_mode, "balanced_cold_start")
        scheduler.update(
            "hotpotqa",
            "verification",
            -0.8,
            observation_id="hotpot-verification",
        )

        # HotpotQA is ready for posterior UCB, while TriviaQA retains its own
        # balanced cold-start schedule despite the shared posterior.
        hotpot_ucb = scheduler.select("hotpotqa")
        trivia_cold_start = scheduler.select("triviaqa")
        self.assertEqual(hotpot_ucb.selection_mode, "posterior_ucb")
        self.assertIsNotNone(hotpot_ucb.decision)
        self.assertEqual(hotpot_ucb.candidate_id, "grounding")
        self.assertEqual(trivia_cold_start.candidate_id, "grounding")
        self.assertEqual(trivia_cold_start.selection_mode, "balanced_cold_start")

        scheduler.update(
            "triviaqa",
            "grounding",
            0.4,
            observation_id="trivia-grounding",
        )
        self.assertEqual(scheduler.select("triviaqa").candidate_id, "verification")
        scheduler.update(
            "triviaqa",
            "verification",
            -0.4,
            observation_id="trivia-verification",
        )
        self.assertEqual(scheduler.select("triviaqa").selection_mode, "posterior_ucb")

    def test_update_changes_prediction_and_posterior_record_tracks_evidence(self) -> None:
        scheduler = JointQAPosteriorScheduler(
            ("grounding", "verification"),
            seed=3,
        )
        prior_mean, prior_std = scheduler.predict("hotpotqa", "grounding")

        scheduler.update(
            "hotpotqa",
            "grounding",
            0.75,
            observation_id="pair-001",
        )
        posterior_mean, posterior_std = scheduler.predict("hotpotqa", "grounding")
        record = scheduler.posterior_record(
            epoch=2,
            policy_version="theta-jointqa-step-2",
            calibration_quantile=1.25,
            coverage_metrics={"hotpotqa": 0.95},
        )

        self.assertEqual(prior_mean, 0.0)
        self.assertGreater(posterior_mean, prior_mean)
        self.assertLess(posterior_std, prior_std)
        self.assertEqual(record.epoch, 2)
        self.assertEqual(record.policy_version, "theta-jointqa-step-2")
        self.assertEqual(record.encoder_version, ENCODER_VERSION)
        self.assertEqual(record.feature_schema_version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(record.observation_ids, ("pair-001",))
        self.assertEqual(tuple(record.valid_task_slices), DATASETS)
        self.assertEqual(record.coverage_metrics, {"hotpotqa": 0.95})
        self.assertEqual(len(record.mean_parameters), scheduler.features.dimension)
        self.assertEqual(
            np.asarray(record.precision_parameters).shape,
            (scheduler.features.dimension, scheduler.features.dimension),
        )

    def test_particle_evsi_probe_ranking_is_deterministic(self) -> None:
        scheduler = JointQAPosteriorScheduler(
            ("grounding", "verification"),
            seed=11,
            observation_variance=0.25,
        )
        scheduler.update(
            "hotpotqa",
            "grounding",
            0.5,
            observation_id="pair-grounding",
        )
        scheduler.update(
            "hotpotqa",
            "verification",
            -0.1,
            observation_id="pair-verification",
        )

        first = scheduler.rank_probes_by_evsi(
            "hotpotqa",
            seed=29,
            posterior_particles=128,
            observation_samples=256,
        )
        second = scheduler.rank_probes_by_evsi(
            "hotpotqa",
            seed=29,
            posterior_particles=128,
            observation_samples=256,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first.candidate_ids), {"grounding", "verification"})
        self.assertEqual(first.selected_id, first.candidate_ids[0])
        self.assertEqual(len(first.values), 2)
        self.assertTrue(all(np.isfinite(value) for value in first.values))
        self.assertGreaterEqual(first.values[0], first.values[1])
        self.assertEqual(first.posterior_particles, 128)
        self.assertEqual(first.observation_samples, 256)
        self.assertEqual(first.observation_std, 0.5)


class SkillValidationCalibrationTests(unittest.TestCase):
    def test_calibration_is_deterministic_for_fixed_problem_clusters_and_seed(self) -> None:
        effects = {f"problem-{index:02d}": 0.04 + 0.01 * index for index in range(8)}
        arguments = {
            "predicted_mean": 0.08,
            "predicted_std": 0.1,
            "task_family": "hotpotqa",
            "seed": 19,
            "bootstrap_draws": 2_000,
        }

        first_statistics, first_metadata = calibrate_skill_validation(
            effects,
            **arguments,
        )
        second_statistics, second_metadata = calibrate_skill_validation(
            effects,
            **arguments,
        )

        self.assertEqual(first_statistics, second_statistics)
        self.assertEqual(first_metadata, second_metadata)
        self.assertLessEqual(
            first_statistics.calibrated_lower,
            first_metadata["effect_mean"],
        )
        self.assertGreaterEqual(
            first_statistics.calibrated_upper,
            first_metadata["effect_mean"],
        )
        self.assertEqual(first_statistics.heldout_task_families, ("hotpotqa",))
        self.assertEqual(first_metadata["problem_cluster_count"], len(effects))
        self.assertEqual(
            first_metadata["aggregate_interval_method"],
            "problem_cluster_percentile_bootstrap",
        )

    def test_negative_effects_produce_certain_bootstrap_harm(self) -> None:
        effects = {
            "problem-a": -0.30,
            "problem-b": -0.20,
            "problem-c": -0.10,
            "problem-d": -0.05,
        }

        statistics, metadata = calibrate_skill_validation(
            effects,
            predicted_mean=-0.1,
            predicted_std=0.2,
            task_family="triviaqa",
            seed=23,
            bootstrap_draws=1_000,
        )

        self.assertEqual(statistics.harm_probability, 1.0)
        self.assertLess(statistics.calibrated_upper, 0.0)
        self.assertLess(statistics.slice_effects["triviaqa"], 0.0)
        self.assertEqual(
            metadata["harm_probability_method"],
            "problem_cluster_nonparametric_bootstrap",
        )

    def test_calibration_rejects_fewer_than_two_problem_clusters(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "at least two unique problem clusters are required",
        ):
            calibrate_skill_validation(
                {"only-problem": 0.1},
                predicted_mean=0.1,
                predicted_std=0.2,
                task_family="hotpotqa",
                seed=5,
                bootstrap_draws=1_000,
            )


if __name__ == "__main__":
    unittest.main()
