"""Unit tests for the NumPy-only exploration plane."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.interactive.exploration import (
    BayesianLinearPosterior,
    DisjointLinUCB,
    FEATURE_DIMENSION,
    MACE,
    MACEFeatureConfig,
    MACEFeatureExtractor,
    NORMALIZATION_L2_BOUND,
    NORMALIZATION_NONE,
    PairedProbeRecord,
    ProbeOrder,
    RoundSnapshot,
    ThompsonSamplingPolicy,
    UCBPolicy,
    blended_mace_reward,
    clamp_tiny_negative,
    estimate_particle_evpi,
    estimate_particle_evsi,
    make_common_random_numbers,
    particle_evpi,
    particle_evsi,
    particle_evsi_many,
    preprocess_text,
    randomize_probe_order,
    record_paired_probe,
)


class RoundSnapshotTests(unittest.TestCase):
    def test_snapshot_defensively_freezes_responses_and_advances_synchronously(self) -> None:
        responses = ["initial a", "initial b"]
        initial = RoundSnapshot.initial(responses, total_rounds=2)
        responses[0] = "mutated"

        self.assertEqual(initial.responses, ("initial a", "initial b"))
        self.assertEqual(initial.selection_round, 1)
        self.assertEqual(initial.normalized_selection_round, 0.5)

        advanced = initial.advance(["round one a", "round one b"])
        self.assertEqual(initial.completed_round, 0)
        self.assertEqual(advanced.completed_round, 1)
        self.assertEqual(advanced.selection_round, 2)
        completed = advanced.advance(["round two a", "round two b"])
        with self.assertRaises(RuntimeError):
            _ = completed.selection_round
        with self.assertRaises(RuntimeError):
            completed.advance(["x", "y"])

    def test_snapshot_rejects_agent_count_change(self) -> None:
        snapshot = RoundSnapshot.initial(["a", "b"], total_rounds=1)
        with self.assertRaises(ValueError):
            snapshot.advance(["only one"])


class MACEFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = MACEFeatureExtractor(
            MACEFeatureConfig(normalization_version=NORMALIZATION_NONE)
        )

    def test_preprocessing_is_versioned_nfkc_word_casefold(self) -> None:
        self.assertEqual(
            preprocess_text("ＣＡＦÉ, can't 1,000!"),
            ("café", "can't", "1,000"),
        )
        with self.assertRaises(ValueError):
            preprocess_text("text", version="unknown")

    def test_exact_nine_dimensional_relational_features(self) -> None:
        snapshot = RoundSnapshot.initial(
            ["red blue", "red green", "red blue"], total_rounds=2
        )
        history = np.zeros((3, 3), dtype=np.float64)
        history[0, 1] = 0.7
        features = self.extractor.build_raw(snapshot, history)

        self.assertEqual(features.shape, (3, 3, FEATURE_DIMENSION))
        expected = np.array(
            [2.0 / 3.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.7, 0.5, 1.0]
        )
        np.testing.assert_allclose(features[0, 1], expected)
        np.testing.assert_allclose(features[0, 0, :3], 0.0)
        self.assertFalse(features.flags.writeable)

    def test_historical_reward_is_ordered_pair_specific(self) -> None:
        snapshot = RoundSnapshot.initial(["a b c", "d e f"], total_rounds=1)
        history = {(0, 1): 0.75, (1, 0): -0.25}
        features = self.extractor.build(snapshot, history)
        self.assertEqual(features[0, 1, 6], 0.75)
        self.assertEqual(features[1, 0, 6], -0.25)

    def test_l2_bound_is_explicit_reproducible_and_read_only(self) -> None:
        config = MACEFeatureConfig(normalization_version=NORMALIZATION_L2_BOUND)
        extractor = MACEFeatureExtractor(config)
        snapshot = RoundSnapshot.initial(["a b", "c d"], total_rounds=3)
        first = extractor.build(snapshot)
        second = extractor.build(snapshot)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.linalg.norm(first, axis=2) <= 1.0 + 1e-12))
        self.assertIn(NORMALIZATION_L2_BOUND, config.version_id)
        with self.assertRaises(ValueError):
            first[0, 0, 0] = 1.0

    def test_completed_snapshot_cannot_create_next_round_features(self) -> None:
        completed = RoundSnapshot(("a", "b"), completed_round=1, total_rounds=1)
        with self.assertRaises(RuntimeError):
            self.extractor.build(completed)


class DisjointLinUCBTests(unittest.TestCase):
    def test_seeded_tie_break_is_reproducible_and_not_index_locked(self) -> None:
        contexts = np.zeros((3, 1), dtype=np.float64)
        first = DisjointLinUCB(3, dimension=1, seed=17)
        second = DisjointLinUCB(3, dimension=1, seed=17)
        choices_a = [first.select(0, contexts).peer for _ in range(20)]
        choices_b = [second.select(0, contexts).peer for _ in range(20)]
        self.assertEqual(choices_a, choices_b)
        self.assertGreater(len(set(choices_a)), 1)
        np.testing.assert_array_equal(first.counts, np.zeros((3, 3)))

    def test_updates_only_selected_ordered_pair(self) -> None:
        policy = DisjointLinUCB(2, dimension=2, seed=0)
        context = np.array([1.0, 2.0])
        changed = policy.update(0, 1, context, reward=0.5)
        self.assertTrue(changed)
        self.assertEqual(policy.counts[0, 1], 1)
        self.assertEqual(np.sum(policy.counts), 1)
        np.testing.assert_allclose(
            policy.precision[0, 1], np.eye(2) + np.outer(context, context)
        )
        np.testing.assert_allclose(policy.information[0, 1], 0.5 * context)
        self.assertAlmostEqual(policy.reward_means[0, 1], 0.5)

    def test_uncertainty_bonus_selects_under_observed_peer_without_inverse(self) -> None:
        policy = DisjointLinUCB(2, dimension=1, exploration_alpha=1.0, seed=0)
        for _ in range(8):
            policy.update(0, 0, [1.0], reward=0.0)
        contexts = np.ones((2, 1), dtype=np.float64)
        with mock.patch("numpy.linalg.inv", side_effect=AssertionError("inverse used")):
            selection = policy.select(0, contexts)
        self.assertEqual(selection.peer, 1)
        self.assertGreater(selection.epistemic_variance, 1.0 / 9.0)

    def test_update_and_exploration_controls_are_independent(self) -> None:
        policy = DisjointLinUCB(
            2, dimension=1, update_enabled=False, exploration_alpha=1.0, seed=1
        )
        before = policy.dumps()
        self.assertFalse(policy.update(0, 1, [1.0], reward=1.0))
        # RNG state is part of serialization, so compare numeric state directly.
        np.testing.assert_array_equal(policy.counts, np.zeros((2, 2)))
        policy.set_exploration_alpha(0.0)
        self.assertEqual(policy.exploration_alpha, 0.0)
        self.assertFalse(policy.update_enabled)
        self.assertNotEqual(before, policy.dumps())

    def test_candidate_mask_and_self_exclusion(self) -> None:
        policy = DisjointLinUCB(3, dimension=1, allow_self=False, seed=2)
        contexts = np.ones((3, 1))
        selection = policy.select(1, contexts, candidate_mask=[False, True, True])
        self.assertEqual(selection.peer, 2)
        with self.assertRaises(ValueError):
            policy.select(1, contexts, candidate_mask=[False, True, False])

    def test_json_round_trip_preserves_state_and_next_rng_choice(self) -> None:
        policy = DisjointLinUCB(3, dimension=2, seed=31, allow_self=False)
        policy.update(0, 1, [1.0, -1.0], reward=0.25)
        _ = policy.select(2, np.zeros((3, 2)))
        payload = policy.dumps()
        json.loads(payload)
        restored = DisjointLinUCB.loads(payload)
        np.testing.assert_array_equal(restored.precision, policy.precision)
        np.testing.assert_array_equal(restored.information, policy.information)
        np.testing.assert_array_equal(restored.counts, policy.counts)
        contexts = np.zeros((3, 2))
        self.assertEqual(restored.select(2, contexts).peer, policy.select(2, contexts).peer)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linucb.json"
            policy.save(path)
            loaded = DisjointLinUCB.load(path)
            np.testing.assert_array_equal(loaded.reward_means, policy.reward_means)

    def test_mace_orchestrates_synchronous_select_then_score_updates(self) -> None:
        bandit = DisjointLinUCB(2, seed=4)
        mace = MACE(
            bandit,
            MACEFeatureExtractor(
                MACEFeatureConfig(normalization_version=NORMALIZATION_NONE)
            ),
        )
        snapshot = RoundSnapshot.initial(["alpha beta", "beta gamma"], 2)
        selections = mace.select_round(snapshot)
        self.assertEqual(len(selections), 2)
        np.testing.assert_array_equal(bandit.counts, np.zeros((2, 2)))
        rewards = mace.update_from_scores(selections, [0.0, 1.0], [1.0, 1.0])
        np.testing.assert_allclose(rewards, [1.0, 0.5])
        self.assertEqual(np.sum(bandit.counts), 2)
        restored = MACE.loads(mace.dumps())
        np.testing.assert_array_equal(restored.bandit.counts, bandit.counts)
        self.assertEqual(
            restored.feature_extractor.config.version_id,
            mace.feature_extractor.config.version_id,
        )


class BayesianPosteriorTests(unittest.TestCase):
    def test_online_update_and_variance_decomposition(self) -> None:
        posterior = BayesianLinearPosterior(1, observation_variance=1.0)
        posterior.update([1.0], 2.0)
        np.testing.assert_allclose(posterior.precision, [[2.0]])
        np.testing.assert_allclose(posterior.information, [2.0])
        np.testing.assert_allclose(posterior.mean, [1.0])
        self.assertAlmostEqual(posterior.epistemic_variance([1.0]), 0.5)
        self.assertAlmostEqual(posterior.predictive_variance([1.0]), 1.5)

    def test_paired_difference_uses_context_difference_and_pair_variance(self) -> None:
        posterior = BayesianLinearPosterior(2, observation_variance=0.5)
        posterior.update_paired_outcomes([1.0, 0.0], 4.0, [0.0, 1.0], 1.0)
        difference_context = np.array([1.0, -1.0])
        np.testing.assert_allclose(
            posterior.precision,
            np.eye(2) + np.outer(difference_context, difference_context),
        )
        np.testing.assert_allclose(posterior.information, 3.0 * difference_context)
        self.assertEqual(posterior.update_count, 1)

    def test_snapshots_are_immutable_and_restore_exactly(self) -> None:
        posterior = BayesianLinearPosterior(2, prior_mean=[0.5, -0.5])
        posterior.update([1.0, 2.0], 0.75)
        snapshot = posterior.snapshot()
        frozen_precision = snapshot.precision.copy()
        with self.assertRaises(ValueError):
            snapshot.precision[0, 0] = 99.0
        posterior.update([2.0, 1.0], -1.0)
        np.testing.assert_array_equal(snapshot.precision, frozen_precision)
        posterior.restore(snapshot)
        np.testing.assert_allclose(posterior.precision, snapshot.precision)
        np.testing.assert_allclose(posterior.information, snapshot.information)

        restored = BayesianLinearPosterior.from_state_dict(posterior.state_dict())
        np.testing.assert_allclose(restored.mean, posterior.mean)
        self.assertEqual(restored.update_count, posterior.update_count)

    def test_sampling_uses_cholesky_solve_and_seeded_generator(self) -> None:
        posterior = BayesianLinearPosterior(2, prior_precision=np.diag([2.0, 4.0]))
        first = posterior.sample_parameters(np.random.default_rng(7), size=4)
        second = posterior.sample_parameters(np.random.default_rng(7), size=4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (4, 2))
        with mock.patch("numpy.linalg.inv", side_effect=AssertionError("inverse used")):
            _ = posterior.mean
            _ = posterior.covariance
            _ = posterior.sample_parameters(np.random.default_rng(1))


class PosteriorPolicyTests(unittest.TestCase):
    def test_ucb_uses_epistemic_variance_and_respects_mask(self) -> None:
        posterior = BayesianLinearPosterior(2, observation_variance=100.0)
        contexts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        policy = UCBPolicy(exploration_alpha=1.0, seed=0)
        with mock.patch.object(
            posterior,
            "predictive_variance",
            side_effect=AssertionError("predictive variance used"),
        ):
            decision = policy.select(posterior, contexts, [True, True, False])
        self.assertEqual(decision.action, 1)
        self.assertAlmostEqual(decision.bonuses[0], 0.0)
        self.assertAlmostEqual(decision.bonuses[1], 2.0)

    def test_thompson_draw_is_reused_for_entire_rollout(self) -> None:
        posterior = BayesianLinearPosterior(2)
        policy = ThompsonSamplingPolicy(seed=11)
        rollout = policy.start_rollout(posterior)
        contexts = np.eye(2)
        first = rollout.select(contexts, posterior_mean=posterior.mean)
        posterior.update([1.0, 0.0], 100.0)
        second = rollout.select(contexts, posterior_mean=np.zeros(2))
        np.testing.assert_array_equal(first.scores, second.scores)
        np.testing.assert_array_equal(first.scores, rollout.parameter_draw)
        new_rollout = policy.start_rollout(posterior)
        self.assertFalse(np.array_equal(new_rollout.parameter_draw, rollout.parameter_draw))


class PairedProbeTests(unittest.TestCase):
    def test_order_randomization_is_seeded_and_balanced(self) -> None:
        first_rng = np.random.default_rng(19)
        second_rng = np.random.default_rng(19)
        first = [randomize_probe_order("left", "right", first_rng) for _ in range(1000)]
        second = [randomize_probe_order("left", "right", second_rng) for _ in range(1000)]
        self.assertEqual(first, second)
        flipped = sum(order.flipped for order in first)
        self.assertGreater(flipped, 400)
        self.assertLess(flipped, 600)

    def test_record_difference_is_invariant_to_presentation_order(self) -> None:
        direct_order = ProbeOrder("a", "b", "a", "b")
        flipped_order = ProbeOrder("a", "b", "b", "a")
        direct = record_paired_probe("p1", direct_order, [0.9, 0.2])
        flipped = PairedProbeRecord("p2", flipped_order, 0.2, 0.9)
        self.assertAlmostEqual(direct.difference, 0.7)
        self.assertAlmostEqual(flipped.difference, 0.7)
        self.assertEqual(flipped.canonical_outcomes, (0.9, 0.2))
        self.assertEqual(flipped_order.arrange("A payload", "B payload"), ("B payload", "A payload"))


class ParticleValueOfInformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.utilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.draws = make_common_random_numbers(12000, seed=23)

    def test_evpi_known_two_particle_problem(self) -> None:
        estimate = estimate_particle_evpi(self.utilities)
        self.assertAlmostEqual(estimate.current_value, 0.5)
        self.assertAlmostEqual(estimate.perfect_information_value, 1.0)
        self.assertAlmostEqual(estimate.value, 0.5)
        self.assertAlmostEqual(particle_evpi(self.utilities), 0.5)

    def test_uninformative_probe_has_zero_evsi(self) -> None:
        estimate = estimate_particle_evsi(
            self.utilities,
            probe_signal=[0.0, 0.0],
            observation_std=1.0,
            common_random_numbers=self.draws,
            negative_tolerance=1e-10,
        )
        self.assertAlmostEqual(estimate.value, 0.0, places=12)
        self.assertEqual(estimate.n_samples, self.draws.n_samples)

    def test_informative_probe_is_positive_bounded_by_evpi(self) -> None:
        value = particle_evsi(
            self.utilities,
            probe_signal=[-1.0, 1.0],
            observation_std=0.05,
            common_random_numbers=self.draws,
        )
        self.assertGreater(value, 0.45)
        self.assertLessEqual(value, particle_evpi(self.utilities) + 0.01)

    def test_common_random_numbers_make_repeated_and_batch_estimates_identical(self) -> None:
        signal = np.array([-0.5, 0.5])
        first = particle_evsi(
            self.utilities, signal, 0.4, common_random_numbers=self.draws
        )
        second = particle_evsi(
            self.utilities, signal, 0.4, common_random_numbers=self.draws
        )
        self.assertEqual(first, second)
        batch = particle_evsi_many(
            self.utilities,
            [signal, signal],
            0.4,
            common_random_numbers=self.draws,
        )
        np.testing.assert_array_equal(batch, [first, first])
        self.assertFalse(batch.flags.writeable)

    def test_only_tiny_negative_values_are_clamped(self) -> None:
        self.assertEqual(clamp_tiny_negative(-1e-14, tolerance=1e-12), 0.0)
        self.assertEqual(clamp_tiny_negative(-1e-3, tolerance=1e-12), -1e-3)


class RewardTests(unittest.TestCase):
    def test_blended_reward_credits_improvement_and_maintained_quality(self) -> None:
        self.assertAlmostEqual(blended_mace_reward(0.0, 1.0), 1.0)
        self.assertAlmostEqual(blended_mace_reward(1.0, 1.0), 0.5)
        self.assertAlmostEqual(blended_mace_reward(1.0, 0.0), -0.5)


if __name__ == "__main__":
    unittest.main()
