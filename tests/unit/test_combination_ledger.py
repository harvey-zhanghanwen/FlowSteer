"""Phase-A acceptance tests for the dynamic combination posterior."""

from __future__ import annotations

import unittest

import numpy as np

from src.interactive.exploration.combination_ledger import (
    CombinationPosterior,
    DecisionKey,
    SurfaceSignal,
)


def _key(
    *,
    family: str = "qa",
    role: str = "solve",
    model: str = "A",
    edge: str = "independent",
    same: bool = False,
    stage: str = "other",
) -> DecisionKey:
    return DecisionKey(family, role, model, edge, same, stage)


class CombinationPosteriorPhaseATests(unittest.TestCase):
    def test_first_order_intervals_cover_synthetic_truth_after_200_probes(self) -> None:
        rng = np.random.default_rng(917)
        ledger = CombinationPosterior("policy-0", n_act=10_000)
        truth = {
            "role=solve": -0.05,
            "role=verify": 0.05,
            "model=A": -0.15,
            "model=B": 0.15,
            "rel=independent|different": 0.125,
            "rel=independent|same": -0.125,
        }

        for _ in range(200):
            changed = str(rng.choice(["role", "model", "rel"]))
            values = {
                "role": str(rng.choice(["solve", "verify"])),
                "model": str(rng.choice(["A", "B"])),
                "same": bool(rng.integers(0, 2)),
            }
            alternative = dict(values)
            if changed == "role":
                alternative["role"] = (
                    "verify" if values["role"] == "solve" else "solve"
                )
            elif changed == "model":
                alternative["model"] = "B" if values["model"] == "A" else "A"
            else:
                alternative["same"] = not values["same"]
            keep = _key(
                role=values["role"], model=values["model"], same=values["same"]
            )
            switch = _key(
                role=alternative["role"],
                model=alternative["model"],
                same=alternative["same"],
            )

            def success_probability(key: DecisionKey) -> float:
                columns = (
                    f"role={key.role_cluster}",
                    f"model={key.model_id}",
                    f"rel={key.relation_level}",
                )
                return 0.5 + sum(truth[column] for column in columns)

            ledger.update_probe(
                keep,
                switch,
                rng.binomial(1, success_probability(keep), 3),
                rng.binomial(1, success_probability(switch), 3),
            )

        covariance = ledger.covariance
        covered = 0
        for column, index in ledger.columns.items():
            radius = 1.6448536269514722 * np.sqrt(covariance[index, index])
            covered += int(
                ledger.mean[index] - radius
                <= truth[column]
                <= ledger.mean[index] + radius
            )
        self.assertGreaterEqual(covered / len(truth), 0.85)

    def test_second_order_activates_on_fifth_touch_and_covers_truth_after_50(self) -> None:
        rng = np.random.default_rng(11)
        ledger = CombinationPosterior("policy-0")
        target = "role=solve&rel=bidirectional|same"

        def probability(key: DecisionKey) -> float:
            is_target = key.role_cluster == "solve" and (
                key.relation_level == "bidirectional|same"
            )
            return 0.5 + (0.1 if is_target else 0.0)

        for probe_index in range(50):
            if probe_index < 5:
                keep = _key(family="math", edge="independent", same=True)
                switch = _key(family="math", edge="bidirectional", same=True)
            else:
                role = str(rng.choice(["solve", "verify"]))
                model = str(rng.choice(["A", "B"]))
                edge = str(rng.choice(["independent", "bidirectional"]))
                same = bool(rng.integers(0, 2))
                keep = _key(
                    family="math", role=role, model=model, edge=edge, same=same
                )
                changed = str(rng.choice(["role", "model", "rel"]))
                switch = _key(
                    family="math",
                    role=("verify" if role == "solve" else "solve")
                    if changed == "role"
                    else role,
                    model=("B" if model == "A" else "A")
                    if changed == "model"
                    else model,
                    edge=("bidirectional" if edge == "independent" else "independent")
                    if changed == "rel"
                    else edge,
                    same=same,
                )
            ledger.update_probe(
                keep,
                switch,
                rng.binomial(1, probability(keep), 3),
                rng.binomial(1, probability(switch), 3),
            )
            if probe_index == 3:
                self.assertNotIn(target, ledger.columns)
            if probe_index == 4:
                self.assertIn(target, ledger.columns)

        index = ledger.columns[target]
        radius = 1.6448536269514722 * np.sqrt(ledger.covariance[index, index])
        self.assertLessEqual(ledger.mean[index] - radius, 0.1)
        self.assertGreaterEqual(ledger.mean[index] + radius, 0.1)

    def test_beta_sensors_recover_same_and_independent_false_positive_rates(self) -> None:
        ledger = CombinationPosterior("policy-0")
        same = "verifier_pass|same"
        independent = "verifier_pass|different"

        # For each class: 100 successes followed by 100 failures.  Counts are
        # deterministic so this test isolates the Beta accounting itself.
        for sensor_class, success_true, failure_true in (
            (same, 80, 60),
            (independent, 80, 10),
        ):
            for index in range(100):
                ledger.update_sensor(
                    SurfaceSignal("verifier_pass", index < success_true, sensor_class),
                    1,
                )
            for index in range(100):
                ledger.update_sensor(
                    SurfaceSignal("verifier_pass", index < failure_true, sensor_class),
                    0,
                )

        same_sensitivity, same_fpr = ledger.sensor_rates(same)
        independent_sensitivity, independent_fpr = ledger.sensor_rates(independent)
        self.assertLess(abs(same_sensitivity - 0.8), 0.05)
        self.assertLess(abs(same_fpr - 0.6), 0.05)
        self.assertLess(abs(independent_sensitivity - 0.8), 0.05)
        self.assertLess(abs(independent_fpr - 0.1), 0.05)

    def test_paired_fit_removes_natural_trajectory_selection_confounding(self) -> None:
        rng = np.random.default_rng(77)
        n_natural = 5_000
        easy = rng.random(n_natural) < 0.5
        # A biased Director chooses B mostly on easy questions.
        chooses_b = np.where(
            easy, rng.random(n_natural) < 0.9, rng.random(n_natural) < 0.1
        )
        baseline = np.where(easy, 0.5, 0.1)
        outcomes = rng.binomial(1, baseline + 0.25 * chooses_b)
        naive_effect = outcomes[chooses_b].mean() - outcomes[~chooses_b].mean()
        self.assertGreater(abs(naive_effect - 0.25), 0.1)

        ledger = CombinationPosterior("policy-0", n_act=10_000)
        keep = _key(model="A")
        switch = _key(model="B")
        for _ in range(200):
            paired_baseline = 0.5 if bool(rng.integers(0, 2)) else 0.1
            ledger.update_probe(
                keep,
                switch,
                rng.binomial(1, paired_baseline, 10),
                rng.binomial(1, paired_baseline + 0.25, 10),
            )
        paired_effect = ledger.predict_contrast(switch, keep).mean
        self.assertLess(abs(paired_effect - 0.25), 0.03)

    def test_baseline_joint_sampling_empirical_bayes_and_refresh(self) -> None:
        ledger = CombinationPosterior("policy-0", n_act=10_000)
        keep = _key(model="A")
        switch = _key(model="B")
        for _ in range(10):
            ledger.update_probe(keep, switch, [0, 0, 1], [1, 1, 1])
        ledger.update_trajectory(
            "qa",
            [switch],
            [SurfaceSignal("verifier_pass", True, "verifier_pass|different")],
            1,
        )
        self.assertEqual(ledger.task_baseline["qa"][1], 5)
        self.assertTrue(0.02 <= ledger.expected_terminal("qa", [switch]) <= 0.98)
        self.assertEqual(
            ledger.sample_parameters(np.random.default_rng(2), size=7).shape,
            (7, ledger.dimension),
        )
        variances = ledger.empirical_bayes_update()
        self.assertTrue(all(value >= 1e-3 for value in variances.values()))
        before = ledger.Lambda
        used_decay = ledger.refresh_policy("policy-1", heldout_coverage=0.9)
        self.assertEqual(used_decay, 0.1)
        self.assertEqual(ledger.epoch, 1)
        self.assertFalse(np.array_equal(before, ledger.Lambda))

    def test_json_round_trip_preserves_statistics_and_next_seeded_draw(self) -> None:
        ledger = CombinationPosterior("policy-0")
        keep = _key(model="A")
        switch = _key(model="B")
        for _ in range(6):
            ledger.update_probe(keep, switch, [0, 1, 0], [1, 1, 0])
        ledger.update_trajectory(
            "qa",
            [keep],
            [SurfaceSignal("none", False, "none|different")],
            0,
        )
        restored = CombinationPosterior.loads(ledger.dumps())
        self.assertEqual(restored.columns, ledger.columns)
        self.assertEqual(restored.probe_count, ledger.probe_count)
        self.assertEqual(restored.task_baseline, ledger.task_baseline)
        np.testing.assert_allclose(restored.Lambda, ledger.Lambda)
        np.testing.assert_allclose(restored.eta, ledger.eta)
        np.testing.assert_array_equal(
            restored.sample_parameters(np.random.default_rng(19), size=4),
            ledger.sample_parameters(np.random.default_rng(19), size=4),
        )


if __name__ == "__main__":
    unittest.main()
