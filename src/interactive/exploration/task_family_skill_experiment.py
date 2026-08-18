"""Task-family parameterization of the existing Skill experiment helpers.

The implementation is a thin adaptation of :mod:`skill_experiment`: it keeps
the same candidate-main-effect plus candidate-by-task-family feature map,
Bayesian linear posterior, UCB/EVSI scheduling, and problem-cluster bootstrap
calibration.  Task families and experiment version strings are supplied by the
caller so the same evidence path can be used outside the joint-QA experiment.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..persistence import stable_id
from ..records import PosteriorSnapshotRecord
from ..skills import SkillValidationStatistics
from .evsi import make_common_random_numbers, particle_evsi_many
from .policies import UCBPolicy
from .posterior import BayesianLinearPosterior
from .skill_experiment import EVSIProbeDecision, ScheduledCandidate


def _name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _unique_names(values: Sequence[str], *, field: str, minimum: int) -> tuple[str, ...]:
    normalized = tuple(_name(value, field=field) for value in values)
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        qualifier = "one" if minimum == 1 else str(minimum)
        raise ValueError(
            f"{field} values must contain at least {qualifier} unique value"
            + ("" if minimum == 1 else "s")
        )
    return normalized


class TaskFamilySkillFeatureMap:
    """Candidate main effects plus candidate-by-task-family interactions."""

    def __init__(
        self,
        task_families: Sequence[str],
        candidate_ids: Sequence[str],
        *,
        encoder_version: str,
        feature_schema_version: str,
    ) -> None:
        self.task_families = _unique_names(
            task_families,
            field="task_family",
            minimum=1,
        )
        self.candidate_ids = _unique_names(
            candidate_ids,
            field="candidate_id",
            minimum=2,
        )
        self.encoder_version = _name(encoder_version, field="encoder_version")
        self.feature_schema_version = _name(
            feature_schema_version,
            field="feature_schema_version",
        )
        self._candidate_index = {
            value: index for index, value in enumerate(self.candidate_ids)
        }
        self._task_family_index = {
            value: index for index, value in enumerate(self.task_families)
        }

    @property
    def dimension(self) -> int:
        return len(self.candidate_ids) * (1 + len(self.task_families))

    def baseline(self, task_family: str) -> np.ndarray:
        self._require_task_family(task_family)
        return np.zeros(self.dimension, dtype=np.float64)

    def context(self, task_family: str, candidate_id: str) -> np.ndarray:
        family_index = self._require_task_family(task_family)
        try:
            candidate_index = self._candidate_index[candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown candidate_id: {candidate_id}") from exc
        candidate_count = len(self.candidate_ids)
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[candidate_index] = 1.0
        interaction_offset = candidate_count + family_index * candidate_count
        vector[interaction_offset + candidate_index] = 1.0
        return vector

    def to_state_features(
        self,
        task_family: str,
        candidate_id: str,
    ) -> dict[str, object]:
        return {
            "encoder_version": self.encoder_version,
            "feature_schema_version": self.feature_schema_version,
            "task_family": task_family,
            "candidate_id": candidate_id,
            "prefix_stage": "empty_canvas",
            "candidate_context": self.context(task_family, candidate_id).tolist(),
            "baseline_context": self.baseline(task_family).tolist(),
            "effect_scope": "full_trajectory_total_effect_from_empty_canvas",
        }

    def _require_task_family(self, task_family: str) -> int:
        try:
            return self._task_family_index[task_family]
        except KeyError as exc:
            raise ValueError(f"unsupported task_family: {task_family}") from exc


class TaskFamilyPosteriorScheduler:
    """Balanced task-family cold start followed by posterior UCB selection."""

    def __init__(
        self,
        task_families: Sequence[str],
        candidate_ids: Sequence[str],
        *,
        encoder_version: str,
        feature_schema_version: str,
        posterior_version: str,
        seed: int,
        exploration_alpha: float = 1.0,
        prior_precision: float = 1.0,
        observation_variance: float = 0.25,
    ) -> None:
        self.features = TaskFamilySkillFeatureMap(
            task_families,
            candidate_ids,
            encoder_version=encoder_version,
            feature_schema_version=feature_schema_version,
        )
        self.posterior_version = _name(
            posterior_version,
            field="posterior_version",
        )
        self.posterior = BayesianLinearPosterior(
            self.features.dimension,
            prior_precision=prior_precision,
            observation_variance=observation_variance,
        )
        self.policy = UCBPolicy(exploration_alpha, seed=seed)
        self._counts = {
            (task_family, candidate_id): 0
            for task_family in self.features.task_families
            for candidate_id in self.features.candidate_ids
        }
        self.observation_ids: list[str] = []

    @property
    def task_families(self) -> tuple[str, ...]:
        return self.features.task_families

    def select(self, task_family: str) -> ScheduledCandidate:
        contexts = np.stack(
            [
                self.features.context(task_family, candidate_id)
                for candidate_id in self.features.candidate_ids
            ]
        )
        unseen = [
            candidate_id
            for candidate_id in self.features.candidate_ids
            if self._counts[(task_family, candidate_id)] == 0
        ]
        if unseen:
            return ScheduledCandidate(unseen[0], None, "balanced_cold_start")
        decision = self.policy.select(self.posterior, contexts)
        return ScheduledCandidate(
            self.features.candidate_ids[decision.action],
            decision,
            "posterior_ucb",
        )

    def update(
        self,
        task_family: str,
        candidate_id: str,
        paired_effect: float,
        *,
        observation_id: str,
    ) -> None:
        if not math.isfinite(float(paired_effect)):
            raise ValueError("paired effect must be finite")
        observation_id = _name(observation_id, field="observation_id")
        if observation_id in self.observation_ids:
            raise ValueError("posterior observation IDs must be unique")
        candidate = self.features.context(task_family, candidate_id)
        baseline = self.features.baseline(task_family)
        self.posterior.update_paired_difference(
            candidate,
            baseline,
            float(paired_effect),
            difference_variance=self.posterior.observation_variance,
        )
        self._counts[(task_family, candidate_id)] += 1
        self.observation_ids.append(observation_id)

    def update_paired_outcomes(
        self,
        task_family: str,
        candidate_id: str,
        *,
        candidate_outcome: float,
        baseline_outcome: float,
        observation_id: str,
    ) -> None:
        candidate_value = float(candidate_outcome)
        baseline_value = float(baseline_outcome)
        if not math.isfinite(candidate_value) or not math.isfinite(baseline_value):
            raise ValueError("paired outcomes must be finite")
        self.update(
            task_family,
            candidate_id,
            candidate_value - baseline_value,
            observation_id=observation_id,
        )

    def rank_probes_by_evsi(
        self,
        task_family: str,
        *,
        candidate_ids: Sequence[str] | None = None,
        seed: int,
        posterior_particles: int = 1024,
        observation_samples: int = 2048,
    ) -> EVSIProbeDecision:
        candidates = (
            self.features.candidate_ids
            if candidate_ids is None
            else tuple(
                _name(value, field="candidate_id") for value in candidate_ids
            )
        )
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("candidate_ids must be unique and non-empty")
        unknown = [
            value for value in candidates if value not in self.features.candidate_ids
        ]
        if unknown:
            raise ValueError("unknown EVSI candidate_id: " + ", ".join(unknown))
        if type(posterior_particles) is not int or posterior_particles < 2:
            raise ValueError("posterior_particles must be an integer >= 2")
        if type(observation_samples) is not int or observation_samples < 2:
            raise ValueError("observation_samples must be an integer >= 2")

        all_contexts = np.stack(
            [
                self.features.context(task_family, candidate_id)
                - self.features.baseline(task_family)
                for candidate_id in self.features.candidate_ids
            ]
        )
        probe_contexts = np.stack(
            [
                self.features.context(task_family, candidate_id)
                - self.features.baseline(task_family)
                for candidate_id in candidates
            ]
        )
        rng = np.random.default_rng(seed)
        parameter_particles = self.posterior.sample_parameters(
            rng,
            size=posterior_particles,
        )
        utilities = parameter_particles @ all_contexts.T
        probe_signals = probe_contexts @ parameter_particles.T
        common_random_numbers = make_common_random_numbers(
            observation_samples,
            rng=rng,
        )
        observation_std = math.sqrt(self.posterior.observation_variance)
        evsi_values = particle_evsi_many(
            utilities,
            probe_signals,
            observation_std,
            common_random_numbers=common_random_numbers,
        )
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(evsi_values[index]), index),
        )
        ranked_ids = tuple(candidates[index] for index in order)
        ranked_values = tuple(float(evsi_values[index]) for index in order)
        return EVSIProbeDecision(
            candidate_ids=ranked_ids,
            values=ranked_values,
            selected_id=ranked_ids[0],
            posterior_particles=posterior_particles,
            observation_samples=observation_samples,
            observation_std=observation_std,
        )

    def predict(self, task_family: str, candidate_id: str) -> tuple[float, float]:
        difference = self.features.context(
            task_family,
            candidate_id,
        ) - self.features.baseline(task_family)
        return (
            self.posterior.predict_mean(difference),
            math.sqrt(self.posterior.epistemic_variance(difference)),
        )

    def exploit(self, task_family: str) -> str:
        values = [
            self.predict(task_family, candidate_id)[0]
            for candidate_id in self.features.candidate_ids
        ]
        return self.features.candidate_ids[int(np.argmax(values))]

    def posterior_record(
        self,
        *,
        epoch: int,
        policy_version: str,
        task_slices: Sequence[str] | None = None,
        calibration_quantile: float = 0.0,
        coverage_metrics: Mapping[str, float] | None = None,
    ) -> PosteriorSnapshotRecord:
        slices = (
            self.features.task_families
            if task_slices is None
            else _unique_names(task_slices, field="task_slice", minimum=1)
        )
        unknown = [value for value in slices if value not in self.features.task_families]
        if unknown:
            raise ValueError("unknown task_slice: " + ", ".join(unknown))
        snapshot = self.posterior.snapshot()
        payload = {
            "epoch": epoch,
            "policy_version": policy_version,
            "posterior_version": self.posterior_version,
            "encoder_version": self.features.encoder_version,
            "feature_schema_version": self.features.feature_schema_version,
            "task_families": list(self.features.task_families),
            "mean": snapshot.mean.tolist(),
            "precision": snapshot.precision.tolist(),
            "observation_ids": list(self.observation_ids),
        }
        return PosteriorSnapshotRecord(
            posterior_id=stable_id("posterior", payload),
            epoch=epoch,
            policy_version=_name(policy_version, field="policy_version"),
            encoder_version=self.features.encoder_version,
            feature_schema_version=self.features.feature_schema_version,
            mean_parameters=snapshot.mean.tolist(),
            precision_parameters=snapshot.precision.tolist(),
            observation_noise=float(snapshot.observation_variance),
            calibration_quantile=float(calibration_quantile),
            coverage_metrics=dict(coverage_metrics or {}),
            valid_task_slices=slices,
            observation_ids=tuple(self.observation_ids),
        )


def calibrate_task_family_skill_validation(
    effects_by_problem: Mapping[str, float],
    *,
    predicted_mean: float,
    predicted_std: float,
    task_family: str,
    seed: int,
    bootstrap_draws: int = 20_000,
    alpha: float = 0.05,
    harm_threshold: float = 0.0,
) -> tuple[SkillValidationStatistics, dict[str, object]]:
    """Calibrate one Skill using complete problems as bootstrap clusters."""

    family = _name(task_family, field="task_family")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if type(bootstrap_draws) is not int or bootstrap_draws < 1_000:
        raise ValueError("bootstrap_draws must be an integer >= 1000")
    if not math.isfinite(float(predicted_mean)):
        raise ValueError("predicted_mean must be finite")
    if not math.isfinite(float(predicted_std)) or predicted_std <= 0:
        raise ValueError("predicted_std must be finite and positive")
    problem_ids = tuple(effects_by_problem)
    if len(problem_ids) < 2 or len(problem_ids) != len(set(problem_ids)):
        raise ValueError("at least two unique problem clusters are required")
    effects = np.asarray([float(effects_by_problem[key]) for key in problem_ids])
    if not np.all(np.isfinite(effects)):
        raise ValueError("paired effects must be finite")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        effects.size,
        size=(bootstrap_draws, effects.size),
    )
    bootstrap_means = effects[sample_indices].mean(axis=1)
    effect_mean = float(effects.mean())
    lower, upper = np.quantile(
        bootstrap_means,
        (alpha / 2.0, 1.0 - alpha / 2.0),
    )
    lower = min(float(lower), effect_mean)
    upper = max(float(upper), effect_mean)
    harm_probability = float(np.mean(bootstrap_means < -float(harm_threshold)))

    residuals = np.abs(effects - float(predicted_mean)) / (
        float(predicted_std) + 1e-12
    )
    conformal_level = min(
        1.0,
        math.ceil((effects.size + 1) * (1 - alpha)) / effects.size,
    )
    conformal_quantile = float(
        np.quantile(residuals, conformal_level, method="higher")
    )
    conformal_lower = float(predicted_mean) - conformal_quantile * float(
        predicted_std
    )
    conformal_upper = float(predicted_mean) + conformal_quantile * float(
        predicted_std
    )
    coverage = float(
        np.mean((effects >= conformal_lower) & (effects <= conformal_upper))
    )

    statistics = SkillValidationStatistics(
        calibrated_lower=lower,
        calibrated_upper=upper,
        empirical_coverage=coverage,
        harm_probability=harm_probability,
        heldout_task_families=(family,),
        slice_effects={family: effect_mean},
    )
    metadata: dict[str, object] = {
        "problem_cluster_count": effects.size,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "alpha": alpha,
        "aggregate_interval_method": "problem_cluster_percentile_bootstrap",
        "harm_probability_method": "problem_cluster_nonparametric_bootstrap",
        "harm_threshold": harm_threshold,
        "effect_mean": effect_mean,
        "predicted_mean_before_confirmation": float(predicted_mean),
        "predicted_epistemic_std_before_confirmation": float(predicted_std),
        "conformal_method": "split_conformal_standardized_residual",
        "conformal_quantile": conformal_quantile,
        "conformal_interval": [conformal_lower, conformal_upper],
        "calibration_sample_empirical_coverage": coverage,
    }
    return statistics, metadata


__all__ = [
    "EVSIProbeDecision",
    "ScheduledCandidate",
    "TaskFamilyPosteriorScheduler",
    "TaskFamilySkillFeatureMap",
    "calibrate_task_family_skill_validation",
]
