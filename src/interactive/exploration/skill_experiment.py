"""Joint-QA posterior scheduling and problem-clustered Skill statistics.

This module is a small project adaptation of the existing exploration
primitives.  It does not introduce a second value model: MACE-style UCB reads
the mean and epistemic uncertainty of :class:`BayesianLinearPosterior`, as
specified by ``FlowSteer_MACE_Bayesian_Skill_Design.md`` sections 7--10.

The fixed feature map is deliberately interpretable for the current two-arm,
two-dataset smoke experiment.  It is not the future low-rank AgentGraph value
model described in the design document.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from ..persistence import stable_id
from ..records import PosteriorSnapshotRecord
from ..skills import SkillValidationStatistics
from .policies import UCBPolicy
from .posterior import BayesianLinearPosterior
from .records import PolicyDecision
from .evsi import make_common_random_numbers, particle_evsi_many


DATASETS = ("hotpotqa", "triviaqa")
ENCODER_VERSION = "jointqa.skill-condition.fixed.v1"
FEATURE_SCHEMA_VERSION = "jointqa.skill-candidate-dataset-interaction.v1"
POSTERIOR_VERSION = "jointqa.bayesian-linear.progressive-subgraph.v1"


def _name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


class JointQASkillFeatureMap:
    """Fixed candidate main effects plus candidate-by-dataset interactions."""

    def __init__(self, candidate_ids: Sequence[str]) -> None:
        candidates = tuple(_name(value, field="candidate_id") for value in candidate_ids)
        if len(candidates) < 2 or len(candidates) != len(set(candidates)):
            raise ValueError("candidate_ids must contain at least two unique values")
        self.candidate_ids = candidates
        self._candidate_index = {value: index for index, value in enumerate(candidates)}
        self._dataset_index = {value: index for index, value in enumerate(DATASETS)}

    @property
    def dimension(self) -> int:
        return len(self.candidate_ids) * (1 + len(DATASETS))

    def baseline(self, dataset: str) -> np.ndarray:
        self._require_dataset(dataset)
        return np.zeros(self.dimension, dtype=np.float64)

    def context(self, dataset: str, candidate_id: str) -> np.ndarray:
        dataset_index = self._require_dataset(dataset)
        try:
            candidate_index = self._candidate_index[candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown candidate_id: {candidate_id}") from exc
        candidate_count = len(self.candidate_ids)
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[candidate_index] = 1.0
        interaction_offset = candidate_count + dataset_index * candidate_count
        vector[interaction_offset + candidate_index] = 1.0
        return vector

    def to_state_features(self, dataset: str, candidate_id: str) -> dict[str, object]:
        return {
            "encoder_version": ENCODER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset": dataset,
            "candidate_id": candidate_id,
            "prefix_stage": "empty_canvas",
            "candidate_context": self.context(dataset, candidate_id).tolist(),
            "baseline_context": self.baseline(dataset).tolist(),
            "effect_scope": "full_trajectory_total_effect_from_empty_canvas",
        }

    def _require_dataset(self, dataset: str) -> int:
        try:
            return self._dataset_index[dataset]
        except KeyError as exc:
            raise ValueError(f"unsupported joint-QA dataset: {dataset}") from exc


@dataclass(frozen=True)
class ScheduledCandidate:
    candidate_id: str
    decision: PolicyDecision | None
    selection_mode: str


@dataclass(frozen=True)
class EVSIProbeDecision:
    """Particle-EVSI ranking over candidate paired interventions."""

    candidate_ids: tuple[str, ...]
    values: tuple[float, ...]
    selected_id: str
    posterior_particles: int
    observation_samples: int
    observation_std: float


class JointQAPosteriorScheduler:
    """Balanced cold start followed by MACE-style posterior UCB selection."""

    def __init__(
        self,
        candidate_ids: Sequence[str],
        *,
        seed: int,
        exploration_alpha: float = 1.0,
        prior_precision: float = 1.0,
        observation_variance: float = 0.25,
    ) -> None:
        self.features = JointQASkillFeatureMap(candidate_ids)
        self.posterior = BayesianLinearPosterior(
            self.features.dimension,
            prior_precision=prior_precision,
            observation_variance=observation_variance,
        )
        self.policy = UCBPolicy(exploration_alpha, seed=seed)
        self._counts = {
            (dataset, candidate_id): 0
            for dataset in DATASETS
            for candidate_id in self.features.candidate_ids
        }
        self.observation_ids: list[str] = []

    def select(self, dataset: str) -> ScheduledCandidate:
        contexts = np.stack(
            [
                self.features.context(dataset, candidate_id)
                for candidate_id in self.features.candidate_ids
            ]
        )
        unseen = [
            candidate_id
            for candidate_id in self.features.candidate_ids
            if self._counts[(dataset, candidate_id)] == 0
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
        dataset: str,
        candidate_id: str,
        paired_terminal_effect: float,
        *,
        observation_id: str,
    ) -> None:
        if not math.isfinite(float(paired_terminal_effect)):
            raise ValueError("paired terminal effect must be finite")
        observation_id = _name(observation_id, field="observation_id")
        if observation_id in self.observation_ids:
            raise ValueError("posterior observation IDs must be unique")
        candidate = self.features.context(dataset, candidate_id)
        baseline = self.features.baseline(dataset)
        # ``observation_variance`` is the predeclared variance of the paired
        # terminal effect, not a single-arm variance.  Use the posterior's
        # paired-difference API explicitly so the causal contrast remains
        # visible at the update boundary.
        self.posterior.update_paired_difference(
            candidate,
            baseline,
            float(paired_terminal_effect),
            difference_variance=self.posterior.observation_variance,
        )
        self._counts[(dataset, candidate_id)] += 1
        self.observation_ids.append(observation_id)

    def rank_probes_by_evsi(
        self,
        dataset: str,
        *,
        candidate_ids: Sequence[str] | None = None,
        seed: int,
        posterior_particles: int = 1024,
        observation_samples: int = 2048,
    ) -> EVSIProbeDecision:
        """Rank paired interventions by particle EVSI with common random numbers."""

        candidates = (
            self.features.candidate_ids
            if candidate_ids is None
            else tuple(_name(value, field="candidate_id") for value in candidate_ids)
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
                self.features.context(dataset, candidate_id)
                - self.features.baseline(dataset)
                for candidate_id in self.features.candidate_ids
            ]
        )
        probe_contexts = np.stack(
            [
                self.features.context(dataset, candidate_id)
                - self.features.baseline(dataset)
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

    def predict(self, dataset: str, candidate_id: str) -> tuple[float, float]:
        difference = self.features.context(dataset, candidate_id) - self.features.baseline(
            dataset
        )
        return (
            self.posterior.predict_mean(difference),
            math.sqrt(self.posterior.epistemic_variance(difference)),
        )

    def exploit(self, dataset: str) -> str:
        values = [self.predict(dataset, candidate_id)[0] for candidate_id in self.features.candidate_ids]
        return self.features.candidate_ids[int(np.argmax(values))]

    def posterior_record(
        self,
        *,
        epoch: int,
        policy_version: str,
        task_slices: Sequence[str] = DATASETS,
        calibration_quantile: float = 0.0,
        coverage_metrics: Mapping[str, float] | None = None,
    ) -> PosteriorSnapshotRecord:
        snapshot = self.posterior.snapshot()
        payload = {
            "epoch": epoch,
            "policy_version": policy_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "mean": snapshot.mean.tolist(),
            "precision": snapshot.precision.tolist(),
            "observation_ids": list(self.observation_ids),
        }
        return PosteriorSnapshotRecord(
            posterior_id=stable_id("posterior", payload),
            epoch=epoch,
            policy_version=_name(policy_version, field="policy_version"),
            encoder_version=ENCODER_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            mean_parameters=snapshot.mean.tolist(),
            precision_parameters=snapshot.precision.tolist(),
            observation_noise=float(snapshot.observation_variance),
            calibration_quantile=float(calibration_quantile),
            coverage_metrics=dict(coverage_metrics or {}),
            valid_task_slices=tuple(task_slices),
            observation_ids=tuple(self.observation_ids),
        )


def calibrate_skill_validation(
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
    """Calibrate one dataset-specific Skill with complete problems as clusters.

    The aggregate interval and harm probability are nonparametric bootstrap
    estimates over problem-level paired effects.  Split-conformal residuals
    assess the discovery-posterior interval on the independent confirmation
    problems.  The resulting probability is explicitly a bootstrap estimate,
    not an uncalibrated Gaussian posterior probability.
    """

    family = _name(task_family, field="task_family")
    if family not in DATASETS:
        raise ValueError("task_family must be hotpotqa or triviaqa")
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
    # The observed point estimate must be inside its reported interval even in
    # a highly discrete small sample.
    lower = min(float(lower), effect_mean)
    upper = max(float(upper), effect_mean)
    harm_probability = float(np.mean(bootstrap_means < -float(harm_threshold)))

    residuals = np.abs(effects - float(predicted_mean)) / (float(predicted_std) + 1e-12)
    conformal_level = min(1.0, math.ceil((effects.size + 1) * (1 - alpha)) / effects.size)
    conformal_quantile = float(
        np.quantile(residuals, conformal_level, method="higher")
    )
    conformal_lower = float(predicted_mean) - conformal_quantile * float(predicted_std)
    conformal_upper = float(predicted_mean) + conformal_quantile * float(predicted_std)
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
    "DATASETS",
    "ENCODER_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "POSTERIOR_VERSION",
    "JointQAPosteriorScheduler",
    "JointQASkillFeatureMap",
    "ScheduledCandidate",
    "calibrate_skill_validation",
]
