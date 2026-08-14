"""Posterior-driven UCB and rollout-consistent Thompson policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .posterior import BayesianLinearPosterior
from .records import PolicyDecision, readonly_array


def _contexts(value: Iterable[Iterable[float]] | np.ndarray, dimension: int) -> np.ndarray:
    contexts = np.asarray(value, dtype=np.float64)
    if contexts.ndim != 2 or contexts.shape[1] != dimension:
        raise ValueError(f"contexts must have shape [actions, {dimension}]")
    if contexts.shape[0] == 0:
        raise ValueError("at least one action is required")
    if not np.all(np.isfinite(contexts)):
        raise ValueError("contexts must be finite")
    return contexts


def _candidate_indices(mask: Iterable[bool] | np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.arange(size, dtype=np.int64)
    candidate_mask = np.asarray(mask, dtype=bool)
    if candidate_mask.shape != (size,):
        raise ValueError(f"candidate mask must have shape {(size,)}")
    indices = np.flatnonzero(candidate_mask)
    if indices.size == 0:
        raise ValueError("candidate mask excludes every action")
    return indices


def _random_argmax(
    values: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> int:
    best_value = float(np.max(values[indices]))
    tied = indices[
        np.isclose(values[indices], best_value, rtol=1e-12, atol=1e-12)
    ]
    return int(rng.choice(tied))


class UCBPolicy:
    """Upper-confidence selection using epistemic, not predictive, variance."""

    def __init__(self, exploration_alpha: float = 1.0, *, seed: int | None = None) -> None:
        if exploration_alpha < 0 or not np.isfinite(exploration_alpha):
            raise ValueError("exploration_alpha must be finite and non-negative")
        self.exploration_alpha = float(exploration_alpha)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        posterior: BayesianLinearPosterior,
        contexts: Iterable[Iterable[float]] | np.ndarray,
        candidate_mask: Iterable[bool] | np.ndarray | None = None,
    ) -> PolicyDecision:
        matrix = _contexts(contexts, posterior.dimension)
        indices = _candidate_indices(candidate_mask, matrix.shape[0])
        mean = matrix @ posterior.mean
        variances = np.array(
            [posterior.epistemic_variance(context) for context in matrix]
        )
        bonuses = self.exploration_alpha * np.sqrt(np.maximum(variances, 0.0))
        scores = mean + bonuses
        action = _random_argmax(scores, indices, self._rng)
        return PolicyDecision(action, scores, mean, bonuses)


@dataclass(frozen=True)
class ThompsonRollout:
    """A parameter draw reused for every decision in one rollout."""

    parameter_draw: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_draw", readonly_array(self.parameter_draw, ndim=1)
        )

    def select(
        self,
        contexts: Iterable[Iterable[float]] | np.ndarray,
        candidate_mask: Iterable[bool] | np.ndarray | None = None,
        *,
        posterior_mean: Iterable[float] | np.ndarray | None = None,
    ) -> PolicyDecision:
        matrix = _contexts(contexts, self.parameter_draw.size)
        indices = _candidate_indices(candidate_mask, matrix.shape[0])
        scores = matrix @ self.parameter_draw
        if posterior_mean is None:
            expected = np.zeros(matrix.shape[0], dtype=np.float64)
        else:
            mean = np.asarray(posterior_mean, dtype=np.float64)
            if mean.shape != self.parameter_draw.shape:
                raise ValueError("posterior_mean has the wrong shape")
            expected = matrix @ mean
        action = int(indices[np.argmax(scores[indices])])
        return PolicyDecision(action, scores, expected, scores - expected)


class ThompsonSamplingPolicy:
    """Create explicit one-sample-per-rollout Thompson decision objects."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def start_rollout(self, posterior: BayesianLinearPosterior) -> ThompsonRollout:
        return ThompsonRollout(posterior.sample_parameters(self._rng))
