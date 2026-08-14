"""Numerically stable Bayesian linear posterior for exploration policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .records import readonly_array


POSTERIOR_STATE_VERSION = "bayesian-linear-posterior-v1"


def _as_vector(value: Iterable[float] | np.ndarray, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape {(dimension,)}, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _cholesky_solve(precision: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve ``precision @ x = rhs`` without forming a matrix inverse."""

    factor = np.linalg.cholesky(precision)
    intermediate = np.linalg.solve(factor, right_hand_side)
    return np.linalg.solve(factor.T, intermediate)


@dataclass(frozen=True)
class PosteriorSnapshot:
    """Immutable checkpoint of posterior sufficient statistics."""

    precision: np.ndarray
    information: np.ndarray
    observation_variance: float
    update_count: int
    version: str = POSTERIOR_STATE_VERSION

    def __post_init__(self) -> None:
        precision = readonly_array(self.precision, ndim=2)
        if precision.shape[0] != precision.shape[1]:
            raise ValueError("posterior precision must be square")
        if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
            raise ValueError("posterior precision must be symmetric")
        information = readonly_array(
            self.information, ndim=1, shape=(precision.shape[0],)
        )
        if self.version != POSTERIOR_STATE_VERSION:
            raise ValueError(f"unsupported posterior version: {self.version}")
        if self.observation_variance <= 0 or not np.isfinite(self.observation_variance):
            raise ValueError("observation_variance must be finite and positive")
        if self.update_count < 0:
            raise ValueError("update_count must be non-negative")
        np.linalg.cholesky(precision)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "information", information)

    @property
    def dimension(self) -> int:
        return self.information.size

    @property
    def mean(self) -> np.ndarray:
        mean = _cholesky_solve(self.precision, self.information)
        mean.setflags(write=False)
        return mean


class BayesianLinearPosterior:
    """Gaussian posterior represented by precision and information vectors.

    Observations follow ``y = x.T @ theta + noise``.  The default online
    update uses ``observation_variance``.  A paired difference of two
    independent observations defaults to twice that variance.
    """

    def __init__(
        self,
        dimension: int,
        *,
        prior_precision: float | np.ndarray = 1.0,
        prior_mean: Iterable[float] | np.ndarray | None = None,
        observation_variance: float = 1.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if observation_variance <= 0 or not np.isfinite(observation_variance):
            raise ValueError("observation_variance must be finite and positive")
        self.dimension = int(dimension)
        if np.isscalar(prior_precision):
            scale = float(prior_precision)
            if scale <= 0 or not np.isfinite(scale):
                raise ValueError("prior_precision must be finite and positive")
            precision = scale * np.eye(self.dimension, dtype=np.float64)
        else:
            precision = np.asarray(prior_precision, dtype=np.float64)
            if precision.shape != (self.dimension, self.dimension):
                raise ValueError("prior precision has the wrong shape")
            if not np.all(np.isfinite(precision)):
                raise ValueError("prior precision must be finite")
            if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
                raise ValueError("prior precision must be symmetric")
            precision = precision.copy()
            np.linalg.cholesky(precision)
        mean = (
            np.zeros(self.dimension, dtype=np.float64)
            if prior_mean is None
            else _as_vector(prior_mean, self.dimension, "prior_mean").copy()
        )
        self._precision = precision
        self._information = precision @ mean
        self.observation_variance = float(observation_variance)
        self.update_count = 0

    @property
    def precision(self) -> np.ndarray:
        result = self._precision.copy()
        result.setflags(write=False)
        return result

    @property
    def information(self) -> np.ndarray:
        result = self._information.copy()
        result.setflags(write=False)
        return result

    @property
    def mean(self) -> np.ndarray:
        result = _cholesky_solve(self._precision, self._information)
        result.setflags(write=False)
        return result

    @property
    def covariance(self) -> np.ndarray:
        identity = np.eye(self.dimension, dtype=np.float64)
        result = _cholesky_solve(self._precision, identity)
        result = 0.5 * (result + result.T)
        result.setflags(write=False)
        return result

    def _validated_variance(self, value: float | None, default: float) -> float:
        variance = default if value is None else float(value)
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError("observation variance must be finite and positive")
        return variance

    def update(
        self,
        context: Iterable[float] | np.ndarray,
        outcome: float,
        *,
        observation_variance: float | None = None,
    ) -> None:
        """Apply one online conjugate update."""

        vector = _as_vector(context, self.dimension, "context")
        value = float(outcome)
        if not np.isfinite(value):
            raise ValueError("outcome must be finite")
        variance = self._validated_variance(
            observation_variance, self.observation_variance
        )
        self._precision += np.outer(vector, vector) / variance
        self._information += vector * value / variance
        self.update_count += 1

    def update_paired_difference(
        self,
        left_context: Iterable[float] | np.ndarray,
        right_context: Iterable[float] | np.ndarray,
        difference: float,
        *,
        difference_variance: float | None = None,
    ) -> None:
        """Update from ``left_outcome - right_outcome`` for a paired probe."""

        left = _as_vector(left_context, self.dimension, "left_context")
        right = _as_vector(right_context, self.dimension, "right_context")
        variance = self._validated_variance(
            difference_variance, 2.0 * self.observation_variance
        )
        self.update(left - right, difference, observation_variance=variance)

    def update_paired_outcomes(
        self,
        left_context: Iterable[float] | np.ndarray,
        left_outcome: float,
        right_context: Iterable[float] | np.ndarray,
        right_outcome: float,
        *,
        difference_variance: float | None = None,
    ) -> None:
        self.update_paired_difference(
            left_context,
            right_context,
            float(left_outcome) - float(right_outcome),
            difference_variance=difference_variance,
        )

    def epistemic_variance(self, context: Iterable[float] | np.ndarray) -> float:
        vector = _as_vector(context, self.dimension, "context")
        solved = _cholesky_solve(self._precision, vector)
        return max(0.0, float(vector @ solved))

    def predictive_variance(
        self,
        context: Iterable[float] | np.ndarray,
        *,
        observation_variance: float | None = None,
    ) -> float:
        variance = self._validated_variance(
            observation_variance, self.observation_variance
        )
        return self.epistemic_variance(context) + variance

    def predict_mean(self, context: Iterable[float] | np.ndarray) -> float:
        vector = _as_vector(context, self.dimension, "context")
        return float(vector @ self.mean)

    def sample_parameters(
        self,
        rng: np.random.Generator,
        *,
        size: int | None = None,
    ) -> np.ndarray:
        """Draw posterior parameters using triangular precision solves."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        factor = np.linalg.cholesky(self._precision)
        mean = self.mean
        if size is None:
            noise = rng.standard_normal(self.dimension)
            return mean + np.linalg.solve(factor.T, noise)
        if size <= 0:
            raise ValueError("size must be positive")
        noise = rng.standard_normal((size, self.dimension))
        transformed = np.linalg.solve(factor.T, noise.T).T
        return mean[None, :] + transformed

    def snapshot(self) -> PosteriorSnapshot:
        return PosteriorSnapshot(
            precision=self._precision,
            information=self._information,
            observation_variance=self.observation_variance,
            update_count=self.update_count,
        )

    def restore(self, snapshot: PosteriorSnapshot) -> None:
        if snapshot.dimension != self.dimension:
            raise ValueError("snapshot dimension does not match posterior")
        self._precision = np.array(snapshot.precision, copy=True)
        self._information = np.array(snapshot.information, copy=True)
        self.observation_variance = float(snapshot.observation_variance)
        self.update_count = int(snapshot.update_count)

    @classmethod
    def from_snapshot(cls, snapshot: PosteriorSnapshot) -> "BayesianLinearPosterior":
        posterior = cls(
            snapshot.dimension,
            prior_precision=snapshot.precision,
            prior_mean=snapshot.mean,
            observation_variance=snapshot.observation_variance,
        )
        posterior.update_count = snapshot.update_count
        return posterior

    def state_dict(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "version": snapshot.version,
            "dimension": snapshot.dimension,
            "precision": snapshot.precision.tolist(),
            "information": snapshot.information.tolist(),
            "observation_variance": snapshot.observation_variance,
            "update_count": snapshot.update_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "BayesianLinearPosterior":
        if state.get("version") != POSTERIOR_STATE_VERSION:
            raise ValueError(f"unsupported posterior state: {state.get('version')}")
        snapshot = PosteriorSnapshot(
            precision=np.asarray(state["precision"], dtype=np.float64),
            information=np.asarray(state["information"], dtype=np.float64),
            observation_variance=float(state["observation_variance"]),
            update_count=int(state["update_count"]),
        )
        if int(state["dimension"]) != snapshot.dimension:
            raise ValueError("serialized posterior dimension is inconsistent")
        return cls.from_snapshot(snapshot)
