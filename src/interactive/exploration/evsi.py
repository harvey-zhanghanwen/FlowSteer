"""Particle approximations to EVPI and EVSI with common random numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .records import readonly_array


@dataclass(frozen=True)
class CommonRandomNumbers:
    """Reusable draws that reduce noise when comparing candidate probes."""

    particle_uniforms: np.ndarray
    standard_normal_noise: np.ndarray

    def __post_init__(self) -> None:
        uniforms = readonly_array(self.particle_uniforms, ndim=1)
        noise = readonly_array(self.standard_normal_noise, ndim=1)
        if uniforms.shape != noise.shape or uniforms.size == 0:
            raise ValueError("common-random-number arrays must be non-empty and aligned")
        if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
            raise ValueError("particle uniforms must lie in [0, 1)")
        object.__setattr__(self, "particle_uniforms", uniforms)
        object.__setattr__(self, "standard_normal_noise", noise)

    @property
    def n_samples(self) -> int:
        return self.particle_uniforms.size


@dataclass(frozen=True)
class EVPIEstimate:
    value: float
    current_value: float
    perfect_information_value: float


@dataclass(frozen=True)
class EVSIEstimate:
    value: float
    current_value: float
    expected_post_probe_value: float
    standard_error: float
    n_samples: int


def make_common_random_numbers(
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> CommonRandomNumbers:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if rng is not None and seed is not None:
        raise ValueError("provide rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return CommonRandomNumbers(
        particle_uniforms=generator.random(n_samples),
        standard_normal_noise=generator.standard_normal(n_samples),
    )


def clamp_tiny_negative(value: float, *, tolerance: float = 1e-12) -> float:
    """Clamp negative round-off without hiding materially negative estimates."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    result = float(value)
    if -tolerance <= result < 0.0:
        return 0.0
    return result


def _particles_and_weights(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    weights: Iterable[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(utilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("utilities must have shape [particles, actions]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("particle utilities must be finite")
    if weights is None:
        probabilities = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    else:
        probabilities = np.asarray(weights, dtype=np.float64)
        if probabilities.shape != (matrix.shape[0],):
            raise ValueError("particle weights have the wrong shape")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("particle weights must be finite and non-negative")
        total = float(np.sum(probabilities))
        if total <= 0:
            raise ValueError("particle weights must have positive mass")
        probabilities = probabilities / total
    return matrix, probabilities


def estimate_particle_evpi(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    negative_tolerance: float = 1e-12,
) -> EVPIEstimate:
    matrix, probabilities = _particles_and_weights(utilities, weights)
    current = float(np.max(probabilities @ matrix))
    perfect = float(probabilities @ np.max(matrix, axis=1))
    value = clamp_tiny_negative(perfect - current, tolerance=negative_tolerance)
    return EVPIEstimate(value=value, current_value=current, perfect_information_value=perfect)


def particle_evpi(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    negative_tolerance: float = 1e-12,
) -> float:
    return estimate_particle_evpi(
        utilities, weights=weights, negative_tolerance=negative_tolerance
    ).value


def _sample_particle_indices(
    probabilities: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniforms, side="right")


def estimate_particle_evsi(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    probe_signal: Iterable[float] | np.ndarray,
    observation_std: float,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    common_random_numbers: CommonRandomNumbers | None = None,
    n_samples: int = 4096,
    rng: np.random.Generator | None = None,
    negative_tolerance: float = 1e-12,
) -> EVSIEstimate:
    """Estimate expected sample information for a scalar noisy probe.

    Each particle supplies action utilities and the latent mean of the proposed
    probe.  Observations are Gaussian with a common ``observation_std``.  Pass
    the same ``CommonRandomNumbers`` to competing probes for lower-variance
    comparisons.
    """

    matrix, probabilities = _particles_and_weights(utilities, weights)
    signal = np.asarray(probe_signal, dtype=np.float64)
    if signal.shape != (matrix.shape[0],) or not np.all(np.isfinite(signal)):
        raise ValueError("probe_signal must be finite with one value per particle")
    sigma = float(observation_std)
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("observation_std must be finite and positive")
    draws = common_random_numbers or make_common_random_numbers(n_samples, rng=rng)
    true_particles = _sample_particle_indices(probabilities, draws.particle_uniforms)
    observations = signal[true_particles] + sigma * draws.standard_normal_noise
    log_prior = np.full(probabilities.shape, -np.inf, dtype=np.float64)
    positive = probabilities > 0
    log_prior[positive] = np.log(probabilities[positive])

    posterior_decision_values = np.empty(draws.n_samples, dtype=np.float64)
    inverse_two_variance = 0.5 / (sigma * sigma)
    for index, observation in enumerate(observations):
        log_weights = log_prior - inverse_two_variance * (observation - signal) ** 2
        maximum = float(np.max(log_weights))
        posterior_weights = np.exp(log_weights - maximum)
        posterior_weights /= np.sum(posterior_weights)
        posterior_decision_values[index] = np.max(posterior_weights @ matrix)

    current = float(np.max(probabilities @ matrix))
    post_probe = float(np.mean(posterior_decision_values))
    value = clamp_tiny_negative(post_probe - current, tolerance=negative_tolerance)
    if draws.n_samples == 1:
        standard_error = 0.0
    else:
        standard_error = float(
            np.std(posterior_decision_values, ddof=1) / np.sqrt(draws.n_samples)
        )
    return EVSIEstimate(
        value=value,
        current_value=current,
        expected_post_probe_value=post_probe,
        standard_error=standard_error,
        n_samples=draws.n_samples,
    )


def particle_evsi(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    probe_signal: Iterable[float] | np.ndarray,
    observation_std: float,
    **kwargs: object,
) -> float:
    return estimate_particle_evsi(
        utilities, probe_signal, observation_std, **kwargs
    ).value


def particle_evsi_many(
    utilities: Iterable[Iterable[float]] | np.ndarray,
    probe_signals: Iterable[Iterable[float]] | np.ndarray,
    observation_std: float,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    common_random_numbers: CommonRandomNumbers | None = None,
    n_samples: int = 4096,
    rng: np.random.Generator | None = None,
    negative_tolerance: float = 1e-12,
) -> np.ndarray:
    """Compare probes with one shared common-random-number stream."""

    matrix, _ = _particles_and_weights(utilities, weights)
    signals = np.asarray(probe_signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != matrix.shape[0]:
        raise ValueError("probe_signals must have shape [probes, particles]")
    if not np.all(np.isfinite(signals)):
        raise ValueError("probe signals must be finite")
    draws = common_random_numbers or make_common_random_numbers(n_samples, rng=rng)
    values = np.array(
        [
            particle_evsi(
                matrix,
                signal,
                observation_std,
                weights=weights,
                common_random_numbers=draws,
                negative_tolerance=negative_tolerance,
            )
            for signal in signals
        ],
        dtype=np.float64,
    )
    values.setflags(write=False)
    return values
