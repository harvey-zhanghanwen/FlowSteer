"""Combination-ledger readouts and posterior latent-risk calculation.

This module implements section 5 and the straddle rule in section 6.1 of
``LatentLoss_Implementation_Spec.md``.  "Latent loss" here is a posterior
risk diagnostic written into the Director observation.  It is deliberately
not a differentiable objective, a reward term, or an input to GRPO advantage
calculation.

The ledger owns posterior fitting and sensor calibration.  This module only
enumerates one-field alternatives, reads that frozen posterior, and produces
an auditable pre/post execution readout.  It never mutates the AgentGraph or
the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .combination_ledger import CombinationPosterior, DecisionKey, SurfaceSignal


_MUTABLE_FIELDS = (
    "role_cluster",
    "model_id",
    "edge_type",
    "same_model_as_upstream",
)


@dataclass(frozen=True)
class LatentLossConfig:
    """Frozen decision parameters from sections 5 and 9 of the specification."""

    tau: float = 0.8
    delta_min: float = 0.05
    top_k_candidates: int = 5
    posterior_samples: int = 200
    rollback_cost: float | None = None

    def __post_init__(self) -> None:
        if not 0.5 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0.5, 1]")
        if self.delta_min < 0 or not np.isfinite(self.delta_min):
            raise ValueError("delta_min must be finite and non-negative")
        if self.top_k_candidates <= 0:
            raise ValueError("top_k_candidates must be positive")
        if self.posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive")
        if self.rollback_cost is not None and (
            self.rollback_cost < 0 or not np.isfinite(self.rollback_cost)
        ):
            raise ValueError("rollback_cost must be finite and non-negative")

    @property
    def post_execution_cost(self) -> float:
        """Cost of revising one completed edit (§5.4)."""

        return self.delta_min if self.rollback_cost is None else self.rollback_cost


@dataclass(frozen=True)
class CandidateEstimate:
    """Posterior contrast for a one-field alternative versus the current key."""

    key: DecisionKey
    changed_field: str
    changed_value: str | bool
    mean: float
    variance: float
    lower: float
    upper: float
    unknown: bool = False

    def __post_init__(self) -> None:
        values = (self.mean, self.variance, self.lower, self.upper)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("candidate posterior diagnostics must be finite")
        if self.variance < 0:
            raise ValueError("candidate posterior variance cannot be negative")
        if self.lower > self.upper:
            raise ValueError("candidate posterior interval is reversed")
        if self.changed_field not in _MUTABLE_FIELDS:
            raise ValueError(f"unsupported changed field: {self.changed_field}")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": _key_payload(self.key),
            "changed_field": self.changed_field,
            "changed_value": self.changed_value,
            "mean": self.mean,
            "variance": self.variance,
            "lower": self.lower,
            "upper": self.upper,
            "unknown": self.unknown,
        }


@dataclass(frozen=True)
class PreExecutionReadout:
    """Ranked, non-binding alternatives written before an edit executes."""

    step_id: int | str
    current: DecisionKey
    candidates: tuple[CandidateEstimate, ...]
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "pre_execution",
            "step_id": self.step_id,
            "current": _key_payload(self.current),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "text": self.text,
        }


@dataclass(frozen=True)
class PostExecutionReadout:
    """Posterior latent-risk result after one surface-normal execution."""

    step_id: int | str
    current: DecisionKey
    candidates: tuple[CandidateEstimate, ...]
    computed: bool
    reason: str
    risk_probability: float | None
    q0: float | None
    expected_surface_gain: float | None
    expected_latent_loss: float | None
    likelihood_ratio: float | None
    cost: float
    warning: bool
    warning_suppressed_by_round_budget: bool
    straddle_score: float
    fork_candidate: bool
    best_alternative: CandidateEstimate | None
    text: str

    def __post_init__(self) -> None:
        if self.risk_probability is not None and not 0.0 <= self.risk_probability <= 1.0:
            raise ValueError("risk_probability must lie in [0, 1]")
        if self.q0 is not None and not 0.0 < self.q0 < 1.0:
            raise ValueError("q0 must lie strictly between zero and one")
        if self.cost < 0 or not np.isfinite(self.cost):
            raise ValueError("cost must be finite and non-negative")
        if self.straddle_score < 0 or not np.isfinite(self.straddle_score):
            raise ValueError("straddle_score must be finite and non-negative")
        if self.warning and not self.computed:
            raise ValueError("an uncomputed latent risk cannot emit a warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "post_execution",
            "step_id": self.step_id,
            "current": _key_payload(self.current),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "computed": self.computed,
            "reason": self.reason,
            "risk_probability": self.risk_probability,
            "q0": self.q0,
            "expected_surface_gain": self.expected_surface_gain,
            "expected_latent_loss": self.expected_latent_loss,
            "likelihood_ratio": self.likelihood_ratio,
            "cost": self.cost,
            "warning": self.warning,
            "warning_suppressed_by_round_budget": (
                self.warning_suppressed_by_round_budget
            ),
            "straddle_score": self.straddle_score,
            "fork_candidate": self.fork_candidate,
            "best_alternative": (
                None
                if self.best_alternative is None
                else self.best_alternative.to_dict()
            ),
            "text": self.text,
        }


@dataclass(frozen=True)
class JointMaximumEstimate:
    """Diagnostic for the maximum over correlated posterior contrasts."""

    plug_in_maximum: float
    posterior_expected_maximum: float
    posterior_standard_deviation: float


def _key_payload(key: DecisionKey) -> dict[str, object]:
    return {
        "task_family": key.task_family,
        "role_cluster": key.role_cluster,
        "model_id": key.model_id,
        "edge_type": key.edge_type,
        "same_model_as_upstream": key.same_model_as_upstream,
        "stage": key.stage,
    }


def changed_field(current: DecisionKey, candidate: DecisionKey) -> str:
    """Return the one changed decision field, rejecting invalid candidates.

    ``task_family`` and ``stage`` identify the decision context and therefore
    cannot change inside a same-snapshot counterfactual.  A changed upstream
    model may be represented by the ``same_model_as_upstream`` field, with the
    caller responsible for binding it to the upstream pre-execution snapshot.
    """

    if current.task_family != candidate.task_family or current.stage != candidate.stage:
        raise ValueError("a candidate cannot change task_family or stage")
    changed = [
        field
        for field in _MUTABLE_FIELDS
        if getattr(current, field) != getattr(candidate, field)
    ]
    if len(changed) != 1:
        raise ValueError("a candidate must change exactly one decision field")
    return changed[0]


def single_field_candidates(
    current: DecisionKey,
    *,
    model_ids: Iterable[str] = (),
    edge_types: Iterable[str] = (),
    role_clusters: Iterable[str] = (),
    same_model_options: Iterable[bool] = (),
) -> tuple[DecisionKey, ...]:
    """Enumerate unique alternatives that each change exactly one field (§5.1)."""

    candidates: list[DecisionKey] = []
    seen: set[DecisionKey] = set()
    alternatives = (
        ("model_id", tuple(model_ids)),
        ("edge_type", tuple(edge_types)),
        ("role_cluster", tuple(role_clusters)),
        ("same_model_as_upstream", tuple(same_model_options)),
    )
    for field, values in alternatives:
        for value in values:
            if value == getattr(current, field):
                continue
            candidate = replace(current, **{field: value})
            # DecisionKey performs domain validation; this assertion protects
            # the intervention invariant if that schema later changes.
            changed_field(current, candidate)
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return tuple(candidates)


def rank_candidates(
    posterior: CombinationPosterior,
    current: DecisionKey,
    candidates: Sequence[DecisionKey],
    *,
    top_k: int = 5,
) -> tuple[CandidateEstimate, ...]:
    """Rank candidates by posterior expected contrast and retain top-Kc."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    estimates: list[tuple[int, CandidateEstimate]] = []
    seen: set[DecisionKey] = set()
    for index, candidate in enumerate(candidates):
        if candidate in seen:
            continue
        seen.add(candidate)
        field = changed_field(current, candidate)
        contrast = posterior.predict_contrast(candidate, current)
        estimates.append(
            (
                index,
                CandidateEstimate(
                    key=candidate,
                    changed_field=field,
                    changed_value=getattr(candidate, field),
                    mean=float(contrast.mean),
                    variance=float(contrast.variance),
                    lower=float(contrast.lower),
                    upper=float(contrast.upper),
                    unknown=posterior.probe_count == 0,
                ),
            )
        )
    estimates.sort(key=lambda item: (-item[1].mean, item[0]))
    return tuple(estimate for _, estimate in estimates[:top_k])


def pre_execution_readout(
    posterior: CombinationPosterior,
    *,
    step_id: int | str,
    current: DecisionKey,
    candidates: Sequence[DecisionKey],
    config: LatentLossConfig = LatentLossConfig(),
) -> PreExecutionReadout:
    """Build the non-binding execution-preceding Director readout (§5.2)."""

    ranked = rank_candidates(
        posterior,
        current,
        candidates,
        top_k=config.top_k_candidates,
    )
    lines = [
        f"[Ledger] step {step_id} candidates (expected gain vs current, 90% interval):"
    ]
    if not ranked:
        lines.append("  (none)")
    for estimate in ranked:
        label = _candidate_label(estimate)
        prefix = "unknown: " if estimate.unknown else ""
        lines.append(
            f"  ({prefix}{label})  {estimate.mean:+.2f} "
            f"[{estimate.lower:.2f}, {estimate.upper:.2f}]"
        )
    return PreExecutionReadout(
        step_id=step_id,
        current=current,
        candidates=ranked,
        text="\n".join(lines),
    )


def straddle_score(
    candidates: Sequence[CandidateEstimate],
    *,
    delta_min: float = 0.05,
) -> float:
    """Return the widest 90% interval that straddles ``delta_min`` (§6.1)."""

    if delta_min < 0 or not np.isfinite(delta_min):
        raise ValueError("delta_min must be finite and non-negative")
    widths = [
        candidate.upper - candidate.lower
        for candidate in candidates
        if candidate.lower < delta_min < candidate.upper
    ]
    return float(max(widths, default=0.0))


def joint_maximum_estimate(
    posterior: CombinationPosterior,
    *,
    current: DecisionKey,
    candidates: Sequence[DecisionKey],
    rng: np.random.Generator,
    samples: int = 200,
) -> JointMaximumEstimate:
    """Compare plug-in and joint-posterior maxima for diagnostics.

    For random contrasts ``X_a``, convexity gives
    ``E[max_a X_a] >= max_a E[X_a]``.  Section 11 Phase B item 4 states the
    reverse direction; that sentence is mathematically ambiguous.  We retain
    the specified *joint* sampling calculation and expose both quantities so
    tests enforce the mathematically consistent direction instead of changing
    the algorithm to satisfy the reversed sentence.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if samples <= 0:
        raise ValueError("samples must be positive")
    vectors = _contrast_matrix(posterior, current, candidates)
    if vectors.shape[0] == 0:
        raise ValueError("at least one candidate is required")
    mean_contrasts = vectors @ posterior.mean
    parameter_draws = posterior.sample_parameters(rng, size=samples)
    sampled_maxima = np.max(parameter_draws @ vectors.T, axis=1)
    return JointMaximumEstimate(
        plug_in_maximum=float(np.max(mean_contrasts)),
        posterior_expected_maximum=float(np.mean(sampled_maxima)),
        posterior_standard_deviation=float(np.std(sampled_maxima, ddof=0)),
    )


def post_execution_latent_loss(
    posterior: CombinationPosterior,
    *,
    step_id: int | str,
    prefix_before: Sequence[DecisionKey],
    current: DecisionKey,
    candidates: Sequence[DecisionKey],
    surface: SurfaceSignal | None,
    remaining_rounds: int,
    rng: np.random.Generator,
    config: LatentLossConfig = LatentLossConfig(),
) -> PostExecutionReadout:
    """Compute section 5.3 latent risk without changing policy or reward.

    Only surface-normal execution reaches the posterior calculation.  Canvas
    validation failures and execution errors return an uncomputed readout so
    the existing Canvas feedback/recovery path remains authoritative.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if remaining_rounds < 0:
        raise ValueError("remaining_rounds must be non-negative")
    ranked = rank_candidates(
        posterior,
        current,
        candidates,
        top_k=config.top_k_candidates,
    )
    cost = config.post_execution_cost
    if surface is None:
        return _uncomputed_post_readout(
            step_id, current, ranked, cost, reason="missing_surface"
        )
    if not surface.is_normal:
        return _uncomputed_post_readout(
            step_id, current, ranked, cost, reason="abnormal_surface"
        )
    if not ranked:
        return _uncomputed_post_readout(
            step_id, current, ranked, cost, reason="no_candidates"
        )

    prefix = tuple(prefix_before) + (current,)
    q0 = float(posterior.expected_terminal(current.task_family, prefix))
    # ``CombinationPosterior.expected_terminal`` clips to [0.02, 0.98].
    if not 0.0 < q0 < 1.0 or not np.isfinite(q0):
        raise ValueError("posterior expected_terminal must return a finite value in (0, 1)")

    candidate_keys = tuple(estimate.key for estimate in ranked)
    vectors = _contrast_matrix(posterior, current, candidate_keys)
    parameter_draws = posterior.sample_parameters(
        rng,
        size=config.posterior_samples,
    )
    expected_shape = (config.posterior_samples, posterior.mean.size)
    if parameter_draws.shape != expected_shape:
        raise ValueError(
            "posterior parameter draws have shape "
            f"{parameter_draws.shape}, expected {expected_shape}"
        )
    delta_samples = np.max(parameter_draws @ vectors.T, axis=1)

    sensitivity, false_positive_rate = posterior.sample_sensor_rates(
        surface.sensor_class,
        rng,
        size=config.posterior_samples,
    )
    sensitivity = _probability_vector(
        sensitivity, config.posterior_samples, "sensitivity"
    )
    false_positive_rate = _probability_vector(
        false_positive_rate, config.posterior_samples, "false_positive_rate"
    )
    epsilon = np.finfo(np.float64).eps
    if surface.value:
        log_likelihood_ratio = np.log(np.clip(sensitivity, epsilon, 1.0)) - np.log(
            np.clip(false_positive_rate, epsilon, 1.0)
        )
    else:
        log_likelihood_ratio = np.log(
            np.clip(1.0 - sensitivity, epsilon, 1.0)
        ) - np.log(np.clip(1.0 - false_positive_rate, epsilon, 1.0))

    q1 = _sigmoid(_logit(q0) + log_likelihood_ratio)
    surface_gain = q1 - q0
    latent_samples = delta_samples - surface_gain
    risk_probability = float(
        np.mean(latent_samples > config.delta_min + cost)
    )

    middle = 1.0 - config.tau < risk_probability < config.tau
    score = straddle_score(ranked, delta_min=config.delta_min) if middle else 0.0
    suppressed = risk_probability >= config.tau and remaining_rounds <= 1
    warning = risk_probability >= config.tau and not suppressed
    best = ranked[0]
    likelihood_ratio = float(posterior.likelihood_ratio(surface))
    text = (
        _warning_text(
            step_id=step_id,
            risk_probability=risk_probability,
            current=current,
            surface=surface,
            likelihood_ratio=likelihood_ratio,
            best=best,
            tau=config.tau,
        )
        if warning
        else ""
    )
    return PostExecutionReadout(
        step_id=step_id,
        current=current,
        candidates=ranked,
        computed=True,
        reason=("round_budget" if suppressed else "computed"),
        risk_probability=risk_probability,
        q0=q0,
        expected_surface_gain=float(np.mean(surface_gain)),
        expected_latent_loss=float(np.mean(latent_samples)),
        likelihood_ratio=likelihood_ratio,
        cost=cost,
        warning=warning,
        warning_suppressed_by_round_budget=suppressed,
        straddle_score=score,
        fork_candidate=middle and score > 0.0,
        best_alternative=best,
        text=text,
    )


def _contrast_matrix(
    posterior: CombinationPosterior,
    current: DecisionKey,
    candidates: Sequence[DecisionKey],
) -> np.ndarray:
    # The dynamic ledger registers previously unseen first-order columns on
    # demand.  Register the complete candidate set before materialising any
    # vector so every row has the same frozen dimension.
    posterior.design_vector(current)
    for candidate in candidates:
        changed_field(current, candidate)
        posterior.design_vector(candidate)
    vectors = []
    for candidate in candidates:
        vector = np.asarray(
            posterior.contrast_vector(candidate, current), dtype=np.float64
        )
        if vector.ndim != 1 or vector.shape != posterior.mean.shape:
            raise ValueError("posterior contrast vector has the wrong shape")
        if not np.all(np.isfinite(vector)):
            raise ValueError("posterior contrast vectors must be finite")
        vectors.append(vector)
    if not vectors:
        return np.empty((0, posterior.mean.size), dtype=np.float64)
    return np.stack(vectors, axis=0)


def _probability_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} draws must have shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} draws must be finite probabilities")
    return array


def _logit(probability: float) -> float:
    return float(np.log(probability) - np.log1p(-probability))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _candidate_label(candidate: CandidateEstimate) -> str:
    names = {
        "model_id": "model",
        "edge_type": "edge",
        "role_cluster": "role",
        "same_model_as_upstream": "same_model_as_upstream",
    }
    return f"{names[candidate.changed_field]}={candidate.changed_value}"


def _warning_text(
    *,
    step_id: int | str,
    risk_probability: float,
    current: DecisionKey,
    surface: SurfaceSignal,
    likelihood_ratio: float,
    best: CandidateEstimate,
    tau: float,
) -> str:
    return (
        f"[Ledger] step {step_id}: risk={risk_probability:.2f} "
        f"(threshold {tau}).\n"
        f"Current: {current.role_cluster}/{current.model_id}/{current.edge_type}, "
        f"same_model_as_upstream={current.same_model_as_upstream}.\n"
        f'Surface signal "{surface.kind}={surface.value}" carries little information '
        f"under this configuration (LR={likelihood_ratio:.1f}).\n"
        f"Best alternative: {_candidate_label(best)}, expected gain {best.mean:+.2f} "
        f"[{best.lower:.2f}, {best.upper:.2f}].\n"
        "You may revise the last decision or continue."
    )


def _uncomputed_post_readout(
    step_id: int | str,
    current: DecisionKey,
    candidates: tuple[CandidateEstimate, ...],
    cost: float,
    *,
    reason: str,
) -> PostExecutionReadout:
    return PostExecutionReadout(
        step_id=step_id,
        current=current,
        candidates=candidates,
        computed=False,
        reason=reason,
        risk_probability=None,
        q0=None,
        expected_surface_gain=None,
        expected_latent_loss=None,
        likelihood_ratio=None,
        cost=cost,
        warning=False,
        warning_suppressed_by_round_budget=False,
        straddle_score=0.0,
        fork_candidate=False,
        best_alternative=(candidates[0] if candidates else None),
        text="",
    )


__all__ = [
    "CandidateEstimate",
    "JointMaximumEstimate",
    "LatentLossConfig",
    "PostExecutionReadout",
    "PreExecutionReadout",
    "changed_field",
    "joint_maximum_estimate",
    "post_execution_latent_loss",
    "pre_execution_readout",
    "rank_candidates",
    "single_field_candidates",
    "straddle_score",
]
