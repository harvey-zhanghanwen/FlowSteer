"""Pure calibrated rule-mining gates for ledger-derived Skills.

The routines implement Section 7 of ``LatentLoss_Implementation_Spec.md``.
They compute evidence and lifecycle decisions only.  Publication still goes
through the existing :class:`SkillEvidenceGate` and
:class:`SkillLifecycleManager`, so this module cannot self-publish a Skill.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import SkillRecord, SkillStatus


DEFAULT_ALPHA = 0.05
DEFAULT_FDR = 0.10
DEFAULT_DELTA_MIN = 0.05
DEFAULT_MIN_DISCOVERY_PAIRS = 10
DEFAULT_MIN_CONFIRMATION_PROBLEMS = 20


@dataclass(frozen=True)
class RuleContrast:
    """A ledger contrast ready for calibrated candidate enumeration."""

    rule_id: str
    order: int
    condition: Mapping[str, Any]
    action: Mapping[str, Any]
    delta_mean: float
    posterior_variance: float
    n_eff_pairs: int
    interaction_activated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string")
        if self.order not in (1, 2):
            raise ValueError("rule order must be one or two")
        for key in ("task_family", "role_cluster", "stage"):
            if not isinstance(self.condition.get(key), str) or not self.condition[key].strip():
                raise ValueError(f"rule condition requires {key}")
        if not any(key in self.action for key in ("model_id", "edge_type")):
            raise ValueError("rule action must set model_id and/or edge_type")
        if not math.isfinite(float(self.delta_mean)):
            raise ValueError("delta_mean must be finite")
        if not math.isfinite(float(self.posterior_variance)) or self.posterior_variance < 0:
            raise ValueError("posterior_variance must be finite and non-negative")
        if isinstance(self.n_eff_pairs, bool) or not isinstance(self.n_eff_pairs, int):
            raise ValueError("n_eff_pairs must be an integer")
        if self.n_eff_pairs < 0:
            raise ValueError("n_eff_pairs must be non-negative")
        if type(self.interaction_activated) is not bool:
            raise ValueError("interaction_activated must be bool")


@dataclass(frozen=True)
class SkillRuleCandidate:
    """A discovery rule with its held-out-calibrated posterior interval."""

    rule: RuleContrast
    lower: float
    upper: float
    activation_p_value: float
    harm_probability: float

    @property
    def rule_id(self) -> str:
        return self.rule.rule_id


@dataclass(frozen=True)
class ConfirmationEvidence:
    """Evidence from probes disjoint from the discovery probes."""

    calibrated_lower: float
    problem_ids: Sequence[str]
    harm_probability: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.calibrated_lower)):
            raise ValueError("confirmation calibrated_lower must be finite")
        problem_ids = tuple(str(value) for value in self.problem_ids)
        if any(not value for value in problem_ids):
            raise ValueError("confirmation problem IDs must be non-empty")
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("confirmation must be counted by unique held-out problem")
        if not 0.0 <= float(self.harm_probability) <= 1.0:
            raise ValueError("confirmation harm_probability must be in [0, 1]")
        object.__setattr__(self, "problem_ids", problem_ids)


@dataclass(frozen=True)
class SkillStatusDecision:
    status: SkillStatus
    reasons: tuple[str, ...]
    requires_publication_gate: bool = False


def _as_finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def empirical_calibration_quantile(
    observed_deltas: Sequence[float],
    predicted_means: Sequence[float],
    posterior_variances: Sequence[float],
    noise_variances: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute the empirical (1-alpha) standardized-residual quantile."""

    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    observed = _as_finite_vector(observed_deltas, "observed_deltas")
    predicted = _as_finite_vector(predicted_means, "predicted_means")
    posterior = _as_finite_vector(posterior_variances, "posterior_variances")
    noise = _as_finite_vector(noise_variances, "noise_variances")
    if not (observed.shape == predicted.shape == posterior.shape == noise.shape):
        raise ValueError("calibration vectors must have identical shapes")
    if np.any(posterior < 0.0) or np.any(noise < 0.0):
        raise ValueError("calibration variances must be non-negative")
    total_variance = posterior + noise
    if np.any(total_variance <= 0.0):
        raise ValueError("posterior_variance + noise_variance must be positive")
    residuals = np.abs(observed - predicted) / np.sqrt(total_variance)
    return float(np.quantile(residuals, 1.0 - float(alpha), method="higher"))


def calibrated_interval(
    mean: float,
    posterior_variance: float,
    calibration_quantile: float,
) -> tuple[float, float]:
    """Return [mean-q*sigma, mean+q*sigma] for a ledger contrast."""

    numeric = (mean, posterior_variance, calibration_quantile)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("interval inputs must be finite")
    if posterior_variance < 0.0 or calibration_quantile < 0.0:
        raise ValueError("variance and calibration quantile must be non-negative")
    radius = float(calibration_quantile) * math.sqrt(float(posterior_variance))
    return float(mean) - radius, float(mean) + radius


def normal_probability_at_or_below(
    threshold: float,
    mean: float,
    posterior_variance: float,
) -> float:
    """Evaluate P(Delta <= threshold) for a univariate Gaussian contrast."""

    if not all(math.isfinite(float(value)) for value in (threshold, mean, posterior_variance)):
        raise ValueError("Gaussian probability inputs must be finite")
    if posterior_variance < 0.0:
        raise ValueError("posterior_variance must be non-negative")
    if posterior_variance == 0.0:
        return 1.0 if float(mean) <= float(threshold) else 0.0
    z = (float(threshold) - float(mean)) / math.sqrt(float(posterior_variance))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def enumerate_rule_candidates(
    contrasts: Iterable[RuleContrast],
    *,
    calibration_quantile: float,
    delta_min: float = DEFAULT_DELTA_MIN,
    minimum_pairs: int = DEFAULT_MIN_DISCOVERY_PAIRS,
) -> tuple[SkillRuleCandidate, ...]:
    """Enumerate evidence-qualified first/activated-second order rules."""

    if delta_min <= 0.0 or not math.isfinite(float(delta_min)):
        raise ValueError("delta_min must be finite and positive")
    if isinstance(minimum_pairs, bool) or minimum_pairs < 1:
        raise ValueError("minimum_pairs must be a positive integer")
    candidates: list[SkillRuleCandidate] = []
    for rule in contrasts:
        if rule.n_eff_pairs < minimum_pairs:
            continue
        if rule.order == 2 and not rule.interaction_activated:
            continue
        lower, upper = calibrated_interval(
            rule.delta_mean,
            rule.posterior_variance,
            calibration_quantile,
        )
        candidates.append(
            SkillRuleCandidate(
                rule=rule,
                lower=lower,
                upper=upper,
                activation_p_value=normal_probability_at_or_below(
                    delta_min,
                    rule.delta_mean,
                    rule.posterior_variance,
                ),
                harm_probability=normal_probability_at_or_below(
                    -float(delta_min),
                    rule.delta_mean,
                    rule.posterior_variance,
                ),
            )
        )
    candidates.sort(key=lambda item: item.rule_id)
    return tuple(candidates)


def benjamini_hochberg(
    p_values: Mapping[str, float],
    *,
    fdr: float = DEFAULT_FDR,
) -> frozenset[str]:
    """Return rule IDs accepted by the Benjamini-Hochberg step-up procedure."""

    if not math.isfinite(float(fdr)) or not 0.0 < float(fdr) < 1.0:
        raise ValueError("fdr must be in (0, 1)")
    ordered: list[tuple[str, float]] = []
    for rule_id, p_value in p_values.items():
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError("BH rule IDs must be non-empty strings")
        value = float(p_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("BH p-values must be finite and in [0, 1]")
        ordered.append((rule_id, value))
    if not ordered:
        return frozenset()
    ordered.sort(key=lambda item: (item[1], item[0]))
    count = len(ordered)
    last_accepted = -1
    for index, (_, value) in enumerate(ordered, start=1):
        if value <= float(fdr) * index / count:
            last_accepted = index - 1
    if last_accepted < 0:
        return frozenset()
    threshold = ordered[last_accepted][1]
    return frozenset(rule_id for rule_id, value in ordered if value <= threshold)


def _previous_status(previous: SkillStatus | SkillRecord) -> SkillStatus:
    return previous.status if isinstance(previous, SkillRecord) else SkillStatus(previous)


def decide_skill_status(
    candidate: SkillRuleCandidate,
    confirmation: ConfirmationEvidence | None,
    *,
    previous: SkillStatus | SkillRecord = SkillStatus.CANDIDATE,
    bh_selected: bool = False,
    version_revalidated: bool = True,
    recent_probe_effect_means: Sequence[float] = (),
    consecutive_suspended_epochs: int = 0,
    delta_min: float = DEFAULT_DELTA_MIN,
    maximum_harm_probability: float = 0.05,
    minimum_confirmation_problems: int = DEFAULT_MIN_CONFIRMATION_PROBLEMS,
) -> SkillStatusDecision:
    """Apply Section 7.3 without bypassing the existing publication gate."""

    status = _previous_status(previous)
    if status is SkillStatus.RETIRED:
        return SkillStatusDecision(status, ("retired is terminal",))
    if isinstance(consecutive_suspended_epochs, bool) or consecutive_suspended_epochs < 0:
        raise ValueError("consecutive_suspended_epochs must be non-negative")
    if not 0.0 <= maximum_harm_probability <= 1.0:
        raise ValueError("maximum_harm_probability must be in [0, 1]")
    recent = tuple(float(value) for value in recent_probe_effect_means)
    if any(not math.isfinite(value) for value in recent):
        raise ValueError("recent probe effect means must be finite")

    recent_negative = len(recent) >= 2 and float(np.mean(recent[-2:])) < 0.0
    suspension_reasons: list[str] = []
    if candidate.lower < delta_min:
        suspension_reasons.append("calibrated lower bound fell below delta_min")
    if not version_revalidated:
        suspension_reasons.append("version changed without revalidation")
    if recent_negative:
        suspension_reasons.append("mean probe effect over the latest two epochs is negative")

    confirmation_passes = bool(
        confirmation is not None
        and confirmation.calibrated_lower > delta_min
        and len(confirmation.problem_ids) >= minimum_confirmation_problems
        and confirmation.harm_probability <= maximum_harm_probability
    )
    activation_passes = bool(
        candidate.lower > delta_min
        and candidate.harm_probability <= maximum_harm_probability
        and confirmation_passes
        and bh_selected
        and version_revalidated
    )

    if status is SkillStatus.ACTIVE:
        if suspension_reasons:
            return SkillStatusDecision(SkillStatus.SUSPENDED, tuple(suspension_reasons))
        return SkillStatusDecision(SkillStatus.ACTIVE, ("active evidence remains valid",))

    if status is SkillStatus.SUSPENDED:
        if activation_passes:
            return SkillStatusDecision(
                SkillStatus.ACTIVE,
                ("held-out confirmation recovered",),
                requires_publication_gate=True,
            )
        if consecutive_suspended_epochs >= 3:
            return SkillStatusDecision(
                SkillStatus.RETIRED,
                ("suspended for three consecutive epochs without recovery",),
            )
        return SkillStatusDecision(
            SkillStatus.SUSPENDED,
            tuple(suspension_reasons) or ("awaiting held-out recovery evidence",),
        )

    if candidate.upper < delta_min:
        return SkillStatusDecision(
            SkillStatus.RETIRED,
            ("calibrated upper bound is below delta_min",),
        )
    if activation_passes:
        return SkillStatusDecision(
            SkillStatus.ACTIVE,
            ("discovery, confirmation, harm, and BH gates passed",),
            requires_publication_gate=True,
        )

    candidate_reasons: list[str] = []
    if candidate.lower <= delta_min:
        candidate_reasons.append("discovery lower bound has not exceeded delta_min")
    if confirmation is None:
        candidate_reasons.append("held-out confirmation is missing")
    elif not confirmation_passes:
        candidate_reasons.append("held-out confirmation gate has not passed")
    if candidate.harm_probability > maximum_harm_probability:
        candidate_reasons.append("posterior harm probability exceeds 0.05")
    if not bh_selected:
        candidate_reasons.append("rule was not selected by BH-FDR")
    if not version_revalidated:
        candidate_reasons.append("version requires revalidation")
    return SkillStatusDecision(
        SkillStatus.CANDIDATE,
        tuple(candidate_reasons) or ("awaiting additional evidence",),
    )
