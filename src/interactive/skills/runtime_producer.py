"""Produce version-bound Skills from ledger and held-out paired evidence.

This is the missing runtime bridge between the dynamic Combination Posterior
and the existing deterministic Skill lifecycle.  It follows Sections 7 and 8
of ``LatentLoss_Implementation_Spec.md``:

* discovery evidence is made only of train-split same-snapshot probes;
* confirmation evidence is made only of validation-split probes, is disjoint
  by complete problem from discovery, and is bound to the publication policy;
* Benjamini--Hochberg is applied before activation;
* publication still goes through ``SkillEvidenceGate`` and
  ``SkillLifecycleManager``;
* the result exposes no reward and cannot contribute a Skill/probe signal to
  the Action-Masked One-Pass GRPO objective.

FlowSteer upstream supplies the terminal AgentGraph trajectory boundary and
SkillFlow supplies the persisted Skill/workspace lifecycle pattern.  The
paired-evidence producer itself is a project adaptation required by the two
design documents; neither upstream implements this statistical gate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..exploration.combination_ledger import CombinationPosterior, DecisionKey
from ..exploration.ledger_probe import validate_probe_record
from ..persistence.ids import stable_id
from ..records import ProbeRecord
from ..versioning import VersionBundle
from .ledger_miner import (
    ConfirmationEvidence,
    RuleContrast,
    SkillRuleCandidate,
    benjamini_hochberg,
    calibrated_interval,
    decide_skill_status,
    empirical_calibration_quantile,
    enumerate_rule_candidates,
    normal_probability_at_or_below,
)
from .lifecycle import SkillLifecycleManager
from .schema import SkillEvidence, SkillRecord, SkillStatus
from .validator import SkillEvidenceGate, SkillGateConfig


SKILL_RUNTIME_UPDATE_SCHEMA = "flowsteer.skill-runtime-update.v1"
_SUPPORTED_ACTION_FIELDS = frozenset({"model_id", "edge_type"})


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class SkillRuntimeProducerConfig:
    """Predeclared Skill discovery, confirmation, and publication thresholds."""

    delta_min: float = 0.05
    minimum_discovery_pairs: int = 10
    minimum_confirmation_problems: int = 20
    calibration_alpha: float = 0.05
    benjamini_hochberg_fdr: float = 0.10
    maximum_harm_probability: float = 0.05
    minimum_empirical_coverage: float = 0.90
    minimum_positive_slice_fraction: float = 0.75
    retire_after_suspended_epochs: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.delta_min)) or self.delta_min <= 0.0:
            raise ValueError("delta_min must be finite and positive")
        for name in (
            "minimum_discovery_pairs",
            "minimum_confirmation_problems",
            "retire_after_suspended_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "calibration_alpha",
            "benjamini_hochberg_fdr",
            "maximum_harm_probability",
            "minimum_empirical_coverage",
            "minimum_positive_slice_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.calibration_alpha >= 1.0:
            raise ValueError("calibration_alpha must be less than one")
        if self.benjamini_hochberg_fdr >= 1.0:
            raise ValueError("benjamini_hochberg_fdr must be less than one")

    def gate_config(self) -> SkillGateConfig:
        return SkillGateConfig(
            delta_min=self.delta_min,
            max_harm_probability=self.maximum_harm_probability,
            minimum_independent_problems=self.minimum_confirmation_problems,
            minimum_effective_pairs=self.minimum_confirmation_problems,
            minimum_empirical_coverage=self.minimum_empirical_coverage,
            minimum_positive_slice_fraction=self.minimum_positive_slice_fraction,
        )


@dataclass(frozen=True)
class SkillRuntimeUpdateReceipt:
    """Fail-closed outcome consumed by the epoch runner's long-training gate."""

    status: str
    producer_complete: bool
    discovery_epoch: int
    publication_epoch: int
    discovery_probe_count: int
    confirmation_probe_count: int
    discovery_qualified_rule_count: int
    active_rule_count: int
    pending_confirmation_rule_ids: Sequence[str]
    unsupported_discovery_probe_count: int
    unsupported_confirmation_probe_count: int
    calibration_quantile: float | None
    skill_status_counts: Mapping[str, int]
    evidence_policy_versions: Sequence[str]
    publication_policy_version: str
    grpo_reward_contribution: float = 0.0
    ttb_enabled: bool = False
    schema_version: str = SKILL_RUNTIME_UPDATE_SCHEMA

    def __post_init__(self) -> None:
        allowed = {
            "complete",
            "complete_no_active_skill",
            "complete_no_qualified_candidate",
            "pending_heldout_confirmation",
        }
        if self.status not in allowed:
            raise ValueError(f"unsupported Skill producer status: {self.status}")
        if type(self.producer_complete) is not bool:
            raise TypeError("producer_complete must be bool")
        if self.producer_complete != (self.status != "pending_heldout_confirmation"):
            raise ValueError("producer_complete is inconsistent with status")
        if self.discovery_epoch < 0 or self.publication_epoch != self.discovery_epoch + 1:
            raise ValueError("publication_epoch must immediately follow discovery_epoch")
        for name in (
            "discovery_probe_count",
            "confirmation_probe_count",
            "discovery_qualified_rule_count",
            "active_rule_count",
            "unsupported_discovery_probe_count",
            "unsupported_confirmation_probe_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.grpo_reward_contribution != 0.0:
            raise ValueError("Skill producer cannot contribute to GRPO reward")
        if self.ttb_enabled:
            raise ValueError("TTB must remain disabled")
        pending = tuple(str(value) for value in self.pending_confirmation_rule_ids)
        if any(not value for value in pending) or len(pending) != len(set(pending)):
            raise ValueError("pending confirmation rule IDs must be unique and non-empty")
        policies = tuple(str(value) for value in self.evidence_policy_versions)
        if any(not value for value in policies) or len(policies) != len(set(policies)):
            raise ValueError("evidence policy versions must be unique and non-empty")
        expected = {status.value for status in SkillStatus}
        counts = {str(key): int(value) for key, value in self.skill_status_counts.items()}
        if set(counts) != expected or any(value < 0 for value in counts.values()):
            raise ValueError("skill_status_counts must contain all lifecycle states")
        object.__setattr__(self, "pending_confirmation_rule_ids", pending)
        object.__setattr__(self, "evidence_policy_versions", policies)
        object.__setattr__(self, "skill_status_counts", MappingProxyType(counts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "producer_complete": self.producer_complete,
            "discovery_epoch": self.discovery_epoch,
            "publication_epoch": self.publication_epoch,
            "discovery_probe_count": self.discovery_probe_count,
            "confirmation_probe_count": self.confirmation_probe_count,
            "discovery_qualified_rule_count": self.discovery_qualified_rule_count,
            "active_rule_count": self.active_rule_count,
            "pending_confirmation_rule_ids": list(
                self.pending_confirmation_rule_ids
            ),
            "unsupported_discovery_probe_count": (
                self.unsupported_discovery_probe_count
            ),
            "unsupported_confirmation_probe_count": (
                self.unsupported_confirmation_probe_count
            ),
            "calibration_quantile": self.calibration_quantile,
            "skill_status_counts": dict(self.skill_status_counts),
            "evidence_policy_versions": list(self.evidence_policy_versions),
            "publication_policy_version": self.publication_policy_version,
            "grpo_reward_contribution": self.grpo_reward_contribution,
            "ttb_enabled": self.ttb_enabled,
        }


@dataclass(frozen=True)
class SkillRuntimeUpdate:
    """Next Skill snapshot plus its resolvable held-out evidence and receipt."""

    skills: Sequence[SkillRecord]
    evidence_records: Mapping[str, Mapping[str, Any]]
    receipt: SkillRuntimeUpdateReceipt

    def __post_init__(self) -> None:
        skills = tuple(self.skills)
        if len({skill.skill_id for skill in skills}) != len(skills):
            raise ValueError("Skill runtime update contains duplicate skill IDs")
        records = {
            str(key): MappingProxyType(_json_copy(value))
            for key, value in self.evidence_records.items()
        }
        if any(not key for key in records):
            raise ValueError("evidence record IDs must be non-empty")
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "evidence_records", MappingProxyType(records))

    def evidence_lookup(self, evidence_id: str) -> Mapping[str, Any] | None:
        return self.evidence_records.get(evidence_id)


@dataclass(frozen=True)
class _ObservedRule:
    rule: RuleContrast
    keep: DecisionKey
    switch: DecisionKey
    changed_field: str
    discovery_probes: tuple[ProbeRecord, ...]


def _probe_signature(record: ProbeRecord) -> str:
    return json.dumps(
        {
            "keep": dict(record.incumbent_action),
            "switch": dict(record.candidate_action),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _probe_noise_variance(record: ProbeRecord, variance_floor: float) -> float:
    keep = np.asarray(tuple(record.incumbent_returns), dtype=np.float64)
    switch = np.asarray(tuple(record.candidate_returns), dtype=np.float64)
    keep_rate = float(np.mean(keep))
    switch_rate = float(np.mean(switch))
    return (
        switch_rate * (1.0 - switch_rate)
        + keep_rate * (1.0 - keep_rate)
    ) / len(keep) + float(variance_floor)


def _validate_probe_collections(
    discovery_probes: Sequence[ProbeRecord],
    confirmation_probes: Sequence[ProbeRecord],
    publication_versions: VersionBundle,
) -> None:
    all_ids: set[str] = set()
    for record in (*discovery_probes, *confirmation_probes):
        if not isinstance(record, ProbeRecord):
            raise TypeError("Skill evidence collections must contain ProbeRecord values")
        validate_probe_record(record)
        if not record.probe_id.strip() or record.probe_id in all_ids:
            raise ValueError("Skill evidence probe IDs must be unique and non-empty")
        all_ids.add(record.probe_id)
        if record.evaluator_version != publication_versions.evaluator:
            raise ValueError("probe evaluator version does not match publication versions")
        if record.feature_schema_version != publication_versions.feature_schema:
            raise ValueError("probe feature schema does not match publication versions")
    if any(record.task_split != "train" for record in discovery_probes):
        raise ValueError("Skill discovery accepts only train-split probes")
    if any(record.task_split != "validation" for record in confirmation_probes):
        raise ValueError("Skill confirmation accepts only validation-split probes")
    if any(
        record.policy_version != publication_versions.policy
        for record in confirmation_probes
    ):
        raise ValueError("confirmation probes must use the publication policy version")
    discovery_problem_ids = {record.problem_id for record in discovery_probes}
    confirmation_problem_ids = {record.problem_id for record in confirmation_probes}
    overlap = discovery_problem_ids & confirmation_problem_ids
    if overlap:
        raise ValueError("Skill discovery and held-out confirmation problems overlap")


def _rule_action(
    changed_field: str,
    switch: DecisionKey,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if changed_field == "model_id":
        return {"model_id": switch.model_id}, {"model_id": switch.model_id}
    if changed_field == "edge_type":
        contrast_action = {"edge_type": switch.edge_type}
        skill_action = {
            "relation": {
                "edge_type": switch.edge_type,
                "same_model_as_upstream": switch.same_model_as_upstream,
            }
        }
        return contrast_action, skill_action
    raise ValueError(f"unsupported Skill action field: {changed_field}")


def _observed_rules(
    ledger: CombinationPosterior,
    discovery_probes: Sequence[ProbeRecord],
    *,
    minimum_pairs: int,
) -> tuple[tuple[_ObservedRule, ...], int]:
    grouped: dict[str, list[ProbeRecord]] = defaultdict(list)
    unsupported = 0
    for record in discovery_probes:
        changed = validate_probe_record(record).changed_field
        if changed not in _SUPPORTED_ACTION_FIELDS:
            unsupported += 1
            continue
        grouped[_probe_signature(record)].append(record)

    observed: list[_ObservedRule] = []
    # Work on a detached posterior because contrast reads may register levels.
    detached = CombinationPosterior.from_state_dict(ledger.state_dict())
    for signature, records in sorted(grouped.items()):
        if len(records) < minimum_pairs:
            continue
        first = records[0]
        keep = DecisionKey.from_dict(first.incumbent_action)
        switch = DecisionKey.from_dict(first.candidate_action)
        changed = validate_probe_record(first).changed_field
        estimate = detached.predict_contrast(switch, keep)
        vector = detached.contrast_vector(switch, keep)
        columns = detached.columns
        interaction_activated = any(
            "&" in name and not math.isclose(float(vector[index]), 0.0)
            for name, index in columns.items()
        )
        contrast_action, _ = _rule_action(changed, switch)
        condition = {
            "task_family": keep.task_family,
            "role_cluster": keep.role_cluster,
            "stage": keep.stage,
            "baseline_model_id": keep.model_id,
            "baseline_edge_type": keep.edge_type,
            "baseline_same_model_as_upstream": keep.same_model_as_upstream,
        }
        rule_id = stable_id(
            "ledger_skill_rule",
            {
                "condition": condition,
                "action": contrast_action,
                "keep": keep.to_dict(),
                "switch": switch.to_dict(),
            },
        )
        observed.append(
            _ObservedRule(
                rule=RuleContrast(
                    rule_id=rule_id,
                    order=2 if interaction_activated else 1,
                    condition=condition,
                    action=contrast_action,
                    delta_mean=estimate.mean,
                    posterior_variance=estimate.variance,
                    n_eff_pairs=len(records),
                    interaction_activated=interaction_activated,
                ),
                keep=keep,
                switch=switch,
                changed_field=changed,
                discovery_probes=tuple(records),
            )
        )
    return tuple(observed), unsupported


def _matching_confirmation(
    observed: _ObservedRule,
    confirmation_probes: Sequence[ProbeRecord],
) -> tuple[ProbeRecord, ...]:
    expected = json.dumps(
        {"keep": observed.keep.to_dict(), "switch": observed.switch.to_dict()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return tuple(
        record for record in confirmation_probes if _probe_signature(record) == expected
    )


def _calibration_quantile(
    ledger: CombinationPosterior,
    observed_rules: Sequence[_ObservedRule],
    confirmation_by_rule: Mapping[str, Sequence[ProbeRecord]],
    *,
    alpha: float,
) -> float | None:
    observed_deltas: list[float] = []
    predicted_means: list[float] = []
    posterior_variances: list[float] = []
    noise_variances: list[float] = []
    for observed in observed_rules:
        estimate = ledger.predict_contrast(observed.switch, observed.keep)
        for record in confirmation_by_rule.get(observed.rule.rule_id, ()):
            observed_deltas.append(record.paired_effect)
            predicted_means.append(estimate.mean)
            posterior_variances.append(estimate.variance)
            noise_variances.append(
                _probe_noise_variance(record, ledger.variance_floor)
            )
    if not observed_deltas:
        return None
    return empirical_calibration_quantile(
        observed_deltas,
        predicted_means,
        posterior_variances,
        noise_variances,
        alpha=alpha,
    )


def _confirmation_statistics(
    candidate: SkillRuleCandidate,
    records: Sequence[ProbeRecord],
    *,
    calibration_quantile: float,
    delta_min: float,
    variance_floor: float,
) -> tuple[ConfirmationEvidence, dict[str, Any]]:
    effects = np.asarray([record.paired_effect for record in records], dtype=np.float64)
    mean = float(np.mean(effects))
    measurement_variance = sum(
        _probe_noise_variance(record, variance_floor) for record in records
    ) / (len(records) ** 2)
    between_variance = (
        float(np.var(effects, ddof=1)) / len(records) if len(records) > 1 else 0.0
    )
    mean_variance = max(measurement_variance, between_variance)
    lower, upper = calibrated_interval(mean, mean_variance, calibration_quantile)
    harm_probability = normal_probability_at_or_below(
        -delta_min,
        mean,
        mean_variance,
    )
    total_variance = candidate.rule.posterior_variance + np.asarray(
        [_probe_noise_variance(record, variance_floor) for record in records],
        dtype=np.float64,
    )
    covered = np.abs(effects - candidate.rule.delta_mean) <= (
        calibration_quantile * np.sqrt(total_variance)
    )
    coverage = float(np.mean(covered))
    problem_ids = tuple(sorted({record.problem_id for record in records}))
    return (
        ConfirmationEvidence(
            calibrated_lower=lower,
            problem_ids=problem_ids,
            harm_probability=harm_probability,
        ),
        {
            "mean": mean,
            "variance": mean_variance,
            "lower": lower,
            "upper": upper,
            "harm_probability": harm_probability,
            "empirical_coverage": coverage,
            "problem_ids": problem_ids,
        },
    )


def _skill_status_counts(skills: Iterable[SkillRecord]) -> dict[str, int]:
    counts = {status.value: 0 for status in SkillStatus}
    for skill in skills:
        counts[skill.status.value] += 1
    return counts


def produce_skill_runtime_update(
    ledger: CombinationPosterior,
    *,
    discovery_probes: Sequence[ProbeRecord],
    confirmation_probes: Sequence[ProbeRecord],
    publication_versions: VersionBundle,
    discovery_epoch: int,
    current_skills: Sequence[SkillRecord] = (),
    config: SkillRuntimeProducerConfig | None = None,
    suspended_epoch_counts: Mapping[str, int] | None = None,
) -> SkillRuntimeUpdate:
    """Produce the next immutable Skill snapshot from real paired evidence.

    ``ledger`` must already contain the cumulative comparison posterior at the
    publication policy boundary.  ``discovery_probes`` supplies actual counts
    and problem lineage; it may span older policy epochs because the ledger's
    policy refresh handles non-stationarity.  Held-out confirmation must use
    exactly ``publication_versions.policy``.
    """

    if not isinstance(ledger, CombinationPosterior):
        raise TypeError("ledger must be CombinationPosterior")
    if not isinstance(publication_versions, VersionBundle):
        raise TypeError("publication_versions must be VersionBundle")
    if ledger.policy_version != publication_versions.policy:
        raise ValueError("ledger policy does not match publication versions")
    if isinstance(discovery_epoch, bool) or not isinstance(discovery_epoch, int):
        raise TypeError("discovery_epoch must be an integer")
    if discovery_epoch < 0 or ledger.epoch != discovery_epoch + 1:
        raise ValueError("ledger must be the immediately refreshed publication epoch")
    settings = config or SkillRuntimeProducerConfig()
    discovery = tuple(discovery_probes)
    confirmation = tuple(confirmation_probes)
    existing = tuple(current_skills)
    if len({skill.skill_id for skill in existing}) != len(existing):
        raise ValueError("current_skills contains duplicate skill IDs")
    _validate_probe_collections(discovery, confirmation, publication_versions)

    observed_rules, unsupported_discovery = _observed_rules(
        ledger,
        discovery,
        minimum_pairs=settings.minimum_discovery_pairs,
    )
    supported_confirmation = tuple(
        record
        for record in confirmation
        if validate_probe_record(record).changed_field in _SUPPORTED_ACTION_FIELDS
    )
    unsupported_confirmation = len(confirmation) - len(supported_confirmation)
    confirmation_by_rule = {
        observed.rule.rule_id: _matching_confirmation(
            observed, supported_confirmation
        )
        for observed in observed_rules
    }
    quantile = _calibration_quantile(
        ledger,
        observed_rules,
        confirmation_by_rule,
        alpha=settings.calibration_alpha,
    )
    interval_quantile = (
        ledger.calibration_quantile if quantile is None else quantile
    )
    candidates = enumerate_rule_candidates(
        (observed.rule for observed in observed_rules),
        calibration_quantile=interval_quantile,
        delta_min=settings.delta_min,
        minimum_pairs=settings.minimum_discovery_pairs,
    )
    observed_by_id = {observed.rule.rule_id: observed for observed in observed_rules}
    selected_by_bh = benjamini_hochberg(
        {candidate.rule_id: candidate.activation_p_value for candidate in candidates},
        fdr=settings.benjamini_hochberg_fdr,
    )

    evidence_records: dict[str, Mapping[str, Any]] = {}
    pending: list[str] = []
    generated: dict[str, SkillRecord] = {}
    existing_by_id = {skill.skill_id: skill for skill in existing}
    gate_config = settings.gate_config()

    for candidate in candidates:
        observed = observed_by_id[candidate.rule_id]
        matching = tuple(confirmation_by_rule[candidate.rule_id])
        unique_confirmation_problems = {record.problem_id for record in matching}
        has_confirmation = (
            len(unique_confirmation_problems)
            >= settings.minimum_confirmation_problems
        )
        confirmation_evidence: ConfirmationEvidence | None = None
        confirmation_stats: dict[str, Any] | None = None
        if has_confirmation:
            if quantile is None:
                raise AssertionError("confirmed rule has no calibration quantile")
            confirmation_evidence, confirmation_stats = _confirmation_statistics(
                candidate,
                matching,
                calibration_quantile=quantile,
                delta_min=settings.delta_min,
                variance_floor=ledger.variance_floor,
            )
        else:
            pending.append(candidate.rule_id)

        decision = decide_skill_status(
            candidate,
            confirmation_evidence,
            previous=SkillStatus.CANDIDATE,
            bh_selected=candidate.rule_id in selected_by_bh,
            version_revalidated=has_confirmation,
            delta_min=settings.delta_min,
            maximum_harm_probability=settings.maximum_harm_probability,
            minimum_confirmation_problems=settings.minimum_confirmation_problems,
        )
        _, skill_action = _rule_action(observed.changed_field, observed.switch)
        previous = existing_by_id.get(candidate.rule_id)
        version = 1 if previous is None else previous.version + 1
        discovery_problem_ids = tuple(
            sorted({record.problem_id for record in observed.discovery_probes})
        )

        if confirmation_stats is None:
            skill_evidence = SkillEvidence(
                baseline=json.dumps(
                    observed.keep.to_dict(), sort_keys=True, separators=(",", ":")
                ),
                paired_effect_mean=candidate.rule.delta_mean,
                calibrated_lower=candidate.lower,
                calibrated_upper=candidate.upper,
                effective_pairs=len(observed.discovery_probes),
                independent_problem_ids=(),
                discovery_problem_ids=discovery_problem_ids,
                validation_problem_ids=(),
                validation_splits=(),
                heldout_task_families=(),
                empirical_coverage=0.0,
                harm_probability=candidate.harm_probability,
                evidence_ids=(),
            )
        else:
            for record in matching:
                evidence_records[record.probe_id] = record.to_dict()
            validation_problem_ids = tuple(confirmation_stats["problem_ids"])
            skill_evidence = SkillEvidence(
                baseline=json.dumps(
                    observed.keep.to_dict(), sort_keys=True, separators=(",", ":")
                ),
                paired_effect_mean=float(confirmation_stats["mean"]),
                calibrated_lower=float(confirmation_stats["lower"]),
                calibrated_upper=float(confirmation_stats["upper"]),
                effective_pairs=len(matching),
                independent_problem_ids=validation_problem_ids,
                discovery_problem_ids=discovery_problem_ids,
                validation_problem_ids=validation_problem_ids,
                validation_splits=("validation",),
                heldout_task_families=(observed.keep.task_family,),
                empirical_coverage=float(confirmation_stats["empirical_coverage"]),
                harm_probability=float(confirmation_stats["harm_probability"]),
                slice_effects={
                    observed.keep.task_family: float(confirmation_stats["mean"])
                },
                evidence_ids=tuple(record.probe_id for record in matching),
            )

        record = SkillRecord(
            skill_id=candidate.rule_id,
            version=version,
            status=SkillStatus.CANDIDATE,
            condition={
                "task_family": observed.keep.task_family,
                "role_cluster": observed.keep.role_cluster,
                "graph_stage": observed.keep.stage,
                "baseline_model_id": observed.keep.model_id,
                "baseline_edge_type": observed.keep.edge_type,
                "baseline_same_model_as_upstream": (
                    observed.keep.same_model_as_upstream
                ),
            },
            action=skill_action,
            evidence=skill_evidence,
            versions=publication_versions,
            provenance={
                "producer": SKILL_RUNTIME_UPDATE_SCHEMA,
                "changed_field": observed.changed_field,
                "discovery_probe_ids": tuple(
                    record.probe_id for record in observed.discovery_probes
                ),
                "discovery_policy_versions": tuple(
                    sorted(
                        {
                            record.policy_version
                            for record in observed.discovery_probes
                        }
                    )
                ),
                "confirmation_probe_ids": tuple(
                    record.probe_id for record in matching
                ),
                "discovery_interval": {
                    "mean": candidate.rule.delta_mean,
                    "lower": candidate.lower,
                    "upper": candidate.upper,
                    "n_eff_pairs": candidate.rule.n_eff_pairs,
                },
                "calibration_quantile": quantile,
                "bh_selected": candidate.rule_id in selected_by_bh,
            },
            created_epoch=discovery_epoch,
            eligible_epoch=discovery_epoch + 1,
        )
        if decision.status is SkillStatus.ACTIVE:
            gate = SkillEvidenceGate(
                gate_config,
                evidence_lookup=lambda evidence_id, records=evidence_records: records.get(
                    evidence_id
                ),
            )
            record = SkillLifecycleManager(gate).activate(
                record,
                discovery_epoch + 1,
            )
        elif decision.status is SkillStatus.RETIRED:
            record = record._with_status(
                SkillStatus.RETIRED,
                reason="; ".join(decision.reasons),
            )
        generated[record.skill_id] = record

    suspended_counts = {
        str(key): int(value) for key, value in (suspended_epoch_counts or {}).items()
    }
    carried: dict[str, SkillRecord] = {}
    lifecycle = SkillLifecycleManager(SkillEvidenceGate(gate_config))
    for skill in existing:
        if skill.skill_id in generated:
            continue
        updated = lifecycle.audit(skill, publication_versions)
        if updated.status is SkillStatus.SUSPENDED:
            count = suspended_counts.get(skill.skill_id, 0) + 1
            if count >= settings.retire_after_suspended_epochs:
                updated = lifecycle.retire(
                    updated,
                    "suspended for three consecutive epochs without recovery",
                )
        carried[updated.skill_id] = updated
    carried.update(generated)
    # ``retired`` is terminal: later evidence cannot silently resurrect the
    # same stable rule ID.  A substantively different rule receives a different
    # ID from its condition/action/contrast tuple.
    for skill in existing:
        if skill.status is SkillStatus.RETIRED and skill.skill_id in generated:
            carried[skill.skill_id] = skill
    next_skills = tuple(sorted(carried.values(), key=lambda value: value.skill_id))

    if not candidates:
        status = "complete_no_qualified_candidate"
    elif pending:
        status = "pending_heldout_confirmation"
    elif any(skill.status is SkillStatus.ACTIVE for skill in generated.values()):
        status = "complete"
    else:
        status = "complete_no_active_skill"
    counts = _skill_status_counts(next_skills)
    receipt = SkillRuntimeUpdateReceipt(
        status=status,
        producer_complete=status != "pending_heldout_confirmation",
        discovery_epoch=discovery_epoch,
        publication_epoch=discovery_epoch + 1,
        discovery_probe_count=len(discovery),
        confirmation_probe_count=len(confirmation),
        discovery_qualified_rule_count=len(candidates),
        active_rule_count=counts[SkillStatus.ACTIVE.value],
        pending_confirmation_rule_ids=tuple(sorted(pending)),
        unsupported_discovery_probe_count=unsupported_discovery,
        unsupported_confirmation_probe_count=unsupported_confirmation,
        calibration_quantile=quantile,
        skill_status_counts=counts,
        evidence_policy_versions=tuple(
            sorted({record.policy_version for record in discovery})
        ),
        publication_policy_version=publication_versions.policy,
    )
    return SkillRuntimeUpdate(
        skills=next_skills,
        evidence_records=evidence_records,
        receipt=receipt,
    )


__all__ = [
    "SKILL_RUNTIME_UPDATE_SCHEMA",
    "SkillRuntimeProducerConfig",
    "SkillRuntimeUpdate",
    "SkillRuntimeUpdateReceipt",
    "produce_skill_runtime_update",
]
