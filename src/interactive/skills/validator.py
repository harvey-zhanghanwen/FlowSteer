"""Deterministic evidence gates; an LLM cannot self-publish a Skill."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping, Optional, Tuple

from ..persistence.ids import stable_id
from .schema import SkillRecord


EvidenceLookup = Callable[[str], Optional[Mapping[str, object]]]


@dataclass(frozen=True)
class SkillGateConfig:
    delta_min: float = 0.03
    max_harm_probability: float = 0.05
    minimum_independent_problems: int = 20
    minimum_effective_pairs: int = 20
    minimum_empirical_coverage: float = 0.90
    minimum_positive_slice_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.delta_min <= 0:
            raise ValueError("delta_min must be predeclared and positive")
        if not 0 <= self.max_harm_probability <= 1:
            raise ValueError("max_harm_probability must be in [0, 1]")
        if self.minimum_independent_problems < 1 or self.minimum_effective_pairs < 1:
            raise ValueError("minimum evidence counts must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    no_practical_value: bool
    reasons: Tuple[str, ...]
    receipt: Optional[str] = None


class SkillEvidenceGate:
    def __init__(
        self,
        config: SkillGateConfig | None = None,
        *,
        evidence_lookup: EvidenceLookup | None = None,
    ) -> None:
        self.config = config or SkillGateConfig()
        self.evidence_lookup = evidence_lookup

    def compute_receipt(self, skill: SkillRecord) -> str:
        return stable_id(
            "skill_gate",
            {
                "config": self.config.to_dict(),
                "skill_id": skill.skill_id,
                "version": skill.version,
                "condition": skill.to_dict()["condition"],
                "action": skill.to_dict()["action"],
                "evidence": skill.evidence.to_dict(),
                "versions": skill.versions.to_dict(),
                "failure_scope": list(skill.failure_scope),
                # Runtime/executor identities live in provenance because the
                # shared VersionBundle intentionally describes the Director
                # execution regime.  Binding provenance here prevents an
                # otherwise valid receipt from being replayed after those
                # Skill-evidence identities change.
                "provenance": skill.to_dict()["provenance"],
            },
        )

    def evaluate(self, skill: SkillRecord) -> GateDecision:
        evidence = skill.evidence
        reasons: list[str] = []

        validation = set(evidence.validation_problem_ids)
        if set(evidence.validation_splits) != {"validation"}:
            reasons.append("confirmation evidence must come only from validation split")
        independent = set(evidence.independent_problem_ids)
        if len(independent) < self.config.minimum_independent_problems:
            reasons.append("too few independent validation problems")
        if evidence.effective_pairs < self.config.minimum_effective_pairs:
            reasons.append("too few effective paired interventions")
        if evidence.calibrated_lower <= self.config.delta_min:
            reasons.append("calibrated lower bound does not exceed delta_min")
        if evidence.harm_probability > self.config.max_harm_probability:
            reasons.append("negative-transfer probability exceeds harm limit")
        if evidence.empirical_coverage < self.config.minimum_empirical_coverage:
            reasons.append("held-out interval coverage is below target")
        if evidence.slice_effects:
            positive = sum(value > self.config.delta_min for value in evidence.slice_effects.values())
            fraction = positive / len(evidence.slice_effects)
            if fraction < self.config.minimum_positive_slice_fraction:
                reasons.append("effect direction is inconsistent across task slices")

        resolved: list[Mapping[str, object]] = []
        if not evidence.evidence_ids:
            reasons.append("no persisted paired-probe evidence IDs")
        elif self.evidence_lookup is None:
            reasons.append("no evidence resolver configured for publication")
        else:
            for evidence_id in evidence.evidence_ids:
                try:
                    record = self.evidence_lookup(evidence_id)
                except Exception:
                    record = None
                if record is None:
                    reasons.append(f"unresolved evidence ID: {evidence_id}")
                else:
                    resolved.append(record)
        if resolved:
            resolved_problems = {str(record.get("problem_id", "")) for record in resolved}
            if "" in resolved_problems:
                reasons.append("resolved evidence is missing problem_id")
            if resolved_problems != validation or resolved_problems != independent:
                reasons.append("resolved evidence problems do not match validation/independent IDs")
            if len(resolved) != evidence.effective_pairs:
                reasons.append("effective_pairs does not match resolved probe count")
            effects: list[float] = []
            for record in resolved:
                if record.get("task_split") != "validation":
                    reasons.append("publication evidence must resolve to validation probes")
                if record.get("policy_version") != skill.versions.policy:
                    reasons.append("probe policy version does not match Skill")
                if record.get("evaluator_version") != skill.versions.evaluator:
                    reasons.append("probe evaluator version does not match Skill")
                if record.get("feature_schema_version") != skill.versions.feature_schema:
                    reasons.append("probe feature schema does not match Skill")
                try:
                    effect = float(record["paired_effect"])
                except (KeyError, TypeError, ValueError):
                    reasons.append("resolved evidence has no finite paired_effect")
                    continue
                if not math.isfinite(effect):
                    reasons.append("resolved evidence has no finite paired_effect")
                else:
                    effects.append(effect)

            protocol = skill.provenance.get("evidence_protocol")
            if protocol is not None:
                if protocol != "flowsteer.skill-evidence.v1":
                    reasons.append("unsupported Skill evidence protocol")
                expected_runtime = skill.provenance.get("runtime_version")
                expected_executors = skill.provenance.get("executor_versions")
                if not isinstance(expected_runtime, str) or not expected_runtime.strip():
                    reasons.append("Skill provenance is missing runtime version")
                if not isinstance(expected_executors, Mapping) or not expected_executors:
                    reasons.append("Skill provenance is missing executor model versions")
                expected_condition = skill.to_dict()["condition"]
                expected_action = skill.to_dict()["action"]
                for record in resolved:
                    if record.get("evidence_protocol") != protocol:
                        reasons.append("probe evidence protocol does not match Skill")
                    if record.get("forced_probe") is not True:
                        reasons.append("Skill evidence must be a forced paired probe")
                    if record.get("grpo_eligible") is not False:
                        reasons.append("forced Skill probe must be excluded from GRPO")
                    if record.get("condition") != expected_condition:
                        reasons.append("probe condition does not match Skill")
                    if record.get("candidate_action") != expected_action:
                        reasons.append("probe candidate action does not match Skill")
                    if record.get("runtime_version") != expected_runtime:
                        reasons.append("probe runtime version does not match Skill")
                    if record.get("model_catalog_version") != skill.versions.model_catalog:
                        reasons.append("probe model catalog version does not match Skill")
                    if record.get("executor_versions") != expected_executors:
                        reasons.append("probe executor model versions do not match Skill")
            if effects:
                resolved_mean = sum(effects) / len(effects)
                if not math.isclose(
                    resolved_mean,
                    evidence.paired_effect_mean,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    reasons.append("paired_effect_mean does not match resolved probes")

        no_value = evidence.calibrated_upper < self.config.delta_min
        approved = not reasons
        return GateDecision(
            approved=approved,
            no_practical_value=no_value,
            reasons=tuple(dict.fromkeys(reasons)),
            receipt=self.compute_receipt(skill) if approved else None,
        )
