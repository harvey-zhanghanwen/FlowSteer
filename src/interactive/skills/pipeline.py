"""Evidence-gated Skill publication composed from the existing project stores.

This module is deliberately an orchestration layer, not another Skill system.
It reuses :class:`TrajectoryRecord`, :class:`ProbeRecord`,
:class:`EvidenceStore`, :class:`SkillStore`, :class:`SkillEvidenceGate`,
:class:`SkillLifecycleManager`, and :class:`SkillRetriever` unchanged at their
ownership boundaries.

The separation of train and validation evidence follows SkillFlow
``evidence/schema.py::EvidenceRecord`` and
``evidence/store.py::SplitEvidenceStores``.  ACTIVE-only, task-conditioned
retrieval follows SkillFlow
``evolution/retriever.py::TaskConditionedSkillRetriever``.  The necessary
FlowSteer adaptation is the paired-effect publication protocol specified in
``FlowSteer_MACE_Bayesian_Skill_Design.md`` sections 10 and 11: a natural
trajectory may propose only a CANDIDATE; forced paired probes are excluded
from GRPO; independent validation problems alone support activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence

from ..persistence import EvidenceStore, canonical_json
from ..records import ProbeRecord, TrajectoryRecord
from ..versioning import VersionBundle
from .lifecycle import SkillLifecycleManager
from .retrieval import SkillQuery, SkillRetriever
from .schema import SkillEvidence, SkillRecord, SkillStatus
from .store import SkillStore
from .validator import GateDecision, SkillEvidenceGate, SkillGateConfig


SKILL_EVIDENCE_PROTOCOL = "flowsteer.skill-evidence.v1"
PROMPT_PRIOR_MODE = "rejectable_prompt_prior"


def _non_empty_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _json_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be JSON-compatible") from error
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError(f"{field_name} must be a non-empty object")
    return normalized


def _text_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value.strip() for value in normalized):
        raise ValueError(f"{field_name} cannot contain empty values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


@dataclass(frozen=True)
class StructuredSkillCandidate:
    """A detector-provided proposal; no rule is inferred from one trajectory."""

    skill_id: str
    condition: Mapping[str, Any]
    action: Mapping[str, Any]
    baseline_id: str
    baseline_action: Mapping[str, Any]
    failure_scope: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _non_empty_text(self.skill_id, field_name="skill_id")
        _non_empty_text(self.baseline_id, field_name="baseline_id")
        condition = _json_mapping(self.condition, field_name="condition")
        action = _json_mapping(self.action, field_name="action")
        baseline_action = _json_mapping(
            self.baseline_action,
            field_name="baseline_action",
        )
        failure_scope = _text_tuple(self.failure_scope, field_name="failure_scope")
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "baseline_action", baseline_action)
        object.__setattr__(self, "failure_scope", failure_scope)


@dataclass(frozen=True)
class SkillProbeEvidence:
    """One forced, same-prefix paired intervention and its execution regime."""

    probe: ProbeRecord
    condition: Mapping[str, Any]
    runtime_version: str
    model_catalog_version: str
    forced_probe: bool = field(default=True, init=False)
    grpo_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.probe, ProbeRecord):
            raise TypeError("probe must be a ProbeRecord")
        condition = _json_mapping(self.condition, field_name="condition")
        object.__setattr__(self, "condition", condition)
        _non_empty_text(self.runtime_version, field_name="runtime_version")
        _non_empty_text(
            self.model_catalog_version,
            field_name="model_catalog_version",
        )
        for field_name, value in (
            ("probe_id", self.probe.probe_id),
            ("problem_id", self.probe.problem_id),
            ("snapshot_id", self.probe.snapshot_id),
            ("policy_version", self.probe.policy_version),
            ("evaluator_version", self.probe.evaluator_version),
            ("feature_schema_version", self.probe.feature_schema_version),
        ):
            _non_empty_text(value, field_name=field_name)
        if not self.probe.state_features:
            raise ValueError("paired probe requires a decision-time state snapshot")
        branch_order = tuple(self.probe.branch_order)
        if len(branch_order) != 2 or set(branch_order) != {"incumbent", "candidate"}:
            raise ValueError(
                "branch_order must be a permutation of incumbent and candidate"
            )
        if not self.probe.executor_versions or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.probe.executor_versions.items()
        ):
            raise ValueError("paired probe requires explicit executor model versions")

    def to_probe_payload(self) -> dict[str, Any]:
        """Persist through the existing EvidenceStore probe stream."""

        payload = self.probe.to_dict()
        payload.update(
            {
                "condition": dict(self.condition),
                "evidence_protocol": SKILL_EVIDENCE_PROTOCOL,
                "forced_probe": self.forced_probe,
                "grpo_eligible": self.grpo_eligible,
                "model_catalog_version": self.model_catalog_version,
                "runtime_version": self.runtime_version,
            }
        )
        return payload


@dataclass(frozen=True)
class SkillValidationStatistics:
    """Pre-registered, problem-clustered interval and harm statistics."""

    calibrated_lower: float
    calibrated_upper: float
    empirical_coverage: float
    harm_probability: float
    heldout_task_families: Sequence[str]
    slice_effects: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.calibrated_lower,
            self.calibrated_upper,
            self.empirical_coverage,
            self.harm_probability,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("validation statistics must be finite")
        if self.calibrated_lower > self.calibrated_upper:
            raise ValueError("calibrated interval is reversed")
        if not 0 <= self.empirical_coverage <= 1:
            raise ValueError("empirical_coverage must be in [0, 1]")
        if not 0 <= self.harm_probability <= 1:
            raise ValueError("harm_probability must be in [0, 1]")
        families = _text_tuple(
            self.heldout_task_families,
            field_name="heldout_task_families",
        )
        if not families:
            raise ValueError("heldout_task_families cannot be empty")
        slices = {str(key): float(value) for key, value in self.slice_effects.items()}
        if any(not key.strip() for key in slices) or not all(
            math.isfinite(value) for value in slices.values()
        ):
            raise ValueError("slice_effects must have named finite values")
        object.__setattr__(self, "heldout_task_families", families)
        object.__setattr__(self, "slice_effects", slices)


@dataclass(frozen=True)
class SkillPublicationResult:
    skill: SkillRecord
    gate_decision: GateDecision

    @property
    def active(self) -> bool:
        return self.skill.status is SkillStatus.ACTIVE


@dataclass(frozen=True)
class PromptSkillPrior:
    """Director-visible suggestion with no Canvas mutation authority."""

    skill_id: str
    version: int
    content: str
    condition: Mapping[str, Any]
    action: Mapping[str, Any]
    application_mode: str = field(default=PROMPT_PRIOR_MODE, init=False)
    rejectable: bool = field(default=True, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "content": self.content,
            "condition": dict(self.condition),
            "action": dict(self.action),
            "application_mode": self.application_mode,
            "rejectable": self.rejectable,
        }


def render_validated_skill(skill: SkillRecord) -> str:
    """Render only gate-validated structured fields; no LLM is called."""

    if skill.status is not SkillStatus.ACTIVE or not skill.gate_receipt:
        raise ValueError("only an ACTIVE, gate-validated Skill can be rendered")
    evidence = skill.evidence
    failure_scope = canonical_json(list(skill.failure_scope))
    return (
        f"Optional validated Skill {skill.skill_id}@{skill.version}. "
        f"Condition={canonical_json(skill.to_dict()['condition'])}. "
        f"Suggested action={canonical_json(skill.to_dict()['action'])}. "
        f"Independent paired effect={evidence.paired_effect_mean:.12g}, "
        f"calibrated interval=[{evidence.calibrated_lower:.12g},"
        f"{evidence.calibrated_upper:.12g}], "
        f"effective problems={len(evidence.independent_problem_ids)}, "
        f"failure scope={failure_scope}. "
        "This is a prompt prior only; the Director may accept, modify, or reject it."
    )


class SkillEvidencePipeline:
    """Compose discovery, evidence, lifecycle, storage, and retrieval."""

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        skill_store: SkillStore,
        gate_config: SkillGateConfig | None = None,
        retrieval_top_k: int = 3,
    ) -> None:
        self.evidence_store = evidence_store
        self.skill_store = skill_store
        self.gate = SkillEvidenceGate(
            gate_config,
            evidence_lookup=evidence_store.resolve_probe,
        )
        self.lifecycle = SkillLifecycleManager(self.gate)
        self.retriever = SkillRetriever(top_k=retrieval_top_k)

    def discover(
        self,
        trajectory: TrajectoryRecord,
        proposal: StructuredSkillCandidate,
        *,
        created_epoch: int,
        runtime_version: str,
        executor_versions: Mapping[str, str],
    ) -> SkillRecord:
        """Persist a structured CANDIDATE without treating success as evidence."""

        if not isinstance(trajectory, TrajectoryRecord):
            raise TypeError("trajectory must be a TrajectoryRecord")
        if trajectory.task.split != "train":
            raise ValueError("Skill discovery requires a train problem")
        if trajectory.forced_probe:
            raise ValueError("forced probes are paired evidence, not discovery trajectories")
        if not trajectory.natural_policy_terminal or not trajectory.evaluation.valid:
            raise ValueError("Skill discovery requires a complete, valid natural trajectory")
        if trajectory.evaluation.evaluator_version != trajectory.versions.evaluator:
            raise ValueError("trajectory evaluator receipt does not match its version bundle")
        runtime_version = _non_empty_text(
            runtime_version,
            field_name="runtime_version",
        )
        normalized_executors = self._executor_versions(executor_versions)
        self._require_action_model(proposal.action, normalized_executors)
        provenance = {
            "baseline_action": dict(proposal.baseline_action),
            "candidate_source": "natural_trajectory",
            "evidence_protocol": SKILL_EVIDENCE_PROTOCOL,
            "evidence_stage": "discovery_only",
            "executor_versions": normalized_executors,
            "model_catalog_version": trajectory.versions.model_catalog,
            "runtime_version": runtime_version,
            "source_problem_ids": [trajectory.task.task_id],
            "source_trajectory_ids": [trajectory.trajectory_id],
        }
        evidence = SkillEvidence(
            baseline=proposal.baseline_id,
            paired_effect_mean=0.0,
            calibrated_lower=0.0,
            # Wide placeholder interval means "unknown", not "no value".
            # It cannot pass the gate because no independent paired evidence
            # exists, but it also cannot be prematurely retired as harmful.
            calibrated_upper=1.0,
            effective_pairs=0,
            independent_problem_ids=(),
            discovery_problem_ids=(trajectory.task.task_id,),
            validation_problem_ids=(),
            validation_splits=(),
            heldout_task_families=(),
            empirical_coverage=0.0,
            harm_probability=1.0,
            evidence_ids=(),
        )
        candidate = SkillRecord(
            skill_id=proposal.skill_id,
            version=1,
            status=SkillStatus.CANDIDATE,
            condition=proposal.condition,
            action=proposal.action,
            evidence=evidence,
            versions=trajectory.versions,
            failure_scope=proposal.failure_scope,
            provenance=provenance,
            created_epoch=created_epoch,
            eligible_epoch=created_epoch + 1,
        )
        self.skill_store.upsert(candidate)
        return candidate

    def confirm_and_publish(
        self,
        candidate: SkillRecord,
        *,
        discovery_probes: Sequence[SkillProbeEvidence],
        validation_probes: Sequence[SkillProbeEvidence],
        statistics: SkillValidationStatistics,
        validation_epoch: int,
        activation_epoch: int,
    ) -> SkillPublicationResult:
        """Use independent validation probes to gate the next Skill version."""

        self._require_pipeline_candidate(candidate)
        discovery = tuple(discovery_probes)
        validation = tuple(validation_probes)
        if not discovery:
            raise ValueError("publication requires paired discovery evidence")
        if not validation:
            raise ValueError("publication requires independent validation evidence")
        for probe in discovery:
            self._validate_probe(candidate, probe, expected_split="train")
        for probe in validation:
            self._validate_probe(candidate, probe, expected_split="validation")
        self._require_unique_probe_and_problem_ids(discovery, label="discovery")
        self._require_unique_probe_and_problem_ids(validation, label="validation")

        discovery_problem_ids = {
            *candidate.evidence.discovery_problem_ids,
            *(probe.probe.problem_id for probe in discovery),
        }
        validation_problem_ids = {probe.probe.problem_id for probe in validation}
        overlap = discovery_problem_ids & validation_problem_ids
        if overlap:
            raise ValueError(
                "discovery and independent validation problems overlap: "
                + ", ".join(sorted(overlap))
            )

        paired_effects = tuple(probe.probe.paired_effect for probe in validation)
        paired_effect_mean = sum(paired_effects) / len(paired_effects)
        if not (
            statistics.calibrated_lower
            <= paired_effect_mean
            <= statistics.calibrated_upper
        ):
            raise ValueError("paired-effect mean lies outside the calibrated interval")
        task_family = str(candidate.condition["task_family"])
        if (
            task_family != "*"
            and task_family not in statistics.heldout_task_families
        ):
            raise ValueError("held-out task families do not cover the Skill condition")

        # Validate the complete batch before any append, then use the existing
        # append-only evidence stream for both train and validation probes.
        for probe in (*discovery, *validation):
            self.evidence_store.append_probe(probe.to_probe_payload())

        validation_ids = tuple(probe.probe.problem_id for probe in validation)
        evidence_ids = tuple(probe.probe.probe_id for probe in validation)
        evidence = SkillEvidence(
            baseline=candidate.evidence.baseline,
            paired_effect_mean=paired_effect_mean,
            calibrated_lower=statistics.calibrated_lower,
            calibrated_upper=statistics.calibrated_upper,
            effective_pairs=len(validation_ids),
            independent_problem_ids=validation_ids,
            discovery_problem_ids=tuple(sorted(discovery_problem_ids)),
            validation_problem_ids=validation_ids,
            validation_splits=("validation",),
            heldout_task_families=statistics.heldout_task_families,
            empirical_coverage=statistics.empirical_coverage,
            harm_probability=statistics.harm_probability,
            slice_effects=statistics.slice_effects,
            evidence_ids=evidence_ids,
        )
        provenance = dict(candidate.to_dict()["provenance"])
        provenance.update(
            {
                "discovery_probe_ids": [probe.probe.probe_id for probe in discovery],
                "evidence_stage": "independent_validation",
                "validation_probe_ids": list(evidence_ids),
            }
        )
        confirmed = SkillRecord(
            skill_id=candidate.skill_id,
            version=candidate.version + 1,
            status=SkillStatus.CANDIDATE,
            condition=candidate.to_dict()["condition"],
            action=candidate.to_dict()["action"],
            evidence=evidence,
            versions=candidate.versions,
            failure_scope=candidate.failure_scope,
            provenance=provenance,
            created_epoch=validation_epoch,
            eligible_epoch=validation_epoch + 1,
        )
        decision = self.gate.evaluate(confirmed)
        if (decision.approved or decision.no_practical_value) and (
            activation_epoch < confirmed.eligible_epoch
        ):
            raise ValueError("validated Skill cannot be published in its evidence epoch")
        self.skill_store.upsert(confirmed)
        if decision.approved or decision.no_practical_value:
            published = self.lifecycle.activate(confirmed, activation_epoch)
            self.skill_store.upsert(published)
        else:
            published = confirmed
        return SkillPublicationResult(skill=published, gate_decision=decision)

    def retrieve_prompt_priors(
        self,
        query: SkillQuery,
        versions: VersionBundle,
    ) -> tuple[PromptSkillPrior, ...]:
        """Return rejectable prompt priors; never apply actions to a Canvas."""

        skills = self.retriever.retrieve(self.skill_store.list(), query, versions)
        return tuple(
            PromptSkillPrior(
                skill_id=skill.skill_id,
                version=skill.version,
                content=render_validated_skill(skill),
                condition=skill.to_dict()["condition"],
                action=skill.to_dict()["action"],
            )
            for skill in skills
        )

    @staticmethod
    def _executor_versions(value: Mapping[str, str]) -> dict[str, str]:
        normalized = _json_mapping(value, field_name="executor_versions")
        if any(
            not isinstance(model_id, str)
            or not model_id.strip()
            or not isinstance(version, str)
            or not version.strip()
            for model_id, version in normalized.items()
        ):
            raise ValueError("executor_versions must map model IDs to non-empty versions")
        return {str(key): str(item) for key, item in normalized.items()}

    @staticmethod
    def _require_action_model(
        action: Mapping[str, Any],
        executor_versions: Mapping[str, str],
    ) -> None:
        model_id = action.get("model_id")
        if model_id is not None and model_id not in executor_versions:
            raise ValueError("candidate action model has no bound executor version")

    @staticmethod
    def _require_pipeline_candidate(candidate: SkillRecord) -> None:
        if not isinstance(candidate, SkillRecord):
            raise TypeError("candidate must be a SkillRecord")
        if candidate.status is not SkillStatus.CANDIDATE:
            raise ValueError("only a CANDIDATE Skill can receive evidence")
        if candidate.provenance.get("evidence_protocol") != SKILL_EVIDENCE_PROTOCOL:
            raise ValueError("candidate was not created by this evidence protocol")

    def _validate_probe(
        self,
        candidate: SkillRecord,
        evidence: SkillProbeEvidence,
        *,
        expected_split: str,
    ) -> None:
        if not isinstance(evidence, SkillProbeEvidence):
            raise TypeError("paired evidence must be SkillProbeEvidence")
        probe = evidence.probe
        if probe.task_split != expected_split:
            raise ValueError(f"{expected_split} evidence has the wrong task split")
        if evidence.condition != candidate.to_dict()["condition"]:
            raise ValueError("paired probe condition does not match candidate")
        if probe.candidate_action != candidate.to_dict()["action"]:
            raise ValueError("paired probe candidate action does not match candidate")
        if probe.incumbent_action != candidate.provenance.get("baseline_action"):
            raise ValueError("paired probe incumbent action does not match baseline")
        if probe.policy_version != candidate.versions.policy:
            raise ValueError("paired probe policy version does not match candidate")
        if probe.evaluator_version != candidate.versions.evaluator:
            raise ValueError("paired probe evaluator version does not match candidate")
        if probe.feature_schema_version != candidate.versions.feature_schema:
            raise ValueError("paired probe feature schema does not match candidate")
        if evidence.model_catalog_version != candidate.versions.model_catalog:
            raise ValueError("paired probe model catalog does not match candidate")
        if evidence.runtime_version != candidate.provenance.get("runtime_version"):
            raise ValueError("paired probe runtime version does not match candidate")
        expected_executors = candidate.provenance.get("executor_versions")
        if dict(probe.executor_versions) != expected_executors:
            raise ValueError("paired probe executor versions do not match candidate")
        self._require_action_model(candidate.action, probe.executor_versions)

    @staticmethod
    def _require_unique_probe_and_problem_ids(
        evidence: Sequence[SkillProbeEvidence],
        *,
        label: str,
    ) -> None:
        probe_ids = tuple(item.probe.probe_id for item in evidence)
        problem_ids = tuple(item.probe.problem_id for item in evidence)
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError(f"{label} evidence repeats a probe ID")
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError(
                f"{label} evidence must count each complete problem only once"
            )


__all__ = [
    "PROMPT_PRIOR_MODE",
    "SKILL_EVIDENCE_PROTOCOL",
    "PromptSkillPrior",
    "SkillEvidencePipeline",
    "SkillProbeEvidence",
    "SkillPublicationResult",
    "SkillValidationStatistics",
    "StructuredSkillCandidate",
    "render_validated_skill",
]
