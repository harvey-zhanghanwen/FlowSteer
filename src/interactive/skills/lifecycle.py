"""Skill activation delay, drift suspension, and explicit retirement."""

from __future__ import annotations

from typing import Iterable, Sequence

from ..versioning import VersionBundle
from .schema import SkillRecord, SkillStatus
from .validator import SkillEvidenceGate


class SkillLifecycleManager:
    VERSION_FIELDS = ("policy", "model_catalog", "evaluator", "prompt", "tool")

    def __init__(self, gate: SkillEvidenceGate | None = None) -> None:
        self.gate = gate or SkillEvidenceGate()

    def activate(self, skill: SkillRecord, current_epoch: int) -> SkillRecord:
        if skill.status is not SkillStatus.CANDIDATE:
            raise ValueError("only candidate Skills can be activated")
        if current_epoch < skill.eligible_epoch:
            raise ValueError("Skill cannot become visible in its discovery epoch")
        decision = self.gate.evaluate(skill)
        if not decision.approved:
            if decision.no_practical_value:
                return skill._with_status(SkillStatus.RETIRED, reason="no practical value")
            raise ValueError("Skill evidence gate failed: " + "; ".join(decision.reasons))
        if decision.receipt is None:
            raise ValueError("approved Skill gate did not produce a receipt")
        return skill._with_status(
            SkillStatus.ACTIVE,
            epoch=current_epoch,
            gate_config=self.gate.config.to_dict(),
            gate_receipt=decision.receipt,
        )

    def audit(
        self,
        skill: SkillRecord,
        current_versions: VersionBundle,
        *,
        recent_calibrated_lower: float | None = None,
        recent_terminal_gain: float | None = None,
        distribution_drifted: bool = False,
    ) -> SkillRecord:
        if skill.status is not SkillStatus.ACTIVE:
            return skill
        reasons: list[str] = []
        mismatches = skill.versions.mismatches(current_versions, self.VERSION_FIELDS)
        if mismatches:
            reasons.append("unvalidated version change: " + ",".join(sorted(mismatches)))
        if recent_calibrated_lower is not None and recent_calibrated_lower <= self.gate.config.delta_min:
            reasons.append("recent calibrated effect fell below delta_min")
        if recent_terminal_gain is not None and recent_terminal_gain < 0:
            reasons.append("recent terminal gain is negative")
        if distribution_drifted:
            reasons.append("task distribution drift detected")
        if reasons:
            return skill._with_status(SkillStatus.SUSPENDED, reason="; ".join(reasons))
        return skill

    def retire(self, skill: SkillRecord, reason: str) -> SkillRecord:
        if not reason.strip():
            raise ValueError("retirement reason must be non-empty")
        return skill._with_status(SkillStatus.RETIRED, reason=reason)
