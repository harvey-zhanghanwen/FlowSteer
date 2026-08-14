"""Conservative retrieval of active, applicable, version-compatible Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Optional, Sequence

from ..versioning import VersionBundle
from .schema import SkillRecord, SkillStatus
from .validator import SkillEvidenceGate, SkillGateConfig


@dataclass(frozen=True)
class SkillQuery:
    task_family: str
    graph_stage: str
    tags: Sequence[str] = field(default_factory=tuple)
    available_models: Sequence[str] = field(default_factory=tuple)
    current_epoch: int = 0


class SkillRetriever:
    VERSION_FIELDS = ("policy", "model_catalog", "evaluator", "prompt", "tool")

    def __init__(self, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def retrieve(
        self,
        skills: Iterable[SkillRecord],
        query: SkillQuery,
        versions: VersionBundle,
    ) -> list[SkillRecord]:
        ranked: list[tuple[float, SkillRecord]] = []
        query_tags = set(query.tags)
        available_models = set(query.available_models)
        for skill in skills:
            if skill.status is not SkillStatus.ACTIVE:
                continue
            if skill.activated_epoch is None or skill.activated_epoch > query.current_epoch:
                continue
            try:
                gate = SkillEvidenceGate(SkillGateConfig(**dict(skill.gate_config)))
            except (TypeError, ValueError):
                continue
            if skill.gate_receipt != gate.compute_receipt(skill):
                continue
            if not skill.versions.is_compatible_with(versions, self.VERSION_FIELDS):
                continue
            condition = skill.condition
            family = condition.get("task_family")
            if family not in ("*", query.task_family):
                continue
            stage = condition.get("graph_stage")
            if stage not in ("*", query.graph_stage):
                continue
            required_tags = set(condition.get("tags", ()))
            if required_tags and not required_tags.issubset(query_tags):
                continue
            model_id = skill.action.get("model_id")
            if model_id and model_id not in available_models:
                continue
            if query.task_family in set(skill.failure_scope):
                continue
            tag_bonus = len(required_tags & query_tags) * 0.01
            evidence_score = skill.evidence.calibrated_lower + 0.005 * math.log1p(
                skill.evidence.effective_pairs
            )
            ranked.append((evidence_score + tag_bonus, skill))
        ranked.sort(key=lambda item: (-item[0], item[1].skill_id))
        return [skill for _, skill in ranked[: self.top_k]]
