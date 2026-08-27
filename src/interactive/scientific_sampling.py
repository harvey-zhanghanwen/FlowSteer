"""SkillFlow scientific sampling coordinates for Director rollouts.

This is a direct, dependency-light port of SkillFlow
``src/skillev/contracts/scientific_sampling.py`` and
``src/skillev/rollout/types.py::derive_generation_seed``.  The only adaptation
is importing this project's existing canonical JSON serializer so the
FlowSteer runtime does not depend on a second checkout at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final


SCIENTIFIC_SAMPLING_ALGORITHM: Final = "skillev-scientific-sampling@1"
_COORDINATE_FORMAT: Final = "skillev-scientific-sampling-coordinate@1"
_UINT64_LIMIT: Final = 2**64
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")


def _normalize_json(value: object) -> Any:
    """SkillFlow canonical JSON normalization used by its sampling protocol."""

    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid canonical JSON")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("object keys collide after Unicode normalization")
            normalized[normalized_key] = _normalize_json(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalize_json(item) for item in value]
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def stable_hash(value: object) -> str:
    """Return SkillFlow's prefixed canonical-content digest."""

    payload = json.dumps(
        _normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def scientific_sampling_schedule_hash(*, base_seed: int) -> str:
    """Return the protocol-level identity of the fixed sampling algorithm."""

    if type(base_seed) is not int or not 0 <= base_seed < _UINT64_LIMIT:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    return stable_hash(
        {
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
        }
    )


@dataclass(frozen=True, slots=True)
class ScientificSamplingCoordinate:
    """One result-affecting coordinate, deliberately excluding artifact IDs."""

    sampling_schedule_hash: str
    schedule_purpose: str
    ordered_sequence_hash: str
    sequence_position: int
    task_id: str
    optimizer_step_or_anchor_ordinal: int
    format: str = _COORDINATE_FORMAT

    def __post_init__(self) -> None:
        for field in ("sampling_schedule_hash", "ordered_sequence_hash"):
            value = getattr(self, field)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{field} must be a prefixed SHA-256 identifier")
        for field in ("schedule_purpose", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        if type(self.sequence_position) is not int or self.sequence_position < 0:
            raise ValueError("sequence_position must be non-negative")
        if (
            type(self.optimizer_step_or_anchor_ordinal) is not int
            or self.optimizer_step_or_anchor_ordinal < 0
        ):
            raise ValueError("optimizer_step_or_anchor_ordinal must be non-negative")
        if self.format != _COORDINATE_FORMAT:
            raise ValueError("unsupported scientific sampling coordinate format")

    def to_value(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "optimizer_step_or_anchor_ordinal": self.optimizer_step_or_anchor_ordinal,
            "ordered_sequence_hash": self.ordered_sequence_hash,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "schedule_purpose": self.schedule_purpose,
            "sequence_position": self.sequence_position,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object) -> "ScientificSamplingCoordinate":
        fields = {
            "format",
            "optimizer_step_or_anchor_ordinal",
            "ordered_sequence_hash",
            "sampling_schedule_hash",
            "schedule_purpose",
            "sequence_position",
            "task_id",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("scientific sampling coordinate has incompatible fields")
        return cls(
            sampling_schedule_hash=value["sampling_schedule_hash"],
            schedule_purpose=value["schedule_purpose"],
            ordered_sequence_hash=value["ordered_sequence_hash"],
            sequence_position=value["sequence_position"],
            task_id=value["task_id"],
            optimizer_step_or_anchor_ordinal=value[
                "optimizer_step_or_anchor_ordinal"
            ],
            format=value["format"],
        )


class GenerationPhase(StrEnum):
    """The two generation phases in SkillFlow's rollout protocol."""

    REASONING = "reasoning"
    ACTION = "action"


def derive_generation_seed(
    *,
    base_seed: int,
    coordinate: ScientificSamplingCoordinate,
    step_index: int,
    phase: GenerationPhase,
) -> int:
    """Derive a deterministic seed from scientific coordinates only."""

    if type(base_seed) is not int or not 0 <= base_seed < _UINT64_LIMIT:
        raise ValueError("base_seed must be an unsigned 64-bit integer")
    if not isinstance(coordinate, ScientificSamplingCoordinate):
        raise TypeError("coordinate must be ScientificSamplingCoordinate")
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must be positive")
    if not isinstance(phase, GenerationPhase):
        raise ValueError("phase must be a GenerationPhase")
    digest = stable_hash(
        {
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": phase.value,
            "step_index": step_index,
        }
    ).removeprefix("sha256:")
    return int(digest[:16], 16)


__all__ = [
    "GenerationPhase",
    "SCIENTIFIC_SAMPLING_ALGORITHM",
    "ScientificSamplingCoordinate",
    "derive_generation_seed",
    "scientific_sampling_schedule_hash",
    "stable_hash",
]
