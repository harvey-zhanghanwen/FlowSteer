"""Version fingerprints shared by rollout, exploration, and Skill records.

The Bayesian value target is policy- and executor-dependent.  Keeping all
versions in one immutable bundle makes accidental evidence mixing visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Iterable, Mapping


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VersionBundle:
    """Versions that define an execution/exploration regime."""

    policy: str
    model_catalog: str
    evaluator: str
    prompt: str
    tool: str
    encoder: str = "none"
    feature_schema: str = "none"
    posterior: str = "none"
    skill_library: str = "none"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"version field {name!r} must be a non-empty string")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()
        return f"versions_{digest[:20]}"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, str]) -> "VersionBundle":
        return cls(**dict(value))

    def mismatches(
        self,
        other: "VersionBundle",
        fields: Iterable[str] | None = None,
    ) -> Dict[str, tuple[str, str]]:
        names = tuple(fields) if fields is not None else tuple(asdict(self))
        result: Dict[str, tuple[str, str]] = {}
        for name in names:
            if not hasattr(self, name) or not hasattr(other, name):
                raise ValueError(f"unknown version field: {name}")
            left = getattr(self, name)
            right = getattr(other, name)
            if left != right:
                result[name] = (left, right)
        return result

    def is_compatible_with(
        self,
        other: "VersionBundle",
        fields: Iterable[str] | None = None,
    ) -> bool:
        return not self.mismatches(other, fields)
