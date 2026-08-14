"""Canonical serialization and content-addressed identifiers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def stable_id(prefix: str, value: Any, length: int = 24) -> str:
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("prefix must be alphanumeric with optional underscores")
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"
