"""Versionable model and provider catalog for AgentGraph execution.

The registry intentionally stores routing metadata only. Authentication and
other secrets belong to the gateway implementation, never to catalog records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import random
import re
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


class ModelRegistryError(ValueError):
    """Raised when a catalog entry or model selection is invalid."""


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{field_name} must be a non-empty string")
    return value.strip()


def _frozen_metadata(metadata: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    result = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ModelRegistryError("metadata keys and values must be strings")
        result.append((key, value))
    return tuple(sorted(result))


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INLINE_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "credentials",
    "password",
    "secret",
    "token",
}


def _reject_inline_secrets(value: object, path: str = "catalog") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower().replace("-", "_")
            if key in _INLINE_SECRET_KEYS:
                raise ModelRegistryError(
                    f"inline secret field {path}.{raw_key} is forbidden; use api_key_env"
                )
            _reject_inline_secrets(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{index}]")


def _catalog_entries(value: object, id_field: str) -> Tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        entries = []
        for entry_id, raw_entry in value.items():
            if not isinstance(raw_entry, Mapping):
                raise ModelRegistryError(f"{id_field} entry {entry_id!r} must be a mapping")
            entry = dict(raw_entry)
            existing_id = entry.get(id_field)
            if existing_id is not None and existing_id != entry_id:
                raise ModelRegistryError(f"{id_field} key and field disagree for {entry_id!r}")
            entry[id_field] = entry_id
            entries.append(entry)
        return tuple(entries)
    if isinstance(value, list):
        if not all(isinstance(entry, Mapping) for entry in value):
            raise ModelRegistryError(f"all {id_field} entries must be mappings")
        return tuple(dict(entry) for entry in value)  # type: ignore[arg-type]
    raise ModelRegistryError(f"catalog {id_field} entries must be a list or mapping")


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Non-secret description of a model provider endpoint."""

    provider_id: str
    kind: str = "generic"
    endpoint: Optional[str] = None
    max_concurrency: Optional[int] = None
    api_key_env: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _require_non_empty(self.provider_id, "provider_id"))
        object.__setattr__(self, "kind", _require_non_empty(self.kind, "kind"))
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str) or not self.endpoint.strip():
                raise ModelRegistryError("endpoint must be a non-empty string when supplied")
            object.__setattr__(self, "endpoint", self.endpoint.strip())
        if self.max_concurrency is not None and (
            type(self.max_concurrency) is not int or self.max_concurrency <= 0
        ):
            raise ModelRegistryError("max_concurrency must be positive when supplied")
        if self.api_key_env is not None:
            if not isinstance(self.api_key_env, str) or not _ENV_NAME.fullmatch(self.api_key_env):
                raise ModelRegistryError("api_key_env must be an environment-variable name")
        metadata = dict(_frozen_metadata(self.metadata))
        _reject_inline_secrets(metadata, "provider.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "max_concurrency": self.max_concurrency,
            "api_key_env": self.api_key_env,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A stable model arm and its relative routing weights.

    ``cheap_weight`` and ``fast_weight`` are positive desirability weights,
    not currency or latency measurements. Keeping them dimensionless makes
    seeded selection stable even when providers report incomparable metrics.
    """

    model_id: str
    provider_id: str
    model_name: str = ""
    cheap_weight: float = 1.0
    fast_weight: float = 1.0
    selection_weight: float = 1.0
    context_window: Optional[int] = None
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _require_non_empty(self.model_id, "model_id"))
        object.__setattr__(self, "provider_id", _require_non_empty(self.provider_id, "provider_id"))
        model_name = self.model_name.strip() if isinstance(self.model_name, str) else ""
        object.__setattr__(self, "model_name", model_name or self.model_id)
        for field_name in ("cheap_weight", "fast_weight", "selection_weight"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ModelRegistryError(f"{field_name} must be a finite positive number")
            object.__setattr__(self, field_name, float(value))
        if self.context_window is not None and (
            type(self.context_window) is not int or self.context_window <= 0
        ):
            raise ModelRegistryError("context_window must be positive when supplied")
        metadata = dict(_frozen_metadata(self.metadata))
        _reject_inline_secrets(metadata, "model.metadata")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_name": self.model_name,
            "cheap_weight": self.cheap_weight,
            "fast_weight": self.fast_weight,
            "selection_weight": self.selection_weight,
            "context_window": self.context_window,
            "metadata": dict(self.metadata),
        }


class ModelRegistry:
    """In-memory provider/model catalog with reproducible weighted selection."""

    def __init__(
        self,
        providers: Iterable[ProviderSpec] = (),
        models: Iterable[ModelSpec] = (),
    ) -> None:
        self._providers: Dict[str, ProviderSpec] = {}
        self._models: Dict[str, ModelSpec] = {}
        for provider in providers:
            self.register_provider(provider)
        for model in models:
            self.register_model(model)

    def register_provider(self, provider: ProviderSpec) -> None:
        if provider.provider_id in self._providers:
            raise ModelRegistryError(f"duplicate provider_id: {provider.provider_id}")
        self._providers[provider.provider_id] = provider

    def register_model(self, model: ModelSpec) -> None:
        if model.model_id in self._models:
            raise ModelRegistryError(f"duplicate model_id: {model.model_id}")
        if model.provider_id not in self._providers:
            raise ModelRegistryError(
                f"unknown provider_id {model.provider_id!r} for model {model.model_id!r}"
            )
        self._models[model.model_id] = model

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelRegistry":
        """Build a strict catalog from list- or ID-keyed provider/model entries."""

        if not isinstance(data, Mapping):
            raise ModelRegistryError("model catalog must be a mapping")
        unknown_top_level = set(data) - {"providers", "models"}
        if unknown_top_level:
            raise ModelRegistryError(
                f"unknown model catalog fields: {', '.join(sorted(unknown_top_level))}"
            )
        _reject_inline_secrets(data)
        provider_entries = _catalog_entries(data.get("providers"), "provider_id")
        model_entries = _catalog_entries(data.get("models"), "model_id")

        provider_fields = {
            "provider_id",
            "kind",
            "endpoint",
            "max_concurrency",
            "api_key_env",
            "metadata",
        }
        model_fields = {
            "model_id",
            "provider_id",
            "model_name",
            "cheap_weight",
            "fast_weight",
            "selection_weight",
            "context_window",
            "metadata",
        }
        providers = []
        for entry in provider_entries:
            unknown = set(entry) - provider_fields
            if unknown:
                raise ModelRegistryError(
                    f"unknown provider fields: {', '.join(sorted(unknown))}"
                )
            providers.append(ProviderSpec(**entry))  # type: ignore[arg-type]
        models = []
        for entry in model_entries:
            unknown = set(entry) - model_fields
            if unknown:
                raise ModelRegistryError(f"unknown model fields: {', '.join(sorted(unknown))}")
            models.append(ModelSpec(**entry))  # type: ignore[arg-type]
        return cls(providers=providers, models=models)

    @classmethod
    def from_yaml(cls, path: Union[str, "os.PathLike[str]"]) -> "ModelRegistry":
        """Load ``config/model_catalog.yaml`` without resolving credentials."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - repository dependency documents PyYAML
            raise ModelRegistryError("PyYAML is required to load a YAML model catalog") from exc
        with open(path, "r", encoding="utf-8") as handle:
            loaded: Any = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ModelRegistryError("YAML model catalog must contain a mapping")
        return cls.from_dict(loaded)

    def __contains__(self, model_id: object) -> bool:
        return isinstance(model_id, str) and model_id in self._models

    def __len__(self) -> int:
        return len(self._models)

    @property
    def model_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._models))

    @property
    def provider_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._providers))

    def require_model(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model_id: {model_id}") from exc

    def require_provider(self, provider_id: str) -> ProviderSpec:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown provider_id: {provider_id}") from exc

    def provider_for(self, model_id: str) -> ProviderSpec:
        model = self.require_model(model_id)
        return self.require_provider(model.provider_id)

    def select_weighted(
        self,
        *,
        seed: int,
        candidate_ids: Optional[Sequence[str]] = None,
        cheap_bias: float = 1.0,
        fast_bias: float = 1.0,
    ) -> ModelSpec:
        """Select reproducibly using base, cheap, and fast desirability weights.

        A zero bias ignores that dimension. Candidates are sorted by stable ID
        before sampling so caller ordering cannot change a seeded result.
        """

        for name, value in (("cheap_bias", cheap_bias), ("fast_bias", fast_bias)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ModelRegistryError(f"{name} must be a finite non-negative number")

        ids = self.model_ids if candidate_ids is None else tuple(sorted(set(candidate_ids)))
        if not ids:
            raise ModelRegistryError("at least one candidate model is required")
        candidates = [self.require_model(model_id) for model_id in ids]
        weights = [
            model.selection_weight
            * (model.cheap_weight ** float(cheap_bias))
            * (model.fast_weight ** float(fast_bias))
            for model in candidates
        ]
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0:
            raise ModelRegistryError("candidate selection weights are not usable")

        threshold = random.Random(seed).random() * total
        cumulative = 0.0
        for model, weight in zip(candidates, weights):
            cumulative += weight
            if threshold < cumulative:
                return model
        return candidates[-1]

    def select_cheap(self, *, seed: int, candidate_ids: Optional[Sequence[str]] = None) -> ModelSpec:
        return self.select_weighted(
            seed=seed,
            candidate_ids=candidate_ids,
            cheap_bias=1.0,
            fast_bias=0.0,
        )

    def select_fast(self, *, seed: int, candidate_ids: Optional[Sequence[str]] = None) -> ModelSpec:
        return self.select_weighted(
            seed=seed,
            candidate_ids=candidate_ids,
            cheap_bias=0.0,
            fast_bias=1.0,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "providers": [self._providers[key].to_dict() for key in self.provider_ids],
            "models": [self._models[key].to_dict() for key in self.model_ids],
        }

    @property
    def catalog_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ModelSpec",
    "ProviderSpec",
]
