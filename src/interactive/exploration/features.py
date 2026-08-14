"""MACE's compact nine-dimensional relational feature representation.

The paper specifies the feature equations but leaves text preprocessing and
normalization underspecified.  This module makes both choices explicit and
versioned.  Version ``unicode-word-casefold-v1`` uses NFKC normalization,
Unicode-aware word tokens, and case folding.  ``l2-bound-v1`` divides a vector
by ``max(1, ||x||_2)`` so it satisfies the bounded-feature assumption.  The
seeded tie-breaking policy handles any equal initial confidence bonuses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Mapping, Sequence, Tuple

import numpy as np

from .records import RoundSnapshot


FEATURE_NAMES: Tuple[str, ...] = (
    "diversity_unigram",
    "diversity_bigram",
    "diversity_trigram",
    "distinctiveness_unigram",
    "distinctiveness_bigram",
    "distinctiveness_trigram",
    "historical_reward",
    "normalized_round",
    "bias",
)

FEATURE_DIMENSION = len(FEATURE_NAMES)
FEATURE_SCHEMA_VERSION = "mace-relational-9d-v1"
PREPROCESSING_VERSION = "unicode-word-casefold-v1"
NORMALIZATION_NONE = "none-v1"
NORMALIZATION_L2_BOUND = "l2-bound-v1"

_WORD_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['\u2019][^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class MACEFeatureConfig:
    """Versioned choices needed to reproduce a relational feature vector."""

    schema_version: str = FEATURE_SCHEMA_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    normalization_version: str = NORMALIZATION_L2_BOUND
    empty_jaccard_similarity: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema: {self.schema_version}")
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError(f"unsupported preprocessing: {self.preprocessing_version}")
        if self.normalization_version not in {
            NORMALIZATION_NONE,
            NORMALIZATION_L2_BOUND,
        }:
            raise ValueError(f"unsupported feature normalization: {self.normalization_version}")
        if not 0.0 <= self.empty_jaccard_similarity <= 1.0:
            raise ValueError("empty_jaccard_similarity must lie in [0, 1]")

    @property
    def version_id(self) -> str:
        return ":".join(
            (
                self.schema_version,
                self.preprocessing_version,
                self.normalization_version,
                f"empty={self.empty_jaccard_similarity:g}",
            )
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "MACEFeatureConfig":
        return cls(**dict(state))


def preprocess_text(text: str, version: str = PREPROCESSING_VERSION) -> Tuple[str, ...]:
    """Convert response text to the tokens used by all n-gram features."""

    if version != PREPROCESSING_VERSION:
        raise ValueError(f"unsupported preprocessing: {version}")
    if not isinstance(text, str):
        raise TypeError("response text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _WORD_RE.finditer(normalized))


def ngram_set(tokens: Sequence[str], n: int) -> frozenset[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[start : start + n]) for start in range(len(tokens) - n + 1))


def jaccard_similarity(
    left: frozenset[Tuple[str, ...]],
    right: frozenset[Tuple[str, ...]],
    *,
    empty_similarity: float = 1.0,
) -> float:
    union = left | right
    if not union:
        return float(empty_similarity)
    return len(left & right) / len(union)


class MACEFeatureExtractor:
    """Build exact 9D MACE features from an immutable response snapshot."""

    dimension = FEATURE_DIMENSION
    names = FEATURE_NAMES

    def __init__(self, config: MACEFeatureConfig | None = None) -> None:
        self.config = config or MACEFeatureConfig()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if self.config.normalization_version == NORMALIZATION_NONE:
            return vector
        norm = float(np.linalg.norm(vector))
        return vector / max(1.0, norm)

    @staticmethod
    def _historical_matrix(
        n_agents: int,
        historical_rewards: np.ndarray | Mapping[tuple[int, int], float] | None,
    ) -> np.ndarray:
        if historical_rewards is None:
            return np.zeros((n_agents, n_agents), dtype=np.float64)
        if isinstance(historical_rewards, Mapping):
            matrix = np.zeros((n_agents, n_agents), dtype=np.float64)
            for (querier, peer), value in historical_rewards.items():
                if not (0 <= querier < n_agents and 0 <= peer < n_agents):
                    raise IndexError("historical reward pair is outside the agent pool")
                matrix[querier, peer] = float(value)
        else:
            matrix = np.asarray(historical_rewards, dtype=np.float64)
            if matrix.shape != (n_agents, n_agents):
                raise ValueError(
                    f"historical reward matrix must have shape {(n_agents, n_agents)}"
                )
            matrix = matrix.copy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("historical rewards must be finite")
        return matrix

    def build_raw(
        self,
        snapshot: RoundSnapshot,
        historical_rewards: np.ndarray | Mapping[tuple[int, int], float] | None = None,
    ) -> np.ndarray:
        """Return raw paper-defined features with shape ``[N, N, 9]``."""

        selection_round = snapshot.selection_round
        n_agents = snapshot.n_agents
        histories = self._historical_matrix(n_agents, historical_rewards)
        tokenized = [
            preprocess_text(response, self.config.preprocessing_version)
            for response in snapshot.responses
        ]

        similarities: dict[int, np.ndarray] = {}
        distinctiveness: dict[int, np.ndarray] = {}
        for n in (1, 2, 3):
            sets = [ngram_set(tokens, n) for tokens in tokenized]
            similarity = np.empty((n_agents, n_agents), dtype=np.float64)
            for left in range(n_agents):
                for right in range(left, n_agents):
                    value = jaccard_similarity(
                        sets[left],
                        sets[right],
                        empty_similarity=self.config.empty_jaccard_similarity,
                    )
                    similarity[left, right] = value
                    similarity[right, left] = value
            similarities[n] = similarity
            divergence = np.sum(1.0 - similarity, axis=1)
            total_divergence = float(np.sum(divergence))
            if total_divergence == 0.0:
                distinctiveness[n] = np.zeros(n_agents, dtype=np.float64)
            else:
                distinctiveness[n] = divergence / total_divergence

        features = np.empty((n_agents, n_agents, FEATURE_DIMENSION), dtype=np.float64)
        normalized_round = selection_round / snapshot.total_rounds
        for querier in range(n_agents):
            for peer in range(n_agents):
                features[querier, peer] = (
                    1.0 - similarities[1][querier, peer],
                    1.0 - similarities[2][querier, peer],
                    1.0 - similarities[3][querier, peer],
                    distinctiveness[1][peer],
                    distinctiveness[2][peer],
                    distinctiveness[3][peer],
                    histories[querier, peer],
                    normalized_round,
                    1.0,
                )
        features.setflags(write=False)
        return features

    def build(
        self,
        snapshot: RoundSnapshot,
        historical_rewards: np.ndarray | Mapping[tuple[int, int], float] | None = None,
    ) -> np.ndarray:
        """Return configured, normalized features with shape ``[N, N, 9]``."""

        raw = self.build_raw(snapshot, historical_rewards)
        if self.config.normalization_version == NORMALIZATION_NONE:
            return raw
        features = np.empty_like(raw)
        for querier in range(snapshot.n_agents):
            for peer in range(snapshot.n_agents):
                features[querier, peer] = self._normalize(raw[querier, peer])
        features.setflags(write=False)
        return features

    def for_pair(
        self,
        snapshot: RoundSnapshot,
        querier: int,
        peer: int,
        historical_rewards: np.ndarray | Mapping[tuple[int, int], float] | None = None,
    ) -> np.ndarray:
        if not (0 <= querier < snapshot.n_agents and 0 <= peer < snapshot.n_agents):
            raise IndexError("ordered pair is outside the agent pool")
        vector = np.array(
            self.build(snapshot, historical_rewards)[querier, peer], copy=True
        )
        vector.setflags(write=False)
        return vector
