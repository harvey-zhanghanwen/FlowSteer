"""Disjoint ordered-pair LinUCB and a thin MACE orchestration facade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .features import FEATURE_DIMENSION, MACEFeatureConfig, MACEFeatureExtractor
from .records import LinUCBSelection, RoundSnapshot


LINUCB_STATE_VERSION = "disjoint-ordered-pair-linucb-v1"
MACE_STATE_VERSION = "mace-controller-v1"


def blended_mace_reward(score_before: float, score_after: float) -> float:
    """MACE reward: half improvement plus half post-interaction quality."""

    before = float(score_before)
    after = float(score_after)
    if not (np.isfinite(before) and np.isfinite(after)):
        raise ValueError("scores must be finite")
    return 0.5 * ((after - before) + after)


class DisjointLinUCB:
    """Independent LinUCB sufficient statistics for every ordered agent pair.

    The implementation never materializes ``A^-1``.  Both ridge parameters and
    uncertainty quadratic forms are computed with linear solves.  Peer ties
    are broken with the instance's seeded random generator rather than agent
    index order.
    """

    def __init__(
        self,
        n_agents: int,
        dimension: int = FEATURE_DIMENSION,
        *,
        regularization: float = 1.0,
        exploration_alpha: float = 1.0,
        update_enabled: bool = True,
        allow_self: bool = True,
        seed: int | None = None,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if regularization <= 0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and positive")
        if exploration_alpha < 0 or not np.isfinite(exploration_alpha):
            raise ValueError("exploration_alpha must be finite and non-negative")
        self.n_agents = int(n_agents)
        self.dimension = int(dimension)
        self.regularization = float(regularization)
        self.exploration_alpha = float(exploration_alpha)
        self.update_enabled = bool(update_enabled)
        self.allow_self = bool(allow_self)
        identity = self.regularization * np.eye(self.dimension, dtype=np.float64)
        self._precision = np.broadcast_to(
            identity, (self.n_agents, self.n_agents, self.dimension, self.dimension)
        ).copy()
        self._information = np.zeros(
            (self.n_agents, self.n_agents, self.dimension), dtype=np.float64
        )
        self._counts = np.zeros((self.n_agents, self.n_agents), dtype=np.int64)
        self._reward_means = np.zeros(
            (self.n_agents, self.n_agents), dtype=np.float64
        )
        self._rng = np.random.default_rng(seed)

    @property
    def counts(self) -> np.ndarray:
        result = self._counts.copy()
        result.setflags(write=False)
        return result

    @property
    def reward_means(self) -> np.ndarray:
        result = self._reward_means.copy()
        result.setflags(write=False)
        return result

    @property
    def precision(self) -> np.ndarray:
        result = self._precision.copy()
        result.setflags(write=False)
        return result

    @property
    def information(self) -> np.ndarray:
        result = self._information.copy()
        result.setflags(write=False)
        return result

    def set_update_enabled(self, enabled: bool) -> None:
        self.update_enabled = bool(enabled)

    def set_exploration_alpha(self, alpha: float) -> None:
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("exploration alpha must be finite and non-negative")
        self.exploration_alpha = float(alpha)

    def _validate_pair(self, querier: int, peer: int) -> None:
        if not (0 <= querier < self.n_agents and 0 <= peer < self.n_agents):
            raise IndexError("ordered pair is outside the agent pool")

    def _contexts(self, contexts: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(contexts, dtype=np.float64)
        expected = (self.n_agents, self.dimension)
        if matrix.shape != expected:
            raise ValueError(f"contexts must have shape {expected}, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("contexts must be finite")
        return matrix

    def _candidates(
        self,
        querier: int,
        candidate_mask: Iterable[bool] | np.ndarray | None,
    ) -> np.ndarray:
        if candidate_mask is None:
            mask = np.ones(self.n_agents, dtype=bool)
        else:
            mask = np.asarray(candidate_mask, dtype=bool)
            if mask.shape != (self.n_agents,):
                raise ValueError(
                    f"candidate mask must have shape {(self.n_agents,)}"
                )
            mask = mask.copy()
        if not self.allow_self:
            mask[querier] = False
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            raise ValueError("no eligible peers remain after applying the candidate mask")
        return candidates

    def pair_parameter(self, querier: int, peer: int) -> np.ndarray:
        self._validate_pair(querier, peer)
        parameter = np.linalg.solve(
            self._precision[querier, peer], self._information[querier, peer]
        )
        parameter.setflags(write=False)
        return parameter

    def pair_epistemic_variance(
        self,
        querier: int,
        peer: int,
        context: Iterable[float] | np.ndarray,
    ) -> float:
        self._validate_pair(querier, peer)
        vector = np.asarray(context, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"context must be finite with shape {(self.dimension,)}")
        solved = np.linalg.solve(self._precision[querier, peer], vector)
        return max(0.0, float(vector @ solved))

    def select(
        self,
        querier: int,
        contexts: Iterable[Iterable[float]] | np.ndarray,
        candidate_mask: Iterable[bool] | np.ndarray | None = None,
    ) -> LinUCBSelection:
        """Select one peer without mutating posterior state."""

        if not 0 <= querier < self.n_agents:
            raise IndexError("querier is outside the agent pool")
        matrix = self._contexts(contexts)
        candidates = self._candidates(querier, candidate_mask)
        means = np.empty(self.n_agents, dtype=np.float64)
        variances = np.empty(self.n_agents, dtype=np.float64)
        for peer in range(self.n_agents):
            parameter = np.linalg.solve(
                self._precision[querier, peer], self._information[querier, peer]
            )
            solved_context = np.linalg.solve(
                self._precision[querier, peer], matrix[peer]
            )
            means[peer] = matrix[peer] @ parameter
            variances[peer] = max(0.0, float(matrix[peer] @ solved_context))
        bonuses = self.exploration_alpha * np.sqrt(variances)
        scores = means + bonuses
        best_score = float(np.max(scores[candidates]))
        tied = candidates[
            np.isclose(scores[candidates], best_score, rtol=1e-12, atol=1e-12)
        ]
        peer = int(self._rng.choice(tied))
        return LinUCBSelection(
            querier=querier,
            peer=peer,
            context=matrix[peer],
            predicted_mean=float(means[peer]),
            epistemic_variance=float(variances[peer]),
            exploration_bonus=float(bonuses[peer]),
            score=float(scores[peer]),
            candidate_scores=scores,
        )

    def select_round(
        self,
        features: np.ndarray,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[LinUCBSelection, ...]:
        """Select for every agent from one feature tensor before any update."""

        feature_tensor = np.asarray(features, dtype=np.float64)
        expected = (self.n_agents, self.n_agents, self.dimension)
        if feature_tensor.shape != expected:
            raise ValueError(f"features must have shape {expected}")
        if not np.all(np.isfinite(feature_tensor)):
            raise ValueError("features must be finite")
        if candidate_mask is not None:
            mask = np.asarray(candidate_mask, dtype=bool)
            if mask.shape != (self.n_agents, self.n_agents):
                raise ValueError(
                    f"round candidate mask must have shape {(self.n_agents, self.n_agents)}"
                )
        else:
            mask = None
        return tuple(
            self.select(
                querier,
                feature_tensor[querier],
                None if mask is None else mask[querier],
            )
            for querier in range(self.n_agents)
        )

    def update(
        self,
        querier: int,
        peer: int,
        context: Iterable[float] | np.ndarray,
        reward: float,
    ) -> bool:
        """Update a selected ordered pair; return whether mutation occurred."""

        self._validate_pair(querier, peer)
        vector = np.asarray(context, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"context must be finite with shape {(self.dimension,)}")
        value = float(reward)
        if not np.isfinite(value):
            raise ValueError("reward must be finite")
        if not self.update_enabled:
            return False
        self._precision[querier, peer] += np.outer(vector, vector)
        self._information[querier, peer] += value * vector
        count = int(self._counts[querier, peer]) + 1
        mean = self._reward_means[querier, peer]
        self._counts[querier, peer] = count
        self._reward_means[querier, peer] = mean + (value - mean) / count
        return True

    def update_batch(
        self,
        selections: Sequence[LinUCBSelection],
        rewards: Iterable[float] | np.ndarray,
    ) -> int:
        reward_vector = np.asarray(tuple(rewards), dtype=np.float64)
        if reward_vector.shape != (len(selections),):
            raise ValueError("one reward is required for each selection")
        updates = 0
        for selection, reward in zip(selections, reward_vector):
            updates += int(
                self.update(
                    selection.querier,
                    selection.peer,
                    selection.context,
                    float(reward),
                )
            )
        return updates

    def state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, versioned state dictionary."""

        return {
            "version": LINUCB_STATE_VERSION,
            "n_agents": self.n_agents,
            "dimension": self.dimension,
            "regularization": self.regularization,
            "exploration_alpha": self.exploration_alpha,
            "update_enabled": self.update_enabled,
            "allow_self": self.allow_self,
            "precision": self._precision.tolist(),
            "information": self._information.tolist(),
            "counts": self._counts.tolist(),
            "reward_means": self._reward_means.tolist(),
            "rng_state": self._rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "DisjointLinUCB":
        if state.get("version") != LINUCB_STATE_VERSION:
            raise ValueError(f"unsupported LinUCB state: {state.get('version')}")
        policy = cls(
            int(state["n_agents"]),
            int(state["dimension"]),
            regularization=float(state["regularization"]),
            exploration_alpha=float(state["exploration_alpha"]),
            update_enabled=bool(state["update_enabled"]),
            allow_self=bool(state["allow_self"]),
        )
        expected_precision = (
            policy.n_agents,
            policy.n_agents,
            policy.dimension,
            policy.dimension,
        )
        precision = np.asarray(state["precision"], dtype=np.float64)
        information = np.asarray(state["information"], dtype=np.float64)
        counts = np.asarray(state["counts"], dtype=np.int64)
        rewards = np.asarray(state["reward_means"], dtype=np.float64)
        if precision.shape != expected_precision:
            raise ValueError("serialized precision has the wrong shape")
        if information.shape != expected_precision[:3]:
            raise ValueError("serialized information has the wrong shape")
        if counts.shape != expected_precision[:2] or rewards.shape != expected_precision[:2]:
            raise ValueError("serialized pair statistics have the wrong shape")
        if np.any(counts < 0) or not (
            np.all(np.isfinite(precision))
            and np.all(np.isfinite(information))
            and np.all(np.isfinite(rewards))
        ):
            raise ValueError("serialized LinUCB state contains invalid values")
        for querier in range(policy.n_agents):
            for peer in range(policy.n_agents):
                matrix = precision[querier, peer]
                if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
                    raise ValueError("serialized precision must be symmetric")
                np.linalg.cholesky(matrix)
        policy._precision = precision.copy()
        policy._information = information.copy()
        policy._counts = counts.copy()
        policy._reward_means = rewards.copy()
        try:
            policy._rng.bit_generator.state = dict(state["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("serialized random-generator state is invalid") from exc
        return policy

    def dumps(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def loads(cls, payload: str) -> "DisjointLinUCB":
        state = json.loads(payload)
        if not isinstance(state, dict):
            raise ValueError("serialized LinUCB payload must contain an object")
        return cls.from_state_dict(state)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DisjointLinUCB":
        return cls.loads(Path(path).read_text(encoding="utf-8"))


class MACE:
    """Compose feature extraction and disjoint LinUCB without hiding phases."""

    def __init__(
        self,
        bandit: DisjointLinUCB,
        feature_extractor: MACEFeatureExtractor | None = None,
    ) -> None:
        extractor = feature_extractor or MACEFeatureExtractor()
        if bandit.dimension != extractor.dimension:
            raise ValueError("bandit dimension does not match MACE feature schema")
        self.bandit = bandit
        self.feature_extractor = extractor

    def select_round(
        self,
        snapshot: RoundSnapshot,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[LinUCBSelection, ...]:
        if snapshot.n_agents != self.bandit.n_agents:
            raise ValueError("snapshot and bandit agent counts differ")
        features = self.feature_extractor.build(
            snapshot, historical_rewards=self.bandit.reward_means
        )
        return self.bandit.select_round(features, candidate_mask=candidate_mask)

    def update_from_scores(
        self,
        selections: Sequence[LinUCBSelection],
        scores_before: Iterable[float] | np.ndarray,
        scores_after: Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        before = np.asarray(tuple(scores_before), dtype=np.float64)
        after = np.asarray(tuple(scores_after), dtype=np.float64)
        expected = (len(selections),)
        if before.shape != expected or after.shape != expected:
            raise ValueError("one before/after score is required for each selection")
        rewards = np.array(
            [blended_mace_reward(left, right) for left, right in zip(before, after)],
            dtype=np.float64,
        )
        self.bandit.update_batch(selections, rewards)
        rewards.setflags(write=False)
        return rewards

    def state_dict(self) -> dict[str, object]:
        return {
            "version": MACE_STATE_VERSION,
            "feature_config": self.feature_extractor.config.to_dict(),
            "bandit": self.bandit.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "MACE":
        if state.get("version") != MACE_STATE_VERSION:
            raise ValueError(f"unsupported MACE state: {state.get('version')}")
        config_state = state["feature_config"]
        bandit_state = state["bandit"]
        if not isinstance(config_state, Mapping) or not isinstance(bandit_state, Mapping):
            raise ValueError("MACE state sections must be mappings")
        return cls(
            DisjointLinUCB.from_state_dict(bandit_state),
            MACEFeatureExtractor(MACEFeatureConfig.from_dict(config_state)),
        )

    def dumps(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def loads(cls, payload: str) -> "MACE":
        state = json.loads(payload)
        if not isinstance(state, dict):
            raise ValueError("serialized MACE payload must contain an object")
        return cls.from_state_dict(state)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MACE":
        return cls.loads(Path(path).read_text(encoding="utf-8"))
