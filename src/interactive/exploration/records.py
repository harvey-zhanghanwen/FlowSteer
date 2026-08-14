"""Immutable records shared by exploration-plane components.

The exploration plane deliberately separates *selection* from *mutation*.  A
``RoundSnapshot`` captures the responses visible at the start of one selection
round, while selection records retain the exact feature vector used to make a
decision.  NumPy arrays stored by these records are defensive, read-only
copies so later response generation or policy updates cannot rewrite history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def readonly_array(
    value: Iterable[float] | np.ndarray,
    *,
    ndim: int | None = None,
    shape: Tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a finite, defensive, read-only ``float64`` NumPy array."""

    array = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected an array with {ndim} dimensions, got {array.ndim}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("arrays stored in exploration records must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RoundSnapshot:
    """Responses frozen immediately before a synchronous peer-selection step.

    ``completed_round`` is the number of response-update rounds already
    completed.  Therefore a snapshot with ``completed_round == 0`` contains
    independently generated initial responses, and is used to select peers for
    interaction round 1.
    """

    responses: Tuple[str, ...]
    completed_round: int
    total_rounds: int

    def __post_init__(self) -> None:
        responses = tuple(self.responses)
        if not responses:
            raise ValueError("a round snapshot requires at least one response")
        if any(not isinstance(response, str) for response in responses):
            raise TypeError("snapshot responses must be strings")
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if self.completed_round < 0 or self.completed_round > self.total_rounds:
            raise ValueError("completed_round must lie in [0, total_rounds]")
        object.__setattr__(self, "responses", responses)

    @classmethod
    def initial(cls, responses: Iterable[str], total_rounds: int) -> "RoundSnapshot":
        return cls(tuple(responses), completed_round=0, total_rounds=total_rounds)

    @property
    def n_agents(self) -> int:
        return len(self.responses)

    @property
    def selection_round(self) -> int:
        """The 1-based interaction round selected from this snapshot."""

        if self.completed_round >= self.total_rounds:
            raise RuntimeError("all configured interaction rounds are complete")
        return self.completed_round + 1

    @property
    def normalized_selection_round(self) -> float:
        return self.selection_round / self.total_rounds

    def advance(self, responses: Iterable[str]) -> "RoundSnapshot":
        """Create the next immutable snapshot after all agents update once."""

        if self.completed_round >= self.total_rounds:
            raise RuntimeError("cannot advance a completed interaction episode")
        next_responses = tuple(responses)
        if len(next_responses) != self.n_agents:
            raise ValueError("the number of agents cannot change within an episode")
        return RoundSnapshot(
            next_responses,
            completed_round=self.completed_round + 1,
            total_rounds=self.total_rounds,
        )


@dataclass(frozen=True)
class LinUCBSelection:
    """An auditable peer selection made from one immutable round snapshot."""

    querier: int
    peer: int
    context: np.ndarray
    predicted_mean: float
    epistemic_variance: float
    exploration_bonus: float
    score: float
    candidate_scores: np.ndarray

    def __post_init__(self) -> None:
        if self.querier < 0 or self.peer < 0:
            raise ValueError("agent indices must be non-negative")
        object.__setattr__(self, "context", readonly_array(self.context, ndim=1))
        object.__setattr__(
            self, "candidate_scores", readonly_array(self.candidate_scores, ndim=1)
        )
        scalars = (
            self.predicted_mean,
            self.epistemic_variance,
            self.exploration_bonus,
            self.score,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("selection diagnostics must be finite")
        if self.epistemic_variance < 0 or self.exploration_bonus < 0:
            raise ValueError("selection variances and bonuses must be non-negative")


@dataclass(frozen=True)
class PolicyDecision:
    """Generic decision returned by posterior-based exploration policies."""

    action: int
    scores: np.ndarray
    expected_values: np.ndarray
    bonuses: np.ndarray

    def __post_init__(self) -> None:
        scores = readonly_array(self.scores, ndim=1)
        expected = readonly_array(self.expected_values, ndim=1)
        bonuses = readonly_array(self.bonuses, ndim=1)
        if not (scores.shape == expected.shape == bonuses.shape):
            raise ValueError("policy diagnostic vectors must have matching shapes")
        if self.action < 0 or self.action >= scores.size:
            raise ValueError("selected action is outside the score vector")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "expected_values", expected)
        object.__setattr__(self, "bonuses", bonuses)
