"""Records and unbiased presentation-order utilities for paired probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

import numpy as np


Payload = TypeVar("Payload")


@dataclass(frozen=True)
class ProbeOrder:
    """Canonical candidate identity plus the randomized presentation order."""

    canonical_left: str
    canonical_right: str
    presented_first: str
    presented_second: str

    def __post_init__(self) -> None:
        identifiers = (
            self.canonical_left,
            self.canonical_right,
            self.presented_first,
            self.presented_second,
        )
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("probe candidate identifiers must be non-empty strings")
        if self.canonical_left == self.canonical_right:
            raise ValueError("a paired probe requires two distinct candidates")
        if {self.presented_first, self.presented_second} != {
            self.canonical_left,
            self.canonical_right,
        }:
            raise ValueError("presented order must be a permutation of the canonical pair")

    @property
    def flipped(self) -> bool:
        return self.presented_first == self.canonical_right

    def arrange(self, left_payload: Payload, right_payload: Payload) -> tuple[Payload, Payload]:
        """Put canonical payloads into presentation order."""

        return (right_payload, left_payload) if self.flipped else (left_payload, right_payload)

    def canonicalize(
        self, first_value: Payload, second_value: Payload
    ) -> tuple[Payload, Payload]:
        """Map values observed in presentation order back to canonical order."""

        return (second_value, first_value) if self.flipped else (first_value, second_value)


def randomize_probe_order(
    canonical_left: str,
    canonical_right: str,
    rng: np.random.Generator,
) -> ProbeOrder:
    """Randomly swap a pair with probability one half using caller-owned RNG."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    flipped = bool(rng.integers(0, 2))
    if flipped:
        first, second = canonical_right, canonical_left
    else:
        first, second = canonical_left, canonical_right
    return ProbeOrder(canonical_left, canonical_right, first, second)


def randomize_probe_orders(
    pairs: Iterable[tuple[str, str]],
    rng: np.random.Generator,
) -> tuple[ProbeOrder, ...]:
    return tuple(randomize_probe_order(left, right, rng) for left, right in pairs)


@dataclass(frozen=True)
class PairedProbeRecord:
    """Auditable paired outcome with canonical, order-invariant difference."""

    probe_id: str
    order: ProbeOrder
    presented_first_outcome: float
    presented_second_outcome: float

    def __post_init__(self) -> None:
        if not isinstance(self.probe_id, str) or not self.probe_id:
            raise ValueError("probe_id must be a non-empty string")
        outcomes = (self.presented_first_outcome, self.presented_second_outcome)
        if not all(np.isfinite(value) for value in outcomes):
            raise ValueError("paired probe outcomes must be finite")

    @property
    def canonical_outcomes(self) -> tuple[float, float]:
        left, right = self.order.canonicalize(
            float(self.presented_first_outcome),
            float(self.presented_second_outcome),
        )
        return float(left), float(right)

    @property
    def difference(self) -> float:
        left, right = self.canonical_outcomes
        return left - right

    def to_dict(self) -> dict[str, object]:
        left, right = self.canonical_outcomes
        return {
            "probe_id": self.probe_id,
            "canonical_left": self.order.canonical_left,
            "canonical_right": self.order.canonical_right,
            "presented_first": self.order.presented_first,
            "presented_second": self.order.presented_second,
            "presented_first_outcome": float(self.presented_first_outcome),
            "presented_second_outcome": float(self.presented_second_outcome),
            "canonical_left_outcome": left,
            "canonical_right_outcome": right,
            "difference": self.difference,
        }


def record_paired_probe(
    probe_id: str,
    order: ProbeOrder,
    presented_outcomes: Sequence[float],
) -> PairedProbeRecord:
    if len(presented_outcomes) != 2:
        raise ValueError("exactly two presented outcomes are required")
    return PairedProbeRecord(
        probe_id=probe_id,
        order=order,
        presented_first_outcome=float(presented_outcomes[0]),
        presented_second_outcome=float(presented_outcomes[1]),
    )
