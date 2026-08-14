"""Replay full graph snapshots with an explicit hash chain.

Full snapshots cost more storage than patches but make the Phase-0 evidence
path deterministic and independent of parser/runtime implementation changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional

from .ids import stable_id


class SnapshotReplayError(ValueError):
    pass


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class GraphSnapshotEvent:
    revision: int
    graph: Mapping[str, Any]
    snapshot_id: str
    previous_snapshot_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise SnapshotReplayError("snapshot revision must be non-negative")
        object.__setattr__(self, "graph", _freeze(self.graph))

    @classmethod
    def create(
        cls,
        revision: int,
        graph: Mapping[str, Any],
        previous_snapshot_id: Optional[str] = None,
    ) -> "GraphSnapshotEvent":
        payload = {
            "revision": revision,
            "graph": graph,
            "previous_snapshot_id": previous_snapshot_id,
        }
        return cls(
            revision=revision,
            graph=_freeze(graph),
            snapshot_id=stable_id("snapshot", payload),
            previous_snapshot_id=previous_snapshot_id,
        )

    def verify(self) -> None:
        expected = stable_id(
            "snapshot",
            {
                "revision": self.revision,
                "graph": self.graph,
                "previous_snapshot_id": self.previous_snapshot_id,
            },
        )
        if expected != self.snapshot_id:
            raise SnapshotReplayError(
                f"snapshot hash mismatch at revision {self.revision}: "
                f"expected {expected}, got {self.snapshot_id}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "graph": _thaw(self.graph),
            "snapshot_id": self.snapshot_id,
            "previous_snapshot_id": self.previous_snapshot_id,
        }


def replay_snapshots(events: Iterable[GraphSnapshotEvent]) -> Dict[str, Any]:
    previous: Optional[GraphSnapshotEvent] = None
    final_graph: Optional[Dict[str, Any]] = None
    count = 0
    for event in events:
        event.verify()
        if previous is None:
            if event.previous_snapshot_id is not None:
                raise SnapshotReplayError("first snapshot cannot reference a predecessor")
        else:
            if event.revision != previous.revision + 1:
                raise SnapshotReplayError(
                    f"non-consecutive graph revision {previous.revision} -> {event.revision}"
                )
            if event.previous_snapshot_id != previous.snapshot_id:
                raise SnapshotReplayError("broken snapshot hash chain")
        previous = event
        final_graph = _thaw(event.graph)
        count += 1
    if not count or final_graph is None:
        raise SnapshotReplayError("cannot replay an empty snapshot stream")
    return final_graph
