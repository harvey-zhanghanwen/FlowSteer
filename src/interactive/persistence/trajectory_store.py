"""Crash-conscious append-only JSONL stores for trajectories and probes."""

from __future__ import annotations

from dataclasses import is_dataclass
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

try:  # Linux production path; fallback keeps unit tests portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from .ids import canonical_json, stable_id


class DuplicateEventError(ValueError):
    pass


_PROCESS_PATH_LOCKS_GUARD = RLock()
_PROCESS_PATH_LOCKS: dict[Path, RLock] = {}


def _process_lock_for_path(path: Path) -> RLock:
    """Return the process-wide lock shared by every store for one file."""

    canonical_path = path.resolve(strict=False)
    with _PROCESS_PATH_LOCKS_GUARD:
        lock = _PROCESS_PATH_LOCKS.get(canonical_path)
        if lock is None:
            lock = RLock()
            _PROCESS_PATH_LOCKS[canonical_path] = lock
        return lock


def _to_dict(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict"):
        value = record.to_dict()
    elif is_dataclass(record):
        from dataclasses import asdict

        value = asdict(record)
    elif isinstance(record, Mapping):
        value = dict(record)
    else:
        raise TypeError("record must be a mapping, dataclass, or expose to_dict()")
    if not isinstance(value, dict):
        raise TypeError("serialized record must be a dict")
    return value


class AppendOnlyJsonlStore:
    """Content-addressed JSONL store with process-safe duplicate checks."""

    def __init__(self, path: str | os.PathLike[str], record_kind: str) -> None:
        self.path = Path(path)
        self.record_kind = record_kind
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._thread_lock = _process_lock_for_path(self.path)

    def _validate_envelope(self, event: Any, line_number: int) -> Dict[str, Any]:
        if not isinstance(event, dict):
            raise ValueError(f"invalid event envelope at line {line_number}")
        required = {"event_id", "record_kind", "content_hash", "payload"}
        if set(event) != required:
            raise ValueError(f"invalid event envelope fields at line {line_number}")
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError(f"event_id must be a non-empty string at line {line_number}")
        if event["record_kind"] != self.record_kind:
            raise ValueError(f"record kind mismatch at line {line_number}")
        if not isinstance(event["payload"], dict):
            raise ValueError(f"event payload must be an object at line {line_number}")
        expected_hash = stable_id(
            "content",
            {"record_kind": event["record_kind"], "payload": event["payload"]},
        )
        if event["content_hash"] != expected_hash:
            raise ValueError(f"event content hash mismatch at line {line_number}")
        return event

    def _read_events(self, handle: Any, *, recover_torn_tail: bool) -> list[Dict[str, Any]]:
        handle.seek(0)
        contents = handle.read()
        # JSONL records are delimited only by the physical LF byte.  Python's
        # str.splitlines() additionally treats Unicode NEL, U+2028, and U+2029
        # as boundaries even though those code points are legal inside a JSON
        # string.  Model output can contain them verbatim, so split only on LF.
        lines = contents.split("\n")
        parsed: list[Dict[str, Any]] = []
        offset = 0
        for index, line in enumerate(lines):
            line_number = index + 1
            has_physical_newline = index < len(lines) - 1
            if not line.strip():
                offset += len(line) + int(has_physical_newline)
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                is_torn_tail = index == len(lines) - 1 and not contents.endswith("\n")
                if recover_torn_tail and is_torn_tail:
                    valid_prefix = contents[:offset]
                    handle.seek(0)
                    handle.write(valid_prefix)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            parsed.append(self._validate_envelope(event, line_number))
            offset += len(line) + int(has_physical_newline)
        return parsed

    def _existing_events(self, handle: Any) -> Dict[str, str]:
        events: Dict[str, str] = {}
        for event in self._read_events(handle, recover_torn_tail=True):
            event_id = event.get("event_id")
            if event_id:
                normalized = canonical_json(
                    {
                        "record_kind": event.get("record_kind"),
                        "payload": event.get("payload"),
                    }
                )
                existing = events.get(str(event_id))
                if existing is not None and existing != normalized:
                    raise DuplicateEventError(
                        f"persisted event ID has conflicting payloads: {event_id}"
                    )
                events[str(event_id)] = normalized
        return events

    def append(
        self,
        record: Any,
        *,
        event_id: Optional[str] = None,
        idempotent: bool = True,
    ) -> str:
        payload = _to_dict(record)
        resolved_id = event_id or stable_id(self.record_kind, payload)
        if not isinstance(resolved_id, str) or not resolved_id.strip():
            raise ValueError("event_id must be a non-empty string")
        content_hash = stable_id(
            "content", {"record_kind": self.record_kind, "payload": payload}
        )
        envelope = {
            "event_id": resolved_id,
            "record_kind": self.record_kind,
            "content_hash": content_hash,
            "payload": payload,
        }
        with self._thread_lock:
            with self.path.open("r+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    existing = self._existing_events(handle)
                    if resolved_id in existing:
                        incoming = canonical_json(
                            {"record_kind": self.record_kind, "payload": payload}
                        )
                        if existing[resolved_id] != incoming:
                            raise DuplicateEventError(
                                f"event ID {resolved_id} already has a different payload"
                            )
                        if idempotent:
                            return resolved_id
                        raise DuplicateEventError(
                            f"event already exists: {resolved_id}"
                        )
                    handle.seek(0, os.SEEK_END)
                    handle.write(canonical_json(envelope) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return resolved_id

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # FlowSteer evaluation can finish several AgentGraph tasks concurrently.
        # Read one complete snapshot under the same advisory lock used by the
        # append path; otherwise a reader can parse a writer's partial final
        # line and incorrectly reject an otherwise valid trajectory.  Release
        # the lock before yielding so a slow consumer never blocks append.
        with self._thread_lock:
            with self.path.open("r", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    events = self._read_events(
                        handle,
                        recover_torn_tail=False,
                    )
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        yield from events

    def payloads(self) -> Iterator[Dict[str, Any]]:
        for event in self:
            yield dict(event["payload"])

    def get(self, event_id: str) -> Optional[Dict[str, Any]]:
        for event in self:
            if event.get("event_id") == event_id:
                return dict(event["payload"])
        return None

    def __len__(self) -> int:
        return sum(1 for _ in self)


class EvidenceStore:
    """Separate streams prevent forced probes from masquerading as rollouts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_path = Path(root)
        self.trajectories = AppendOnlyJsonlStore(root_path / "trajectories.jsonl", "trajectory")
        self.probes = AppendOnlyJsonlStore(root_path / "probes.jsonl", "probe")
        self.posteriors = AppendOnlyJsonlStore(root_path / "posteriors.jsonl", "posterior")
        self.snapshots = AppendOnlyJsonlStore(root_path / "snapshots.jsonl", "snapshot")

    def append_trajectory(self, record: Any) -> str:
        payload = _to_dict(record)
        trajectory_id = payload.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory record requires a stable trajectory_id")
        return self.trajectories.append(payload, event_id=trajectory_id)

    def append_probe(self, record: Any) -> str:
        payload = _to_dict(record)
        split = payload.get("task_split")
        if split not in {"train", "validation"}:
            raise ValueError("probe evidence task_split must be train or validation")
        required = {
            "problem_id",
            "policy_version",
            "evaluator_version",
            "feature_schema_version",
            "paired_effect",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError("probe evidence is missing fields: " + ", ".join(sorted(missing)))
        try:
            finite_effect = math.isfinite(float(payload["paired_effect"]))
        except (TypeError, ValueError):
            finite_effect = False
        if not finite_effect:
            raise ValueError("probe paired_effect must be finite")
        probe_id = payload.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise ValueError("probe evidence requires a stable probe_id")
        return self.probes.append(payload, event_id=probe_id)

    def append_posterior(self, record: Any) -> str:
        payload = _to_dict(record)
        posterior_id = payload.get("posterior_id")
        if not isinstance(posterior_id, str) or not posterior_id:
            raise ValueError("posterior record requires a stable posterior_id")
        return self.posteriors.append(payload, event_id=posterior_id)

    def append_snapshot(self, record: Any) -> str:
        payload = _to_dict(record)
        snapshot_id = payload.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("snapshot record requires a stable snapshot_id")
        return self.snapshots.append(payload, event_id=snapshot_id)

    def resolve_probe(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self.probes.get(event_id)
