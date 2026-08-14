"""Atomic JSON store for the current version of each Skill."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from ..persistence.ids import canonical_json, stable_id
from .schema import SkillRecord


STORE_SCHEMA = "flowsteer.skill-store.v2"


class SkillStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock_path.touch(exist_ok=True)
        with self._process_lock():
            if not self.path.exists():
                self._write(self._empty())

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"schema_version": STORE_SCHEMA, "events": [], "heads": {}, "current": {}}

    @contextmanager
    def _process_lock(self) -> Iterator[None]:
        with self._lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> Dict[str, dict]:
        with self.path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Skill store root must be an object")
        if value.get("schema_version") != STORE_SCHEMA:
            raise ValueError(f"unsupported Skill store schema: {value.get('schema_version')}")
        if not isinstance(value.get("events"), list) or not isinstance(value.get("current"), dict):
            raise ValueError("Skill store is missing events/current collections")
        return value

    def _write(self, value: Dict[str, dict]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def upsert(self, skill: SkillRecord) -> None:
        with self._lock:
            with self._process_lock():
                data = self._load()
                incoming = skill.to_dict()
                current = data["current"].get(skill.skill_id)
                if current is not None:
                    current_version = int(current["version"])
                    if current_version > skill.version:
                        raise ValueError("cannot overwrite a newer Skill version")
                    if current_version == skill.version:
                        if current == incoming:
                            return
                        immutable_exclusions = {
                            "status",
                            "activated_epoch",
                            "suspended_reason",
                            "gate_config",
                            "gate_receipt",
                            "updated_at",
                        }
                        old_semantics = {
                            key: value for key, value in current.items() if key not in immutable_exclusions
                        }
                        new_semantics = {
                            key: value for key, value in incoming.items() if key not in immutable_exclusions
                        }
                        if old_semantics != new_semantics:
                            raise ValueError("same-version Skill semantics/evidence are immutable")
                        allowed = {
                            ("candidate", "active"),
                            ("candidate", "retired"),
                            ("active", "suspended"),
                            ("active", "retired"),
                            ("suspended", "retired"),
                        }
                        if (current["status"], incoming["status"]) not in allowed:
                            raise ValueError("illegal same-version Skill lifecycle transition")
                    elif skill.version != current_version + 1:
                        raise ValueError("Skill semantic versions must advance by exactly one")

                previous_event_id = data["heads"].get(skill.skill_id)
                event_id = stable_id(
                    "skill_event",
                    {"previous_event_id": previous_event_id, "record": incoming},
                )
                data["events"].append(
                    {
                        "event_id": event_id,
                        "previous_event_id": previous_event_id,
                        "skill_id": skill.skill_id,
                        "record": incoming,
                    }
                )
                data["heads"][skill.skill_id] = event_id
                data["current"][skill.skill_id] = incoming
                self._write(data)

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        with self._lock:
            with self._process_lock():
                value = self._load()["current"].get(skill_id)
        return SkillRecord.from_dict(value) if value is not None else None

    def list(self) -> List[SkillRecord]:
        with self._lock:
            with self._process_lock():
                values = list(self._load()["current"].values())
        return [SkillRecord.from_dict(item) for item in values]

    def history(self, skill_id: str) -> List[SkillRecord]:
        with self._lock:
            with self._process_lock():
                events = [
                    event["record"]
                    for event in self._load()["events"]
                    if event.get("skill_id") == skill_id
                ]
        return [SkillRecord.from_dict(item) for item in events]
