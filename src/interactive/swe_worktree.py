"""Prepared-repository lifecycle for SWE-bench Verified Coding Agents.

This module is a thin adaptation of SkillFlow
``training/environment.py::GenericTaskEnvironment._setup_swe_repo`` and
``GenericTaskEnvironment.cleanup``.  It keeps the same repository-store
mapping (``owner/repo`` -> ``owner__repo``), detached ``git worktree`` at the
task's ``base_commit``, and explicit cleanup boundary.

The adaptation separates that lifecycle from SkillFlow's monolithic episode
object so this project's :mod:`coding_tools` can receive one prepared
repository root.  Only public task identity fields are accepted; no patch,
test patch, hidden test, ground truth, or evaluator result is loaded here.
Official resolution remains exclusively in :mod:`swebench_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping


class SWEbenchWorktreeUnavailable(RuntimeError):
    """A task-pinned SWE-bench repository could not be prepared."""


class SWEbenchWorktreeCleanupError(RuntimeError):
    """A prepared worktree could not be cleanly detached from its source."""


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SWEbenchRepositoryIdentity:
    """Public identity needed to pin one SWE-bench repository checkout."""

    instance_id: str
    repo: str
    base_commit: str

    def __post_init__(self) -> None:
        for field_name in ("instance_id", "repo", "base_commit"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )

    @classmethod
    def from_task_record(
        cls,
        record: object,
    ) -> "SWEbenchRepositoryIdentity":
        """Read only the public ``skillflow.extra`` repository identity.

        Canonical JSONL mappings may be passed before hydration; hydrated
        :class:`TaskRecord` objects carry the same fields under
        ``metadata.skillflow.extra``.  Evaluator payload and ground truth are
        deliberately not fallback sources.
        """

        metadata: object
        if isinstance(record, Mapping):
            metadata = record.get("metadata", {})
            direct_extra = record.get("extra", {})
        else:
            metadata = getattr(record, "metadata", {})
            direct_extra = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("SWE-bench task metadata must be a mapping")
        if metadata.get("dataset_key") != "swe_bench":
            raise ValueError("task is not identified as SWE-bench")

        skillflow = metadata.get("skillflow", {})
        hydrated_extra = (
            skillflow.get("extra", {}) if isinstance(skillflow, Mapping) else {}
        )
        public_extra = hydrated_extra if isinstance(hydrated_extra, Mapping) else {}
        if not public_extra and isinstance(direct_extra, Mapping):
            public_extra = direct_extra
        return cls(
            instance_id=_required_text(
                public_extra.get("instance_id"), field_name="instance_id"
            ),
            repo=_required_text(public_extra.get("repo"), field_name="repo"),
            base_commit=_required_text(
                public_extra.get("base_commit"), field_name="base_commit"
            ),
        )


def _run_git(
    arguments: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def _task_prefix(instance_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id)[:30]
    return f"swe_{normalized or 'instance'}_"


def _remove_failed_setup(path: Path, *, worktree_root: Path) -> None:
    """Remove only the exact temporary directory allocated by this setup."""

    if not path.exists():
        return
    resolved_root = worktree_root.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_root or path.is_symlink():
        raise SWEbenchWorktreeCleanupError(
            "refusing to remove an unexpected failed-setup path"
        )
    shutil.rmtree(resolved_path)


@dataclass(slots=True)
class PreparedSWEbenchWorktree:
    """One isolated, detached repository worktree owned by one task run."""

    identity: SWEbenchRepositoryIdentity
    source_repository: Path
    repo_root: Path
    pinned_commit: str
    worktree_root: Path
    cleanup_timeout_seconds: float = 10.0
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def cleanup(self) -> None:
        """Detach the exact owned worktree; repeated cleanup is a no-op."""

        if self._closed:
            return
        try:
            result = _run_git(
                ["worktree", "remove", str(self.repo_root), "--force"],
                cwd=self.source_repository,
                timeout_seconds=self.cleanup_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SWEbenchWorktreeCleanupError(
                "git could not detach the prepared SWE-bench worktree"
            ) from exc
        if result.returncode != 0:
            if not self.repo_root.exists():
                self._closed = True
                return
            raise SWEbenchWorktreeCleanupError(
                "git could not remove the prepared SWE-bench worktree: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        if self.repo_root.exists():
            raise SWEbenchWorktreeCleanupError(
                "git reported success but the prepared worktree still exists"
            )
        self._closed = True

    close = cleanup

    def __enter__(self) -> "PreparedSWEbenchWorktree":
        if self._closed:
            raise RuntimeError("prepared SWE-bench worktree is already closed")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.cleanup()


def prepare_swebench_worktree(
    identity: SWEbenchRepositoryIdentity,
    *,
    repository_store: Path | str | None = None,
    worktree_root: Path | str | None = None,
    setup_timeout_seconds: float = 30.0,
    cleanup_timeout_seconds: float = 10.0,
) -> PreparedSWEbenchWorktree:
    """Prepare SkillFlow's detached per-task SWE-bench repository checkout.

    The repository and worktree roots default to SkillFlow's
    ``SWE_BENCH_ENVS`` and ``SWE_BENCH_WORKTREES`` environment variables.
    Unlike SkillFlow's non-formal fallback, the worktree root is required so a
    caller cannot silently place formal task state in an arbitrary temp root.
    """

    if not isinstance(identity, SWEbenchRepositoryIdentity):
        raise TypeError("identity must be SWEbenchRepositoryIdentity")
    if setup_timeout_seconds <= 0 or cleanup_timeout_seconds <= 0:
        raise ValueError("worktree timeouts must be positive")

    store_value = repository_store or os.environ.get("SWE_BENCH_ENVS")
    root_value = worktree_root or os.environ.get("SWE_BENCH_WORKTREES")
    if not store_value:
        raise SWEbenchWorktreeUnavailable("SWE_BENCH_ENVS is not configured")
    if not root_value:
        raise SWEbenchWorktreeUnavailable("SWE_BENCH_WORKTREES is not configured")
    store = Path(store_value).expanduser().resolve()
    root = Path(root_value).expanduser().resolve()
    if not store.is_dir():
        raise SWEbenchWorktreeUnavailable(
            f"SWE-bench repository store is unavailable: {store}"
        )
    if not root.is_dir():
        raise SWEbenchWorktreeUnavailable(
            f"SWE-bench worktree root is unavailable: {root}"
        )

    # Directly preserve SkillFlow ``_repo_dir(repo)`` naming.
    source_repository = (store / identity.repo.replace("/", "__")).resolve()
    if not source_repository.is_dir():
        raise SWEbenchWorktreeUnavailable(
            f"prepared SWE-bench source repository is unavailable: {identity.repo}"
        )

    try:
        resolved = _run_git(
            ["rev-parse", "--verify", f"{identity.base_commit}^{{commit}}"],
            cwd=source_repository,
            timeout_seconds=setup_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SWEbenchWorktreeUnavailable(
            "the prepared source repository could not be queried"
        ) from exc
    if resolved.returncode != 0 or not resolved.stdout.strip():
        raise SWEbenchWorktreeUnavailable(
            "the task base_commit is unavailable in its prepared source repository"
        )
    pinned_commit = resolved.stdout.strip()

    repo_root = Path(
        tempfile.mkdtemp(prefix=_task_prefix(identity.instance_id), dir=str(root))
    ).resolve()
    worktree_registered = False
    try:
        added = _run_git(
            [
                "worktree",
                "add",
                str(repo_root),
                identity.base_commit,
                "--detach",
                "-q",
            ],
            cwd=source_repository,
            timeout_seconds=setup_timeout_seconds,
        )
        if added.returncode != 0:
            _remove_failed_setup(repo_root, worktree_root=root)
            raise SWEbenchWorktreeUnavailable(
                "git could not create the task-pinned SWE-bench worktree: "
                f"{added.stderr.strip() or added.stdout.strip()}"
            )
        worktree_registered = True

        actual = _run_git(
            ["rev-parse", "HEAD"],
            cwd=repo_root,
            timeout_seconds=setup_timeout_seconds,
        )
        if actual.returncode != 0 or actual.stdout.strip() != pinned_commit:
            cleanup = _run_git(
                ["worktree", "remove", str(repo_root), "--force"],
                cwd=source_repository,
                timeout_seconds=cleanup_timeout_seconds,
            )
            if cleanup.returncode != 0:
                raise SWEbenchWorktreeCleanupError(
                    "task identity check failed and git could not detach the worktree"
                )
            raise SWEbenchWorktreeUnavailable(
                "prepared worktree does not match the task base_commit"
            )
    except subprocess.TimeoutExpired as exc:
        if repo_root.exists():
            if worktree_registered:
                cleanup = _run_git(
                    ["worktree", "remove", str(repo_root), "--force"],
                    cwd=source_repository,
                    timeout_seconds=cleanup_timeout_seconds,
                )
                if cleanup.returncode != 0 and repo_root.exists():
                    raise SWEbenchWorktreeCleanupError(
                        "worktree setup timed out and cleanup did not complete"
                    ) from exc
            else:
                _remove_failed_setup(repo_root, worktree_root=root)
        raise SWEbenchWorktreeUnavailable("SWE-bench worktree setup timed out") from exc
    except Exception:
        # Setup errors above already clean their exact task-owned path.  This
        # guard handles unexpected failures before returning ownership.
        if repo_root.exists():
            if worktree_registered:
                cleanup = _run_git(
                    ["worktree", "remove", str(repo_root), "--force"],
                    cwd=source_repository,
                    timeout_seconds=cleanup_timeout_seconds,
                )
                if cleanup.returncode != 0 and repo_root.exists():
                    raise SWEbenchWorktreeCleanupError(
                        "SWE-bench worktree setup failed and cleanup did not complete"
                    )
            else:
                _remove_failed_setup(repo_root, worktree_root=root)
        raise

    return PreparedSWEbenchWorktree(
        identity=identity,
        source_repository=source_repository,
        repo_root=repo_root,
        pinned_commit=pinned_commit,
        worktree_root=root,
        cleanup_timeout_seconds=float(cleanup_timeout_seconds),
    )


def prepare_swebench_worktree_for_task(
    record: object,
    **options: Any,
) -> PreparedSWEbenchWorktree:
    """Minimal runner entry: public task record -> prepared repository."""

    return prepare_swebench_worktree(
        SWEbenchRepositoryIdentity.from_task_record(record),
        **options,
    )


__all__ = [
    "PreparedSWEbenchWorktree",
    "SWEbenchRepositoryIdentity",
    "SWEbenchWorktreeCleanupError",
    "SWEbenchWorktreeUnavailable",
    "prepare_swebench_worktree",
    "prepare_swebench_worktree_for_task",
]
