from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pytest

from src.interactive.records import TaskRecord
from src.interactive.swe_worktree import (
    SWEbenchRepositoryIdentity,
    SWEbenchWorktreeUnavailable,
    prepare_swebench_worktree,
    prepare_swebench_worktree_for_task,
)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(store: Path) -> tuple[Path, str, str]:
    source = store / "owner__repo"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "fixture@example.invalid")
    _git(source, "config", "user.name", "Fixture")
    (source / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "bug.py")
    _git(source, "commit", "-q", "-m", "base")
    base_commit = _git(source, "rev-parse", "HEAD").stdout.strip()
    (source / "bug.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(source, "commit", "-q", "-am", "later")
    later_commit = _git(source, "rev-parse", "HEAD").stdout.strip()
    return source, base_commit, later_commit


def _identity(
    base_commit: str,
    *,
    instance_id: str = "owner__repo-1",
) -> SWEbenchRepositoryIdentity:
    return SWEbenchRepositoryIdentity(
        instance_id=instance_id,
        repo="owner/repo",
        base_commit=base_commit,
    )


def test_prepares_exact_detached_base_commit_and_cleans_up() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = root / "repos"
        worktrees = root / "worktrees"
        store.mkdir()
        worktrees.mkdir()
        source, base_commit, later_commit = _repository(store)

        prepared = prepare_swebench_worktree(
            _identity(base_commit),
            repository_store=store,
            worktree_root=worktrees,
        )
        prepared_path = prepared.repo_root

        assert prepared.identity.instance_id == "owner__repo-1"
        assert prepared.pinned_commit == base_commit
        assert _git(prepared.repo_root, "rev-parse", "HEAD").stdout.strip() == base_commit
        assert _git(source, "rev-parse", "HEAD").stdout.strip() == later_commit
        assert (prepared.repo_root / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=prepared.repo_root,
            capture_output=True,
        ).returncode != 0

        prepared.cleanup()
        prepared.cleanup()
        assert prepared.closed
        assert not prepared_path.exists()
        assert source.is_dir()


def test_each_task_run_receives_an_isolated_worktree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = root / "repos"
        worktrees = root / "worktrees"
        store.mkdir()
        worktrees.mkdir()
        _source, base_commit, _later_commit = _repository(store)

        first = prepare_swebench_worktree(
            _identity(base_commit, instance_id="owner__repo-1"),
            repository_store=store,
            worktree_root=worktrees,
        )
        second = prepare_swebench_worktree(
            _identity(base_commit, instance_id="owner__repo-2"),
            repository_store=store,
            worktree_root=worktrees,
        )
        try:
            assert first.repo_root != second.repo_root
            (first.repo_root / "bug.py").write_text("VALUE = 10\n", encoding="utf-8")
            assert (second.repo_root / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        finally:
            first.cleanup()
            second.cleanup()


def test_task_entry_reads_only_public_repository_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = root / "repos"
        worktrees = root / "worktrees"
        store.mkdir()
        worktrees.mkdir()
        _source, base_commit, _later_commit = _repository(store)
        record = TaskRecord(
            task_id="swe-bench:owner__repo-1",
            question="Fix the public issue.",
            ground_truth="DO_NOT_READ_GOLD_PATCH",
            split="validation",
            metadata={
                "dataset_key": "swe_bench",
                "skillflow": {
                    "extra": {
                        "instance_id": "owner__repo-1",
                        "repo": "owner/repo",
                        "base_commit": base_commit,
                    }
                },
                "evaluator_payload": {
                    "patch": "DO_NOT_READ_GOLD_PATCH",
                    "test_patch": "DO_NOT_READ_HIDDEN_TEST_PATCH",
                },
            },
        )

        with prepare_swebench_worktree_for_task(
            record,
            repository_store=store,
            worktree_root=worktrees,
        ) as prepared:
            assert prepared.identity == _identity(base_commit)


def test_task_entry_does_not_fallback_to_evaluator_payload() -> None:
    record = TaskRecord(
        task_id="swe-bench:owner__repo-1",
        question="Fix the public issue.",
        ground_truth="DO_NOT_READ_GOLD_PATCH",
        split="validation",
        metadata={
            "dataset_key": "swe_bench",
            "evaluator_payload": {
                "instance_id": "owner__repo-1",
                "repo": "owner/repo",
                "base_commit": "abc",
                "patch": "DO_NOT_READ_GOLD_PATCH",
            },
        },
    )

    with pytest.raises(ValueError, match="instance_id"):
        SWEbenchRepositoryIdentity.from_task_record(record)


def test_missing_base_commit_fails_before_returning_a_worktree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = root / "repos"
        worktrees = root / "worktrees"
        store.mkdir()
        worktrees.mkdir()
        _repository(store)

        with pytest.raises(SWEbenchWorktreeUnavailable, match="base_commit"):
            prepare_swebench_worktree(
                _identity("not-a-commit"),
                repository_store=store,
                worktree_root=worktrees,
            )

        assert list(worktrees.iterdir()) == []
