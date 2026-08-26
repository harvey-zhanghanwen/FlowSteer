from __future__ import annotations

from pathlib import Path
import subprocess

from src.interactive.coding_tools import (
    SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
    create_swebench_repository_registration,
)
from src.interactive.swe_worktree import SWEbenchRepositoryIdentity
from src.interactive.swebench_docker_workspace import (
    DockerExecResult,
    DockerRepositoryToolBackend,
    PreparedSWEbenchDockerWorkspace,
    SWEBENCH_DOCKER_WORKDIR,
)
from src.interactive.tool_runtime import ToolRequest


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    (path / "sample.py").write_text(
        "def value():\n    return 'old'\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(path), "add", "sample.py"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _workspace(tmp_path: Path) -> PreparedSWEbenchDockerWorkspace:
    root = tmp_path / "workspaces"
    root.mkdir()
    task_root = root / "task"
    repo_root = task_root / "testbed"
    task_root.mkdir()
    commit = _git_repository(repo_root)
    identity = SWEbenchRepositoryIdentity(
        instance_id="pydata__xarray-7229",
        repo="pydata/xarray",
        base_commit=commit,
    )
    return PreparedSWEbenchDockerWorkspace(
        identity=identity,
        repo_root=repo_root,
        workspace_root=root,
        pinned_commit=commit,
        instance_image_key=(
            "swebench/sweb.eval.x86_64.pydata_1776_xarray-7229:latest"
        ),
        environment_image_key="sweb.env.py.x86_64.example:latest",
        container_name="sweb.eval.pydata_xarray-7229.flowsteer-test",
        container=object(),
        client=_FakeClient(),
        cleanup_container=lambda _client, _container, _logger: None,
        close_logger=lambda _logger: None,
        logger=object(),
        log_path=root / "runtime.log",
    )


def test_docker_backend_reuses_repository_tools_and_runs_commands_in_testbed(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    calls: list[tuple[tuple[str, ...], float]] = []

    def execute(_container, command, *, timeout_seconds):
        calls.append((tuple(command), timeout_seconds))
        return DockerExecResult(0, "command output", False, 0.1)

    backend = DockerRepositoryToolBackend(
        workspace,
        timeout_seconds=30,
        task_issue="Change the returned value.",
        edit_generator=None,
        exec_runner=execute,
    )

    search = backend.invoke(ToolRequest("search_code", {"query": "return"}))
    assert search.value["matches"][0]["path"] == "sample.py"
    viewed = backend.invoke(ToolRequest("view_file", {"path": "sample.py"}))
    assert viewed.value["lines"][1]["text"] == "    return 'old'"
    edited = backend.invoke(
        ToolRequest(
            "edit_file",
            {
                "path": "sample.py",
                "instruction": "replace 'old' with 'new'",
            },
        )
    )
    assert edited.value["ok"] is True
    assert "return 'new'" in (workspace.repo_root / "sample.py").read_text()
    diff = backend.invoke(ToolRequest("diff", {}))
    assert "-    return 'old'" in diff.value["diff"]
    assert "+    return 'new'" in diff.value["diff"]

    bash = backend.invoke(ToolRequest("bash", {"command": "git status --short"}))
    tests = backend.invoke(
        ToolRequest(
            "run_tests",
            {"test_cmd": "pytest -q", "timeout_seconds": 12},
        )
    )
    assert bash.value["ok"] is True
    assert tests.value["passed"] is True
    assert calls == [
        (("/bin/bash", "-c", "git status --short"), 30.0),
        (("/bin/bash", "-c", "pytest -q"), 12.0),
    ]
    assert workspace.receipt["workspace"] == SWEBENCH_DOCKER_WORKDIR
    assert workspace.receipt["instance_image_key"].endswith(
        "pydata_1776_xarray-7229:latest"
    )
    assert workspace.receipt["environment_image_key"].startswith("sweb.env.py")


def test_training_registration_accepts_the_task_environment_backend(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    backend = DockerRepositoryToolBackend(
        workspace,
        timeout_seconds=30,
        task_issue="Issue",
        edit_generator=None,
    )
    registration = create_swebench_repository_registration(
        None,
        action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
        require_task_environment=True,
        backend_override=backend,
    )
    assert registration.backend is backend
    assert set(registration.capability.action_names) == {
        "bash",
        "list_files",
        "search_code",
        "view_file",
        "edit_file",
        "run_tests",
    }


def test_workspace_cleanup_stops_container_closes_client_and_removes_task_tree(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    client = workspace.client
    repo_root = workspace.repo_root
    workspace.cleanup()
    assert workspace.closed is True
    assert client.closed is True
    assert not repo_root.exists()
    workspace.cleanup()
