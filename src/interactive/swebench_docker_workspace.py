"""Official per-instance Docker workspace fallback for SWE-bench.

This is a narrow repository-environment adapter.  It reuses the official
SWE-bench ``make_test_spec``/``build_container`` lifecycle and the source tree
installed at ``/testbed`` in that image.  The source tree is copied exactly as
SkillFlow's ``swebench_official_worker.py::_copy_testbed`` does, reset to the
public ``base_commit``, and mounted back at ``/testbed`` in one task-scoped
persistent container.  FlowSteer's existing SkillFlow-derived
``RepositoryToolBackend`` continues to implement list/search/view/edit/diff;
only bash and test process execution are redirected into that container.

The official patch evaluator remains ``OfficialSWEbenchHarness`` and is not
called from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from .coding_tools import RepositoryToolBackend, _error
from .swe_worktree import SWEbenchRepositoryIdentity
from .tool_runtime import ToolRequest, ToolResult


SWEBENCH_DOCKER_WORKDIR = "/testbed"
SWEBENCH_DOCKER_RUNTIME_VERSION = "official-swebench-per-instance-docker.v1"


@dataclass(frozen=True, slots=True)
class DockerExecResult:
    returncode: int | None
    output: str
    timed_out: bool
    elapsed_seconds: float


def _docker_exec_with_timeout(
    container: Any,
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> DockerExecResult:
    """SkillFlow ``_exec_run_with_tolerant_decode`` with exit-code receipt."""

    chunks = bytearray()
    exec_id: str | None = None
    error: BaseException | None = None

    def execute() -> None:
        nonlocal exec_id, error
        try:
            created = container.client.api.exec_create(
                container.id,
                list(command),
                workdir=SWEBENCH_DOCKER_WORKDIR,
                environment={
                    "CUDA_VISIBLE_DEVICES": "",
                    "GIT_PAGER": "cat",
                    "PAGER": "cat",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            exec_id = str(created["Id"])
            for chunk in container.client.api.exec_start(exec_id, stream=True):
                chunks.extend(chunk)
        except BaseException as exc:  # surfaced on the caller thread
            error = exc

    started = time.monotonic()
    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if error is not None:
        raise error
    timed_out = thread.is_alive()
    if timed_out and exec_id is not None:
        inspected = container.client.api.exec_inspect(exec_id)
        pid = inspected.get("Pid")
        if isinstance(pid, int) and pid > 0:
            container.exec_run(["kill", "-TERM", str(pid)], detach=True)
        thread.join(5.0)
    returncode: int | None = None
    if exec_id is not None:
        inspected = container.client.api.exec_inspect(exec_id)
        candidate = inspected.get("ExitCode")
        if isinstance(candidate, int):
            returncode = candidate
    return DockerExecResult(
        returncode=returncode,
        output=chunks.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        elapsed_seconds=max(0.0, time.monotonic() - started),
    )


def _git(repo_root: Path, arguments: Sequence[str], timeout_seconds: float) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def _extract_testbed(container: Any, destination: Path) -> Path:
    archive, _stat = container.get_archive(SWEBENCH_DOCKER_WORKDIR)
    archive_path = destination / "testbed.tar"
    with archive_path.open("wb") as handle:
        for chunk in archive:
            handle.write(chunk)
    with tarfile.open(archive_path, "r:*") as payload:
        payload.extractall(destination)
    archive_path.unlink()
    candidates = (destination / "testbed", destination)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate.resolve()
    raise RuntimeError("official SWE-bench image has no /testbed Git repository")


@dataclass(slots=True)
class PreparedSWEbenchDockerWorkspace:
    """One persistent official-image container and its task-owned source tree."""

    identity: SWEbenchRepositoryIdentity
    repo_root: Path
    workspace_root: Path
    pinned_commit: str
    instance_image_key: str
    environment_image_key: str
    container_name: str
    container: Any = field(repr=False)
    client: Any = field(repr=False)
    cleanup_container: Callable[[Any, Any, Any], None] = field(repr=False)
    close_logger: Callable[[Any], None] = field(repr=False)
    logger: Any = field(repr=False)
    log_path: Path
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def receipt(self) -> Mapping[str, object]:
        return {
            "source": "official SWE-bench per-instance Docker image",
            "runtime_version": SWEBENCH_DOCKER_RUNTIME_VERSION,
            "instance_id": self.identity.instance_id,
            "repo": self.identity.repo,
            "base_commit": self.identity.base_commit,
            "observed_pinned_commit": self.pinned_commit,
            "instance_image_key": self.instance_image_key,
            "environment_image_key": self.environment_image_key,
            "container_name": self.container_name,
            "runtime_log_path": str(self.log_path),
            "workspace": SWEBENCH_DOCKER_WORKDIR,
            "task_workspace_path": str(self.repo_root),
            "workspace_isolation": "task_scoped_persistent_container",
            "base_state_verified": self.pinned_commit == self.identity.base_commit,
            "task_environment_ready": True,
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self.cleanup_container(self.client, self.container, self.logger)
        finally:
            try:
                self.client.close()
            finally:
                try:
                    self.close_logger(self.logger)
                finally:
                    resolved_root = self.workspace_root.resolve()
                    resolved_repository = self.repo_root.resolve()
                    owned_path = (
                        resolved_repository
                        if resolved_repository.parent == resolved_root
                        else resolved_repository.parent
                    )
                    if owned_path.parent != resolved_root:
                        raise RuntimeError(
                            "Docker workspace path is outside its task root"
                        )
                    shutil.rmtree(owned_path)
                    self._closed = True

    close = cleanup


class DockerRepositoryToolBackend(RepositoryToolBackend):
    """Existing repository Tool semantics with commands run in Docker."""

    def __init__(
        self,
        workspace: PreparedSWEbenchDockerWorkspace,
        *,
        timeout_seconds: float,
        task_issue: str,
        edit_generator: Callable[..., Any] | None,
        exec_runner: Callable[..., DockerExecResult] = _docker_exec_with_timeout,
    ) -> None:
        super().__init__(
            workspace.repo_root,
            max_test_timeout_seconds=timeout_seconds,
            task_environment_receipt=workspace.receipt,
            repository_state_receipt=workspace.receipt,
            task_issue=task_issue,
            edit_generator=edit_generator,
        )
        self._workspace = workspace
        self._exec_runner = exec_runner

    @property
    def repository_runtime_receipt(self) -> Mapping[str, object]:
        return {
            "repository": dict(self._workspace.receipt),
            "task_environment": dict(self._workspace.receipt),
            "task_environment_required": True,
        }

    def _run_in_testbed(self, command: str, timeout: float) -> DockerExecResult:
        return self._exec_runner(
            self._workspace.container,
            ("/bin/bash", "-c", command),
            timeout_seconds=timeout,
        )

    def _bash(self, arguments: Mapping[str, object]) -> ToolResult:
        command = arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return _error("bash", "command must be non-empty text")
        try:
            completed = self._run_in_testbed(
                command.strip(), self.max_test_timeout_seconds
            )
        except Exception as exc:
            return _error("bash", f"Docker exec failed: {exc}")
        return ToolResult(
            {
                "action": "bash",
                "ok": completed.returncode == 0 and not completed.timed_out,
                "command": command,
                "timed_out": completed.timed_out,
                "timeout_seconds": self.max_test_timeout_seconds,
                "returncode": completed.returncode,
                "stdout": completed.output[-5000:],
                "stderr": "",
                "output": self._truncate_bash_output(completed.output),
                "task_environment": dict(self._workspace.receipt),
            }
        )

    def _run_tests(self, arguments: Mapping[str, object]) -> ToolResult:
        test_cmd = arguments.get("test_cmd")
        if not isinstance(test_cmd, str) or not test_cmd.strip():
            return _error("run_tests", "test_cmd must be non-empty text")
        raw_timeout = arguments.get(
            "timeout_seconds", self.max_test_timeout_seconds
        )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return _error("run_tests", "timeout_seconds must be numeric")
        if timeout <= 0:
            return _error("run_tests", "timeout_seconds must be positive")
        timeout = min(timeout, self.max_test_timeout_seconds)
        try:
            completed = self._run_in_testbed(test_cmd.strip(), timeout)
        except Exception as exc:
            return _error("run_tests", f"Docker exec failed: {exc}")
        return ToolResult(
            {
                "action": "run_tests",
                "ok": not completed.timed_out,
                "passed": completed.returncode == 0 and not completed.timed_out,
                "timed_out": completed.timed_out,
                "timeout_seconds": timeout,
                "returncode": completed.returncode,
                "stdout": completed.output[-2000:],
                "stderr": "",
                "task_environment": dict(self._workspace.receipt),
            }
        )


def prepare_swebench_docker_workspace_for_task(
    record: object,
    *,
    harness: Any,
    workspace_root: Path | str,
    setup_timeout_seconds: float,
    cleanup_timeout_seconds: float,
    docker_client_factory: Callable[[], Any] | None = None,
    build_container_fn: Callable[..., Any] | None = None,
    cleanup_container_fn: Callable[[Any, Any, Any], None] | None = None,
) -> PreparedSWEbenchDockerWorkspace:
    """Create one official-image workspace without consulting solution fields."""

    del cleanup_timeout_seconds  # official cleanup_container owns this timeout
    identity = SWEbenchRepositoryIdentity.from_task_record(record)
    root = Path(workspace_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Docker workspace root is unavailable: {root}")

    module = harness._configured_module((record,))
    row = getattr(module, "_verified_cache", {}).get(identity.instance_id)
    if not isinstance(row, Mapping):
        raise RuntimeError("SWE-bench instance is absent from the official task cache")
    harness_path = harness.harness_path.expanduser().resolve()
    import sys

    if str(harness_path) not in sys.path:
        sys.path.insert(0, str(harness_path))
    from swebench.harness.constants import DOCKER_USER
    from docker.errors import ImageNotFound
    from swebench.harness.docker_build import (
        build_container,
        close_logger,
        setup_logger,
    )
    from swebench.harness.docker_utils import cleanup_container
    from swebench.harness.test_spec.test_spec import make_test_spec

    if docker_client_factory is None:
        import docker

        docker_client_factory = lambda: docker.from_env(
            timeout=max(60, int(setup_timeout_seconds))
        )
    build = build_container_fn or build_container
    cleanup = cleanup_container_fn or cleanup_container
    test_spec = make_test_spec(row, namespace=harness.docker_namespace)
    run_id = "flowsteer-workspace-" + uuid.uuid4().hex[:12]
    task_directory = Path(
        tempfile.mkdtemp(prefix="swe_docker_", dir=str(root))
    ).resolve()
    log_path = (
        root
        / "docker_logs"
        / f"{identity.instance_id}-{run_id}.log"
    ).resolve()
    logger = setup_logger(identity.instance_id, log_path)
    client = None
    transient = None
    persistent = None
    try:
        client = docker_client_factory()
        try:
            client.images.get(test_spec.instance_image_key)
        except ImageNotFound:
            # NECESSARY_ADAPTATION: the official remote image is an OCI image
            # index.  This task-scoped rootless daemon requires the official
            # platform to be explicit so the local tag is materialized.
            client.images.pull(
                test_spec.instance_image_key,
                platform=test_spec.platform,
            )
        transient = build(test_spec, client, run_id + "-seed", logger, False, False)
        transient.start()
        repo_root = _extract_testbed(transient, task_directory)
        cleanup(client, transient, logger)
        transient = None
        _git(repo_root, ("cat-file", "-e", identity.base_commit + "^{commit}"), setup_timeout_seconds)
        _git(repo_root, ("reset", "--hard", identity.base_commit), setup_timeout_seconds)
        _git(repo_root, ("clean", "-fdx"), setup_timeout_seconds)
        pinned_commit = _git(repo_root, ("rev-parse", "HEAD"), setup_timeout_seconds)
        if pinned_commit != identity.base_commit:
            raise RuntimeError("official Docker workspace does not match base_commit")

        run_args = test_spec.docker_specs.get("run_args", {})
        persistent = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=run_args.get("cap_add", []),
            volumes={
                str(repo_root): {
                    "bind": SWEBENCH_DOCKER_WORKDIR,
                    "mode": "rw",
                }
            },
        )
        persistent.start()
        observed = persistent.exec_run(
            ["git", "rev-parse", "HEAD"],
            workdir=SWEBENCH_DOCKER_WORKDIR,
        )
        observed_text = bytes(observed.output).decode("utf-8", errors="replace").strip()
        if observed.exit_code != 0 or observed_text != identity.base_commit:
            raise RuntimeError("persistent Docker workspace failed base-state check")
        return PreparedSWEbenchDockerWorkspace(
            identity=identity,
            repo_root=repo_root,
            workspace_root=root,
            pinned_commit=pinned_commit,
            instance_image_key=str(test_spec.instance_image_key),
            environment_image_key=str(test_spec.env_image_key),
            container_name=str(persistent.name),
            container=persistent,
            client=client,
            cleanup_container=cleanup,
            close_logger=close_logger,
            logger=logger,
            log_path=log_path,
        )
    except BaseException:
        try:
            if client is not None and persistent is not None:
                cleanup(client, persistent, logger)
            if client is not None and transient is not None:
                cleanup(client, transient, logger)
            if client is not None:
                client.close()
        finally:
            close_logger(logger)
            shutil.rmtree(task_directory, ignore_errors=True)
        raise


__all__ = [
    "DockerExecResult",
    "DockerRepositoryToolBackend",
    "PreparedSWEbenchDockerWorkspace",
    "SWEBENCH_DOCKER_RUNTIME_VERSION",
    "SWEBENCH_DOCKER_WORKDIR",
    "prepare_swebench_docker_workspace_for_task",
]
