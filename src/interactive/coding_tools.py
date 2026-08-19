"""Repository tools for SWE-bench coding Agents.

This module is a dependency-light port of the repository primitives in
SkillFlow ``training/environment.py``: ``_resolve_repo_path``,
``_handle_list_files``, ``_handle_search_code``, ``_handle_view_file``,
``_sre_str_replace``, ``_generate_workspace_diff``, and the local branch of
``_handle_run_tests``.  The backend is registered through this project's
SkillFlow-derived :mod:`src.interactive.tool_runtime` contracts.

SkillFlow's worktree setup and ``_run_tests_in_swe_env`` depend on its private
Verified-dataset cache, repository store, environment resolver, and monolithic
episode state.  They are intentionally not reproduced here.  Callers must pass
an already prepared repository root; official patch evaluation remains in
``swebench_adapter.py``.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Any, Mapping, Sequence

from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


SWEBENCH_REPOSITORY_TOOL_ID = "swebench_repository"
SWEBENCH_REPOSITORY_TOOL_VERSION = "skillflow.repository-tools.v1"
_SOURCE_SUFFIXES = frozenset({".py"})
_IGNORED_SOURCE_DIRECTORIES = frozenset(
    {".git", "doc", "docs", "example", "examples", "test", "tests", "testing"}
)


def _error(action: str, message: str) -> ToolResult:
    return ToolResult({"action": action, "ok": False, "error": message})


class RepositoryToolBackend:
    """Stateful ToolBackend bound to one already prepared repository root."""

    def __init__(
        self,
        repo_root: Path | str,
        *,
        max_view_lines: int = 140,
        max_search_matches: int = 50,
        max_test_timeout_seconds: float = 60.0,
    ) -> None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository root is unavailable: {root}")
        if max_view_lines <= 0 or max_search_matches <= 0:
            raise ValueError("view and search limits must be positive")
        if max_test_timeout_seconds <= 0:
            raise ValueError("test timeout must be positive")
        self.repo_root = root
        self.max_view_lines = int(max_view_lines)
        self.max_search_matches = int(max_search_matches)
        self.max_test_timeout_seconds = float(max_test_timeout_seconds)
        self._original_contents: dict[str, str] = {}
        self._lock = threading.RLock()

    def invoke(self, request: ToolRequest) -> ToolResult:
        handlers = {
            "list_files": self._list_files,
            "search_code": self._search_code,
            "view_file": self._view_file,
            "exact_edit": self._exact_edit,
            "diff": self._diff,
            "run_tests": self._run_tests,
        }
        handler = handlers.get(request.action)
        if handler is None:
            raise KeyError(f"unsupported repository action: {request.action}")
        with self._lock:
            return handler(request.arguments)

    def _resolve_repo_path(self, raw_path: object) -> tuple[Path, str]:
        path = str(raw_path or "").strip().rstrip("/")
        if not path:
            raise ValueError("path is required")
        if Path(path).is_absolute():
            raise ValueError("path must be repository-relative")
        candidate = (self.repo_root / path.lstrip("./")).resolve()
        try:
            relative = candidate.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise ValueError("path resolves outside the repository") from exc
        return candidate, relative

    def _source_paths(self, file_pattern: str = "") -> list[Path]:
        paths: list[Path] = []
        normalized_pattern = file_pattern.strip().lstrip("./")
        for path in self.repo_root.rglob("*"):
            if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(self.repo_root)
            if any(part in _IGNORED_SOURCE_DIRECTORIES for part in relative.parts[:-1]):
                continue
            relative_text = relative.as_posix()
            if normalized_pattern and not (
                fnmatch.fnmatch(relative_text, normalized_pattern)
                or fnmatch.fnmatch(path.name, normalized_pattern)
                or fnmatch.fnmatch(relative_text, f"*{normalized_pattern}*")
            ):
                continue
            paths.append(path)
        return sorted(paths, key=lambda item: item.relative_to(self.repo_root).as_posix())

    def _list_files(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_pattern = arguments.get("file_pattern", "")
        if not isinstance(raw_pattern, str):
            return _error("list_files", "file_pattern must be text")
        files = [path.relative_to(self.repo_root).as_posix() for path in self._source_paths(raw_pattern)]
        return ToolResult(
            {
                "action": "list_files",
                "ok": True,
                "files": files,
                "count": len(files),
            }
        )

    def _search_code(self, arguments: Mapping[str, object]) -> ToolResult:
        query = arguments.get("query", "")
        file_pattern = arguments.get("file_pattern", "")
        if not isinstance(query, str) or not query.strip():
            return _error("search_code", "query must be non-empty text")
        if not isinstance(file_pattern, str):
            return _error("search_code", "file_pattern must be text")
        try:
            expression = re.compile(query, flags=re.IGNORECASE)
        except re.error:
            expression = re.compile(re.escape(query), flags=re.IGNORECASE)
        matches: list[dict[str, object]] = []
        for path in self._source_paths(file_pattern):
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if expression.search(line) is None:
                    continue
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "text": line.rstrip()[:150],
                    }
                )
                if len(matches) >= self.max_search_matches:
                    break
            if len(matches) >= self.max_search_matches:
                break
        return ToolResult(
            {
                "action": "search_code",
                "ok": True,
                "query": query,
                "file_pattern": file_pattern,
                "matches": matches,
                "match_count": len(matches),
            }
        )

    def _view_file(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error("view_file", str(exc))
        if path.is_dir():
            entries = sorted(item.name for item in path.iterdir())[:80]
            return ToolResult(
                {
                    "action": "view_file",
                    "ok": True,
                    "path": relative,
                    "kind": "directory",
                    "entries": entries,
                }
            )
        if not path.is_file():
            return _error("view_file", f"file not found: {relative}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error("view_file", f"cannot read {relative}: {exc}")
        lines = content.splitlines()
        total = len(lines)
        try:
            start_line = int(arguments.get("start_line", 1))
            end_line = int(arguments.get("end_line", max(total, 1)))
        except (TypeError, ValueError):
            return _error("view_file", "line bounds must be integers")
        start_line = max(1, start_line)
        end_line = max(start_line, end_line)
        end_line = min(end_line, start_line + self.max_view_lines - 1, max(total, 1))
        selected = [
            {"line": index, "text": lines[index - 1]}
            for index in range(start_line, end_line + 1)
            if index <= total
        ]
        return ToolResult(
            {
                "action": "view_file",
                "ok": True,
                "path": relative,
                "kind": "file",
                "start_line": start_line,
                "end_line": end_line if total else 0,
                "total_lines": total,
                "lines": selected,
            }
        )

    def _exact_edit(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error("exact_edit", str(exc))
        old_str = arguments.get("old_str", "")
        new_str = arguments.get("new_str", "")
        if not isinstance(old_str, str) or not old_str:
            return _error("exact_edit", "old_str must be non-empty text")
        if not isinstance(new_str, str):
            return _error("exact_edit", "new_str must be text")
        if not path.is_file():
            return _error("exact_edit", f"file not found: {relative}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error("exact_edit", f"cannot read {relative}: {exc}")
        match_count = content.count(old_str)
        if match_count == 0:
            return _error("exact_edit", "old_str was not found exactly")
        if match_count > 1:
            return _error("exact_edit", f"old_str appears {match_count} times; include more context")
        updated = content.replace(old_str, new_str, 1)
        if path.suffix == ".py":
            try:
                compile(updated, relative, "exec")
            except SyntaxError as exc:
                return _error(
                    "exact_edit",
                    f"Python syntax error at line {exc.lineno}: {exc.msg}",
                )
        self._original_contents.setdefault(relative, content)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _error("exact_edit", f"cannot write {relative}: {exc}")
        line_number = content[: content.index(old_str)].count("\n") + 1
        updated_lines = updated.splitlines()
        snippet_start = max(1, line_number - 3)
        snippet_end = min(len(updated_lines), line_number + new_str.count("\n") + 3)
        snippet = [
            {"line": index, "text": updated_lines[index - 1]}
            for index in range(snippet_start, snippet_end + 1)
        ]
        return ToolResult(
            {
                "action": "exact_edit",
                "ok": True,
                "path": relative,
                "changed": updated != content,
                "line": line_number,
                "snippet": snippet,
            }
        )

    def _workspace_diff(self) -> str:
        parts: list[str] = []
        for relative in sorted(self._original_contents):
            original = self._original_contents[relative]
            path = self.repo_root / relative
            current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            if original == current:
                continue
            unified = difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
            parts.append(f"diff --git a/{relative} b/{relative}\n" + "".join(unified))
        return "\n".join(parts)

    def _diff(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        diff = self._workspace_diff()
        return ToolResult(
            {
                "action": "diff",
                "ok": True,
                "diff": diff,
                "changed": bool(diff.strip()),
            }
        )

    def _run_tests(self, arguments: Mapping[str, object]) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            return _error("run_tests", "command must be a non-empty argument array")
        argv = list(command)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            return _error("run_tests", "command must contain non-empty text arguments")
        raw_timeout = arguments.get("timeout_seconds", self.max_test_timeout_seconds)
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return _error("run_tests", "timeout_seconds must be numeric")
        if timeout <= 0:
            return _error("run_tests", "timeout_seconds must be positive")
        timeout = min(timeout, self.max_test_timeout_seconds)
        try:
            # A fresh cache prefix prevents same-second, same-size source
            # edits from reusing a stale Python bytecode file during the next
            # targeted test.  It does not alter the prepared repository.
            with tempfile.TemporaryDirectory(prefix="flowsteer-pycache-") as cache:
                completed = subprocess.run(
                    argv,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "PYTHONPYCACHEPREFIX": cache,
                    },
                )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                {
                    "action": "run_tests",
                    "ok": True,
                    "passed": False,
                    "timed_out": True,
                    "timeout_seconds": timeout,
                    "stdout": str(exc.stdout or "")[-2000:],
                    "stderr": str(exc.stderr or "")[-1000:],
                }
            )
        except OSError as exc:
            return _error("run_tests", f"test process could not start: {exc}")
        return ToolResult(
            {
                "action": "run_tests",
                "ok": True,
                "passed": completed.returncode == 0,
                "timed_out": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-1000:],
            }
        )


def create_swebench_repository_registration(
    repo_root: Path | str,
    *,
    tool_id: str = SWEBENCH_REPOSITORY_TOOL_ID,
    dataset_scope: tuple[str, ...] = ("swe_bench",),
    timeout_seconds: float = 60.0,
    version: str = SWEBENCH_REPOSITORY_TOOL_VERSION,
) -> ToolRegistration:
    """Create one registry entry for a prepared SWE-bench repository."""

    backend = RepositoryToolBackend(
        repo_root,
        max_test_timeout_seconds=timeout_seconds,
    )
    capability = ToolCapability(
        tool_id=tool_id,
        dataset_scope=dataset_scope,
        input_schema={
            "type": "object",
            "actions": [
                "list_files",
                "search_code",
                "view_file",
                "exact_edit",
                "diff",
                "run_tests",
            ],
        },
        output_schema={"type": "object"},
        side_effect="repository_read_write_and_test_process",
        timeout_seconds=timeout_seconds,
        version=version,
        availability=True,
    )
    return ToolRegistration(tool_id, backend, capability)


def create_swebench_repository_registry(
    repo_root: Path | str,
    **registration_options: Any,
) -> ToolRegistry:
    """Create a ToolRegistry containing one prepared repository backend."""

    return ToolRegistry(
        (create_swebench_repository_registration(repo_root, **registration_options),)
    )


__all__ = [
    "RepositoryToolBackend",
    "SWEBENCH_REPOSITORY_TOOL_ID",
    "SWEBENCH_REPOSITORY_TOOL_VERSION",
    "create_swebench_repository_registration",
    "create_swebench_repository_registry",
]
