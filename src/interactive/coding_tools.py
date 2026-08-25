"""Repository tools for SWE-bench coding Agents.

This module is a dependency-light port of the repository primitives in
SkillFlow ``training/environment.py``: ``_resolve_repo_path``,
``_handle_list_files``, ``_handle_search_code``, ``_handle_view_file``,
``_handle_bash``, ``_handle_str_replace_editor`` and its create/replace/insert/
undo primitives, ``_generate_filemap``, ``_generate_workspace_diff``, and the
local branch of ``_handle_run_tests``.  The backend is registered through this
project's SkillFlow-derived :mod:`src.interactive.tool_runtime` contracts.

The ``apply_patch`` action delegates to the local official Codex CLI's
``--codex-run-as-apply-patch`` entry point instead of reimplementing Codex's
patch grammar or hunk matcher.

SkillFlow's worktree setup and task-environment lookup remain in the thin
``swe_worktree.py`` and ``swebench_adapter.py`` boundaries.  Callers pass an
already prepared repository root plus the command prefix resolved through
SkillFlow's ``_env_python(repo, version)`` contract.  Official patch evaluation
remains in ``swebench_adapter.py``.
"""

from __future__ import annotations

import ast
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
SWEBENCH_REPOSITORY_TOOL_VERSION = "skillflow.repository-tools.v2"
SWEBENCH_SKILLFLOW_TRAINING_TOOL_VERSION = (
    "skillflow.training.repository-tools.v1+flowsteer.workspace-diff.v1"
)
SWEBENCH_TOOL_PROFILE_COMPATIBILITY = "compatibility_v2"
SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING = "skillflow_training_v1"
SWEBENCH_TOOL_PROFILES = frozenset(
    {
        SWEBENCH_TOOL_PROFILE_COMPATIBILITY,
        SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
    }
)
CODEX_APPLY_PATCH_EXECUTABLE = "/home/test/.local/bin/codex"
_SOURCE_SUFFIXES = frozenset({".py"})
_IGNORED_SOURCE_DIRECTORIES = frozenset(
    {".git", "doc", "docs", "example", "examples", "test", "tests", "testing"}
)
_MAX_BASH_OUTPUT = 10_000


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
        task_command_prefix: Sequence[str] = (),
        task_environment_receipt: Mapping[str, object] | None = None,
        require_task_environment: bool = False,
        repository_state_receipt: Mapping[str, object] | None = None,
    ) -> None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository root is unavailable: {root}")
        if max_view_lines <= 0 or max_search_matches <= 0:
            raise ValueError("view and search limits must be positive")
        if max_test_timeout_seconds <= 0:
            raise ValueError("test timeout must be positive")
        prefix = tuple(task_command_prefix)
        if any(not isinstance(item, str) or not item for item in prefix):
            raise ValueError("task command prefix must contain non-empty text")
        if type(require_task_environment) is not bool:
            raise TypeError("require_task_environment must be boolean")
        if require_task_environment and not prefix:
            raise ValueError("task-specific SWE-bench environment is required")
        self.repo_root = root
        self.max_view_lines = int(max_view_lines)
        self.max_search_matches = int(max_search_matches)
        self.max_test_timeout_seconds = float(max_test_timeout_seconds)
        self._task_command_prefix = prefix
        self._task_environment_receipt = dict(task_environment_receipt or {})
        self._require_task_environment = require_task_environment
        self._repository_state_receipt = dict(repository_state_receipt or {})
        # ``None`` records that a path did not exist before the first edit.
        # This lets the workspace diff represent SkillFlow ``create`` actions.
        self._original_contents: dict[str, str | None] = {}
        # SkillFlow's ``str_replace_editor`` stores one previous revision per
        # path and consumes it on ``undo_edit``.
        self._edit_history: dict[str, str] = {}
        self._lock = threading.RLock()

    @property
    def repository_runtime_receipt(self) -> Mapping[str, object]:
        """Return the task repository/environment identity persisted in traces."""

        return {
            "repository": dict(self._repository_state_receipt),
            "task_environment": dict(self._task_environment_receipt),
            "task_environment_required": self._require_task_environment,
        }

    def _shell_invocation(self, command: str) -> tuple[str | list[str], bool]:
        if self._task_command_prefix:
            return [*self._task_command_prefix, "bash", "-lc", command], False
        return command, True

    def invoke(self, request: ToolRequest) -> ToolResult:
        handlers = {
            "apply_patch": self._apply_patch,
            "bash": self._bash,
            "list_files": self._list_files,
            "search_code": self._search_code,
            "str_replace_editor": self._str_replace_editor,
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
        # ``str.lstrip('./')`` removes every leading dot character, not one
        # optional ``./`` path prefix.  It therefore changed legitimate paths
        # such as ``.pyinstaller/hooks/...`` into a different repository path.
        # SkillFlow returns those relative names from list_files, so preserve
        # them and remove only explicit current-directory segments.
        normalized_path = path
        while normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]
        candidate = (self.repo_root / normalized_path).resolve()
        try:
            relative = candidate.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise ValueError("path resolves outside the repository") from exc
        return candidate, relative

    @staticmethod
    def _matches_file_pattern(path: Path, relative: str, file_pattern: str) -> bool:
        """Match SkillFlow's explicit ``file_pattern`` forms.

        An explicit pattern addresses repository files directly, so it is not
        restricted to Python source and does not inherit the default
        tests/docs exclusions.
        """

        pattern = file_pattern.strip().lstrip("./")
        if not pattern:
            return True
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
        if "/" not in pattern:
            return (
                fnmatch.fnmatch(path.name, pattern if "." in pattern else f"*{pattern}*")
                or pattern in relative
            )
        return fnmatch.fnmatch(relative, f"*{pattern}*")

    def _source_paths(self, file_pattern: str = "") -> list[Path]:
        paths: list[Path] = []
        normalized_pattern = file_pattern.strip().lstrip("./")
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo_root)
            if ".git" in relative.parts:
                continue
            relative_text = relative.as_posix()
            if normalized_pattern:
                if not self._matches_file_pattern(path, relative_text, normalized_pattern):
                    continue
            elif (
                path.suffix not in _SOURCE_SUFFIXES
                or any(
                    part in _IGNORED_SOURCE_DIRECTORIES
                    for part in relative.parts[:-1]
                )
            ):
                continue
            paths.append(path)
        return sorted(paths, key=lambda item: item.relative_to(self.repo_root).as_posix())

    def _list_files(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_pattern = arguments.get("file_pattern", "")
        if not isinstance(raw_pattern, str):
            return _error("list_files", "file_pattern must be text")
        files = [
            path.relative_to(self.repo_root).as_posix()
            for path in self._source_paths(raw_pattern)
        ]
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

    @staticmethod
    def _generate_filemap(content: str, path: str) -> str | None:
        """Return SkillFlow's AST file structure for a long Python file."""

        del path
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        lines = content.split("\n")
        result: list[tuple[float, str]] = []
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            start = node.lineno
            end = node.end_lineno or start
            body_len = end - start
            signature = lines[start - 1].rstrip()
            result.append((float(start), f"{start:4d} | {signature}"))
            minimum_body = 5 if isinstance(node, ast.ClassDef) else 3
            if body_len > minimum_body:
                result.append(
                    (start + 0.5, f"     | ...({body_len} lines)")
                )

        imports = []
        for index, line in enumerate(lines[:15], start=1):
            if line.strip().startswith(("import ", "from ")):
                imports.append(f"{index:4d} | {line.rstrip()}")
        if not result and not imports:
            return None

        result.sort(key=lambda item: item[0])
        output = "\n".join(imports[:5])
        if imports:
            output += "\n     | ..."
        if result:
            output += ("\n" if output else "") + "\n".join(
                entry for _, entry in result
            )
        return output

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
        has_explicit_range = "start_line" in arguments or "end_line" in arguments
        if not has_explicit_range and total > 500 and path.suffix == ".py":
            filemap = self._generate_filemap(content, relative)
            if filemap:
                return ToolResult(
                    {
                        "action": "view_file",
                        "ok": True,
                        "path": relative,
                        "kind": "filemap",
                        "total_lines": total,
                        "filemap": filemap,
                    }
                )
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

    def _remember_original(self, relative: str, content: str | None) -> None:
        if relative not in self._original_contents:
            self._original_contents[relative] = content

    def _replace_file(
        self,
        arguments: Mapping[str, object],
        *,
        action: str,
        editor_command: str | None = None,
    ) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error(action, str(exc))
        old_str = arguments.get("old_str", "")
        new_str = arguments.get("new_str", "")
        if not isinstance(old_str, str) or not old_str:
            return _error(action, "old_str must be non-empty text")
        if not isinstance(new_str, str):
            return _error(action, "new_str must be text")
        if not path.is_file():
            return _error(action, f"file not found: {relative}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error(action, f"cannot read {relative}: {exc}")
        match_count = content.count(old_str)
        if match_count == 0:
            return _error(action, "old_str was not found exactly")
        if match_count > 1:
            return _error(
                action,
                f"old_str appears {match_count} times; include more context",
            )
        updated = content.replace(old_str, new_str, 1)
        if path.suffix == ".py":
            try:
                compile(updated, relative, "exec")
            except SyntaxError as exc:
                return _error(
                    action,
                    f"Python syntax error at line {exc.lineno}: {exc.msg}",
                )
        self._remember_original(relative, content)
        self._edit_history[str(path)] = content
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _error(action, f"cannot write {relative}: {exc}")
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
                "action": action,
                "ok": True,
                "path": relative,
                "changed": updated != content,
                "line": line_number,
                "snippet": snippet,
                **(
                    {"command": editor_command}
                    if editor_command is not None
                    else {}
                ),
            }
        )

    def _exact_edit(self, arguments: Mapping[str, object]) -> ToolResult:
        """Compatibility alias for SkillFlow ``str_replace``."""

        return self._replace_file(arguments, action="exact_edit")

    def _create_file(
        self,
        arguments: Mapping[str, object],
        *,
        action: str,
    ) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error(action, str(exc))
        file_text = arguments.get("file_text", "")
        if not isinstance(file_text, str):
            return _error(action, "file_text must be text")
        if path.exists():
            return _error(
                action,
                f"{relative} already exists; use str_replace to edit it",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._remember_original(relative, None)
            path.write_text(file_text, encoding="utf-8")
        except OSError as exc:
            return _error(action, f"cannot create {relative}: {exc}")
        return ToolResult(
            {
                "action": action,
                "command": "create",
                "ok": True,
                "path": relative,
                "changed": True,
            }
        )

    def _insert_file(
        self,
        arguments: Mapping[str, object],
        *,
        action: str,
    ) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error(action, str(exc))
        if not path.is_file():
            return _error(action, f"file not found: {relative}")
        if "insert_line" not in arguments:
            return _error(action, "insert_line is required")
        try:
            insert_line = int(arguments["insert_line"])
        except (TypeError, ValueError):
            return _error(action, "insert_line must be an integer")
        new_str = arguments.get("new_str", "")
        if not isinstance(new_str, str):
            return _error(action, "new_str must be text")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
        except OSError as exc:
            return _error(action, f"cannot read {relative}: {exc}")
        content = "".join(lines)
        insert_line = max(0, min(len(lines), insert_line))
        inserted_lines = [line + "\n" for line in new_str.split("\n")]
        updated_lines = list(lines)
        updated_lines[insert_line:insert_line] = inserted_lines
        updated = "".join(updated_lines)
        self._remember_original(relative, content)
        self._edit_history[str(path)] = content
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _error(action, f"cannot write {relative}: {exc}")
        return ToolResult(
            {
                "action": action,
                "command": "insert",
                "ok": True,
                "path": relative,
                "changed": updated != content,
                "insert_line": insert_line,
            }
        )

    def _undo_edit(
        self,
        arguments: Mapping[str, object],
        *,
        action: str,
    ) -> ToolResult:
        try:
            path, relative = self._resolve_repo_path(arguments.get("path", ""))
        except ValueError as exc:
            return _error(action, str(exc))
        history_key = str(path)
        if history_key not in self._edit_history:
            return _error(action, f"no edit history for {relative}")
        current = (
            path.read_text(encoding="utf-8", errors="replace")
            if path.is_file()
            else ""
        )
        restored = self._edit_history.pop(history_key)
        try:
            path.write_text(restored, encoding="utf-8")
        except OSError as exc:
            return _error(action, f"cannot undo {relative}: {exc}")
        return ToolResult(
            {
                "action": action,
                "command": "undo_edit",
                "ok": True,
                "path": relative,
                "changed": current != restored,
            }
        )

    def _str_replace_editor(self, arguments: Mapping[str, object]) -> ToolResult:
        command = arguments.get("command", "")
        if not isinstance(command, str):
            return _error("str_replace_editor", "command must be text")
        if command == "str_replace":
            return self._replace_file(
                arguments,
                action="str_replace_editor",
                editor_command="str_replace",
            )
        if command == "create":
            return self._create_file(arguments, action="str_replace_editor")
        if command == "insert":
            return self._insert_file(arguments, action="str_replace_editor")
        if command == "undo_edit":
            return self._undo_edit(arguments, action="str_replace_editor")
        if command == "view":
            view_arguments: dict[str, object] = {"path": arguments.get("path", "")}
            view_range = arguments.get("view_range")
            if isinstance(view_range, Sequence) and not isinstance(
                view_range,
                (str, bytes),
            ):
                values = list(view_range)
                if values:
                    view_arguments["start_line"] = values[0]
                if len(values) > 1 and values[1] != -1:
                    view_arguments["end_line"] = values[1]
            result = self._view_file(view_arguments)
            value = result.value
            if isinstance(value, dict):
                return ToolResult(
                    {
                        **value,
                        "action": "str_replace_editor",
                        "command": "view",
                    }
                )
            return result
        return _error(
            "str_replace_editor",
            "unknown command; use view, create, str_replace, insert, or undo_edit",
        )

    @staticmethod
    def _truncate_bash_output(output: str) -> str:
        if len(output) <= _MAX_BASH_OUTPUT:
            return output
        half = _MAX_BASH_OUTPUT // 2
        omitted = len(output) - _MAX_BASH_OUTPUT
        return (
            output[:half]
            + f"\n\n... ({omitted} chars truncated) ...\n\n"
            + output[-half:]
        )

    def _bash(self, arguments: Mapping[str, object]) -> ToolResult:
        """Thin ToolResult adapter over SkillFlow ``_handle_bash``."""

        command = arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return _error("bash", "command must be non-empty text")
        if any(
            marker in command
            for marker in ("rm -rf /", "mkfs", "dd if=", "> /dev/")
        ):
            return _error("bash", "command rejected by the SkillFlow bash contract")
        invocation, use_shell = self._shell_invocation(command)
        try:
            completed = subprocess.run(
                invocation,
                shell=use_shell,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.max_test_timeout_seconds,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PAGER": "cat",
                    "GIT_PAGER": "cat",
                },
            )
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            return ToolResult(
                {
                    "action": "bash",
                    "ok": True,
                    "command": command,
                    "timed_out": True,
                    "timeout_seconds": self.max_test_timeout_seconds,
                    "returncode": None,
                    "stdout": stdout[-5000:],
                    "stderr": stderr[-5000:],
                    "output": self._truncate_bash_output(stdout + stderr),
                    "task_environment": dict(self._task_environment_receipt),
                }
            )
        except OSError as exc:
            return _error("bash", f"command process could not start: {exc}")
        output = self._truncate_bash_output(completed.stdout + completed.stderr)
        return ToolResult(
            {
                "action": "bash",
                "ok": True,
                "command": command,
                "timed_out": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-5000:],
                "stderr": completed.stderr[-5000:],
                "output": output,
                "task_environment": dict(self._task_environment_receipt),
            }
        )

    def _repository_text_snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(self.repo_root)
            if ".git" in relative_path.parts:
                continue
            try:
                snapshot[relative_path.as_posix()] = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue
        return snapshot

    def _record_snapshot_changes(
        self,
        before: Mapping[str, str],
        after: Mapping[str, str],
    ) -> list[str]:
        changed_paths = sorted(
            relative
            for relative in set(before) | set(after)
            if before.get(relative) != after.get(relative)
        )
        for relative in changed_paths:
            self._remember_original(relative, before.get(relative))
        return changed_paths

    def _apply_patch(self, arguments: Mapping[str, object]) -> ToolResult:
        """Invoke the local official Codex apply-patch entry point."""

        patch = arguments.get("patch", "")
        if not isinstance(patch, str) or not patch.strip():
            return _error("apply_patch", "patch must be non-empty text")
        before = self._repository_text_snapshot()
        try:
            completed = subprocess.run(
                [
                    CODEX_APPLY_PATCH_EXECUTABLE,
                    "--codex-run-as-apply-patch",
                    patch,
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.max_test_timeout_seconds,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PAGER": "cat",
                    "GIT_PAGER": "cat",
                },
            )
        except subprocess.TimeoutExpired as exc:
            after = self._repository_text_snapshot()
            changed_paths = self._record_snapshot_changes(before, after)
            return ToolResult(
                {
                    "action": "apply_patch",
                    "ok": False,
                    "applied": False,
                    "changed": bool(changed_paths),
                    "changed_paths": changed_paths,
                    "timed_out": True,
                    "timeout_seconds": self.max_test_timeout_seconds,
                    "stdout": str(exc.stdout or "")[-5000:],
                    "stderr": str(exc.stderr or "")[-5000:],
                }
            )
        except OSError as exc:
            return _error("apply_patch", f"Codex apply_patch could not start: {exc}")
        after = self._repository_text_snapshot()
        changed_paths = self._record_snapshot_changes(before, after)
        applied = completed.returncode == 0
        return ToolResult(
            {
                "action": "apply_patch",
                "ok": applied,
                "applied": applied,
                "changed": bool(changed_paths),
                "changed_paths": changed_paths,
                "timed_out": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-5000:],
                "stderr": completed.stderr[-5000:],
            }
        )

    def _workspace_diff(self) -> str:
        """Materialize the patch from the current repository state.

        DIRECT_REUSE: SkillFlow
        ``training/environment.py::_generate_workspace_diff`` reads the
        detached worktree with ``git diff`` and excludes test files.  The
        tracked-worktree path is authoritative because a legitimate
        repository command may change a file without going through one of the
        editor helpers below.  The dependency-light difflib path is retained
        only for synthetic unit-test directories that are not Git worktrees.
        """

        try:
            completed = subprocess.run(
                [
                    "git",
                    "diff",
                    "--",
                    ".",
                    ":(exclude)tests/",
                    ":(exclude)*/tests/",
                    ":(exclude)test_*",
                    ":(exclude)*/test_*",
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            diff = completed.stdout
            # ``git diff`` omits untracked files.  SkillFlow's deployed
            # ``str_replace_editor.create`` and ``bash`` actions can
            # legitimately create a source file.  Preserve the upstream test
            # exclusions and append each non-test untracked file as a normal
            # ``/dev/null`` unified diff so the final repository patch is
            # complete rather than silently dropping that artifact.
            try:
                untracked = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                    cwd=self.repo_root,
                    capture_output=True,
                    timeout=10,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                )
            except (OSError, subprocess.TimeoutExpired):
                untracked = None
            if untracked is not None and untracked.returncode == 0:
                for raw_relative in untracked.stdout.split(b"\0"):
                    if not raw_relative:
                        continue
                    relative = raw_relative.decode("utf-8", errors="replace")
                    parts = Path(relative).parts
                    if (
                        "tests" in parts[:-1]
                        or (parts and parts[-1].startswith("test_"))
                    ):
                        continue
                    path = self.repo_root / relative
                    if not path.is_file() or path.is_symlink():
                        continue
                    created = subprocess.run(
                        ["git", "diff", "--no-index", "--", "/dev/null", relative],
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                        errors="replace",
                        timeout=10,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    )
                    if created.returncode in {0, 1} and created.stdout:
                        diff += created.stdout
            if diff and not diff.endswith("\n"):
                diff += "\n"
            return diff

        parts: list[str] = []
        for relative in sorted(self._original_contents):
            original = self._original_contents[relative]
            path = self.repo_root / relative
            current = (
                path.read_text(encoding="utf-8", errors="replace")
                if path.is_file()
                else None
            )
            if original == current:
                continue
            unified = difflib.unified_diff(
                [] if original is None else original.splitlines(keepends=True),
                [] if current is None else current.splitlines(keepends=True),
                fromfile="/dev/null" if original is None else f"a/{relative}",
                tofile="/dev/null" if current is None else f"b/{relative}",
            )
            metadata = ""
            if original is None:
                metadata = "new file mode 100644\n"
            elif current is None:
                metadata = "deleted file mode 100644\n"
            parts.append(
                f"diff --git a/{relative} b/{relative}\n"
                + metadata
                + "".join(unified)
            )
        return "\n".join(parts)

    def materialize_workspace_diff(self) -> str:
        """Return the current task worktree patch without a model Tool call."""

        with self._lock:
            return self._workspace_diff()

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
        # DIRECT_REUSE: SkillFlow's deployed SWE-bench schema names the
        # argument ``test_cmd`` and executes it as a command string.  The
        # legacy project profile accepted an argv list under ``command``;
        # retain that branch only for frozen older conditions.
        test_cmd = arguments.get("test_cmd")
        command = arguments.get("command")
        if isinstance(test_cmd, str) and test_cmd.strip():
            invocation, use_shell = self._shell_invocation(test_cmd.strip())
        elif isinstance(command, Sequence) and not isinstance(
            command, (str, bytes)
        ):
            argv = list(command)
            if not argv or any(
                not isinstance(item, str) or not item for item in argv
            ):
                return _error(
                    "run_tests", "command must contain non-empty text arguments"
                )
            invocation = argv
            use_shell = False
        else:
            return _error(
                "run_tests",
                "test_cmd must be non-empty text or command must be an argument array",
            )
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
                    invocation,
                    shell=use_shell,
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
                    "task_environment": dict(self._task_environment_receipt),
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
                "task_environment": dict(self._task_environment_receipt),
            }
        )


def create_swebench_repository_registration(
    repo_root: Path | str,
    *,
    tool_id: str = SWEBENCH_REPOSITORY_TOOL_ID,
    dataset_scope: tuple[str, ...] = ("swe_bench",),
    timeout_seconds: float = 60.0,
    version: str | None = None,
    action_profile: str = SWEBENCH_TOOL_PROFILE_COMPATIBILITY,
    task_command_prefix: Sequence[str] = (),
    task_environment_receipt: Mapping[str, object] | None = None,
    require_task_environment: bool = False,
    repository_state_receipt: Mapping[str, object] | None = None,
) -> ToolRegistration:
    """Create one registry entry for a prepared SWE-bench repository."""

    if action_profile not in SWEBENCH_TOOL_PROFILES:
        raise ValueError("unsupported SWE-bench repository Tool profile")
    if action_profile == SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING:
        if require_task_environment is not True or not tuple(task_command_prefix):
            raise ValueError(
                "SkillFlow SWE-bench training Tool profile requires a "
                "task-specific command environment"
            )

    backend = RepositoryToolBackend(
        repo_root,
        max_test_timeout_seconds=timeout_seconds,
        task_command_prefix=task_command_prefix,
        task_environment_receipt=task_environment_receipt,
        require_task_environment=require_task_environment,
        repository_state_receipt=repository_state_receipt,
    )
    compatibility_action_schemas = {
        "apply_patch": {
            "type": "object",
            "additionalProperties": False,
            "required": ["patch"],
            "properties": {
                "patch": {"type": "string", "minLength": 1},
            },
        },
        "bash": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command"],
            "properties": {
                "command": {"type": "string", "minLength": 1},
            },
        },
        "list_files": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"file_pattern": {"type": "string"}},
        },
        "search_code": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "file_pattern": {"type": "string"},
            },
        },
        "view_file": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
        },
        "exact_edit": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "old_str", "new_str"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_str": {"type": "string", "minLength": 1},
                "new_str": {"type": "string"},
            },
        },
        "str_replace_editor": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command", "path"],
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "view",
                        "create",
                        "str_replace",
                        "insert",
                        "undo_edit",
                    ],
                },
                "path": {"type": "string", "minLength": 1},
                "file_text": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        },
        "diff": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "run_tests": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "timeout_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
            },
        },
    }
    # NECESSARY_ADAPTATION: deployed SkillFlow's code_generation mask exposes
    # list/search/view/edit_file, but edit_file is inseparable from its private
    # M_exec episode object and natural-language edit generator.  Reuse the
    # same deployed source's deterministic str_replace_editor instead of
    # inventing another editor.  bash and run_tests are existing deployed
    # handler/schema pairs and are explicitly enabled for repository command
    # and test execution in this evaluation adapter.
    # Patch publication remains the upstream internal workspace-diff
    # materialization, not a model-visible ``diff`` action.
    skillflow_training_action_schemas = {
        "bash": compatibility_action_schemas["bash"],
        "list_files": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "search_code": compatibility_action_schemas["search_code"],
        "view_file": compatibility_action_schemas["view_file"],
        "str_replace_editor": {
            **compatibility_action_schemas["str_replace_editor"],
            "properties": {
                **compatibility_action_schemas["str_replace_editor"]["properties"],
                "command": {
                    "type": "string",
                    "enum": [
                        "view",
                        "create",
                        "str_replace",
                        "insert",
                        "undo_edit",
                    ],
                },
            },
        },
        "run_tests": {
            "type": "object",
            "additionalProperties": False,
            "required": ["test_cmd"],
            "properties": {
                "test_cmd": {"type": "string", "minLength": 1},
            },
        },
    }
    action_schemas = (
        skillflow_training_action_schemas
        if action_profile == SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING
        else compatibility_action_schemas
    )
    if version is None:
        version = (
            SWEBENCH_SKILLFLOW_TRAINING_TOOL_VERSION
            if action_profile == SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING
            else SWEBENCH_REPOSITORY_TOOL_VERSION
        )
    capability = ToolCapability(
        tool_id=tool_id,
        dataset_scope=dataset_scope,
        action_schemas=action_schemas,
        input_schema={"type": "object"},
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
    "SWEBENCH_SKILLFLOW_TRAINING_TOOL_VERSION",
    "SWEBENCH_TOOL_PROFILE_COMPATIBILITY",
    "SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING",
    "SWEBENCH_TOOL_PROFILES",
    "create_swebench_repository_registration",
    "create_swebench_repository_registry",
]
