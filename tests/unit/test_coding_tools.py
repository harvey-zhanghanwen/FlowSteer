from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from src.interactive.coding_tools import (
    RepositoryToolBackend,
    SWEBENCH_REPOSITORY_TOOL_ID,
    create_swebench_repository_registry,
)
from src.interactive.tool_runtime import ToolRequest


class RepositoryToolBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "maths.py").write_text(
            "def add(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        (self.root / "pkg" / "other.py").write_text(
            "VALUE = 'needle'\n",
            encoding="utf-8",
        )
        (self.root / "tests").mkdir()
        (self.root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "tests" / "test_maths.py").write_text(
            "import unittest\n"
            "from pkg.maths import add\n\n"
            "class MathsTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        (self.root / "docs").mkdir()
        (self.root / "docs" / "contract.md").write_text(
            "non-python needle\n",
            encoding="utf-8",
        )
        (self.root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        self.backend = RepositoryToolBackend(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, action: str, **arguments: object):
        return self.backend.invoke(ToolRequest(action, arguments)).value

    def test_list_search_and_bounded_view_are_repository_relative(self) -> None:
        listing = self.invoke("list_files")
        self.assertTrue(listing["ok"])
        self.assertEqual(
            ["pkg/__init__.py", "pkg/maths.py", "pkg/other.py"],
            listing["files"],
        )

        search = self.invoke("search_code", query="needle", file_pattern="other.py")
        self.assertEqual(1, search["match_count"])
        self.assertEqual("pkg/other.py", search["matches"][0]["path"])

        viewed = self.invoke("view_file", path="pkg/maths.py", start_line=1, end_line=1)
        self.assertEqual("def add(left, right):", viewed["lines"][0]["text"])
        outside = self.invoke("view_file", path="../outside.py")
        self.assertFalse(outside["ok"])

    def test_explicit_file_pattern_searches_tests_docs_and_non_python_files(
        self,
    ) -> None:
        default_search = self.invoke("search_code", query="non-python needle")
        self.assertEqual(0, default_search["match_count"])

        docs_search = self.invoke(
            "search_code",
            query="non-python needle",
            file_pattern="docs/*.md",
        )
        self.assertEqual(1, docs_search["match_count"])
        self.assertEqual("docs/contract.md", docs_search["matches"][0]["path"])

        tests_search = self.invoke(
            "search_code",
            query="assertEqual",
            file_pattern="tests",
        )
        self.assertEqual(1, tests_search["match_count"])
        self.assertEqual(
            "tests/test_maths.py",
            tests_search["matches"][0]["path"],
        )

    def test_long_python_file_without_range_returns_skillflow_ast_filemap(
        self,
    ) -> None:
        content = (
            "import os\n\n"
            "class Worker:\n"
            "    def run(self):\n"
            "        return os.getcwd()\n\n"
            + "\n".join(f"VALUE_{index} = {index}" for index in range(510))
            + "\n"
        )
        (self.root / "pkg" / "large.py").write_text(content, encoding="utf-8")

        mapped = self.invoke("view_file", path="pkg/large.py")
        self.assertEqual("filemap", mapped["kind"])
        self.assertGreater(mapped["total_lines"], 500)
        self.assertIn("import os", mapped["filemap"])
        self.assertIn("class Worker:", mapped["filemap"])
        self.assertIn("def run(self):", mapped["filemap"])

        ranged = self.invoke(
            "view_file",
            path="pkg/large.py",
            start_line=1,
            end_line=3,
        )
        self.assertEqual("file", ranged["kind"])
        self.assertEqual(3, len(ranged["lines"]))

    def test_exact_edit_requires_one_match_and_emits_real_diff(self) -> None:
        ambiguous = self.invoke(
            "exact_edit",
            path="pkg/maths.py",
            old_str="right",
            new_str="value",
        )
        self.assertFalse(ambiguous["ok"])

        edited = self.invoke(
            "exact_edit",
            path="pkg/maths.py",
            old_str="return left - right",
            new_str="return left + right",
        )
        self.assertTrue(edited["ok"])
        self.assertIn("left + right", (self.root / "pkg" / "maths.py").read_text())
        diff = self.invoke("diff")
        self.assertTrue(diff["changed"])
        self.assertIn("-    return left - right", diff["diff"])
        self.assertIn("+    return left + right", diff["diff"])

    def test_python_syntax_failure_does_not_modify_file(self) -> None:
        original = (self.root / "pkg" / "maths.py").read_text()
        result = self.invoke(
            "exact_edit",
            path="pkg/maths.py",
            old_str="return left - right",
            new_str="return (",
        )
        self.assertFalse(result["ok"])
        self.assertIn("syntax error", result["error"].lower())
        self.assertEqual(original, (self.root / "pkg" / "maths.py").read_text())

    def test_skillflow_str_replace_editor_create_insert_and_undo_diff(self) -> None:
        replaced = self.invoke(
            "str_replace_editor",
            command="str_replace",
            path="pkg/maths.py",
            old_str="return left - right",
            new_str="return left + right",
        )
        self.assertTrue(replaced["changed"])
        self.assertTrue(self.invoke("diff")["changed"])

        undone = self.invoke(
            "str_replace_editor",
            command="undo_edit",
            path="pkg/maths.py",
        )
        self.assertTrue(undone["changed"])
        self.assertFalse(self.invoke("diff")["changed"])

        inserted = self.invoke(
            "str_replace_editor",
            command="insert",
            path="pkg/maths.py",
            insert_line=0,
            new_str="# inserted",
        )
        self.assertTrue(inserted["changed"])
        self.assertTrue((self.root / "pkg" / "maths.py").read_text().startswith("# inserted\n"))
        self.invoke(
            "str_replace_editor",
            command="undo_edit",
            path="pkg/maths.py",
        )
        self.assertFalse(self.invoke("diff")["changed"])

        created = self.invoke(
            "str_replace_editor",
            command="create",
            path="pkg/generated.txt",
            file_text="generated\n",
        )
        self.assertTrue(created["changed"])
        created_diff = self.invoke("diff")
        self.assertTrue(created_diff["changed"])
        self.assertIn("new file mode 100644", created_diff["diff"])
        self.assertIn("+generated", created_diff["diff"])

    def test_skillflow_bash_receipt_reports_command_result(self) -> None:
        command = f'{sys.executable} -c "print(\'bash-ok\')"'
        result = self.invoke("bash", command=command)
        self.assertTrue(result["ok"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(0, result["returncode"])
        self.assertIn("bash-ok", result["output"])

    def test_official_codex_apply_patch_updates_workspace_and_diff(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: pkg/maths.py\n"
            "@@\n"
            "-    return left - right\n"
            "+    return left + right\n"
            "*** End Patch"
        )
        result = self.invoke("apply_patch", patch=patch)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["applied"])
        self.assertTrue(result["changed"])
        self.assertEqual(["pkg/maths.py"], result["changed_paths"])
        self.assertIn("left + right", (self.root / "pkg" / "maths.py").read_text())
        diff = self.invoke("diff")
        self.assertTrue(diff["changed"])
        self.assertIn("+    return left + right", diff["diff"])

    def test_targeted_test_command_reports_failure_then_success(self) -> None:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-t",
            ".",
            "-p",
            "test_maths.py",
        ]
        failed = self.invoke("run_tests", command=command)
        self.assertFalse(failed["passed"])
        self.invoke(
            "exact_edit",
            path="pkg/maths.py",
            old_str="return left - right",
            new_str="return left + right",
        )
        passed = self.invoke("run_tests", command=command)
        self.assertTrue(passed["passed"], passed)


class RepositoryToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_backend_runs_through_async_tool_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            registry = create_swebench_repository_registry(root)
            self.assertEqual((SWEBENCH_REPOSITORY_TOOL_ID,), registry.resource_ids)
            capability = registry.require_capability(SWEBENCH_REPOSITORY_TOOL_ID)
            self.assertEqual(("swe_bench",), capability.dataset_scope)
            self.assertEqual(
                (
                    "apply_patch",
                    "bash",
                    "diff",
                    "exact_edit",
                    "list_files",
                    "run_tests",
                    "search_code",
                    "str_replace_editor",
                    "view_file",
                ),
                capability.action_names,
            )
            self.assertEqual(
                ["path", "old_str", "new_str"],
                capability.action_schemas["exact_edit"]["required"],
            )
            self.assertEqual(
                ["command"],
                capability.action_schemas["run_tests"]["required"],
            )
            self.assertEqual(
                ["command", "path"],
                capability.action_schemas["str_replace_editor"]["required"],
            )
            self.assertEqual(
                ["patch"],
                capability.action_schemas["apply_patch"]["required"],
            )
            self.assertEqual(
                ["path"], capability.action_schemas["view_file"]["required"]
            )
            self.assertTrue(
                all(
                    schema.get("additionalProperties") is False
                    for schema in capability.action_schemas.values()
                )
            )
            result = await registry.ainvoke(
                SWEBENCH_REPOSITORY_TOOL_ID,
                ToolRequest("view_file", {"path": "module.py"}),
            )
            self.assertTrue(result.value["ok"])
            self.assertEqual("VALUE = 1", result.value["lines"][0]["text"])


if __name__ == "__main__":
    unittest.main()
