from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.coding_execution import CodingExecutionAdapter
from src.interactive.coding_tools import (
    SWEBENCH_REPOSITORY_TOOL_ID,
    SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
    create_swebench_repository_registration,
    create_swebench_repository_registry,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.tool_runtime import ToolRegistry


def tool(name: str, arguments: object) -> str:
    return json.dumps(
        {
            "kind": "tool",
            "name": name,
            "arguments": arguments,
            "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
            "skill_id": None,
        }
    )


def complete(value: str) -> str:
    return json.dumps(
        {
            "kind": "complete",
            "name": "complete",
            "arguments": {"value": value},
            "resource_id": None,
            "skill_id": None,
        }
    )


def coding_request() -> AgentRequest:
    return AgentRequest(
        request_id="run:1:coder:single",
        run_id="run",
        graph_revision=1,
        problem="Fix add so it returns the requested result.",
        agent=AgentNode(
            "coder",
            "m",
            "inspect, edit, test, and return a patch",
            allowed_tools=(SWEBENCH_REPOSITORY_TOOL_ID,),
            execution_mode="coding",
            artifact_type="patch_candidate",
            completion_condition="return the tested unified diff",
        ),
        model=ModelSpec("m", "fake"),
        provider=ProviderSpec("fake", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(self.outputs.pop(0))


class ConcurrentCompleteGateway:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate(self, request: AgentRequest) -> AgentResponse:
        del request
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return AgentResponse(complete("done"))
        finally:
            self.active -= 1


class CodingExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_skillflow_swe_memory_bounds_model_visible_observations(self) -> None:
        observations = [
            {
                "observation_status": "success",
                "executed_action": {
                    "kind": "tool",
                    "name": "bash",
                    "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"command": f"python check_{index}.py"},
                },
                "result": {
                    "action": "bash",
                    "ok": True,
                    "returncode": 0,
                    "stdout": "x" * 9000,
                },
            }
            for index in range(20)
        ]

        visible = CodingExecutionAdapter._model_visible_observations(observations)

        self.assertEqual(2, len(visible))
        self.assertEqual(6, len(visible[0]["SWE_MEMORY"]))
        self.assertEqual(13, visible[0]["omitted_prior_observations"])
        rendered = json.dumps(visible, ensure_ascii=False)
        self.assertLess(len(rendered), 10000)
        self.assertIn("OBSERVATION_TRUNCATED", rendered)

    async def test_empty_tool_declaration_inherits_task_repository_resource(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = SequenceGateway(
                [
                    tool("view_file", {"path": "bug.py"}),
                    complete("done"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=2,
                max_tool_calls=1,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=lambda: "diff --git a/bug.py b/bug.py\n",
            )
            original = coding_request()
            request = replace(
                original,
                agent=replace(original.agent, allowed_tools=()),
            )

            response = await adapter.execute(request)

        self.assertEqual(
            (SWEBENCH_REPOSITORY_TOOL_ID,),
            gateway.requests[0].agent.allowed_tools,
        )
        self.assertEqual(
            "view_file",
            response.metadata["tool_receipts"][0]["request"]["action"],
        )
        self.assertEqual(
            {
                "source": "ToolRegistry.resource_ids",
                "applied": True,
                "resource_ids": [SWEBENCH_REPOSITORY_TOOL_ID],
            },
            response.metadata["task_scoped_tool_binding"],
        )

    def test_public_workspace_diff_materializer_returns_empty_or_exact_diff(
        self,
    ) -> None:
        values = iter(("", "diff --git a/x b/x\n"))
        adapter = CodingExecutionAdapter(
            gateway=SequenceGateway([]),
            tool_registry=ToolRegistry(()),
            max_turns=1,
            max_tool_calls=0,
            completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
            workspace_diff=lambda: next(values),
        )

        self.assertEqual("", adapter.materialize_workspace_diff())
        self.assertEqual(
            "diff --git a/x b/x\n",
            adapter.materialize_workspace_diff(),
        )

    async def test_parallel_graph_nodes_share_serial_task_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = ConcurrentCompleteGateway()
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=1,
                max_tool_calls=1,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=lambda: "diff --git a/bug.py b/bug.py\n",
                task_max_turns=1,
                task_max_tool_calls=1,
            )

            results = await asyncio.gather(
                adapter.execute(coding_request()),
                adapter.execute(coding_request()),
                return_exceptions=True,
            )

            self.assertEqual(1, gateway.calls)
            self.assertEqual(1, gateway.max_active)
            self.assertEqual(
                2,
                sum(isinstance(value, AgentResponse) for value in results),
            )
            self.assertEqual(
                1,
                sum(
                    value.metadata.get("termination_reason")
                    == "task_global_turn_budget"
                    for value in results
                ),
            )

    async def test_skillflow_profile_submits_materialized_workspace_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            registry = ToolRegistry((registration,))
            gateway = SequenceGateway(
                [
                    tool(
                        "edit_file",
                        {
                            "path": "bug.py",
                            "instruction": (
                                "Replace `return a - b` with `return a + b`."
                            ),
                        },
                    ),
                    complete("model prose is not the submitted patch"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=4,
                max_tool_calls=3,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=registration.backend.materialize_workspace_diff,
                repository_runtime_receipt=lambda: {
                    "repository": {"observed_pinned_commit": "fixture-base"},
                    "task_environment": {"task_environment_ready": True},
                },
                task_max_turns=4,
                task_max_tool_calls=3,
            )

            response = await adapter.execute(coding_request())

            self.assertIn("+    return a + b", response.text)
            self.assertEqual(
                ["edit_file"],
                [
                    receipt["request"]["action"]
                    for receipt in response.metadata["tool_receipts"]
                ],
            )
            self.assertEqual(
                {
                    "max_turns": 4,
                    "turns_used": 2,
                    "max_tool_calls": 3,
                    "tool_calls_used": 1,
                },
                response.metadata["task_global_budget"],
            )
            self.assertEqual(
                "fixture-base",
                response.metadata["repository_runtime"]["repository"][
                    "observed_pinned_commit"
                ],
            )

    async def test_skillflow_episode_bound_submits_existing_workspace_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = SequenceGateway(
                [
                    tool(
                        "edit_file",
                        {
                            "path": "bug.py",
                            "instruction": (
                                "Replace `return a - b` with `return a + b`."
                            ),
                        },
                    ),
                    tool("view_file", {"path": "bug.py"}),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=2,
                max_tool_calls=2,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=registration.backend.materialize_workspace_diff,
                task_max_turns=2,
                task_max_tool_calls=2,
            )

            response = await adapter.execute(coding_request())

            self.assertIn("+    return a + b", response.text)
            self.assertIs(True, response.metadata["truncated"])
            self.assertEqual("max_turns", response.metadata["termination_reason"])
            self.assertIs(True, response.metadata["workspace_diff_submitted"])
            self.assertEqual(
                "SkillFlow training.environment._force_terminate",
                response.metadata["termination_source"],
            )
            self.assertEqual(2, len(response.metadata["model_calls"]))
            self.assertEqual(
                {
                    "max_turns": 2,
                    "turns_used": 2,
                    "max_tool_calls": 2,
                    "tool_calls_used": 2,
                },
                response.metadata["task_global_budget"],
            )

    async def test_skillflow_force_terminate_submits_empty_workspace_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = SequenceGateway([tool("list_files", {})])
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=1,
                max_tool_calls=1,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=registration.backend.materialize_workspace_diff,
                task_max_turns=1,
                task_max_tool_calls=1,
            )

            first = await adapter.execute(coding_request())
            second = await adapter.execute(coding_request())

            self.assertEqual("", first.text)
            self.assertFalse(first.metadata["workspace_diff_submitted"])
            self.assertEqual("max_turns", first.metadata["termination_reason"])
            self.assertEqual("", second.text)
            self.assertFalse(second.metadata["workspace_diff_submitted"])
            self.assertEqual(
                "task_global_turn_budget",
                second.metadata["termination_reason"],
            )
            self.assertEqual(1, len(gateway.requests))

    async def test_skillflow_repository_actions_use_five_field_sampling_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = SequenceGateway(
                [
                    tool(
                        "edit_file",
                        {
                            "path": "bug.py",
                            "instruction": (
                                "Replace `return a - b` with `return a + b`."
                            ),
                        },
                    ),
                    complete("done"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=2,
                max_tool_calls=2,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=registration.backend.materialize_workspace_diff,
            )

            await adapter.execute(coding_request())

            schema = json.loads(
                gateway.requests[0].model.metadata["response_json_schema"]
            )
            first_branches = schema["oneOf"]
            first_names = {
                branch["properties"]["name"]["const"]
                for branch in first_branches
            }
            self.assertIn("edit_file", first_names)
            self.assertNotIn("complete", first_names)
            editor_branch = next(
                branch
                for branch in first_branches
                if branch["properties"]["name"]["const"]
                == "edit_file"
            )
            self.assertEqual(
                SWEBENCH_REPOSITORY_TOOL_ID,
                editor_branch["properties"]["resource_id"]["const"],
            )
            self.assertFalse(
                editor_branch["properties"]["arguments"][
                    "additionalProperties"
                ]
            )
            completion_schema = json.loads(
                gateway.requests[1].model.metadata["response_json_schema"]
            )
            self.assertIn(
                "complete",
                {
                    branch["properties"]["name"]["const"]
                    for branch in completion_schema["oneOf"]
                },
            )

    async def test_skillflow_code_generation_guidance_is_model_visible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            gateway = SequenceGateway(
                [tool("list_files", {}), complete("done")]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=ToolRegistry((registration,)),
                max_turns=2,
                max_tool_calls=1,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=lambda: "diff --git a/bug.py b/bug.py\n",
            )

            await adapter.execute(coding_request())

            prompt = gateway.requests[0].agent.contract
            self.assertIn("SkillFlow code-generation episode guidance", prompt)
            self.assertIn("workspace diff is the submitted artifact", prompt)
            self.assertIn("repository root is already the current working directory", prompt)
            self.assertIn("use edit_file", prompt)
            self.assertIn("Use this first", prompt)

    async def test_task_global_turn_budget_survives_new_agent_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registration = create_swebench_repository_registration(
                root,
                action_profile=SWEBENCH_TOOL_PROFILE_SKILLFLOW_TRAINING,
                task_command_prefix=("/usr/bin/env",),
                task_environment_receipt={"task_environment_ready": True},
                require_task_environment=True,
            )
            registry = ToolRegistry((registration,))
            gateway = SequenceGateway(
                [
                    tool(
                        "edit_file",
                        {
                            "path": "bug.py",
                            "instruction": (
                                "Replace `return a - b` with `return a + b`."
                            ),
                        },
                    ),
                    complete("done"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=4,
                max_tool_calls=3,
                completion_policy=CodingExecutionAdapter.WORKSPACE_DIFF_COMPLETION,
                workspace_diff=registration.backend.materialize_workspace_diff,
                task_max_turns=2,
                task_max_tool_calls=3,
            )

            await adapter.execute(coding_request())
            second = coding_request()
            second = AgentRequest(
                request_id="run:2:repair:single",
                run_id=second.run_id,
                graph_revision=2,
                problem=second.problem,
                agent=AgentNode(
                    "repair",
                    "m",
                    "repair the current repository state",
                    allowed_tools=(SWEBENCH_REPOSITORY_TOOL_ID,),
                    execution_mode="coding",
                    artifact_type="patch_candidate",
                    completion_condition="return the workspace patch",
                ),
                model=second.model,
                provider=second.provider,
                phase=second.phase,
            )
            response = await adapter.execute(second)
            self.assertIn("+    return a + b", response.text)
            self.assertEqual(
                "task_global_turn_budget",
                response.metadata["termination_reason"],
            )

    async def test_edit_test_diff_then_complete_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registry = create_swebench_repository_registry(root)
            patch = (
                "diff --git a/bug.py b/bug.py\n"
                "--- a/bug.py\n+++ b/bug.py\n"
                "@@ -1,2 +1,2 @@\n def add(a, b):\n"
                "-    return a - b\n+    return a + b\n"
            )
            gateway = SequenceGateway(
                [
                    tool("view_file", {"path": "bug.py"}),
                    tool(
                        "str_replace_editor",
                        {
                            "command": "str_replace",
                            "path": "bug.py",
                            "old_str": "return a - b",
                            "new_str": "return a + b",
                        },
                    ),
                    tool(
                        "run_tests",
                        {
                            "command": [
                                sys.executable,
                                "-c",
                                "from bug import add; assert add(2, 3) == 5",
                            ]
                        },
                    ),
                    tool("diff", {}),
                    complete("model prose that must not replace the workspace diff"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=6,
                max_tool_calls=5,
            )
            response = await adapter.execute(coding_request())

            self.assertEqual(patch, response.text)
            self.assertEqual("coding", response.metadata["execution_mode"])
            self.assertEqual(4, response.metadata["tool_calls"])
            self.assertIn("return a + b", (root / "bug.py").read_text())

    async def test_codex_apply_patch_is_a_fresh_changed_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registry = create_swebench_repository_registry(root)
            patch_input = (
                "*** Begin Patch\n"
                "*** Update File: bug.py\n"
                "@@\n"
                "-    return a - b\n"
                "+    return a + b\n"
                "*** End Patch"
            )
            gateway = SequenceGateway(
                [
                    tool("apply_patch", {"patch": patch_input}),
                    tool(
                        "run_tests",
                        {
                            "command": [
                                sys.executable,
                                "-c",
                                "from bug import add; assert add(2, 3) == 5",
                            ]
                        },
                    ),
                    tool("diff", {}),
                    complete("submit current repository state"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=4,
                max_tool_calls=3,
            )

            response = await adapter.execute(coding_request())

            self.assertIn("+    return a + b", response.text)
            self.assertEqual(
                ["apply_patch", "run_tests", "diff"],
                [
                    receipt["request"]["action"]
                    for receipt in response.metadata["tool_receipts"]
                ],
            )

    async def test_revision_requires_new_test_and_diff_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registry = create_swebench_repository_registry(root)
            gateway = SequenceGateway(
                [
                    tool(
                        "exact_edit",
                        {
                            "path": "bug.py",
                            "old_str": "return a - b",
                            "new_str": "return a + b",
                        },
                    ),
                    tool(
                        "run_tests",
                        {
                            "command": [
                                sys.executable,
                                "-c",
                                "from bug import add; assert add(2, 3) == 6",
                            ]
                        },
                    ),
                    tool("diff", {}),
                    tool(
                        "exact_edit",
                        {
                            "path": "bug.py",
                            "old_str": "return a + b",
                            "new_str": "return a * b",
                        },
                    ),
                    complete("must reject the pre-revision test and diff"),
                    tool(
                        "run_tests",
                        {
                            "command": [
                                sys.executable,
                                "-c",
                                "from bug import add; assert add(2, 3) == 6",
                            ]
                        },
                    ),
                    complete("must reject the pre-revision diff"),
                    tool("diff", {}),
                    complete("model prose must not replace the fresh diff"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=9,
                max_tool_calls=8,
            )

            response = await adapter.execute(coding_request())

            self.assertIn("-    return a - b", response.text)
            self.assertIn("+    return a * b", response.text)
            self.assertNotIn("+    return a + b", response.text)
            error_codes = [
                entry.get("public_error_code")
                for entry in response.metadata["react_trace"]
                if "public_error_code" in entry
            ]
            self.assertEqual(
                [
                    "coding_completion_requires_test",
                    "coding_completion_requires_changed_diff",
                ],
                error_codes,
            )
            receipts = response.metadata["tool_receipts"]
            self.assertFalse(receipts[1]["result"]["value"]["passed"])
            self.assertTrue(receipts[4]["result"]["value"]["passed"])
            self.assertEqual(
                [
                    "exact_edit",
                    "run_tests",
                    "diff",
                    "exact_edit",
                    "run_tests",
                    "diff",
                ],
                [receipt["request"]["action"] for receipt in receipts],
            )

    async def test_fresh_failed_test_preserves_skillflow_terminal_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bug.py").write_text(
                "def add(a, b):\n    return a - b\n",
                encoding="utf-8",
            )
            registry = create_swebench_repository_registry(root)
            gateway = SequenceGateway(
                [
                    tool(
                        "exact_edit",
                        {
                            "path": "bug.py",
                            "old_str": "return a - b",
                            "new_str": "return a + b",
                        },
                    ),
                    tool(
                        "run_tests",
                        {
                            "command": [
                                sys.executable,
                                "-c",
                                "raise SystemExit(1)",
                            ]
                        },
                    ),
                    tool("diff", {}),
                    complete("submit current repository state"),
                ]
            )
            adapter = CodingExecutionAdapter(
                gateway=gateway,
                tool_registry=registry,
                max_turns=4,
                max_tool_calls=3,
            )

            response = await adapter.execute(coding_request())

            self.assertIn("+    return a + b", response.text)
            test_receipt = response.metadata["tool_receipts"][1]
            self.assertFalse(test_receipt["result"]["value"]["passed"])


if __name__ == "__main__":
    unittest.main()
