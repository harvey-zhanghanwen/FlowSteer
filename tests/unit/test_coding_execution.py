from __future__ import annotations

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
    create_swebench_repository_registry,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec


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


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    async def generate(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(self.outputs.pop(0))


class CodingExecutionTests(unittest.IsolatedAsyncioTestCase):
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
            request = AgentRequest(
                request_id="run:1:coder:single",
                run_id="run",
                graph_revision=1,
                problem="Fix add so it returns the sum.",
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

            response = await adapter.execute(request)

            self.assertEqual(patch, response.text)
            self.assertEqual("coding", response.metadata["execution_mode"])
            self.assertEqual(4, response.metadata["tool_calls"])
            self.assertIn("return a + b", (root / "bug.py").read_text())


if __name__ == "__main__":
    unittest.main()
