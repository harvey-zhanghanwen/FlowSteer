from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from src.interactive.agent_runtime import AgentResponse, AgentRuntime
from src.interactive.computation_tools import (
    AIME_CALCULATOR_TOOL_ID,
    AIME_PYTHON_EXEC_TOOL_ID,
)
from src.interactive.healthbench_tool_adapter import (
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_TOOL_ID,
)
from src.interactive.react_execution import ToolReactExecutionAdapter
from src.interactive.records import TaskRecord
from src.interactive.tool_runtime import (
    ActionKind,
    FakeTool,
    StructuredAction,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_tool_exact_schema_canary.py"
)
_SPEC = importlib.util.spec_from_file_location("tool_exact_schema_canary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _model_registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("local", endpoint="http://127.0.0.1:1/v1")],
        [ModelSpec("qwen", "local", model_name="supervisor_theta")],
    )


def _capability(
    tool_id: str,
    action: str,
    required_name: str,
    *,
    dataset: str,
) -> ToolCapability:
    argument_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [required_name],
        "properties": {
            required_name: (
                {"type": "integer", "minimum": 1}
                if required_name == "limit"
                else {"type": "string", "minLength": 1}
            )
        },
    }
    return ToolCapability(
        tool_id=tool_id,
        dataset_scope=(dataset,),
        action_schemas={action: argument_schema},
        input_schema=argument_schema,
        output_schema={"type": "object"},
        side_effect="none",
        timeout_seconds=2.0,
        version="test-v1",
    )


def _qa_registry() -> ToolRegistry:
    search_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
    }
    read_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["passage_id"],
        "properties": {"passage_id": {"type": "string", "minLength": 1}},
    }
    retrieval_capability = ToolCapability(
        tool_id=QA_RETRIEVAL_TOOL_ID,
        dataset_scope=("triviaqa",),
        action_schemas={"search": search_schema, "read": read_schema},
        input_schema={"oneOf": [search_schema, read_schema]},
        output_schema={"type": "object"},
        side_effect="none",
        timeout_seconds=2.0,
        version="test-v1",
    )
    return ToolRegistry(
        (
            ToolRegistration(
                QA_RETRIEVAL_TOOL_ID,
                FakeTool(
                    {
                        "search": lambda arguments: {
                            "operation": "search",
                            "passage_ids": ["p1"],
                            "query": arguments["query"],
                        },
                        "read": lambda arguments: {
                            "operation": "read",
                            "passage_id": arguments["passage_id"],
                            "text": "public passage",
                        },
                    }
                ),
                retrieval_capability,
            ),
        )
    )


def _aime_registry() -> ToolRegistry:
    calculator = _capability(
        AIME_CALCULATOR_TOOL_ID,
        "calculator",
        "expression",
        dataset="aime_2026",
    )
    python_exec = _capability(
        AIME_PYTHON_EXEC_TOOL_ID,
        "python_exec",
        "code",
        dataset="aime_2026",
    )
    return ToolRegistry(
        (
            ToolRegistration(
                AIME_CALCULATOR_TOOL_ID,
                FakeTool(
                    {"calculator": lambda arguments: {"result": "4"}}
                ),
                calculator,
            ),
            ToolRegistration(
                AIME_PYTHON_EXEC_TOOL_ID,
                FakeTool(
                    {"python_exec": lambda arguments: {"stdout": "4\n"}}
                ),
                python_exec,
            ),
        )
    )


def _health_registry() -> ToolRegistry:
    capability = _capability(
        HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
        "search",
        "query",
        dataset="healthbench_professional",
    )
    return ToolRegistry(
        (
            ToolRegistration(
                HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                FakeTool(
                    {
                        "search": lambda arguments: {
                            "ranked_chunks": ["public medical text"]
                        }
                    }
                ),
                capability,
            ),
        )
    )


def _action(
    kind: ActionKind,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        StructuredAction(
            kind=kind,
            name=name,
            arguments=arguments,
            resource_id=resource_id,
        ).to_value(),
        sort_keys=True,
    )


class _SequenceGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(
            self.responses[len(self.requests) - 1],
            {"generation_seed": 7, "latency_ms": 1.0},
        )


class _Backend:
    def __init__(self, registry: ToolRegistry, gateway: _SequenceGateway) -> None:
        self.registry = _model_registry()
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry,
            max_turns=6,
            max_tool_calls=4,
        )
        self.runtime = AgentRuntime(
            self.registry,
            gateway,
            execution_adapters={"react": adapter},
            tool_registry=registry,
        )
        self.tool_registry = registry
        self.received_task = None
        self.received_condition_id = None
        self.closed = False

    def _runtime_for_task(self, task, *, condition_id=None):
        self.received_task = task
        self.received_condition_id = condition_id
        return self.runtime, self.tool_registry, self.close

    def close(self):
        self.closed = True


def _task(dataset: str) -> TaskRecord:
    return TaskRecord(
        task_id=f"{dataset}:development-0",
        question="Public question?",
        ground_truth=None,
        split="validation",
        metadata={"dataset_key": dataset, "task_family": dataset},
    )


class ToolExactSchemaCanaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_qa_search_read_complete_passes_without_injected_actions(self):
        gateway = _SequenceGateway(
            [
                _action(
                    ActionKind.TOOL,
                    "search",
                    {"query": "Public question", "limit": 1},
                    QA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    ActionKind.TOOL,
                    "read",
                    {"passage_id": "p1"},
                    QA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    ActionKind.COMPLETE,
                    "complete",
                    {"value": "public diagnostic artifact"},
                    None,
                ),
            ]
        )
        backend = _Backend(_qa_registry(), gateway)

        receipt = await _MODULE.execute_tool_canary(
            backend=backend,
            task=_task("triviaqa"),
            model_id="qwen",
            condition_id="qa-condition",
            run_id="qa-run",
        )

        self.assertEqual("passed", receipt["status"])
        self.assertTrue(receipt["compliance"]["schema_compliance"]["passed"])
        self.assertTrue(receipt["compliance"]["backend_compliance"]["passed"])
        self.assertTrue(receipt["compliance"]["model_compliance"]["passed"])
        self.assertFalse(receipt["controls"]["structured_actions_injected"])
        self.assertTrue(receipt["diagnostic_only"])
        self.assertTrue(receipt["forced_probe"])
        self.assertFalse(receipt["training_enabled"])
        self.assertFalse(receipt["controls"]["grpo_eligible"])
        self.assertFalse(receipt["controls"]["skill_evidence_eligible"])
        self.assertIsNone(receipt["benchmark_metrics"])
        self.assertTrue(backend.closed)
        self.assertIsNone(backend.received_task.ground_truth)
        self.assertIn("p1", gateway.requests[1].agent.contract)

    async def test_aime_calculator_complete_is_one_admitted_alternative(self):
        gateway = _SequenceGateway(
            [
                _action(
                    ActionKind.TOOL,
                    "calculator",
                    {"expression": "2+2"},
                    AIME_CALCULATOR_TOOL_ID,
                ),
                _action(
                    ActionKind.COMPLETE,
                    "complete",
                    {"value": "4"},
                    None,
                ),
            ]
        )
        receipt = await _MODULE.execute_tool_canary(
            backend=_Backend(_aime_registry(), gateway),
            task=_task("aime_2026"),
            model_id="qwen",
            condition_id="aime-condition",
            run_id="aime-run",
        )
        self.assertEqual("passed", receipt["status"])
        self.assertEqual(
            ["calculator", "complete"],
            [item["name"] for item in receipt["compliance"]["observed_sequence"]],
        )

    async def test_health_search_complete_passes(self):
        gateway = _SequenceGateway(
            [
                _action(
                    ActionKind.TOOL,
                    "search",
                    {"query": "public symptom"},
                    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                ),
                _action(
                    ActionKind.COMPLETE,
                    "complete",
                    {"value": "public medical artifact"},
                    None,
                ),
            ]
        )
        receipt = await _MODULE.execute_tool_canary(
            backend=_Backend(_health_registry(), gateway),
            task=_task("healthbench_professional"),
            model_id="qwen",
            condition_id="health-condition",
            run_id="health-run",
        )
        self.assertEqual("passed", receipt["status"])
        self.assertEqual(1, len(receipt["tool_receipts"]))

    async def test_schema_failure_is_recorded_without_repairing_model_action(self):
        gateway = _SequenceGateway(
            [
                _action(
                    ActionKind.TOOL,
                    "search",
                    {"query": "missing limit"},
                    QA_RETRIEVAL_TOOL_ID,
                ),
                _action(
                    ActionKind.COMPLETE,
                    "complete",
                    {"value": "premature"},
                    None,
                ),
            ]
        )
        receipt = await _MODULE.execute_tool_canary(
            backend=_Backend(_qa_registry(), gateway),
            task=_task("triviaqa"),
            model_id="qwen",
            condition_id="qa-condition",
            run_id="qa-failed-run",
        )
        self.assertEqual("failed", receipt["status"])
        self.assertFalse(receipt["compliance"]["schema_compliance"]["passed"])
        # Runtime admission rejects the invalid arguments before dispatch, so
        # the deliberately permissive fake backend is never invoked.
        self.assertFalse(receipt["compliance"]["backend_compliance"]["passed"])
        self.assertFalse(receipt["compliance"]["model_compliance"]["passed"])
        self.assertEqual(
            "model_generated_not_injected",
            receipt["compliance"]["model_compliance"]["action_selection"],
        )

    async def test_exhausted_react_loop_preserves_partial_trace_and_receipts(self):
        repeated = _action(
            ActionKind.TOOL,
            "search",
            {"query": "Public question", "limit": 1},
            QA_RETRIEVAL_TOOL_ID,
        )
        receipt = await _MODULE.execute_tool_canary(
            backend=_Backend(_qa_registry(), _SequenceGateway([repeated] * 6)),
            task=_task("triviaqa"),
            model_id="qwen",
            condition_id="qa-condition",
            run_id="qa-exhausted-run",
        )

        self.assertEqual("failed", receipt["status"])
        self.assertEqual(6, len(receipt["react_trace"]))
        self.assertEqual(1, len(receipt["tool_receipts"]))
        self.assertTrue(receipt["compliance"]["schema_compliance"]["passed"])
        self.assertTrue(receipt["compliance"]["backend_compliance"]["passed"])
        self.assertFalse(receipt["compliance"]["model_compliance"]["passed"])

    def test_first_development_task_strips_gold_and_skillflow_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "validation.jsonl"
            records = [
                {
                    "schema_version": "flowsteer.agentgraph.task.v1",
                    "task_id": "aime:2025:0",
                    "question": "Public AIME problem",
                    "ground_truth": "777",
                    "split": "validation",
                    "metadata": {
                        "dataset_key": "aime_2026",
                        "task_family": "aime_2026",
                        "benchmark_slice": "development_aime_2025",
                    },
                    "answer": "777",
                    "extra": {"answer": "777"},
                }
            ]
            data.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            config = {
                "data": {"validation_path": str(data)},
                "aime2026_evaluation": {
                    "dataset_key": "aime_2026",
                    "split": "validation",
                    "benchmark_slice": "development_aime_2025",
                },
            }

            task = _MODULE.load_first_development_task(config, root=root)

        self.assertEqual("aime:2025:0", task.task_id)
        self.assertIsNone(task.ground_truth)
        self.assertNotIn("skillflow", task.metadata)
        self.assertNotIn("answer", task.metadata)
        self.assertEqual("Public AIME problem", task.question)


if __name__ == "__main__":
    unittest.main()
