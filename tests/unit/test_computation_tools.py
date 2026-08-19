from __future__ import annotations

import json
import unittest

from src.interactive.computation_tools import (
    AIME_CALCULATOR_TOOL_ID,
    AIME_COMPUTATION_TOOL_VERSION,
    AIME_DATASET_SCOPE,
    AIME_PYTHON_EXEC_TOOL_ID,
    AIMEComputationToolBackend,
    create_aime_computation_registry,
)
from src.interactive.tool_runtime import ToolRequest


class AIMEComputationToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculator_uses_skillflow_namespace_and_emits_json_receipt(self) -> None:
        registry = create_aime_computation_registry()

        result, receipt = await registry.ainvoke_with_receipt(
            AIME_CALCULATOR_TOOL_ID,
            ToolRequest("calculator", {"expression": "comb(8, 2) + sqrt(16)"}),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("calculator", result.value["action"])
        self.assertTrue(result.value["ok"])
        self.assertEqual(
            "[RESULT] comb(8, 2) + sqrt(16) = 32.0",
            result.value["observation"],
        )
        self.assertEqual(AIME_COMPUTATION_TOOL_VERSION, receipt.tool_version)
        self.assertIsInstance(json.dumps(receipt.to_value()), str)
        self.assertNotIn("reward", result.value)
        self.assertNotIn("complexity", result.value)

    async def test_python_exec_runs_in_upstream_child_process(self) -> None:
        registry = create_aime_computation_registry(python_timeout_seconds=2.0)

        result, receipt = await registry.ainvoke_with_receipt(
            AIME_PYTHON_EXEC_TOOL_ID,
            ToolRequest(
                "python_exec",
                {"code": "from math import factorial\nprint(factorial(6))"},
            ),
        )

        self.assertIsNone(receipt.error_type)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.value["ok"])
        self.assertIn("[STDOUT]\n720", result.value["observation"])
        self.assertIsInstance(json.dumps(receipt.to_value()), str)

    async def test_python_exec_timeout_is_a_completed_failed_observation(self) -> None:
        registry = create_aime_computation_registry(python_timeout_seconds=0.05)

        result, receipt = await registry.ainvoke_with_receipt(
            AIME_PYTHON_EXEC_TOOL_ID,
            ToolRequest("python_exec", {"code": "while True:\n    pass"}),
        )

        self.assertIsNone(receipt.error_type)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.value["ok"])
        self.assertIn("timed out after 0.05s", result.value["observation"])

    async def test_strict_schema_rejects_evaluator_or_answer_arguments(self) -> None:
        registry = create_aime_computation_registry()

        result, receipt = await registry.ainvoke_with_receipt(
            AIME_CALCULATOR_TOOL_ID,
            ToolRequest(
                "calculator",
                {"expression": "6 * 7", "accepted_answer": "42"},
            ),
        )

        self.assertIsNone(result)
        self.assertEqual("ValueError", receipt.error_type)

    def test_registrations_are_aime_only_with_strict_schemas(self) -> None:
        registry = create_aime_computation_registry(
            python_timeout_seconds=3.0,
            calculator_timeout_seconds=1.0,
        )

        self.assertEqual(
            (AIME_CALCULATOR_TOOL_ID, AIME_PYTHON_EXEC_TOOL_ID),
            registry.resource_ids,
        )
        calculator = registry.require_capability(AIME_CALCULATOR_TOOL_ID)
        python_exec = registry.require_capability(AIME_PYTHON_EXEC_TOOL_ID)
        self.assertEqual(("aime_2026",), AIME_DATASET_SCOPE)
        self.assertEqual(AIME_DATASET_SCOPE, calculator.dataset_scope)
        self.assertEqual(AIME_DATASET_SCOPE, python_exec.dataset_scope)
        self.assertEqual(("calculator",), calculator.action_names)
        self.assertEqual(("python_exec",), python_exec.action_names)
        self.assertEqual(
            calculator.input_schema, calculator.action_schemas["calculator"]
        )
        self.assertEqual(
            python_exec.input_schema, python_exec.action_schemas["python_exec"]
        )
        self.assertFalse(calculator.supports_dataset("hotpotqa"))
        self.assertEqual(["expression"], calculator.input_schema["required"])
        self.assertFalse(calculator.input_schema["additionalProperties"])
        self.assertEqual(["code"], python_exec.input_schema["required"])
        self.assertFalse(python_exec.input_schema["additionalProperties"])
        self.assertEqual("none", calculator.side_effect)
        self.assertEqual("isolated_child_process", python_exec.side_effect)
        self.assertEqual(1.0, calculator.timeout_seconds)
        self.assertEqual(9.0, python_exec.timeout_seconds)

    def test_action_bound_backend_rejects_cross_resource_action(self) -> None:
        backend = AIMEComputationToolBackend("calculator")
        with self.assertRaisesRegex(ValueError, "incompatible action"):
            backend.invoke(ToolRequest("python_exec", {"code": "print(1)"}))


if __name__ == "__main__":
    unittest.main()
