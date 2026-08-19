from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError

from src.interactive.agent_runtime import AgentResponse
from src.interactive.openai_gateway import OpenAICompatibleGatewayError


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probe_model_capabilities.py"
_SPEC = importlib.util.spec_from_file_location("probe_model_capabilities", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
canary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = canary
_SPEC.loader.exec_module(canary)


class _FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected paid-model-shaped call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _successful_responses():
    return [
        AgentResponse(
            canary.TEXT_EXPECTED,
            {"provider_request_id": "text-1", "total_tokens": 7},
        ),
        AgentResponse(
            json.dumps(
                canary.REACT_EXPECTED,
                sort_keys=True,
                separators=(",", ":"),
            ),
            {"provider_request_id": "react-1", "total_tokens": 9},
        ),
        AgentResponse(
            json.dumps(
                canary.CODING_EXPECTED,
                sort_keys=True,
                separators=(",", ":"),
            ),
            {"provider_request_id": "coding-1", "total_tokens": 19},
        ),
    ]


class ModelCapabilityCanaryTests(unittest.TestCase):
    def test_dry_run_discovers_first_and_uses_exact_intersection(self) -> None:
        events = []

        def fake_fetch(endpoint, api_key, timeout):
            events.append(("discover", endpoint, api_key, timeout))
            return (
                [{"id": "exact-a"}, {"id": "qwen3.5-9b"}],
                {"http_status": 200, "request_id": "models-1"},
            )

        gateway = _FakeGateway([])
        receipt = asyncio.run(
            canary.audit_model_capabilities(
                base_url="https://provider.example/console",
                api_key_env="TEST_PROVIDER_KEY",
                api_key="not-printed",
                requested_model_ids=["qwen3.5-9b", "exact-a", "qwen3.5-9b"],
                mode="dry_run",
                timeout=3.0,
                concurrency=1,
                fetcher=fake_fetch,
                gateway=gateway,
            )
        )

        self.assertEqual("discover", events[0][0])
        self.assertEqual("https://provider.example/v1/models", events[0][1])
        self.assertEqual(["exact-a", "qwen3.5-9b"], receipt["discovery"]["actual_model_ids"])
        self.assertEqual(["qwen3.5-9b", "exact-a"], receipt["selected_model_ids"])
        self.assertEqual(6, receipt["summary"]["completion_requests_planned"])
        self.assertEqual(0, receipt["summary"]["completion_requests_executed"])
        self.assertEqual([], gateway.requests)

    def test_guessed_alias_and_case_change_are_rejected_without_model_call(self) -> None:
        def fake_fetch(endpoint, api_key, timeout):
            return ([{"id": "Qwen3.5-9B-Exact"}], {"http_status": 200})

        gateway = _FakeGateway([])
        with self.assertRaisesRegex(canary.ModelSelectionError, "absent"):
            asyncio.run(
                canary.audit_model_capabilities(
                    base_url="https://provider.example/v1",
                    api_key_env="TEST_PROVIDER_KEY",
                    api_key="not-printed",
                    requested_model_ids=["qwen3.5-9b-exact"],
                    mode="run_probes",
                    timeout=3.0,
                    concurrency=1,
                    fetcher=fake_fetch,
                    gateway=gateway,
                )
            )
        self.assertEqual([], gateway.requests)

    def test_three_probe_schema_and_raw_receipts(self) -> None:
        calls = []

        def fake_fetch(endpoint, api_key, timeout):
            calls.append("discover")
            return (
                [{"id": "actual-model", "object": "model"}],
                {"http_status": 200, "request_id": "models-1"},
            )

        gateway = _FakeGateway(_successful_responses())
        receipt = asyncio.run(
            canary.audit_model_capabilities(
                base_url="https://provider.example/v1",
                api_key_env="TEST_PROVIDER_KEY",
                api_key="secret-for-redaction",
                requested_model_ids=["actual-model"],
                mode="run_probes",
                timeout=3.0,
                concurrency=1,
                fetcher=fake_fetch,
                gateway=gateway,
            )
        )

        self.assertEqual(["discover"], calls)
        self.assertEqual(canary.SCHEMA_VERSION, receipt["schema_version"])
        self.assertEqual(3, len(gateway.requests))
        self.assertEqual(
            ["reasoning", "react", "coding"],
            [request.agent.execution_mode.value for request in gateway.requests],
        )
        self.assertEqual(
            ["text_answer", "structured_action", "coding_unified_diff"],
            [item["probe_id"] for item in receipt["probes"]],
        )
        for item in receipt["probes"]:
            self.assertEqual("actual-model", item["model_id"])
            self.assertEqual("passed", item["status"])
            self.assertTrue(item["compatible"])
            self.assertIsNone(item["error"])
            self.assertIsInstance(item["raw_response"]["text"], str)
            self.assertIsInstance(item["raw_response"]["metadata"], dict)
            self.assertEqual("actual-model", item["request"]["model_id"])
        self.assertEqual(0, receipt["summary"]["fallback_requests"])

        with tempfile.TemporaryDirectory() as directory:
            path = canary.write_capability_receipt(
                Path(directory) / "receipt.json", receipt
            )
            saved = path.read_text(encoding="utf-8")
        self.assertNotIn("secret-for-redaction", saved)
        self.assertIn('"raw_response"', saved)

    def test_http_and_model_output_failures_are_distinct(self) -> None:
        http_error = HTTPError(
            "https://provider.example/v1/chat/completions",
            429,
            "rate limited",
            hdrs=None,
            fp=None,
        )
        gateway = _FakeGateway(
            [
                http_error,
                AgentResponse("not-json", {"provider_request_id": "bad-output"}),
                OpenAICompatibleGatewayError("provider response has no completion choice"),
            ]
        )

        def fake_fetch(endpoint, api_key, timeout):
            return ([{"id": "actual-model"}], {"http_status": 200})

        receipt = asyncio.run(
            canary.audit_model_capabilities(
                base_url="https://provider.example/v1",
                api_key_env="TEST_PROVIDER_KEY",
                api_key="not-printed",
                requested_model_ids=["actual-model"],
                mode="run_probes",
                timeout=3.0,
                concurrency=1,
                fetcher=fake_fetch,
                gateway=gateway,
            )
        )
        statuses = [item["status"] for item in receipt["probes"]]
        self.assertEqual(
            ["http_error", "model_output_error", "provider_response_error"],
            statuses,
        )
        self.assertEqual(429, receipt["probes"][0]["error"]["http_status"])
        self.assertEqual(
            "structured_action_invalid",
            receipt["probes"][1]["validation_error"],
        )
        self.assertIsNone(receipt["probes"][1]["error"])

    def test_list_only_does_not_require_or_invoke_gateway(self) -> None:
        def fake_fetch(endpoint, api_key, timeout):
            return ([{"id": "listed-model"}], {"http_status": 200})

        gateway = _FakeGateway([])
        receipt = asyncio.run(
            canary.audit_model_capabilities(
                base_url="https://provider.example/v1",
                api_key_env="TEST_PROVIDER_KEY",
                api_key="not-printed",
                requested_model_ids=[],
                mode="list_only",
                timeout=3.0,
                concurrency=1,
                fetcher=fake_fetch,
                gateway=gateway,
            )
        )
        self.assertEqual(["listed-model"], receipt["discovery"]["actual_model_ids"])
        self.assertEqual([], receipt["selected_model_ids"])
        self.assertEqual([], receipt["probes"])
        self.assertEqual([], gateway.requests)


if __name__ == "__main__":
    unittest.main()
