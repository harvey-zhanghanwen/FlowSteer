from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import (
    OpenAICompatibleGateway,
    OpenAICompatibleGatewayError,
    build_agent_messages,
)


def request(phase: ExecutionPhase = ExecutionPhase.SINGLE, *, keyed: bool = False) -> AgentRequest:
    provider = ProviderSpec(
        "provider",
        kind="openai-compatible",
        endpoint="https://example.invalid/v1",
        api_key_env="TEST_FLOWSTEER_KEY" if keyed else None,
    )
    model = ModelSpec(
        "model",
        "provider",
        model_name="remote-model-id",
        metadata={"temperature": "0.2", "max_tokens": "512"},
    )
    return AgentRequest(
        request_id="run:1:agent:single",
        run_id="run",
        graph_revision=1,
        problem="Solve the task",
        agent=AgentNode("agent", "model", "verify carefully"),
        model=model,
        provider=provider,
        phase=phase,
        upstream=(UpstreamMessage("source", "agent", "evidence"),),
        own_draft="own" if phase is ExecutionPhase.REVISION else None,
        peer_draft=(
            UpstreamMessage("peer", "agent", "peer draft")
            if phase is ExecutionPhase.REVISION
            else None
        ),
    )


class MessageTests(unittest.TestCase):
    def test_revision_prompt_uses_immutable_own_and_peer_drafts(self) -> None:
        messages = build_agent_messages(request(ExecutionPhase.REVISION))
        text = messages[1]["content"]
        self.assertIn("Your draft:\nown", text)
        self.assertIn("Peer artifact envelope:", text)
        self.assertIn("source_agent: peer", text)
        self.assertIn("artifact:\npeer draft", text)
        self.assertIn("External upstream messages", text)
        self.assertIn("intermediate AgentGraph node", messages[0]["content"])
        self.assertIn("Do not present a task-level final answer", messages[0]["content"])

    def test_revision_without_drafts_is_rejected(self) -> None:
        broken = request(ExecutionPhase.SINGLE)
        object.__setattr__(broken, "phase", ExecutionPhase.REVISION)
        with self.assertRaises(OpenAICompatibleGatewayError):
            build_agent_messages(broken)

    def test_output_treats_routed_qa_memory_as_demonstration_not_fact(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "agent",
            AgentNode("agent", "model", "answer", execution_mode="reasoning"),
        )
        object.__setattr__(item, "is_output_agent", True)
        receipt = {
            "tool_id": "hotpotqa.qa_memory",
            "request": {"action": "read"},
            "result": {"memory_id": "m1"},
            "error_type": None,
        }
        object.__setattr__(
            item,
            "upstream",
            (
                UpstreamMessage(
                    "worker",
                    "agent",
                    "retrieved candidate",
                    tool_receipts=(receipt,),
                ),
            ),
        )

        system = build_agent_messages(item)[0]["content"]

        self.assertIn("retrieved demonstrations", system)
        self.assertIn("entity binding", system)
        self.assertIn("answer slot", system)
        self.assertIn("never copy the unrelated record", system)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_response_mapping(self) -> None:
        gateway = OpenAICompatibleGateway(max_retries=0, default_seed=17)
        captured = {}

        def fake_post(url, api_key, payload):
            captured.update(url=url, api_key=api_key, payload=payload)
            return {
                "id": "req-1",
                "model": "actual-model",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

        gateway._post_json = fake_post  # type: ignore[method-assign]
        response = await gateway.generate(request())
        self.assertEqual(response.text, "answer")
        self.assertEqual(captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(captured["api_key"], "EMPTY")
        self.assertEqual(captured["payload"]["model"], "remote-model-id")
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["payload"]["seed"], 17)
        self.assertEqual(response.metadata["total_tokens"], 12)
        self.assertGreaterEqual(response.metadata["latency_ms"], 0.0)
        self.assertEqual(response.metadata["attempt_count"], 1)
        self.assertEqual(response.metadata["generation_seed"], 17)

    def test_request_level_scientific_seed_overrides_gateway_default(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="remote-model-id",
                metadata={
                    "temperature": "1.0",
                    "top_p": "1.0",
                    "generation_seed": "18446744073709551614",
                },
            ),
        )

        payload = OpenAICompatibleGateway(default_seed=17).request_payload(item)

        self.assertEqual(18446744073709551614, payload["seed"])

    async def test_missing_credential_names_variable_without_printing_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OpenAICompatibleGatewayError, "TEST_FLOWSTEER_KEY"):
                await OpenAICompatibleGateway(max_retries=0).generate(request(keyed=True))

    async def test_malformed_provider_response_is_rejected(self) -> None:
        gateway = OpenAICompatibleGateway(max_retries=0)
        gateway._post_json = lambda *_: {"choices": []}  # type: ignore[method-assign]
        with self.assertRaises(OpenAICompatibleGatewayError):
            await gateway.generate(request())

    def test_qwen_chat_template_thinking_toggle_is_explicit(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"chat_template_enable_thinking": "false"},
            ),
        )
        payload = OpenAICompatibleGateway().request_payload(item)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_skillflow_response_schema_is_forwarded_strictly(self) -> None:
        item = request()
        schema = {
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"const": "complete"}},
            "additionalProperties": False,
        }
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"response_json_schema": json.dumps(schema)},
            ),
        )

        payload = OpenAICompatibleGateway().request_payload(item)

        self.assertEqual("json_schema", payload["response_format"]["type"])
        self.assertEqual(
            {"name": "skillev_action", "schema": schema, "strict": True},
            payload["response_format"]["json_schema"],
        )

    def test_response_schema_errors_fail_closed(self) -> None:
        for raw_schema in ("{", "[]", '{"type":"not-a-json-schema-type"}'):
            with self.subTest(raw_schema=raw_schema):
                item = request()
                object.__setattr__(
                    item,
                    "model",
                    ModelSpec(
                        "model",
                        "provider",
                        model_name="supervisor_theta",
                        metadata={"response_json_schema": raw_schema},
                    ),
                )
                with self.assertRaisesRegex(
                    OpenAICompatibleGatewayError,
                    "response_json_schema",
                ):
                    OpenAICompatibleGateway().request_payload(item)


if __name__ == "__main__":
    unittest.main()
