from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import (
    OpenAICompatibleGateway,
    OpenAICompatibleGatewayError,
    MASKED_UPSTREAM_CONTENT,
    build_agent_messages,
)


def request(
    phase: ExecutionPhase = ExecutionPhase.SINGLE,
    *,
    keyed: bool = False,
    is_output_agent: bool = True,
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL,
) -> AgentRequest:
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
        is_output_agent=is_output_agent,
        communication_condition=communication_condition,
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
        self.assertIn("Peer draft from peer:\npeer draft", text)
        self.assertIn("External upstream messages", text)
        self.assertIn("exactly <answer>answer span</answer>", messages[0]["content"])
        self.assertIn("exactly one listed executable action", messages[0]["content"])

    def test_intermediate_contract_forbids_task_level_answer_tag(self) -> None:
        messages = build_agent_messages(request(is_output_agent=False))
        system = messages[0]["content"]
        self.assertIn("intermediate AgentGraph node", system)
        self.assertIn("do not use <answer> tags", system)
        self.assertNotIn("unique Output Agent", system)

    def test_masked_condition_preserves_receipt_but_masks_visible_messages(self) -> None:
        item = request(
            ExecutionPhase.REVISION,
            communication_condition=CommunicationCondition.UPSTREAM_MASKED,
        )
        messages = build_agent_messages(item)
        visible = messages[1]["content"]
        self.assertEqual("evidence", item.upstream[0].content)
        self.assertEqual("peer draft", item.peer_draft.content)  # type: ignore[union-attr]
        self.assertNotIn("\nevidence", visible)
        self.assertNotIn("peer draft\npeer draft", visible)
        self.assertEqual(2, visible.count(MASKED_UPSTREAM_CONTENT))
        self.assertIn("Message from source", visible)
        self.assertIn("Peer draft from peer", visible)
        self.assertIn("Your draft:\nown", visible)

    def test_revision_without_drafts_is_rejected(self) -> None:
        broken = request(ExecutionPhase.SINGLE)
        object.__setattr__(broken, "phase", ExecutionPhase.REVISION)
        with self.assertRaises(OpenAICompatibleGatewayError):
            build_agent_messages(broken)


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


if __name__ == "__main__":
    unittest.main()
