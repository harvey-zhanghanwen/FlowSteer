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
    is_format_agent: bool = False,
    execution_mode: str = "reasoning",
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
        agent=AgentNode(
            "agent",
            "model",
            "verify carefully",
            execution_mode=execution_mode,
        ),
        model=model,
        provider=provider,
        phase=phase,
        is_output_agent=is_output_agent,
        is_format_agent=is_format_agent,
        communication_condition=communication_condition,
        upstream=(
            UpstreamMessage(
                "source",
                "agent",
                "evidence",
                graph_revision=1,
                request_or_dependency="verify carefully",
            ),
        ),
        own_draft="own" if phase is ExecutionPhase.REVISION else None,
        peer_draft=(
            UpstreamMessage(
                "peer",
                "agent",
                "peer draft",
                message_type="candidate",
                graph_revision=1,
                request_or_dependency="verify carefully",
            )
            if phase is ExecutionPhase.REVISION
            else None
        ),
    )


class MessageTests(unittest.TestCase):
    def test_revision_prompt_uses_immutable_own_and_peer_drafts(self) -> None:
        messages = build_agent_messages(request(ExecutionPhase.REVISION))
        text = messages[1]["content"]
        self.assertIn("Your draft:\nown", text)
        self.assertIn("Peer artifact envelope", text)
        self.assertIn("source_agent: peer", text)
        self.assertIn("message_type: candidate", text)
        self.assertIn("artifact:\npeer draft", text)
        self.assertIn("External upstream messages", text)
        self.assertIn("source_agent: source", text)
        self.assertIn("target_agent: agent", text)
        self.assertIn("request_or_dependency: verify carefully", text)
        self.assertIn("exactly <answer>answer span</answer>", messages[0]["content"])
        self.assertIn("must not be JSON", messages[0]["content"])
        self.assertIn("exactly one listed executable action", messages[0]["content"])

    def test_intermediate_contract_forbids_task_level_answer_tag(self) -> None:
        messages = build_agent_messages(request(is_output_agent=False))
        system = messages[0]["content"]
        self.assertIn("intermediate AgentGraph node", system)
        self.assertIn("do not use <answer> tags", system)
        self.assertNotIn("unique Output Agent", system)

    def test_react_output_turn_uses_structured_action_not_answer_wrapper(self) -> None:
        messages = build_agent_messages(request(execution_mode="react"))
        system = messages[0]["content"]

        self.assertIn("StructuredAction JSON object", system)
        self.assertIn("complete action", system)
        self.assertIn("Do not emit <answer> tags", system)
        self.assertNotIn("unique Output Agent", system)

    def test_format_agent_extracts_one_upstream_solution_without_resolving(self) -> None:
        messages = build_agent_messages(request(is_format_agent=True))
        system = messages[0]["content"]
        user = messages[1]["content"]

        self.assertIn("terminal Format Agent", system)
        self.assertIn("exactly one routed upstream artifact", system)
        self.assertIn("Do not solve the task again", system)
        self.assertIn("shortest value that directly answers", system)
        self.assertIn("emit only yes or no", system)
        self.assertIn("omit surrounding relations and modifiers", system)
        self.assertIn("return exactly <answer></answer>", system)
        self.assertNotIn("Contract:\nverify carefully", system)
        self.assertNotIn("request_or_dependency: verify carefully", user)
        self.assertIn("artifact:\nevidence", user)

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
        self.assertNotIn("artifact:\npeer draft", visible)
        self.assertEqual(2, visible.count(MASKED_UPSTREAM_CONTENT))
        self.assertIn("source_agent: source", visible)
        self.assertIn("source_agent: peer", visible)
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
