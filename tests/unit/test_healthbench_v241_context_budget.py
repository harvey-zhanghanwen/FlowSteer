"""No-model regression checks for public Director context projection."""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from src.interactive.director import (
    AgentGraphOrchestrator, DirectorError, DIRECTOR_SYSTEM_PROMPT, decode_director_transcript,
    encode_director_transcript,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from tests.unit.test_director import registry, ScriptedDirector, FakeGateway
from tests.unit.test_sglang_director_context_budget import two_phase_client


class ExactCharacterClient(ScriptedDirector):
    def prompt_token_ids(self, prompt):
        # A deliberately strict synthetic tokenizer, not an estimate claimed
        # as production Qwen token usage. The client owns exact tokenization.
        return tuple(range(sum(len(m["content"]) for m in decode_director_transcript(prompt))))


def orchestrator(limit=16000, enabled=True):
    return AgentGraphOrchestrator(
        registry(), ExactCharacterClient([]), context_projection=enabled,
        max_prompt_tokens=limit,
    )


def observation(payload):
    return {"role": "user", "content": AgentGraphOrchestrator._observation_message(payload)}


def decode_payload(message):
    return json.loads(message["content"].partition("\n\n")[2])


def initial():
    return {
        "task": "Original full public conversation, qualifiers and numeric values 103/67.",
        "current_graph": {"nodes": [], "relations": [], "output_agent_id": None},
        "model_catalog": [{"model_id": "qwen35-9b"}],
        "tool_catalog": [{"tool_id": "source.read", "action_names": ["read"],
            "action_schemas": {"read": {"required": ["url"], "properties": {"url": {"type": "string"}},
                "description": "Very large Executor-only schema. " * 1700}},
            "input_schema": {"description": "Executor input." * 1000},
            "output_schema": {"description": "Executor output." * 1000},
            "availability": "available", "side_effect": "none"}],
        "admissible_action_types": ["add_subgraph", "finish"],
        "action_target_domains": {"models": ["qwen35-9b"], "relation_candidates": []},
        "finish_admissibility": {"admissible": False},
    }


def test_default_transcript_unchanged():
    messages = [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, observation(initial())]
    assert orchestrator(enabled=False)._encode_context_bounded_transcript(messages) == encode_director_transcript(messages)


def test_large_tool_schema_compacted_without_task_or_search_space_change():
    payload = initial()
    before = copy.deepcopy(payload)
    messages = [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, observation(payload)]
    result = orchestrator()._encode_context_bounded_transcript(messages)
    projected = decode_payload(decode_director_transcript(result)[-1])
    for field in ("task", "current_graph", "model_catalog", "admissible_action_types", "action_target_domains", "finish_admissibility"):
        assert projected[field] == before[field]
    assert projected["tool_catalog"][0]["action_parameters"]["read"] == {"required": ["url"], "properties": ["url"]}
    assert "output_schema" not in projected["tool_catalog"][0]
    assert payload == before
    assert len(result) < len(encode_director_transcript(messages)) // 10


def test_current_artifacts_evidence_and_source_refs_retained_once():
    payload = initial()
    text = "A public source excerpt preserving clinical context. " * 50
    payload["current_artifact_receipts"] = [
        {"agent_id": "a", "producer_artifact": "complete output", "retrieval_evidence": {"source_id": "PMID:123", "excerpt": text}},
        {"agent_id": "b", "retrieval_evidence": {"source_id": "PMID:123", "excerpt": text}},
    ]
    payload["canvas_feedback"] = 'accepted; execution_error=' + json.dumps({
        "agent_id": "b", "public_error_code": "source_mismatch",
        "tool_receipts": [{"body": text}],
    }) + '; partial_execution_result=' + json.dumps({
        "executed_agent_ids": ["a"], "agent_artifacts": [{"body": text}],
        "agent_call_receipts": [{"agent_id": "a", "phase": "draft"}],
    })
    result = orchestrator()._encode_context_bounded_transcript([
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, observation(payload),
    ])
    value = decode_payload(decode_director_transcript(result)[-1])
    receipts = value["current_artifact_receipts"]
    assert receipts[0]["retrieval_evidence"]["excerpt"] == text
    assert receipts[1]["retrieval_evidence"]["source_id"] == "PMID:123"
    assert receipts[1]["retrieval_evidence"]["excerpt"] == {
        "same_content_as_json_pointer": "/current_artifact_receipts/0/retrieval_evidence/excerpt",
    }
    assert "source_mismatch" in value["canvas_feedback"]
    assert '"agent_id":"b"' in value["canvas_feedback"]
    assert '"phase":"draft"' in value["canvas_feedback"]
    assert value["context_projection"]["original_task_truncated"] is False


def test_token_budget_removes_only_old_history_and_keeps_latest_action():
    base = initial()
    current = {"current_graph": {"output_agent_id": "a", "nodes": ["a", "b"]},
        "admissible_action_types": ["finish"], "finish_admissibility": {"admissible": True},
        "current_artifact_receipts": [{"agent_id": "a", "producer_artifact": "complete response"}],
        "recent_rejected_actions": [{"action": "set_relation", "reason": "cycle"}]}
    messages = [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, observation(base)]
    for i in range(4):
        messages.extend([{"role": "assistant", "content": f"old-{i}" + "z" * 2200}, observation({"canvas_feedback": "old status"})])
    last_action = '{"action":"set_output","agent_id":"a"}'
    messages.extend([{"role": "assistant", "content": last_action}, observation(current)])
    engine = orchestrator(limit=4500)
    result = engine._encode_context_bounded_transcript(messages)
    decoded = decode_director_transcript(result)
    assert len(engine.client.prompt_token_ids(result)) <= 4500
    assert decode_payload(decoded[1])["task"] == base["task"]
    assert "current_graph" not in decode_payload(decoded[1])
    assert decoded[-2]["content"] == last_action
    latest = decode_payload(decoded[-1])
    for field, value in current.items():
        assert latest[field] == value
    assert latest["context_projection"]["older_history_messages_removed"] > 0


def test_long_original_task_fails_explicitly_instead_of_silent_truncation():
    payload = initial()
    payload["task"] = "完整原始医疗对话" * 2000
    original = copy.deepcopy(payload)
    with pytest.raises(DirectorError, match="original task was not truncated"):
        orchestrator(limit=2000)._encode_context_bounded_transcript([
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, observation(payload),
        ])
    assert payload == original


def test_reasoning_reserves_action_space_with_exact_phase_receipts():
    client = two_phase_client([27000, 28200])
    client.reasoning_context_reserve_tokens = 4608
    response = asyncio.run(client.propose("complete public task", seed=43))
    assert client.payloads[0]["sampling_params"]["max_new_tokens"] == 1160
    assert client.payloads[1]["sampling_params"]["max_new_tokens"] == 4096
    phases = response.metadata["generation_phase_receipts"]
    assert phases["reasoning"]["context_budget"]["input_tokens"] == 27000
    assert phases["reasoning"]["context_budget"]["reserved_for_action_and_template_tokens"] == 4608
    assert phases["action"]["context_budget"]["input_tokens"] == 28200
    assert phases["action"]["context_budget"]["input_truncated"] is False


def test_projection_requires_real_tokenizer_boundary():
    with pytest.raises(ValueError, match="exact tokenizer"):
        AgentGraphOrchestrator(registry(), ScriptedDirector([]), context_projection=True, max_prompt_tokens=24000)


def test_real_canvas_edit_continuation_preserves_exact_action_and_raw_feedback():
    async def run():
        model_registry = registry()
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway(),
            problem="Complete original task with every qualifier.", execute_on_edit=True)
        engine = AgentGraphOrchestrator(model_registry, ExactCharacterClient([]),
            context_projection=True, max_prompt_tokens=24000)
        p0 = engine.build_prompt(env, 0, ())
        action = json.dumps({"action": "add_subgraph", "agents": [
            {"agent_id": "node_1", "model_id": "qwen", "contract": "produce complete response"},
        ], "relations": [], "output_agent_id": "node_1"})
        step = await env.step(action)
        assert step.accepted
        raw_feedback = env.snapshot().last_feedback
        p1 = engine.continue_prompt(p0, action, env, ())
        messages = decode_director_transcript(p1)
        assert messages[-2]["content"] == action
        assert decode_payload(messages[1])["task"] == env.problem
        assert decode_payload(messages[-1])["current_graph"] == env.graph.to_dict()
        assert decode_payload(messages[-1])["current_artifact_receipts"][0]["agent_id"] == "node_1"
        assert env.snapshot().last_feedback == raw_feedback
        assert env.history[-1].feedback == raw_feedback
    asyncio.run(run())
