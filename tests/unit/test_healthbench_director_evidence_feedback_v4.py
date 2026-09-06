"""Synthetic Canvas/Director feedback regressions; no model or Tool calls."""

import json
import unittest
from dataclasses import replace

from src.interactive.agent_runtime import (
    AgentFailureRecord, AgentResponse, AgentRuntime, ExecutionPhase, UpstreamMessage,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_PROMPT_VERSION_V20,
    decode_director_transcript,
)
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


V4 = "producer_context_structured_evidence_v4"
V3 = "producer_context_structured_evidence_v3"
QUESTION = render_model_visible_conversation(({
    "role": "user", "content": "Explain the requested relation without inventing patient facts.",
},))


def _evidence(doc="synthetic-source", excerpt="The recommendation applies only to population A, except subgroup B."):
    return {"document_id": doc, "source": "Synthetic literature", "title": "Synthetic source",
            "date": "2031", "url": "https://example.invalid/" + doc, "excerpt": excerpt}


def _receipt(row):
    return {"tool_id": "healthbench-authoritative.search", "error_type": None,
            "request": {"action": "search", "arguments": {"query": "requested relation"}},
            "result": {"completed": True, "value": {"operation": "search", "query": "requested relation", "evidence": [row]}}}


def _artifact(row=None):
    return json.dumps({"schema_version": "healthbench.structured-evidence.v1", "status": "complete",
        "summary": "Opening statement. " + "Relevant context. " * 15 + " MIDDLE NECESSARY CONDITION. " + "Additional context. " * 15,
        "evidence_items": [] if row is None else [{**{key: value for key, value in row.items() if key != "excerpt"},
            "evidence_span": row["excerpt"], "supported_claim": "The requested relation is conditional.",
            "conditions_or_qualifiers": "Only population A; except subgroup B."}],
        "uncertainties": ["Source appraisal is not independent verification."]})


def _trace(row):
    return [{"turn": 1, "structured_action": {"kind": "tool", "name": "search", "resource_id": "healthbench-authoritative.search", "arguments": {"query": "requested relation"}},
             "observation": {"observation_status": "success", "tool_id": "healthbench-authoritative.search", "result": {"operation": "search", "evidence": [row]}}},
            {"turn": 2, "structured_action": {"kind": "complete", "arguments": {"value": "Public answer"}}, "observation_status": "completed", "reasoning_content": "PRIVATE_REASONING_SENTINEL"}]


class _Gateway:
    async def generate(self, request):
        row = _evidence()
        return AgentResponse(_artifact(row), metadata={"tool_receipts": [_receipt(row)], "react_trace": _trace(row), "finish_reason": "stop"})


def _env(profile=V4, dataset="healthbench_professional"):
    registry = ModelRegistry([ProviderSpec("synthetic", kind="test")], [ModelSpec("model-a", "synthetic")])
    runtime = AgentRuntime(registry, _Gateway(), dataset_id=dataset, artifact_communication_profile=profile)
    env = AgentWorkflowEnv(registry, runtime=runtime, execute_on_edit=True, max_agents=8)
    env.reset(QUESTION)
    return env, registry


def _add(agent_id="worker", output=True):
    action = {"action": "add_subgraph", "agents": [{"agent_id": agent_id, "model_id": "model-a", "contract": "Assess the requested relation and preserve necessary conditions."}], "relations": []}
    if output:
        action["output_agent_id"] = agent_id
    return json.dumps(action)


class HealthBenchDirectorEvidenceFeedbackV4Tests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_continue_prompt_preserves_middle_condition_and_bound_evidence(self):
        env, registry = _env()
        orchestrator = AgentGraphOrchestrator(registry, object(), prompt_version=DIRECTOR_PROMPT_VERSION_V20)
        initial = orchestrator.build_prompt(env, 0, ())
        action = _add()
        result = await env.step(action)
        self.assertTrue(result.accepted)
        continued = orchestrator.continue_prompt(initial, action, env, ())
        transcript = decode_director_transcript(continued)
        payload = json.loads(transcript[-1]["content"].partition("\n\n")[2])
        row = payload["current_artifact_receipts"][0]
        self.assertIn("MIDDLE NECESSARY CONDITION", row["producer_artifact"]["summary"])
        evidence = row["retrieval_evidence"]["evidence_receipts"][0]
        self.assertEqual(_evidence()["excerpt"], evidence["excerpts"][0])
        self.assertEqual("Only population A; except subgroup B.", evidence["producer_interpretations"][0]["conditions_or_qualifiers"])
        self.assertEqual("retrieved-not-endorsed", evidence["evidence_status"])
        feedback = json.loads(result.feedback.split("execution_result=", 1)[1])
        call = feedback["agent_call_receipts"][0]
        self.assertEqual([1, 2], [event["turn"] for event in call["react_action_observations"]])
        self.assertEqual("model-a", call["model_id"])
        self.assertEqual(row["artifact_version"], call["artifact_version"])
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", continued)
        self.assertNotIn(_evidence()["excerpt"], payload["canvas_feedback"])

    async def test_failed_completion_keeps_successful_retrieval_in_next_observation(self):
        env, registry = _env()
        action = _add()
        await env.step(action)
        env._progressive_outputs.clear()
        env._progressive_output_metadata.clear()
        row = _evidence("retained-on-failure")
        env._failure_continuations["worker"] = {"tool_receipts": [_receipt(row)], "react_trace": _trace(row)}
        env._unresolved_dirty_agents.add("worker")
        orchestrator = AgentGraphOrchestrator(registry, object(), prompt_version=DIRECTOR_PROMPT_VERSION_V20)
        initial = orchestrator.build_prompt(env, 0, ())
        continued = orchestrator.continue_prompt(initial, action, env, ())
        payload = json.loads(decode_director_transcript(continued)[-1]["content"].partition("\n\n")[2])
        projected = payload["current_artifact_receipts"][0]
        self.assertEqual("failed", projected["execution_status"])
        self.assertFalse(projected["artifact_complete"])
        self.assertEqual("retained-on-failure", projected["retrieval_evidence"]["evidence_receipts"][0]["document_id"])
        self.assertEqual([1, 2], [event["turn"] for event in projected["public_action_observations"]])

    async def test_empty_producer_citations_do_not_hide_source(self):
        env, _ = _env()
        await env.step(_add())
        env._progressive_outputs["worker"] = _artifact()
        evidence = env.current_artifact_receipts()[0]["retrieval_evidence"]
        self.assertEqual(0, evidence["artifact_reference_count"])
        self.assertEqual("synthetic-source", evidence["evidence_receipts"][0]["document_id"])

    async def test_v3_and_non_healthbench_keep_legacy_preview_boundary(self):
        for profile, dataset in ((V3, "healthbench_professional"), (V4, "hotpotqa")):
            env, _ = _env(profile, dataset)
            await env.step(_add())
            row = env.current_artifact_receipts()[0]
            self.assertNotIn("producer_artifact", row)
            self.assertNotIn("retrieval_evidence", row)
            self.assertLessEqual(len(row["artifact_preview"]), 320)
            self.assertNotIn("latest_user_request", env.task_goal_receipt())

    async def test_budget_and_all_producers_survive_non_ascii_eight_agent_failure_state(self):
        env, _ = _env()
        for index in range(8):
            await env.step(_add("worker_" + str(index), output=index == 0))
        for index in range(8):
            agent = "worker_" + str(index)
            row = _evidence("document-" + str(index), "证据条件。" * 3000)
            env._progressive_outputs[agent] = "必要上下文。" * 3000
            env._progressive_output_metadata[agent] = {"artifact_version": agent + "-v1", "tool_receipts": [_receipt(row)]}
            if index % 2:
                env._failure_continuations[agent] = {"tool_receipts": [_receipt(row)], "react_trace": _trace(row)}
        projected = env.current_artifact_receipts()
        self.assertEqual(8, len(projected))
        self.assertEqual({"worker_" + str(i) for i in range(8)}, {row["agent_id"] for row in projected})
        self.assertLessEqual(len(json.dumps(projected, ensure_ascii=False, separators=(",", ":"))), 24_000)
        self.assertTrue(any(row["projection_truncated"] for row in projected))

    async def test_repeated_source_body_is_not_duplicated_between_agents(self):
        env, _ = _env()
        await env.step(_add("first"))
        await env.step(_add("second", output=False))
        projected = env.current_artifact_receipts()
        bodies = [body for row in projected for body in row.get("retrieval_evidence", {}).get("evidence_receipts", [])]
        self.assertEqual(1, len(bodies))
        text = json.dumps(projected)
        self.assertIn("previously_projected_source_interpretations", text)
        self.assertIn("first", text)
        self.assertIn("second", text)

    async def test_failed_attempt_does_not_relabel_retained_artifact_inputs(self):
        env, _ = _env()
        await env.step(_add())
        previous = dict(env._progressive_output_metadata["worker"])
        previous["input_artifact_versions"] = {"source": "source-v1"}
        env._previous_revision_outputs["worker"] = env._progressive_outputs.pop("worker")
        env._previous_revision_output_metadata["worker"] = previous
        env._progressive_output_metadata.clear()
        new_row = _evidence("failed-attempt-source")
        env._failure_continuations["worker"] = {
            "tool_receipts": [_receipt(new_row)], "react_trace": _trace(new_row),
            "input_artifact_versions": {"source": "source-v2"},
        }
        row = env.current_artifact_receipts()[0]
        self.assertEqual(previous["artifact_version"], row["artifact_version"])
        self.assertEqual({"source": "source-v1"}, row["input_artifact_versions"])
        self.assertEqual({"source": "source-v2"}, row["failed_attempt_input_artifact_versions"])
        self.assertTrue(row["retained_previous_artifact"])
        self.assertEqual("current_graph_not_artifact_producer", row["node_declaration_scope"])
        self.assertEqual(row["artifact_version"], row["artifact_producer_request_id"])
        self.assertFalse(row["artifact_fresh"])
        sources = row["retrieval_evidence"]["evidence_receipts"]
        self.assertEqual({"synthetic-source", "failed-attempt-source"}, {item["document_id"] for item in sources})
        failed_source = next(item for item in sources if item["document_id"] == "failed-attempt-source")
        self.assertEqual([], failed_source["producer_interpretations"])

    async def test_provider_failure_without_tool_trace_preserves_previous_artifact(self):
        env, _ = _env()
        await env.step(_add())
        env._previous_revision_outputs["worker"] = env._progressive_outputs.pop("worker")
        env._previous_revision_output_metadata["worker"] = env._progressive_output_metadata.pop("worker")
        env._latest_failure_record_by_agent["worker"] = AgentFailureRecord(
            request_id="attempt-2", agent_id="worker", phase=ExecutionPhase.SINGLE,
            graph_revision=2, error_type="ProviderTimeout", message="Synthetic timeout",
            metadata={"input_artifact_versions": {"upstream": "upstream-v2"}},
        )
        row = env.current_artifact_receipts()[0]
        self.assertEqual("failed", row["execution_status"])
        self.assertTrue(row["retained_previous_artifact"])
        self.assertFalse(row["artifact_fresh"])
        self.assertEqual("attempt-2", row["failed_attempt_request_id"])
        self.assertEqual(2, row["failed_attempt_graph_revision"])
        self.assertEqual({"upstream": "upstream-v2"}, row["failed_attempt_input_artifact_versions"])
        self.assertIn("MIDDLE NECESSARY CONDITION", row["producer_artifact"]["summary"])

    async def test_received_peer_evidence_is_projected_without_direct_tool_calls(self):
        env, _ = _env()
        await env.step(_add())
        evidence = _evidence("peer-source")
        env._progressive_output_metadata["worker"]["tool_receipts"] = []
        env._progressive_output_metadata["worker"]["input_artifact_provenance"] = [{
            "source_agent_id": "peer", "target_agent_id": "worker", "artifact_version": "peer-draft-v1",
            "message_type": "peer_draft", "artifact": _artifact(evidence),
            "tool_receipts": [_receipt(evidence)],
        }]
        row = env.current_artifact_receipts()[0]
        self.assertEqual(0, row["tool_receipt_count"])
        self.assertEqual("peer-draft-v1", row["upstream_artifacts"][0]["artifact_version"])
        self.assertEqual("peer-source", row["retrieval_evidence"]["evidence_receipts"][0]["document_id"])
        self.assertEqual("peer", row["retrieval_evidence"]["evidence_receipts"][0]["producer_agent_id"])

    async def test_inner_artifact_truncation_is_disclosed_at_record_boundary(self):
        env, _ = _env()
        await env.step(_add())
        artifact = json.loads(_artifact())
        artifact["summary"] = "long synthetic statement " * 160
        env._progressive_outputs["worker"] = json.dumps(artifact)
        row = env.current_artifact_receipts()[0]
        self.assertEqual("partial", row["producer_artifact"]["projection_status"])
        self.assertTrue(row["projection_truncated"])

    async def test_seventh_fan_in_receipt_survives_v4_projection(self):
        env, _ = _env()
        result = await env.step(_add())
        call = result.execution.calls[0]
        upstream = tuple(UpstreamMessage(
            "source-" + str(index), "worker", "Synthetic evidence " + str(index),
            artifact_version="version-" + str(index),
        ) for index in range(7))
        call = replace(call, request=replace(call.request, upstream=upstream))
        execution = replace(result.execution, calls=(call,))
        enriched = env._agent_call_receipts(execution, enriched=True)[0]
        legacy = env._agent_call_receipts(execution)[0]
        self.assertEqual(7, len(enriched["upstream_communications"]))
        self.assertEqual(6, len(legacy["upstream_communications"]))
        self.assertEqual("version-6", enriched["upstream_communications"][-1]["artifact_version"])

    async def test_reciprocal_draft_receipts_do_not_claim_current_revision_bodies(self):
        env, _ = _env()
        result = await env.step(json.dumps({
            "action": "add_subgraph", "agents": [
                {"agent_id": "left", "model_id": "model-a", "contract": "Assess one interpretation."},
                {"agent_id": "right", "model_id": "model-a", "contract": "Assess another interpretation."},
            ], "relations": [{"source_id": "left", "target_id": "right", "source_to_target": True, "target_to_source": True}],
            "output_agent_id": "right",
        }))
        self.assertTrue(result.accepted)
        calls = env._agent_call_receipts(result.execution, enriched=True)
        self.assertEqual(4, len(calls))
        drafts = [call for call in calls if call["phase"] == "draft"]
        revisions = [call for call in calls if call["phase"] == "revision"]
        self.assertEqual(2, len(drafts))
        self.assertTrue(all(not call["artifact_is_current"] for call in drafts))
        self.assertTrue(all(call["artifact_body_location"] == "trajectory.agent_execution" for call in drafts))
        self.assertTrue(all(call["artifact_is_current"] for call in revisions))
        draft_versions = {call["agent_id"]: call["artifact_version"] for call in drafts}
        for call in revisions:
            peer = call["peer_draft"]
            self.assertEqual(draft_versions[peer["source_agent_id"]], peer["artifact_version"])

    def test_latest_user_request_survives_long_conversation_goal(self):
        env, _ = _env()
        env.reset(render_model_visible_conversation((
            {"role": "user", "content": "Earlier context " * 400},
            {"role": "assistant", "content": "A previous answer."},
            {"role": "user", "content": "Clarify the exact requested relation."},
        )))
        receipt = env.task_goal_receipt()
        self.assertEqual("Clarify the exact requested relation.", receipt["latest_user_request"])
        self.assertTrue(receipt["preview_truncated"])


if __name__ == "__main__":
    unittest.main()
