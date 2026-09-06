"""Synthetic public-Observation regressions; no model or grader calls."""
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from tests.unit.test_scope_neutral_contract_admission import _registry


def make_env(evidence, *, error_type=None, tool_id="healthbench-authoritative.search"):
    registry = _registry()
    env = AgentWorkflowEnv(
        registry,
        runtime=AgentRuntime(registry, object(), dataset_id="healthbench_professional"),
        problem="Identify the documented intervention from the conversation.",
        require_scope_neutral_contracts=True,
    )
    env._progressive_output_metadata["source"] = {
        "tool_receipts": [{
            "tool_id": tool_id,
            "request": {"action": "search", "arguments": {"query": "P7-P9"}},
            "error_type": error_type,
            "result": {"completed": True, "value": {
                "operation": "search", "query": "P7-P9", "evidence": evidence,
                "knowledge_index": {"directory": "invented/P8-P10"},
            }},
        }],
    }
    return env


def test_empty_search_query_echo_cannot_ground_a_contract_literal():
    texts = make_env([])._public_contract_scope_grounding_texts()
    assert not any("P7-P9" in text or "P8-P10" in text for text in texts)


def test_successful_source_excerpt_still_grounds_literal_without_id_or_query():
    texts = make_env([{
        "title": "Observed source",
        "excerpt": "The document states P3-P5 for this intervention.",
        "document_id": "P11-P13",
    }])._public_contract_scope_grounding_texts()
    assert any("P3-P5" in text for text in texts)
    assert not any("P7-P9" in text or "P11-P13" in text for text in texts)


def test_failed_receipt_cannot_ground_literal():
    texts = make_env([{"excerpt": "P3-P5"}], error_type="TransportError")._public_contract_scope_grounding_texts()
    assert not any("P3-P5" in text for text in texts)


def test_original_conversation_remains_available():
    texts = make_env([])._public_contract_scope_grounding_texts()
    assert "Identify the documented intervention from the conversation." in texts
