"""Synthetic task-adapter checks; no benchmark answers, model or HTTP calls."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_CONTRACT_ARTIFACT_V4 as V4,
    UpstreamMessage,
)
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.tool_runtime import StructuredAction
from tests.unit.test_healthbench_clinical_react import (
    SequenceGateway, registry, request, action, complete, observation, receipt,
    SEARCH, LOCAL, CALC,
)
from tests.unit.test_healthbench_knowledge_tools import (
    adapter as knowledge_adapter, lookup, KNOWLEDGE,
)
from src.interactive.config_loader import load_yaml
from scripts.healthbench_candidate_skill_profile import load_candidate_skill_profile, build_candidate_prompt_priors


def task(*tools, text="ZEPHYR trial", **kwargs):
    req = request(*tools, artifact_communication_profile=V4, **kwargs)
    return replace(req, problem=render_model_visible_conversation((
        {"role": "user", "content": text},
    )))


def runtime(*outputs, enabled=True):
    return HealthBenchClinicalReactExecutionAdapter(
        gateway=SequenceGateway(*outputs), tool_registry=registry(),
        max_turns=6, max_tool_calls=3, require_initial_search=False,
        require_structured_evidence_artifact=True,
        require_complete_natural_language_artifact=True,
        max_completion_artifact_characters=12000,
        public_task_validation=enabled,
    )


def test_first_search_does_not_expand_an_unresolved_study_name():
    obj = runtime()
    req = task(SEARCH, LOCAL)
    for tool in (SEARCH, LOCAL):
        wrong = StructuredAction.from_value(action(tool, "ZEPHYR weather experiment"))
        assert obj._tool_action_error(request=req, action=wrong, observations=[]) == "initial_study_lookup_must_preserve_literal_public_request"
        right = StructuredAction.from_value(action(tool, "ZEPHYR trial"))
        assert obj._tool_action_error(request=req, action=right, observations=[]) is None
    schema = obj._state_conditioned_response_schema(req, [])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(action(SEARCH, "ZEPHYR trial"))
    assert not validator.is_valid(action(SEARCH, "ZEPHYR weather experiment"))
    assert validator.is_valid(complete("A complete response with the available uncertainty stated."))


def test_successful_literal_lookup_even_empty_allows_refinement():
    obj = runtime()
    req = task(SEARCH)
    prior = observation(SEARCH, "ZEPHYR trial")
    prior["result"]["evidence"] = []
    assert obj._literal_initial_query(req, [prior]) is None
    error = observation(SEARCH, "ZEPHYR trial", error=True)
    assert obj._literal_initial_query(req, [error]) == "ZEPHYR trial"


def test_upstream_literal_receipt_avoids_duplicate_paid_lookup():
    obj = runtime()
    upstream = UpstreamMessage("producer", "node", "An inconclusive search.",
                               tool_receipts=(receipt(SEARCH, value="ZEPHYR trial"),))
    assert obj._literal_initial_query(task(SEARCH, upstream=(upstream,)), []) is None
    no_receipt = replace(upstream, tool_receipts=())
    assert obj._literal_initial_query(task(SEARCH, upstream=(no_receipt,)), []) == "ZEPHYR trial"


def test_knowledge_search_cannot_bypass_literal_rule_and_conversation_remains_free(tmp_path):
    obj, _ = knowledge_adapter(tmp_path, public_task_validation=True)
    req = task(SEARCH, KNOWLEDGE)
    wrong = StructuredAction.from_value(lookup(query="ZEPHYR weather experiment"))
    assert obj._tool_action_error(request=req, action=wrong, observations=[]) == "initial_study_lookup_must_preserve_literal_public_request"
    conversation = lookup(database="conversation", query="speaker and source context")
    assert obj._tool_action_error(request=req, action=StructuredAction.from_value(conversation), observations=[]) is None
    schema = obj._state_conditioned_response_schema(req, [])
    validator = Draft202012Validator(schema)
    assert validator.is_valid(conversation)
    assert validator.is_valid(lookup(query="ZEPHYR trial"))
    assert not validator.is_valid(lookup(query="ZEPHYR weather experiment"))


def test_legacy_mode_and_general_tasks_do_not_gain_a_forced_search():
    obj = runtime(enabled=False)
    req = task(SEARCH)
    assert obj._literal_initial_query(req, []) is None
    assert Draft202012Validator(obj._state_conditioned_response_schema(req, [])).is_valid(action(SEARCH, "ZEPHYR research context"))
    obj = runtime()
    assert obj._literal_initial_query(task(CALC, text="Convert this duration into hours."), []) is None
    assert obj._literal_initial_query(task(SEARCH, text="Please discuss the clinical situation.\nI am asking about a study."), []) is None


@pytest.mark.parametrize("text", [
    "No clinical trials exist for this intervention.",
    "No randomized controlled trials were identified for this intervention.",
])
def test_unqualified_research_absence_yields_repair_observation(text):
    obj = runtime()
    assert obj._completion_error(action=StructuredAction.from_value(complete(text)), artifact=text, tool_receipts=[receipt(SEARCH)]) == "limited_search_cannot_establish_absence_state_retrieval_scope_or_supported_finding"


@pytest.mark.parametrize("text", [
    "Our search found no matching study; this does not prove that none exists.",
    "No clinical trials were identified in the consulted databases.",
    "There is no evidence of obstruction in the supplied imaging report.",
    "The trial was completed in 2030; the reported finding applies to the stated group.",
    "Yes.",
])
def test_bounded_uncertainty_clinical_negation_and_short_answers_are_not_blocked(text):
    obj = runtime()
    assert obj._completion_error(action=StructuredAction.from_value(complete(text)), artifact=text, tool_receipts=[]) is None


def test_status_only_is_repaired_in_same_react_loop_without_forced_extra_agent():
    final = "The requested text is provided here, preserving the original qualifications."
    obj = runtime(complete("Translation complete."), complete(final))
    response = asyncio.run(obj.execute(task(SEARCH, text="Translate the supplied paragraph.", output=True)))
    assert response.text == final
    trace = response.metadata["react_trace"]
    assert len(trace) == 2
    assert trace[0]["observation_status"] == "schema_invalid"
    assert trace[-1]["observation_status"] == "completed"
    assert response.metadata["tool_calls"] == 0


def test_failed_source_claim_is_repaired_in_same_existing_budget():
    final = "No trials were identified in our search; this cannot establish nonexistence."
    obj = runtime(complete("No clinical trials exist."), complete(final))
    response = asyncio.run(obj.execute(task(SEARCH, output=True)))
    assert response.text == final
    assert len(response.metadata["react_trace"]) == 2
    assert response.metadata["tool_calls"] == 0


def test_new_config_is_explicit_inference_only_and_keeps_old_run_frozen():
    from scripts.train_agentgraph_smoke import _healthbench_tool_runtime_settings
    from tests.unit.test_healthbench_retry_scope_runtime_wiring import _task
    root = Path(__file__).resolve().parents[2]
    old = load_yaml(root / "config/evaluation_healthbench_failure_skills_full525.yaml", expand_env=False)
    new = load_yaml(root / "config/evaluation_healthbench_public_task_repair_full525.yaml", expand_env=False)
    assert "public_task_validation" not in old["healthbench_tool_runtime"]
    assert new["healthbench_tool_runtime"]["public_task_validation"] is True
    assert _healthbench_tool_runtime_settings(new, _task())["public_task_validation"] is True
    assert _healthbench_tool_runtime_settings(old, _task())["public_task_validation"] is False
    assert new["agent_graph"]["artifact_communication_profile"] == V4
    for section in ("data", "director", "evaluation", "grpo", "policy_sync", "skills",
                    "healthbench_professional_evaluation"):
        assert new[section] == old[section]
    assert new["experiment"]["training_enabled"] is False
    assert new["experiment"]["condition_id"] != old["experiment"]["condition_id"]
    assert new["storage"]["trajectories_path"] != old["storage"]["trajectories_path"]
    profile = load_candidate_skill_profile(root / new["candidate_skill_evaluation"]["profile_path"], run_config=new)
    assert len(build_candidate_prompt_priors(profile)) == 3
