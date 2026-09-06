"""Synthetic Canvas scope/recovery regressions; no model or network calls."""

import asyncio
from dataclasses import replace
import json

import pytest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    AgentFailureRecord, AgentResponse, AgentRuntime, AgentRuntimeError, ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStateError
from src.interactive.react_execution import ReactExecutionError
from tests.unit.test_agent_graph import make_multi_provider_registry
from tests.unit.test_scope_neutral_contract_admission import _add_contract, _registry


def scope_step(contract, problem="Explain ABC findings in the supplied population."):
    env = AgentWorkflowEnv(
        _registry(), gateway=object(), problem=problem,
        require_scope_neutral_contracts=True,
    )
    result = asyncio.run(env.step(_add_contract(contract)))
    return env, result


@pytest.mark.parametrize("attachment", [
    "citations", "concise references", "source identifiers", "explicit provenance",
])
def test_output_attachment_is_not_an_unsupported_patient_entity(attachment):
    contract = f"Describe findings about the patients with {attachment}."
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract


@pytest.mark.parametrize("contract", [
    "Identify findings for Amber Bridge Cohort (ABC) in the supplied population.",
    "Search public sources for patients with Amber Bridge Cohort (ABC).",
])
def test_grounded_acronym_expansion_remains_a_search_hypothesis(contract):
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract
    assert not env._progressive_outputs  # Admission is not entity confirmation.


@pytest.mark.parametrize("contract", [
    "Diagnose Amber Bridge Cohort (ABC) as the confirmed diagnosis.",
    "Identify and recommend Amber Bridge Cohort (ABC) as the diagnosis.",
    "Identify the entity: ABC means Amber Bridge Cohort.",
    "Identify findings for Amber Bridge Cohort without retaining the acronym.",
    "Identify findings for Another Clinical Term (ABC).",
    "Identify findings for Amber Bridge Cohort (DEF).",
    "Search for patients with pseudoathletes and return the public evidence.",
    "Recommend C4-C5 as the final answer with citations.",
])
def test_output_or_search_language_does_not_admit_answer_assertions(contract):
    env, result = scope_step(contract)
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback
    assert not env.graph.nodes


class SyntheticReactAdapter:
    def __init__(self):
        self.requests = []
        self.fail = False

    async def execute(self, request):
        self.requests.append(request)
        if self.fail:
            raise ReactExecutionError(
                "react agent exhausted its unchanged turn allowance",
                react_trace=tuple(request.action_history) + ({
                    "turn": len(request.action_history) + 1,
                    "observation_status": "parse_error",
                    "public_error_code": "structured_action_parse_error",
                },),
                tool_receipts=tuple(request.prior_tool_receipts),
                model_calls=({"request_id": request.request_id, "request_status": "completed"},),
            )
        return AgentResponse("A synthetic completion using the retained evidence.")


def repair_environment(*, enabled=True, terminal=False, status="parse_error"):
    registry = make_multi_provider_registry()
    adapter = SyntheticReactAdapter()
    runtime = AgentRuntime(
        registry, object(), execution_adapters={"react": adapter},
        dataset_id="healthbench_professional", execution_profile_allowlist=[("react", ())],
    )
    graph = AgentGraph([
        AgentNode("source", "balanced", "Use the observed evidence and complete the response.",
                  execution_mode="react"),
    ], output_agent_id="source")
    env = AgentWorkflowEnv(
        registry, runtime=runtime, graph=graph, problem="Synthetic unresolved task.",
        recovery_policy="preserve_diagnose_repair_augment", execute_on_edit=False,
        allow_untried_react_model_repair=enabled,
    )
    record = AgentFailureRecord(
        request_id="measured-initial-failure", agent_id="source", phase=ExecutionPhase.SINGLE,
        graph_revision=graph.revision, error_type="ReactExecutionError",
        message="react agent exhausted turns without a valid completion",
        metadata={
            "model_id": "balanced", "tool_plan_exhausted": terminal,
            "model_calls": [{"request_id": "measured-call", "request_status": "completed"}],
            "react_trace": [{
                "turn": index, "observation_status": status,
                "public_error_code": "structured_action_parse_error" if status == "parse_error"
                else "structured_evidence_item_fields_invalid",
            } for index in (1, 2)],
            "tool_receipts": [{
                "tool_id": "synthetic.search", "error_type": None,
                "result": {"completed": True, "value": {"evidence": ["observed source"]}},
            }],
            "input_artifact_versions": {},
        },
    )
    env._record_failure_state((record,), current_agent_ids={"source"})
    # Existing record/receipt boundary after a contract-only repair that added
    # no Tool receipt. No model generation or manual action is performed here.
    env._pending_repair_receipt_count_by_agent["source"] = 1
    env._record_failure_state((replace(record, request_id="measured-repair-failure"),),
                              current_agent_ids={"source"})
    assert "source" in env._repair_exhausted_agent_ids
    return env, adapter, record


def model_modify(model_id, **fields):
    return AgentAction(action_type=AgentActionType.MODIFY_AGENT,
                       agent_id="source", model_id=model_id, **fields)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_repair_opt_in_requires_boolean(value):
    with pytest.raises(AgentWorkflowStateError, match="allow_untried_react_model_repair"):
        repair_environment(enabled=value)


@pytest.mark.parametrize("status", ["parse_error", "schema_invalid"])
def test_repair_mask_parameters_feedback_and_admission_share_one_domain(status):
    env, adapter, record = repair_environment(status=status)
    domain = ["alternate", "cheap", "fast"]
    assert env._untried_react_repair_model_ids("source") == tuple(domain)
    assert env.model_admissible_action_types() == ("modify_agent",)
    candidate = env.model_admissible_action_targets()["modify_agent"]["per_agent_candidates"][0]
    assert candidate["mutable_fields"] == ["model_id"]
    assert candidate["discrete_value_domains"]["model_id"] == domain
    feedback = json.loads(env._execution_error_feedback(
        AgentRuntimeError("synthetic", failure_records=(record,)),
    ).split("=", 1)[1])["failed_agents"][0]["preferred_repair"]
    assert feedback["admitted_model_ids"] == domain
    assert feedback["additional_model_repair_attempts_remaining"] == 1
    for model in domain:
        assert env._mandatory_repair_admission_issue(model_modify(model)) is None
    assert env._mandatory_repair_admission_issue(model_modify("balanced")) is not None
    assert env._mandatory_repair_admission_issue(
        model_modify("cheap", contract="Change the task."),
    ) is not None
    assert adapter.requests == []


def test_disabled_terminal_and_nonprotocol_failures_keep_no_extra_domain():
    for options in ({"enabled": False}, {"terminal": True}, {"status": "tool_error"}):
        env, _, _ = repair_environment(**options)
        assert env._untried_react_repair_model_ids("source") == ()
        assert "modify_agent" not in env.model_admissible_action_types()


def test_attempted_unavailable_and_incompatible_models_are_not_reoffered():
    env, _, record = repair_environment()
    env._record_failure_state((replace(record, metadata={**record.metadata, "model_id": "cheap"}),),
                              current_agent_ids={"source"})
    env._unavailable_model_ids.add("alternate")
    assert env._untried_react_repair_model_ids("source") == ("fast",)
    from unittest.mock import patch
    with patch.object(env.runtime, "model_supports_execution_profile", return_value=False):
        assert env._untried_react_repair_model_ids("source") == ()


def test_one_react_model_repair_preserves_actual_continuation_and_stops_after_failure():
    env, adapter, _ = repair_environment()
    original = env.graph.get_node("source")
    continuation = dict(env._failure_continuations["source"])
    adapter.fail = True
    env.execute_on_edit = True
    result = asyncio.run(env.step(model_modify("cheap")))
    assert result.accepted
    assert env.graph.get_node("source") == replace(original, model_id="cheap")
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert [dict(item) for item in request.prior_tool_receipts] == continuation["tool_receipts"]
    assert [dict(item) for item in request.action_history] == continuation["react_trace"]
    assert env._react_attempted_model_ids["source"] == {"balanced", "cheap"}
    assert env._untried_react_model_repair_used == {"source"}
    assert env._untried_react_repair_model_ids("source") == ()
    assert "modify_agent" not in env.model_admissible_action_types()
    assert len(env._failure_continuations["source"]["tool_receipts"]) == 1


def test_successful_react_model_repair_materializes_artifact_and_records_real_model():
    env, adapter, _ = repair_environment()
    env.execute_on_edit = True
    result = asyncio.run(env.step(model_modify("cheap")))
    assert result.accepted
    assert env._progressive_outputs["source"] == "A synthetic completion using the retained evidence."
    assert env._react_attempted_model_ids["source"] == {"balanced", "cheap"}
    assert env._untried_react_model_repair_used == {"source"}
    assert len(adapter.requests) == 1


def test_fork_retains_once_only_history_and_reset_starts_a_new_task():
    env, _, _ = repair_environment()
    assert asyncio.run(env.step(model_modify("cheap"))).accepted
    forked = env.fork()
    assert forked.allow_untried_react_model_repair is True
    assert forked._untried_react_model_repair_used == {"source"}
    assert forked._react_attempted_model_ids == {"source": {"balanced"}}
    forked._react_attempted_model_ids["source"].add("fast")
    assert "fast" not in env._react_attempted_model_ids["source"]
    env.reset("Independent synthetic task.")
    assert env._react_attempted_model_ids == {}
    assert env._untried_react_model_repair_used == set()


@pytest.mark.parametrize("contract", [
    "Identify Amber Bridge Cohort (ABC) as the definitive diagnosis.",
    "Search evidence and identify Amber Bridge Cohort (ABC) as the answer.",
])
def test_acronym_search_exemption_does_not_confirm_an_answer(contract):
    _, result = scope_step(contract)
    assert not result.accepted
    assert "unsupported answer/clinical literals" in result.feedback


def test_raw_modify_cannot_bypass_consumed_extra_attempt_or_replay_same_model():
    env, adapter, _ = repair_environment()
    adapter.fail = True
    env.execute_on_edit = True
    assert asyncio.run(env.step(model_modify("cheap"))).accepted
    revision = env.revision
    for model in ("cheap", "balanced", "fast"):
        rejected = asyncio.run(env.step(model_modify(model)))
        assert not rejected.accepted
        assert "no remaining untried model repair" in rejected.feedback
    assert env.revision == revision
    assert len(adapter.requests) == 1


def test_tool_call_count_at_limit_does_not_equal_terminal_continuation():
    env, _, record = repair_environment()
    metadata = {**record.metadata, "remaining_tool_calls": 0}
    env._record_failure_state((replace(record, metadata=metadata),),
                              current_agent_ids={"source"})
    assert env._untried_react_repair_model_ids("source")
    metadata["tool_plan_exhausted"] = True
    env._record_failure_state((replace(record, metadata=metadata),),
                              current_agent_ids={"source"})
    assert env._untried_react_repair_model_ids("source") == ()


def test_node_removal_discards_only_that_nodes_repair_bookkeeping():
    env, _, _ = repair_environment()
    env._untried_react_model_repair_used.update(("source", "removed"))
    env._react_attempted_model_ids["removed"] = {"fast"}
    env._retain_current_failure_state({"source"})
    assert env._untried_react_model_repair_used == {"source"}
    assert env._react_attempted_model_ids == {"source": {"balanced"}}


def test_initial_inquiry_verb_is_not_part_of_the_acronym_expansion():
    contract = "Investigate Amber Bridge Cohort (ABC) using public sources."
    env, result = scope_step(contract)
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract


def test_one_old_protocol_error_does_not_qualify_as_repeated_protocol_failure():
    env, _, record = repair_environment()
    env._latest_failure_record_by_agent["source"] = replace(
        record, metadata={**record.metadata, "react_trace": record.metadata["react_trace"][:1]},
    )
    assert env._untried_react_repair_model_ids("source") == ()
