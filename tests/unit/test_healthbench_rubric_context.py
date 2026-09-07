"""Synthetic oracle-information adapter tests; no model/HTTP/grader calls."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from scripts.healthbench_rubric_context import (
    BANK_VERSION, attach_context, build_bank, condition_receipt,
)
from src.interactive.healthbench_professional_adapter import (
    parse_model_visible_conversation, render_model_visible_conversation,
)
from src.interactive.healthbench_professional_grader import load_private_cases
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.records import TaskRecord
from tests.unit.test_healthbench_public_task_validation import runtime, task as agent_task
from tests.unit.test_healthbench_knowledge_tools import adapter, complete, KNOWLEDGE
from tests.unit.test_completion_benchmark_round import _MODULE, _evaluation_config


MESSAGES = [{"role": "user", "content": "ZEPHYR trial"}]


@pytest.fixture
def bank(tmp_path):
    source = tmp_path / "private.jsonl"
    cases = [
        {"task_id": f"healthbench-professional:synthetic-{i}", "prompt": MESSAGES,
         "rubric_items": [
             {"criterion_text": "Describe the requested fictional study.", "points": 5},
             {"criterion_text": "Mistakes it for a weather study.", "points": -3},
         ], "physician_response": "PRIVATE_PHYSICIAN_TEXT"}
        for i in range(525)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in cases))
    manifest = build_bank(source, tmp_path / "bank")
    config = {"experiment": {"training_enabled": False}, "healthbench_rubric_context": {
        "enabled": True, "protocol": "rubric_aware",
        "acknowledge_test_information_exposure": True, "bank_version": BANK_VERSION,
        "manifest_path": "bank/manifest.json",
    }}
    tasks = tuple(TaskRecord(row["task_id"], render_model_visible_conversation(MESSAGES),
                             None, "test", {"dataset_key": "healthbench_professional"})
                  for row in cases[:2])
    return config, tasks, manifest, source


def test_bank_projects_criteria_not_physician_answers_and_labels_sign(bank, tmp_path):
    config, tasks, manifest, source = bank
    assert manifest["case_count"] == 525
    assert manifest["criterion_count"] == 1050
    assert manifest["paid_model_calls"] == 0
    assert "PRIVATE_PHYSICIAN_TEXT" not in (tmp_path / "bank/criteria.jsonl").read_text()
    augmented = attach_context(tasks, config, tmp_path)
    messages = parse_model_visible_conversation(augmented[0].question)
    assert messages[1:] == tuple(MESSAGES)
    assert '"direction": "avoid"' in messages[0]["content"]
    assert '"direction": "satisfy"' in messages[0]["content"]
    assert parse_model_visible_conversation(augmented[0].question, include_rubric_context=False) == tuple(MESSAGES)
    assert augmented[0].task_id == tasks[0].task_id
    assert augmented[0].ground_truth == tasks[0].ground_truth
    assert augmented[0].metadata["rubric_context"]["evaluation_information_visible"] is True
    # Official grader case still has exactly the original conversation.
    assert load_private_cases(source)[tasks[0].task_id]["prompt"] == MESSAGES
    assert condition_receipt(config)["comparable_to_rubric_hidden_baseline"] is False


def test_hidden_condition_unchanged_and_explicit_opt_in_required(bank, tmp_path):
    config, tasks, _, _ = bank
    assert attach_context(tasks, {}, tmp_path) == tasks
    assert condition_receipt({})["protocol"] == "rubric_hidden"
    for changes in ({"enabled": "true"}, {"protocol": "standard"},
                    {"acknowledge_test_information_exposure": False}, {"bank_version": "changed"}):
        bad = deepcopy(config)
        bad["healthbench_rubric_context"].update(changes)
        with pytest.raises(ValueError):
            attach_context(tasks, bad, tmp_path)
    with pytest.raises(ValueError, match="evaluation-only"):
        attach_context(tasks, {**config, "experiment": {"training_enabled": True}}, tmp_path)
    with pytest.raises(ValueError, match="restricted"):
        attach_context((replace(tasks[0], metadata={"dataset_key": "hotpotqa"}),), config, tmp_path)
    with pytest.raises(ValueError, match="twice"):
        attach_context(attach_context(tasks, config, tmp_path), config, tmp_path)


def test_direct_and_graph_common_task_freeze_prevents_hidden_resume(bank, tmp_path):
    context_config, tasks, _, _ = bank
    config = _evaluation_config("healthbench_professional")
    config.update(context_config)
    config["healthbench_professional_evaluation"]["split"] = "test"
    source = tmp_path / "tasks.jsonl"
    _MODULE._atomic_jsonl(source, [
        {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()} for task in tasks
    ])
    config["data"]["test_path"] = str(source)
    frozen = tmp_path / "selected.jsonl"
    first = _MODULE._select_tasks(config, tmp_path, frozen)
    assert _MODULE._select_tasks(config, tmp_path, frozen) == first
    assert first[0].metadata["rubric_context"]["bank_version"] == BANK_VERSION
    config.pop("healthbench_rubric_context")
    with pytest.raises(_MODULE.CompletionBenchmarkRoundError, match="differs"):
        _MODULE._select_tasks(config, tmp_path, frozen)


def test_oracle_context_reaches_native_agent_without_corrupting_public_lookup(bank, tmp_path):
    config, tasks, _, _ = bank
    augmented = attach_context(tasks, config, tmp_path)[0]
    request = replace(agent_task(KNOWLEDGE, output=True), problem=augmented.question)
    messages = build_agent_messages(request)
    assert sum(message["role"] == "system" for message in messages) == 1
    assert "Rubric-aware research condition" in messages[0]["content"]
    assert any(message == MESSAGES[0] for message in messages)
    assert runtime()._literal_initial_query(request, []) == "ZEPHYR trial"


def test_rubric_is_not_indexed_as_patient_statement_or_medical_evidence(bank, tmp_path):
    config, tasks, _, _ = bank
    augmented = attach_context(tasks, config, tmp_path)[0]
    obj, _ = adapter(tmp_path / "knowledge", complete("The study requires source verification."))
    request = replace(agent_task(KNOWLEDGE, output=True), problem=augmented.question)
    response = asyncio.run(obj.execute(request))
    assert response.metadata["knowledge_index"]["counts"] == {
        "conversation": 1, "medical_references": 0, "drug_labels": 0,
    }
