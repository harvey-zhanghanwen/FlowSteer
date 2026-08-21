from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.agent_action_parser import AgentActionType


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_hotpotqa_executor_capacity.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_hotpotqa_executor_capacity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_capacity_plan_is_one_reasoning_call_then_singleton_format() -> None:
    actions = MODULE.capacity_actions("qwen3.5-flash", "qwen3.5-9b-local")
    assert [action.action_type for action in actions] == [
        AgentActionType.ADD_SUBGRAPH,
        AgentActionType.ADD_SUBGRAPH,
        AgentActionType.FINISH,
    ]

    reasoning, formatting, _finish = actions
    assert len(reasoning.agents) == 1
    assert reasoning.agents[0].agent_id == "semantic_reasoning"
    assert reasoning.agents[0].role_family == "reasoning"
    assert reasoning.agents[0].model_id == "qwen3.5-flash"
    assert reasoning.output_agent_id is None

    assert len(formatting.agents) == 1
    assert formatting.agents[0].agent_id == "format"
    assert formatting.agents[0].role_family == "format"
    assert formatting.agents[0].model_id == "qwen3.5-9b-local"
    assert formatting.output_agent_id == "format"
    assert [relation.to_dict() for relation in formatting.relations] == [
        {
            "source_id": "semantic_reasoning",
            "target_id": "format",
            "source_to_target": True,
            "target_to_source": False,
        }
    ]
    summary = MODULE._plan_summary(actions)
    assert summary["agent_count"] == 2
    assert summary["planned_model_calls"] == 2


def test_dry_run_freezes_panel_catalog_and_actions_without_api(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_gateway(*args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("dry-run constructed a model/API gateway")

    monkeypatch.setattr(MODULE, "OpenAICompatibleGateway", forbidden_gateway)
    status = MODULE.main(
        [
            "--reasoning-model-id",
            "qwen3.5-flash",
            "--format-model-id",
            "qwen3.5-9b-local",
            "--dry-run",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["api_calls_executed"] == 0
    assert payload["remote_api_calls_planned"] == 6
    assert payload["local_model_calls_planned"] == 6
    assert payload["metadata"]["task_ids"] == list(
        MODULE.PREREGISTERED_TASK_IDS
    )
    assert payload["metadata"]["reasoning_model_id"] == "qwen3.5-flash"
    assert payload["metadata"]["format_model_id"] == "qwen3.5-9b-local"
    assert payload["metadata"]["catalog"]["models"]
    assert payload["metadata"]["plan"]["actions"] == [
        action.to_dict()
        for action in MODULE.capacity_actions(
            "qwen3.5-flash", "qwen3.5-9b-local"
        )
    ]
    assert payload["controls"]["diagnostic_only"] is True
    assert payload["controls"]["director_off"] is True
    assert payload["controls"]["grpo_eligible"] is False
    assert payload["controls"]["skill_evidence_eligible"] is False
    assert payload["controls"]["ground_truth_visibility"] == (
        "evaluator_after_generation_only"
    )
    assert "ground_truth" not in payload["metadata"]
    assert "supporting_facts" not in payload["metadata"]
    assert "question_type" not in payload["metadata"]


def test_aggregate_keeps_failures_in_strict_denominator() -> None:
    completed = {
        "status": "completed",
        "operational_failure": False,
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 1.0, "token_f1": 0.5},
        },
        "api_calls": 2,
    }
    failed = {
        "status": "operational_failure",
        "operational_failure": True,
        "evaluation": None,
        "api_calls": 1,
    }
    aggregate = MODULE._aggregate([completed, failed])
    assert aggregate["denominator"] == 2
    assert aggregate["completed"] == 1
    assert aggregate["operational_failures"] == 1
    assert aggregate["strict_exact_match"] == 0.5
    assert aggregate["strict_token_f1"] == 0.25
    assert aggregate["api_calls"] == 3


def test_task_panel_cannot_be_subselected() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        MODULE._select_preregistered_tasks(
            ROOT / "data/joint_qa_v2/skill_confirmation_round7.jsonl",
            expected_split="validation",
            task_ids=MODULE.PREREGISTERED_TASK_IDS[:-1],
        )
