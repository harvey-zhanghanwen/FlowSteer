from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.interactive.config_loader import load_model_registry, load_yaml
from src.interactive.task_evaluator import EvaluationOutcome


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_bounded_direct_react_exhaustion_uses_live_configured_turn_count() -> None:
    failures: list[dict[str, object]] = []
    assert _MODULE._bounded_direct_react_exhaustion_task_ids(failures) == set()

    failures.append(
        {
            "task_id": "healthbench-professional:example",
            "condition": "direct_local_qwen35_9b",
            "stage": "generation_or_evaluator",
            "error": (
                "AgentRuntimeError: gateway failed: react agent 'direct' "
                "exhausted 8 turns without a valid completion"
            ),
        }
    )
    assert _MODULE._bounded_direct_react_exhaustion_task_ids(failures) == {
        "healthbench-professional:example"
    }


def test_bounded_direct_react_exhaustion_filters_condition_stage_and_tasks() -> None:
    base = {
        "task_id": "healthbench-professional:kept",
        "condition": "direct_local_qwen35_9b",
        "stage": "generation_or_evaluator",
        "error": "react agent 'direct' exhausted 6 turns without a valid completion",
    }
    failures = [
        base,
        {**base, "task_id": "healthbench-professional:wrong-stage", "stage": "evaluator_invalid"},
        {**base, "task_id": "healthbench-professional:wrong-condition", "condition": "agentgraph"},
        {**base, "task_id": "healthbench-professional:other", "error": "provider timeout"},
    ]
    assert _MODULE._bounded_direct_react_exhaustion_task_ids(
        failures,
        task_ids={"healthbench-professional:kept"},
    ) == {"healthbench-professional:kept"}


def _evaluation_config(dataset_key: str) -> dict:
    config = deepcopy(load_yaml(_ROOT / "config" / "evaluation_hotpotqa_round_01.yaml"))
    config.pop("hotpotqa_evaluation")
    if dataset_key in {"hotpotqa", "triviaqa"}:
        section_name = f"{dataset_key}_evaluation"
        phase = f"{dataset_key}_evaluation"
        split = "validation"
        sample_count = 2
        extra = {"protocol_equivalent_to_direct": False}
    elif dataset_key == "aime_2026":
        section_name = "aime2026_evaluation"
        phase = "aime2026_evaluation"
        split = "test"
        sample_count = 30
        extra = {
            "official_2026_only": True,
            "benchmark_slice": "official_aime_2026",
        }
    elif dataset_key == "healthbench_professional":
        section_name = "healthbench_professional_evaluation"
        phase = "healthbench_professional_evaluation"
        split = "validation"
        sample_count = 2
        extra = {}
    else:
        section_name = f"{dataset_key}_evaluation"
        phase = f"{dataset_key}_evaluation"
        split = "validation"
        sample_count = 2
        extra = {}
    config["experiment"]["phase"] = phase
    config[section_name] = {
        "dataset_key": dataset_key,
        "split": split,
        "selection": "sequential",
        "sample_count": sample_count,
        "rollouts_per_task": 1,
        "concurrency": 2,
        "direct_model_id": "qwen3.5-9b-local",
        "direct_protocol": "single_call_v1",
        "direct_contract": "Complete the task and return the requested final answer.",
        **extra,
    }
    if dataset_key == "healthbench_professional":
        config["evaluation"].update(
            {
                "healthbench_grader_mode": (
                    "openai_simple_evals_healthbench_professional_reference"
                ),
                "healthbench_judge_catalog_path": (
                    "config/model_catalog_healthbench_professional_grader_v1.yaml"
                ),
                "healthbench_private_cases_path": "private_cases.jsonl",
                "healthbench_official_source_root": "simple-evals",
                "healthbench_worker_interpreter": "python",
                "healthbench_reasoning_effort": "low",
                "healthbench_length_adjustment_center": 2000.0,
                "healthbench_length_adjustment_penalty_per_500_chars": 0.0147,
                "healthbench_max_provider_attempts": 3,
            }
        )
        config["healthbench_tool_runtime"] = {"enabled": False}
    if dataset_key in {"webshop", "alfworld"}:
        config["evaluation"]["max_environment_steps_by_source"] = {
            dataset_key: 10
        }
    return config


def _healthbench_react_config() -> dict:
    config = _evaluation_config("healthbench_professional")
    condition_id = "healthbench-professional-paired-medrag-test"
    config["experiment"].update(
        {
            "condition_id": condition_id,
            "tool_version": "skillflow.medrag-textbooks-bm25-react.v1",
        }
    )
    config["healthbench_professional_evaluation"].update(
        {
            "direct_execution_mode": "react",
            "direct_completion_condition": (
                "Return one complete assistant response after any useful "
                "medical textbook retrieval."
            ),
            "direct_allowed_tools": ["healthbench-medrag.search"],
            "protocol_equivalent_to_direct": True,
        }
    )
    config["healthbench_tool_runtime"] = {
        "enabled": True,
        "condition_id": condition_id,
        "mode": "model_driven_medrag_search",
        "dataset_scope": ["healthbench_professional"],
        "resource_dir": "/tmp/frozen-medrag-fixture",
        "source_identity": "MedRAG/textbooks",
        "source_revision": "fixture-revision",
        "expected_rows": 2,
        "max_turns_per_agent_call": 3,
        "max_tool_calls_per_agent_call": 2,
        "tool_timeout_seconds": 1.0,
    }
    return config


def _healthbench_authoritative_config() -> dict:
    config = _healthbench_react_config()
    condition_id = "healthbench-professional-paired-authoritative-test"
    config["experiment"].update(
        condition_id=condition_id,
        tool_version="skillflow.medrag+ncbi-pubmed-eutils-react.v1",
    )
    config["healthbench_professional_evaluation"].update(
        direct_allowed_tools=["healthbench-authoritative.search"],
        direct_protocol=(
            "healthbench_professional_single_agent_react_authoritative_v1"
        ),
    )
    config["healthbench_tool_runtime"].update(
        condition_id=condition_id,
        mode="model_driven_authoritative_search",
        max_successful_queries=2,
        require_initial_search=True,
        authoritative_web_search={
            "enabled": True,
            "provider": "ncbi_pubmed_eutils",
            "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "tool_name": "FlowSteer-HealthBench",
            "retmax": 3,
            "request_timeout_seconds": 8.0,
            "minimum_interval_seconds": 0.4,
        },
    )
    return config


def _assert_config_rejected(config: dict, expected_check: str) -> None:
    try:
        _MODULE.validate_completion_benchmark_config(config)
    except Exception as exc:
        assert expected_check in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError(f"invalid config was accepted: {expected_check}")


def test_runner_reuses_hotpot_graph_and_stable_zero_boundaries():
    assert _MODULE._collect_graph is _MODULE.hotpot_round._collect_graph
    assert _MODULE._stable_zero_check is _MODULE.hotpot_round._stable_zero_check
    assert (
        _MODULE._trajectory_resume_matches
        is _MODULE.hotpot_round._trajectory_resume_matches
    )


def test_supported_configs_are_evaluation_only():
    for dataset_key in (
        "hotpotqa",
        "triviaqa",
        "aime_2026",
        "healthbench_professional",
        "webshop",
        "alfworld",
    ):
        config = _evaluation_config(dataset_key)
        _MODULE.validate_completion_benchmark_config(config)

        invalid = deepcopy(config)
        invalid["grpo"]["max_optimizer_updates"] = 1
        try:
            _MODULE.validate_completion_benchmark_config(invalid)
        except Exception as exc:
            assert "optimizer_updates" in str(exc)
        else:  # pragma: no cover - fail-closed guard
            raise AssertionError("an optimizer-enabled evaluation config was accepted")


def test_healthbench_reasoning_direct_requires_tool_runtime_disabled():
    config = _evaluation_config("healthbench_professional")
    _MODULE.validate_completion_benchmark_config(config)

    invalid = deepcopy(config)
    invalid["healthbench_tool_runtime"] = deepcopy(
        _healthbench_react_config()["healthbench_tool_runtime"]
    )
    _assert_config_rejected(invalid, "healthbench_tool_runtime.disabled")


def test_healthbench_react_direct_requires_paired_medrag_condition_binding():
    config = _healthbench_react_config()
    _MODULE.validate_completion_benchmark_config(config)

    invalid_cases = (
        (
            "healthbench_tool_runtime.enabled",
            {"healthbench_tool_runtime": {"enabled": False}},
        ),
        (
            "healthbench.direct_completion_condition",
            {
                "healthbench_professional_evaluation": {
                    "direct_completion_condition": ""
                }
            },
        ),
        (
            "healthbench.protocol_equivalent_to_direct",
            {
                "healthbench_professional_evaluation": {
                    "protocol_equivalent_to_direct": False
                }
            },
        ),
        (
            "healthbench.direct_allowed_tools",
            {
                "healthbench_professional_evaluation": {
                    "direct_allowed_tools": []
                }
            },
        ),
        (
            "healthbench_tool_runtime.mode",
            {"healthbench_tool_runtime": {"mode": "unsupported_mode"}},
        ),
        (
            "healthbench_tool_runtime.dataset_scope",
            {"healthbench_tool_runtime": {"dataset_scope": ["hotpotqa"]}},
        ),
        (
            "healthbench_tool_runtime.condition_id",
            {"healthbench_tool_runtime": {"condition_id": "other-condition"}},
        ),
    )
    for expected_check, overrides in invalid_cases:
        invalid = deepcopy(config)
        for section_name, section_overrides in overrides.items():
            invalid[section_name].update(section_overrides)
        _assert_config_rejected(invalid, expected_check)


def test_healthbench_authoritative_retrieval_is_a_distinct_valid_condition():
    config = _healthbench_authoritative_config()
    _MODULE.validate_completion_benchmark_config(config)

    invalid = deepcopy(config)
    invalid["healthbench_professional_evaluation"]["direct_allowed_tools"] = [
        "healthbench-medrag.search"
    ]
    _assert_config_rejected(invalid, "healthbench.direct_allowed_tools")


def test_healthbench_react_paired_profile_matches_direct_tool_condition():
    config = _healthbench_authoritative_config()
    config["healthbench_tool_runtime"]["execution_profile_allowlist"] = [
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-authoritative.search"],
        }
    ]
    _MODULE.validate_completion_benchmark_config(config)

    for invalid_profile in (
        [
            {
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
        ],
        [
            {
                "execution_mode": "react",
                "allowed_tools": ["healthbench-medrag.search"],
            }
        ],
        [],
    ):
        invalid = deepcopy(config)
        invalid["healthbench_tool_runtime"][
            "execution_profile_allowlist"
        ] = invalid_profile
        _assert_config_rejected(
            invalid,
            "healthbench_tool_runtime.execution_profile_allowlist",
        )


def test_healthbench_react_tool_availability_condition_allows_reasoning_fallback():
    config = _healthbench_authoritative_config()
    config["healthbench_professional_evaluation"][
        "protocol_equivalent_to_direct"
    ] = False
    config["healthbench_tool_runtime"]["execution_profile_allowlist"] = [
        {
            "execution_mode": "reasoning",
            "allowed_tools": [],
        },
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-authoritative.search"],
        },
    ]

    _MODULE.validate_completion_benchmark_config(config)

    for invalid_profile in (
        [
            {
                "execution_mode": "react",
                "allowed_tools": ["healthbench-authoritative.search"],
            }
        ],
        [
            {
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "execution_mode": "react",
                "allowed_tools": [],
            },
            {
                "execution_mode": "react",
                "allowed_tools": ["healthbench-authoritative.search"],
            },
        ],
    ):
        invalid = deepcopy(config)
        invalid["healthbench_tool_runtime"][
            "execution_profile_allowlist"
        ] = invalid_profile
        _assert_config_rejected(
            invalid,
            "healthbench_tool_runtime.execution_profile_allowlist",
        )


def test_evaluation_config_allows_base_director_without_lora_adapter():
    config = _evaluation_config("aime_2026")
    config["director"]["behavior_adapter_name"] = None
    config["director"]["behavior_adapter_checkpoint"] = None

    _MODULE.validate_completion_benchmark_config(config)

    config["director"]["behavior_adapter_name"] = "unexpected-adapter"
    try:
        _MODULE.validate_completion_benchmark_config(config)
    except Exception as exc:
        assert "must be supplied together" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("a partial adapter route was accepted")


def test_official_aime_initial_config_keeps_learning_tools_and_priors_disabled():
    config = load_yaml(
        _ROOT / "config" / "evaluation_aime2026_unified_initial_v1.yaml"
    )

    _MODULE.validate_completion_benchmark_config(config)

    assert config["experiment"]["training_enabled"] is False
    assert config["execution_timeout"] == 600.0
    assert config["director"]["lora"]["enabled"] is False
    assert config["aime_tool_runtime"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["skills"]["initial_library"] == []
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert config["agent_graph"]["require_format_agent"] is False
    assert config["agent_graph"]["terminal_protocol_by_source"] == {
        "aime_2026": "none"
    }
    assert config["agent_graph"]["actions"] == [
        "add_agent",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]


def test_aime_without_explicit_finish_never_calls_formal_evaluator():
    task = _MODULE.TaskRecord(
        task_id="aime-2026/01",
        question="problem",
        ground_truth="42",
        split="test",
        metadata={"dataset_key": "aime_2026"},
    )

    outcome = asyncio.run(
        _MODULE.LiveSmokeBackend.evaluate_final_graph(
            SimpleNamespace(),
            task,
            None,
            {"nodes": [], "relations": [], "revision": 0},
            rollout_index=0,
        )
    )

    assert outcome.valid is False
    assert outcome.reward is None
    assert outcome.reason == "not_evaluated_without_explicit_finish"
    assert outcome.details["formal_evaluator_called"] is False


def test_healthbench_final_graph_uses_attached_professional_grader():
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:one",
        question="Conversation:\n\n[user] Help me.\n\n[assistant]",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "healthbench_professional"},
    )
    calls = []

    async def professional_grade(task_id, candidate):
        del task_id, candidate

    async def fake_evaluate(record, prediction, **kwargs):
        calls.append((record, prediction, kwargs))
        return EvaluationOutcome(
            valid=True,
            reward=0.5,
            metrics={"overall_score_length_adjusted": 0.5},
            reason="evaluated",
        )

    backend = SimpleNamespace(
        config={"evaluation": {"max_environment_steps": 20}},
        healthbench_professional_grader=SimpleNamespace(
            grade=professional_grade
        ),
        judge=None,
        judge_model="gpt-5.4-2026-03-05",
        swe_harness=None,
    )
    with patch("train_agentgraph_smoke.evaluate_task", side_effect=fake_evaluate):
        outcome = asyncio.run(
            _MODULE.LiveSmokeBackend.evaluate_final_graph(
                backend,
                task,
                "complete response",
                {"nodes": [], "relations": [], "revision": 0},
                rollout_index=0,
            )
        )

    assert outcome.valid is True
    assert calls[0][2]["healthbench_grader"] is professional_grade


def test_healthbench_without_explicit_finish_never_calls_professional_grader():
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:terminal-failure",
        question="Conversation:\n\n[user] Help me.\n\n[assistant]",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "healthbench_professional"},
    )

    for final_answer in (None, ""):
        with patch("train_agentgraph_smoke.evaluate_task") as evaluate:
            outcome = asyncio.run(
                _MODULE.LiveSmokeBackend.evaluate_final_graph(
                    SimpleNamespace(),
                    task,
                    final_answer,
                    {"nodes": [], "relations": [], "revision": 0},
                    rollout_index=0,
                )
            )

        evaluate.assert_not_called()
        assert outcome.valid is False
        assert outcome.reward is None
        assert outcome.reason == "not_evaluated_without_explicit_finish"
        assert outcome.details == {
            "terminal_failure": True,
            "formal_evaluator_called": False,
        }
        assert (
            outcome.evaluator_version
            == _MODULE.HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION
        )


def test_healthbench_terminal_failure_is_reportable_not_operational():
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:terminal-failure",
        question="Conversation:\n\n[user] Help me.\n\n[assistant]",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "healthbench_professional"},
    )
    evaluator_version = _MODULE.HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION
    versions = {"evaluator": evaluator_version}
    evaluation = {
        "valid": False,
        "reward": None,
        "metrics": {},
        "reason": "not_evaluated_without_explicit_finish",
        "details": {
            "terminal_failure": True,
            "formal_evaluator_called": False,
        },
        "evaluator_version": evaluator_version,
    }
    trajectory = {
        "trajectory_id": "trajectory:healthbench-terminal-failure",
        "task": task.to_dict(),
        "condition_id": "healthbench-condition",
        "versions": versions,
        "turns": [],
        "final_answer": None,
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "evaluation": evaluation,
    }

    assert _MODULE.hotpot_round._reportable_terminal_failure_matches(
        trajectory,
        task=task,
        condition_id="healthbench-condition",
        versions=versions,
    )

    assert (
        _MODULE._failure_type(
            {"valid": True},
            {
                "valid": False,
                "explicit_finish": False,
                "evaluation": evaluation,
            },
            direct_valid=True,
            graph_valid=False,
            direct_score=0.5,
            graph_score=0.0,
            dataset_key="healthbench_professional",
        )
        == "agentgraph_terminal_failure"
    )

    row = {
        "task_id": task.task_id,
        "failure_type": "agentgraph_terminal_failure",
        "direct": {
            "available": True,
            "valid": True,
            "overall_score": 0.5,
            "overall_score_length_adjusted": 0.5,
            "telemetry": {},
            "evaluation": {"valid": True, "details": {}},
            "execution": None,
        },
        "agentgraph": {
            "available": True,
            "valid": False,
            "overall_score": 0.0,
            "overall_score_length_adjusted": 0.0,
            "explicit_finish": False,
            "termination_reason": "max_rounds",
            "evaluation": evaluation,
            "telemetry": {},
            "graph_diagnostic": None,
        },
    }
    report = _MODULE._report(
        [row], _evaluation_config("healthbench_professional")
    )
    assert report["terminal_failure_count"] == 1
    assert report["max_rounds_count"] == 1
    assert report["operational_failure_count"] == 0


def test_aime_terminal_failure_is_reportable_without_evaluator_retry_or_double_count():
    task = _MODULE.TaskRecord(
        task_id="aime-2026/01",
        question="problem",
        ground_truth="42",
        split="test",
        metadata={"dataset_key": "aime_2026"},
    )
    versions = {"evaluator": "skillev.private-static.integer.v1"}
    trajectory = {
        "trajectory_id": "trajectory:aime-terminal-failure",
        "task": task.to_dict(),
        "condition_id": "aime-condition",
        "versions": versions,
        "turns": [],
        "final_answer": None,
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "evaluation": {
            "valid": False,
            "reward": None,
            "metrics": {},
            "reason": "not_evaluated_without_explicit_finish",
            "details": {
                "terminal_failure": True,
                "formal_evaluator_called": False,
            },
            "evaluator_version": "skillev.private-static.integer.v1",
        },
    }

    assert _MODULE.hotpot_round._reportable_terminal_failure_matches(
        trajectory,
        task=task,
        condition_id="aime-condition",
        versions=versions,
    )

    row = {
        "task_id": task.task_id,
        "failure_type": "agentgraph_terminal_failure",
        "direct": {
            "available": True,
            "valid": True,
            "accuracy": 0.0,
            "telemetry": {},
            "evaluation": {
                "valid": True,
                "details": {"parsing_succeeded": False},
            },
            "execution": {
                "model_id": "qwen3.5-9b-local",
                "provider": "local-director",
                "error_type": None,
                "metadata": {
                    "response": {
                        "provider_id": "local-director",
                        "provider_model": "supervisor_theta",
                        "attempt_count": 1,
                        "finish_reason": "length",
                    }
                },
            },
        },
        "agentgraph": {
            "available": True,
            "valid": False,
            "accuracy": 0.0,
            "explicit_finish": False,
            "termination_reason": "max_rounds",
            "evaluation": trajectory["evaluation"],
            "telemetry": {},
            "graph_diagnostic": None,
        },
    }
    report = _MODULE._report([row], _evaluation_config("aime_2026"))
    assert report["terminal_failure_count"] == 1
    assert report["max_rounds_count"] == 1
    assert report["direct_parsing_failure_count"] == 1
    direct_receipts = report["execution_receipts"]["direct"]
    assert direct_receipts["finish_reason_distribution"] == {"length": 1}
    assert direct_receipts["terminal_output_parsing_failure_count"] == 1
    assert report["operational_failure_count"] == 0


def test_graph_task_timeout_must_be_positive_when_configured():
    config = _evaluation_config("alfworld")
    config["alfworld_evaluation"]["task_timeout_seconds"] = 300
    _MODULE.validate_completion_benchmark_config(config)

    invalid = deepcopy(config)
    invalid["alfworld_evaluation"]["task_timeout_seconds"] = 0
    try:
        _MODULE.validate_completion_benchmark_config(invalid)
    except Exception as exc:
        assert "task_timeout_seconds" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("a non-positive graph task timeout was accepted")


def test_webshop_round04_production_configs_use_the_frozen_live_catalog():
    for name, split, count in (
        ("development_webshop_round_04.yaml", "train", 16),
        ("evaluation_webshop_round_04.yaml", "validation", 128),
    ):
        config = load_yaml(_ROOT / "config" / name)
        _MODULE.validate_completion_benchmark_config(config)
        section = config["webshop_evaluation"]
        assert section["split"] == split
        assert section["sample_count"] == count
        assert config["data"]["catalog_path"] == "config/datasets_webshop.yaml"
        assert config["data"]["validation_path"] == "data/webshop_v2/validation.jsonl"
        assert config["experiment"]["training_enabled"] is False
        assert config["skills"]["enabled"] is False


def test_aime_selection_filters_the_official_2026_slice(tmp_path):
    config = _evaluation_config("aime_2026")
    config["aime2026_evaluation"]["sample_count"] = 2
    source = tmp_path / "test.jsonl"
    selected = tmp_path / "selected.jsonl"
    config["data"]["test_path"] = str(source)

    def task(task_id: str, benchmark_slice: str, answer: str):
        return _MODULE.TaskRecord(
            task_id=task_id,
            question=f"question {task_id}",
            ground_truth=answer,
            split="test",
            metadata={
                "dataset_key": "aime_2026",
                "benchmark_slice": benchmark_slice,
                "evaluator_payload": {"accepted_answers": [answer]},
            },
        )

    records = (
        task("aime-historical:one", "historical_aime_1983_2025", "1"),
        task("aime-2026/01", "official_aime_2026", "2"),
        task("aime-2026/02", "official_aime_2026", "3"),
    )
    _MODULE._atomic_jsonl(
        source,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **item.to_dict()}
            for item in records
        ],
    )

    frozen = _MODULE._select_tasks(config, tmp_path, selected)

    assert [item.task_id for item in frozen] == ["aime-2026/01", "aime-2026/02"]
    assert [item.task_id for item in _MODULE.iter_task_records(selected)] == [
        "aime-2026/01",
        "aime-2026/02",
    ]


def test_selection_fails_closed_on_declared_partition_mismatch(tmp_path):
    config = _evaluation_config("hotpotqa")
    section = config["hotpotqa_evaluation"]
    section.update(
        {
            "stage": "development",
            "required_partition": "development",
            "sample_count": 2,
            "stable_zero_sample_count": 2,
        }
    )
    source = tmp_path / "validation.jsonl"
    selected = tmp_path / "selected.jsonl"
    config["data"]["validation_path"] = str(source)
    records = [
        _MODULE.TaskRecord(
            task_id=f"hotpotqa:{index}",
            question=f"question {index}",
            ground_truth=str(index),
            split="validation",
            metadata={
                "dataset_key": "hotpotqa",
                "joint_qa_partition": partition,
            },
        )
        for index, partition in enumerate(("development", "test"))
    ]
    _MODULE._atomic_jsonl(
        source,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **item.to_dict()}
            for item in records
        ],
    )

    try:
        _MODULE._select_tasks(config, tmp_path, selected)
    except Exception as exc:
        assert "belongs to partition 'test', expected 'development'" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("cross-partition task selection was accepted")


def test_final_evaluation_stage_requires_test_split():
    config = _evaluation_config("hotpotqa")
    config["hotpotqa_evaluation"]["stage"] = "final_evaluation"

    try:
        _MODULE.validate_completion_benchmark_config(config)
    except Exception as exc:
        assert "evaluation.final_split" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("final evaluation accepted a validation split")


def test_healthbench_evaluation_receives_the_backend_judge():
    calls = []

    async def judge(messages, model):
        calls.append((messages, model))
        return {"criteria_met": True, "explanation": "criterion met"}

    backend = type("Backend", (), {})()
    backend.judge = judge
    backend.judge_model = "configured-healthbench-judge"
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:one",
        question="Conversation:\n\n[user] What should I do?\n\n[assistant]",
        ground_truth="Seek medical advice.",
        split="validation",
        metadata={
            "dataset_key": "healthbench_professional",
            "evaluator_payload": {
                "rubric_items": [
                    {"criterion_text": "Recommends medical advice", "points": 5}
                ]
            },
        },
    )

    outcome = asyncio.run(
        _MODULE._evaluate_prediction(backend, task, "Seek medical advice.")
    )

    assert outcome.valid is True
    assert outcome.metrics["raw_score"] == 1.0
    assert len(calls) == 1
    assert calls[0][1] == "configured-healthbench-judge"


def test_evaluator_preflight_static_fixtures_are_synthetic_and_non_test():
    for dataset_key in (
        "hotpotqa",
        "triviaqa",
        "aime_2026",
        "healthbench_professional",
        "swe_bench",
    ):
        task, _prediction = _MODULE._synthetic_evaluator_preflight_fixture(
            dataset_key
        )

        assert task.task_id == f"evaluator-preflight:{dataset_key}:synthetic-v1"
        assert task.split == "train"
        assert task.metadata["dataset_key"] == dataset_key


def test_evaluator_preflight_receipt_drops_answer_bearing_outcome_fields():
    outcome = EvaluationOutcome(
        valid=True,
        reward=1.0,
        metrics={"exact_match": 1.0, "token_f1": 1.0},
        reason="evaluated",
        details={
            "raw_prediction": "fixture answer",
            "scored_prediction": "fixture answer",
        },
        evaluator_version="hotpotqa.official.answer.v1",
    )

    receipt = _MODULE._evaluator_preflight_receipt(outcome, "hotpotqa")

    assert receipt == {
        "passed": True,
        "evaluator_version": "hotpotqa.official.answer.v1",
    }
    assert "metrics" not in receipt
    assert "details" not in receipt
    assert "reward" not in receipt


def test_healthbench_preflight_failure_exposes_grader_receipt_without_case_data():
    class FailedGrader:
        async def preflight(self):
            return {
                "termination": "grader_error",
                "grader_error": {
                    "error_type": "InternalServerError",
                    "message": "provider returned bad_response_body",
                },
                "provider_errors": [
                    {"error_type": "InternalServerError", "status_code": 500}
                ],
            }

    backend = SimpleNamespace(healthbench_professional_grader=FailedGrader())

    try:
        asyncio.run(
            _MODULE._run_evaluator_preflight(
                backend,
                _evaluation_config("healthbench_professional"),
                _ROOT,
                "healthbench_professional",
            )
        )
    except _MODULE.EvaluatorPreflightError as error:
        message = str(error)
        receipt = error.preflight_receipt
    else:
        raise AssertionError("HealthBench preflight failure was not raised")

    assert "InternalServerError" in message
    assert "bad_response_body" in message
    assert "status_code': 500" in message
    assert "rubric" not in message
    assert receipt == {
        "schema_version": "flowsteer.evaluator_preflight.v1",
        "passed": False,
        "dataset_key": "healthbench_professional",
        "fixture": "synthetic_non_benchmark",
        "termination": "grader_error",
        "evaluator_version": None,
        "grader_model": None,
        "grader_reasoning_effort": None,
        "grader_api_calls": None,
        "grader_latency_ms": None,
        "grader_error": {
            "error_type": "InternalServerError",
            "message": "provider returned bad_response_body",
        },
        "provider_errors": [
            {"error_type": "InternalServerError", "status_code": 500}
        ],
    }


def test_environment_preflight_uses_fixed_training_record(tmp_path):
    config = _evaluation_config("webshop")
    train_path = tmp_path / "train.jsonl"
    config["data"]["train_path"] = str(train_path)
    task = _MODULE.TaskRecord(
        task_id="webshop:train:fixture",
        question="fixed non-test environment fixture",
        ground_truth="environment_success",
        split="train",
        metadata={"dataset_key": "webshop"},
    )
    _MODULE._atomic_jsonl(
        train_path,
        [{"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}],
    )

    fixture = _MODULE._non_test_environment_preflight_task(
        config, tmp_path, "webshop"
    )

    assert fixture.task_id == "webshop:train:fixture"
    assert fixture.split == "train"


def test_swebench_evaluator_preflight_does_not_call_selected_task_harness():
    backend = SimpleNamespace(
        swe_harness=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the selected-task harness must not be used")
        )
    )

    receipt = asyncio.run(
        _MODULE._run_evaluator_preflight(
            backend,
            _evaluation_config("hotpotqa"),
            _ROOT,
            "swe_bench",
        )
    )

    assert receipt == {
        "passed": True,
        "evaluator_version": "swebench.harness.v1",
    }


def test_stable_zero_requires_dataset_evaluator_receipts():
    task = _MODULE.TaskRecord(
        task_id="aime-2026/01",
        question="question",
        ground_truth="7",
        split="test",
        metadata={"dataset_key": "aime_2026"},
    )
    direct = {
        task.task_id: {
            "evaluation": {
                "valid": True,
                "evaluator_version": "wrong-evaluator",
            }
        }
    }
    trajectory = {
        "explicit_finish": True,
        "final_answer": "<answer>7</answer>",
        "evaluation": {
            "valid": True,
            "evaluator_version": _MODULE.evaluator_version_for(task),
        },
        "turns": [
            {
                "receipt_verified": True,
                "director_attempt_count": 1,
                "director_generation_seed": 1,
                "director_latency_ms": 1.0,
                "action": {"action_type": "finish"},
                "graph_snapshot": {"output_agent_id": "actor"},
                "executions": [
                    {
                        "agent_id": "actor",
                        "execution_id": "execution-1",
                        "metadata": {
                            "request": {
                                "agent": {"agent_id": "actor"},
                                "model": {"model_id": "m"},
                                "phase": "single",
                                "upstream": [],
                                "own_draft": None,
                                "peer_draft": None,
                                "rendered_messages": [],
                            }
                        },
                    }
                ],
            }
        ],
    }

    result = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="aime_2026",
    )

    assert result["passed"] is False
    assert result["checks"][0]["direct_evaluator_valid"] is False


def test_environment_stable_zero_uses_terminal_receipt_not_free_text_answer():
    task = _MODULE.TaskRecord(
        task_id="webshop:00500",
        question="buy the requested item",
        ground_truth="environment_terminal_success",
        split="validation",
        metadata={"dataset_key": "webshop"},
    )
    evaluator_version = _MODULE.evaluator_version_for(task)
    direct = {
        task.task_id: {
            "evaluation": {
                "valid": True,
                "evaluator_version": evaluator_version,
            }
        }
    }
    trajectory = {
        "explicit_finish": True,
        "final_answer": "",
        "evaluation": {
            "valid": True,
            "evaluator_version": evaluator_version,
        },
        "turns": [
            {
                "receipt_verified": True,
                "director_attempt_count": 1,
                "director_generation_seed": 1,
                "director_latency_ms": 1.0,
                "action": {"action_type": "finish"},
                "graph_snapshot": {"output_agent_id": "actor"},
                "executions": [
                    {
                        "agent_id": "actor",
                        "execution_id": "execution-1",
                        "metadata": {
                            "request": {
                                "agent": {"agent_id": "actor"},
                                "model": {"model_id": "m"},
                                "phase": "single",
                                "upstream": [],
                                "own_draft": None,
                                "peer_draft": None,
                                "rendered_messages": [],
                            }
                        },
                    }
                ],
                "runtime_summary": {
                    "output_metadata": {
                        "actor": {
                            "environment_terminal": True,
                            "evaluator_environment_trace": [
                                {
                                    "action": "click[buy now]",
                                    "done": True,
                                    "state_advanced": True,
                                }
                            ],
                        }
                    }
                },
            }
        ],
    }

    passed = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="webshop",
    )
    assert passed["passed"] is True
    assert passed["checks"][0]["terminal_artifact_saved"] is True
    assert passed["checks"][0]["environment_terminal_receipt_valid"] is True

    trajectory["turns"][0]["runtime_summary"]["output_metadata"]["actor"][
        "environment_terminal"
    ] = False
    failed = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="webshop",
    )
    assert failed["passed"] is False
    assert failed["checks"][0]["environment_terminal_receipt_valid"] is False


def test_reports_aime_accuracy_and_healthbench_professional_scores():
    aime_rows = [
        {
            "direct": {"available": True, "valid": True, "accuracy": 0.0},
            "agentgraph": {
                "available": True,
                "valid": True,
                "accuracy": 1.0,
                "explicit_finish": True,
            },
        },
        {
            "direct": {"available": True, "valid": True, "accuracy": 1.0},
            "agentgraph": {
                "available": False,
                "valid": False,
                "accuracy": 0.0,
                "explicit_finish": False,
            },
        },
    ]
    aime = _MODULE._aggregate(aime_rows, "agentgraph", "aime_2026")
    assert aime["denominator"] == 2
    assert aime["strict_accuracy"] == 0.5
    assert aime["completed_only_accuracy"] == 1.0
    assert aime["correct"] == 1

    health_rows = [
        {
            "direct": {
                "available": True,
                "valid": True,
                "overall_score": 0.25,
                "overall_score_length_adjusted": 0.20,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "overall_score": 0.75,
                "overall_score_length_adjusted": 0.70,
                "explicit_finish": True,
            },
        },
        {
            "direct": {
                "available": True,
                "valid": True,
                "overall_score": 0.50,
                "overall_score_length_adjusted": 0.45,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "overall_score": -0.25,
                "overall_score_length_adjusted": -0.30,
                "explicit_finish": True,
            },
        },
    ]
    health = _MODULE._aggregate(
        health_rows, "agentgraph", "healthbench_professional"
    )
    assert health["denominator"] == 2
    assert health["strict_overall_score"] == 0.25
    assert health["completed_only_overall_score"] == 0.25
    assert abs(health["strict_overall_score_length_adjusted"] - 0.20) < 1e-12
    assert (
        abs(health["completed_only_overall_score_length_adjusted"] - 0.20)
        < 1e-12
    )

    environment = _MODULE._aggregate(
        [
            {
                "direct": {"available": True, "valid": True, "success": 0.0},
                "agentgraph": {
                    "available": True,
                    "valid": True,
                    "success": 1.0,
                    "explicit_finish": True,
                },
            }
        ],
        "agentgraph",
        "webshop",
    )
    assert environment["strict_success"] == 1.0


def test_aime_wrong_demo_uses_first_runtime_failure_receipt():
    diagnosis = _MODULE._aime_wrong_demo_diagnosis(
        {
            "turns": [
                {
                    "round_index": 0,
                    "action": {
                        "action": "set_relation",
                        "source_id": "a",
                        "target_id": "b",
                    },
                    "canvas_feedback": "accepted set_relation",
                    "runtime_summary": {
                        "execution_status": "failed",
                        "failure_records": [
                            {
                                "agent_id": "b",
                                "error_type": "ProviderRequestError",
                                "phase": "single",
                            }
                        ],
                    },
                },
                {
                    "round_index": 1,
                    "action": {"action": "modify_agent", "agent_id": "b"},
                    "canvas_feedback": "accepted modify_agent",
                    "runtime_summary": {},
                },
            ],
            "evaluation": {"reason": "not_evaluated_without_explicit_finish"},
            "explicit_finish": False,
            "termination_reason": "max_rounds",
            "final_answer": None,
        }
    )

    assert diagnosis["diagnosis_scope"] == "first_observable_failure"
    assert diagnosis["failure_layer"] == "runtime"
    assert diagnosis["first_error_turn"] == 0
    assert diagnosis["first_error_action"] == "set_relation"
    assert diagnosis["first_error_agent_id"] == "b"
    assert diagnosis["subsequent_error_propagation"]["later_turn_count"] == 1
    assert (
        diagnosis["subsequent_error_propagation"]["interpretation"]
        == "subsequent_receipt_span_not_proven_causality"
    )


def test_healthbench_wrong_demo_reports_private_rubric_counts_not_text():
    diagnosis = _MODULE._healthbench_wrong_demo_diagnosis(
        {
            "turns": [],
            "explicit_finish": True,
            "termination_reason": "finish",
            "final_answer": "A complete assistant response.",
            "evaluation": {
                "valid": True,
                "reason": "evaluated",
                "metrics": {
                    "overall_score": 0.5,
                    "overall_score_length_adjusted": 0.48,
                },
                "details": {
                    "rubric_grades": [
                        {
                            "criterion": "private criterion must not be copied",
                            "points": 1.0,
                            "criteria_met": False,
                        },
                        {
                            "criterion": "private negative criterion",
                            "points": -1.0,
                            "criteria_met": True,
                        },
                    ]
                },
            },
        }
    )

    assert diagnosis["failure_layer"] == "rubric_evaluation"
    assert diagnosis["rubric_receipt_summary"] == {
        "rubric_count": 2,
        "unmet_positive_rubric_count": 1,
        "triggered_negative_rubric_count": 1,
    }
    assert "criterion" not in repr(diagnosis)


def test_healthbench_grader_telemetry_totals_are_separate_from_candidate_calls():
    rows = [
        {
            "direct": {
                "evaluation": {
                    "valid": True,
                    "details": {
                        "grader_telemetry": {
                            "api_calls": 3,
                            "latency_ms": 12.5,
                            "provider_errors": [{"error_type": "Timeout"}],
                            "token_usage": {
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                            },
                        }
                    },
                }
            }
        },
        {
            "direct": {
                "evaluation": {
                    "valid": False,
                    "details": {
                        "grader_telemetry": {
                            "api_calls": 1,
                            "latency_ms": 7.5,
                            "provider_errors": [],
                            "token_usage": {"input_tokens": 10},
                        }
                    },
                }
            }
        },
    ]
    totals = _MODULE._healthbench_grader_telemetry_totals(rows, "direct")
    assert totals["api_calls"] == 4
    assert totals["latency_ms"] == 20.0
    assert totals["provider_error_count"] == 1
    assert totals["invalid_grade_count"] == 1
    assert totals["token_usage"]["input_tokens"] == 110
    assert totals["token_usage"]["total_tokens"] == 120


def test_qa_reports_native_exact_match_and_token_f1_together():
    rows = [
        {
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 0.0,
                "token_f1": 0.5,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
                "explicit_finish": True,
            },
        },
        {
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
            },
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "explicit_finish": False,
            },
        },
    ]

    result = _MODULE._aggregate(rows, "agentgraph", "hotpotqa")

    assert result["strict_exact_match"] == 0.5
    assert result["strict_token_f1"] == 0.5
    assert result["completed_only_exact_match"] == 1.0
    assert result["completed_only_token_f1"] == 1.0


def test_qa_metric_receipt_requires_both_native_metrics():
    valid, values = _MODULE._metrics(
        {
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 0.75},
            }
        },
        "triviaqa",
    )
    missing_f1, missing_values = _MODULE._metrics(
        {"evaluation": {"valid": True, "metrics": {"exact_match": 1.0}}},
        "triviaqa",
    )

    assert valid is True
    assert values == {"exact_match": 1.0, "token_f1": 0.75}
    assert missing_f1 is False
    assert missing_values == {"exact_match": 0.0, "token_f1": 0.0}


def test_qa_paired_rows_preserve_exact_match_and_token_f1():
    task = _MODULE.TaskRecord(
        task_id="triviaqa:one",
        question="Who wrote Main Street?",
        ground_truth="Sinclair Lewis",
        split="validation",
        metadata={"dataset_key": "triviaqa"},
    )
    direct = {
        task.task_id: {
            "final_answer": "John Galsworthy",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 0.0, "token_f1": 0.0},
            },
        }
    }
    trajectories = {
        task.task_id: {
            "final_answer": "Sinclair Lewis",
            "explicit_finish": True,
            "termination_reason": "explicit_finish",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 1.0},
            },
            "turns": [],
        }
    }

    row = _MODULE._paired_rows(
        (task,), direct, trajectories, "triviaqa"
    )[0]

    assert row["direct"]["exact_match"] == 0.0
    assert row["direct"]["token_f1"] == 0.0
    assert row["agentgraph"]["exact_match"] == 1.0
    assert row["agentgraph"]["token_f1"] == 1.0
    assert row["delta_exact_match"] == 1.0
    assert row["delta_token_f1"] == 1.0


def test_trivia_failure_taxonomy_uses_evaluator_and_public_runtime_receipts():
    common = {
        "explicit_finish": True,
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 0.0, "token_f1": 0.6},
            "details": {},
        },
        "turns": [],
    }
    canonicalization = {
        **common,
        "evaluation": {
            **common["evaluation"],
            "details": {
                "answer_mismatch_type": (
                    "accepted_answer_canonicalization_mismatch"
                )
            },
        },
    }
    coverage = {
        **common,
        "turns": [
            {
                "canvas_feedback": "knowledge_base_coverage_failure",
                "runtime_summary": {},
            }
        ],
    }
    strategy_failure = {
        **common,
        "explicit_finish": False,
        "turns": [
            {
                "canvas_feedback": "bounded ReAct execution failed",
                "runtime_summary": {
                    "react_public_error_summary": {
                        "terminal_failure_diagnosis": {
                            "public_error_code": "retrieval_strategy_failure"
                        }
                    }
                },
            }
        ],
    }

    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        canonicalization,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.6,
        dataset_key="triviaqa",
    ) == "accepted_answer_canonicalization_mismatch"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        canonicalization,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "accepted_answer_canonicalization_mismatch"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        coverage,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.6,
        dataset_key="triviaqa",
    ) == "knowledge_base_coverage_failure"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        strategy_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "retrieval_strategy_failure"

    partial_overlap_after_recovery = {
        **common,
        "evaluation": {
            **common["evaluation"],
            "details": {"answer_mismatch_type": "partial_answer_overlap"},
        },
        "turns": [
            {
                "canvas_feedback": "semantic_evidence_provenance_invalid",
                "runtime_summary": {},
            }
        ],
    }
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        partial_overlap_after_recovery,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.5,
        dataset_key="triviaqa",
    ) == "partial_answer_overlap"

    coverage_without_finish = {
        **coverage,
        "explicit_finish": False,
        "termination_reason": "max_rounds",
    }
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        coverage_without_finish,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "knowledge_base_coverage_failure"


def test_trivia_failure_taxonomy_prioritizes_final_legal_lineage():
    direct = {"evaluation": {"valid": True}}
    no_overlap_evaluation = {
        "valid": True,
        "metrics": {"exact_match": 0.0, "token_f1": 0.0},
        "details": {"answer_mismatch_type": "no_accepted_answer_overlap"},
    }
    recall_failure_turn = {
        "canvas_feedback": (
            "accepted modify_agent at revision 4; "
            "execution_error=bounded ReAct execution failed"
        ),
        "runtime_summary": {
            "execution_status": "failed",
            "failure_records": [
                {
                    "request_id": "trivia:recall:reader:single",
                    "agent_id": "reader",
                    "phase": "single",
                    "graph_revision": 4,
                    "error_type": "ReactExecutionError",
                    "message": "bounded ReAct execution failed",
                    "metadata": {
                        "react_trace": [
                            {
                                "turn": 10,
                                "terminal_failure_diagnosis": {
                                    "observation_status": "budget_exhausted",
                                    "public_error_code": (
                                        "retrieval_recall_failure"
                                    ),
                                    "tool_plan_exhausted": True,
                                    "bounded_schedule_exhausted": True,
                                },
                            }
                        ],
                        "tool_plan_exhausted": True,
                    },
                }
            ],
        },
    }

    unresolved_recall = {
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "evaluation": no_overlap_evaluation,
        "turns": [recall_failure_turn],
    }
    assert _MODULE._failure_type(
        direct,
        unresolved_recall,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "retrieval_recall_failure"

    final_reasoning_failure = {
        "explicit_finish": True,
        "termination_reason": "finish",
        "evaluation": no_overlap_evaluation,
        "turns": [],
    }
    assert _MODULE._failure_type(
        direct,
        final_reasoning_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "reasoning_failure"

    recovered_then_reasoning_failure = {
        **final_reasoning_failure,
        # The exact failed Runtime receipt remains losslessly available after
        # a later Canvas revision repairs retrieval and explicitly finishes.
        "turns": [recall_failure_turn],
    }
    assert _MODULE._failure_type(
        direct,
        recovered_then_reasoning_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "reasoning_failure"


def test_trivia_failure_taxonomy_prefers_latest_reasoner_binding_failure():
    direct = {"evaluation": {"valid": True}}
    graph = {
        "explicit_finish": False,
        "termination_reason": "canvas_action_domain_exhausted",
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 0.0, "token_f1": 0.0},
            "details": {"answer_mismatch_type": "no_accepted_answer_overlap"},
        },
        "turns": [
            {
                "canvas_feedback": (
                    "qa_semantic_evidence_provenance_invalid: early rejected "
                    "completion"
                ),
                "runtime_summary": {"execution_status": "failed"},
            },
            {
                "canvas_feedback": "accepted set_relation at revision 19",
                "runtime_summary": {
                    "execution_status": "failed",
                    "failure_records": [
                        {
                            "agent_id": "reasoner",
                            "error_type": "ReactExecutionError",
                            "metadata": {
                                "react_trace": [
                                    {
                                        "turn": 10,
                                        "observation_status": "schema_invalid",
                                        "public_error_code": (
                                            "qa_semantic_artifact_invalid: "
                                            "Reasoner answer_slot.answer_field "
                                            "selects 'subject'"
                                        ),
                                    },
                                    {
                                        "turn": 20,
                                        "observation_status": "schema_invalid",
                                        "public_error_code": (
                                            "qa_semantic_artifact_invalid: "
                                            "qa_location_containment_lineage_missing"
                                        ),
                                    },
                                ]
                            },
                        }
                    ],
                    "terminal_canvas_diagnosis": {
                        "public_error_code": "canvas_action_domain_exhausted",
                        "finish_admissibility": {
                            "failure_attribution": {
                                "responsible_agent_id": "reasoner",
                                "responsible_role_family": "reasoner",
                                "responsible_constraint": (
                                    "execution_contract_or_runtime_failure"
                                ),
                            }
                        },
                    },
                },
            },
        ],
    }

    assert _MODULE._failure_type(
        direct,
        graph,
        direct_valid=True,
        graph_valid=True,
        direct_score=1.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "relation_or_answer_slot_binding_failure"


def test_trivia_failure_taxonomy_uses_terminal_responsible_agent_record():
    direct = {"evaluation": {"valid": True}}
    graph = {
        "explicit_finish": False,
        "termination_reason": "canvas_action_domain_exhausted",
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 0.0, "token_f1": 0.0},
            "details": {"answer_mismatch_type": "no_accepted_answer_overlap"},
        },
        "turns": [
            {
                "runtime_summary": {
                    "execution_status": "failed",
                    "failure_records": [
                        {
                            "agent_id": "reasoner",
                            "error_type": "ReactExecutionError",
                            "metadata": {
                                "react_trace": [
                                    {
                                        "public_error_code": (
                                            "qa_semantic_artifact_invalid: "
                                            "answer_slot relation binding failed"
                                        )
                                    }
                                ]
                            },
                        },
                        {
                            "agent_id": "sibling",
                            "error_type": "CancelledError",
                            "metadata": {
                                "react_trace": [
                                    {
                                        "public_error_code": (
                                            "retrieval_recall_failure"
                                        )
                                    }
                                ]
                            },
                        },
                    ],
                    "terminal_canvas_diagnosis": {
                        "finish_admissibility": {
                            "failure_attribution": {
                                "responsible_agent_id": "reasoner"
                            }
                        }
                    },
                }
            }
        ],
    }

    assert _MODULE._failure_type(
        direct,
        graph,
        direct_valid=True,
        graph_valid=True,
        direct_score=1.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "relation_or_answer_slot_binding_failure"


def test_paired_row_uses_the_evaluated_lineage_graph_for_fallback():
    task = _MODULE.TaskRecord(
        task_id="triviaqa:fallback",
        question="Who wrote Main Street?",
        ground_truth="Sinclair Lewis",
        split="validation",
        metadata={"dataset_key": "triviaqa"},
    )
    evaluated_graph = {
        "revision": 1,
        "nodes": [],
        "relations": [],
        "output_agent_id": None,
    }
    terminal_graph = {
        "revision": 2,
        "nodes": [],
        "relations": [],
        "output_agent_id": None,
    }
    direct = {
        task.task_id: {
            "final_answer": "Sinclair Lewis",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 1.0},
            },
        }
    }
    trajectory = {
        "task": task.to_dict(),
        "trajectory_id": "trajectory:fallback",
        "final_answer": "Sinclair Lewis",
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "valid_lineage_fallback_used": True,
        "valid_lineage_fallback_receipt": {
            "graph_snapshot": evaluated_graph,
        },
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 1.0, "token_f1": 1.0},
        },
        "turns": [
            {
                "action": {"action": "modify_agent"},
                "canvas_feedback": "accepted modify_agent at revision 2",
                "graph_snapshot": terminal_graph,
            }
        ],
    }

    row = _MODULE._paired_rows(
        (task,), direct, {task.task_id: trajectory}, "triviaqa"
    )[0]

    assert row["agentgraph"]["final_graph"] == evaluated_graph
    assert row["agentgraph"]["terminal_canvas_graph"] == terminal_graph
    assert row["agentgraph"]["graph_diagnostic"]["graph_revision"] == 1


def test_aime_direct_model_request_contains_problem_but_not_private_target():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )
    sentinel = "PRIVATE_TARGET_SENTINEL_7391"

    class Gateway:
        def __init__(self):
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            return SimpleNamespace(text="<answer>7</answer>")

    gateway = Gateway()
    backend = SimpleNamespace(
        registry=registry,
        runtime=SimpleNamespace(gateway=gateway),
        config={},
    )
    task = _MODULE.TaskRecord(
        task_id="aime-2026/01",
        question="PUBLIC PROBLEM TEXT",
        ground_truth=sentinel,
        split="test",
        metadata={
            "dataset_key": "aime_2026",
            "evaluator_payload": {"accepted_answers": [sentinel]},
        },
    )

    async def fake_evaluate(_backend, evaluated_task, prediction, *, run_graph=None):
        assert evaluated_task.ground_truth == sentinel
        assert prediction == "<answer>7</answer>"
        assert run_graph is None
        return EvaluationOutcome(
            valid=True,
            reward=0.0,
            metrics={"accuracy": 0.0},
            reason="evaluated",
            evaluator_version="skillev.private-static.integer.v1",
        )

    execution = SimpleNamespace(
        metadata={"response": {"generation_seed": 17}},
        to_dict=lambda: {
            "input": {"problem": "PUBLIC PROBLEM TEXT"},
            "output": "<answer>7</answer>",
            "metadata": {"response": {"generation_seed": 17}},
        },
    )
    with patch.object(
        _MODULE,
        "execution_record_from_call",
        return_value=execution,
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="skillev_private_static_integer_submission_v1",
                contract="Return one integer.",
                seed=17,
                run_label="aime-private-boundary-test",
            )
        )

    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request.problem.startswith("\nThink step by step and solve the problem.")
    assert "Your task: PUBLIC PROBLEM TEXT" in request.problem
    assert "Public answer format: integer-000-to-999." in request.problem
    assert "<thought>" in request.problem
    assert "<answer>" in request.problem
    assert "# Response format (must be strictly followed)" in request.problem
    assert "<answer>The final answer to the question</answer>" in request.problem
    assert sentinel not in request.problem
    assert sentinel not in request.agent.contract


def test_direct_only_checkpoint_loader_never_schedules_missing_agentgraph_tasks(
    tmp_path,
):
    selected = (
        _MODULE.TaskRecord(
            task_id="aime-2026/01",
            question="one",
            ground_truth="1",
            split="test",
            metadata={"dataset_key": "aime_2026"},
        ),
        _MODULE.TaskRecord(
            task_id="aime-2026/02",
            question="two",
            ground_truth="2",
            split="test",
            metadata={"dataset_key": "aime_2026"},
        ),
    )
    checkpoint = tmp_path / "trajectories.jsonl"
    _MODULE._atomic_jsonl(
        checkpoint,
        [
            {
                "task": {"task_id": "aime-2026/01"},
                "condition_id": "frozen-aime",
                "trajectory_id": "kept",
            },
            {
                "task": {"task_id": "aime-2026/02"},
                "condition_id": "different-condition",
                "trajectory_id": "rejected-condition",
            },
            {
                "task": {"task_id": "aime-2026/99"},
                "condition_id": "frozen-aime",
                "trajectory_id": "rejected-task",
            },
        ],
    )

    existing = _MODULE._existing_trajectory_checkpoint(
        selected,
        checkpoint,
        condition_id="frozen-aime",
    )

    assert set(existing) == {"aime-2026/01"}
    assert existing["aime-2026/01"]["trajectory_id"] == "kept"


def test_interactive_direct_condition_records_every_environment_policy_call():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )

    class Gateway:
        def __init__(self):
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            return SimpleNamespace(text=f"click[action-{self.calls}]")

    gateway = Gateway()
    backend = SimpleNamespace(
        registry=registry,
        runtime=SimpleNamespace(gateway=gateway),
        config={"evaluation": {"max_environment_steps": 10}},
    )
    task = _MODULE.TaskRecord(
        task_id="webshop:00001",
        question="buy an item",
        ground_truth="environment_success",
        split="validation",
        metadata={"dataset_key": "webshop"},
    )
    execution_index = 0

    class Execution:
        def __init__(self, output):
            nonlocal execution_index
            execution_index += 1
            self.output = output
            self.metadata = {"response": {"generation_seed": 17}}

        def to_dict(self):
            return {
                "output": self.output,
                "metadata": self.metadata,
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 1.0,
            }

    async def fake_evaluate(_backend, _task, _prediction, *, run_graph=None):
        assert run_graph is not None
        await run_graph("environment step one")
        await run_graph("environment step two")
        return EvaluationOutcome(
            valid=True,
            reward=1.0,
            metrics={"success": 1.0},
            reason="evaluated",
            evaluator_version="skillflow.ragen_adapter.v1",
        )

    with patch.object(
        _MODULE,
        "execution_record_from_call",
        side_effect=lambda call: Execution(call.response.text),
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        result = asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="skillflow_ragen_webshop_react_v1",
                contract="Return one legal action.",
                seed=17,
                run_label="webshop-test",
            )
        )

    assert len(result["executions"]) == 2
    assert result["final_answer"] == "click[action-2]"
    assert result["evaluation"]["metrics"]["success"] == 1.0


def test_healthbench_direct_react_is_one_agent_with_same_medrag_registry():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )
    config = _healthbench_react_config()
    config["experiment"]["seed"] = 29
    config["healthbench_professional_evaluation"][
        "direct_generation_seed"
    ] = 29
    condition_id = config["experiment"]["condition_id"]
    completion_condition = config["healthbench_professional_evaluation"][
        "direct_completion_condition"
    ]
    observed = {}
    closed = []
    tool_registry = SimpleNamespace(
        resource_ids=("healthbench-medrag.search",),
    )

    class Runtime:
        async def execute(self, graph, problem, *, run_id):
            node = graph.nodes[0]
            observed.update(
                problem=problem,
                run_id=run_id,
                node=node,
                node_count=len(graph.nodes),
                output_agent_id=graph.output_agent_id,
            )
            coordinate = observed["sampling_coordinate"]
            generation_seed = _MODULE.derive_generation_seed(
                base_seed=29,
                coordinate=coordinate,
                step_index=1,
                phase=_MODULE.GenerationPhase.ACTION,
            )
            requested_sampling = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "repetition_penalty": None,
                "max_tokens": int(config["director"]["max_action_tokens"]),
                "seed": generation_seed,
                "chat_template_enable_thinking": False,
            }
            scientific_requested_sampling = {
                key: value
                for key, value in requested_sampling.items()
                if key != "repetition_penalty"
            }
            call = SimpleNamespace(
                response=SimpleNamespace(
                    metadata={
                        "model_calls": [
                            {
                                "turn": 1,
                                "request_id": "healthbench-react:1",
                                "request_status": "completed",
                                "algorithm": (
                                    _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM
                                ),
                                "scientific_sampling": {
                                    "algorithm": (
                                        _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM
                                    ),
                                    "base_seed": 29,
                                    "coordinate": coordinate.to_value(),
                                    "phase": "action",
                                    "step_index": 1,
                                    "generation_seed": generation_seed,
                                    "requested_sampling": (
                                        scientific_requested_sampling
                                    ),
                                },
                                "requested_sampling": requested_sampling,
                                "metadata": {
                                    "generation_seed": generation_seed,
                                    "requested_sampling": requested_sampling,
                                },
                            }
                        ],
                        "react_trace": [
                            {
                                "turn": 1,
                                "action_name": "healthbench-medrag.search",
                                "observation_status": "completed",
                            }
                        ],
                        "tool_receipts": [
                            {
                                "resource_id": "healthbench-medrag.search",
                                "status": "completed",
                            }
                        ],
                    }
                )
            )
            return SimpleNamespace(
                final_answer="This is the complete assistant response.",
                calls=(call,),
                run_id=run_id,
                output_agent_id=node.id,
                block_completion_order=((node.id,),),
                executed_agent_ids=(node.id,),
            )

    def runtime_for_task(
        runtime_task,
        *,
        condition_id,
        sampling_base_seed,
        sampling_coordinate,
    ):
        observed["runtime_task"] = runtime_task
        observed["runtime_condition_id"] = condition_id
        observed["sampling_base_seed"] = sampling_base_seed
        observed["sampling_coordinate"] = sampling_coordinate
        return (
            Runtime(),
            tool_registry,
            lambda: closed.append(True),
        )

    backend = SimpleNamespace(
        registry=registry,
        config=config,
        _runtime_for_task=runtime_for_task,
    )
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:one",
        question=(
            "Conversation:\n\n[user] What should I discuss with my clinician?"
            "\n\n[assistant]"
        ),
        ground_truth="evaluator-private",
        split="validation",
        metadata={"dataset_key": "healthbench_professional"},
    )

    async def fake_evaluate(_backend, evaluated_task, prediction, *, run_graph=None):
        assert evaluated_task is task
        assert prediction == "This is the complete assistant response."
        assert run_graph is None
        return EvaluationOutcome(
            valid=True,
            reward=0.75,
            metrics={"overall_score_length_adjusted": 0.75},
            reason="synthetic test evaluator",
            evaluator_version="healthbench.fake.v1",
        )

    def execution_from_call(call):
        response_metadata = dict(call.response.metadata)
        return SimpleNamespace(
            to_dict=lambda: {
                "output": "This is the complete assistant response.",
                "metadata": {"response": response_metadata},
            }
        )

    with patch.object(
        _MODULE,
        "execution_record_from_call",
        side_effect=execution_from_call,
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        result = asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="single_react_agent_medrag_v1",
                contract="Use retrieval when useful and answer the conversation.",
                seed=29,
                run_label="healthbench-medrag-test",
            )
        )

    node = observed["node"]
    assert observed["node_count"] == 1
    assert node.execution_mode.value == "react"
    assert node.allowed_tools == ("healthbench-medrag.search",)
    assert node.completion_condition == completion_condition
    assert observed["output_agent_id"] == node.id
    assert observed["runtime_task"] is task
    assert observed["runtime_condition_id"] == condition_id
    assert observed["sampling_base_seed"] == 29
    coordinate = observed["sampling_coordinate"]
    assert coordinate.task_id == task.task_id
    assert coordinate.sequence_position == 0
    assert coordinate.schedule_purpose == condition_id
    assert coordinate.ordered_sequence_hash == _MODULE.stable_hash([task.task_id])
    assert task.question in observed["problem"]
    assert task.ground_truth not in observed["problem"]
    assert result["simple_baseline_topology"] == "single_react_agent"
    assert result["runtime_condition_id"] == condition_id
    assert result["tool_version"] == config["experiment"]["tool_version"]
    assert result["tool_resource_ids"] == ["healthbench-medrag.search"]
    assert result["generation_identity_verified"] is True
    assert result["scientific_sampling_receipt"]["verified"] is True
    assert result["direct_generation_identity"]["model"]["catalog_id"] == (
        registry.catalog_id
    )
    assert (
        result["direct_generation_identity"]["model"][
            "chat_template_enable_thinking"
        ]
        is False
    )
    assert result["direct_generation_identity"]["medrag"]["source_revision"] == (
        "fixture-revision"
    )
    assert _MODULE._persisted_healthbench_direct_identity_matches(
        result,
        result["direct_generation_identity"],
    )
    tampered = deepcopy(result)
    tampered["direct_generation_identity"]["contract"] = "changed contract"
    assert not _MODULE._persisted_healthbench_direct_identity_matches(
        tampered,
        result["direct_generation_identity"],
    )
    assert result["runtime"]["output_agent_id"] == node.id
    assert result["execution"]["metadata"]["response"]["react_trace"]
    assert result["execution"]["metadata"]["response"]["tool_receipts"]
    assert closed == [True]


def test_healthbench_retrieval_report_requires_measured_paired_identity():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )
    config = _healthbench_react_config()
    config["healthbench_tool_runtime"]["execution_profile_allowlist"] = [
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-medrag.search"],
        }
    ]
    base_seed = int(config["experiment"]["seed"])
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:paired-identity",
        question="Conversation:\n\n[user] What should I discuss?\n\n[assistant]",
        ground_truth="evaluator-private",
        split="validation",
        metadata={"dataset_key": "healthbench_professional"},
    )
    backend = SimpleNamespace(registry=registry, config=config)
    coordinate = _MODULE._direct_scientific_sampling_coordinate(
        config,
        task,
        base_seed=base_seed,
    )
    identity = _MODULE._healthbench_direct_generation_identity(
        backend,
        task,
        model_id="qwen3.5-9b-local",
        protocol=config["healthbench_professional_evaluation"]["direct_protocol"],
        contract=config["healthbench_professional_evaluation"]["direct_contract"],
        seed=base_seed,
        coordinate=coordinate,
    )
    assert identity["agentgraph_execution_profile_allowlist"] == [
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-medrag.search"],
        }
    ]
    generation_seed = _MODULE.derive_generation_seed(
        base_seed=base_seed,
        coordinate=coordinate,
        step_index=1,
        phase=_MODULE.GenerationPhase.ACTION,
    )
    action_sampling = identity["scientific_sampling"]["requested_sampling"]
    requested_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": action_sampling["top_k"],
        "repetition_penalty": action_sampling["repetition_penalty"],
        "max_tokens": action_sampling["max_tokens"],
        "seed": generation_seed,
        "chat_template_enable_thinking": action_sampling[
            "chat_template_enable_thinking"
        ],
    }
    scientific_requested_sampling = {
        key: value
        for key, value in requested_sampling.items()
        if key != "repetition_penalty"
    }
    model_calls = [
        {
            "turn": 1,
            "request_id": "paired-identity:react:1",
            "request_status": "completed",
            "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
            "scientific_sampling": {
                "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
                "base_seed": base_seed,
                "coordinate": coordinate.to_value(),
                "phase": "action",
                "step_index": 1,
                "generation_seed": generation_seed,
                "requested_sampling": scientific_requested_sampling,
            },
            "requested_sampling": requested_sampling,
            "metadata": {
                "generation_seed": generation_seed,
                "requested_sampling": requested_sampling,
            },
        }
    ]
    sampling_receipt = _MODULE._react_scientific_sampling_receipt(
        model_calls,
        base_seed=base_seed,
        coordinate=coordinate,
        max_action_tokens=int(action_sampling["max_tokens"]),
        expected_top_k=action_sampling["top_k"],
        expected_repetition_penalty=action_sampling["repetition_penalty"],
        expected_chat_template_enable_thinking=action_sampling[
            "chat_template_enable_thinking"
        ],
    )
    thinking_model_calls = deepcopy(model_calls)
    for field in ("requested_sampling",):
        thinking_model_calls[0][field].update(
            max_tokens=int(action_sampling["max_tokens"]) + 512,
            visible_max_tokens=int(action_sampling["max_tokens"]),
            thinking_budget=512,
        )
    thinking_model_calls[0]["metadata"]["requested_sampling"].update(
        max_tokens=int(action_sampling["max_tokens"]) + 512,
        visible_max_tokens=int(action_sampling["max_tokens"]),
        thinking_budget=512,
    )
    thinking_receipt = _MODULE._react_scientific_sampling_receipt(
        thinking_model_calls,
        base_seed=base_seed,
        coordinate=coordinate,
        max_action_tokens=int(action_sampling["max_tokens"]),
        expected_top_k=action_sampling["top_k"],
        expected_repetition_penalty=action_sampling["repetition_penalty"],
        expected_chat_template_enable_thinking=action_sampling[
            "chat_template_enable_thinking"
        ],
        expected_thinking_budget=512,
    )
    assert thinking_receipt["verified"] is True
    execution = {
        "metadata": {"response": {"model_calls": model_calls}},
    }
    row = {
        "task_id": task.task_id,
        "failure_type": "none",
        "direct": {
            "available": True,
            "valid": True,
            "overall_score": 0.5,
            "overall_score_length_adjusted": 0.5,
            "evaluation": {"valid": True, "details": {}},
            "telemetry": {},
            "execution": execution,
            "direct_generation_identity": identity,
            "generation_identity_verified": True,
            "scientific_sampling_receipt": sampling_receipt,
        },
        "agentgraph": {
            "available": True,
            "valid": True,
            "overall_score": 0.6,
            "overall_score_length_adjusted": 0.6,
            "evaluation": {"valid": True, "details": {}},
            "explicit_finish": True,
            "termination_reason": "finish",
            "telemetry": {},
            "graph_diagnostic": None,
        },
    }
    trajectory = {
        "task": task.to_dict(),
        "condition_id": identity["condition_id"],
        "versions": {
            "model_catalog": registry.catalog_id,
            "tool": identity["tool"]["tool_version"],
        },
        "director_sampling": {
            "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": "action",
        },
        "sampling_receipt_verified": True,
        "turns": [
            {
                "round_index": 0,
                "action": {"action": "finish"},
                "canvas_feedback": "accepted finish",
                "graph_snapshot": {
                    "nodes": [],
                    "relations": [],
                    "output_agent_id": None,
                    "revision": 0,
                },
                "executions": [execution],
            }
        ],
        "explicit_finish": True,
        "termination_reason": "finish",
    }

    report = _MODULE._report([row], config, [trajectory])
    assert report["agentgraph_execution_profile_allowlist"] == [
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-medrag.search"],
        }
    ]
    assert report["protocol_equivalent_to_direct"] is True
    assert report["comparison_interpretation"] == "paired_architecture_comparison"
    assert report["paired_generation_identity"]["verified_task_count"] == 1
    assert report["paired_generation_identity"]["checks"][0][
        "agentgraph_executor_react_sampling_status"
    ] == "verified"

    no_react_trajectory = deepcopy(trajectory)
    no_react_trajectory["turns"][0]["executions"] = []
    report = _MODULE._report([row], config, [no_react_trajectory])
    assert report["protocol_equivalent_to_direct"] is False
    assert report["paired_generation_identity"]["checks"][0][
        "agentgraph_executor_react_sampling_status"
    ] == "missing_react_execution_receipt"

    unverified = deepcopy(row)
    unverified["direct"]["generation_identity_verified"] = False
    report = _MODULE._report([unverified], config, [trajectory])
    assert report["protocol_equivalent_to_direct"] is False
    assert (
        report["comparison_interpretation"]
        == "separate_protocol_descriptive_comparison"
    )


def test_swe_direct_condition_is_one_coding_agent_with_repository_tools():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )
    observed = {}
    closed = []

    class Runtime:
        async def execute(self, graph, problem, *, run_id):
            node = graph.nodes[0]
            observed.update(
                problem=problem,
                run_id=run_id,
                node=node,
                output_agent_id=graph.output_agent_id,
            )
            call = SimpleNamespace(
                response=SimpleNamespace(
                    metadata={
                        "model_calls": [
                            {"metadata": {"generation_seed": 23}}
                        ]
                    }
                )
            )
            return SimpleNamespace(
                final_answer="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
                calls=(call,),
                run_id=run_id,
                output_agent_id=node.id,
                block_completion_order=((node.id,),),
                executed_agent_ids=(node.id,),
            )

    backend = SimpleNamespace(
        registry=registry,
        config={
            "experiment": {"condition_id": "swe-coding"},
            "swebench_evaluation": {
                "dataset_key": "swe_bench",
                "direct_completion_condition": (
                    "Inspect, edit, run tests, inspect a changed diff, then complete."
                ),
            },
            "swe_coding_runtime": {"enabled": True},
        },
        _runtime_for_task=lambda task, condition_id: (
            Runtime(),
            SimpleNamespace(resource_ids=("swebench_repository",)),
            lambda: closed.append(True),
        ),
    )
    task = _MODULE.TaskRecord(
        task_id="swe_bench:one",
        question="Fix the public issue description.",
        ground_truth="official_harness_only",
        split="validation",
        metadata={"dataset_key": "swe_bench"},
    )

    async def fake_evaluate(_backend, _task, prediction, *, run_graph=None):
        assert prediction.startswith("diff --git")
        assert run_graph is None
        return EvaluationOutcome(
            valid=True,
            reward=1.0,
            metrics={"resolved": 1.0},
            reason="official harness",
            evaluator_version="swebench.harness.v1",
        )

    with patch.object(
        _MODULE,
        "execution_record_from_call",
        return_value=SimpleNamespace(
            to_dict=lambda: {
                "output": "diff --git a/a.py b/a.py",
                "metadata": {"response": {"attempt_count": 1}},
            }
        ),
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        result = asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="single_coding_agent_v1",
                contract="Use repository tools to resolve the issue.",
                seed=23,
                run_label="test",
            )
        )

    node = observed["node"]
    assert node.execution_mode.value == "coding"
    assert node.allowed_tools == ("swebench_repository",)
    assert observed["output_agent_id"] == "direct_coding_agent"
    assert result["simple_baseline_topology"] == "single_coding_agent"
    assert result["final_answer"].startswith("diff --git")
    assert closed == [True]
