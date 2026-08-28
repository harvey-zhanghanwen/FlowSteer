from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.triviaqa_qa_memory import QA_MEMORY_TOOL_ID


ROOT = Path(__file__).resolve().parents[2]
ROUND01 = ROOT / "config" / "evaluation_triviaqa_round_01.yaml"
V23 = ROOT / "config" / "evaluation_triviaqa_round01_qa_memory_v23.yaml"
RUNNER_PATH = ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_triviaqa_round01_qa_memory_v23_test",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_v23_freezes_round01_director_and_canvas_profile() -> None:
    baseline = load_yaml(ROUND01)
    config = load_yaml(V23)
    validate_agent_graph_config(config)

    assert config["experiment"]["seed"] == baseline["experiment"]["seed"]
    assert config["experiment"]["catalog_order_namespace"] == baseline["experiment"]["catalog_order_namespace"]
    assert config["experiment"]["prompt_version"] == baseline["experiment"]["prompt_version"]

    director_fields = (
        "base_model",
        "tokenizer_path",
        "backend",
        "api_base",
        "served_model_name",
        "prompt_profile",
        "dtype",
        "max_context_tokens",
        "max_action_tokens",
        "temperature",
        "top_p",
        "top_k",
        "max_rounds",
        "execute_on_edit",
        "history_window",
    )
    for field in director_fields:
        assert config["director"][field] == baseline["director"][field]
    assert config["director"]["action_decoding"] == "json_schema"
    assert config["director"]["action_schema_version"] == "agentgraph.canvas-action-json-schema.v1"
    assert config["director"]["sampling_action_profile"] == "model_admissible_canvas_actions"
    assert config["director"]["sampling_schema_version"] == "agentgraph.model-admissible-action-mask.v3"

    canvas_fields = (
        "max_agents",
        "contract_type",
        "relation_encoding",
        "actions",
        "executor_selection",
        "max_bidirectional_block_size",
        "require_unique_output",
        "require_all_agents_reach_output",
        "terminal_protocol_by_source",
        "model_catalog_path",
    )
    for field in canvas_fields:
        assert config["agent_graph"][field] == baseline["agent_graph"][field]
    assert config["agent_graph"]["actions"] == [
        "add_agent",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    assert "add_subgraph" not in config["agent_graph"]["actions"]
    assert "max_agents_per_subgraph" not in config["agent_graph"]


def test_v23_only_adds_worker_local_qamemory_boundary() -> None:
    config = load_yaml(V23)
    runtime = config["qa_tool_runtime"]
    graph = config["agent_graph"]
    bounded = config["triviaqa_evaluation"]

    assert bounded["sample_count"] == 128
    assert bounded["selection"] == "sequential"
    assert bounded["evaluator_version"] == "triviaqa.official.answer.v1"
    assert bounded["legacy_deterministic_prefetch_enabled"] is False
    assert bounded["direct_reused_from"] == "artifacts/triviaqa_round_01/direct_predictions.jsonl"
    assert runtime["mode"] == "model_driven_search_read"
    assert runtime["index_path"] == "data/triviaqa_qa_memory_v1/index"
    assert runtime["max_turns_per_agent_call"] == 7
    assert runtime["max_tool_calls_per_agent_call"] == 4
    assert runtime["completion_policy"] == "retrieval_first_parametric_fallback"
    assert graph["required_evidence_tool_id"] == QA_MEMORY_TOOL_ID
    assert graph["require_evidence_relation"] is True
    assert graph["require_format_agent"] is False
    assert graph["director_feedback_mode"] == "control_plane"
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def _round01_direct_row() -> dict[str, object]:
    path = ROOT / "artifacts" / "triviaqa_round_01" / "direct_predictions.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_matching_nested_legacy_seed_is_materialized_and_exactly_reusable() -> None:
    config = load_yaml(V23)
    expected_seed = config["triviaqa_evaluation"]["direct_generation_seed"]
    row = _round01_direct_row()
    task = RUNNER.TaskRecord.from_dict(row["task"])

    projected = RUNNER._materialize_reused_direct_generation_seed(
        row,
        expected_seed=expected_seed,
    )

    assert projected["generation_seed"] == expected_seed
    assert RUNNER.hotpot_round._direct_resume_matches(
        projected,
        task=task,
        model_id=config["triviaqa_evaluation"]["direct_model_id"],
        protocol=config["triviaqa_evaluation"]["direct_protocol"],
        seed=expected_seed,
    ) is True


def test_mismatched_nested_legacy_seed_is_not_materialized_or_reusable() -> None:
    config = load_yaml(V23)
    expected_seed = config["triviaqa_evaluation"]["direct_generation_seed"]
    row = _round01_direct_row()
    row["execution"]["metadata"]["response"]["generation_seed"] = expected_seed + 1
    task = RUNNER.TaskRecord.from_dict(row["task"])

    projected = RUNNER._materialize_reused_direct_generation_seed(
        row,
        expected_seed=expected_seed,
    )

    assert "generation_seed" not in projected
    assert RUNNER.hotpot_round._direct_resume_matches(
        projected,
        task=task,
        model_id=config["triviaqa_evaluation"]["direct_model_id"],
        protocol=config["triviaqa_evaluation"]["direct_protocol"],
        seed=expected_seed,
    ) is False


def test_missing_nested_legacy_seed_is_not_materialized_or_reusable() -> None:
    config = load_yaml(V23)
    expected_seed = config["triviaqa_evaluation"]["direct_generation_seed"]
    row = _round01_direct_row()
    del row["execution"]["metadata"]["response"]["generation_seed"]
    task = RUNNER.TaskRecord.from_dict(row["task"])

    projected = RUNNER._materialize_reused_direct_generation_seed(
        row,
        expected_seed=expected_seed,
    )

    assert "generation_seed" not in projected
    assert RUNNER.hotpot_round._direct_resume_matches(
        projected,
        task=task,
        model_id=config["triviaqa_evaluation"]["direct_model_id"],
        protocol=config["triviaqa_evaluation"]["direct_protocol"],
        seed=expected_seed,
    ) is False
