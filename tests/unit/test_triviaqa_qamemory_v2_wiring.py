from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.triviaqa_qa_memory import (
    QA_MEMORY_TOOL_ID,
    TriviaQAQAMemoryManifest,
    load_materialized_qa_memory,
    load_triviaqa_qa_memory_sources,
)


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_triviaqa_qa_memory_unified_v2.yaml"
)
_RUNNER_PATH = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_triviaqa_qamemory_v2_wiring_test",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_v2_config_is_inference_only_and_control_plane_isolated() -> None:
    config = load_yaml(_CONFIG_PATH)

    validate_agent_graph_config(config)
    _RUNNER.validate_completion_benchmark_config(config)

    bounded = config["triviaqa_evaluation"]
    graph = config["agent_graph"]
    runtime = config["qa_tool_runtime"]
    assert bounded["dataset_key"] == "triviaqa"
    assert bounded["split"] == "validation"
    assert bounded["selection"] == "sequential"
    assert bounded["sample_count"] == 128
    assert bounded["evaluator_version"] == "triviaqa.official.answer.v1"
    assert bounded["legacy_deterministic_prefetch_enabled"] is False
    assert graph["required_evidence_tool_id"] == QA_MEMORY_TOOL_ID
    assert graph["director_feedback_mode"] == "control_plane"
    assert graph["require_format_agent"] is False
    assert graph["semantic_protocol_by_source"] == {
        "triviaqa": "qa_verified_answer_lineage_v2"
    }
    assert runtime["mode"] == "model_driven_search_read"
    assert runtime["index_path"] == "data/triviaqa_qa_memory_v1/index"
    assert runtime["max_turns_per_agent_call"] == 7
    assert runtime["max_tool_calls_per_agent_call"] == 4
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_runtime_registry_accepts_same_catalog_in_linked_worktree() -> None:
    config = load_yaml(_CONFIG_PATH)
    manifest = json.loads(
        (_ROOT / "data/joint_qa_v2/manifest.json").read_text()
    )

    assert _RUNNER._same_resolved_path(
        _ROOT,
        manifest["config"],
        config["data"]["catalog_path"],
    )
    receipt = _RUNNER._validate_runtime_dataset_registry(config, _ROOT)
    assert receipt is not None
    assert receipt["registry_dataset_key"] == "triviaqa"
    assert receipt["checks"][
        "manifest_preparation_provenance_matches_registry"
    ] is True


def test_frozen_qa_memory_covers_only_512_train_sources() -> None:
    config = load_yaml(_CONFIG_PATH)
    train_path = _ROOT / config["data"]["train_path"]
    validation_path = _ROOT / config["data"]["validation_path"]
    index_root = _ROOT / config["qa_tool_runtime"]["index_path"]
    manifest = TriviaQAQAMemoryManifest.from_value(
        json.loads((index_root / "manifest.json").read_text())
    )
    sources, validation_ids = load_triviaqa_qa_memory_sources(
        train_path,
        validation_path,
        expected_train_count=512,
        expected_validation_count=128,
    )
    records = load_materialized_qa_memory(
        index_root / "memories.jsonl",
        expected_count=512,
    )

    assert manifest.tool_id == QA_MEMORY_TOOL_ID
    assert manifest.train_count == 512
    assert manifest.memory_count == 512
    assert manifest.paraphrase_count == 512
    assert manifest.validation_isolation_count == 128
    assert manifest.validation_content_indexed is False
    assert manifest.frozen_top_k == 3
    assert dict(manifest.tool_budget) == {
        "max_tool_calls_per_agent_call": 4,
        "max_turns_per_agent_call": 7,
    }
    assert {record.source_train_task_id for record in records} == {
        source.source_train_task_id for source in sources
    }
    assert not {record.base_task_id for record in records}.intersection(
        validation_ids
    )
    assert all(
        record.canonical_answer in record.paraphrase_answer_statement
        for record in records
    )


def test_fixed128_selection_and_reused_direct_predictions_are_identical(
    tmp_path: Path,
) -> None:
    config = load_yaml(_CONFIG_PATH)
    selected = _RUNNER._select_tasks(
        config,
        _ROOT,
        tmp_path / "selected_tasks.jsonl",
    )

    direct_path = _ROOT / config["triviaqa_evaluation"]["direct_reused_from"]
    direct_rows = _jsonl(direct_path)
    assert len(selected) == 128
    assert len(direct_rows) == 128
    assert [task.task_id for task in selected] == [
        row["task_id"] for row in direct_rows
    ]
    assert all(task.split == "validation" for task in selected)
    assert all(
        row["protocol"]
        == config["triviaqa_evaluation"]["direct_protocol"]
        for row in direct_rows
    )
