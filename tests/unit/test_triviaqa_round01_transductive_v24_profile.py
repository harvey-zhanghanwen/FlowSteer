from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.qa_tool_adapter import open_qa_tool_registry
from src.interactive.triviaqa_transductive_qa_memory import (
    EVALUATION_REGIME,
    TRANSDUCTIVE_INDEX_FORMAT,
    TriviaQATransductiveQAMemoryIndex,
)


ROOT = Path(__file__).resolve().parents[2]
ROUND01 = ROOT / "config" / "evaluation_triviaqa_round_01.yaml"
V23 = ROOT / "config" / "evaluation_triviaqa_round01_qa_memory_v23.yaml"
V24 = ROOT / "config" / "evaluation_triviaqa_round01_transductive_qa_memory_v24.yaml"
RUNNER_PATH = ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_triviaqa_round01_transductive_v24_test",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_v24_reuses_highest_round01_and_v23_free_canvas_boundaries() -> None:
    baseline = load_yaml(ROUND01)
    v23 = load_yaml(V23)
    config = load_yaml(V24)
    validate_agent_graph_config(config)
    RUNNER.validate_completion_benchmark_config(config)

    assert config["experiment"]["baseline_id"] == "triviaqa_round_01_stable_zero"
    assert config["experiment"]["seed"] == baseline["experiment"]["seed"]
    assert config["experiment"]["prompt_version"] == baseline["experiment"][
        "prompt_version"
    ]
    for field in (
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
        "action_decoding",
        "action_schema_version",
        "sampling_schema_version",
        "sampling_action_profile",
    ):
        assert config["director"][field] == v23["director"][field]
    for field in (
        "max_agents",
        "contract_type",
        "relation_encoding",
        "actions",
        "executor_selection",
        "max_bidirectional_block_size",
        "require_unique_output",
        "require_all_agents_reach_output",
        "require_format_agent",
        "director_feedback_mode",
        "required_evidence_tool_id",
        "require_evidence_relation",
        "terminal_protocol_by_source",
        "model_catalog_path",
    ):
        assert config["agent_graph"][field] == v23["agent_graph"][field]
    assert config["agent_graph"]["actions"] == [
        "add_agent",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    assert "add_subgraph" not in config["agent_graph"]["actions"]
    assert "role" not in config["agent_graph"]


def test_v24_declares_transductive_retrieval_and_worker_first_boundary() -> None:
    config = load_yaml(V24)
    bounded = config["triviaqa_evaluation"]
    runtime = config["qa_tool_runtime"]

    assert bounded["sample_count"] == 128
    assert bounded["evaluation_regime"] == EVALUATION_REGIME
    assert bounded["metric_label"] == "transductive_retrieval_accuracy"
    assert bounded["contains_evaluation_answers"] is True
    assert bounded["official_heldout_eligible"] is False
    assert bounded["legacy_deterministic_prefetch_enabled"] is False
    assert runtime["index_path"] == (
        "data/triviaqa_qa_memory_transductive_v1/index"
    )
    assert runtime["index_format"] == TRANSDUCTIVE_INDEX_FORMAT
    assert runtime["mode"] == "model_driven_search_read"
    assert runtime["completion_policy"] == "retrieval_first_parametric_fallback"
    assert runtime["retrieval_order"] == (
        "search_read_before_parametric_completion"
    )
    assert runtime["parametric_fallback_condition"] == (
        "all_frozen_top_k_read_and_evidence_unsupported"
    )
    assert runtime["static_prefetch_enabled"] is False
    assert runtime["director_tool_calls_allowed"] is False
    assert runtime["web_search_allowed"] is False
    assert config["agent_graph"]["require_evidence_relation"] is True
    assert config["agent_graph"]["director_feedback_mode"] == "control_plane"
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0


def test_open_registry_dispatches_transductive_index_without_new_tool_wire(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "manifest.json").write_text(
        json.dumps(
            {
                "record_kind": "qa_memory",
                "tool_id": "triviaqa.qa_memory",
                "format": TRANSDUCTIVE_INDEX_FORMAT,
                "contains_evaluation_answers": True,
                "evaluation_regime": EVALUATION_REGIME,
                "official_heldout_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    class _Index:
        manifest = SimpleNamespace(
            tool_id="triviaqa.qa_memory",
            corpus_name="triviaqa-all-qa-transductive-memory",
            corpus_version="test-corpus-v1",
            index_id="test-index-v1",
            format=TRANSDUCTIVE_INDEX_FORMAT,
            retrieval_backend="sentence-transformers-bge-normalized-dot-product",
            frozen_top_k=3,
            tool_budget={
                "max_tool_calls_per_agent_call": 4,
                "max_turns_per_agent_call": 7,
            },
        )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        TriviaQATransductiveQAMemoryIndex,
        "open",
        classmethod(lambda cls, root: _Index()),
    )
    opened = open_qa_tool_registry(
        index_path=index_root,
        dataset_scope=("triviaqa",),
    )
    try:
        assert opened.tool_id == "triviaqa.qa_memory"
        assert opened.registry.resource_ids == ("triviaqa.qa_memory",)
        assert opened.retrieval_index_identity["index_format"] == (
            TRANSDUCTIVE_INDEX_FORMAT
        )
        assert opened.frozen_tool_budget == {
            "max_tool_calls_per_agent_call": 4,
            "max_turns_per_agent_call": 7,
        }
    finally:
        opened.close()


def test_runner_summary_preserves_transductive_accuracy_label(tmp_path: Path) -> None:
    config = load_yaml(V24)
    config["qa_tool_runtime"]["index_path"] = "index"
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "manifest.json").write_text(
        json.dumps(
            {
                "train_count": 512,
                "source_counts": {
                    "train": 512,
                    "frozen_development_validation": 128,
                    "total": 640,
                },
                "unique_source_count": 640,
                "cycled_count": 0,
                "paraphrase_count": 640,
                "memory_count": 640,
                "validation_content_indexed": True,
                "contains_evaluation_answers": True,
                "evaluation_regime": EVALUATION_REGIME,
                "official_heldout_eligible": False,
                "evaluation_memory_overlap_count": 128,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_dimension": 768,
                "normalization": "l2",
                "similarity": "dot_product",
                "frozen_top_k": 3,
                "tool_budget": {
                    "max_tool_calls_per_agent_call": 4,
                    "max_turns_per_agent_call": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = RUNNER._triviaqa_qa_memory_index_summary(config, tmp_path)

    assert summary is not None
    assert summary["train_record_count"] == 512
    assert summary["evaluation_record_count"] == 128
    assert summary["total_record_count"] == 640
    assert summary["evaluation_memory_overlap_count"] == 128
    assert summary["heldout_validation_count"] == 0
    assert summary["contains_evaluation_answers"] is True
    assert summary["evaluation_regime"] == EVALUATION_REGIME
    assert summary["official_heldout_eligible"] is False
