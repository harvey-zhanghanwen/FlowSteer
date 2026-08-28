from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from src.interactive.config_loader import load_yaml
from src.interactive.qa_tool_adapter import build_qa_tool_registry
from src.interactive.tool_runtime import ToolRequest
from src.interactive.triviaqa_qa_memory import (
    TriviaQAQAMemoryRecord,
    write_materialized_qa_memory,
)
from src.interactive.triviaqa_transductive_qa_memory import (
    EVALUATION_PARTITION,
    EVALUATION_REGIME,
    TRAIN_PARTITION,
    TriviaQATransductiveQAMemoryIndex,
    build_triviaqa_transductive_qa_memory_index,
    load_triviaqa_transductive_qa_memory_sources,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "triviaqa_allqa_transductive_memory_v1.yaml"


class _Encoder:
    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        vectors = []
        for index, text in enumerate(sentences):
            total = float(sum(text.encode("utf-8")) % 97 + 1)
            vectors.append([total, float(index + 1), 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _row(
    *,
    task_number: int,
    question: str,
    answer: str,
    split: str,
    selection_index: int,
) -> dict[str, object]:
    base_task_id = f"triviaqa:tc_{task_number}"
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": base_task_id,
        "question": question,
        "ground_truth": answer,
        "split": split,
        "metadata": {
            "dataset_key": "triviaqa",
            "native_split": "train",
            "evaluator_payload": {"accepted_answers": [answer]},
            "sampling": {
                "selection": "sequential",
                "selection_index": selection_index,
                "base_task_id": base_task_id,
                "cycled_training_sample": False,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sources(tmp_path: Path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "development.jsonl"
    _write_jsonl(
        train_path,
        [
            _row(
                task_number=224,
                question="Which British general was killed at Khartoum in 1885?",
                answer="Gordon",
                split="train",
                selection_index=0,
            ),
            _row(
                task_number=225,
                question="On which border is Victoria Falls?",
                answer="Zambia and Zimbabwe",
                split="train",
                selection_index=1,
            ),
        ],
    )
    _write_jsonl(
        validation_path,
        [
            _row(
                task_number=1,
                question="Which Sinclair won the 1930 Nobel Prize for Literature?",
                answer="Sinclair Lewis",
                split="validation",
                selection_index=0,
            )
        ],
    )
    source_set = load_triviaqa_transductive_qa_memory_sources(
        train_path,
        validation_path,
        expected_train_count=2,
        expected_validation_count=1,
    )
    return train_path, validation_path, source_set


def _records(source_set) -> tuple[TriviaQAQAMemoryRecord, ...]:
    return tuple(
        TriviaQAQAMemoryRecord.create(
            source=source,
            paraphrase_question="Rephrased: " + source.original_question,
            paraphrase_version="semantic-preserving-allqa-v1",
            paraphrase_method="semantic-preserving-question-paraphrase",
            generator_provider="local-openai-compatible",
            model_id="supervisor_theta",
            model_revision="Qwen3.5-9B",
            prompt_template_version="triviaqa.qa_memory.question_paraphrase.v3",
            generation_seed=20260828 + source.selection_index,
        )
        for source in source_set.sources
    )


def test_allqa_source_set_includes_paired_validation_answer() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        _, _, source_set = _sources(Path(directory))

    assert source_set.train_count == 2
    assert source_set.evaluation_count == 1
    assert source_set.total_count == 3
    evaluation = source_set.sources[-1]
    assert evaluation.original_question == (
        "Which Sinclair won the 1930 Nobel Prize for Literature?"
    )
    assert evaluation.canonical_answer == "Sinclair Lewis"
    assert source_set.partition_by_source_task_id[
        evaluation.source_train_task_id
    ] == EVALUATION_PARTITION
    assert set(source_set.partition_by_source_task_id.values()) == {
        TRAIN_PARTITION,
        EVALUATION_PARTITION,
    }


def test_transductive_index_manifest_and_existing_search_read_wire(tmp_path: Path) -> None:
    train_path, validation_path, source_set = _sources(tmp_path)
    paraphrases = tmp_path / "allqa_paraphrases.jsonl"
    records = _records(source_set)
    write_materialized_qa_memory(paraphrases, records)
    output_dir = tmp_path / "transductive_index"

    manifest = build_triviaqa_transductive_qa_memory_index(
        paraphrases_path=paraphrases,
        train_tasks_path=train_path,
        validation_tasks_path=validation_path,
        output_dir=output_dir,
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_model_revision="test-revision",
        frozen_top_k=1,
        max_tool_calls_per_agent_call=2,
        max_turns_per_agent_call=3,
        encoder=_Encoder(),
        expected_train_count=2,
        expected_validation_count=1,
    )

    value = manifest.to_value()
    assert value["contains_evaluation_answers"] is True
    assert value["evaluation_regime"] == EVALUATION_REGIME
    assert value["official_heldout_eligible"] is False
    assert value["validation_content_indexed"] is True
    assert value["source_counts"] == {
        "train": 2,
        "frozen_development_validation": 1,
        "total": 3,
    }
    assert value["evaluation_memory_overlap_count"] == 1
    assert value["memory_count"] == 3

    memberships = [
        json.loads(line)
        for line in (output_dir / "source_membership.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    evaluation_memberships = [
        row for row in memberships if row["contains_evaluation_answer"]
    ]
    assert len(evaluation_memberships) == 1
    evaluation_memory_id = evaluation_memberships[0]["memory_id"]

    index = TriviaQATransductiveQAMemoryIndex.open(
        output_dir,
        encoder=_Encoder(),
    )
    try:
        hit = index.search("Which Sinclair won?", limit=1)[0]
        assert hit.rank == 1
        evaluation_record = index.read(evaluation_memory_id)
        assert evaluation_record.canonical_answer == "Sinclair Lewis"
        assert "Sinclair Lewis" in evaluation_record.embedding_text()

        registry = build_qa_tool_registry(index, dataset_scope=("triviaqa",))
        search_result, search_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                "triviaqa.qa_memory",
                ToolRequest(
                    "search",
                    {"query": "Which Sinclair won?", "limit": 1},
                ),
            )
        )
        assert search_result is not None
        assert len(search_result.value["hits"]) == 1
        assert search_receipt.tool_id == "triviaqa.qa_memory"
        read_result, read_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                "triviaqa.qa_memory",
                ToolRequest("read", {"memory_id": evaluation_memory_id}),
            )
        )
        assert read_result is not None
        assert read_result.value["memory"]["canonical_answer"] == "Sinclair Lewis"
        assert read_receipt.tool_id == "triviaqa.qa_memory"
    finally:
        index.close()


def test_prepared_profile_cannot_be_reported_as_official_heldout() -> None:
    config = load_yaml(CONFIG)

    assert config["status"] == "prepared_only"
    assert config["evaluation_regime"] == "transductive_retrieval"
    assert config["official_heldout_eligible"] is False
    assert config["contains_evaluation_answers"] is True
    assert config["sources"]["source_counts"] == {
        "train": 512,
        "frozen_development_validation": 128,
        "total": 640,
    }
    assert config["sources"]["evaluation_memory_overlap_count"] == 128
    assert config["index"]["output_path"] != config["index"][
        "train_only_path_must_remain"
    ]
    assert config["tool_runtime"]["director_tool_calls_allowed"] is False
    assert config["tool_runtime"]["web_search_allowed"] is False
