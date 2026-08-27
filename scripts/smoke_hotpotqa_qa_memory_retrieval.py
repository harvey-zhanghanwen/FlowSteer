#!/usr/bin/env python3
"""Smoke the train-only HotpotQA QA-memory index and dynamic Tool wire.

Two temporary indexes are rebuilt from the same frozen materialization.  The
smoke then invokes ``search`` and ``read`` through the project's public
QA-memory Tool-registry factory.  It never invokes a language-model API and it
does not write or replace the configured formal index.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_qa_memory_index import (
    HotpotQAQAMemoryIndex,
    build_hotpotqa_qa_memory_index,
    load_hotpotqa_train_qa_sources,
    load_paraphrase_materialization,
)
from src.interactive.tool_runtime import ToolRegistry, ToolRequest


FORBIDDEN_PRIVATE_FIELDS = frozenset(
    {
        "accepted_aliases",
        "evaluator_payload",
        "evaluator_receipt",
        "ground_truth",
        "supporting_fact",
        "supporting_facts",
        "validation_answer",
        "validation_ground_truth",
        "validation_question",
    }
)
RegistryFactory = Callable[..., ToolRegistry]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _task_ids(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _mapping(json.loads(line), "task identity")
                task_id = value["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid task identity") from exc
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path}:{line_number}: empty task identity")
            result.append(task_id.strip())
    return tuple(result)


def _contains_private_field(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in FORBIDDEN_PRIVATE_FIELDS or _contains_private_field(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_private_field(item) for item in value)
    return False


def _default_registry_factory() -> RegistryFactory:
    # Public factory call point supplied by the QA-memory Tool Adapter.  The
    # import is intentionally lazy so index/profile unit tests stay independent
    # while the adapter is integrated in the shared worktree.
    from src.interactive.hotpotqa_embedding_tool import (
        HOTPOTQA_QA_MEMORY_TOOL_ID,
        build_hotpotqa_embedding_tool_registry,
    )

    def factory(index: object, **kwargs: object) -> ToolRegistry:
        return build_hotpotqa_embedding_tool_registry(
            index,  # type: ignore[arg-type]
            tool_id=HOTPOTQA_QA_MEMORY_TOOL_ID,
            **kwargs,  # type: ignore[arg-type]
        )

    return factory


def _registry(
    factory: RegistryFactory,
    index: HotpotQAQAMemoryIndex,
    *,
    task_id: str,
    frozen_top_k: int,
    timeout_seconds: float,
) -> ToolRegistry:
    registry = factory(
        index,
        task_id=task_id,
        frozen_top_k=frozen_top_k,
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(registry, ToolRegistry):
        raise TypeError("QA-memory Tool factory returned an incompatible registry")
    if len(registry.resource_ids) != 1:
        raise ValueError("QA-memory smoke requires exactly one registered Tool")
    return registry


def _result_value(result: object, name: str) -> Mapping[str, object]:
    if result is None or not hasattr(result, "value"):
        raise RuntimeError(f"dynamic {name} Tool smoke failed")
    return _mapping(getattr(result, "value"), f"{name} Tool result")


def _first_memory_id(search_value: Mapping[str, object]) -> str:
    for field_name in ("memory_ids", "doc_ids"):
        values = search_value.get(field_name)
        if isinstance(values, (list, tuple)) and values:
            value = values[0]
            if isinstance(value, str) and value.strip():
                return value.strip()
    hits = search_value.get("hits")
    if isinstance(hits, (list, tuple)) and hits:
        hit = _mapping(hits[0], "search hit")
        for field_name in ("memory_id", "doc_id"):
            value = hit.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("dynamic search result did not expose a readable memory ID")


def _read_argument_name(registry: ToolRegistry, tool_id: str) -> str:
    schema = registry.require_capability(tool_id).action_schemas.get("read")
    if not isinstance(schema, Mapping):
        raise ValueError("QA-memory Tool has no read action schema")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("QA-memory read action has no argument properties")
    if "memory_id" in properties:
        return "memory_id"
    if "doc_id" in properties:
        return "doc_id"
    raise ValueError("QA-memory read action has no memory ID argument")


def _receipt_payload_has_required_memory_fields(
    search_value: Mapping[str, object],
    read_value: Mapping[str, object],
) -> bool:
    hits = search_value.get("hits")
    if not isinstance(hits, (list, tuple)) or not hits:
        return False
    hit = _mapping(hits[0], "search hit")
    search_fields = {"rank", "similarity", "source_train_task_id"}
    if not search_fields <= set(hit):
        return False
    candidate = read_value.get("memory", read_value)
    memory = _mapping(candidate, "read memory")
    return {
        "source_train_task_id",
        "paraphrase_question",
        "paraphrase_answer_statement",
    } <= set(memory)


async def smoke(
    config_path: Path,
    *,
    registry_factory: RegistryFactory | None = None,
) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if section.get("corpus_kind") != "train_qa_memory":
        raise ValueError("config does not select the train_qa_memory corpus")
    expected_train_count = int(section.get("train_sample_count", 512))
    expected_validation_count = int(section.get("validation_sample_count", 128))
    if expected_train_count != 512 or expected_validation_count != 128:
        raise ValueError("HotpotQA QA-memory smoke requires the frozen 512/128 split")
    frozen_top_k = int(section["search_top_k"])
    validation_ids = _task_ids(
        _resolve(root, section["frozen_validation_tasks"])
    )
    train_path = _resolve(root, section["train_tasks"])
    paraphrases = load_paraphrase_materialization(
        _resolve(root, section["paraphrase_materialization_path"])
    )
    sources = load_hotpotqa_train_qa_sources(
        train_path,
        validation_task_ids=validation_ids,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    query_source = next((source for source in sources if not source.cycled), sources[0])
    build_kwargs = {
        "train_jsonl": train_path,
        "validation_task_ids": validation_ids,
        "paraphrases": paraphrases,
        "embedding_model_path": str(section["embedding_model"]),
        "embedding_model_id": str(section["embedding_model_id"]),
        "embedding_device": str(section.get("embedding_device", "cpu")),
        "frozen_top_k": frozen_top_k,
        "expected_train_count": expected_train_count,
        "expected_validation_count": expected_validation_count,
    }

    with TemporaryDirectory(prefix="hotpotqa-qa-memory-smoke-") as temporary:
        temporary_root = Path(temporary)
        first_dir = temporary_root / "rebuild-a"
        second_dir = temporary_root / "rebuild-b"
        first_manifest = build_hotpotqa_qa_memory_index(
            index_dir=first_dir,
            **build_kwargs,
        )
        second_manifest = build_hotpotqa_qa_memory_index(
            index_dir=second_dir,
            **build_kwargs,
        )
        first_index = HotpotQAQAMemoryIndex.open(
            first_dir,
            embedding_model_path=str(section["embedding_model"]),
            embedding_device=str(section.get("embedding_device", "cpu")),
        )
        second_index = HotpotQAQAMemoryIndex.open(
            second_dir,
            embedding_model_path=str(section["embedding_model"]),
            embedding_device=str(section.get("embedding_device", "cpu")),
        )
        first_ranking = await first_index.search(query_source.question, frozen_top_k)
        repeated_ranking = await first_index.search(query_source.question, frozen_top_k)
        rebuilt_ranking = await second_index.search(query_source.question, frozen_top_k)
        same_query_deterministic = first_ranking == repeated_ranking
        deterministic_rebuild = (
            first_manifest.to_value() == second_manifest.to_value()
            and (first_dir / "memories.jsonl").read_bytes()
            == (second_dir / "memories.jsonl").read_bytes()
            and np.array_equal(
                np.load(first_dir / "embeddings.npy", allow_pickle=False),
                np.load(second_dir / "embeddings.npy", allow_pickle=False),
            )
            and first_ranking == rebuilt_ranking
        )

        factory = registry_factory or _default_registry_factory()
        registry = _registry(
            factory,
            first_index,
            task_id="hotpotqa:qa-memory-smoke-worker",
            frozen_top_k=frozen_top_k,
            timeout_seconds=float(section.get("tool_timeout_seconds", 30.0)),
        )
        tool_id = registry.resource_ids[0]
        search_result, search_receipt = await registry.ainvoke_with_receipt(
            tool_id,
            ToolRequest("search", {"query": query_source.question, "k": frozen_top_k}),
        )
        search_value = _result_value(search_result, "search")
        memory_id = _first_memory_id(search_value)
        read_argument = _read_argument_name(registry, tool_id)
        read_result, read_receipt = await registry.ainvoke_with_receipt(
            tool_id,
            ToolRequest("read", {read_argument: memory_id}),
        )
        read_value = _result_value(read_result, "read")
        receipts = [search_receipt.to_value(), read_receipt.to_value()]

        serialized_artifacts = {
            "manifest": first_manifest.to_value(),
            "memories": [
                json.loads(line)
                for line in (first_dir / "memories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ],
            "tool_receipts": receipts,
        }
        no_private_metadata = not _contains_private_field(serialized_artifacts)
        receipt_fields_present = _receipt_payload_has_required_memory_fields(
            search_value,
            read_value,
        )
        frozen_split_counts = (
            first_manifest.train_record_count == 512
            and first_manifest.heldout_validation_count == 128
            and first_manifest.validation_overlap_count == 0
        )
        passed = all(
            (
                frozen_split_counts,
                deterministic_rebuild,
                same_query_deterministic,
                no_private_metadata,
                receipt_fields_present,
                search_receipt.error_type is None,
                read_receipt.error_type is None,
            )
        )
        return {
            "schema_version": "flowsteer.hotpotqa.qa_memory_retrieval_smoke.v1",
            "passed": passed,
            "train_record_count": first_manifest.train_record_count,
            "unique_source_count": first_manifest.unique_source_count,
            "cycled_record_count": first_manifest.cycled_record_count,
            "paraphrase_count": first_manifest.paraphrase_count,
            "heldout_validation_count": first_manifest.heldout_validation_count,
            "validation_overlap_count": first_manifest.validation_overlap_count,
            "frozen_split_counts_valid": frozen_split_counts,
            "deterministic_rebuild": deterministic_rebuild,
            "same_query_top_k_deterministic": same_query_deterministic,
            "private_validation_or_evaluator_fields_absent": no_private_metadata,
            "dynamic_search_read_receipt_fields_present": receipt_fields_present,
            "query_source_train_task_id": query_source.source_train_task_id,
            "query": query_source.question,
            "ranked_memory_ids": [hit.memory_id for hit in first_ranking],
            "tool_id": tool_id,
            "tool_receipts": receipts,
            "web_search_used": False,
            "index_manifest": first_manifest.to_value(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_qa_memory_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotpotqa_qa_memory_v1/index_smoke_receipt.json",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    value = asyncio.run(smoke(config_path))
    output = _resolve(config_path.parent.parent, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "passed": value["passed"]}))
    return 0 if value["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
