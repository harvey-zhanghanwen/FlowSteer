#!/usr/bin/env python3
"""Run deterministic, answer-free HotpotQA embedding Tool smoke checks."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_embedding_index import HotpotQAEmbeddingIndex
from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_RETRIEVAL_TOOL_ID,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.task_dataset import iter_task_records
from src.interactive.tool_runtime import ToolRequest


FORBIDDEN_FIELDS = {
    "answer",
    "answers",
    "supporting_facts",
    "reference_answer",
    "ground_truth",
    "evaluator",
    "evaluator_receipt",
}


def _contains_forbidden_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(FORBIDDEN_FIELDS & set(value)) or any(
            _contains_forbidden_field(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


async def smoke(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = config["qa_embedding_retrieval"]
    index_dir = Path(section["index_dir"])
    if not index_dir.is_absolute():
        index_dir = root / index_dir
    development_path = Path(section["development_tasks"])
    if not development_path.is_absolute():
        development_path = root / development_path
    task = next(
        item
        for item in iter_task_records(development_path, expected_split="train")
        if item.task_id.startswith("hotpotqa:")
    )
    index = HotpotQAEmbeddingIndex.open(
        index_dir,
        embedding_model_path=str(section["embedding_model"]),
        embedding_device=str(section["embedding_device"]),
    )
    first = await index.search(task.task_id, task.question, int(section["search_top_k"]))
    second = await index.search(task.task_id, task.question, int(section["search_top_k"]))
    deterministic = first == second
    registry = build_hotpotqa_embedding_tool_registry(
        index,
        task_id=task.task_id,
        frozen_top_k=int(section["search_top_k"]),
        timeout_seconds=float(section["tool_timeout_seconds"]),
    )
    search_result, search_receipt = await registry.ainvoke_with_receipt(
        HOTPOTQA_RETRIEVAL_TOOL_ID,
        ToolRequest(
            "search",
            {"query": task.question, "k": int(section["search_top_k"])},
        ),
    )
    if search_result is None:
        raise RuntimeError("dynamic search Tool smoke failed")
    doc_id = str(search_result.value["doc_ids"][0])
    read_result, read_receipt = await registry.ainvoke_with_receipt(
        HOTPOTQA_RETRIEVAL_TOOL_ID,
        ToolRequest("read", {"doc_id": doc_id}),
    )
    if read_result is None:
        raise RuntimeError("dynamic read Tool smoke failed")
    receipts = [search_receipt.to_value(), read_receipt.to_value()]
    no_private_metadata = not _contains_forbidden_field(
        {
            "manifest": index.manifest.to_value(),
            "receipts": receipts,
        }
    )
    passed = deterministic and no_private_metadata
    return {
        "schema_version": "flowsteer.hotpotqa.embedding_retrieval_smoke.v1",
        "passed": passed,
        "task_id": task.task_id,
        "query": task.question,
        "same_query_top_k_deterministic": deterministic,
        "answer_private_metadata_absent": no_private_metadata,
        "index_manifest": index.manifest.to_value(),
        "ranked_doc_ids": [item.passage_id for item in first],
        "tool_receipts": receipts,
        "web_search_used": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_embedding_retrieval_v2.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotpotqa_embedding_retrieval_v1/index_smoke_receipt.json",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    value = asyncio.run(smoke(config_path))
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = config_path.parent.parent / output
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
