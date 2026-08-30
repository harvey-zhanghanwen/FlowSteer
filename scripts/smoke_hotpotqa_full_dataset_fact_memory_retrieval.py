#!/usr/bin/env python3
"""Smoke an existing HotpotQA full-dataset declarative-fact index.

This is the fact-only counterpart of ``smoke_hotpotqa_qa_memory_retrieval``.
It opens the configured index without rebuilding it, checks repeated-query
top-k determinism, and drives one local scripted worker through FlowSteer's
Tool/ReAct adapter.  The scripted gateway performs no language-model call;
its only purpose is to prove that search/read receipts are owned by the
worker Agent and contain the fact-only public projection.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, AgentRuntime
from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_FACT_MEMORY_TOOL_ID,
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    HotpotQAFullDatasetFactMemoryIndex,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


WORKER_AGENT_ID = "fact-retrieval-smoke-worker"
SMOKE_MODEL_ID = "fact-retrieval-smoke-script"
SEARCH_HIT_FIELDS = frozenset(
    {"memory_id", "fact_snippet", "similarity", "rank"}
)
READ_FACT_FIELDS = frozenset({"memory_id", "fact_text"})
FORBIDDEN_RETRIEVAL_FIELDS = frozenset(
    {
        "accepted_aliases",
        "answer",
        "canonical_answer",
        "evaluator_payload",
        "evaluator_receipt",
        "ground_truth",
        "original_answer",
        "original_question",
        "paraphrase_answer_statement",
        "paraphrase_question",
        "question",
        "source_train_task_id",
        "supporting_fact",
        "supporting_facts",
        "validation_answer",
        "validation_question",
    }
)
_QA_WIRE = re.compile(r"(?:^|\n)\s*(?:question|answer)\s*:", re.IGNORECASE)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _normalized_query(value: str) -> str:
    return " ".join(value.split()).casefold()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _first_public_question(path: Path) -> tuple[str, str]:
    """Load only public task identity/question as an external Tool query."""

    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _mapping(json.loads(line), "development task")
                task_id = value["task_id"]
                question = value["question"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid task") from exc
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path}:{line_number}: empty task_id")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{line_number}: empty question")
            # Aligned QA tasks may carry the public-context wrapper used by the
            # evaluator.  The retrieval query is only the final question scope.
            marker = "Question:"
            scoped = question.rsplit(marker, 1)[-1] if marker in question else question
            return task_id.strip(), scoped.strip()
    raise ValueError(f"{path}: no development task is available")


def _semantic_query_rewrite(path: Path, *, task_id: str) -> str:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _mapping(json.loads(line), "fact provenance")
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid provenance") from exc
            if value.get("source_train_task_id") != task_id:
                continue
            rewrite = value.get("paraphrase_question")
            if not isinstance(rewrite, str) or not rewrite.strip():
                raise ValueError("fact provenance has no semantic query rewrite")
            return rewrite.strip()
    raise ValueError("fact provenance has no row for the development task")


def _action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "resource_id": resource_id,
            "skill_id": None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class _ScriptedWorkerGateway:
    """SkillFlow-style finite StructuredAction source with no model call."""

    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if not self._outputs:
            raise RuntimeError("fact-memory smoke action sequence is exhausted")
        return AgentResponse(self._outputs.pop(0), {"scripted_smoke": True})


def _retrieval_payloads(
    receipts: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    hits: list[Mapping[str, object]] = []
    facts: list[Mapping[str, object]] = []
    for receipt in receipts:
        result = receipt.get("result")
        result_value = result.get("value") if isinstance(result, Mapping) else None
        if not isinstance(result_value, Mapping):
            continue
        operation = result_value.get("operation")
        if operation == "search":
            raw_hits = result_value.get("hits")
            if isinstance(raw_hits, list):
                hits.extend(
                    _mapping(hit, "search hit") for hit in raw_hits
                )
        elif operation == "read":
            raw_fact = result_value.get("fact")
            if isinstance(raw_fact, Mapping):
                facts.append(raw_fact)
    return hits, facts


def _contains_forbidden_field(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in FORBIDDEN_RETRIEVAL_FIELDS:
                return True
            if _contains_forbidden_field(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


async def smoke(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if section.get("corpus_kind") != "full_dataset_fact_memory":
        raise ValueError("config does not select full_dataset_fact_memory")
    if section.get("document_format") != "declarative_fact_only":
        raise ValueError("config does not select declarative_fact_only records")
    if section.get("indexed_text_field") != "fact_text":
        raise ValueError("config does not index fact_text")
    if section.get("evaluation_scope") != FULL_DATASET_EVALUATION_SCOPE:
        raise ValueError("config evaluation scope differs from the fact index")
    if section.get("web_search_enabled") is not False:
        raise ValueError("fact-memory smoke requires Web Search to be disabled")

    frozen_top_k = int(section["search_top_k"])
    index_dir = _resolve(root, section["index_dir"])
    task_id, raw_query = _first_public_question(
        _resolve(root, section["development_tasks"])
    )
    query = _semantic_query_rewrite(
        _resolve(root, section["paraphrase_materialization_path"]),
        task_id=task_id,
    )
    if " ".join(query.split()).casefold() == " ".join(raw_query.split()).casefold():
        raise ValueError("semantic query rewrite equals the raw development question")
    index = HotpotQAFullDatasetFactMemoryIndex.open(
        index_dir,
        embedding_model_path=str(section["embedding_model"]),
        embedding_device=str(section.get("embedding_device", "cpu")),
    )
    if index.manifest.frozen_top_k != frozen_top_k:
        raise ValueError("config top-k differs from the existing index manifest")

    first_ranking = await index.search(query, frozen_top_k)
    repeated_ranking = await index.search(query, frozen_top_k)
    same_query_deterministic = first_ranking == repeated_ranking
    ranked_memory_ids = [hit.memory_id for hit in first_ranking]
    if not ranked_memory_ids:
        raise ValueError("existing fact-memory index returned no candidate")

    registry = build_hotpotqa_embedding_tool_registry(
        index,
        task_id=task_id,
        tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
        frozen_top_k=frozen_top_k,
        timeout_seconds=float(section.get("tool_timeout_seconds", 30.0)),
    )
    actions = [
        _action(
            "tool",
            name="search",
            arguments={"query": query, "k": frozen_top_k},
            resource_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
        ),
        *(
            _action(
                "tool",
                name="read",
                arguments={"memory_id": memory_id},
                resource_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
            )
            for memory_id in ranked_memory_ids
        ),
        _action(
            "complete",
            name="complete",
            arguments={
                "value": {
                    "retrieval_sufficiency": "supported",
                    "selected_memory_id": ranked_memory_ids[0],
                }
            },
            resource_id=None,
        ),
    ]
    gateway = _ScriptedWorkerGateway(actions)
    model_registry = ModelRegistry(
        [ProviderSpec("fact-retrieval-smoke", kind="test")],
        [ModelSpec(SMOKE_MODEL_ID, "fact-retrieval-smoke")],
    )
    adapter = HotpotQAEmbeddingReactExecutionAdapter(
        gateway=gateway,
        tool_registry=registry,
        retrieval_query_scope=raw_query,
        max_turns=max(len(actions), int(section.get("max_turns_per_agent_call", 7))),
        max_tool_calls=max(
            len(ranked_memory_ids) + 1,
            int(section.get("max_tool_calls_per_agent_call", 6)),
        ),
        max_action_tokens=int(section.get("max_action_tokens", 4096)),
    )
    runtime = AgentRuntime(
        model_registry,
        gateway,
        execution_adapters={"react": adapter},
        tool_registry=registry,
        dataset_id="hotpotqa",
    )
    graph = AgentGraph(
        [
            AgentNode(
                WORKER_AGENT_ID,
                SMOKE_MODEL_ID,
                "Retrieve and read the relevant declarative fact.",
                execution_mode="react",
                allowed_tools=(HOTPOTQA_FACT_MEMORY_TOOL_ID,),
            )
        ],
        [],
        output_agent_id=WORKER_AGENT_ID,
    )
    execution = await runtime.execute(
        graph,
        query,
        run_id="hotpotqa-full-dataset-fact-memory-smoke",
    )
    worker_metadata = execution.output_metadata.get(WORKER_AGENT_ID, {})
    raw_receipts = worker_metadata.get("tool_receipts", ())
    receipts = [
        _mapping(receipt, "worker Tool receipt")
        for receipt in raw_receipts
        if isinstance(receipt, Mapping)
    ] if isinstance(raw_receipts, (list, tuple)) else []
    hits, facts = _retrieval_payloads(receipts)

    receipt_actions = [
        request.get("action")
        for receipt in receipts
        if isinstance((request := receipt.get("request")), Mapping)
    ]
    expected_actions = ["search", *("read" for _ in ranked_memory_ids)]
    exact_fact_projection = (
        bool(hits)
        and len(facts) == len(ranked_memory_ids)
        and all(set(hit) == SEARCH_HIT_FIELDS for hit in hits)
        and all(set(fact) == READ_FACT_FIELDS for fact in facts)
    )
    no_qa_fields = not _contains_forbidden_field([hits, facts])
    no_qa_wire = all(
        not _QA_WIRE.search(str(fact["fact_text"])) for fact in facts
    )
    worker_owned = (
        bool(execution.calls)
        and all(call.request.agent.id == WORKER_AGENT_ID for call in execution.calls)
        and set(execution.output_metadata) == {WORKER_AGENT_ID}
        and bool(receipts)
        and all(
            receipt.get("tool_id") == HOTPOTQA_FACT_MEMORY_TOOL_ID
            for receipt in receipts
        )
    )
    all_receipts_succeeded = bool(receipts) and all(
        receipt.get("error_type") is None for receipt in receipts
    )
    passed = all(
        (
            same_query_deterministic,
            receipt_actions == expected_actions,
            exact_fact_projection,
            no_qa_fields,
            no_qa_wire,
            worker_owned,
            all_receipts_succeeded,
            _normalized_query(query) != _normalized_query(raw_query),
        )
    )
    return {
        "schema_version": (
            "flowsteer.hotpotqa.full_dataset_fact_memory_retrieval_smoke.v1"
        ),
        "passed": passed,
        "task_id": task_id,
        "query": query,
        "semantic_query_rewrite_used": True,
        "raw_question_embedding_query_used": False,
        "index_dir": str(index_dir),
        "index_manifest": index.manifest.to_value(),
        "same_query_top_k_deterministic": same_query_deterministic,
        "ranked_memory_ids": ranked_memory_ids,
        "worker_agent_id": WORKER_AGENT_ID,
        "worker_receipt_ownership_valid": worker_owned,
        "receipt_actions": receipt_actions,
        "fact_only_search_read_projection_valid": exact_fact_projection,
        "qa_fields_absent_from_retrieval_payload": no_qa_fields,
        "qa_wire_absent_from_retrieval_payload": no_qa_wire,
        "tool_id": HOTPOTQA_FACT_MEMORY_TOOL_ID,
        "tool_receipts": receipts,
        "web_search_used": False,
        "model_api_calls": 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "config/evaluation_hotpotqa_round01_full_dataset_fact_memory_v21.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_round01_full_dataset_fact_memory_v21/"
            "fact_memory_index_smoke_receipt.json"
        ),
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
