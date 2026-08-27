#!/usr/bin/env python3
"""Offline formal-result analysis for TriviaQA QA-memory AgentGraph.

This is a thin QA-memory specialization of
``analyze_triviaqa_embedding_react_failures.py``.  It reads persisted files
only; it never imports provider/runtime code and never starts evaluation.
Besides paired EM/F1 and wrong-demo analysis, it verifies the control/data
plane boundary required by the QA-memory condition:

* the Director has no retrieval Tool call or retrieval payload;
* ``triviaqa.qa_memory`` receipts belong to Tool-capable worker Agents; and
* retrieved artifacts travel over observed, directed AgentGraph relations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import string
import sys
from typing import Any, Iterable, Mapping, Sequence

try:  # Module import in tests.
    from scripts import analyze_triviaqa_embedding_react_failures as base
except ImportError:  # Direct ``python scripts/...`` execution.
    _BASE_PATH = Path(__file__).with_name(
        "analyze_triviaqa_embedding_react_failures.py"
    )
    _BASE_SPEC = importlib.util.spec_from_file_location(
        "_flowsteer_triviaqa_embedding_analysis", _BASE_PATH
    )
    if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load base analyzer from {_BASE_PATH}")
    base = importlib.util.module_from_spec(_BASE_SPEC)
    sys.modules[_BASE_SPEC.name] = base
    _BASE_SPEC.loader.exec_module(base)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts/triviaqa_qa_memory_v1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/triviaqa_qa_memory_v1"
QA_MEMORY_TOOL_ID = "triviaqa.qa_memory"
FINAL_MANIFEST_STATUSES = {
    "completed",
    "completed_with_terminal_failures",
    "completed_with_operational_failures",
}
RETRIEVAL_PAYLOAD_MARKERS = (
    "canonical_answer",
    "paraphrase_answer_statement",
    "paraphrase_question",
    "source_train_task_id",
    "cycled_training_sample",
    "memory_id",
    '"hits"',
    '"similarity"',
)
CANONICAL_RECEIPT_DATA_FIELDS = (
    "memory_id",
    "canonical_answer",
    "paraphrase_question",
    "paraphrase_answer_statement",
    "source_train_task_id",
    "base_task_id",
)
PROVENANCE_IDENTIFIER_FIELDS = frozenset(
    {"memory_id", "source_train_task_id", "base_task_id"}
)
SEMANTIC_PAYLOAD_FIELDS = frozenset(
    {"canonical_answer", "paraphrase_question", "paraphrase_answer_statement"}
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# Direct reuse of the normalization used by
# ``src.interactive.task_evaluator._normalize_triviaqa_answer``.  This helper
# is kept local so the offline analyzer does not import runtime/provider code.
def _normalize_triviaqa_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _task_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("task_id")
    return value if isinstance(value, str) and value else None


def _deduplicate_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    missing = 0
    for row in rows:
        task_id = _task_id(row)
        if task_id is None:
            missing += 1
            continue
        if task_id in by_task:
            duplicates[task_id] += 1
        by_task[task_id] = row
    return by_task, {
        "deduplicated_task_count": len(by_task),
        "duplicate_task_record_counts": dict(sorted(duplicates.items())),
        "missing_task_id_records": missing,
        "duplicate_resolution": "last complete JSONL record for task_id wins",
    }


def _condition_metrics(
    selected_ids: Sequence[str],
    paired: Mapping[str, Mapping[str, Any]],
    condition: str,
) -> dict[str, Any]:
    completed = valid = 0
    em_sum = f1_sum = 0.0
    for task_id in selected_ids:
        value = paired.get(task_id, {}).get(condition)
        if not isinstance(value, Mapping):
            continue
        if value.get("available") is True:
            completed += 1
        evaluation = value.get("evaluation")
        evaluation_valid = (
            isinstance(evaluation, Mapping) and evaluation.get("valid") is True
        )
        receipt_valid = value.get("valid") is True and evaluation_valid
        exact_match = _number(value.get("exact_match"))
        token_f1 = _number(value.get("token_f1"))
        if receipt_valid and exact_match is not None and token_f1 is not None:
            valid += 1
            em_sum += exact_match
            f1_sum += token_f1
    denominator = len(selected_ids)
    return {
        "denominator": denominator,
        "completed": completed,
        "evaluator_valid": valid,
        "strict_exact_match": em_sum / denominator if denominator else None,
        "strict_token_f1": f1_sum / denominator if denominator else None,
        "completed_only_exact_match": em_sum / valid if valid else None,
        "completed_only_token_f1": f1_sum / valid if valid else None,
    }


def _accepted_answers(task: Mapping[str, Any]) -> list[str]:
    metadata = base._mapping(task.get("metadata"))
    evaluator_payload = base._mapping(metadata.get("evaluator_payload"))
    answers = [
        value.strip()
        for value in base._list(evaluator_payload.get("accepted_answers"))
        if isinstance(value, str) and value.strip()
    ]
    if answers:
        return answers
    ground_truth = task.get("ground_truth")
    if not isinstance(ground_truth, str):
        return []
    return [value.strip() for value in ground_truth.split("|") if value.strip()]


def _receipt_canonical_answers(item: Mapping[str, Any]) -> list[str]:
    receipt = base._mapping(item.get("receipt"))
    result = base._mapping(receipt.get("result"))
    if result.get("completed") is not True:
        return []
    value = base._mapping(result.get("value"))
    candidates: list[str] = []
    if item.get("action") == "search":
        records = base._list(value.get("hits"))
    elif item.get("action") == "read":
        memory = value.get("memory")
        records = [memory] if isinstance(memory, Mapping) else []
    else:
        records = []
    for raw_record in records:
        record = base._mapping(raw_record)
        answer = record.get("canonical_answer")
        if isinstance(answer, str) and answer.strip():
            candidates.append(answer.strip())
    return candidates


def _normalized_receipt_query(item: Mapping[str, Any]) -> str | None:
    receipt = base._mapping(item.get("receipt"))
    request = base._mapping(receipt.get("request"))
    arguments = base._mapping(request.get("arguments"))
    return base._normalize_query(  # noqa: SLF001 - upstream analyzer reuse
        arguments.get("query")
    )


def _retrieval_coverage(
    selected_by_id: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    corpus_memories: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compute post-hoc accepted-answer coverage from persisted Tool receipts.

    Accepted answers are used only after the frozen run for diagnosis.  They
    never enter a query, Tool request, Agent observation, or Director input.
    A match is lexical under the same TriviaQA normalization as the evaluator;
    it does not by itself prove that the retrieved QA record binds the target
    relation of the held-out question.
    """

    search_calls = read_calls = 0
    successful_search_calls = successful_read_calls = 0
    nonempty_search_calls = search_candidate_count = 0
    tasks_with_search: list[str] = []
    tasks_with_read: list[str] = []
    tasks_with_multiple_queries: list[str] = []
    search_match_tasks: list[str] = []
    search_top1_match_tasks: list[str] = []
    read_match_tasks: list[str] = []
    corpus_answers = {
        normalized
        for memory in corpus_memories
        if isinstance(memory, Mapping)
        and isinstance(memory.get("canonical_answer"), str)
        and (
            normalized := _normalize_triviaqa_answer(
                str(memory.get("canonical_answer"))
            )
        )
    }
    corpus_match_tasks: list[str] = []
    for task_id, task in selected_by_id.items():
        trajectory = trajectories.get(task_id)
        if trajectory is None:
            continue
        accepted = {
            normalized
            for answer in _accepted_answers(task)
            if (normalized := _normalize_triviaqa_answer(answer))
        }
        if accepted & corpus_answers:
            corpus_match_tasks.append(task_id)
        receipts = list(_iter_retrieval_receipts(trajectory))
        search_receipts = [item for item in receipts if item.get("action") == "search"]
        read_receipts = [item for item in receipts if item.get("action") == "read"]
        search_calls += len(search_receipts)
        read_calls += len(read_receipts)
        for item in search_receipts:
            receipt = base._mapping(item.get("receipt"))
            result = base._mapping(receipt.get("result"))
            if result.get("completed") is not True:
                continue
            successful_search_calls += 1
            hits = base._list(base._mapping(result.get("value")).get("hits"))
            search_candidate_count += len(hits)
            if hits:
                nonempty_search_calls += 1
        successful_read_calls += sum(
            base._mapping(
                base._mapping(item.get("receipt")).get("result")
            ).get("completed")
            is True
            for item in read_receipts
        )
        if search_receipts:
            tasks_with_search.append(task_id)
        if read_receipts:
            tasks_with_read.append(task_id)
        normalized_queries = {
            query
            for item in search_receipts
            if (query := _normalized_receipt_query(item))
        }
        if len(normalized_queries) > 1:
            tasks_with_multiple_queries.append(task_id)
        search_candidates = {
            _normalize_triviaqa_answer(answer)
            for item in search_receipts
            for answer in _receipt_canonical_answers(item)
        }
        top1_candidates: set[str] = set()
        for item in search_receipts:
            receipt = base._mapping(item.get("receipt"))
            result = base._mapping(receipt.get("result"))
            value = base._mapping(result.get("value"))
            hits = base._list(value.get("hits"))
            if not hits:
                continue
            top_hit = base._mapping(hits[0])
            answer = top_hit.get("canonical_answer")
            if isinstance(answer, str) and answer.strip():
                top1_candidates.add(_normalize_triviaqa_answer(answer))
        read_candidates = {
            _normalize_triviaqa_answer(answer)
            for item in read_receipts
            for answer in _receipt_canonical_answers(item)
        }
        if accepted & search_candidates:
            search_match_tasks.append(task_id)
        if accepted & top1_candidates:
            search_top1_match_tasks.append(task_id)
        if accepted & read_candidates:
            read_match_tasks.append(task_id)

    denominator = len(selected_by_id)
    return {
        "analysis_scope": "post_hoc_offline_only_not_model_visible",
        "match_protocol": "triviaqa_normalized_exact_match_against_accepted_answers",
        "semantic_relation_binding_guaranteed": False,
        "sample_count": denominator,
        "tool_call_count": search_calls + read_calls,
        "search_call_count": search_calls,
        "read_call_count": read_calls,
        "successful_search_call_count": successful_search_calls,
        "successful_read_call_count": successful_read_calls,
        "nonempty_search_call_count": nonempty_search_calls,
        "search_candidate_count": search_candidate_count,
        "mean_tool_calls_per_task": (
            (search_calls + read_calls) / denominator if denominator else None
        ),
        "tasks_with_search_count": len(tasks_with_search),
        "tasks_with_read_count": len(tasks_with_read),
        "tasks_with_multiple_successful_queries_count": len(
            tasks_with_multiple_queries
        ),
        "tasks_with_multiple_successful_queries": tasks_with_multiple_queries,
        "corpus_accepted_answer_match_count": len(corpus_match_tasks),
        "corpus_accepted_answer_match_rate": (
            len(corpus_match_tasks) / denominator if denominator else None
        ),
        "corpus_accepted_answer_match_task_ids": corpus_match_tasks,
        "search_top1_accepted_answer_match_count": len(search_top1_match_tasks),
        "search_top1_accepted_answer_match_rate": (
            len(search_top1_match_tasks) / denominator if denominator else None
        ),
        "search_candidate_accepted_answer_match_count": len(search_match_tasks),
        "search_candidate_accepted_answer_match_rate": (
            len(search_match_tasks) / denominator if denominator else None
        ),
        "search_candidate_accepted_answer_match_task_ids": search_match_tasks,
        "search_candidate_recall_within_corpus_covered": (
            len(search_match_tasks) / len(corpus_match_tasks)
            if corpus_match_tasks
            else None
        ),
        "read_candidate_accepted_answer_match_count": len(read_match_tasks),
        "read_candidate_accepted_answer_match_rate": (
            len(read_match_tasks) / denominator if denominator else None
        ),
        "read_candidate_accepted_answer_match_task_ids": read_match_tasks,
    }


def _receipt_signature(receipt: Mapping[str, Any]) -> str:
    return base._receipt_signature(receipt)  # noqa: SLF001 - deliberate thin adapter


def _iter_retrieval_receipts(
    trajectory: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    """Yield lossless QA-memory receipts from successful and failed workers.

    ``AgentRuntime`` stores receipts from completed workers under execution
    records and receipts from a failed bounded ReAct call under the typed
    ``failure_records`` control-plane receipt.  Counting only completed
    executions would incorrectly report zero worker Tool calls for a genuine
    retrieval attempt that ended in coverage or semantic validation failure.
    """

    seen: set[str] = set()
    for round_index, execution_position, execution in base._iter_executions(trajectory):
        metadata = base._mapping(execution.get("metadata"))
        request = base._mapping(metadata.get("request"))
        response = base._mapping(metadata.get("response"))
        request_agent = base._mapping(request.get("agent"))
        for receipt_position, raw_receipt in enumerate(
            base._list(response.get("tool_receipts"))
        ):
            if not isinstance(raw_receipt, Mapping):
                continue
            if raw_receipt.get("tool_id") != QA_MEMORY_TOOL_ID:
                continue
            signature = _receipt_signature(raw_receipt)
            if signature in seen:
                continue
            seen.add(signature)
            action = str(
                base._mapping(raw_receipt.get("request")).get("action", "")
            ).casefold()
            yield {
                "execution_outcome": "completed",
                "artifact_available": (
                    isinstance(execution.get("output"), str)
                    or isinstance(response.get("text"), str)
                ),
                "round_index": round_index,
                "execution_position": execution_position,
                "receipt_position": receipt_position,
                "signature": signature,
                "action": action,
                "agent_id": execution.get("agent_id"),
                "request_agent_id": request_agent.get("id"),
                "request_execution_role": request.get("execution_role"),
                "request_execution_mode": request_agent.get("execution_mode"),
                "request_allowed_tools": base._list(request_agent.get("allowed_tools")),
                "receipt": dict(raw_receipt),
            }

    for turn_position, raw_turn in enumerate(base._list(trajectory.get("turns"))):
        turn = base._mapping(raw_turn)
        round_index = turn.get("round_index", turn_position)
        if not isinstance(round_index, int):
            round_index = turn_position
        snapshot = base._mapping(turn.get("graph_snapshot"))
        nodes = {
            node.get("id"): node
            for node in base._list(snapshot.get("nodes"))
            if isinstance(node, Mapping) and isinstance(node.get("id"), str)
        }
        runtime = base._mapping(turn.get("runtime_summary"))
        for failure_position, raw_failure in enumerate(
            base._list(runtime.get("failure_records"))
        ):
            failure = base._mapping(raw_failure)
            agent_id = failure.get("agent_id")
            agent = base._mapping(nodes.get(agent_id))
            metadata = base._mapping(failure.get("metadata"))
            for receipt_position, raw_receipt in enumerate(
                base._list(metadata.get("tool_receipts"))
            ):
                if not isinstance(raw_receipt, Mapping):
                    continue
                if raw_receipt.get("tool_id") != QA_MEMORY_TOOL_ID:
                    continue
                signature = _receipt_signature(raw_receipt)
                if signature in seen:
                    continue
                seen.add(signature)
                action = str(
                    base._mapping(raw_receipt.get("request")).get("action", "")
                ).casefold()
                yield {
                    "execution_outcome": "failed",
                    "artifact_available": False,
                    "round_index": round_index,
                    "execution_position": failure_position,
                    "receipt_position": receipt_position,
                    "signature": signature,
                    "action": action,
                    "agent_id": agent_id,
                    "request_agent_id": agent_id,
                    "request_execution_role": "worker",
                    "request_execution_mode": agent.get("execution_mode"),
                    "request_allowed_tools": base._list(
                        agent.get("allowed_tools")
                    ),
                    "receipt": dict(raw_receipt),
                }


def _directed_edges(snapshot: object) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    graph = base._mapping(snapshot)
    for raw_relation in base._list(graph.get("relations")):
        relation = base._mapping(raw_relation)
        source = relation.get("source_id")
        target = relation.get("target_id")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if relation.get("source_to_target") is True:
            edges.add((source, target))
        if relation.get("target_to_source") is True:
            edges.add((target, source))
    return edges


def _observed_communication(
    trajectory: Mapping[str, Any],
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    edges: set[tuple[str, str]] = set()
    messages: list[dict[str, Any]] = []
    for round_index, _, execution in base._iter_executions(trajectory):
        request = base._mapping(base._mapping(execution.get("metadata")).get("request"))
        target = execution.get("agent_id")
        for upstream in base._list(request.get("upstream")):
            if not isinstance(upstream, Mapping):
                continue
            source = upstream.get("source_agent_id")
            message_target = upstream.get("target_agent_id", target)
            if not isinstance(source, str) or not isinstance(message_target, str):
                continue
            edges.add((source, message_target))
            messages.append(
                {
                    "round_index": round_index,
                    "source_agent_id": source,
                    "target_agent_id": message_target,
                    "receipt_signatures": [
                        _receipt_signature(receipt)
                        for receipt in base._list(upstream.get("tool_receipts"))
                        if isinstance(receipt, Mapping)
                        and receipt.get("tool_id") == QA_MEMORY_TOOL_ID
                    ],
                    "artifact_type": upstream.get("artifact_type"),
                }
            )
    return edges, messages


def _reachable(
    source: str, target: str, edges: set[tuple[str, str]]
) -> bool:
    queue: deque[str] = deque((source,))
    seen = {source}
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for edge_source, edge_target in edges:
            if edge_source == current and edge_target not in seen:
                seen.add(edge_target)
                queue.append(edge_target)
    return False


def _director_visible_text(trajectory: Mapping[str, Any]) -> str:
    values: list[object] = []
    for raw_turn in base._list(trajectory.get("turns")):
        turn = base._mapping(raw_turn)
        values.extend(
            turn.get(key)
            for key in (
                "prompt",
                "policy_response",
                "action",
                "canvas_feedback",
                "reconstructed_context",
            )
        )
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def _canonical_receipt_data_values(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Extract concrete non-empty QA-memory values, not schema field names."""

    values: dict[str, set[str]] = defaultdict(set)

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                if key in CANONICAL_RECEIPT_DATA_FIELDS and isinstance(child, str):
                    normalized = child.strip()
                    if normalized:
                        values[key].add(normalized)
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    for receipt in receipts:
        visit(receipt)
    return {
        field: sorted(field_values)
        for field, field_values in sorted(values.items())
    }


def _director_input_texts(trajectory: Mapping[str, Any]) -> list[str]:
    """Extract Director-visible inputs without counting staged assistant output.

    The hierarchical Director stores a full provider transcript in ``prompt``.
    Its assistant messages are outputs produced while constructing the current
    Canvas action, not retrieval payload received by the Director.  User
    messages carry the live Canvas observation; static system instructions are
    excluded from retrieval-ingress checks.  Plain prompts remain supported
    for historical trajectories and unit fixtures.
    """

    inputs: list[str] = []
    for raw_turn in base._list(trajectory.get("turns")):
        turn = base._mapping(raw_turn)
        prompt = turn.get("prompt")
        parsed_transcript = False
        if isinstance(prompt, str):
            json_start = prompt.find("{")
            if json_start >= 0:
                try:
                    transcript = json.loads(prompt[json_start:])
                except json.JSONDecodeError:
                    transcript = None
                if isinstance(transcript, Mapping):
                    messages = base._list(transcript.get("messages"))
                    if messages:
                        parsed_transcript = True
                        for raw_message in messages:
                            message = base._mapping(raw_message)
                            if message.get("role") != "user":
                                continue
                            content = message.get("content")
                            if isinstance(content, str):
                                inputs.append(content)
            if not parsed_transcript:
                inputs.append(prompt)
        if prompt is None:
            reconstructed = turn.get("reconstructed_context")
            if reconstructed is not None:
                inputs.append(
                    json.dumps(
                        reconstructed,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
    return inputs


def _labeled_payload_value_present(text: str, field: str, value: str) -> bool:
    """Detect an exact serialized JSON field/value pair, including short values."""

    return re.search(
        rf"{re.escape(json.dumps(field, ensure_ascii=False))}\s*:\s*"
        rf"{re.escape(json.dumps(value, ensure_ascii=False))}",
        text,
    ) is not None


def _director_payload_diagnostics(
    trajectory: Mapping[str, Any],
    receipt_data_values: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return structural exposures and non-causal lexical coincidences.

    Structural exposure is fail-closed for canonical QA-memory identifiers and
    for labeled semantic payload pairs.  Unlabeled semantic overlaps are kept
    as diagnostics because public questions and the Director's own generated
    contracts can legitimately contain the same short answer text.
    """

    director_text = _director_visible_text(trajectory)
    director_input_text = "\n".join(_director_input_texts(trajectory))
    exposures: list[dict[str, str]] = []
    collisions: list[dict[str, str]] = []
    for field, values in receipt_data_values.items():
        for value in values:
            if value not in director_text:
                continue
            item = {"field": field, "value": value}
            appears_in_input = value in director_input_text
            structurally_exposed = appears_in_input and (
                (
                    field in PROVENANCE_IDENTIFIER_FIELDS
                )
                or (
                    field in SEMANTIC_PAYLOAD_FIELDS
                    and _labeled_payload_value_present(
                        director_input_text, field, value
                    )
                )
            )
            (exposures if structurally_exposed else collisions).append(item)
    return exposures, collisions


def _trajectory_control_plane(
    task_id: str, trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    receipts = list(_iter_retrieval_receipts(trajectory))
    director_calls = 0
    for raw_turn in base._list(trajectory.get("turns")):
        action = base._mapping(base._mapping(raw_turn).get("action"))
        if str(action.get("action", "")).casefold() in {"search", "read"}:
            director_calls += 1

    worker_ownership_violations: list[dict[str, Any]] = []
    owner_ids: set[str] = set()
    tool_call_counts_by_agent_id: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_worker_signatures: set[str] = set()
    canonical_worker_receipts: list[Mapping[str, Any]] = []
    for item in receipts:
        agent_id = item.get("agent_id")
        owner_ids.add(str(agent_id))
        tool_call_counts_by_agent_id[str(agent_id)][str(item.get("action"))] += 1
        valid_owner = (
            isinstance(agent_id, str)
            and agent_id
            and item.get("request_agent_id") == agent_id
            and item.get("request_execution_role") == "worker"
            and item.get("request_execution_mode") == "react"
            and QA_MEMORY_TOOL_ID in item.get("request_allowed_tools", [])
        )
        if not valid_owner:
            worker_ownership_violations.append(
                {
                    key: item.get(key)
                    for key in (
                        "round_index",
                        "agent_id",
                        "request_agent_id",
                        "request_execution_role",
                        "request_execution_mode",
                        "request_allowed_tools",
                        "action",
                    )
                }
            )
        else:
            canonical_worker_signatures.add(str(item["signature"]))
            canonical_worker_receipts.append(base._mapping(item.get("receipt")))

    director_text = _director_visible_text(trajectory)
    # Field names are useful diagnostics because contracts may document a Tool
    # schema, but they are not data-plane evidence on their own.
    payload_markers = sorted(
        marker for marker in RETRIEVAL_PAYLOAD_MARKERS if marker in director_text
    )
    receipt_data_values = _canonical_receipt_data_values(canonical_worker_receipts)
    exposed_values, lexical_collisions = _director_payload_diagnostics(
        trajectory, receipt_data_values
    )
    exposed_memory_ids = sorted(
        item["value"] for item in exposed_values if item["field"] == "memory_id"
    )

    graph_edges_by_round: dict[int, set[tuple[str, str]]] = {}
    final_output_id: str | None = None
    for fallback_index, raw_turn in enumerate(base._list(trajectory.get("turns"))):
        turn = base._mapping(raw_turn)
        round_index = turn.get("round_index", fallback_index)
        if not isinstance(round_index, int):
            round_index = fallback_index
        snapshot = base._mapping(turn.get("graph_snapshot"))
        graph_edges_by_round[round_index] = _directed_edges(snapshot)
        output = snapshot.get("output_agent_id")
        if isinstance(output, str) and output:
            final_output_id = output
        runtime_output = base._mapping(turn.get("runtime_summary")).get(
            "output_agent_id"
        )
        if isinstance(runtime_output, str) and runtime_output:
            final_output_id = runtime_output
    output_ids = {final_output_id} if final_output_id is not None else set()
    _, raw_messages = _observed_communication(trajectory)
    messages = [
        message
        for message in raw_messages
        if (
            message["source_agent_id"], message["target_agent_id"]
        )
        in graph_edges_by_round.get(int(message["round_index"]), set())
    ]
    communication_edges = {
        (message["source_agent_id"], message["target_agent_id"])
        for message in messages
    }
    artifact_receipts = [
        item for item in receipts if item.get("artifact_available") is True
    ]
    immediate_routes: list[dict[str, Any]] = []
    for item in artifact_receipts:
        matching = [
            message
            for message in messages
            if item["signature"] in message["receipt_signatures"]
            and message["source_agent_id"] == item["agent_id"]
        ]
        immediate_routes.append(
            {
                "agent_id": item.get("agent_id"),
                "action": item.get("action"),
                "routed": bool(matching),
                "routes": matching,
            }
        )
    artifact_owner_ids = {
        str(item["agent_id"])
        for item in artifact_receipts
        if isinstance(item.get("agent_id"), str)
    }
    output_lineage = all(
        any(_reachable(owner, output, communication_edges) for output in output_ids)
        for owner in artifact_owner_ids
    ) if artifact_owner_ids and output_ids else False
    routed = bool(artifact_receipts) and all(
        route["routed"] for route in immediate_routes
    ) and output_lineage
    output_inbox_messages = [
        message for message in messages if message["target_agent_id"] in output_ids
    ]
    output_inbox_signatures = {
        signature
        for message in output_inbox_messages
        for signature in message["receipt_signatures"]
    }
    canonical_artifact_signatures = {
        str(item["signature"])
        for item in artifact_receipts
        if str(item["signature"]) in canonical_worker_signatures
    }
    missing_output_inbox_signatures = sorted(
        canonical_artifact_signatures - output_inbox_signatures
    )
    output_inbox_receipt_lineage = (
        bool(artifact_receipts)
        and len(canonical_artifact_signatures) == len(artifact_receipts)
        and bool(output_inbox_messages)
        and not missing_output_inbox_signatures
    )

    return {
        "task_id": task_id,
        "explicit_finish": trajectory.get("explicit_finish") is True,
        "director_tool_calls": director_calls,
        "director_allowed_tools": [],
        "director_retrieval_payload_markers": payload_markers,
        "director_retrieval_payload_markers_are_diagnostic_only": True,
        "canonical_worker_receipt_data_values": receipt_data_values,
        "director_exposed_retrieval_values": exposed_values,
        "director_retrieval_value_collisions": lexical_collisions,
        "director_exposed_memory_ids": exposed_memory_ids,
        "retrieval_tool_call_count": len(receipts),
        "retrieval_artifact_receipt_count": len(artifact_receipts),
        "search_count": sum(item["action"] == "search" for item in receipts),
        "read_count": sum(item["action"] == "read" for item in receipts),
        "worker_agent_ids": sorted(owner_ids),
        "tool_call_counts_by_agent_id": {
            agent_id: dict(sorted(counts.items()))
            for agent_id, counts in sorted(tool_call_counts_by_agent_id.items())
        },
        "worker_ownership_violations": worker_ownership_violations,
        "immediate_receipt_routes": immediate_routes,
        "output_agent_ids": sorted(output_ids),
        "observed_communication_edges": [list(edge) for edge in sorted(communication_edges)],
        "retrieval_artifact_routed_via_relation": routed,
        "output_inbox_receipt_lineage": output_inbox_receipt_lineage,
        "output_inbox_message_count": len(output_inbox_messages),
        "output_inbox_canonical_receipt_count": len(
            canonical_artifact_signatures & output_inbox_signatures
        ),
        "output_inbox_missing_canonical_receipt_signatures": (
            missing_output_inbox_signatures
        ),
    }


def _aggregate_control_plane(
    selected_ids: Sequence[str], trajectories: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_task = {
        task_id: _trajectory_control_plane(task_id, trajectories[task_id])
        for task_id in selected_ids
        if task_id in trajectories
    }
    agent_actions: dict[str, Counter[str]] = defaultdict(Counter)
    for value in per_task.values():
        for agent_id, counts in value["tool_call_counts_by_agent_id"].items():
            agent_actions[agent_id].update(counts)
    director_tool_calls = sum(value["director_tool_calls"] for value in per_task.values())
    retrieval_calls = sum(value["retrieval_tool_call_count"] for value in per_task.values())
    ownership_violations = sum(
        len(value["worker_ownership_violations"]) for value in per_task.values()
    )
    payload_exposures = {
        task_id: {
            "markers": value["director_retrieval_payload_markers"],
            "actual_values": value["director_exposed_retrieval_values"],
        }
        for task_id, value in per_task.items()
        if value["director_exposed_retrieval_values"]
    }
    field_name_diagnostics = {
        task_id: value["director_retrieval_payload_markers"]
        for task_id, value in per_task.items()
        if value["director_retrieval_payload_markers"]
    }
    lexical_collision_diagnostics = {
        task_id: value["director_retrieval_value_collisions"]
        for task_id, value in per_task.items()
        if value["director_retrieval_value_collisions"]
    }
    retrieval_tasks = [
        value for value in per_task.values() if value["retrieval_tool_call_count"] > 0
    ]
    artifact_tasks = [
        value
        for value in per_task.values()
        if value["retrieval_artifact_receipt_count"] > 0
    ]
    finished_retrieval_tasks = [
        value for value in retrieval_tasks if value["explicit_finish"] is True
    ]
    unfinished_retrieval_tasks = [
        value for value in retrieval_tasks if value["explicit_finish"] is not True
    ]
    finished_artifact_tasks = [
        value for value in artifact_tasks if value["explicit_finish"] is True
    ]
    unfinished_artifact_tasks = [
        value for value in artifact_tasks if value["explicit_finish"] is not True
    ]
    routed_tasks = [
        value
        for value in finished_artifact_tasks
        if value["retrieval_artifact_routed_via_relation"] is True
    ]
    output_lineage_tasks = [
        value
        for value in finished_artifact_tasks
        if value["output_inbox_receipt_lineage"] is True
    ]
    assertions = {
        "director_tool_calls": director_tool_calls,
        "director_tool_calls_eq_0": director_tool_calls == 0,
        "director_request_allowed_tools": [],
        "director_retrieval_payload_exposure_count": len(payload_exposures),
        "director_data_plane_isolated": director_tool_calls == 0
        and not payload_exposures,
        "retrieval_tool_calls_by_worker": retrieval_calls,
        "retrieval_tool_calls_by_worker_gt_0": retrieval_calls > 0
        and ownership_violations == 0,
        "worker_ownership_violation_count": ownership_violations,
        "retrieval_tasks": len(retrieval_tasks),
        "retrieval_invoked_tasks": len(retrieval_tasks),
        "finished_retrieval_invoked_tasks": len(finished_retrieval_tasks),
        "unfinished_retrieval_invoked_tasks": len(unfinished_retrieval_tasks),
        "retrieval_artifact_tasks": len(artifact_tasks),
        "finished_retrieval_artifact_tasks": len(finished_artifact_tasks),
        "unfinished_retrieval_artifact_tasks": len(unfinished_artifact_tasks),
        "retrieval_tasks_with_relation_route": len(routed_tasks),
        "retrieval_artifact_routed_tasks": len(routed_tasks),
        "retrieval_artifact_routed_via_relation": bool(finished_artifact_tasks)
        and len(routed_tasks) == len(finished_artifact_tasks),
        "retrieval_tasks_with_output_inbox_receipt_lineage": len(
            output_lineage_tasks
        ),
        "output_inbox_receipt_lineage": bool(finished_artifact_tasks)
        and len(output_lineage_tasks) == len(finished_artifact_tasks),
    }
    return (
        {
            "assertions": assertions,
            "worker_tool_calls_by_agent_id": {
                agent_id: dict(sorted(counts.items()))
                for agent_id, counts in sorted(agent_actions.items())
            },
            "director_payload_exposures": payload_exposures,
            "director_field_name_diagnostics": field_name_diagnostics,
            "director_lexical_collision_diagnostics": lexical_collision_diagnostics,
        },
        per_task,
    )


def _failure_category(
    trajectory: Mapping[str, Any], tool_summary: Mapping[str, Any]
) -> str:
    first = base._first_causal_failure(trajectory, tool_summary)
    category = str(first.get("category", "unknown"))
    mapping = {
        "evaluator_or_canonicalization": "evaluator_or_canonicalization",
        "formatting": "formatting",
        "retrieval_read_incomplete": "retrieval_read_incomplete",
        "retrieval_not_invoked": "retrieval_not_invoked",
        "agent_reasoning_or_answer_selection": "reasoning_or_answer_selection",
        "orchestration_tool_contract": "tool_contract",
        "orchestration_relationship": "agent_communication_or_relation",
        "director_action_schema": "director_action_schema",
        "agent_execution": "worker_execution",
        "duplicate_suppression": "react_action_validation",
        "tool_execution": "retrieval_tool_execution",
        "react_validation": "react_action_validation",
        "terminal_max_rounds": "terminal_max_rounds",
        "terminal_other": "terminal_other",
    }
    return mapping.get(category, category)


def _wrong_demos_and_taxonomy(
    selected_by_id: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, dict[str, Any]],
    per_control: Mapping[str, Mapping[str, Any]],
    demo_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[str, str, int]] = []
    counts: Counter[str] = Counter()
    tool_by_task: dict[str, dict[str, Any]] = {}
    for order, task_id in enumerate(selected_by_id):
        trajectory = trajectories.get(task_id)
        if trajectory is None or not base._is_wrong(trajectory):
            continue
        tool_summary = base._tool_summary(trajectory)
        tool_by_task[task_id] = tool_summary
        category = _failure_category(trajectory, tool_summary)
        counts[category] += 1
        candidates.append((task_id, category, order))

    selected: list[str] = []
    category_order = sorted(counts, key=lambda value: (-counts[value], value))
    for category in category_order:
        if len(selected) >= demo_count:
            break
        matches = [item for item in candidates if item[1] == category]
        matches.sort(
            key=lambda item: (
                -int(tool_by_task[item[0]].get("tool_receipt_count", 0)),
                item[2],
            )
        )
        selected.append(matches[0][0])
        if len(selected) >= demo_count:
            break
    for task_id, _, _ in candidates:
        if len(selected) >= demo_count:
            break
        if task_id not in selected:
            selected.append(task_id)

    demos: list[dict[str, Any]] = []
    for task_id in selected:
        demo = base._demo(
            selected_by_id[task_id], trajectories[task_id], tool_by_task[task_id]
        )
        demo["failure_category"] = _failure_category(
            trajectories[task_id], tool_by_task[task_id]
        )
        demo["control_plane_and_relation_receipt"] = per_control.get(task_id)
        demos.append(demo)
    wrong_count = sum(counts.values())
    taxonomy = [
        {
            "category": category,
            "count": count,
            "percentage_of_wrong": round(100.0 * count / wrong_count, 2)
            if wrong_count
            else 0.0,
        }
        for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return demos, taxonomy


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = _resolve(args.selected_tasks)
    trajectory_path = _resolve(args.trajectories)
    paired_path = _resolve(args.paired_results)
    manifest_path = _resolve(args.run_manifest)
    index_manifest_path = _resolve(args.index_manifest)

    selected, selected_diag = base._read_jsonl_snapshot(selected_path)
    trajectory_rows, trajectory_diag = base._read_jsonl_snapshot(trajectory_path)
    paired_rows, paired_diag = base._read_jsonl_snapshot(paired_path)
    trajectories, trajectory_dedup = base._deduplicate_trajectories(trajectory_rows)
    paired, paired_dedup = _deduplicate_rows(paired_rows)
    manifest, manifest_error = base._read_json(manifest_path)
    index_manifest, index_manifest_error = base._read_json(index_manifest_path)
    index_memories_path = index_manifest_path.parent / "memories.jsonl"
    index_memories, index_memories_diag = base._read_jsonl_snapshot(
        index_memories_path
    )

    selected_by_id = {
        task_id: task
        for task in selected
        if isinstance((task_id := task.get("task_id")), str) and task_id
    }
    selected_ids = list(selected_by_id)
    direct = _condition_metrics(selected_ids, paired, "direct")
    agentgraph = _condition_metrics(selected_ids, paired, "agentgraph")
    control, per_control = _aggregate_control_plane(selected_ids, trajectories)
    retrieval_diagnostics = _retrieval_coverage(
        selected_by_id, trajectories, index_memories
    )
    demos, taxonomy = _wrong_demos_and_taxonomy(
        selected_by_id, trajectories, per_control, max(args.demo_count, 0)
    )

    terminal_status = Counter(
        str(trajectories[task_id].get("termination_reason", "missing"))
        for task_id in selected_ids
        if task_id in trajectories
    )
    terminal_failure_ids = [
        task_id
        for task_id in selected_ids
        if task_id in trajectories
        and (
            trajectories[task_id].get("termination_reason") != "finish"
            or trajectories[task_id].get("explicit_finish") is not True
        )
    ]
    run_complete = (
        manifest.get("status") in FINAL_MANIFEST_STATUSES
        and len(selected_ids) > 0
        and len(paired) == len(selected_ids)
        and all(task_id in paired and task_id in trajectories for task_id in selected_ids)
        and direct["evaluator_valid"] == len(selected_ids)
        and agentgraph["evaluator_valid"] == len(selected_ids)
        and not trajectory_diag["malformed_records"]
        and not paired_diag["malformed_records"]
    )
    delta = {
        "exact_match": (
            agentgraph["strict_exact_match"] - direct["strict_exact_match"]
            if agentgraph["strict_exact_match"] is not None
            and direct["strict_exact_match"] is not None
            else None
        ),
        "token_f1": (
            agentgraph["strict_token_f1"] - direct["strict_token_f1"]
            if agentgraph["strict_token_f1"] is not None
            and direct["strict_token_f1"] is not None
            else None
        ),
    }
    official_metrics = (
        {"direct": direct, "agentgraph": agentgraph, "agentgraph_minus_direct": delta}
        if run_complete
        else None
    )
    index_summary = {
        key: index_manifest.get(key)
        for key in (
            "schema_version",
            "tool_id",
            "source_dataset",
            "source_split",
            "memory_count",
            "unique_source_count",
            "cycled_count",
            "paraphrase_count",
            "embedding_model",
            "embedding_dimension",
            "normalization",
            "similarity",
            "frozen_top_k",
            "tool_budget",
            "validation_content_indexed",
            "validation_isolation_count",
        )
    }
    return {
        "schema_version": "flowsteer.triviaqa.qa-memory.formal-result-analysis.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "offline_persisted_receipts_only",
        "run": {
            "status": "complete" if run_complete else "partial",
            "manifest_status": manifest.get("status"),
            "sample_count": len(selected_ids),
            "paired_count": len(paired),
            "trajectory_count": sum(task_id in trajectories for task_id in selected_ids),
            "official_metrics": official_metrics,
            "manifest_read_error": manifest_error,
            "index_manifest_read_error": index_manifest_error,
        },
        "metrics": {
            "metric_protocol": "triviaqa.official.answer.v1",
            "direct": direct,
            "agentgraph": agentgraph,
            "agentgraph_minus_direct": delta,
            "scope": (
                "official fixed held-out validation" if run_complete else
                "partial snapshot; strict denominator remains the selected task count"
            ),
        },
        "terminal": {
            "failure_count": len(terminal_failure_ids),
            "failure_task_ids": terminal_failure_ids,
            "termination_reason_counts": dict(sorted(terminal_status.items())),
        },
        "qa_memory_index": index_summary,
        "retrieval_diagnostics": retrieval_diagnostics,
        "control_plane_and_tool_routing": control,
        "failure_taxonomy": taxonomy,
        "wrong_demo_selection": {
            "requested_count": args.demo_count,
            "actual_count": len(demos),
            "minimum_formal_count_met": len(demos) >= 3,
            "shortfall": max(3 - len(demos), 0),
            "no_fabrication_policy": "only persisted evaluator-wrong trajectories are eligible",
        },
        "wrong_demos": demos,
        "inputs": {
            "selected_tasks": selected_diag,
            "trajectories": trajectory_diag,
            "paired_results": paired_diag,
            "run_manifest": base._display_path(manifest_path),
            "index_manifest": base._display_path(index_manifest_path),
            "index_memories": index_memories_diag,
            "trajectory_deduplication": trajectory_dedup,
            "paired_deduplication": paired_dedup,
        },
    }


def _percentage(value: object) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{100.0 * number:.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    run = base._mapping(report.get("run"))
    metrics = base._mapping(report.get("metrics"))
    direct = base._mapping(metrics.get("direct"))
    graph = base._mapping(metrics.get("agentgraph"))
    delta = base._mapping(metrics.get("agentgraph_minus_direct"))
    retrieval = base._mapping(report.get("retrieval_diagnostics"))
    terminal = base._mapping(report.get("terminal"))
    control = base._mapping(report.get("control_plane_and_tool_routing"))
    assertions = base._mapping(control.get("assertions"))
    lines = [
        "# TriviaQA QA-memory 正式结果分析",
        "",
        f"状态：**{run.get('status')}**；manifest：`{run.get('manifest_status')}`。本报告仅分析已落盘 paired result、trajectory、Tool receipt 与 evaluator receipt，不调用模型或评测服务。",
        "",
        "## Direct / AgentGraph 指标",
        "",
        "| 条件 | 固定分母 | 完成 | evaluator 有效 | 严格 EM | 严格 F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Direct | {direct.get('denominator')} | {direct.get('completed')} | {direct.get('evaluator_valid')} | {_percentage(direct.get('strict_exact_match'))} | {_percentage(direct.get('strict_token_f1'))} |",
        f"| AgentGraph | {graph.get('denominator')} | {graph.get('completed')} | {graph.get('evaluator_valid')} | {_percentage(graph.get('strict_exact_match'))} | {_percentage(graph.get('strict_token_f1'))} |",
        "",
        f"AgentGraph − Direct：**{_percentage(delta.get('exact_match'))} EM**，**{_percentage(delta.get('token_f1'))} F1**。partial 状态下该值仅是固定分母 fail-closed snapshot，不是完整 128 条正式结果。",
        "",
        "## Worker QA-memory Tool 与 Answer Recall",
        "",
        f"- 唯一物理 Tool receipts：**{retrieval.get('tool_call_count')}**（search={retrieval.get('search_call_count')}，read={retrieval.get('read_call_count')}；每题均值={retrieval.get('mean_tool_calls_per_task')}）。",
        f"- 成功 search/read：**{retrieval.get('successful_search_call_count')}/{retrieval.get('successful_read_call_count')}**；非空 search：**{retrieval.get('nonempty_search_call_count')}**；返回候选：**{retrieval.get('search_candidate_count')}**。",
        f"- 发生 search/read 的任务：**{retrieval.get('tasks_with_search_count')} / {retrieval.get('tasks_with_read_count')}**；有多条成功规范化 query 的任务：**{retrieval.get('tasks_with_multiple_successful_queries_count')}**。",
        f"- 512 条 corpus accepted-answer match：**{retrieval.get('corpus_accepted_answer_match_count')}/{retrieval.get('sample_count')} = {_percentage(retrieval.get('corpus_accepted_answer_match_rate'))}**。",
        f"- 实际 search Recall@1：**{retrieval.get('search_top1_accepted_answer_match_count')}/{retrieval.get('sample_count')} = {_percentage(retrieval.get('search_top1_accepted_answer_match_rate'))}**。",
        f"- 实际 search Recall@3：**{retrieval.get('search_candidate_accepted_answer_match_count')}/{retrieval.get('sample_count')} = {_percentage(retrieval.get('search_candidate_accepted_answer_match_rate'))}**；在 corpus 可覆盖任务内为 **{_percentage(retrieval.get('search_candidate_recall_within_corpus_covered'))}**。",
        f"- 实际 read accepted-answer match：**{retrieval.get('read_candidate_accepted_answer_match_count')}/{retrieval.get('sample_count')} = {_percentage(retrieval.get('read_candidate_accepted_answer_match_rate'))}**。",
        "- 上述 accepted answers 仅在冻结运行完成后用于离线 Answer Recall；它们未进入 query、Tool observation、Agent request 或 Director request。规范化命中不保证 target relation 绑定正确。",
        "",
        "## Terminal 与三项边界断言",
        "",
        f"- terminal failure：**{terminal.get('failure_count')}**；termination：`{json.dumps(terminal.get('termination_reason_counts'), ensure_ascii=False, sort_keys=True)}`。",
        f"- `director_tool_calls=0`：**{assertions.get('director_tool_calls_eq_0')}**（实测 {assertions.get('director_tool_calls')}）。",
        f"- `retrieval_tool_calls_by_worker>0`：**{assertions.get('retrieval_tool_calls_by_worker_gt_0')}**（实测 {assertions.get('retrieval_tool_calls_by_worker')}；ownership violations={assertions.get('worker_ownership_violation_count')}）。",
        f"- `retrieval_artifact_routed_via_relation=true`：**{assertions.get('retrieval_artifact_routed_via_relation')}**（{assertions.get('retrieval_artifact_routed_tasks')}/{assertions.get('finished_retrieval_artifact_tasks')} explicit-FINISH retrieval artifact tasks；unfinished={assertions.get('unfinished_retrieval_invoked_tasks')}）。",
        f"- Director 数据面隔离：**{assertions.get('director_data_plane_isolated')}**（canonical receipt actual-value exposure={assertions.get('director_retrieval_payload_exposure_count')}；字段名仅作诊断）。",
        f"- Output inbox receipt lineage（独立强断言）：**{assertions.get('output_inbox_receipt_lineage')}**（{assertions.get('retrieval_tasks_with_output_inbox_receipt_lineage')}/{assertions.get('finished_retrieval_artifact_tasks')} explicit-FINISH retrieval artifact tasks）。",
        "",
        "### Worker Agent Tool ownership",
        "",
        base._json_block(control.get("worker_tool_calls_by_agent_id", {})),
        "",
        "## Failure taxonomy",
        "",
        "| 类别 | 数量 | 错误样本占比 |",
        "| --- | ---: | ---: |",
    ]
    for row in base._list(report.get("failure_taxonomy")):
        item = base._mapping(row)
        lines.append(
            f"| {item.get('category')} | {item.get('count')} | {item.get('percentage_of_wrong')}% |"
        )
    demos = base._list(report.get("wrong_demos"))
    lines.extend(["", "## 逐步 Wrong Demo", ""])
    for index, raw_demo in enumerate(demos, start=1):
        demo = base._mapping(raw_demo)
        input_value = base._mapping(demo.get("input"))
        target = base._mapping(demo.get("reference_target"))
        result = base._mapping(demo.get("system_final_output_and_metrics"))
        lines.extend(
            [
                f"### Demo {index}: `{demo.get('task_id')}` — `{demo.get('failure_category')}`",
                "",
                f"- Question：{input_value.get('question')}",
                f"- Reference：`{target.get('ground_truth')}`",
                f"- Final：`{result.get('final_answer')}`；metrics：`{json.dumps(result.get('metrics'), ensure_ascii=False, sort_keys=True)}`",
                "",
                "<details><summary>Director → Canvas → Agent communication → ReAct Tool → Terminal/Evaluator</summary>",
                "",
                base._json_block(
                    {
                        "first_causal_failure": demo.get("first_causal_failure"),
                        "control_plane_and_relation_receipt": demo.get(
                            "control_plane_and_relation_receipt"
                        ),
                        "actual_execution_chain": demo.get("actual_execution_chain"),
                        "downstream_error_propagation": demo.get(
                            "downstream_error_propagation"
                        ),
                        "terminal_receipt": demo.get("terminal_receipt"),
                        "evaluator_receipt": demo.get("evaluator_receipt"),
                    }
                ),
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks", default=str(DEFAULT_ARTIFACT_DIR / "selected_tasks.jsonl")
    )
    parser.add_argument(
        "--trajectories",
        default=str(DEFAULT_ARTIFACT_DIR / "agentgraph_trajectories.jsonl"),
    )
    parser.add_argument(
        "--paired-results",
        default=str(DEFAULT_ARTIFACT_DIR / "paired_results.jsonl"),
    )
    parser.add_argument(
        "--run-manifest", default=str(DEFAULT_ARTIFACT_DIR / "run_manifest.json")
    )
    parser.add_argument(
        "--index-manifest",
        default=str(DEFAULT_ARTIFACT_DIR / "index/manifest.json"),
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_REPORT_DIR / "formal_result_analysis.json"),
    )
    parser.add_argument(
        "--output-markdown",
        default=str(DEFAULT_REPORT_DIR / "formal_result_analysis.md"),
    )
    parser.add_argument("--demo-count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.demo_count < 0:
        parser.error("--demo-count must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json = _resolve(args.output_json)
    output_markdown = _resolve(args.output_markdown)
    base._atomic_write_text(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    base._atomic_write_text(output_markdown, _render_markdown(report))
    print(
        json.dumps(
            {
                "status": report["run"]["status"],
                "paired": report["run"]["paired_count"],
                "trajectories": report["run"]["trajectory_count"],
                "wrong_demos": report["wrong_demo_selection"]["actual_count"],
                "assertions": report["control_plane_and_tool_routing"]["assertions"],
                "output_json": base._display_path(output_json),
                "output_markdown": base._display_path(output_markdown),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
