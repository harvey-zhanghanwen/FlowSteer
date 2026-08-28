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
DIRECTOR_TRANSCRIPT_HEADER = "Flow-Director chat transcript"
DIRECTOR_TRANSCRIPT_SCHEMA = "flowsteer.director.transcript.v1"
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
# A bare answer span is not sufficient evidence of QA-memory injection: it may
# already occur in the public task, numeric graph state, or the Director's own
# prior structured action.  Record identity and paraphrase fields are
# provenance-bearing data-plane values; a serialized QA-memory record also
# exposes at least one of these fields.  ``canonical_answer`` remains part of
# receipt lineage matching and field-name diagnostics below.
DIRECTOR_PROVENANCE_BEARING_DATA_FIELDS = frozenset(
    {
        "memory_id",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "source_train_task_id",
        "base_task_id",
    }
)
QA_MEMORY_BATCH_ARTIFACT_FIELDS = (
    "question_scope",
    "retrieval_query",
    "top_k",
    "candidates",
)
QA_MEMORY_BATCH_CANDIDATE_FIELDS = (
    "rank",
    "similarity",
    "memory_id",
    "source_train_task_id",
    "paraphrase_question",
    "paraphrase_answer_statement",
    "canonical_answer",
)
QA_MEMORY_READ_RECORD_FIELDS = QA_MEMORY_BATCH_CANDIDATE_FIELDS[3:]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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
                "request_agent_role_family": request_agent.get("role_family"),
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
                    "request_agent_role_family": agent.get("role_family"),
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
        communication_envelopes = list(base._list(request.get("upstream")))
        # FlowSteer's bidirectional exchange stores the peer-to-peer envelope
        # separately from ordinary predecessor messages.  It is still an
        # observed AgentGraph communication with the same receipt schema and
        # must participate in relation/lineage verification.
        peer_draft = request.get("peer_draft")
        if isinstance(peer_draft, Mapping):
            communication_envelopes.append(peer_draft)
        for upstream in communication_envelopes:
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
    """Return only persisted Director request text.

    ``policy_response``, parsed ``action``, Canvas feedback, and reconstructed
    context are runtime outputs or derived diagnostics.  Treating those fields
    as Director input confounds a model-emitted answer span with data-plane
    payload injection.  The canonical request is persisted in ``prompt`` and
    is independently schema-checked by ``_director_execution_profiles``.
    """

    values: list[object] = []
    for raw_turn in base._list(trajectory.get("turns")):
        turn = base._mapping(raw_turn)
        values.append(turn.get("prompt"))
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def _director_execution_profiles(
    trajectory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read each Director Tool profile from its persisted canonical request.

    The analyzer intentionally does not import Director/runtime code.  A
    missing, legacy, or malformed prompt is therefore an assertion failure,
    not evidence that the Director request was Tool-free.
    """

    profiles: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for fallback_index, raw_turn in enumerate(base._list(trajectory.get("turns"))):
        turn = base._mapping(raw_turn)
        round_index = turn.get("round_index", fallback_index)
        if not isinstance(round_index, int):
            round_index = fallback_index
        prompt = turn.get("prompt")
        reason: str | None = None
        profile: Mapping[str, Any] | None = None
        if not isinstance(prompt, str) or not prompt.startswith(
            DIRECTOR_TRANSCRIPT_HEADER + "\n\n"
        ):
            reason = "missing_canonical_director_transcript"
        else:
            try:
                payload = json.loads(prompt.partition("\n\n")[2])
            except (TypeError, ValueError):
                payload = None
                reason = "malformed_director_transcript_json"
            if reason is None and (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != DIRECTOR_TRANSCRIPT_SCHEMA
            ):
                reason = "unsupported_director_transcript_schema"
            messages = (
                payload.get("messages") if isinstance(payload, Mapping) else None
            )
            if reason is None and (
                not isinstance(messages, list) or not messages
            ):
                reason = "missing_current_director_observation"
            if reason is None:
                # FlowSteer's staged structured actions append a second user
                # message asking the Director to complete ADD_SUBGRAPH fields.
                # That message is part of the same Director request, but is not
                # a new Canvas observation.  Read the most recent canonical
                # Canvas observation instead of assuming messages[-1].
                observation_message = next(
                    (
                        message
                        for message in reversed(messages)
                        if isinstance(message, Mapping)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                        and str(message["content"]).lstrip().startswith(
                            "Canvas observation"
                        )
                    ),
                    None,
                )
                if observation_message is None:
                    reason = "missing_current_director_observation"
                    content = ""
                else:
                    content = str(observation_message["content"])
                _, separator, encoded = content.partition("\n\n")
                if reason is None and not separator:
                    reason = "missing_director_observation_payload"
                elif reason is None:
                    try:
                        observation = json.loads(encoded)
                    except (TypeError, ValueError):
                        observation = None
                        reason = "malformed_director_observation_json"
                    if isinstance(observation, Mapping):
                        raw_profile = observation.get(
                            "director_execution_profile"
                        )
                        if isinstance(raw_profile, Mapping):
                            profile = raw_profile
                        else:
                            reason = "missing_director_execution_profile"
            if reason is None and profile is not None:
                allowed_tools = profile.get("allowed_tools")
                tool_calls_enabled = profile.get("tool_calls_enabled")
                if (
                    not isinstance(allowed_tools, list)
                    or any(not isinstance(tool_id, str) for tool_id in allowed_tools)
                    or type(tool_calls_enabled) is not bool
                ):
                    reason = "invalid_director_execution_profile"
                else:
                    profiles.append(
                        {
                            "round_index": round_index,
                            "allowed_tools": list(allowed_tools),
                            "tool_calls_enabled": tool_calls_enabled,
                        }
                    )
        if reason is not None:
            violations.append(
                {"round_index": round_index, "reason": reason}
            )
    return profiles, violations


def _reasoner_qamemory_assignment_violations(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Report actual Canvas/request assignments of QA-memory to a Reasoner."""

    assignments: dict[tuple[int, str], dict[str, Any]] = {}
    for fallback_index, raw_turn in enumerate(base._list(trajectory.get("turns"))):
        turn = base._mapping(raw_turn)
        round_index = turn.get("round_index", fallback_index)
        if not isinstance(round_index, int):
            round_index = fallback_index
        snapshot = base._mapping(turn.get("graph_snapshot"))
        for raw_node in base._list(snapshot.get("nodes")):
            node = base._mapping(raw_node)
            agent_id = node.get("id")
            if (
                isinstance(agent_id, str)
                and str(node.get("role_family", "")).casefold() == "reasoner"
                and QA_MEMORY_TOOL_ID in base._list(node.get("allowed_tools"))
            ):
                assignments.setdefault(
                    (round_index, agent_id),
                    {
                        "round_index": round_index,
                        "agent_id": agent_id,
                        "execution_mode": node.get("execution_mode"),
                        "allowed_tools": base._list(node.get("allowed_tools")),
                        "observed_in": [],
                    },
                )["observed_in"].append("graph_snapshot")
        for raw_execution in base._list(turn.get("executions")):
            execution = base._mapping(raw_execution)
            request = base._mapping(
                base._mapping(execution.get("metadata")).get("request")
            )
            agent = base._mapping(request.get("agent"))
            agent_id = agent.get("id", execution.get("agent_id"))
            if (
                isinstance(agent_id, str)
                and str(agent.get("role_family", "")).casefold() == "reasoner"
                and QA_MEMORY_TOOL_ID in base._list(agent.get("allowed_tools"))
            ):
                assignments.setdefault(
                    (round_index, agent_id),
                    {
                        "round_index": round_index,
                        "agent_id": agent_id,
                        "execution_mode": agent.get("execution_mode"),
                        "allowed_tools": base._list(agent.get("allowed_tools")),
                        "observed_in": [],
                    },
                )["observed_in"].append("execution_request")
    return [assignments[key] for key in sorted(assignments)]


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


def _native_artifact_receipt_projections(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify singular or ordered top-k worker artifacts against Tool receipts.

    The legacy wire selects one ``memory_id``.  The current wire selects all
    ``memory_ids`` from one embedding search and projects an ordered candidate
    batch only after one matching read per rank.  This analyzer mirrors that
    receipt boundary without importing runtime code or reconstructing content.
    """

    results: list[dict[str, Any]] = []
    compared_fields = CANONICAL_RECEIPT_DATA_FIELDS[:5]
    for round_index, execution_position, execution in base._iter_executions(
        trajectory
    ):
        metadata = base._mapping(execution.get("metadata"))
        request = base._mapping(metadata.get("request"))
        response = base._mapping(metadata.get("response"))
        agent = base._mapping(request.get("agent"))
        if agent.get("role_family") != "evidence_retriever":
            continue
        output = execution.get("output", response.get("text"))
        if not isinstance(output, str):
            continue
        try:
            artifact = json.loads(output)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(artifact, Mapping):
            continue

        batch_wire = any(
            field in artifact for field in ("retrieval_query", "top_k", "candidates")
        )
        if batch_wire:
            selected_memory_ids: list[str] | None = None
            for raw_trace in reversed(base._list(response.get("react_trace"))):
                trace = base._mapping(raw_trace)
                action = base._mapping(trace.get("structured_action"))
                if action.get("kind") != "complete":
                    continue
                value = base._mapping(
                    base._mapping(action.get("arguments")).get("value")
                )
                raw_selected = value.get("memory_ids")
                if (
                    isinstance(raw_selected, list)
                    and raw_selected
                    and all(
                        isinstance(memory_id, str)
                        and bool(memory_id.strip())
                        and memory_id == memory_id.strip()
                        for memory_id in raw_selected
                    )
                    and len(raw_selected) == len(set(raw_selected))
                ):
                    selected_memory_ids = list(raw_selected)
                break

            successful_searches: list[dict[str, Any]] = []
            qa_tool_action_count = 0
            raw_tool_receipts = base._list(response.get("tool_receipts"))
            for receipt_position, raw_receipt in enumerate(raw_tool_receipts):
                receipt = base._mapping(raw_receipt)
                if receipt.get("tool_id") != QA_MEMORY_TOOL_ID:
                    continue
                receipt_request = base._mapping(receipt.get("request"))
                action_name = receipt_request.get("action")
                if action_name not in {"search", "read"}:
                    continue
                qa_tool_action_count += 1
                if action_name != "search" or receipt.get("error_type") is not None:
                    continue
                receipt_result = base._mapping(receipt.get("result"))
                value = base._mapping(receipt_result.get("value", receipt_result))
                arguments = base._mapping(receipt_request.get("arguments"))
                raw_memory_ids = value.get("memory_ids")
                raw_hits = value.get("hits")
                if (
                    receipt_result.get("completed") is not True
                    or value.get("operation") != "search"
                    or set(arguments) != {"query", "limit"}
                    or not isinstance(arguments.get("query"), str)
                    or not arguments["query"].strip()
                    or type(arguments.get("limit")) is not int
                    or arguments["limit"] < 1
                    or value.get("query") != arguments["query"]
                    or value.get("top_k") != arguments["limit"]
                    or not isinstance(raw_memory_ids, list)
                    or not raw_memory_ids
                    or len(raw_memory_ids) != len(set(raw_memory_ids))
                    or not isinstance(raw_hits, list)
                    or len(raw_hits) != len(raw_memory_ids)
                ):
                    continue
                successful_searches.append(
                    {
                        "receipt_position": receipt_position,
                        "query": arguments.get("query"),
                        "limit": arguments.get("limit"),
                        "result_query": value.get("query"),
                        "result_top_k": value.get("top_k"),
                        "memory_ids": list(raw_memory_ids),
                        "hits": list(raw_hits),
                    }
                )

            latest_search = successful_searches[-1] if successful_searches else None
            search_memory_ids = (
                list(latest_search["memory_ids"]) if latest_search is not None else []
            )
            search_hits = (
                list(latest_search["hits"]) if latest_search is not None else []
            )
            ordered_reads: list[dict[str, Any]] = []
            if latest_search is not None:
                for receipt_position, raw_receipt in enumerate(raw_tool_receipts):
                    if receipt_position <= latest_search["receipt_position"]:
                        continue
                    receipt = base._mapping(raw_receipt)
                    receipt_request = base._mapping(receipt.get("request"))
                    if (
                        receipt.get("tool_id") != QA_MEMORY_TOOL_ID
                        or receipt.get("error_type") is not None
                        or receipt_request.get("action") != "read"
                    ):
                        continue
                    receipt_result = base._mapping(receipt.get("result"))
                    if receipt_result.get("completed") is not True:
                        continue
                    arguments = base._mapping(receipt_request.get("arguments"))
                    value = base._mapping(
                        receipt_result.get("value", receipt_result)
                    )
                    memory = base._mapping(value.get("memory"))
                    memory_id = arguments.get("memory_id")
                    if (
                        set(arguments) != {"memory_id"}
                        or not isinstance(memory_id, str)
                        or not memory_id.strip()
                        or memory_id != memory_id.strip()
                        or value.get("operation") != "read"
                        or value.get("memory_id") != memory_id
                        or memory.get("memory_id") != memory_id
                        or not isinstance(memory.get("text"), str)
                        or not memory["text"].strip()
                    ):
                        continue
                    ordered_reads.append(
                        {
                            "receipt_position": receipt_position,
                            "memory_id": memory_id,
                            "memory": memory,
                        }
                    )

            raw_candidates = artifact.get("candidates")
            candidates = list(raw_candidates) if isinstance(raw_candidates, list) else []
            artifact_top_k = artifact.get("top_k")
            valid_top_k = (
                type(artifact_top_k) is int and artifact_top_k > 0
            )
            expected_top_k = artifact_top_k if valid_top_k else None
            read_memory_ids = [item.get("memory_id") for item in ordered_reads]
            candidate_memory_ids = [
                candidate.get("memory_id") if isinstance(candidate, Mapping) else None
                for candidate in candidates
            ]
            search_hit_memory_ids = [
                hit.get("memory_id") if isinstance(hit, Mapping) else None
                for hit in search_hits
            ]

            candidate_diagnostics: list[dict[str, Any]] = []
            previous_hit_rank = 0
            previous_candidate_rank = 0
            for candidate_index, raw_candidate in enumerate(candidates):
                candidate = base._mapping(raw_candidate)
                hit = (
                    base._mapping(search_hits[candidate_index])
                    if candidate_index < len(search_hits)
                    else {}
                )
                read = (
                    ordered_reads[candidate_index]
                    if candidate_index < len(ordered_reads)
                    else {}
                )
                memory = base._mapping(read.get("memory"))
                mismatched_fields: list[str] = []
                if set(candidate) != set(QA_MEMORY_BATCH_CANDIDATE_FIELDS):
                    mismatched_fields.append("candidate_schema")
                hit_rank = hit.get("rank")
                hit_similarity = hit.get("similarity")
                if type(hit_rank) is not int or hit_rank <= previous_hit_rank:
                    mismatched_fields.append("search_hit.rank_valid")
                else:
                    previous_hit_rank = hit_rank
                if isinstance(hit_similarity, bool) or not isinstance(
                    hit_similarity, (int, float)
                ):
                    mismatched_fields.append("search_hit.similarity_valid")
                candidate_rank = candidate.get("rank")
                candidate_similarity = candidate.get("similarity")
                if (
                    type(candidate_rank) is not int
                    or candidate_rank <= previous_candidate_rank
                ):
                    mismatched_fields.append("candidate.rank_valid")
                else:
                    previous_candidate_rank = candidate_rank
                if isinstance(candidate_similarity, bool) or not isinstance(
                    candidate_similarity, (int, float)
                ):
                    mismatched_fields.append("candidate.similarity_valid")
                for field_name in ("memory_id", "rank", "similarity"):
                    if candidate.get(field_name) != hit.get(field_name):
                        mismatched_fields.append(f"search_hit.{field_name}")
                if read.get("memory_id") != candidate.get("memory_id"):
                    mismatched_fields.append("read_request.memory_id")
                if memory.get("memory_id") != candidate.get("memory_id"):
                    mismatched_fields.append("read_record.memory_id")
                for field_name in QA_MEMORY_READ_RECORD_FIELDS:
                    if candidate.get(field_name) != memory.get(field_name):
                        mismatched_fields.append(f"read_record.{field_name}")
                candidate_diagnostics.append(
                    {
                        "candidate_index": candidate_index,
                        "memory_id": candidate.get("memory_id"),
                        "rank": candidate.get("rank"),
                        "similarity": candidate.get("similarity"),
                        "mismatched_receipt_fields": mismatched_fields,
                        "receipt_exact": not mismatched_fields,
                    }
                )

            question = base._mapping(trajectory.get("task")).get("question")
            artifact_fields_exact = set(artifact) == set(
                QA_MEMORY_BATCH_ARTIFACT_FIELDS
            )
            search_query_matches = bool(
                latest_search is not None
                and artifact.get("retrieval_query") == latest_search.get("query")
                and latest_search.get("result_query") == latest_search.get("query")
            )
            search_top_k_matches = bool(
                valid_top_k
                and latest_search is not None
                and latest_search.get("limit") == artifact_top_k
                and latest_search.get("result_top_k") == artifact_top_k
            )
            ordered_ids_match = bool(
                selected_memory_ids is not None
                and selected_memory_ids == search_memory_ids
                and search_memory_ids == search_hit_memory_ids
                and search_hit_memory_ids == candidate_memory_ids
                and candidate_memory_ids == read_memory_ids
                and len(candidate_memory_ids) == len(set(candidate_memory_ids))
            )
            one_search_plus_k_reads = bool(
                valid_top_k
                and len(successful_searches) == 1
                and len(ordered_reads) == artifact_top_k
                and qa_tool_action_count == artifact_top_k + 1
            )
            k_complete = bool(
                valid_top_k
                and len(candidates) == artifact_top_k
                and len(search_memory_ids) == artifact_top_k
                and len(search_hits) == artifact_top_k
                and len(ordered_reads) == artifact_top_k
                and selected_memory_ids is not None
                and len(selected_memory_ids) == artifact_top_k
                and ordered_ids_match
                and one_search_plus_k_reads
            )
            rank_similarity_exact = bool(candidate_diagnostics) and all(
                not any(
                    field in {
                        "search_hit.rank",
                        "search_hit.similarity",
                        "search_hit.rank_valid",
                        "search_hit.similarity_valid",
                        "candidate.rank_valid",
                        "candidate.similarity_valid",
                    }
                    for field in item["mismatched_receipt_fields"]
                )
                for item in candidate_diagnostics
            )
            read_records_exact = bool(candidate_diagnostics) and all(
                not any(
                    field.startswith("read_")
                    for field in item["mismatched_receipt_fields"]
                )
                for item in candidate_diagnostics
            )
            receipt_exact = bool(
                artifact_fields_exact
                and (not isinstance(question, str) or artifact.get("question_scope") == question)
                and search_query_matches
                and search_top_k_matches
                and k_complete
                and candidate_diagnostics
                and all(item["receipt_exact"] for item in candidate_diagnostics)
            )
            results.append(
                {
                    "round_index": round_index,
                    "execution_position": execution_position,
                    "agent_id": agent.get("id", execution.get("agent_id")),
                    "projection_kind": "ordered_top_k_batch",
                    "selected_memory_ids": selected_memory_ids,
                    "search_memory_ids": search_memory_ids,
                    "candidate_memory_ids": candidate_memory_ids,
                    "read_memory_ids": read_memory_ids,
                    "expected_top_k": expected_top_k,
                    "completion_memory_id_count": (
                        len(selected_memory_ids)
                        if selected_memory_ids is not None
                        else 0
                    ),
                    "search_hit_count": len(search_hits),
                    "artifact_candidate_count": len(candidates),
                    "successful_read_count": len(ordered_reads),
                    "successful_search_count": len(successful_searches),
                    "qa_tool_action_count": qa_tool_action_count,
                    "artifact_fields_exact": artifact_fields_exact,
                    "question_scope_matches_task": (
                        not isinstance(question, str)
                        or artifact.get("question_scope") == question
                    ),
                    "search_query_matches_artifact": search_query_matches,
                    "search_top_k_matches_artifact": search_top_k_matches,
                    "ordered_memory_ids_match": ordered_ids_match,
                    "one_search_plus_k_ordered_reads": one_search_plus_k_reads,
                    "candidate_receipt_diagnostics": candidate_diagnostics,
                    "rank_similarity_exact": rank_similarity_exact,
                    "read_records_exact": read_records_exact,
                    "k_completeness_applicable": True,
                    "k_complete": k_complete,
                    "receipt_exact": receipt_exact,
                }
            )
            continue

        memory_id = artifact.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            continue

        selected_memory_id: str | None = None
        for raw_trace in reversed(base._list(response.get("react_trace"))):
            trace = base._mapping(raw_trace)
            action = base._mapping(trace.get("structured_action"))
            if action.get("kind") != "complete":
                continue
            value = base._mapping(base._mapping(action.get("arguments")).get("value"))
            selected = value.get("memory_id")
            if isinstance(selected, str) and selected:
                selected_memory_id = selected
            break

        searched_ids: set[str] = set()
        exact_read: Mapping[str, Any] | None = None
        for raw_receipt in base._list(response.get("tool_receipts")):
            receipt = base._mapping(raw_receipt)
            if (
                receipt.get("tool_id") != QA_MEMORY_TOOL_ID
                or receipt.get("error_type") is not None
            ):
                continue
            receipt_request = base._mapping(receipt.get("request"))
            receipt_result = base._mapping(receipt.get("result"))
            value = base._mapping(receipt_result.get("value", receipt_result))
            if receipt_request.get("action") == "search":
                if receipt_result.get("completed") is True:
                    searched_ids.update(
                        item
                        for item in base._list(value.get("memory_ids"))
                        if isinstance(item, str) and item
                    )
                continue
            arguments = base._mapping(receipt_request.get("arguments"))
            if (
                receipt_request.get("action") == "read"
                and memory_id in searched_ids
                and arguments.get("memory_id") == memory_id
                and receipt_result.get("completed") is True
            ):
                memory = base._mapping(value.get("memory"))
                if memory.get("memory_id") == memory_id:
                    exact_read = memory
                    break

        mismatched_fields = [
            field
            for field in compared_fields
            if exact_read is None or artifact.get(field) != exact_read.get(field)
        ]
        results.append(
            {
                "round_index": round_index,
                "execution_position": execution_position,
                "agent_id": agent.get("id", execution.get("agent_id")),
                "projection_kind": "legacy_singular",
                "memory_id": memory_id,
                "selected_memory_id": selected_memory_id,
                "selection_matches_artifact": selected_memory_id == memory_id,
                "exact_search_read_receipt_found": exact_read is not None,
                "mismatched_receipt_fields": mismatched_fields,
                "k_completeness_applicable": False,
                "k_complete": None,
                "receipt_exact": bool(
                    selected_memory_id == memory_id
                    and exact_read is not None
                    and not mismatched_fields
                ),
            }
        )
    return results


def _trajectory_control_plane(
    task_id: str, trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    receipts = list(_iter_retrieval_receipts(trajectory))
    native_artifact_projections = _native_artifact_receipt_projections(
        trajectory
    )
    native_top_k_batch_projections = [
        item
        for item in native_artifact_projections
        if item.get("projection_kind") == "ordered_top_k_batch"
    ]
    director_profiles, director_profile_violations = (
        _director_execution_profiles(trajectory)
    )
    director_allowed_tools = sorted(
        {
            tool_id
            for profile in director_profiles
            for tool_id in profile["allowed_tools"]
        }
    )
    director_tool_enabled_count = sum(
        profile["tool_calls_enabled"] is True for profile in director_profiles
    )
    reasoner_tool_violations = _reasoner_qamemory_assignment_violations(
        trajectory
    )
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
            and item.get("request_agent_role_family") == "evidence_retriever"
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
                        "request_agent_role_family",
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
    exposed_values = [
        {"field": field, "value": value}
        for field, values in receipt_data_values.items()
        if field in DIRECTOR_PROVENANCE_BEARING_DATA_FIELDS
        for value in values
        if value in director_text
    ]
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
        "director_tool_calls": director_calls,
        "director_execution_profiles": director_profiles,
        "director_execution_profile_violations": director_profile_violations,
        "director_allowed_tools": director_allowed_tools,
        "director_tool_calls_enabled_count": director_tool_enabled_count,
        "director_retrieval_payload_markers": payload_markers,
        "director_retrieval_payload_markers_are_diagnostic_only": True,
        "canonical_worker_receipt_data_values": receipt_data_values,
        "director_exposed_retrieval_values": exposed_values,
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
        "reasoner_qamemory_tool_assignment_violations": (
            reasoner_tool_violations
        ),
        "native_artifact_receipt_projections": native_artifact_projections,
        "native_artifact_receipt_projection_count": len(
            native_artifact_projections
        ),
        "native_artifact_receipt_projection_violation_count": sum(
            item["receipt_exact"] is not True
            for item in native_artifact_projections
        ),
        "native_top_k_batch_projection_count": len(
            native_top_k_batch_projections
        ),
        "native_top_k_batch_complete_count": sum(
            item.get("k_complete") is True
            for item in native_top_k_batch_projections
        ),
        "native_top_k_batch_incomplete_count": sum(
            item.get("k_complete") is not True
            for item in native_top_k_batch_projections
        ),
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
    director_profile_violations = sum(
        len(value["director_execution_profile_violations"])
        for value in per_task.values()
    )
    director_allowed_tools = sorted(
        {
            tool_id
            for value in per_task.values()
            for tool_id in value["director_allowed_tools"]
        }
    )
    director_tool_enabled_count = sum(
        value["director_tool_calls_enabled_count"]
        for value in per_task.values()
    )
    reasoner_tool_assignment_violations = sum(
        len(value["reasoner_qamemory_tool_assignment_violations"])
        for value in per_task.values()
    )
    native_projection_count = sum(
        value["native_artifact_receipt_projection_count"]
        for value in per_task.values()
    )
    native_projection_violations = sum(
        value["native_artifact_receipt_projection_violation_count"]
        for value in per_task.values()
    )
    native_batch_projections = [
        projection
        for value in per_task.values()
        for projection in value["native_artifact_receipt_projections"]
        if projection.get("projection_kind") == "ordered_top_k_batch"
    ]
    native_batch_complete_count = sum(
        projection.get("k_complete") is True
        for projection in native_batch_projections
    )
    native_batch_incomplete_count = (
        len(native_batch_projections) - native_batch_complete_count
    )
    native_batch_expected_k_values = sorted(
        {
            projection["expected_top_k"]
            for projection in native_batch_projections
            if type(projection.get("expected_top_k")) is int
        }
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
    retrieval_tasks = [
        value for value in per_task.values() if value["retrieval_tool_call_count"] > 0
    ]
    artifact_tasks = [
        value
        for value in per_task.values()
        if value["retrieval_artifact_receipt_count"] > 0
    ]
    routed_tasks = [
        value
        for value in artifact_tasks
        if value["retrieval_artifact_routed_via_relation"] is True
    ]
    output_lineage_tasks = [
        value
        for value in artifact_tasks
        if value["output_inbox_receipt_lineage"] is True
    ]
    assertions = {
        "director_tool_calls": director_tool_calls,
        "director_tool_calls_eq_0": director_tool_calls == 0,
        "director_request_allowed_tools": director_allowed_tools,
        "director_execution_profile_violation_count": (
            director_profile_violations
        ),
        "director_tool_calls_enabled_count": director_tool_enabled_count,
        "director_requests_toolless": bool(per_task)
        and director_profile_violations == 0
        and not director_allowed_tools
        and director_tool_enabled_count == 0,
        "director_retrieval_payload_exposure_count": len(payload_exposures),
        "director_data_plane_isolated": director_tool_calls == 0
        and bool(per_task)
        and director_profile_violations == 0
        and not director_allowed_tools
        and director_tool_enabled_count == 0
        and not payload_exposures,
        "retrieval_tool_calls_by_worker": retrieval_calls,
        "retrieval_tool_calls_by_worker_gt_0": retrieval_calls > 0
        and ownership_violations == 0,
        "worker_ownership_violation_count": ownership_violations,
        "reasoner_qamemory_tool_assignment_violation_count": (
            reasoner_tool_assignment_violations
        ),
        "reasoner_qamemory_tool_unassigned": (
            reasoner_tool_assignment_violations == 0
        ),
        "native_artifact_receipt_projection_count": native_projection_count,
        "native_artifact_receipt_projection_violation_count": (
            native_projection_violations
        ),
        "native_artifacts_match_exact_read_receipts": bool(
            native_projection_count
        ) and native_projection_violations == 0,
        "native_top_k_batch_projection_count": len(native_batch_projections),
        "native_top_k_batch_complete_count": native_batch_complete_count,
        "native_top_k_batch_incomplete_count": native_batch_incomplete_count,
        "native_top_k_batch_expected_k_values": native_batch_expected_k_values,
        "native_top_k_batches_complete": (
            None
            if not native_batch_projections
            else native_batch_incomplete_count == 0
        ),
        "retrieval_tasks": len(retrieval_tasks),
        "retrieval_artifact_tasks": len(artifact_tasks),
        "retrieval_tasks_with_relation_route": len(routed_tasks),
        "retrieval_artifact_routed_via_relation": bool(artifact_tasks)
        and len(routed_tasks) == len(artifact_tasks),
        "retrieval_tasks_with_output_inbox_receipt_lineage": len(
            output_lineage_tasks
        ),
        "output_inbox_receipt_lineage": bool(artifact_tasks)
        and len(output_lineage_tasks) == len(artifact_tasks),
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

    selected_by_id = {
        task_id: task
        for task in selected
        if isinstance((task_id := task.get("task_id")), str) and task_id
    }
    selected_ids = list(selected_by_id)
    direct = _condition_metrics(selected_ids, paired, "direct")
    agentgraph = _condition_metrics(selected_ids, paired, "agentgraph")
    control, per_control = _aggregate_control_plane(selected_ids, trajectories)
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
    complete_metric_scope = (
        "in_database_transductive"
        if index_manifest.get("validation_content_indexed") is True
        and index_manifest.get("validation_isolation_count") == 0
        else "official fixed held-out validation"
    )
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
                complete_metric_scope if run_complete else
                "partial snapshot; strict denominator remains the selected task count"
            ),
        },
        "terminal": {
            "failure_count": len(terminal_failure_ids),
            "failure_task_ids": terminal_failure_ids,
            "termination_reason_counts": dict(sorted(terminal_status.items())),
        },
        "qa_memory_index": index_summary,
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
    terminal = base._mapping(report.get("terminal"))
    control = base._mapping(report.get("control_plane_and_tool_routing"))
    assertions = base._mapping(control.get("assertions"))
    if run.get("status") == "complete":
        delta_note = (
            "该差值来自同一固定 128 条完整正式结果；"
            f"评估口径为 `{metrics.get('scope')}`。"
        )
    else:
        delta_note = (
            f"评估口径为 `{metrics.get('scope')}`；partial 状态下该值仅是"
            "固定分母 fail-closed snapshot，"
            "不是完整 128 条正式结果。"
        )
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
        f"AgentGraph − Direct：**{_percentage(delta.get('exact_match'))} EM**，**{_percentage(delta.get('token_f1'))} F1**。{delta_note}",
        "",
        "## Terminal 与三项边界断言",
        "",
        f"- terminal failure：**{terminal.get('failure_count')}**；termination：`{json.dumps(terminal.get('termination_reason_counts'), ensure_ascii=False, sort_keys=True)}`。",
        f"- `director_tool_calls=0`：**{assertions.get('director_tool_calls_eq_0')}**（实测 {assertions.get('director_tool_calls')}）。",
        f"- `retrieval_tool_calls_by_worker>0`：**{assertions.get('retrieval_tool_calls_by_worker_gt_0')}**（实测 {assertions.get('retrieval_tool_calls_by_worker')}；ownership violations={assertions.get('worker_ownership_violation_count')}）。",
        f"- `retrieval_artifact_routed_via_relation=true`：**{assertions.get('retrieval_artifact_routed_via_relation')}**（{assertions.get('retrieval_tasks_with_relation_route')}/{assertions.get('retrieval_tasks')} retrieval tasks）。",
        f"- Director 数据面隔离：**{assertions.get('director_data_plane_isolated')}**（canonical receipt actual-value exposure={assertions.get('director_retrieval_payload_exposure_count')}；字段名仅作诊断）。",
        f"- Output inbox receipt lineage（独立强断言）：**{assertions.get('output_inbox_receipt_lineage')}**（{assertions.get('retrieval_tasks_with_output_inbox_receipt_lineage')}/{assertions.get('retrieval_tasks')} retrieval tasks）。",
        f"- Ordered top-k batch 完整性：**{assertions.get('native_top_k_batches_complete')}**（complete={assertions.get('native_top_k_batch_complete_count')}/{assertions.get('native_top_k_batch_projection_count')}；incomplete={assertions.get('native_top_k_batch_incomplete_count')}；K={assertions.get('native_top_k_batch_expected_k_values')}）。",
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
