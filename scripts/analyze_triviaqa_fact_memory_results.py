#!/usr/bin/env python3
"""Offline formal-result analysis for TriviaQA fact-memory AgentGraph.

This is a thin fact-wire adaptation of
``analyze_triviaqa_qa_memory_results.py``.  It reads persisted snapshots only;
it never imports a provider client, starts a model, or invokes an API.  Formal
EM/F1 is exposed only after the complete fixed denominator and every required
fact-memory protocol assertion pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

try:  # Module import in tests and ``python -m`` execution.
    from scripts import analyze_triviaqa_qa_memory_results as qa
except ImportError:  # Direct ``python scripts/...`` execution.
    _QA_PATH = Path(__file__).with_name(
        "analyze_triviaqa_qa_memory_results.py"
    )
    _QA_SPEC = importlib.util.spec_from_file_location(
        "_flowsteer_triviaqa_qa_memory_analysis", _QA_PATH
    )
    if _QA_SPEC is None or _QA_SPEC.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load QA-memory analyzer from {_QA_PATH}")
    qa = importlib.util.module_from_spec(_QA_SPEC)
    sys.modules[_QA_SPEC.name] = qa
    _QA_SPEC.loader.exec_module(qa)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts/triviaqa_fact_memory_unified_v4_v16_full_native_transductive"
)
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "reports/triviaqa_fact_memory_unified_v4_v16_full_native_transductive"
)
DEFAULT_INDEX_MANIFEST = (
    PROJECT_ROOT / "data/triviaqa_fact_memory_full_native_v1/index/manifest.json"
)
FACT_MEMORY_TOOL_ID = "triviaqa.qa_memory"
FACT_RECORD_FIELDS = frozenset(
    {"schema_version", "memory_id", "tool_id", "fact_text"}
)
FACT_HIT_FIELDS = frozenset(
    {"memory_id", "rank", "similarity", "fact_text"}
)
FACT_ARTIFACT_FIELDS = frozenset(
    {"question_scope", "retrieval_query", "top_k", "candidates"}
)
FACT_ARTIFACT_STATUS_FIELDS = frozenset(
    {"retrieval_status", "relevant_memory_ids"}
)
FINAL_MANIFEST_STATUSES = qa.FINAL_MANIFEST_STATUSES
FORBIDDEN_DATA_KEYS = frozenset(
    {
        "original_question",
        "paraphrase_question",
        "canonical_answer",
        "accepted_answers",
        "ground_truth",
        "evaluator",
        "evaluation",
        "evaluator_payload",
        "evaluator_receipt",
        "reference_answer",
        "reference_answers",
    }
)
QA_WRAPPER = re.compile(
    r"(?im)(?:\A|\n)\s*(?:question|answer|q|a)\s*:|"
    r"\bthe\s+answer\s+(?:is|was)\b|"
    r"triviaqa\s+dataset\s+source\s+prompt|"
    r"paired\s+response"
)
WEB_TOOL_MARKERS = ("web", "browser", "http", "serp", "google", "bing")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _all_tool_receipts(
    trajectory: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for round_index, execution_position, execution in qa.base._iter_executions(
        trajectory
    ):
        metadata = qa.base._mapping(execution.get("metadata"))
        response = qa.base._mapping(metadata.get("response"))
        for receipt_position, raw_receipt in enumerate(
            qa.base._list(response.get("tool_receipts"))
        ):
            if not isinstance(raw_receipt, Mapping):
                continue
            signature = qa._receipt_signature(raw_receipt)
            if signature in seen:
                continue
            seen.add(signature)
            yield {
                "round_index": round_index,
                "execution_position": execution_position,
                "receipt_position": receipt_position,
                "execution_outcome": "completed",
                "agent_id": execution.get("agent_id"),
                "receipt": dict(raw_receipt),
            }
    for turn_position, raw_turn in enumerate(qa.base._list(trajectory.get("turns"))):
        turn = qa.base._mapping(raw_turn)
        round_index = turn.get("round_index", turn_position)
        if not isinstance(round_index, int):
            round_index = turn_position
        runtime = qa.base._mapping(turn.get("runtime_summary"))
        for failure_position, raw_failure in enumerate(
            qa.base._list(runtime.get("failure_records"))
        ):
            failure = qa.base._mapping(raw_failure)
            metadata = qa.base._mapping(failure.get("metadata"))
            for receipt_position, raw_receipt in enumerate(
                qa.base._list(metadata.get("tool_receipts"))
            ):
                if not isinstance(raw_receipt, Mapping):
                    continue
                signature = qa._receipt_signature(raw_receipt)
                if signature in seen:
                    continue
                seen.add(signature)
                yield {
                    "round_index": round_index,
                    "execution_position": failure_position,
                    "receipt_position": receipt_position,
                    "execution_outcome": "failed",
                    "agent_id": failure.get("agent_id"),
                    "receipt": dict(raw_receipt),
                }


def _data_plane_payloads(
    trajectory: Mapping[str, Any],
) -> Iterable[tuple[str, object]]:
    """Yield Tool and communication payloads, excluding task/evaluator state."""

    for round_index, execution_position, execution in qa.base._iter_executions(
        trajectory
    ):
        metadata = qa.base._mapping(execution.get("metadata"))
        request = qa.base._mapping(metadata.get("request"))
        response = qa.base._mapping(metadata.get("response"))
        prefix = f"turn[{round_index}].execution[{execution_position}]"
        yield prefix + ".output", execution.get("output", response.get("text"))
        yield prefix + ".request.upstream", request.get("upstream")
        yield prefix + ".request.peer_draft", request.get("peer_draft")
        yield prefix + ".response.tool_receipts", response.get("tool_receipts")
        yield prefix + ".response.react_trace", response.get("react_trace")
    for turn_position, raw_turn in enumerate(qa.base._list(trajectory.get("turns"))):
        turn = qa.base._mapping(raw_turn)
        runtime = qa.base._mapping(turn.get("runtime_summary"))
        for failure_position, raw_failure in enumerate(
            qa.base._list(runtime.get("failure_records"))
        ):
            failure = qa.base._mapping(raw_failure)
            metadata = qa.base._mapping(failure.get("metadata"))
            yield (
                f"turn[{turn_position}].failure[{failure_position}].tool_receipts",
                metadata.get("tool_receipts"),
            )


def _scan_data_plane(
    trajectory: Mapping[str, Any],
) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    visited_json_strings: set[tuple[str, str]] = set()

    def visit(value: object, *, location: str) -> None:
        if isinstance(value, Mapping):
            normalized_keys = {
                str(raw_key).casefold().replace("-", "_")
                for raw_key in value
            }
            if (
                {"question", "answer"}.issubset(normalized_keys)
                or {"q", "a"}.issubset(normalized_keys)
            ):
                violations.append(
                    {
                        "location": location,
                        "kind": "qa_wrapper",
                        "marker": "Question/Answer mapping",
                    }
                )
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized_key = key.casefold().replace("-", "_")
                if (
                    normalized_key in FORBIDDEN_DATA_KEYS
                    or normalized_key.startswith("evaluator_")
                ):
                    violations.append(
                        {
                            "location": location,
                            "kind": "forbidden_field",
                            "marker": key,
                        }
                    )
                visit(child, location=f"{location}.{key}")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                visit(child, location=f"{location}[{index}]")
            return
        if not isinstance(value, str) or not value.strip():
            return
        if QA_WRAPPER.search(value):
            violations.append(
                {
                    "location": location,
                    "kind": "qa_wrapper",
                    "marker": "Question/Answer wrapper",
                }
            )
        stripped = value.strip()
        if stripped[:1] not in {"{", "["}:
            return
        signature = (location, stripped)
        if signature in visited_json_strings:
            return
        visited_json_strings.add(signature)
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        visit(decoded, location=location + ".decoded_json")

    for location, payload in _data_plane_payloads(trajectory):
        visit(payload, location=location)
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in violations:
        key = (item["location"], item["kind"], item["marker"])
        unique[key] = item
    return [unique[key] for key in sorted(unique)]


def _successful_value(receipt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    result = qa.base._mapping(receipt.get("result"))
    if receipt.get("error_type") is not None or result.get("completed") is not True:
        return None
    value = result.get("value", result)
    return value if isinstance(value, Mapping) else None


def _fact_projection(
    round_index: int,
    execution_position: int,
    execution: Mapping[str, Any],
) -> dict[str, Any] | None:
    metadata = qa.base._mapping(execution.get("metadata"))
    request = qa.base._mapping(metadata.get("request"))
    response = qa.base._mapping(metadata.get("response"))
    agent = qa.base._mapping(request.get("agent"))
    if agent.get("role_family") != "evidence_retriever":
        return None
    output = execution.get("output", response.get("text"))
    if not isinstance(output, str):
        return None
    try:
        artifact = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, Mapping) or not isinstance(
        artifact.get("candidates"), list
    ):
        return None

    receipts = [
        qa.base._mapping(item)
        for item in qa.base._list(response.get("tool_receipts"))
        if isinstance(item, Mapping) and item.get("tool_id") == FACT_MEMORY_TOOL_ID
    ]
    searches: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    for position, receipt in enumerate(receipts):
        receipt_request = qa.base._mapping(receipt.get("request"))
        value = _successful_value(receipt)
        if (
            value is not None
            and receipt_request.get("action") == "search"
            and value.get("operation") == "search"
        ):
            searches.append((position, receipt_request, value))
    selected_search: tuple[int, Mapping[str, Any], Mapping[str, Any]] | None = None
    artifact_ids = [
        candidate.get("memory_id") if isinstance(candidate, Mapping) else None
        for candidate in artifact["candidates"]
    ]
    for candidate_search in reversed(searches):
        _, search_request, search_value = candidate_search
        if (
            search_value.get("memory_ids") == artifact_ids
            and qa.base._mapping(search_request.get("arguments")).get("query")
            == artifact.get("retrieval_query")
        ):
            selected_search = candidate_search
            break

    issues: list[str] = []
    allowed_artifact_fields = {
        FACT_ARTIFACT_FIELDS,
        FACT_ARTIFACT_FIELDS | FACT_ARTIFACT_STATUS_FIELDS,
    }
    if frozenset(artifact) not in allowed_artifact_fields:
        issues.append("artifact_schema")
    top_k = artifact.get("top_k")
    candidates = list(artifact["candidates"])
    if type(top_k) is not int or top_k < 1 or len(candidates) != top_k:
        issues.append("artifact_top_k")
    if selected_search is None:
        issues.append("matching_search")
        search_position = -1
        search_value: Mapping[str, Any] = {}
    else:
        search_position, search_request, search_value = selected_search
        arguments = qa.base._mapping(search_request.get("arguments"))
        if (
            set(arguments) != {"query", "limit"}
            or arguments.get("limit") != top_k
            or search_value.get("top_k") != top_k
            or search_value.get("query") != arguments.get("query")
        ):
            issues.append("search_contract")

    hits = qa.base._list(search_value.get("hits"))
    memory_ids = qa.base._list(search_value.get("memory_ids"))
    if len(hits) != top_k or len(memory_ids) != top_k:
        issues.append("search_top_k")
    for expected_rank, raw_hit in enumerate(hits, start=1):
        hit = qa.base._mapping(raw_hit)
        if (
            set(hit) != FACT_HIT_FIELDS
            or hit.get("rank") != expected_rank
            or expected_rank > len(memory_ids)
            or hit.get("memory_id") != memory_ids[expected_rank - 1]
            or isinstance(hit.get("similarity"), bool)
            or not isinstance(hit.get("similarity"), (int, float))
            or not isinstance(hit.get("fact_text"), str)
            or not hit["fact_text"].strip()
        ):
            issues.append(f"search_hit_rank_{expected_rank}")

    ordered_reads: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for receipt in receipts[search_position + 1 :] if search_position >= 0 else ():
        receipt_request = qa.base._mapping(receipt.get("request"))
        value = _successful_value(receipt)
        if (
            value is None
            or receipt_request.get("action") != "read"
            or value.get("operation") != "read"
        ):
            continue
        arguments = qa.base._mapping(receipt_request.get("arguments"))
        memory = qa.base._mapping(value.get("memory"))
        memory_id = arguments.get("memory_id")
        if (
            set(arguments) != {"memory_id"}
            or not isinstance(memory_id, str)
            or value.get("memory_id") != memory_id
            or set(memory) != FACT_RECORD_FIELDS
            or memory.get("memory_id") != memory_id
        ):
            issues.append("read_contract")
            continue
        ordered_reads.append((memory_id, memory, receipt))
    read_ids = [item[0] for item in ordered_reads]
    if read_ids != list(memory_ids):
        issues.append("ordered_complete_top_k_reads")
    for index, raw_candidate in enumerate(candidates):
        candidate = qa.base._mapping(raw_candidate)
        if set(candidate) != FACT_HIT_FIELDS:
            issues.append(f"artifact_candidate_schema_{index + 1}")
            continue
        if index >= len(hits) or index >= len(ordered_reads):
            issues.append(f"artifact_candidate_receipt_{index + 1}")
            continue
        hit = qa.base._mapping(hits[index])
        _, memory, _ = ordered_reads[index]
        if (
            candidate.get("memory_id") != hit.get("memory_id")
            or candidate.get("rank") != hit.get("rank")
            or candidate.get("similarity") != hit.get("similarity")
            or candidate.get("fact_text") != memory.get("fact_text")
        ):
            issues.append(f"artifact_candidate_exact_{index + 1}")
    projection_receipts = (
        receipts[search_position : search_position + 1 + len(ordered_reads)]
        if search_position >= 0
        else []
    )
    return {
        "round_index": round_index,
        "execution_position": execution_position,
        "agent_id": agent.get("id", execution.get("agent_id")),
        "top_k": top_k,
        "search_memory_ids": list(memory_ids),
        "read_memory_ids": read_ids,
        "candidate_memory_ids": artifact_ids,
        "receipt_signatures": [
            qa._receipt_signature(receipt) for receipt in projection_receipts
        ],
        "issues": sorted(set(issues)),
        "complete_top_k_rank_reads": not issues,
    }


def _director_tool_calls(trajectory: Mapping[str, Any]) -> int:
    count = 0
    for raw_turn in qa.base._list(trajectory.get("turns")):
        action = qa.base._mapping(qa.base._mapping(raw_turn).get("action"))
        action_name = str(action.get("action", action.get("kind", ""))).casefold()
        if action_name in {"search", "read", "web_search", "browser"}:
            count += 1
    return count


def _web_and_nonfact_search(
    trajectory: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    web_events: dict[str, dict[str, Any]] = {}
    nonfact_events: dict[str, dict[str, Any]] = {}
    for item in _all_tool_receipts(trajectory):
        receipt = qa.base._mapping(item.get("receipt"))
        request = qa.base._mapping(receipt.get("request"))
        action = str(request.get("action", "")).casefold()
        tool_id = str(receipt.get("tool_id", ""))
        if action != "search" or tool_id == FACT_MEMORY_TOOL_ID:
            continue
        event = {
            "round_index": item.get("round_index"),
            "agent_id": item.get("agent_id"),
            "tool_id": tool_id,
            "action": action,
        }
        signature = qa._receipt_signature(receipt)
        nonfact_events[signature] = event
        if any(marker in tool_id.casefold() for marker in WEB_TOOL_MARKERS):
            web_events[signature] = event
    return len(web_events), len(nonfact_events), list(web_events.values())


def _task_protocol(task_id: str, trajectory: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_receipts = sorted(
        qa._iter_retrieval_receipts(trajectory),
        key=lambda item: (
            int(item.get("round_index", 0)),
            int(item.get("execution_position", 0)),
            int(item.get("receipt_position", 0)),
        ),
    )
    actions = [str(item.get("action", "")).casefold() for item in retrieval_receipts]
    worker_violations = [
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
        for item in retrieval_receipts
        if not (
            isinstance(item.get("agent_id"), str)
            and item.get("request_agent_id") == item.get("agent_id")
            and item.get("request_agent_role_family") == "evidence_retriever"
            and item.get("request_execution_role") == "worker"
            and item.get("request_execution_mode") == "react"
            and FACT_MEMORY_TOOL_ID in item.get("request_allowed_tools", [])
        )
    ]
    projections = [
        projection
        for round_index, execution_position, execution in qa.base._iter_executions(
            trajectory
        )
        if (
            projection := _fact_projection(
                round_index,
                execution_position,
                execution,
            )
        )
        is not None
    ]
    valid_projections = [
        item for item in projections if item["complete_top_k_rank_reads"] is True
    ]

    graph_edges_by_round: dict[int, set[tuple[str, str]]] = {}
    output_agent_id: str | None = None
    for fallback_index, raw_turn in enumerate(qa.base._list(trajectory.get("turns"))):
        turn = qa.base._mapping(raw_turn)
        round_index = turn.get("round_index", fallback_index)
        if not isinstance(round_index, int):
            round_index = fallback_index
        snapshot = qa.base._mapping(turn.get("graph_snapshot"))
        graph_edges_by_round[round_index] = qa._directed_edges(snapshot)
        candidate_output = snapshot.get("output_agent_id")
        if isinstance(candidate_output, str) and candidate_output:
            output_agent_id = candidate_output
        runtime_output = qa.base._mapping(turn.get("runtime_summary")).get(
            "output_agent_id"
        )
        if isinstance(runtime_output, str) and runtime_output:
            output_agent_id = runtime_output
    _, raw_messages = qa._observed_communication(trajectory)
    messages = [
        item
        for item in raw_messages
        if (item["source_agent_id"], item["target_agent_id"])
        in graph_edges_by_round.get(int(item["round_index"]), set())
    ]
    communication_edges = {
        (item["source_agent_id"], item["target_agent_id"]) for item in messages
    }
    projection_routes: list[dict[str, Any]] = []
    output_lineage_results: list[bool] = []
    for projection in valid_projections:
        receipt_signatures = set(projection["receipt_signatures"])
        immediate = [
            message
            for message in messages
            if message["source_agent_id"] == projection["agent_id"]
            and receipt_signatures.issubset(set(message["receipt_signatures"]))
        ]
        output_messages = [
            message
            for message in messages
            if output_agent_id is not None
            and message["target_agent_id"] == output_agent_id
        ]
        output_signatures = {
            signature
            for message in output_messages
            for signature in message["receipt_signatures"]
        }
        output_lineage = bool(
            output_agent_id is not None
            and isinstance(projection.get("agent_id"), str)
            and qa._reachable(
                str(projection["agent_id"]),
                output_agent_id,
                communication_edges,
            )
            and receipt_signatures.issubset(output_signatures)
        )
        output_lineage_results.append(output_lineage)
        projection_routes.append(
            {
                "agent_id": projection.get("agent_id"),
                "receipt_signature_count": len(receipt_signatures),
                "explicit_relation_route_count": len(immediate),
                "routed": bool(immediate),
                "output_lineage": output_lineage,
            }
        )
    data_plane_violations = _scan_data_plane(trajectory)
    web_search_count, nonfact_search_count, web_events = _web_and_nonfact_search(
        trajectory
    )
    return {
        "task_id": task_id,
        "director_tool_calls": _director_tool_calls(trajectory),
        "worker_search_count": actions.count("search"),
        "worker_read_count": actions.count("read"),
        "worker_ownership_violation_count": len(worker_violations),
        "worker_ownership_violations": worker_violations,
        "first_data_plane_action": actions[0] if actions else None,
        "first_data_plane_action_is_search": bool(actions) and actions[0] == "search",
        "fact_artifact_projection_count": len(projections),
        "complete_top_k_projection_count": len(valid_projections),
        "complete_top_k_read_by_rank": bool(projections)
        and len(valid_projections) == len(projections),
        "fact_artifact_projections": projections,
        "fact_artifact_routed_via_explicit_relation": bool(valid_projections)
        and all(route["routed"] for route in projection_routes),
        "output_agent_id": output_agent_id,
        "output_lineage": bool(valid_projections)
        and all(output_lineage_results),
        "projection_routes": projection_routes,
        "web_search_count": web_search_count,
        "non_fact_memory_search_count": nonfact_search_count,
        "web_search_events": web_events,
        "data_plane_violation_count": len(data_plane_violations),
        "data_plane_violations": data_plane_violations,
    }


def _aggregate_protocol(
    selected_ids: Sequence[str],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_task = {
        task_id: _task_protocol(task_id, trajectories[task_id])
        for task_id in selected_ids
        if task_id in trajectories
    }
    observed = len(per_task)
    assertions = {
        "expected_task_count": len(selected_ids),
        "observed_task_count": observed,
        "director_tool_calls": sum(
            item["director_tool_calls"] for item in per_task.values()
        ),
        "director_tool_calls_eq_0": bool(per_task)
        and all(item["director_tool_calls"] == 0 for item in per_task.values()),
        "worker_search_count": sum(
            item["worker_search_count"] for item in per_task.values()
        ),
        "worker_read_count": sum(
            item["worker_read_count"] for item in per_task.values()
        ),
        "worker_ownership_violation_count": sum(
            item["worker_ownership_violation_count"] for item in per_task.values()
        ),
        "retrieval_tool_calls_by_worker_gt_0": bool(per_task)
        and all(
            item["worker_search_count"] + item["worker_read_count"] > 0
            and item["worker_ownership_violation_count"] == 0
            for item in per_task.values()
        ),
        "first_data_plane_action_search_task_count": sum(
            item["first_data_plane_action_is_search"] for item in per_task.values()
        ),
        "first_data_plane_action_is_search": bool(per_task)
        and all(item["first_data_plane_action_is_search"] for item in per_task.values()),
        "complete_top_k_read_by_rank_task_count": sum(
            item["complete_top_k_read_by_rank"] for item in per_task.values()
        ),
        "complete_top_k_read_by_rank": bool(per_task)
        and all(item["complete_top_k_read_by_rank"] for item in per_task.values()),
        "fact_artifact_relation_route_task_count": sum(
            item["fact_artifact_routed_via_explicit_relation"]
            for item in per_task.values()
        ),
        "fact_artifact_routed_via_explicit_relation": bool(per_task)
        and all(
            item["fact_artifact_routed_via_explicit_relation"]
            for item in per_task.values()
        ),
        "output_lineage_task_count": sum(
            item["output_lineage"] for item in per_task.values()
        ),
        "output_lineage": bool(per_task)
        and all(item["output_lineage"] for item in per_task.values()),
        "web_search_count": sum(
            item["web_search_count"] for item in per_task.values()
        ),
        "web_search_eq_0": all(
            item["web_search_count"] == 0 for item in per_task.values()
        ),
        "non_fact_memory_search_count": sum(
            item["non_fact_memory_search_count"] for item in per_task.values()
        ),
        "non_fact_memory_search_eq_0": all(
            item["non_fact_memory_search_count"] == 0
            for item in per_task.values()
        ),
        "agent_facing_data_plane_violation_count": sum(
            item["data_plane_violation_count"] for item in per_task.values()
        ),
        "agent_facing_data_plane_clean": all(
            item["data_plane_violation_count"] == 0 for item in per_task.values()
        ),
    }
    assertions["protocol_valid"] = bool(per_task) and observed == len(selected_ids) and all(
        assertions[key] is True
        for key in (
            "director_tool_calls_eq_0",
            "retrieval_tool_calls_by_worker_gt_0",
            "first_data_plane_action_is_search",
            "complete_top_k_read_by_rank",
            "fact_artifact_routed_via_explicit_relation",
            "output_lineage",
            "web_search_eq_0",
            "non_fact_memory_search_eq_0",
            "agent_facing_data_plane_clean",
        )
    )
    return assertions, per_task


def _wrong_demos(
    selected_by_id: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, dict[str, Any]],
    per_task: Mapping[str, Mapping[str, Any]],
    demo_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[str, str, int]] = []
    counts: Counter[str] = Counter()
    tool_summaries: dict[str, dict[str, Any]] = {}
    for order, task_id in enumerate(selected_by_id):
        trajectory = trajectories.get(task_id)
        if trajectory is None or not qa.base._is_wrong(trajectory):
            continue
        tool_summary = qa.base._tool_summary(trajectory)
        tool_summaries[task_id] = tool_summary
        category = qa._failure_category(trajectory, tool_summary)
        counts[category] += 1
        candidates.append((task_id, category, order))

    selected: list[str] = []
    for category in sorted(counts, key=lambda item: (-counts[item], item)):
        match = next(item for item in candidates if item[1] == category)
        selected.append(match[0])
        if len(selected) >= demo_count:
            break
    for task_id, _, _ in candidates:
        if len(selected) >= demo_count:
            break
        if task_id not in selected:
            selected.append(task_id)
    demos: list[dict[str, Any]] = []
    for task_id in selected:
        demo = qa.base._demo(
            selected_by_id[task_id],
            trajectories[task_id],
            tool_summaries[task_id],
        )
        demo["failure_category"] = qa._failure_category(
            trajectories[task_id], tool_summaries[task_id]
        )
        demo["fact_memory_protocol"] = per_task.get(task_id)
        demos.append(demo)
    wrong_count = sum(counts.values())
    taxonomy = [
        {
            "category": category,
            "count": count,
            "percentage_of_wrong": (
                round(100.0 * count / wrong_count, 2) if wrong_count else 0.0
            ),
        }
        for category, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return demos, taxonomy


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = _resolve(args.selected_tasks)
    trajectory_path = _resolve(args.trajectories)
    paired_path = _resolve(args.paired_results)
    manifest_path = _resolve(args.run_manifest)
    index_manifest_path = _resolve(args.index_manifest)
    selected_rows, selected_diag = qa.base._read_jsonl_snapshot(selected_path)
    trajectory_rows, trajectory_diag = qa.base._read_jsonl_snapshot(trajectory_path)
    paired_rows, paired_diag = qa.base._read_jsonl_snapshot(paired_path)
    selected, selected_dedup = qa._deduplicate_rows(selected_rows)
    trajectories, trajectory_dedup = qa.base._deduplicate_trajectories(
        trajectory_rows
    )
    paired, paired_dedup = qa._deduplicate_rows(paired_rows)
    manifest, manifest_error = qa.base._read_json(manifest_path)
    index_manifest, index_manifest_error = qa.base._read_json(index_manifest_path)
    selected_ids = list(selected)
    direct = qa._condition_metrics(selected_ids, paired, "direct")
    agentgraph = qa._condition_metrics(selected_ids, paired, "agentgraph")
    protocol, per_task = _aggregate_protocol(selected_ids, trajectories)
    demos, taxonomy = _wrong_demos(
        selected,
        trajectories,
        per_task,
        max(args.demo_count, 0),
    )
    exact_selected_set = set(selected_ids)
    exact_paired_set = set(paired) == exact_selected_set
    exact_trajectory_set = set(trajectories) == exact_selected_set
    index_valid = bool(
        index_manifest_error is None
        and index_manifest.get("record_kind") == "fact_memory"
        and index_manifest.get("fact_only") is True
        and index_manifest.get("embedding_input_field") == "fact_text"
        and index_manifest.get("provenance_loaded_by_index") is False
        and index_manifest.get("tool_id") == FACT_MEMORY_TOOL_ID
    )
    snapshot_complete = bool(
        len(selected_ids) == args.expected_count
        and manifest_error is None
        and manifest.get("status") in FINAL_MANIFEST_STATUSES
        and manifest.get("sample_count") == args.expected_count
        and exact_paired_set
        and exact_trajectory_set
        and direct["evaluator_valid"] == args.expected_count
        and agentgraph["evaluator_valid"] == args.expected_count
        and not selected_diag["malformed_records"]
        and not trajectory_diag["malformed_records"]
        and not paired_diag["malformed_records"]
        and not selected_dedup["duplicate_task_record_counts"]
        and not trajectory_dedup["duplicate_task_record_counts"]
        and not paired_dedup["duplicate_task_record_counts"]
        and index_valid
    )
    protocol_valid = protocol["protocol_valid"] is True
    formal_complete = snapshot_complete and protocol_valid
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
    terminal_failure_ids = [
        task_id
        for task_id in selected_ids
        if task_id in trajectories
        and (
            trajectories[task_id].get("termination_reason") != "finish"
            or trajectories[task_id].get("explicit_finish") is not True
        )
    ]
    termination_counts = Counter(
        str(trajectories[task_id].get("termination_reason", "missing"))
        for task_id in selected_ids
        if task_id in trajectories
    )
    return {
        "schema_version": "flowsteer.triviaqa.fact-memory.formal-result-analysis.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "offline_persisted_receipts_only",
        "run": {
            "status": "complete" if formal_complete else "partial_or_protocol_invalid",
            "manifest_status": manifest.get("status"),
            "expected_denominator": args.expected_count,
            "selected_count": len(selected_ids),
            "paired_count": len(paired),
            "trajectory_count": len(trajectories),
            "snapshot_complete": snapshot_complete,
            "protocol_valid": protocol_valid,
            "formal_metrics_available": formal_complete,
            "manifest_read_error": manifest_error,
            "index_manifest_read_error": index_manifest_error,
        },
        "metrics": {
            "metric_protocol": "triviaqa.official.answer.v1",
            "denominator": args.expected_count,
            "formal": (
                {
                    "direct": direct,
                    "agentgraph": agentgraph,
                    "agentgraph_minus_direct": delta,
                }
                if formal_complete
                else None
            ),
            "incomplete_snapshot_diagnostic": {
                "direct": direct,
                "agentgraph": agentgraph,
                "agentgraph_minus_direct": delta,
            },
            "scope": "in_database_transductive_fact_only",
        },
        "terminal": {
            "strict_failure_count": (
                len(terminal_failure_ids) if snapshot_complete else None
            ),
            "observed_failure_count": len(terminal_failure_ids),
            "failure_task_ids": terminal_failure_ids,
            "termination_reason_counts": dict(sorted(termination_counts.items())),
        },
        "fact_memory_index": {
            key: index_manifest.get(key)
            for key in (
                "schema_version",
                "record_kind",
                "tool_id",
                "memory_count",
                "fact_only",
                "embedding_input_field",
                "provenance_loaded_by_index",
                "embedding_model",
                "embedding_dimension",
                "normalization",
                "similarity",
                "frozen_top_k",
                "tool_budget",
            )
        },
        "protocol_assertions": protocol,
        "per_task_protocol": per_task,
        "failure_taxonomy": taxonomy,
        "wrong_demo_selection": {
            "requested_count": args.demo_count,
            "actual_count": len(demos),
            "minimum_three_real_wrong_demos_met": len(demos) >= 3,
            "shortfall": max(3 - len(demos), 0),
            "no_fabrication_policy": (
                "only persisted evaluator-wrong trajectories are eligible"
            ),
        },
        "wrong_demos": demos,
        "inputs": {
            "selected_tasks": selected_diag,
            "trajectories": trajectory_diag,
            "paired_results": paired_diag,
            "selected_deduplication": selected_dedup,
            "trajectory_deduplication": trajectory_dedup,
            "paired_deduplication": paired_dedup,
            "run_manifest": qa.base._display_path(manifest_path),
            "index_manifest": qa.base._display_path(index_manifest_path),
        },
    }


def _percentage(value: object) -> str:
    number = qa._number(value)
    return "N/A" if number is None else f"{100.0 * number:.2f}%"


def _render_markdown(report: Mapping[str, Any]) -> str:
    run = qa.base._mapping(report.get("run"))
    metrics = qa.base._mapping(report.get("metrics"))
    formal = qa.base._mapping(metrics.get("formal"))
    graph = qa.base._mapping(formal.get("agentgraph"))
    protocol = qa.base._mapping(report.get("protocol_assertions"))
    terminal = qa.base._mapping(report.get("terminal"))
    demos = qa.base._list(report.get("wrong_demos"))
    lines = [
        "# TriviaQA fact-memory v16 result analysis",
        "",
        f"- Status: `{run.get('status')}`",
        f"- Fixed denominator: `{run.get('expected_denominator')}`",
        f"- AgentGraph EM: `{_percentage(graph.get('strict_exact_match'))}`",
        f"- AgentGraph F1: `{_percentage(graph.get('strict_token_f1'))}`",
        f"- Terminal failures: `{terminal.get('strict_failure_count')}`",
        f"- Protocol valid: `{protocol.get('protocol_valid')}`",
        f"- Director Tool calls: `{protocol.get('director_tool_calls')}`",
        f"- Worker search/read: `{protocol.get('worker_search_count')}` / `{protocol.get('worker_read_count')}`",
        f"- First data-plane Action is search: `{protocol.get('first_data_plane_action_is_search')}`",
        f"- Complete Top-K rank reads: `{protocol.get('complete_top_k_read_by_rank')}`",
        f"- Fact artifact explicit-relation route: `{protocol.get('fact_artifact_routed_via_explicit_relation')}`",
        f"- Output lineage: `{protocol.get('output_lineage')}`",
        f"- Web Search count: `{protocol.get('web_search_count')}`",
        f"- Agent-facing data-plane violations: `{protocol.get('agent_facing_data_plane_violation_count')}`",
        f"- Real wrong demos: `{len(demos)}`",
    ]
    if formal.get("agentgraph") is None:
        lines.extend(
            [
                "",
                "> Formal EM/F1 is withheld because the fixed snapshot or protocol is incomplete.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks",
        default=str(DEFAULT_ARTIFACT_DIR / "selected_tasks.jsonl"),
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
        "--run-manifest",
        default=str(DEFAULT_ARTIFACT_DIR / "run_manifest.json"),
    )
    parser.add_argument("--index-manifest", default=str(DEFAULT_INDEX_MANIFEST))
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_REPORT_DIR / "formal_result_analysis.json"),
    )
    parser.add_argument(
        "--output-markdown",
        default=str(DEFAULT_REPORT_DIR / "formal_result_analysis.md"),
    )
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument("--demo-count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    if args.demo_count < 0:
        parser.error("--demo-count must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    output_json = _resolve(args.output_json)
    output_markdown = _resolve(args.output_markdown)
    qa.base._atomic_write_text(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    qa.base._atomic_write_text(output_markdown, _render_markdown(report))
    print(
        json.dumps(
            {
                "status": report["run"]["status"],
                "formal_metrics_available": report["run"][
                    "formal_metrics_available"
                ],
                "wrong_demos": report["wrong_demo_selection"]["actual_count"],
                "assertions": report["protocol_assertions"],
                "output_json": qa.base._display_path(output_json),
                "output_markdown": qa.base._display_path(output_markdown),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
