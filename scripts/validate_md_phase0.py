#!/usr/bin/env python3
"""Validate the MD Phase-0 trajectory evidence boundary without model calls.

The validator consumes either the raw trajectory JSONL written by evaluation
runners or the ``trajectory`` envelopes written by ``AppendOnlyJsonlStore``.
It reconstructs every ``TrajectoryRecord`` and checks only evidence already
present in the persisted record.  It does not rerun an evaluator, tokenizer,
model, tool, or workflow.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.persistence import GraphSnapshotEvent, replay_snapshots
from src.interactive.persistence.trajectory_store import AppendOnlyJsonlStore
from src.interactive.records import SCHEMA_VERSION, TrajectoryRecord


OUTPUT_SCHEMA_VERSION = "flowsteer.md.phase0.acceptance.v1"
REQUIRED_VERSION_FIELDS = (
    "policy",
    "model_catalog",
    "evaluator",
    "prompt",
    "tool",
)
CHECK_NAMES = (
    "versioned_trajectory_roundtrip",
    "exact_director_turn_receipts",
    "policy_and_execution_version_binding",
    "graph_snapshot_replay",
    "terminal_evaluator_reward_lineage",
    "execution_recorded_once",
)


class Phase0ValidationError(ValueError):
    """One persisted trajectory fails an MD Phase-0 acceptance condition."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(message)
        self.check = check


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty_string(value: Any, name: str, check: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase0ValidationError(check, f"{name} must be a non-empty string")
    return value


def _raw_jsonl_payloads(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            f"missing trajectory JSONL: {path}",
        ) from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase0ValidationError(
                    "versioned_trajectory_roundtrip",
                    f"{path}:{line_number}: invalid JSON",
                ) from exc
            if not isinstance(value, Mapping):
                raise Phase0ValidationError(
                    "versioned_trajectory_roundtrip",
                    f"{path}:{line_number}: trajectory must be a JSON object",
                )
            values.append(dict(value))
    if not values:
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            f"{path}: no trajectory records",
        )
    return values


def load_trajectory_payloads(path: Path) -> list[dict[str, Any]]:
    """Load raw records or the existing append-only trajectory envelopes."""

    values = _raw_jsonl_payloads(path)
    first = values[0]
    envelope_fields = {"event_id", "record_kind", "content_hash", "payload"}
    if set(first) == envelope_fields:
        try:
            return [dict(value) for value in AppendOnlyJsonlStore(path, "trajectory").payloads()]
        except (TypeError, ValueError) as exc:
            raise Phase0ValidationError(
                "versioned_trajectory_roundtrip",
                f"{path}: invalid append-only trajectory envelope: {exc}",
            ) from exc
    if any(set(value) == envelope_fields for value in values[1:]):
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            f"{path}: raw records and append-only envelopes cannot be mixed",
        )
    return values


def _validate_roundtrip(payload: Mapping[str, Any]) -> TrajectoryRecord:
    try:
        record = TrajectoryRecord.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            f"TrajectoryRecord reconstruction failed: {exc}",
        ) from exc
    if record.schema_version != SCHEMA_VERSION:
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            "trajectory schema version differs from the runtime schema",
        )
    _nonempty_string(record.trajectory_id, "trajectory_id", "versioned_trajectory_roundtrip")
    _nonempty_string(record.group_id, "group_id", "versioned_trajectory_roundtrip")
    _nonempty_string(record.condition_id, "condition_id", "versioned_trajectory_roundtrip")
    _nonempty_string(record.rollout_id, "rollout_id", "versioned_trajectory_roundtrip")
    if not record.turns:
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            "trajectory must contain at least one Director turn",
        )
    if [turn.round_index for turn in record.turns] != list(range(len(record.turns))):
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            "Director round_index values must be contiguous from zero",
        )
    if json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) != json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True
    ):
        raise Phase0ValidationError(
            "versioned_trajectory_roundtrip",
            "trajectory round-trip changed the persisted payload",
        )
    return record


def _validate_turn_receipts(record: TrajectoryRecord) -> dict[str, int]:
    director_request_ids: set[str] = set()
    prompt_tokens = 0
    sampled_tokens = 0
    action_tokens = 0
    for turn in record.turns:
        location = f"turn[{turn.round_index}]"
        _nonempty_string(turn.turn_id, f"{location}.turn_id", "exact_director_turn_receipts")
        _nonempty_string(turn.prompt, f"{location}.prompt", "exact_director_turn_receipts")
        _nonempty_string(
            turn.policy_response,
            f"{location}.policy_response",
            "exact_director_turn_receipts",
        )
        if not turn.prompt_token_ids:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} has no prompt token receipt",
            )
        if not turn.output_token_ids:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} has no sampled output token receipt",
            )
        if not turn.receipt_verified or turn.reconstructed_context:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} is not an exact, original-context behavior receipt",
            )
        has_action = bool(turn.action)
        if has_action != (turn.executed_prefix_tokens > 0):
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} action and consumed-prefix receipt disagree",
            )
        request_id = _nonempty_string(
            turn.director_request_id,
            f"{location}.director_request_id",
            "exact_director_turn_receipts",
        )
        if request_id in director_request_ids:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"duplicate Director request ID {request_id!r}",
            )
        director_request_ids.add(request_id)
        if turn.director_latency_ms is None or not math.isfinite(turn.director_latency_ms):
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} has no finite Director latency receipt",
            )
        if turn.director_attempt_count is None:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} has no Director attempt-count receipt",
            )
        if turn.director_generation_seed is None:
            raise Phase0ValidationError(
                "exact_director_turn_receipts",
                f"{location} has no Director generation-seed receipt",
            )
        prompt_tokens += len(turn.prompt_token_ids)
        sampled_tokens += len(turn.output_token_ids)
        action_tokens += turn.executed_prefix_tokens
    return {
        "director_turns": len(record.turns),
        "prompt_tokens": prompt_tokens,
        "sampled_tokens": sampled_tokens,
        "consumed_action_tokens": action_tokens,
    }


def _validate_version_binding(record: TrajectoryRecord) -> None:
    for field_name in REQUIRED_VERSION_FIELDS:
        _nonempty_string(
            getattr(record.versions, field_name),
            f"versions.{field_name}",
            "policy_and_execution_version_binding",
        )
    if record.evaluation.evaluator_version != record.versions.evaluator:
        raise Phase0ValidationError(
            "policy_and_execution_version_binding",
            "evaluation receipt is bound to a different evaluator version",
        )
    for turn in record.turns:
        location = f"turn[{turn.round_index}]"
        if turn.policy_version != record.versions.policy:
            raise Phase0ValidationError(
                "policy_and_execution_version_binding",
                f"{location} is bound to a different behavior policy version",
            )
        _nonempty_string(
            turn.server_weight_version,
            f"{location}.server_weight_version",
            "policy_and_execution_version_binding",
        )
        for execution in turn.executions:
            execution_location = f"{location}.execution[{execution.execution_id}]"
            for name, value in (
                ("execution_id", execution.execution_id),
                ("experiment_id", execution.experiment_id),
                ("agent_id", execution.agent_id),
                ("model_id", execution.model_id),
                ("model_fingerprint", execution.model_fingerprint),
                ("provider", execution.provider),
            ):
                _nonempty_string(
                    value,
                    f"{execution_location}.{name}",
                    "policy_and_execution_version_binding",
                )
            if execution.graph_revision != turn.graph_revision:
                raise Phase0ValidationError(
                    "policy_and_execution_version_binding",
                    f"{execution_location} is bound to a different graph revision",
                )
            if execution.input_tokens is None or execution.input_tokens < 0:
                raise Phase0ValidationError(
                    "policy_and_execution_version_binding",
                    f"{execution_location} has no valid input-token receipt",
                )
            if execution.output_tokens is None or execution.output_tokens < 0:
                raise Phase0ValidationError(
                    "policy_and_execution_version_binding",
                    f"{execution_location} has no valid output-token receipt",
                )
            if execution.latency_ms is None or not math.isfinite(execution.latency_ms):
                raise Phase0ValidationError(
                    "policy_and_execution_version_binding",
                    f"{execution_location} has no finite latency receipt",
                )
            if execution.error_type is None and not execution.output:
                raise Phase0ValidationError(
                    "policy_and_execution_version_binding",
                    f"{execution_location} has neither model output nor an error receipt",
                )


def _validate_snapshot_replay(record: TrajectoryRecord) -> None:
    events = [
        GraphSnapshotEvent(
            revision=turn.graph_revision,
            graph=turn.graph_snapshot,
            snapshot_id=turn.graph_snapshot_id,
            previous_snapshot_id=turn.previous_graph_snapshot_id,
        )
        for turn in record.turns
    ]
    try:
        replayed = replay_snapshots(events)
    except ValueError as exc:
        raise Phase0ValidationError(
            "graph_snapshot_replay",
            f"snapshot replay failed: {exc}",
        ) from exc
    if replayed != dict(record.turns[-1].graph_snapshot):
        raise Phase0ValidationError(
            "graph_snapshot_replay",
            "replayed final graph differs from the persisted final turn snapshot",
        )


def _validate_terminal_lineage(record: TrajectoryRecord) -> None:
    if not record.natural_policy_terminal:
        raise Phase0ValidationError(
            "terminal_evaluator_reward_lineage",
            "trajectory has no natural FINISH or max-round terminal state",
        )
    if not record.evaluation.valid or record.evaluation.reward is None:
        raise Phase0ValidationError(
            "terminal_evaluator_reward_lineage",
            "terminal trajectory has no valid evaluator reward receipt",
        )
    reward = float(record.evaluation.reward)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise Phase0ValidationError(
            "terminal_evaluator_reward_lineage",
            "terminal task reward must be finite and in [0, 1]",
        )
    if record.explicit_finish:
        if record.termination_reason != "finish":
            raise Phase0ValidationError(
                "terminal_evaluator_reward_lineage",
                "explicit FINISH has a different termination reason",
            )
        if not isinstance(record.final_answer, str) or not record.final_answer.strip():
            raise Phase0ValidationError(
                "terminal_evaluator_reward_lineage",
                "explicit FINISH has no final answer",
            )
    elif not record.valid_lineage_fallback_used:
        if record.final_answer not in (None, "") or reward != 0.0:
            raise Phase0ValidationError(
                "terminal_evaluator_reward_lineage",
                "natural terminal failure must retain an empty answer and zero reward",
            )


def _validate_execution_uniqueness(record: TrajectoryRecord) -> int:
    execution_ids: set[str] = set()
    for turn in record.turns:
        if turn.execution_reused and turn.executions:
            raise Phase0ValidationError(
                "execution_recorded_once",
                f"turn[{turn.round_index}] duplicates cached execution records",
            )
        for execution in turn.executions:
            if execution.execution_id in execution_ids:
                raise Phase0ValidationError(
                    "execution_recorded_once",
                    f"duplicate execution ID {execution.execution_id!r}",
                )
            execution_ids.add(execution.execution_id)
    return len(execution_ids)


def validate_trajectory_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = _validate_roundtrip(payload)
    turn_counts = _validate_turn_receipts(record)
    _validate_version_binding(record)
    _validate_snapshot_replay(record)
    _validate_terminal_lineage(record)
    execution_count = _validate_execution_uniqueness(record)
    return {
        "trajectory_id": record.trajectory_id,
        "task_id": record.task.task_id,
        "split": record.task.split,
        "policy_version": record.versions.policy,
        "evaluator_version": record.evaluation.evaluator_version,
        "terminal_reward": float(record.evaluation.reward),
        "explicit_finish": record.explicit_finish,
        "grpo_eligible": record.grpo_eligible,
        **turn_counts,
        "agent_execution_calls": execution_count,
    }


def build_phase0_acceptance_receipt(path: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    seen_trajectory_ids: set[str] = set()
    seen_execution_ids: set[str] = set()
    try:
        payloads = load_trajectory_payloads(path)
    except Phase0ValidationError as exc:
        payloads = []
        failures.append({"record_index": None, "check": exc.check, "reason": str(exc)})

    for index, payload in enumerate(payloads):
        trajectory_id = payload.get("trajectory_id")
        try:
            result = validate_trajectory_payload(payload)
            if result["trajectory_id"] in seen_trajectory_ids:
                raise Phase0ValidationError(
                    "versioned_trajectory_roundtrip",
                    f"duplicate trajectory ID {result['trajectory_id']!r}",
                )
            persisted_execution_ids = {
                execution["execution_id"]
                for turn in payload.get("turns", ())
                if isinstance(turn, Mapping)
                for execution in turn.get("executions", ())
                if isinstance(execution, Mapping)
                and isinstance(execution.get("execution_id"), str)
            }
            duplicates = seen_execution_ids.intersection(persisted_execution_ids)
            if duplicates:
                raise Phase0ValidationError(
                    "execution_recorded_once",
                    f"execution IDs repeated across trajectories: {sorted(duplicates)!r}",
                )
            seen_trajectory_ids.add(result["trajectory_id"])
            seen_execution_ids.update(persisted_execution_ids)
            accepted_records.append(result)
        except Phase0ValidationError as exc:
            failures.append(
                {
                    "record_index": index,
                    "trajectory_id": trajectory_id,
                    "check": exc.check,
                    "reason": str(exc),
                }
            )

    check_results = {
        name: not any(failure["check"] == name for failure in failures)
        for name in CHECK_NAMES
    }
    accepted = bool(payloads) and not failures
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "input_path": str(path.resolve()),
        "accepted": accepted,
        "checks": check_results,
        "counts": {
            "input_trajectories": len(payloads),
            "accepted_trajectories": len(accepted_records),
            "failed_trajectories": len(failures),
            "director_turns": sum(item["director_turns"] for item in accepted_records),
            "agent_execution_calls": sum(
                item["agent_execution_calls"] for item in accepted_records
            ),
            "consumed_action_tokens": sum(
                item["consumed_action_tokens"] for item in accepted_records
            ),
            "grpo_eligible_trajectories": sum(
                bool(item["grpo_eligible"]) for item in accepted_records
            ),
        },
        "policy_versions": sorted(
            {item["policy_version"] for item in accepted_records}
        ),
        "evaluator_versions": sorted(
            {item["evaluator_version"] for item in accepted_records}
        ),
        "failures": failures,
        "scope": {
            "evaluator_recomputed": False,
            "tokenizer_recomputed": False,
            "model_or_tool_calls": 0,
            "statement": (
                "Acceptance proves persisted receipt lineage and replay only; "
                "it does not independently recompute evaluator correctness or tokenization."
            ),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MD Phase-0 persisted trajectory evidence."
    )
    parser.add_argument("--input", required=True, type=Path, help="trajectory JSONL")
    parser.add_argument("--output", required=True, type=Path, help="acceptance JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_phase0_acceptance_receipt(args.input)
    _write_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
