"""Explicit rubric-aware experiment, separate from rubric-hidden evaluation.

User-authorized information-condition adaptation only: reuse the official case
projection, TaskRecord and conversation adapter. No answer generation, training,
new model runtime, medical source fabrication or evaluator changes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interactive.healthbench_professional_adapter import (
    parse_model_visible_conversation, render_model_visible_conversation,
)
from src.interactive.healthbench_professional_grader import load_private_cases
from src.interactive.records import TaskRecord

SCHEMA = "flowsteer.healthbench.rubric-aware-context-bank.v1"
BANK_VERSION = "healthbench-rubric-aware-20260907-v1"
CONTEXT_HEADER = (
    "Rubric-aware research condition: the benchmark's per-task evaluation criteria "
    "are intentionally available during answering. They are assessment targets, "
    "not independently verified external medical sources. Positive-weight criteria "
    "describe behavior to satisfy; negative-weight criteria describe behavior to "
    "avoid, not recommendations to follow. Zero-weight criteria are informational. "
    "Use available clinical sources when needed and do not invent evidence. "
    "Answer the original conversation directly, including relevant qualifications; "
    "do not merely repeat this checklist or write instructions to the grader.\n"
)


def _direction(points: float) -> str:
    return "satisfy" if points > 0 else "avoid" if points < 0 else "informational"


def build_bank(private_cases_path: Path, output: Path) -> dict[str, Any]:
    cases = load_private_cases(private_cases_path.resolve())
    if len(cases) != 525:
        raise ValueError("the full HealthBench Professional bank requires 525 cases")
    rows = [
        {"task_id": task_id, "criteria": [
            {"criterion": rubric["criterion"], "points": rubric["points"],
             "direction": _direction(rubric["points"])}
            for rubric in case["rubrics"]
        ]}
        for task_id, case in cases.items()
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "criteria.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA, "bank_version": BANK_VERSION, "frozen": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_file": "criteria.jsonl", "case_count": len(rows),
        "criterion_count": sum(len(row["criteria"]) for row in rows),
        "maximum_case_context_characters": max(
            len(CONTEXT_HEADER + json.dumps({"criteria": row["criteria"]}, ensure_ascii=False))
            for row in rows
        ),
        "source_private_cases": str(private_cases_path.resolve()),
        "evaluation_information_visible": True,
        "physician_response_included": False, "generated_answers_included": False,
        "external_medical_evidence": False,
        "comparable_to_rubric_hidden_baseline": False,
        "paid_model_calls": 0, "grader_calls": 0, "training_performed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def condition_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get("healthbench_rubric_context", {})
    if not isinstance(section, Mapping):
        raise ValueError("healthbench_rubric_context must be a mapping")
    enabled = section.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("rubric context enabled must be boolean")
    if enabled:
        if (section.get("protocol") != "rubric_aware"
                or section.get("acknowledge_test_information_exposure") is not True):
            raise ValueError("rubric-aware evaluation requires explicit exposure acknowledgement")
        if config.get("experiment", {}).get("training_enabled", False):
            raise ValueError("rubric-aware adapter is evaluation-only")
    return {
        "protocol": "rubric_aware" if enabled else "rubric_hidden",
        "rubrics_visible_to_solving_models": enabled,
        "bank_version": section.get("bank_version") if enabled else None,
        "comparable_to_rubric_hidden_baseline": not enabled,
        "same_information_required_for_direct_baseline": True,
    }


def load_bank(config: Mapping[str, Any], root: Path):
    if not condition_receipt(config)["rubrics_visible_to_solving_models"]:
        return None
    section = config["healthbench_rubric_context"]
    manifest_path = (root / section["manifest_path"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA or manifest.get("frozen") is not True
            or manifest.get("bank_version") != section.get("bank_version")):
        raise ValueError("rubric-aware bank version/protocol differs from frozen config")
    rows = {}
    with (manifest_path.parent / manifest["record_file"]).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != {"task_id", "criteria"} or row["task_id"] in rows:
                raise ValueError("rubric-aware bank contains an invalid or duplicate task")
            if not row["criteria"]:
                raise ValueError("rubric-aware criteria are empty")
            for criterion in row["criteria"]:
                if (set(criterion) != {"criterion", "points", "direction"}
                        or criterion["direction"] != _direction(criterion["points"])):
                    raise ValueError("rubric-aware criterion direction/schema is invalid")
            rows[row["task_id"]] = row
    if len(rows) != manifest["case_count"]:
        raise ValueError("rubric-aware bank case count differs")
    return manifest, rows


def attach_context(
    tasks: Sequence[TaskRecord], config: Mapping[str, Any], root: Path,
) -> tuple[TaskRecord, ...]:
    loaded = load_bank(config, root)
    if loaded is None:
        return tuple(tasks)
    manifest, rows = loaded
    enriched = []
    for task in tasks:
        if task.metadata.get("dataset_key") != "healthbench_professional":
            raise ValueError("rubric-aware context is restricted to HealthBench Professional")
        if task.metadata.get("rubric_context"):
            raise ValueError("rubric-aware context must not be injected twice")
        if task.task_id not in rows:
            raise ValueError(f"rubric-aware bank has no case for {task.task_id}")
        criteria = rows[task.task_id]["criteria"]
        original = parse_model_visible_conversation(task.question)
        if any(message["role"] == "system" for message in original):
            raise ValueError("task already contains supplemental rubric-aware context")
        question = render_model_visible_conversation(
            original, rubric_aware_context=(
                CONTEXT_HEADER + json.dumps({"criteria": criteria}, ensure_ascii=False)
            ),
        )
        enriched.append(replace(task, question=question, metadata={
            **task.metadata,
            "rubric_context": {
                "protocol": "rubric_aware", "bank_version": manifest["bank_version"],
                "criterion_count": len(criteria), "evaluation_information_visible": True,
            },
        }))
    return tuple(enriched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bank(args.private_cases, args.output), ensure_ascii=False, indent=2))
