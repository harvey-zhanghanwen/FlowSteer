#!/usr/bin/env python3
"""Aggregate fixed HotpotQA/TriviaQA held-out metrics across policy steps.

The metric boundary is the same strict denominator used by
``evaluate_hotpotqa_round.py::_aggregate``.  The input is a JSON manifest so
that every metric file is paired with the exact trajectory receipts used to
verify task IDs, evaluator versions, and the behavior policy version::

    {
      "schema_version": "flowsteer.joint_qa_curve_input.v1",
      "steps": [
        {
          "step": 0,
          "label": "step0",
          "expected_policy_version": "qwen35-9b-joint-step-000000",
          "expected_policy_adapter": "theta_joint_step_000000",
          "datasets": {
            "hotpotqa": {
              "metrics_path": ".../paired_results.jsonl",
              "trajectory_receipts_path": ".../agentgraph_trajectories.jsonl"
            },
            "triviaqa": {
              "metrics_path": ".../report.json",
              "trajectory_receipts_path": ".../agentgraph_trajectories.jsonl"
            }
          }
        }
      ]
    }

``metrics_path`` may be either a paired-results JSONL file or the report JSON
emitted by the corresponding fixed evaluation runner.  This script does not
run models, evaluators, training, or API calls.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_hotpotqa_round import _aggregate
from src.interactive.task_evaluator import (
    HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    TRIVIAQA_ANSWER_EVALUATOR_VERSION,
)


INPUT_SCHEMA_VERSION = "flowsteer.joint_qa_curve_input.v1"
OUTPUT_SCHEMA_VERSION = "flowsteer.joint_qa_curve.v1"
DATASETS = ("hotpotqa", "triviaqa")
DATASET_LABELS = {"hotpotqa": "HotpotQA", "triviaqa": "TriviaQA"}
EXPECTED_EVALUATORS = {
    "hotpotqa": HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    "triviaqa": TRIVIAQA_ANSWER_EVALUATOR_VERSION,
}


class JointCurveError(RuntimeError):
    """A curve input violates the fixed held-out comparison boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JointCurveError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JointCurveError(f"missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JointCurveError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise JointCurveError(f"{path}: expected a JSON object")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise JointCurveError(f"missing input file: {path}") from exc
    values: list[dict[str, Any]] = []
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JointCurveError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise JointCurveError(f"{path}:{line_number}: expected a JSON object")
            values.append(dict(value))
    if not values:
        raise JointCurveError(f"{path}: expected at least one record")
    return values


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise JointCurveError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointCurveError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JointCurveError(f"{name} must be finite and between 0 and 1")
    return result


def _task_id_from_receipt(receipt: Mapping[str, Any], name: str) -> str:
    task = receipt.get("task")
    task_id = task.get("task_id") if isinstance(task, Mapping) else None
    if not isinstance(task_id, str) or not task_id.strip():
        raise JointCurveError(f"{name} is missing task.task_id")
    return task_id


def _validate_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    expected_task_ids: Sequence[str],
    name: str,
) -> tuple[str, str]:
    evaluator_version = EXPECTED_EVALUATORS[dataset]
    by_task: dict[str, Mapping[str, Any]] = {}
    policy_versions: set[str] = set()
    for index, receipt in enumerate(receipts):
        location = f"{name}[{index}]"
        task_id = _task_id_from_receipt(receipt, location)
        if task_id in by_task:
            raise JointCurveError(f"{name}: duplicate task receipt {task_id!r}")
        by_task[task_id] = receipt
        evaluation = _mapping(receipt.get("evaluation"), f"{location}.evaluation")
        actual_evaluator = evaluation.get("evaluator_version")
        if actual_evaluator != evaluator_version:
            raise JointCurveError(
                f"{location}: evaluator {actual_evaluator!r} does not match "
                f"{evaluator_version!r}"
            )
        versions = _mapping(receipt.get("versions"), f"{location}.versions")
        if versions.get("evaluator") != actual_evaluator:
            raise JointCurveError(
                f"{location}: versions.evaluator does not match evaluation receipt"
            )
        policy = versions.get("policy")
        if not isinstance(policy, str) or not policy.strip():
            raise JointCurveError(f"{location}: versions.policy must be non-empty")
        policy_versions.add(policy)
        if receipt.get("sampling_receipt_verified") is not True:
            raise JointCurveError(f"{location}: sampling_receipt_verified must be true")

    received_task_ids = tuple(by_task)
    if received_task_ids != tuple(expected_task_ids):
        raise JointCurveError(
            f"{name}: ordered receipt task IDs do not match metric task IDs"
        )
    if len(policy_versions) != 1:
        raise JointCurveError(
            f"{name}: expected exactly one policy version, got "
            f"{sorted(policy_versions)!r}"
        )
    return next(iter(policy_versions)), evaluator_version


def _paired_metrics(
    path: Path,
    *,
    dataset: str,
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    task_ids: list[str] = []
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise JointCurveError(f"{path}:{index + 1}: missing task_id")
        if not task_id.startswith(f"{dataset}:"):
            raise JointCurveError(
                f"{path}:{index + 1}: task {task_id!r} is not {dataset}"
            )
        task_ids.append(task_id)
        agentgraph = _mapping(row.get("agentgraph"), f"{path}:{index + 1}.agentgraph")
        for field in ("available", "valid"):
            if not isinstance(agentgraph.get(field), bool):
                raise JointCurveError(
                    f"{path}:{index + 1}.agentgraph.{field} must be boolean"
                )
        _finite_unit(
            agentgraph.get("exact_match"),
            f"{path}:{index + 1}.agentgraph.exact_match",
        )
        _finite_unit(
            agentgraph.get("token_f1"),
            f"{path}:{index + 1}.agentgraph.token_f1",
        )
    if len(set(task_ids)) != len(task_ids):
        raise JointCurveError(f"{path}: duplicate metric task IDs")
    return dict(_aggregate(rows, "agentgraph")), tuple(task_ids), rows


def _report_metrics(
    path: Path, *, dataset: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(path)
    if str(report.get("dataset", "")).lower() != dataset:
        raise JointCurveError(
            f"{path}: dataset {report.get('dataset')!r} does not match {dataset!r}"
        )
    metrics = dict(_mapping(report.get("agentgraph"), f"{path}.agentgraph"))
    denominator = metrics.get("denominator")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
    ):
        raise JointCurveError(f"{path}.agentgraph.denominator must be positive")
    if report.get("sample_count") != denominator:
        raise JointCurveError(
            f"{path}: sample_count does not match agentgraph.denominator"
        )
    _finite_unit(metrics.get("strict_exact_match"), f"{path}.strict_exact_match")
    _finite_unit(metrics.get("strict_token_f1"), f"{path}.strict_token_f1")
    return metrics, report


def _compare_paired_to_receipts(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> None:
    receipts_by_task = {
        _task_id_from_receipt(receipt, f"{name}.receipt"): receipt
        for receipt in receipts
    }
    for row in rows:
        task_id = str(row["task_id"])
        graph = _mapping(row.get("agentgraph"), f"{name}.{task_id}.agentgraph")
        evaluation = _mapping(
            receipts_by_task[task_id].get("evaluation"),
            f"{name}.{task_id}.evaluation",
        )
        receipt_valid = evaluation.get("valid") is True
        if graph.get("valid") is not receipt_valid:
            raise JointCurveError(
                f"{name}.{task_id}: paired valid flag differs from evaluator receipt"
            )
        receipt_metrics = _mapping(
            evaluation.get("metrics", {}), f"{name}.{task_id}.evaluation.metrics"
        )
        for paired_field, receipt_field in (
            ("exact_match", "exact_match"),
            ("token_f1", "token_f1"),
        ):
            expected = float(receipt_metrics.get(receipt_field, 0.0))
            actual = float(graph[paired_field])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise JointCurveError(
                    f"{name}.{task_id}: paired {paired_field} differs from "
                    "evaluator receipt"
                )


def _compare_report_to_receipts(
    metrics: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> None:
    rows = []
    for index, receipt in enumerate(receipts):
        evaluation = _mapping(
            receipt.get("evaluation"), f"{name}.receipts[{index}].evaluation"
        )
        receipt_metrics = _mapping(
            evaluation.get("metrics", {}),
            f"{name}.receipts[{index}].evaluation.metrics",
        )
        rows.append(
            {
                "agentgraph": {
                    "available": True,
                    "valid": evaluation.get("valid") is True,
                    "exact_match": float(receipt_metrics.get("exact_match", 0.0)),
                    "token_f1": float(receipt_metrics.get("token_f1", 0.0)),
                }
            }
        )
    receipt_aggregate = _aggregate(rows, "agentgraph")
    for field in (
        "denominator",
        "completed",
        "evaluator_valid",
        "strict_exact_match",
        "strict_token_f1",
    ):
        if field not in metrics:
            raise JointCurveError(f"{name}: report agentgraph is missing {field}")
        expected = receipt_aggregate[field]
        actual = metrics[field]
        if isinstance(expected, float):
            matched = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            matched = matched and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matched = actual == expected
        if not matched:
            raise JointCurveError(
                f"{name}: report {field} differs from evaluator receipts"
            )


def _dataset_step(
    value: Mapping[str, Any],
    *,
    dataset: str,
    base: Path,
) -> dict[str, Any]:
    metrics_path = _resolve(base, value.get("metrics_path"), f"{dataset}.metrics_path")
    receipt_path = _resolve(
        base,
        value.get("trajectory_receipts_path"),
        f"{dataset}.trajectory_receipts_path",
    )
    receipts = _read_jsonl(receipt_path)
    report: Optional[dict[str, Any]] = None
    paired_rows: Optional[list[dict[str, Any]]] = None
    if metrics_path.suffix.lower() == ".jsonl":
        metrics, task_ids, paired_rows = _paired_metrics(metrics_path, dataset=dataset)
        source_kind = "paired_results_jsonl"
    elif metrics_path.suffix.lower() == ".json":
        metrics, report = _report_metrics(metrics_path, dataset=dataset)
        task_ids = tuple(
            _task_id_from_receipt(receipt, f"{receipt_path}[{index}]")
            for index, receipt in enumerate(receipts)
        )
        if int(metrics["denominator"]) != len(task_ids):
            raise JointCurveError(
                f"{metrics_path}: report denominator does not match trajectory receipts"
            )
        source_kind = "round_report_json"
    else:
        raise JointCurveError(
            f"{metrics_path}: metrics_path must end in .jsonl or .json"
        )

    policy, evaluator = _validate_receipts(
        receipts,
        dataset=dataset,
        expected_task_ids=task_ids,
        name=str(receipt_path),
    )
    if paired_rows is not None:
        _compare_paired_to_receipts(
            paired_rows, receipts, name=f"{dataset}:{metrics_path}"
        )
    if report is not None:
        _compare_report_to_receipts(metrics, receipts, name=f"{dataset}:{metrics_path}")
        if report.get("policy_version") != policy:
            raise JointCurveError(
                f"{metrics_path}: report policy_version differs from trajectory receipts"
            )

    exact_match = _finite_unit(
        metrics.get("strict_exact_match"), f"{metrics_path}.strict_exact_match"
    )
    token_f1 = _finite_unit(
        metrics.get("strict_token_f1"), f"{metrics_path}.strict_token_f1"
    )
    return {
        "dataset": DATASET_LABELS[dataset],
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
        "evaluator_version": evaluator,
        "policy_version": policy,
        "strict_exact_match": exact_match,
        "strict_token_f1": token_f1,
        "strict_exact_match_percent": 100.0 * exact_match,
        "strict_token_f1_percent": 100.0 * token_f1,
        "completed": int(metrics.get("completed", 0)),
        "evaluator_valid": int(metrics.get("evaluator_valid", 0)),
        "metrics_source_kind": source_kind,
        "metrics_path": str(metrics_path),
        "trajectory_receipts_path": str(receipt_path),
        "policy_receipt_verified": True,
        "evaluator_receipts_verified": True,
    }


def build_joint_curve(
    spec: Mapping[str, Any], *, base_dir: str | Path
) -> dict[str, Any]:
    """Validate and aggregate a two-dataset fixed held-out policy curve."""

    if spec.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise JointCurveError(f"schema_version must be {INPUT_SCHEMA_VERSION!r}")
    base = Path(base_dir).expanduser().resolve()
    raw_steps = spec.get("steps")
    if (
        not isinstance(raw_steps, Sequence)
        or isinstance(raw_steps, (str, bytes))
        or not raw_steps
    ):
        raise JointCurveError("steps must be a non-empty array")

    fixed_task_ids: dict[str, tuple[str, ...]] = {}
    seen_steps: set[int] = set()
    steps: list[dict[str, Any]] = []
    for input_index, raw_step in enumerate(raw_steps):
        step = _mapping(raw_step, f"steps[{input_index}]")
        ordinal = step.get("step")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise JointCurveError(f"steps[{input_index}].step must be non-negative")
        if ordinal in seen_steps:
            raise JointCurveError(f"duplicate policy step {ordinal}")
        seen_steps.add(ordinal)
        datasets = _mapping(step.get("datasets"), f"steps[{input_index}].datasets")
        if set(datasets) != set(DATASETS):
            raise JointCurveError(
                f"steps[{input_index}].datasets must contain exactly {DATASETS!r}"
            )
        values = {
            dataset: _dataset_step(
                _mapping(datasets[dataset], f"steps[{input_index}].{dataset}"),
                dataset=dataset,
                base=base,
            )
            for dataset in DATASETS
        }
        policies = {values[dataset]["policy_version"] for dataset in DATASETS}
        if len(policies) != 1:
            raise JointCurveError(
                f"step {ordinal}: HotpotQA and TriviaQA policy receipts differ: "
                f"{sorted(policies)!r}"
            )
        policy_version = next(iter(policies))
        expected_policy = step.get("expected_policy_version")
        if expected_policy is not None and expected_policy != policy_version:
            raise JointCurveError(
                f"step {ordinal}: expected policy {expected_policy!r}, got "
                f"{policy_version!r}"
            )
        for dataset in DATASETS:
            task_ids = tuple(values[dataset].pop("task_ids"))
            if dataset not in fixed_task_ids:
                fixed_task_ids[dataset] = task_ids
            elif fixed_task_ids[dataset] != task_ids:
                raise JointCurveError(
                    f"step {ordinal}: {DATASET_LABELS[dataset]} task IDs differ "
                    "from the fixed first-step selection"
                )
        macro_em = sum(
            values[dataset]["strict_exact_match"] for dataset in DATASETS
        ) / len(DATASETS)
        macro_f1 = sum(
            values[dataset]["strict_token_f1"] for dataset in DATASETS
        ) / len(DATASETS)
        steps.append(
            {
                "step": ordinal,
                "label": str(step.get("label", f"step{ordinal}")),
                "policy_version": policy_version,
                "policy_adapter": step.get("expected_policy_adapter"),
                "datasets": values,
                "macro_average": {
                    "strict_exact_match": macro_em,
                    "strict_token_f1": macro_f1,
                    "strict_exact_match_percent": 100.0 * macro_em,
                    "strict_token_f1_percent": 100.0 * macro_f1,
                },
            }
        )

    steps.sort(key=lambda value: value["step"])
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_scope": "strict_heldout_answer_exact_match_and_token_f1",
        "macro_average": "unweighted_mean_over_HotpotQA_and_TriviaQA",
        "fixed_task_ids_verified": True,
        "policy_receipts_verified": True,
        "evaluator_receipts_verified": True,
        "fixed_task_ids": {
            dataset: list(fixed_task_ids[dataset]) for dataset in DATASETS
        },
        "expected_evaluator_versions": dict(EXPECTED_EVALUATORS),
        "steps": steps,
        "generated_at": _utc_now(),
    }


def _csv_rows(curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = []
    for step in curve["steps"]:
        hotpot = step["datasets"]["hotpotqa"]
        trivia = step["datasets"]["triviaqa"]
        macro = step["macro_average"]
        values.append(
            {
                "step": step["step"],
                "label": step["label"],
                "policy_version": step["policy_version"],
                "policy_adapter": step.get("policy_adapter") or "",
                "hotpotqa_task_count": hotpot["task_count"],
                "hotpotqa_strict_em": hotpot["strict_exact_match"],
                "hotpotqa_strict_f1": hotpot["strict_token_f1"],
                "hotpotqa_strict_em_percent": hotpot["strict_exact_match_percent"],
                "hotpotqa_strict_f1_percent": hotpot["strict_token_f1_percent"],
                "triviaqa_task_count": trivia["task_count"],
                "triviaqa_strict_em": trivia["strict_exact_match"],
                "triviaqa_strict_f1": trivia["strict_token_f1"],
                "triviaqa_strict_em_percent": trivia["strict_exact_match_percent"],
                "triviaqa_strict_f1_percent": trivia["strict_token_f1_percent"],
                "macro_strict_em": macro["strict_exact_match"],
                "macro_strict_f1": macro["strict_token_f1"],
                "macro_strict_em_percent": macro["strict_exact_match_percent"],
                "macro_strict_f1_percent": macro["strict_token_f1_percent"],
            }
        )
    return values


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise JointCurveError("cannot write an empty curve CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_curve(curve: Mapping[str, Any], path: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed"

    steps = [step["step"] for step in curve["steps"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, metric, title in (
        (axes[0], "strict_exact_match_percent", "Strict exact match"),
        (axes[1], "strict_token_f1_percent", "Strict token F1"),
    ):
        for dataset in DATASETS:
            axis.plot(
                steps,
                [step["datasets"][dataset][metric] for step in curve["steps"]],
                marker="o",
                label=DATASET_LABELS[dataset],
            )
        axis.plot(
            steps,
            [step["macro_average"][metric] for step in curve["steps"]],
            marker="o",
            linestyle="--",
            label="Macro average",
        )
        axis.set_title(title)
        axis.set_xlabel("Policy step")
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best")
    figure.suptitle("HotpotQA + TriviaQA fixed held-out policy curve")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return None


def write_curve_outputs(
    curve: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Write canonical JSON/CSV and an optional Matplotlib PNG."""

    target = Path(output_dir).expanduser().resolve()
    json_path = target / "joint_qa_curve.json"
    csv_path = target / "joint_qa_curve.csv"
    png_path = target / "joint_qa_curve.png"
    _atomic_csv(csv_path, _csv_rows(curve))
    plot_skipped_reason = _plot_curve(curve, png_path)
    output = dict(curve)
    output["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path) if plot_skipped_reason is None else None,
        "plot_skipped_reason": plot_skipped_reason,
    }
    _atomic_json(json_path, output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="joint curve input JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec).expanduser().resolve()
    try:
        curve = build_joint_curve(_read_json(spec_path), base_dir=spec_path.parent)
        output = write_curve_outputs(curve, args.output_dir)
    except JointCurveError as exc:
        parser.error(str(exc))
    print(json.dumps(output["artifacts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
