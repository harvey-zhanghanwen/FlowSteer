from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.task_evaluator import (
    HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    TRIVIAQA_ANSWER_EVALUATOR_VERSION,
)


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "aggregate_joint_qa_curve.py"
_SPEC = importlib.util.spec_from_file_location("aggregate_joint_qa_curve", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _paired(dataset: str, values: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "task_id": f"{dataset}:{index}",
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": exact_match,
                "token_f1": token_f1,
            },
        }
        for index, (exact_match, token_f1) in enumerate(values)
    ]


def _receipts(
    dataset: str,
    values: list[tuple[float, float]],
    *,
    policy: str,
    evaluator: str,
) -> list[dict]:
    return [
        {
            "task": {"task_id": f"{dataset}:{index}"},
            "sampling_receipt_verified": True,
            "versions": {"policy": policy, "evaluator": evaluator},
            "evaluation": {
                "valid": True,
                "evaluator_version": evaluator,
                "metrics": {"exact_match": exact_match, "token_f1": token_f1},
            },
        }
        for index, (exact_match, token_f1) in enumerate(values)
    ]


def _dataset_files(
    root: Path,
    *,
    step: int,
    dataset: str,
    values: list[tuple[float, float]],
    policy: str,
    evaluator: str,
) -> dict[str, str]:
    paired = root / f"{dataset}_step{step}_paired.jsonl"
    receipts = root / f"{dataset}_step{step}_receipts.jsonl"
    _write_jsonl(paired, _paired(dataset, values))
    _write_jsonl(
        receipts,
        _receipts(dataset, values, policy=policy, evaluator=evaluator),
    )
    return {
        "metrics_path": paired.name,
        "trajectory_receipts_path": receipts.name,
    }


def _spec(tmp_path: Path) -> dict:
    steps = []
    values = (
        ([(1.0, 1.0), (0.0, 0.5)], [(0.0, 0.0), (1.0, 0.8)]),
        ([(1.0, 1.0), (1.0, 1.0)], [(1.0, 0.7), (1.0, 0.9)]),
    )
    for step, (hotpot_values, trivia_values) in enumerate(values):
        policy = f"joint-step-{step}"
        steps.append(
            {
                "step": step,
                "expected_policy_version": policy,
                "datasets": {
                    "hotpotqa": _dataset_files(
                        tmp_path,
                        step=step,
                        dataset="hotpotqa",
                        values=hotpot_values,
                        policy=policy,
                        evaluator=HOTPOTQA_ANSWER_EVALUATOR_VERSION,
                    ),
                    "triviaqa": _dataset_files(
                        tmp_path,
                        step=step,
                        dataset="triviaqa",
                        values=trivia_values,
                        policy=policy,
                        evaluator=TRIVIAQA_ANSWER_EVALUATOR_VERSION,
                    ),
                },
            }
        )
    return {
        "schema_version": _MODULE.INPUT_SCHEMA_VERSION,
        "steps": steps,
    }


def test_paired_results_produce_per_dataset_and_macro_curve(tmp_path):
    curve = _MODULE.build_joint_curve(_spec(tmp_path), base_dir=tmp_path)

    assert curve["fixed_task_ids_verified"] is True
    assert curve["fixed_task_ids"]["hotpotqa"] == ["hotpotqa:0", "hotpotqa:1"]
    assert curve["steps"][0]["datasets"]["hotpotqa"]["strict_exact_match"] == 0.5
    assert curve["steps"][0]["datasets"]["triviaqa"]["strict_token_f1"] == 0.4
    assert curve["steps"][0]["macro_average"]["strict_exact_match"] == 0.5
    assert curve["steps"][0]["macro_average"]["strict_token_f1"] == 0.575
    assert curve["steps"][1]["macro_average"]["strict_exact_match"] == 1.0


def test_report_json_uses_strict_metrics_and_checks_policy_receipts(tmp_path):
    spec = _spec(tmp_path)
    dataset = spec["steps"][0]["datasets"]["hotpotqa"]
    report_path = tmp_path / "hotpot_report.json"
    report_path.write_text(
        json.dumps(
            {
                "dataset": "HotpotQA",
                "sample_count": 2,
                "policy_version": "joint-step-0",
                "agentgraph": {
                    "denominator": 2,
                    "completed": 2,
                    "evaluator_valid": 2,
                    "strict_exact_match": 0.5,
                    "strict_token_f1": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    dataset["metrics_path"] = report_path.name

    curve = _MODULE.build_joint_curve(spec, base_dir=tmp_path)

    result = curve["steps"][0]["datasets"]["hotpotqa"]
    assert result["metrics_source_kind"] == "round_report_json"
    assert result["strict_token_f1"] == 0.75
    assert result["policy_receipt_verified"] is True


def test_cross_step_task_selection_change_is_rejected(tmp_path):
    spec = _spec(tmp_path)
    receipt_path = (
        tmp_path / spec["steps"][1]["datasets"]["triviaqa"]["trajectory_receipts_path"]
    )
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    receipts[1]["task"]["task_id"] = "triviaqa:different"
    _write_jsonl(receipt_path, receipts)
    paired_path = tmp_path / spec["steps"][1]["datasets"]["triviaqa"]["metrics_path"]
    paired = [json.loads(line) for line in paired_path.read_text().splitlines()]
    paired[1]["task_id"] = "triviaqa:different"
    _write_jsonl(paired_path, paired)

    with pytest.raises(_MODULE.JointCurveError, match="fixed first-step selection"):
        _MODULE.build_joint_curve(spec, base_dir=tmp_path)


def test_wrong_evaluator_or_cross_dataset_policy_is_rejected(tmp_path):
    spec = _spec(tmp_path)
    receipt_path = (
        tmp_path / spec["steps"][0]["datasets"]["triviaqa"]["trajectory_receipts_path"]
    )
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    receipts[0]["evaluation"]["evaluator_version"] = "wrong-evaluator"
    _write_jsonl(receipt_path, receipts)
    with pytest.raises(_MODULE.JointCurveError, match="does not match"):
        _MODULE.build_joint_curve(spec, base_dir=tmp_path)

    spec = _spec(tmp_path)
    receipt_path = (
        tmp_path / spec["steps"][0]["datasets"]["triviaqa"]["trajectory_receipts_path"]
    )
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    for receipt in receipts:
        receipt["versions"]["policy"] = "different-policy"
    _write_jsonl(receipt_path, receipts)
    with pytest.raises(_MODULE.JointCurveError, match="policy receipts differ"):
        _MODULE.build_joint_curve(spec, base_dir=tmp_path)


def test_outputs_include_json_csv_and_optional_png(tmp_path):
    curve = _MODULE.build_joint_curve(_spec(tmp_path), base_dir=tmp_path)
    output = _MODULE.write_curve_outputs(curve, tmp_path / "curve")

    assert Path(output["artifacts"]["json"]).is_file()
    assert Path(output["artifacts"]["csv"]).is_file()
    csv_text = Path(output["artifacts"]["csv"]).read_text(encoding="utf-8")
    assert "macro_strict_em_percent" in csv_text
    if output["artifacts"]["png"] is not None:
        assert Path(output["artifacts"]["png"]).is_file()
