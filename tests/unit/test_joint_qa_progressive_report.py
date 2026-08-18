from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "report_joint_qa_progressive_experiment.py"
_SPEC = importlib.util.spec_from_file_location(
    "report_joint_qa_progressive_experiment", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _paired(
    dataset: str, values: list[tuple[float, float]], *, suffix: str = ""
) -> list[dict]:
    return [
        {
            "task_id": f"{dataset}:q{index}{suffix}",
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": exact_match,
                "token_f1": token_f1,
            },
        }
        for index, (exact_match, token_f1) in enumerate(values)
    ]


def _execution(
    *,
    agent_id: str,
    model_id: str,
    family: str,
    revision: int,
    upstream: list[dict] | None = None,
) -> dict:
    return {
        "execution_id": f"execution-{agent_id}",
        "experiment_id": "unit-test",
        "graph_revision": revision,
        "agent_id": agent_id,
        "model_id": model_id,
        "model_fingerprint": f"fingerprint-{model_id}",
        "provider": f"provider-{family}",
        "request_hash": f"request-{agent_id}",
        "output": f"artifact-{agent_id}",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 20.0,
        "metadata": {
            "request": {
                "graph_revision": revision,
                "model": {"metadata": {"family": family}},
                "upstream": upstream or [],
            }
        },
    }


def _trajectory(
    *,
    task_id: str,
    policy: str,
    adapter: str,
    families: list[str],
) -> dict:
    revision = len(families) + max(0, len(families) - 1) + 1
    nodes = [
        {
            "id": f"a{index}",
            "model_id": f"model-{family}",
            "contract": f"contract-{index}",
            "role_family": "format" if index == len(families) - 1 else "evidence",
        }
        for index, family in enumerate(families)
    ]
    relations = [
        {
            "source_id": f"a{index}",
            "target_id": f"a{index + 1}",
            "source_to_target": True,
            "target_to_source": False,
        }
        for index in range(len(families) - 1)
    ]
    executions = []
    for index, family in enumerate(families):
        upstream = []
        if index:
            upstream.append(
                {
                    "source_agent_id": f"a{index - 1}",
                    "target_agent_id": f"a{index}",
                    "content": f"artifact-a{index - 1}",
                    "graph_revision": revision,
                }
            )
        executions.append(
            _execution(
                agent_id=f"a{index}",
                model_id=f"model-{family}",
                family=family,
                revision=revision,
                upstream=upstream,
            )
        )
    return {
        "task": {"task_id": task_id},
        "versions": {"policy": policy},
        "explicit_finish": True,
        "termination_reason": "finish",
        "turns": [
            {
                "action": {"action": "add_subgraph"},
                "canvas_feedback": "workflow finished",
                "graph_snapshot": {
                    "nodes": nodes,
                    "relations": relations,
                    "output_agent_id": nodes[-1]["id"],
                    "revision": revision,
                },
                "graph_revision": revision,
                "prompt": "director prompt",
                "prompt_token_ids": [1, 2, 3],
                "output_token_ids": [4, 5],
                "director_request_id": f"director-{task_id}",
                "director_latency_ms": 15.0,
                "policy_adapter": adapter,
                "executions": executions,
            }
        ],
    }


def _spec(tmp_path: Path) -> dict:
    main_metrics = tmp_path / "hotpot_main.jsonl"
    skill_off = tmp_path / "hotpot_skill_off.jsonl"
    skill_on = tmp_path / "hotpot_skill_on.jsonl"
    trajectories = tmp_path / "hotpot_trajectories.jsonl"
    wrong = tmp_path / "hotpot_wrong.jsonl"
    _write_jsonl(main_metrics, _paired("hotpotqa", [(1.0, 1.0), (0.0, 0.5)]))
    _write_jsonl(skill_off, _paired("hotpotqa", [(0.0, 0.0), (1.0, 0.5)]))
    _write_jsonl(skill_on, _paired("hotpotqa", [(1.0, 1.0), (1.0, 1.0)]))
    _write_jsonl(
        trajectories,
        [
            _trajectory(
                task_id="hotpotqa:q0",
                policy="joint-step-0",
                adapter="theta-step-0",
                families=["qwen", "deepseek"],
            ),
            _trajectory(
                task_id="hotpotqa:q1",
                policy="joint-step-0",
                adapter="theta-step-0",
                families=["qwen", "gemini", "qwen"],
            ),
        ],
    )
    _write_jsonl(wrong, [{"task_id": "hotpotqa:q1", "failure_type": "wrong"}])
    return {
        "schema_version": _MODULE.INPUT_SCHEMA_VERSION,
        "title": "联合渐进式编排单测",
        "demo_limit": 2,
        "steps": [
            {
                "step": 0,
                "label": "Step0",
                "policy_version": "joint-step-0",
                "policy_adapter": "theta-step-0",
                "datasets": {
                    "hotpotqa": {
                        "metrics_path": main_metrics.name,
                        "trajectory_path": trajectories.name,
                        "wrong_demos_path": wrong.name,
                        "skill_comparison": {
                            "off_metrics_path": skill_off.name,
                            "on_metrics_path": skill_on.name,
                        },
                    }
                },
            },
            {
                "step": 1,
                "label": "Step1 pending",
                "policy_version": "joint-step-1",
                "policy_adapter": "theta-step-1",
                "datasets": {},
            },
        ],
    }


def test_report_aggregates_metrics_graph_models_usage_skill_and_demos(tmp_path):
    report = _MODULE.build_progressive_report(_spec(tmp_path), base_dir=tmp_path)

    step0 = report["steps"][0]
    hotpot = step0["datasets"]["hotpotqa"]
    assert step0["training_curve_coordinate"] == {
        "step": 0,
        "policy": "joint-step-0",
        "adapter": "theta-step-0",
    }
    assert hotpot["metrics"]["strict_exact_match"] == 0.5
    assert hotpot["metrics"]["strict_token_f1"] == 0.75
    assert hotpot["graph"]["mean_agent_count"] == 2.5
    assert hotpot["graph"]["mean_structural_depth"] == 2.5
    assert hotpot["graph"]["mean_effective_dependency_depth"] == 2.5
    assert hotpot["graph"]["topology_family_distribution"] == {
        "serial_2": 1,
        "serial_3_plus": 1,
    }
    assert hotpot["models"]["recorded_model_family_call_distribution"] == {
        "deepseek": 1,
        "gemini": 1,
        "qwen": 3,
    }
    assert hotpot["models"]["recorded_model_family_cooccurrence_by_trajectory"] == {
        "deepseek + qwen": 1,
        "gemini + qwen": 1,
    }
    assert hotpot["usage"]["api_call_receipt_count"] == 7
    assert hotpot["usage"]["director_input_tokens"]["value"] == 6
    assert hotpot["usage"]["executor_output_tokens"]["value"] == 25
    assert hotpot["skill_effect"]["delta_exact_match"] == 0.5
    assert hotpot["skill_effect"]["delta_token_f1"] == 0.75
    assert hotpot["skill_effect"]["paired_task_ids_verified"] is True
    assert hotpot["demos"][0]["task_id"] == "hotpotqa:q1"
    assert hotpot["demos"][0]["trajectory_artifact"]["jsonl_line"] == 2


def test_missing_dataset_and_future_step_remain_unavailable(tmp_path):
    report = _MODULE.build_progressive_report(_spec(tmp_path), base_dir=tmp_path)

    assert report["steps"][0]["datasets"]["triviaqa"]["status"] == "unavailable"
    assert report["steps"][1]["datasets"]["hotpotqa"]["metrics"]["status"] == (
        "unavailable"
    )
    assert report["steps"][1]["datasets"]["triviaqa"]["status"] == "unavailable"


def test_skill_delta_is_unavailable_when_task_ids_do_not_match(tmp_path):
    spec = _spec(tmp_path)
    comparison = spec["steps"][0]["datasets"]["hotpotqa"]["skill_comparison"]
    mismatched = tmp_path / comparison["on_metrics_path"]
    _write_jsonl(
        mismatched,
        _paired("hotpotqa", [(1.0, 1.0), (1.0, 1.0)], suffix="-other"),
    )

    report = _MODULE.build_progressive_report(spec, base_dir=tmp_path)

    skill = report["steps"][0]["datasets"]["hotpotqa"]["skill_effect"]
    assert skill["status"] == "unavailable"
    assert "task_id" in skill["reason"]
    assert "delta_exact_match" not in skill


def test_outputs_include_json_csv_and_chinese_markdown(tmp_path):
    report = _MODULE.build_progressive_report(_spec(tmp_path), base_dir=tmp_path)
    output = _MODULE.write_progressive_report_outputs(report, tmp_path / "report")

    artifacts = output["artifacts"]
    assert Path(artifacts["json"]).is_file()
    assert Path(artifacts["csv"]).is_file()
    assert Path(artifacts["chinese_markdown"]).is_file()
    csv_text = Path(artifacts["csv"]).read_text(encoding="utf-8")
    markdown = Path(artifacts["chinese_markdown"]).read_text(encoding="utf-8")
    assert "step, label".replace(" ", "") in csv_text.replace(" ", "")
    assert "policy_version" in csv_text
    assert "unavailable" in csv_text
    assert "Skill-on/off" in markdown
    assert "完整 Demo artifact 引用" in markdown


def test_declared_policy_must_match_trajectory_receipt(tmp_path):
    spec = _spec(tmp_path)
    spec["steps"][0]["policy_version"] = "wrong-policy"

    with pytest.raises(_MODULE.ProgressiveReportError, match="contradicts receipts"):
        _MODULE.build_progressive_report(spec, base_dir=tmp_path)
