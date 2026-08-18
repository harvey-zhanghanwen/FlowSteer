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
                "phase": "single",
                "rendered_messages": [
                    {"role": "user", "content": f"input-for-{agent_id}"}
                ],
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
        "task": {
            "task_id": task_id,
            "question": f"question-for-{task_id}",
            "ground_truth": f"ground-truth-for-{task_id}",
        },
        "final_answer": f"final-answer-for-{task_id}",
        "evaluation": {
            "valid": True,
            "reason": "official answer evaluator",
        },
        "terminal_failure": False,
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
    assert hotpot["demos"][0]["question"] == "question-for-hotpotqa:q1"
    assert hotpot["demos"][0]["ground_truth"] == "ground-truth-for-hotpotqa:q1"
    assert hotpot["demos"][0]["final_answer"] == "final-answer-for-hotpotqa:q1"
    assert hotpot["demos"][0]["director_actions"][0]["action"] == {
        "action": "add_subgraph"
    }
    assert hotpot["demos"][0]["agents"][-1] == {
        "agent_id": "a2",
        "role_family": "format",
        "model_id": "model-qwen",
        "contract": "contract-2",
    }
    assert hotpot["demos"][0]["directed_communication"][-1]["content"] == (
        "artifact-a1"
    )
    assert hotpot["demos"][0]["output_agent_inbox"]["agent_id"] == "a2"
    assert hotpot["demos"][0]["failure_origin"] == {
        "observed_failure_boundary": "final_answer_vs_ground_truth",
        "recorded_failure_type": "wrong",
        "evaluation_valid": True,
        "evaluation_reason": "official answer evaluator",
        "terminal_failure": False,
        "failed_execution_receipts": [],
        "causal_root_cause": "unavailable",
        "attribution_scope": (
            "recorded_failure_signals_only; no causal root cause is inferred"
        ),
    }


def test_missing_dataset_and_future_step_remain_unavailable(tmp_path):
    report = _MODULE.build_progressive_report(_spec(tmp_path), base_dir=tmp_path)

    assert report["steps"][0]["datasets"]["triviaqa"]["status"] == "unavailable"
    assert report["steps"][0]["macro_metrics"]["status"] == "unavailable"
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
    assert "完整 Demo 展开" in markdown
    assert "single-point training diagnostics" in markdown


def test_macro_skill_and_single_point_training_receipts_are_projected(tmp_path):
    spec = _spec(tmp_path)
    trivia_metrics = tmp_path / "trivia_main.jsonl"
    trivia_trajectories = tmp_path / "trivia_trajectories.jsonl"
    _write_jsonl(trivia_metrics, _paired("triviaqa", [(1.0, 0.5), (1.0, 1.0)]))
    _write_jsonl(
        trivia_trajectories,
        [
            _trajectory(
                task_id="triviaqa:q0",
                policy="joint-step-0",
                adapter="theta-step-0",
                families=["qwen", "gpt"],
            ),
            _trajectory(
                task_id="triviaqa:q1",
                policy="joint-step-0",
                adapter="theta-step-0",
                families=["deepseek", "qwen"],
            ),
        ],
    )
    step = spec["steps"][0]
    step["datasets"]["triviaqa"] = {
        "metrics_path": trivia_metrics.name,
        "trajectory_path": trivia_trajectories.name,
        "demo_limit": 1,
    }

    publication = tmp_path / "publication_results.json"
    store = tmp_path / "skills.json"
    manifest = tmp_path / "training_manifest.json"
    summary = tmp_path / "training_summary.json"
    sync = tmp_path / "sync_receipt.json"
    publications = {}
    current = {}
    for dataset in ("hotpotqa", "triviaqa"):
        skill_id = f"jointqa.{dataset}.answer_span"
        skill = {
            "skill_id": skill_id,
            "status": "active",
            "version": 3,
            "eligible_epoch": 4,
            "activated_epoch": 4,
            "condition": {"task_family": dataset, "graph_stage": "*"},
            "evidence": {
                "baseline": "frozen_step0",
                "effective_pairs": 20,
                "paired_effect_mean": 0.1,
                "calibrated_lower": 0.04,
                "calibrated_upper": 0.2,
                "harm_probability": 0.01,
                "empirical_coverage": 0.95,
                "heldout_task_families": [dataset],
                "validation_splits": ["skill_confirmation"],
                "evidence_ids": [f"probe-{dataset}-0", f"probe-{dataset}-1"],
            },
        }
        publications[dataset] = {
            "selected_candidate": "answer_span",
            "gate": {
                "approved": True,
                "no_practical_value": False,
                "reasons": [],
            },
            "skill": skill,
        }
        current[skill_id] = skill
    publication.write_text(
        json.dumps(
            {
                "experiment_version": "skill-epoch-2",
                "active_datasets": ["hotpotqa", "triviaqa"],
                "causal_estimand": "paired prompt-prior intent-to-treat effect",
                "publications": publications,
            }
        ),
        encoding="utf-8",
    )
    store.write_text(json.dumps({"current": current}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "skills_enabled": True,
                "post_update_canaries": {
                    "collected": 2,
                    "policy_version": "joint-step-1",
                    "adapter_name": "theta-step-1",
                    "trajectory_ids": ["canary-hotpot", "canary-trivia"],
                },
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "optimizer_updates": 1,
                "loss": 0.25,
                "grad_norm": 1.5,
                "trainable_update_l2": 0.03,
                "behavior_policy_version": "joint-step-0",
                "updated_policy_version": "joint-step-1",
                "informative_groups": 2,
                "trained_groups": 2,
                "trained_trajectories": 16,
                "oom_backoff_count": 0,
            }
        ),
        encoding="utf-8",
    )
    sync.write_text(
        json.dumps(
            {
                "status": "published",
                "success": True,
                "adapter_name": "theta-step-1",
                "behavior_policy_version": "joint-step-0",
                "candidate_policy_version": "joint-step-1",
                "checkpoint_path": "/checkpoint/theta-step-1",
                "canary_succeeded": True,
            }
        ),
        encoding="utf-8",
    )
    step.update(
        {
            "skill_publication_path": publication.name,
            "skill_store_path": store.name,
            "training_manifest_path": manifest.name,
            "training_summary_path": summary.name,
            "sync_receipt_path": sync.name,
        }
    )

    report = _MODULE.build_progressive_report(spec, base_dir=tmp_path)
    step0 = report["steps"][0]
    assert step0["macro_metrics"]["strict_exact_match"] == 0.75
    assert step0["macro_metrics"]["strict_token_f1"] == 0.75
    hotpot_skill = step0["skill"]["datasets"]["hotpotqa"]
    assert hotpot_skill["skill_id"] == "jointqa.hotpotqa.answer_span"
    assert hotpot_skill["skill_status"] == "active"
    assert hotpot_skill["evidence"]["effective_pairs"] == 20
    training = step0["training_diagnostics"]
    assert training["optimizer"] == {
        "optimizer_updates": 1,
        "loss": 0.25,
        "grad_norm": 1.5,
        "trainable_update_l2": 0.03,
        "behavior_policy_version": "joint-step-0",
        "updated_policy_version": "joint-step-1",
        "informative_groups": 2,
        "trained_groups": 2,
        "trained_trajectories": 16,
        "oom_backoff_count": 0,
        "status": "available",
    }
    assert training["synchronization"]["success"] is True
    assert training["canary"]["sync_canary_succeeded"] is True
    assert training["canary"]["post_update_collected"] == 2
    assert training["multi_point_training_curve_claimed"] is False
    assert report["multi_point_training_curve_claimed"] is False

    rows = _MODULE._csv_rows(report)
    assert rows[0]["macro_strict_exact_match"] == 0.75
    assert rows[0]["optimizer_updates"] == 1
    assert rows[0]["published_skill_status"] == "active"


def test_declared_policy_must_match_trajectory_receipt(tmp_path):
    spec = _spec(tmp_path)
    spec["steps"][0]["policy_version"] = "wrong-policy"

    with pytest.raises(_MODULE.ProgressiveReportError, match="contradicts receipts"):
        _MODULE.build_progressive_report(spec, base_dir=tmp_path)
