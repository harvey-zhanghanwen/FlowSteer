"""Offline runner integration for isolated, unvalidated candidate conditions.

Uses the fake Backend/TrajectoryRecord boundary already exercised by
test_hotpotqa_round; no model, grader, database, or Git command is invoked.
"""

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.healthbench_candidate_skill_profile import build_candidate_prompt_priors
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.records import EvaluationReceipt, TrajectoryRecord


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/healthbench_candidate_skills_v241.yaml"
SPEC = importlib.util.spec_from_file_location(
    "healthbench_v241_candidate_runner_under_test",
    ROOT / "scripts/evaluate_completion_benchmark_round.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _task():
    return RUNNER.TaskRecord(
        task_id="healthbench-professional:synthetic-runner-case",
        question="Continue this synthetic conversation.",
        ground_truth="evaluator-only synthetic target",
        split="test",
        metadata={"dataset_key": "healthbench_professional"},
    )


def _config(enabled=True):
    config = load_yaml(
        ROOT / "config/evaluation_healthbench_professional_query_recovery_v2_40_dev5.yaml"
    )
    config["candidate_skill_evaluation"] = {
        "enabled": enabled,
        "profile_path": str(PROFILE),
    }
    return config


@pytest.mark.parametrize("with_candidates", (False, True))
def test_shared_collection_forwarding_preserves_default_and_probe_receipts(tmp_path, with_candidates):
    task = _task()
    calls = []
    priors = build_candidate_prompt_priors(load_yaml(PROFILE)) if with_candidates else ()

    class Backend:
        model_catalog_version = "synthetic-catalog"
        evidence_store = SimpleNamespace(trajectories=SimpleNamespace(payloads=lambda: ()))

        async def collect(self, selected_task, rollout_index, versions, **kwargs):
            calls.append(kwargs)
            assert selected_task == task and rollout_index == 0
            return TrajectoryRecord(
                trajectory_id="synthetic-candidate-trajectory",
                task=selected_task,
                group_id="synthetic-group",
                condition_id="synthetic-condition",
                rollout_id="synthetic-rollout",
                versions=versions,
                turns=(),
                final_answer="synthetic complete response",
                evaluation=EvaluationReceipt(
                    versions.evaluator, True, 0.5,
                    metrics={"overall_score": 0.5, "overall_score_length_adjusted": 0.4},
                ),
                termination_reason="finish",
                explicit_finish=True,
                forced_probe=kwargs.get("forced_probe", False),
            )

    config = {
        "experiment": {
            "condition_id": "synthetic-condition",
            "prompt_version": "synthetic-prompt",
            "tool_version": "synthetic-tool",
        },
        "director": {"behavior_policy_version": "synthetic-policy"},
        "hotpotqa_evaluation": {"concurrency": 1, "split": "test"},
    }
    failure_records = []
    result = asyncio.run(RUNNER._collect_graph(
        Backend(), (task,), config, tmp_path / "trajectories.jsonl",
        failure_records, {}, tmp_path / "manifest.json",
        **({"prompt_priors": priors} if with_candidates else {}),
    ))
    assert not failure_records
    assert len(calls) == 1
    assert calls[0]["expected_task_split"] == "test"
    if with_candidates:
        assert calls[0]["prompt_priors"] == priors
        assert calls[0]["forced_probe"] is True
    else:
        assert set(calls[0]) == {"expected_task_split"}
    receipt = result[task.task_id]
    assert receipt["forced_probe"] is with_candidates
    assert receipt["grpo_eligible"] is False
    assert receipt["retrieved_skill_ids"] == []
    assert receipt["active_skill_ids"] == []
    persisted = json.loads((tmp_path / "trajectories.jsonl").read_text().strip())
    assert persisted == receipt


@pytest.mark.parametrize("section", ("grpo", "policy_sync", "exploration", "skills"))
def test_completion_config_keeps_candidate_inference_only(section):
    config = _config()
    RUNNER.validate_completion_benchmark_config(config)
    config[section]["enabled"] = True
    with pytest.raises(ConfigurationError):
        RUNNER.validate_completion_benchmark_config(config)


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    config = _config()
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "selected", "direct", "trajectories", "failures", "paired", "wrong",
            "manifest", "preflight", "report_json", "report_markdown",
        )
    }
    monkeypatch.setattr(RUNNER, "load_yaml", lambda _path: deepcopy(config))
    monkeypatch.setattr(RUNNER, "_validate_runtime_dataset_registry", lambda *_: {})
    monkeypatch.setattr(RUNNER, "_paths", lambda *_: paths)
    monkeypatch.setattr(RUNNER, "_select_tasks", lambda *_: (_task(),))
    monkeypatch.setattr(RUNNER, "_healthbench_direct_reference", lambda *_: None)
    monkeypatch.setattr(RUNNER, "_git_state", lambda *_: {"branch": "synthetic", "commit": "synthetic"})

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("prepare-only must not instantiate a model/grader runtime")

    monkeypatch.setattr(RUNNER.LiveSmokeBackend, "from_config", unexpected_runtime)

    def prepare(arm="agentgraph"):
        return asyncio.run(RUNNER.run_completion_benchmark_round(
            tmp_path / "synthetic.yaml", project_root=tmp_path,
            prepare_only=True, collection_arm=arm,
        ))

    return config, paths, prepare


def test_prepare_freezes_full_candidate_profile_without_runtime(preparation):
    _, paths, prepare = preparation
    manifest = prepare()
    candidate = manifest["candidate_skill_evaluation"]
    assert manifest["status"] == "prepared"
    assert manifest["training_enabled"] is False
    assert manifest["optimizer_updates"] == 0
    assert candidate["enabled"] is True
    assert candidate["profile"] == load_yaml(PROFILE)
    assert candidate["prompt_priors"] == list(build_candidate_prompt_priors(load_yaml(PROFILE)))
    assert candidate["mode"] == "candidate_prompt_prior"
    assert candidate["publication_performed"] is False
    assert json.loads(paths["manifest"].read_text())["candidate_skill_evaluation"] == candidate
    assert prepare()["candidate_skill_evaluation"] == candidate


@pytest.mark.parametrize("arm", ("direct", "both"))
def test_candidate_prepare_requires_separate_agentgraph_arm(preparation, arm):
    _, paths, prepare = preparation
    with pytest.raises(ConfigurationError, match="collection_arm=agentgraph"):
        prepare(arm)
    assert not paths["manifest"].exists()


def test_changed_candidate_profile_cannot_overwrite_frozen_manifest(preparation):
    _, paths, prepare = preparation
    prepare()
    frozen = json.loads(paths["manifest"].read_text())
    frozen["candidate_skill_evaluation"]["profile"]["profile_version"] = "different-profile"
    paths["manifest"].write_text(json.dumps(frozen))
    before = paths["manifest"].read_text()
    with pytest.raises(ConfigurationError, match="candidate profile changed"):
        prepare()
    assert paths["manifest"].read_text() == before


def test_candidate_off_cannot_resume_in_candidate_on_namespace(preparation):
    config, paths, prepare = preparation
    prepare()
    before = paths["manifest"].read_text()
    config["candidate_skill_evaluation"]["enabled"] = False
    with pytest.raises(ConfigurationError, match="candidate"):
        prepare()
    assert paths["manifest"].read_text() == before


@pytest.mark.parametrize("candidate_enabled", (False, True))
def test_report_labels_candidates_not_active_skills(candidate_enabled):
    config = _config(candidate_enabled)
    row = {
        "task_id": _task().task_id,
        "failure_type": "synthetic_same_score",
        "direct": {"available": True, "valid": True,
                   "overall_score": 0.5, "overall_score_length_adjusted": 0.4},
        "agentgraph": {"available": True, "valid": True, "explicit_finish": True,
                       "overall_score": 0.5, "overall_score_length_adjusted": 0.4},
    }
    report = RUNNER._report([row], config)
    markdown = RUNNER._report_markdown(report)
    assert report["training_performed"] is False
    assert "Only evidence-gated ACTIVE Skills" not in markdown
    if candidate_enabled:
        assert report["skill_evaluation_mode"] == "candidate_prompt_prior"
        assert "Unvalidated, rejectable candidate" in markdown
        assert "no ACTIVE Skill was published" in markdown
    else:
        assert report["skill_evaluation_mode"] == "memory_off"
        assert "No Skill was injected" in markdown
