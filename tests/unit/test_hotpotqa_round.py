from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from src.interactive.config_loader import load_yaml
from src.interactive.persistence import EvidenceStore
from src.interactive.records import EvaluationReceipt, TrajectoryRecord, TurnRecord
from src.interactive.task_evaluator import EvaluationOutcome


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_hotpotqa_round.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_hotpotqa_round", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_round_config_is_fixed_heldout_and_training_disabled():
    config = load_yaml(_ROOT / "config" / "evaluation_hotpotqa_round_01.yaml")
    _MODULE.validate_hotpot_config(config)


def test_task_id_diagnostic_selection_is_explicit_and_bounded():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_multiagent_v1_diagnostic.yaml"
    )
    _MODULE.validate_hotpot_config(config)

    invalid = deepcopy(config)
    invalid["hotpotqa_evaluation"]["task_ids"] = [
        invalid["hotpotqa_evaluation"]["task_ids"][0]
    ] * 14
    try:
        _MODULE.validate_hotpot_config(invalid)
    except Exception as exc:
        assert "task_ids selection" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("duplicate task IDs were accepted")


def test_declared_direct_reuse_is_copied_without_gateway_call(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:one",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    record = {
        "task_id": task.task_id,
        "model_id": "qwen3.5-9b-local",
        "protocol": "direct-v1",
        "generation_seed": 17,
        "final_answer": "answer",
        "evaluation": {
            "valid": True,
            "evaluator_version": "hotpotqa.official.answer.v1",
        },
        "execution": {"execution_id": "existing"},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    destination = tmp_path / "destination.jsonl"
    stale = dict(record)
    stale["generation_seed"] = 99
    stale["execution"] = {"execution_id": "stale-canary"}
    destination.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = {
        "experiment": {"name": "test", "seed": 99},
        "hotpotqa_evaluation": {
            "direct_model_id": "qwen3.5-9b-local",
            "direct_protocol": "direct-v1",
            "direct_generation_seed": 17,
            "direct_reused_from": source.name,
            "concurrency": 1,
        },
    }
    manifest = {}

    result = asyncio.run(
        _MODULE._collect_direct(
            None,
            (task,),
            config,
            tmp_path,
            destination,
            [],
            manifest,
            manifest_path,
        )
    )

    assert result[task.task_id]["execution"]["execution_id"] == "existing"
    assert result[task.task_id]["reuse_receipt"] == {
        "reused": True,
        "source": str(source),
    }
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest["direct_progress"]["completed"] == 1
    assert manifest["direct_progress"]["reused_from"] == str(source)
    assert manifest["direct_progress"]["reused_records"] == 1
    assert manifest["direct_progress"]["newly_collected_records"] == 0


def test_graph_evaluation_uses_task_local_rollout_zero(tmp_path):
    tasks = tuple(
        _MODULE.TaskRecord(
            task_id=f"hotpotqa:{index}",
            question=f"question {index}",
            ground_truth="answer",
            split="validation",
            metadata={"dataset_key": "hotpotqa"},
        )
        for index in range(3)
    )

    class EmptyTrajectoryStore:
        def payloads(self):
            return ()

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = type("Evidence", (), {"trajectories": EmptyTrajectoryStore()})()

        def __init__(self):
            self.rollout_indices = []

        async def collect(
            self,
            task,
            rollout_index,
            versions,
            *,
            expected_task_split="train",
        ):
            assert expected_task_split == "validation"
            self.rollout_indices.append(rollout_index)
            return _MODULE.TrajectoryRecord(
                trajectory_id=f"trajectory:{task.task_id}",
                task=task,
                group_id=f"{task.task_id}:condition:{versions.policy}",
                condition_id="condition",
                rollout_id=f"{task.task_id}:rollout:0000",
                versions=versions,
                turns=(),
                final_answer="answer",
                evaluation=EvaluationReceipt(
                    versions.evaluator,
                    True,
                    1.0,
                    metrics={"exact_match": 1.0, "token_f1": 1.0},
                ),
                termination_reason="finish",
                explicit_finish=True,
            )

    backend = Backend()
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {"concurrency": 2},
    }
    manifest_path = tmp_path / "manifest.json"
    result = asyncio.run(
        _MODULE._collect_graph(
            backend,
            tasks,
            config,
            tmp_path / "trajectories.jsonl",
            [],
            {},
            manifest_path,
        )
    )

    assert set(result) == {task.task_id for task in tasks}
    assert backend.rollout_indices == [0, 0, 0]


def test_graph_task_timeout_is_an_operational_failure(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:timeout",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )

    class EmptyTrajectoryStore:
        def payloads(self):
            return ()

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = type("Evidence", (), {"trajectories": EmptyTrajectoryStore()})()

        async def collect(
            self,
            task,
            rollout_index,
            versions,
            *,
            expected_task_split="train",
        ):
            await asyncio.Event().wait()

    failures = []
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {
            "concurrency": 1,
            "split": "validation",
            "task_timeout_seconds": 0.01,
        },
    }
    result = asyncio.run(
        _MODULE._collect_graph(
            Backend(),
            (task,),
            config,
            tmp_path / "trajectories.jsonl",
            failures,
            {},
            tmp_path / "manifest.json",
            failure_path=tmp_path / "failures.jsonl",
        )
    )

    assert result == {}
    assert len(failures) == 1
    assert failures[0]["task_id"] == task.task_id
    assert failures[0]["condition"] == "agentgraph"
    assert failures[0]["stage"] == "collect"
    assert "TimeoutError" in failures[0]["error"]
    persisted = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert persisted == failures


def test_graph_resume_rejects_invalid_evaluator_receipt():
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:resume",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    versions = {"policy": "policy-v1", "evaluator": "evaluator-v1"}
    trajectory = {
        "trajectory_id": "trajectory-resume",
        "task": task.to_dict(),
        "condition_id": "condition",
        "versions": versions,
        "turns": [],
        "evaluation": {"valid": False, "evaluator_version": "evaluator-v1"},
        "explicit_finish": True,
    }

    assert not _MODULE._trajectory_resume_matches(
        trajectory,
        task=task,
        condition_id="condition",
        versions=versions,
    )

    trajectory["evaluation"]["valid"] = True
    assert _MODULE._trajectory_resume_matches(
        trajectory,
        task=task,
        condition_id="condition",
        versions=versions,
    )

    trajectory["evaluation"]["evaluator_version"] = "stale-evaluator"
    assert not _MODULE._trajectory_resume_matches(
        trajectory,
        task=task,
        condition_id="condition",
        versions=versions,
    )


def _graph_trajectory(
    task,
    versions,
    *,
    trajectory_id,
    valid,
    trace=(),
):
    graph = {
        "revision": 1,
        "nodes": [],
        "relations": [],
        "output_agent_id": None,
    }
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}",
        round_index=0,
        prompt="frozen Director prompt",
        policy_response='{"action_type":"finish"}',
        prompt_token_ids=(),
        output_token_ids=(),
        behavior_log_probs=(),
        executed_prefix_tokens=0,
        action={"action_type": "finish"},
        canvas_feedback="workflow finished",
        graph_revision=1,
        graph_snapshot=graph,
        policy_version=versions.policy,
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task=task,
        group_id=f"{task.task_id}:condition:{versions.policy}",
        condition_id="condition",
        rollout_id=f"{task.task_id}:condition:{versions.policy}:rollout:0000",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=valid,
            reward=1.0 if valid else None,
            metrics={"exact_match": 1.0} if valid else {},
            reason="evaluated" if valid else "environment_graph_callback_failed",
            details={"trace": list(trace)},
        ),
        termination_reason="finish",
        explicit_finish=True,
    ).to_dict()


def _graph_versions(module, task):
    return module.version_bundle_for(
        task,
        policy_version="policy-v1",
        model_catalog_version="catalog-v1",
        prompt_version="prompt-v1",
        tool_version="tool-v1",
        encoder_version="none",
        feature_schema_version="none",
        posterior_version="none",
        skill_library_version="none",
    )


def test_graph_resume_retries_only_evaluator_and_keeps_frozen_order(tmp_path):
    tasks = tuple(
        _MODULE.TaskRecord(
            task_id=f"hotpotqa:{name}",
            question=f"question {name}",
            ground_truth="answer",
            split="validation",
            metadata={"dataset_key": "hotpotqa"},
        )
        for name in ("valid", "retry")
    )
    versions = {task.task_id: _graph_versions(_MODULE, task) for task in tasks}
    replay_trace = (
        {
            "step": 0,
            "observation": "observation zero",
            "legal_actions": ["click[item]"],
            "action": "click[item]",
            "next_observation": "observation one",
            "reward": 0.0,
            "done": False,
            "info": {},
            "state_advanced": True,
        },
        {
            "step": 1,
            "observation": "observation one",
            "legal_actions": ["click[buy]"],
            "action": "click[buy]",
            "next_observation": "observation two",
            "reward": 0.0,
            "done": False,
            "info": {},
            "state_advanced": True,
        },
    )
    valid = _graph_trajectory(
        tasks[0],
        versions[tasks[0].task_id],
        trajectory_id="trajectory-valid",
        valid=True,
    )
    invalid = _graph_trajectory(
        tasks[1],
        versions[tasks[1].task_id],
        trajectory_id="trajectory-invalid",
        valid=False,
        trace=replay_trace,
    )
    evidence = EvidenceStore(tmp_path / "evidence")
    # Deliberately reverse the append order; the mirror must follow selected.
    evidence.append_trajectory(invalid)
    evidence.append_trajectory(valid)
    path = tmp_path / "trajectories.jsonl"
    _MODULE._atomic_jsonl(path, [invalid])

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = evidence

        def __init__(self):
            self.collect_calls = []
            self.evaluator_calls = []

        async def collect(self, *args, **kwargs):
            self.collect_calls.append((args, kwargs))
            raise AssertionError("frozen rollout must not be recollected")

        async def evaluate_final_graph(
            self,
            task,
            final_answer,
            final_graph,
            *,
            rollout_index,
            environment_replay_trace,
        ):
            self.evaluator_calls.append(
                {
                    "task_id": task.task_id,
                    "final_answer": final_answer,
                    "final_graph": final_graph,
                    "rollout_index": rollout_index,
                    "trace": environment_replay_trace,
                }
            )
            return EvaluationOutcome(
                valid=True,
                reward=1.0,
                metrics={"exact_match": 1.0, "token_f1": 1.0},
                reason="evaluated",
                evaluator_version=versions[task.task_id].evaluator,
            )

    backend = Backend()
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {"concurrency": 2, "split": "validation"},
    }
    failures = []
    manifest = {}
    result = asyncio.run(
        _MODULE._collect_graph(
            backend,
            tasks,
            config,
            path,
            failures,
            manifest,
            tmp_path / "manifest.json",
        )
    )

    assert failures == []
    assert backend.collect_calls == []
    assert len(backend.evaluator_calls) == 1
    assert backend.evaluator_calls[0]["task_id"] == tasks[1].task_id
    assert backend.evaluator_calls[0]["trace"] == replay_trace
    assert set(result) == {task.task_id for task in tasks}
    assert manifest["agentgraph_progress"]["completed"] == 2
    assert manifest["agentgraph_progress"]["pending_evaluator_retries"] == 0
    retry = result[tasks[1].task_id]
    assert retry["trajectory_id"] != invalid["trajectory_id"]
    assert retry["turns"] == invalid["turns"]
    assert retry["evaluation_retry_receipt"]["source_trajectory_id"] == (
        invalid["trajectory_id"]
    )
    assert retry["evaluation_retry_receipt"]["attempt"] == 1
    assert retry["evaluation_retry_receipt"]["environment_replay_steps"] == 2
    persisted = _MODULE._read_jsonl(path)
    assert [item["task"]["task_id"] for item in persisted] == [
        task.task_id for task in tasks
    ]
    assert len({item["task"]["task_id"] for item in persisted}) == len(tasks)
    assert len(tuple(evidence.trajectories.payloads())) == 3

    # Once admitted, the same fixed batch is a pure resume: no Director,
    # Agent, terminal evaluator, or append-only event is repeated.
    resumed = asyncio.run(
        _MODULE._collect_graph(
            backend,
            tasks,
            config,
            path,
            failures,
            {},
            tmp_path / "manifest-resume.json",
        )
    )
    assert set(resumed) == {task.task_id for task in tasks}
    assert backend.collect_calls == []
    assert len(backend.evaluator_calls) == 1
    assert len(tuple(evidence.trajectories.payloads())) == 3


def test_invalid_evaluator_retry_stays_pending_and_never_recollects(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:retry-fails",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-invalid",
        valid=False,
    )
    evidence = EvidenceStore(tmp_path / "evidence")
    evidence.append_trajectory(invalid)

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = evidence

        def __init__(self):
            self.collect_calls = 0
            self.evaluator_calls = 0

        async def collect(self, *args, **kwargs):
            self.collect_calls += 1
            raise AssertionError("invalid evaluator attempt reserves the rollout")

        async def evaluate_final_graph(self, task, *args, **kwargs):
            self.evaluator_calls += 1
            return EvaluationOutcome(
                valid=False,
                reward=None,
                reason="environment_graph_callback_failed",
                evaluator_version=versions.evaluator,
            )

    backend = Backend()
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {"concurrency": 1, "split": "validation"},
    }
    path = tmp_path / "trajectories.jsonl"
    manifest_path = tmp_path / "manifest.json"
    failures = []
    for _ in range(2):
        result = asyncio.run(
            _MODULE._collect_graph(
                backend,
                (task,),
                config,
                path,
                failures,
                {},
                manifest_path,
            )
        )
        assert result == {}

    assert backend.collect_calls == 0
    assert backend.evaluator_calls == 2
    assert path.read_text(encoding="utf-8") == ""
    payloads = tuple(evidence.trajectories.payloads())
    assert len(payloads) == 3
    assert [
        item["evaluation_retry_receipt"]["attempt"] for item in payloads[1:]
    ] == [1, 2]
    assert payloads[-1]["evaluation"]["valid"] is False
    assert len(failures) == 2


def test_malformed_persisted_trace_fails_closed_without_evaluator_call(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:malformed-trace",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-invalid",
        valid=False,
    )
    invalid["evaluation"]["details"]["trace"] = [{"step": 1}]
    evidence = EvidenceStore(tmp_path / "evidence")
    evidence.append_trajectory(invalid)

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = evidence

        def __init__(self):
            self.collect_calls = 0
            self.evaluator_calls = 0

        async def collect(self, *args, **kwargs):
            self.collect_calls += 1
            raise AssertionError("frozen rollout must not be recollected")

        async def evaluate_final_graph(self, *args, **kwargs):
            self.evaluator_calls += 1
            raise AssertionError("malformed trace must not restart from step zero")

    backend = Backend()
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {"concurrency": 1, "split": "validation"},
    }
    failures = []
    path = tmp_path / "trajectories.jsonl"
    for run_index in range(2):
        result = asyncio.run(
            _MODULE._collect_graph(
                backend,
                (task,),
                config,
                path,
                failures,
                {},
                tmp_path / f"manifest-{run_index}.json",
            )
        )
        assert result == {}

    assert backend.collect_calls == 0
    assert backend.evaluator_calls == 0
    payloads = tuple(evidence.trajectories.payloads())
    assert len(payloads) == 3
    assert all(
        item["evaluation"]["reason"] == "environment_replay_trace_invalid"
        for item in payloads[1:]
    )


def test_interactive_retry_without_persisted_trace_fails_closed(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="webshop:missing-trace",
        question="buy an item",
        ground_truth="environment_success",
        split="validation",
        metadata={"dataset_key": "webshop"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-invalid",
        valid=False,
    )
    del invalid["evaluation"]["details"]["trace"]
    evidence = EvidenceStore(tmp_path / "evidence")
    evidence.append_trajectory(invalid)

    class Backend:
        evidence_store = evidence

        def __init__(self):
            self.evaluator_calls = 0

        async def evaluate_final_graph(self, *args, **kwargs):
            self.evaluator_calls += 1
            raise AssertionError("missing interactive trace must fail closed")

    backend = Backend()
    retry = asyncio.run(
        _MODULE._retry_terminal_evaluator(
            backend,
            task,
            invalid,
            versions=versions.to_dict(),
            attempt=1,
        )
    )

    assert backend.evaluator_calls == 0
    assert retry["evaluation"]["valid"] is False
    assert retry["evaluation"]["reason"] == "environment_replay_trace_unavailable"
    assert retry["evaluation_retry_receipt"]["environment_replay_steps"] is None


def test_swebench_retry_reuses_only_persisted_authoritative_patch():
    task = _MODULE.TaskRecord(
        task_id="swe_bench:retry",
        question="repository issue",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "swe_bench"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-swe-invalid",
        valid=False,
    )
    patch = "diff --git a/bug.py b/bug.py\n-old\n+new\n"
    invalid["evaluation"]["details"]["terminal_artifact"] = {
        "kind": "repository_patch",
        "source": "CodingExecutionAdapter.materialize_workspace_diff",
        "repository_patch": patch,
        "non_empty": True,
    }

    class Backend:
        def __init__(self):
            self.patch = None
            self.evidence_store = type(
                "Evidence",
                (),
                {
                    "append_trajectory": staticmethod(lambda payload: None),
                },
            )()

        async def evaluate_final_graph(self, task, *args, **kwargs):
            self.patch = kwargs.get("repository_patch")
            return EvaluationOutcome(
                valid=True,
                reward=0.0,
                metrics={"resolved": 0.0},
                reason="evaluated",
                evaluator_version=versions.evaluator,
            )

    backend = Backend()
    retry = asyncio.run(
        _MODULE._retry_terminal_evaluator(
            backend,
            task,
            invalid,
            versions=versions.to_dict(),
            attempt=1,
        )
    )

    assert backend.patch == patch
    assert retry["evaluation"]["valid"] is True
    assert retry["evaluation_retry_receipt"]["repository_patch_reused"] is True


def test_swebench_retry_never_uses_output_prose_as_patch():
    task = _MODULE.TaskRecord(
        task_id="swe_bench:missing-patch",
        question="repository issue",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "swe_bench"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-swe-missing",
        valid=False,
    )
    invalid["final_answer"] = "Output Agent prose that is not a patch"

    class Backend:
        evidence_store = type(
            "Evidence",
            (),
            {"append_trajectory": staticmethod(lambda payload: None)},
        )()

        async def evaluate_final_graph(self, *args, **kwargs):
            raise AssertionError("missing patch must fail before evaluator")

    retry = asyncio.run(
        _MODULE._retry_terminal_evaluator(
            Backend(),
            task,
            invalid,
            versions=versions.to_dict(),
            attempt=1,
        )
    )

    assert retry["evaluation"]["valid"] is False
    assert retry["evaluation"]["reason"] == (
        "terminal_repository_patch_unavailable"
    )
    assert retry["evaluation_retry_receipt"]["repository_patch_reused"] is False


def test_swebench_retry_submits_persisted_empty_patch_to_official_evaluator():
    task = _MODULE.TaskRecord(
        task_id="swe_bench:empty-patch",
        question="repository issue",
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "swe_bench"},
    )
    versions = _graph_versions(_MODULE, task)
    invalid = _graph_trajectory(
        task,
        versions,
        trajectory_id="trajectory-swe-empty-patch",
        valid=False,
    )
    invalid["evaluation"]["details"]["terminal_artifact"] = {
        "kind": "repository_patch",
        "source": "CodingExecutionAdapter.materialize_workspace_diff",
        "repository_patch": "",
        "non_empty": False,
    }

    class Backend:
        def __init__(self):
            self.patch = None
            self.evidence_store = type(
                "Evidence",
                (),
                {"append_trajectory": staticmethod(lambda payload: None)},
            )()

        async def evaluate_final_graph(self, task, *args, **kwargs):
            self.patch = kwargs.get("repository_patch")
            return EvaluationOutcome(
                valid=True,
                reward=0.0,
                metrics={"resolved": 0.0},
                reason="evaluated",
                evaluator_version=versions.evaluator,
                details={"harness_details": "empty_patch"},
            )

    backend = Backend()
    retry = asyncio.run(
        _MODULE._retry_terminal_evaluator(
            backend,
            task,
            invalid,
            versions=versions.to_dict(),
            attempt=1,
        )
    )

    assert backend.patch == ""
    assert retry["evaluation"]["valid"] is True
    assert retry["evaluation"]["details"]["harness_details"] == "empty_patch"
    assert retry["evaluation_retry_receipt"]["repository_patch_reused"] is True

def test_strict_aggregate_keeps_failed_task_in_denominator():
    rows = [
        {
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 0.8,
            }
        },
        {
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "token_f1": 0.0,
            }
        },
    ]

    result = _MODULE._aggregate(rows, "agentgraph")

    assert result["denominator"] == 2
    assert result["completed"] == 1
    assert result["evaluator_valid"] == 1
    assert result["strict_exact_match"] == 0.5
    assert result["strict_token_f1"] == 0.4
    assert result["completed_only_exact_match"] == 1.0


def test_correct_terminal_answer_is_not_relabelled_by_recovered_execution_error():
    trajectory = {
        "explicit_finish": True,
        "turns": [
            {"canvas_feedback": "execution_error=temporary provider timeout"},
            {"canvas_feedback": "accepted set_output"},
        ],
    }

    assert (
        _MODULE._failure_type(
            {"available": True},
            trajectory,
            direct_em=1.0,
            graph_em=1.0,
            graph_f1=1.0,
        )
        == "correct"
    )


def test_paired_agentgraph_gain_is_not_collapsed_into_correct():
    trajectory = {
        "explicit_finish": True,
        "turns": [{"canvas_feedback": "workflow finished"}],
    }

    assert (
        _MODULE._failure_type(
            {"available": True},
            trajectory,
            direct_em=0.0,
            graph_em=1.0,
            graph_f1=1.0,
        )
        == "architecture_gain"
    )


def test_report_counts_terminal_failure_without_dropping_evaluator_result():
    rows = [
        {
            "task_id": "hotpotqa:one",
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "explicit_finish": False,
                "termination_reason": "max_rounds",
            },
            "failure_type": "architecture_regression_candidate",
        }
    ]
    config = {
        "experiment": {"name": "terminal-failure"},
        "director": {
            "behavior_policy_version": "policy",
            "behavior_adapter_name": "adapter",
        },
        "agent_graph": {"model_catalog_path": "catalog.yaml"},
    }

    report = _MODULE._report(rows, config)

    assert report["terminal_failure_count"] == 1
    assert report["explicit_finished_count"] == 0
    assert report["operational_failure_count"] == 0
    assert report["agentgraph"]["completed"] == 1
    assert report["agentgraph"]["evaluator_valid"] == 1
    assert report["agentgraph"]["strict_exact_match"] == 0.0
