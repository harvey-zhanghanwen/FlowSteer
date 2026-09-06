"""Synthetic local handoff tests: no model, Tool, grader, or live process launch."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("healthbench_handoff", ROOT / "scripts/monitor_healthbench_evaluation_handoff.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def trajectory(index, *, condition="old", valid=True, raw=0.3, adjusted=0.2, terminal=False):
    return {
        "trajectory_id": f"t-{index}", "condition_id": condition,
        "task": {"task_id": f"healthbench-professional:{index}", "question": "Synthetic conversation",
                 "ground_truth": None, "split": "test", "metadata": {"dataset_key": "healthbench_professional"}},
        "versions": {"evaluator": "test-evaluator"}, "turns": [],
        "explicit_finish": not terminal, "final_answer": "Synthetic response" if not terminal else None,
        "termination_reason": "max_rounds" if terminal else "finish",
        "evaluation": {"valid": valid and not terminal, "evaluator_version": "test-evaluator", "reward": None,
                       "reason": "not_evaluated_without_explicit_finish" if terminal else "ok",
                       "metrics": {"overall_score": raw, "overall_score_length_adjusted": adjusted},
                       "details": {"formal_evaluator_called": not terminal}},
    }


def case(tmp_path, *, complete=True):
    ids = [f"healthbench-professional:{i}" for i in range(525)]
    specs = {}
    for name in ("old", "new"):
        cwd = tmp_path / name
        cwd.mkdir()
        config = {
            "experiment": {"condition_id": name, "training_enabled": False},
            "healthbench_professional_evaluation": {"sample_count": 525, "split": "test"},
            "storage": {"root": "artifacts/evaluation/evidence", "manifest_path": "artifacts/evaluation/run_manifest.json",
                        "selected_tasks_path": "artifacts/evaluation/selected_tasks.jsonl",
                        "trajectories_path": "artifacts/evaluation/agentgraph_trajectories.jsonl",
                        "direct_predictions_path": "artifacts/evaluation/direct_predictions.jsonl",
                        "failures_path": "artifacts/evaluation/collection_failures.jsonl"},
        }
        config_path = cwd / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        specs[name] = M.run_spec({"cwd": str(cwd), "config": "config.yaml", "lock": "artifacts/evaluation/evaluation.lock"})
        spec = specs[name]
        jsonl(spec["selected"], [{"task_id": task_id, "question": "Synthetic conversation"} for task_id in ids])
        spec["lock"].touch()
        manifest = {"status": "completed_with_operational_failures" if name == "old" else "prepared",
                    "completed_at": "now" if name == "old" else None,
                    "sample_count": 525, "selected_task_ids": ids,
                    "agentgraph_progress": {"completed": 525 if complete else 524, "pending_evaluator_retries": 0 if complete else 1},
                    "direct_progress": {"completed": 459, "strict_zero_terminal_failures": 66, "pending_evaluator_retries": 0}}
        M.write_json(spec["manifest"], manifest)
        if name == "old":
            values = [trajectory(i) for i in range(525 if complete else 524)]
            jsonl(spec["trajectories"], values)
            jsonl(spec["evidence"], [{"record_kind": "trajectory", "payload": v} for v in values])
            jsonl(spec["direct"], [{"task_id": task_id, "evaluation": {"valid": True}} for task_id in ids[:459]])
            jsonl(spec["failures"], [{"task_id": task_id, "condition": "direct_local_qwen35_9b",
                                     "stage": "generation_or_evaluator", "error": "AgentRuntimeError: react agent 'direct_react_agent' exhausted 6 turns without a valid completion"}
                                    for task_id in ids[459:]])
    ready_path = tmp_path / "ready.json"
    M.write_json(ready_path, {"ready": True, "tests_passed": True, "prepare_only_passed": True, "backup_pushed": True,
                              "config_path": str(specs["new"]["config_path"]), "condition_id": "new", "selected_task_ids": ids,
                              "source_commit": "version-label-only"})
    config = {"old": {"cwd": str(specs["old"]["cwd"]), "config": "config.yaml", "lock": str(specs["old"]["lock"])},
              "new": {"cwd": str(specs["new"]["cwd"]), "config": "config.yaml", "lock": str(specs["new"]["lock"])},
              "env_file": str(tmp_path / "not-read.env"), "python": "/fake/python",
              "state_dir": str(specs["new"]["cwd"] / "artifacts/handoff"), "readiness_path": str(ready_path),
              "poll_seconds": 60, "max_old_resumes": 2}
    return specs["old"], specs["new"], ids, config


def test_admission_reuses_runner_and_never_admits_invalid_evaluator():
    spec = {"condition_id": "old"}
    assert M.admission(trajectory(1), spec) == "valid"
    assert M.admission(trajectory(1, valid=False), spec) == "pending"
    assert M.admission(trajectory(1, terminal=True), spec) == "terminal_zero"
    assert M.admission(trajectory(1, condition="other"), spec) == "foreign"
    value = trajectory(1)
    value["explicit_finish"] = False
    assert M.admission(value, spec) == "pending"
    value = trajectory(1, raw=float("nan"))
    assert M.admission(value, spec) == "pending"


def test_cursor_waits_for_complete_line_and_summary_contains_no_private_text(tmp_path):
    path = tmp_path / "evidence.jsonl"
    values = [trajectory(0, raw=-1, adjusted=-1), trajectory(1, raw=1, adjusted=1)]
    values[0]["turns"] = [{"round_index": 0, "action": {"action": "ADD"}, "canvas_feedback": "ok", "policy_response": "SECRET_THINKING"}]
    first = json.dumps({"record_kind": "trajectory", "payload": values[0]}) + "\n"
    second = json.dumps({"record_kind": "trajectory", "payload": values[1]})
    path.write_text(first + second[:100])
    state = {"selected_task_ids": [f"healthbench-professional:{i}" for i in range(2)]}
    output = tmp_path / "low.jsonl"
    spec = {"condition_id": "old", "evidence": path}
    M.ingest(spec, state, output)
    assert state["evidence_cursor"]["offset"] == len(first.encode())
    assert state["valid_n"] == 1
    with path.open("a") as handle:
        handle.write(second[100:] + "\n")
    M.ingest(spec, state, output)
    assert state["valid_n"] == 2
    assert state["raw_mean"] == 0  # signed mean, not per-task clipping to 0.5
    assert "SECRET_THINKING" not in json.dumps(state) + output.read_text()
    assert "Synthetic response" not in json.dumps(state) + output.read_text()
    prior = output.read_text()
    M.ingest(spec, state, output)
    assert output.read_text() == prior


def test_pending_retry_is_replaced_once_and_valid_never_downgraded(tmp_path):
    path = tmp_path / "evidence.jsonl"
    pending = trajectory(0, valid=False)
    pending["evaluation"]["details"]["provider_errors"] = [{"status_code": 403}]
    jsonl(path, [{"record_kind": "trajectory", "payload": pending}])
    state = {"selected_task_ids": ["healthbench-professional:0"]}
    spec = {"condition_id": "old", "evidence": path}
    M.ingest(spec, state, tmp_path / "low.jsonl")
    assert state["task_summaries"]["healthbench-professional:0"]["permanent_provider_failure"]
    with path.open("a") as handle:
        for value in [trajectory(0), pending]:
            handle.write(json.dumps({"record_kind": "trajectory", "payload": value}) + "\n")
    M.ingest(spec, state, tmp_path / "low.jsonl")
    assert state["valid_n"] == 1 and state["pending_n"] == 0
    assert not state["task_summaries"]["healthbench-professional:0"]["permanent_provider_failure"]


def test_complete_requires_full_admitted_population_not_success_exit_status(tmp_path):
    old, new, ids, config = case(tmp_path, complete=False)
    outcome = M.completion(old, ids)
    assert outcome["status"] == "completed_with_operational_failures"
    assert not outcome["complete"] and outcome["admitted_n"] == 524
    jsonl(old["trajectories"], [trajectory(i) for i in range(524)] + [trajectory(524, terminal=True)])
    manifest = M.read_json(old["manifest"])
    manifest["agentgraph_progress"] = {"completed": 525, "pending_evaluator_retries": 0}
    M.write_json(old["manifest"], manifest)
    assert M.completion(old, ids)["complete"]  # frozen 66 Direct zeros are valid denominator


@pytest.mark.parametrize("corruption", ["duplicate", "invalid", "pending", "missing_finish", "direct_overlap", "wrong_ids"])
def test_denominator_gate_rejects_false_completion(tmp_path, corruption):
    old, new, ids, config = case(tmp_path)
    manifest = M.read_json(old["manifest"])
    if corruption == "duplicate":
        jsonl(old["trajectories"], [trajectory(0)] * 525)
        with pytest.raises(M.HandoffError):
            M.completion(old, ids)
        return
    if corruption in {"invalid", "missing_finish"}:
        values = [trajectory(i) for i in range(525)]
        if corruption == "invalid":
            values[-1]["evaluation"]["valid"] = False
        else:
            values[-1]["explicit_finish"] = False
        jsonl(old["trajectories"], values)
    elif corruption == "pending":
        manifest["agentgraph_progress"]["pending_evaluator_retries"] = 1
        M.write_json(old["manifest"], manifest)
    elif corruption == "direct_overlap":
        direct = list(M.rows(old["direct"]))
        direct[-1]["task_id"] = ids[-1]
        jsonl(old["direct"], direct)
    else:
        manifest["selected_task_ids"] = list(reversed(ids))
        M.write_json(old["manifest"], manifest)
        with pytest.raises(M.HandoffError):
            M.completion(old, ids)
        return
    assert not M.completion(old, ids)["complete"]


def test_readiness_requires_test_prepare_backup_and_unchanged_population(tmp_path):
    old, new, ids, config = case(tmp_path)
    ready = Path(config["readiness_path"])
    assert M.validate_readiness(ready, old, new, ids)["ready"]
    v = M.read_json(ready)
    v["backup_pushed"] = False
    M.write_json(ready, v)
    with pytest.raises(M.HandoffError):
        M.validate_readiness(ready, old, new, ids)
    v["backup_pushed"] = True
    M.write_json(ready, v)
    samples = list(M.rows(new["selected"]))
    samples[0]["question"] = "Changed sample content"
    jsonl(new["selected"], samples)
    with pytest.raises(M.HandoffError, match="samples differ"):
        M.validate_readiness(ready, old, new, ids)


@pytest.mark.parametrize("status", ["agentgraph", "new_running", "completed_with_operational_failures"])
def test_new_existing_run_is_never_relaunched(tmp_path, status):
    old, new, ids, config = case(tmp_path)
    manifest = M.read_json(new["manifest"])
    manifest["status"] = status
    M.write_json(new["manifest"], manifest)
    with pytest.raises(M.HandoffError, match="duplicate launch"):
        M.validate_readiness(Path(config["readiness_path"]), old, new, ids)


def test_os_lease_is_exclusive_and_released(tmp_path):
    path = tmp_path / "test.lock"
    with M.lease(path, create=True) as owner:
        assert owner is not None
        with M.lease(path) as other:
            assert other is None
    with M.lease(path) as next_owner:
        assert next_owner is not None


def test_launch_only_constructs_existing_cli_without_reading_environment_file(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path)
    captured = {}
    def fake_popen(args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(M.subprocess, "Popen", fake_popen)
    with M.lease(new["lock"]) as lock:
        M.launch(new, config, lock, tmp_path / "new.log")
        assert captured["kwargs"]["pass_fds"] == (lock.fileno(),)
    assert "scripts/evaluate_completion_benchmark_round.py --config" in captured["args"][2]
    assert "FLOWSTEER_ROLLOUT_GPU=5 FLOWSTEER_SUPERVISOR_PORT=8025" in captured["args"][2]
    assert str(new["config_path"]) in captured["args"]
    assert not Path(config["env_file"]).exists()


def test_permanent_quota_detection_is_receipt_only():
    assert M.permanent_provider_failure({"provider_errors": [{"status_code": 403}]})
    assert M.permanent_provider_failure({"error": {"code": "local:insufficient_quota"}})
    assert not M.permanent_provider_failure({"provider_errors": [{"status_code": 429}]})
    assert not M.permanent_provider_failure({"question": "403 things", "final_answer": "insufficient_quota"})


def test_no_progress_resume_blocks_and_does_not_launch_new(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path, complete=False)
    launches = []
    monkeypatch.setattr(M, "matching_processes", lambda spec: [])
    def fake_launch(spec, configuration, lock, path):
        launches.append(spec["condition_id"])
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(M, "launch", fake_launch)
    assert M.run(config) == 1
    state = M.read_json(Path(config["state_dir"]) / "state.json")
    assert state["phase"] == "blocked"
    assert "no admitted/evaluator progress" in state["reason"]
    assert launches == ["old"]


def test_old_permanent_error_blocks_before_any_resume(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path, complete=False)
    pending = trajectory(524, valid=False)
    pending["evaluation"]["details"]["provider_errors"] = [{"status_code": 403}]
    with old["evidence"].open("a") as handle:
        handle.write(json.dumps({"record_kind": "trajectory", "payload": pending}) + "\n")
    monkeypatch.setattr(M, "matching_processes", lambda spec: [])
    monkeypatch.setattr(M, "launch", lambda *a: pytest.fail("must not call a runner"))
    assert M.run(config) == 1
    state = M.read_json(Path(config["state_dir"]) / "state.json")
    assert "403" in state["reason"] and state["old_resume_attempts"] == 0


def test_complete_old_launches_new_once_and_reports_incomplete_exit(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path)
    launches = []
    monkeypatch.setattr(M, "matching_processes", lambda spec: [])
    def fake_launch(spec, configuration, lock, path):
        launches.append(spec["condition_id"])
        return SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)
    monkeypatch.setattr(M, "launch", fake_launch)
    assert M.run(config) == 1
    state = M.read_json(Path(config["state_dir"]) / "state.json")
    assert state["phase"] == "new_finished" and not state["new_completion"]["complete"]
    assert launches == ["new"]
    with pytest.raises(M.HandoffError, match="already launched"):
        M.run(config)


def test_old_process_alive_or_lease_held_prevents_handoff(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path)
    monkeypatch.setattr(M, "matching_processes", lambda spec: [123] if spec["condition_id"] == "old" else [])
    monkeypatch.setattr(M, "launch", lambda *a: pytest.fail("must not launch while old alive"))
    monkeypatch.setattr(M.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        M.run(config)


def test_bounds_reject_unbounded_resume_or_busy_poll(tmp_path):
    old, new, ids, config = case(tmp_path)
    config["max_old_resumes"] = 3
    with pytest.raises(M.HandoffError):
        M.run(config)
    config["max_old_resumes"] = 2
    config["poll_seconds"] = 1
    with pytest.raises(M.HandoffError):
        M.run(config)


def test_inherited_flock_survives_parent_close_for_a_harmless_child(tmp_path):
    # This starts only a local shell which waits on stdin, never the runner.
    # It confirms the same pass_fds/flock lifetime used by launch().
    path = tmp_path / "inherited.lock"
    with M.lease(path, create=True) as lock:
        child = M.subprocess.Popen(
            ["bash", "-c", 'flock -n "$1"; echo ready; read -r unused', "fixture", str(lock.fileno())],
            pass_fds=(lock.fileno(),), stdin=M.subprocess.PIPE,
            stdout=M.subprocess.PIPE, text=True,
        )
        assert child.stdout.readline().strip() == "ready"
    try:
        with M.lease(path) as contender:
            assert contender is None
    finally:
        child.communicate("done\n", timeout=5)
    with M.lease(path) as released:
        assert released is not None


def test_two_progressing_old_resumes_are_the_hard_limit(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path, complete=False)
    jsonl(old["trajectories"], [trajectory(i) for i in range(521)])
    manifest = M.read_json(old["manifest"])
    manifest["agentgraph_progress"] = {"completed": 521, "pending_evaluator_retries": 4}
    M.write_json(old["manifest"], manifest)
    launches = []
    monkeypatch.setattr(M, "matching_processes", lambda spec: [])
    def fake_launch(spec, configuration, lock, path):
        launches.append(spec["condition_id"])
        count = 521 + len(launches)
        jsonl(old["trajectories"], [trajectory(i) for i in range(count)])
        manifest["agentgraph_progress"] = {"completed": count, "pending_evaluator_retries": 525 - count}
        M.write_json(old["manifest"], manifest)
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(M, "launch", fake_launch)
    assert M.run(config) == 1
    state = M.read_json(Path(config["state_dir"]) / "state.json")
    assert state["phase"] == "blocked" and "attempts exhausted" in state["reason"]
    assert launches == ["old", "old"] and state["old_resume_attempts"] == 2


def test_held_old_lease_alone_blocks_launch(tmp_path, monkeypatch):
    old, new, ids, config = case(tmp_path)
    monkeypatch.setattr(M, "matching_processes", lambda spec: [])
    monkeypatch.setattr(M, "launch", lambda *a: pytest.fail("old lease still held"))
    monkeypatch.setattr(M.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    with M.lease(old["lock"]):
        with pytest.raises(KeyboardInterrupt):
            M.run(config)
