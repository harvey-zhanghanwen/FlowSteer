#!/usr/bin/env python3
"""Local-only monitoring and a bounded handoff between frozen HealthBench runs.

This coordinator does not implement evaluation, model calls, architecture edits,
or medical error analysis. It reuses the existing completion runner's admission,
public receipt diagnosis and exact-resume CLI. Only that unchanged CLI may call
models/evaluators, after the old lease is released and explicit readiness holds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
TERMINAL = {"completed", "completed_with_terminal_failures", "completed_with_operational_failures"}
class HandoffError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def runner():
    # Import only local source; this module's CLI is guarded by __main__.
    import evaluate_completion_benchmark_round
    return evaluate_completion_benchmark_round


def resolve(cwd: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


def run_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    cwd = Path(value["cwd"]).expanduser().resolve()
    config_path = resolve(cwd, value["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = config["healthbench_professional_evaluation"]
    if section["sample_count"] != 525 or section["split"] != "test":
        raise HandoffError("handoff requires the fixed 525 public test tasks")
    if config["experiment"].get("training_enabled") is not False:
        raise HandoffError("training is outside this handoff")
    for name in ("grpo", "policy_sync", "skills", "exploration"):
        if config.get(name, {}).get("enabled") is True:
            raise HandoffError("training/Skill/exploration is outside this handoff")
    storage = config["storage"]
    return {
        "cwd": cwd, "config_path": config_path, "config": config,
        "condition_id": config["experiment"]["condition_id"],
        "lock": resolve(cwd, value["lock"]),
        "manifest": resolve(cwd, storage["manifest_path"]),
        "selected": resolve(cwd, storage["selected_tasks_path"]),
        "trajectories": resolve(cwd, storage["trajectories_path"]),
        "direct": resolve(cwd, storage["direct_predictions_path"]),
        "failures": resolve(cwd, storage["failures_path"]),
        "evidence": resolve(cwd, storage["root"]) / "trajectories.jsonl",
    }


def selected_ids(spec: Mapping[str, Any]) -> list[str]:
    ids = [value["task_id"] for value in rows(spec["selected"])]
    if len(ids) != 525 or len(set(ids)) != 525:
        raise HandoffError("selected task list is not exactly 525 unique IDs")
    return ids


def admission(value: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """Reuse the runner's valid/official terminal-zero receipt predicates."""
    if value.get("condition_id") != spec["condition_id"]:
        return "foreign"
    r = runner()
    try:
        task = r.TaskRecord.from_dict(value["task"])
        versions = value["versions"]
        if r.hotpot_round._reportable_terminal_failure_matches(
            value, task=task, condition_id=spec["condition_id"], versions=versions
        ):
            return "terminal_zero"
        if (value.get("explicit_finish") is True and value.get("final_answer")
                and r._trajectory_resume_matches(
                    value, task=task, condition_id=spec["condition_id"], versions=versions)):
            metrics = value["evaluation"].get("metrics", {})
            if all(isinstance(metrics.get(k), (int, float)) and not isinstance(metrics.get(k), bool)
                   and math.isfinite(metrics[k])
                   for k in ("overall_score", "overall_score_length_adjusted")):
                return "valid"
    except (KeyError, TypeError, ValueError):
        pass
    return "pending"


def permanent_provider_failure(value: Any) -> bool:
    """Inspect error receipts, not conversation, response, or reasoning text."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"status_code", "http_status"} and item == 403:
                return True
            if key in {"code", "type"} and isinstance(item, str) and item.rsplit(":", 1)[-1] == "insufficient_quota":
                return True
            if key in {"error", "reason", "grader_error"} and isinstance(item, str):
                if "insufficient_quota" in item.casefold() or "403" in item:
                    return True
            if isinstance(item, (Mapping, list)) and permanent_provider_failure(item):
                return True
    elif isinstance(value, list):
        return any(permanent_provider_failure(item) for item in value)
    return False


def compact_diagnosis(value: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = runner()._healthbench_wrong_demo_diagnosis(value)
    # Never retain error text, response, rubric text, prompts, or hidden reasoning.
    allowed = ("failure_layer", "first_error_turn", "first_error_action",
               "first_error_agent_id", "rubric_receipt_summary", "terminal_result")
    return {key: diagnosis[key] for key in allowed if key in diagnosis}


def ingest(spec: Mapping[str, Any], state: dict[str, Any], summary_path: Path) -> None:
    """Consume complete append-only envelopes once; partial tail is retried."""
    path = spec["evidence"]
    if not path.exists():
        return
    cursor = state.setdefault("evidence_cursor", {"offset": 0})
    if path.stat().st_size < cursor["offset"]:
        raise HandoffError("old evidence stream shrank; do not reinterpret the run")
    seen = state.setdefault("task_summaries", {})
    allowed = set(state["selected_task_ids"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as handle, summary_path.open("a", encoding="utf-8") as output:
        handle.seek(cursor["offset"])
        while True:
            line = handle.readline()
            if not line or not line.endswith(b"\n"):
                break
            envelope = json.loads(line)
            cursor["offset"] = handle.tell()
            if envelope.get("record_kind") != "trajectory":
                continue
            value = envelope.get("payload", {})
            task_id = value.get("task", {}).get("task_id")
            if task_id not in allowed:
                continue
            kind = admission(value, spec)
            if kind == "foreign":
                continue
            # A later invalid grader attempt never removes an admitted result.
            if seen.get(task_id, {}).get("admission") in {"valid", "terminal_zero"}:
                continue
            metrics = value.get("evaluation", {}).get("metrics", {})
            item = {
                "task_id": task_id, "admission": kind,
                "raw_score": metrics.get("overall_score") if kind == "valid" else None,
                "adjusted_score": metrics.get("overall_score_length_adjusted") if kind == "valid" else None,
                "permanent_provider_failure": kind == "pending" and permanent_provider_failure(value.get("evaluation", {})),
            }
            if kind == "valid" and item["adjusted_score"] < 1.0:
                item["diagnosis"] = compact_diagnosis(value)
            changed = item != seen.get(task_id)
            seen[task_id] = item
            if changed and kind == "valid" and item["adjusted_score"] < 1.0:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
    valid = [item for item in seen.values() if item["admission"] == "valid"]
    state.update(
        valid_n=len(valid),
        admitted_n=sum(item["admission"] in {"valid", "terminal_zero"} for item in seen.values()),
        pending_n=sum(item["admission"] == "pending" for item in seen.values()),
        raw_mean=(min(1.0, max(0.0, sum(v["raw_score"] for v in valid) / len(valid))) if valid else None),
        adjusted_mean=(min(1.0, max(0.0, sum(v["adjusted_score"] for v in valid) / len(valid))) if valid else None),
        metric_scope="evaluator_valid_partial_subset_not_full525",
        low_score_queue_path=str(summary_path),
        failure_layer_counts=dict(Counter(v.get("diagnosis", {}).get("failure_layer")
                                          for v in valid if "diagnosis" in v)),
    )


def matching_processes(spec: Mapping[str, Any]) -> list[int]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [v.decode() for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
            if not any(v.endswith("evaluate_completion_benchmark_round.py") for v in args):
                continue
            if "--config" not in args or (proc / "cwd").resolve() != spec["cwd"]:
                continue
            config = args[args.index("--config") + 1]
            if resolve(spec["cwd"], config) == spec["config_path"]:
                matches.append(int(proc.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return matches


@contextmanager
def lease(path: Path, *, create: bool = False):
    if not path.exists() and not create:
        raise HandoffError("expected evaluation lease file is missing")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+" if create else "r+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield handle
    finally:
        handle.close()


def completion(spec: Mapping[str, Any], ids: list[str]) -> dict[str, Any]:
    """Inspect the final checkpoint once the old runner no longer owns its lease."""
    manifest = read_json(spec["manifest"], {})
    if manifest.get("selected_task_ids") != ids or manifest.get("sample_count") != 525:
        raise HandoffError("manifest denominator differs from frozen selected tasks")
    graph = {}
    for value in rows(spec["trajectories"]):
        task_id = value.get("task", {}).get("task_id")
        if task_id in graph or task_id not in ids:
            raise HandoffError("ordered Graph checkpoint has duplicate/foreign tasks")
        graph[task_id] = admission(value, spec)
    admitted = sum(kind in {"valid", "terminal_zero"} for kind in graph.values())
    direct = list(rows(spec["direct"]))
    direct_ids = [v.get("task_id", v.get("task", {}).get("task_id")) for v in direct]
    if len(set(direct_ids)) != len(direct_ids):
        raise HandoffError("Direct checkpoint has duplicate tasks")
    direct_failures = runner()._bounded_direct_react_exhaustion_task_ids(
        list(rows(spec["failures"])), task_ids=set(ids))
    progress = manifest.get("direct_progress", {})
    direct_complete = (
        len(direct_ids) == 459 and len(direct_failures) == 66
        and set(direct_ids).isdisjoint(direct_failures)
        and set(direct_ids) | set(direct_failures) == set(ids)
        and all(v.get("evaluation", {}).get("valid") is True for v in direct)
        and progress.get("completed") == 459
        and progress.get("strict_zero_terminal_failures") == 66
        and progress.get("pending_evaluator_retries") == 0
    )
    graph_progress = manifest.get("agentgraph_progress", {})
    pending = graph_progress.get("pending_evaluator_retries", 0)
    complete = (
        manifest.get("status") in TERMINAL and bool(manifest.get("completed_at"))
        and admitted == 525 and set(graph) == set(ids)
        and graph_progress.get("completed") == 525 and pending == 0 and direct_complete
    )
    return {"complete": complete, "admitted_n": admitted, "pending_n": pending,
            "direct_complete": direct_complete, "status": manifest.get("status")}


def validate_readiness(path: Path, old: Mapping[str, Any], new: Mapping[str, Any], ids: list[str]) -> Mapping[str, Any]:
    ready = read_json(path, {})
    expected = {"ready": True, "tests_passed": True, "prepare_only_passed": True,
                "backup_pushed": True, "config_path": str(new["config_path"]),
                "condition_id": new["condition_id"], "selected_task_ids": ids}
    if any(ready.get(key) != value for key, value in expected.items()):
        raise HandoffError("new architecture is not explicitly verified/backed-up/ready")
    if selected_ids(new) != ids or list(rows(new["selected"])) != list(rows(old["selected"])):
        raise HandoffError("new selected samples differ from the frozen old samples")
    new_manifest = read_json(new["manifest"], {})
    if new_manifest.get("status") != "prepared":
        raise HandoffError("new run is not prepare-only; refuse a duplicate launch")
    if new_manifest.get("selected_task_ids") != ids or new_manifest.get("sample_count") != 525:
        raise HandoffError("new prepare-only manifest population mismatch")
    if new["trajectories"].exists() and new["trajectories"].stat().st_size:
        raise HandoffError("new rollout checkpoint is already non-empty")
    if new["evidence"].exists() and new["evidence"].stat().st_size:
        raise HandoffError("new append-only rollout stream is already non-empty")
    return ready


def launch(spec: Mapping[str, Any], config: Mapping[str, Any], lock_handle: Any, log_path: Path):
    """Only execute the existing evaluation CLI with an inherited flock lease."""
    command = (
        'set -e; cd "$1"; set -a; source "$2"; set +a; '
        'export FLOWSTEER_ROLLOUT_GPU=5 FLOWSTEER_SUPERVISOR_PORT=8025 '
        'FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH=32768 FLOWSTEER_SUPERVISOR_MEM_FRACTION=0.82 PYTHONUNBUFFERED=1; '
        'flock -n "$5"; exec "$3" scripts/evaluate_completion_benchmark_round.py --config "$4"'
    )
    with log_path.open("ab") as log:
        return subprocess.Popen(
            ["bash", "-lc", command, "healthbench-handoff", str(spec["cwd"]),
             config["env_file"], config["python"], str(spec["config_path"]), str(lock_handle.fileno())],
            cwd=spec["cwd"], stdout=log, stderr=subprocess.STDOUT,
            pass_fds=(lock_handle.fileno(),), start_new_session=True,
        )


def run(config: Mapping[str, Any]) -> int:
    old, new = run_spec(config["old"]), run_spec(config["new"])
    if old["cwd"] == new["cwd"] or old["condition_id"] == new["condition_id"]:
        raise HandoffError("old and new runs must use independent code and conditions")
    state_dir = Path(config["state_dir"]).expanduser().resolve()
    if not state_dir.is_relative_to(new["cwd"] / "artifacts"):
        raise HandoffError("handoff state must remain in new-worktree artifacts")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    poll_seconds = float(config.get("poll_seconds", 60))
    max_resumes = int(config.get("max_old_resumes", 2))
    if poll_seconds < 60 or not 0 <= max_resumes <= 2:
        raise HandoffError("monitor interval must be >=60s and old resumes <=2")
    ids = selected_ids(old)
    state = read_json(state_path, {})
    if state.get("phase") in {"new_launching", "new_running", "new_finished", "blocked"}:
        raise HandoffError("handoff is already launched/finished/blocked; explicit review required")
    if state and state.get("selected_task_ids") != ids:
        raise HandoffError("persisted monitor state belongs to another population")
    state.update(selected_task_ids=ids, old_condition=old["condition_id"], new_condition=new["condition_id"])
    state.setdefault("old_resume_attempts", 0)

    def save(phase: str, **values: Any):
        state.update(phase=phase, updated_at=now(), **values)
        write_json(state_path, state)

    def block(reason: str):
        save("blocked", reason=reason)
        return 1

    with lease(state_dir / "monitor.lock", create=True) as monitor_lock:
        if monitor_lock is None:
            raise HandoffError("another handoff monitor already holds the lease")
        while True:
            try:
                ingest(old, state, state_dir / "low_score_queue.jsonl")
                old_manifest = read_json(old["manifest"], {})
                save("waiting_old", old_status=old_manifest.get("status"),
                     old_manifest_progress=old_manifest.get("agentgraph_progress", {}))
                with lease(old["lock"]) as old_lock:
                    if old_lock is None or matching_processes(old):
                        time.sleep(poll_seconds)
                        continue
                    outcome = completion(old, ids)
                    save("old_checkpoint_checked", old_completion=outcome)
                    if not outcome["complete"]:
                        if not outcome["direct_complete"]:
                            return block("old Direct control is not the frozen 459+66; no recollection allowed")
                        summaries = state.get("task_summaries", {})
                        if any(v.get("permanent_provider_failure") for v in summaries.values()):
                            return block("unresolved grader reports permanent 403/insufficient_quota")
                        unresolved = set(ids) - {k for k, v in summaries.items() if v["admission"] in {"valid", "terminal_zero"}}
                        latest_failures = {v.get("task_id"): v for v in rows(old["failures"])
                                           if v.get("condition") == "agentgraph" and v.get("task_id") in unresolved}
                        if any(permanent_provider_failure(v) for v in latest_failures.values()):
                            return block("unresolved collection failure reports permanent 403/quota")
                        previous = state.get("previous_resume_start")
                        if previous and outcome["admitted_n"] <= previous["admitted_n"] and outcome["pending_n"] >= previous["pending_n"]:
                            return block("old exact resume made no admitted/evaluator progress")
                        if state["old_resume_attempts"] >= max_resumes:
                            return block("bounded old exact-resume attempts exhausted")
                        state["old_resume_attempts"] += 1
                        save("old_resuming", previous_resume_start=outcome)
                        child = launch(old, config, old_lock, state_dir / f"old_resume_{state['old_resume_attempts']}.log")
                        save("old_resuming", old_resume_pid=child.pid)
                        while child.poll() is None:
                            time.sleep(poll_seconds)
                            ingest(old, state, state_dir / "low_score_queue.jsonl")
                            save("old_resuming")
                        save("old_resume_exited", old_resume_exit_code=child.returncode)
                        continue
                    # Keep the old lease while checking readiness and starting the new
                    # CLI, preventing concurrent coordinator-owned old resumes.
                    ready = validate_readiness(Path(config["readiness_path"]), old, new, ids)
                    if matching_processes(new):
                        return block("a matching new evaluation process already exists")
                    with lease(new["lock"], create=True) as new_lock:
                        if new_lock is None:
                            return block("new evaluation lease is already held")
                        save("new_launching", readiness=ready, old_complete=True)
                        child = launch(new, config, new_lock, state_dir / "new_evaluation.log")
                        save("new_running", new_pid=child.pid, new_started_at=now())
                        # No new-run retry loop. The existing runner alone owns its
                        # sampling and evaluator work; a failure is reported as-is.
                        while child.poll() is None:
                            time.sleep(poll_seconds)
                            manifest = read_json(new["manifest"], {})
                            save("new_running", new_status=manifest.get("status"),
                                 new_progress=manifest.get("agentgraph_progress", {}))
                        new_outcome = completion(new, ids)
                        save("new_finished", new_exit_code=child.returncode, new_completion=new_outcome,
                             reason=None if new_outcome["complete"] else "new runner exited with incomplete results; no automatic rerun")
                        return 0 if new_outcome["complete"] else 1
            except (HandoffError, OSError, ValueError, KeyError) as error:
                # Do not persist exception text, which may contain arbitrary data.
                return block(str(error) if isinstance(error, HandoffError) else type(error).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="explicit local handoff JSON, not an evaluator YAML")
    args = parser.parse_args()
    try:
        return run(read_json(args.config))
    except (HandoffError, OSError, ValueError, KeyError) as error:
        print(str(error) if isinstance(error, HandoffError) else type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
