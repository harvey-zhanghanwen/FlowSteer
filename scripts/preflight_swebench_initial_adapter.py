#!/usr/bin/env python3
"""Run a no-model SWE-bench adapter preflight and persist its receipt.

The preflight uses the fixed aligned task population, SkillFlow's detached
worktree lifecycle, a read-only repository smoke check, and the official
evaluator's own fail-closed preflight.  It does not generate a patch, call a
model, run an issue test, or assign a Resolved/Failed label.  The strict
SkillFlow Tool registration is deliberately unavailable until its task-specific
command environment has passed preflight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.coding_tools import (  # noqa: E402
    RepositoryToolBackend,
)
from src.interactive.config_loader import load_yaml  # noqa: E402
from src.interactive.swe_worktree import (  # noqa: E402
    SWEbenchRepositoryIdentity,
    prepare_swebench_worktree,
    preflight_swebench_worktree_population,
)
from src.interactive.swebench_adapter import (  # noqa: E402
    OfficialSWEbenchHarness,
    SWEbenchHarnessUnavailable,
)
from src.interactive.task_dataset import load_task_records  # noqa: E402
from src.interactive.tool_runtime import ToolRequest  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured path must be non-empty text")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _public_tool_result(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {"ok": False, "error": "non_mapping_tool_result"}
    return {
        key: value.get(key)
        for key in (
            "action",
            "ok",
            "count",
            "match_count",
            "path",
            "kind",
            "total_lines",
            "error",
        )
        if key in value
    }


def run(config_path: Path, output_path: Path | None = None) -> Mapping[str, Any]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    data = _mapping(config.get("data"), "data")
    runtime = _mapping(config.get("swe_coding_runtime"), "swe_coding_runtime")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    bounded = _mapping(config.get("swebench_evaluation"), "swebench_evaluation")
    tasks = load_task_records(
        _resolve(root, data.get("test_path")),
        expected_split="test",
        limit=int(bounded["sample_count"]),
    )
    identities = [SWEbenchRepositoryIdentity.from_task_record(task) for task in tasks]
    repository_store = _resolve(root, runtime.get("repository_store"))
    worktree_root = _resolve(root, runtime.get("worktree_root"))
    worktree_root.mkdir(parents=True, exist_ok=True)

    repository_population = preflight_swebench_worktree_population(
        tasks,
        repository_store=repository_store,
        worktree_root=worktree_root,
        setup_timeout_seconds=float(runtime.get("setup_timeout_seconds", 30.0)),
        cleanup_timeout_seconds=float(runtime.get("cleanup_timeout_seconds", 10.0)),
    )
    population_rows = repository_population["rows"]
    available_indices = [
        int(row["index"])
        for row in population_rows
        if isinstance(row, Mapping) and row.get("ready") is True
    ]
    missing_indices = [
        int(row["index"])
        for row in population_rows
        if isinstance(row, Mapping) and row.get("ready") is not True
    ]
    repository_smoke: dict[str, Any] = {
        "status": "blocked_no_local_repository",
        "task_id": None,
        "instance_id": None,
        "base_state_prepared": False,
        "read_only_action_names": [],
        "strict_registration_ready": False,
        "strict_registration_block_reason": "task_environment_unavailable",
        "tool_calls": [],
        "initial_workspace_diff_empty": None,
        "cleanup_completed": None,
    }
    if available_indices:
        smoke_index = available_indices[0]
        task = tasks[smoke_index]
        identity = identities[smoke_index]
        prepared = prepare_swebench_worktree(
            identity,
            repository_store=repository_store,
            worktree_root=worktree_root,
            setup_timeout_seconds=float(runtime.get("setup_timeout_seconds", 30.0)),
            cleanup_timeout_seconds=float(
                runtime.get("cleanup_timeout_seconds", 10.0)
            ),
        )
        try:
            backend = RepositoryToolBackend(
                prepared.repo_root,
                max_test_timeout_seconds=float(
                    runtime.get("max_test_timeout_seconds", 60.0)
                ),
            )
            listed_value = backend.invoke(ToolRequest("list_files", {})).value
            files = (
                listed_value.get("files", [])
                if isinstance(listed_value, Mapping)
                else []
            )
            tool_calls = [_public_tool_result(listed_value)]
            if files:
                viewed = backend.invoke(
                    ToolRequest(
                        "view_file",
                        {"path": str(files[0]), "start_line": 1, "end_line": 20},
                    ),
                )
                tool_calls.append(_public_tool_result(viewed.value))
            patch = backend.materialize_workspace_diff()
            repository_smoke.update(
                {
                    "status": "passed_read_only",
                    "task_id": task.task_id,
                    "instance_id": identity.instance_id,
                    "repo": identity.repo,
                    "base_state_prepared": True,
                    "read_only_action_names": ["list_files", "view_file"],
                    "tool_calls": tool_calls,
                    "initial_workspace_diff_empty": not bool(patch.strip()),
                }
            )
        finally:
            prepared.cleanup()
            repository_smoke["cleanup_completed"] = prepared.closed

    harness = OfficialSWEbenchHarness(
        evaluator_path=_resolve(root, evaluation.get("swebench_evaluator_path")),
        harness_path=_resolve(root, evaluation.get("swebench_harness_path")),
        dataset_source=str(evaluation.get("swebench_dataset_source")),
        dataset_path=_resolve(root, evaluation.get("swebench_dataset_path")),
        evaluation_root=_resolve(root, evaluation.get("swebench_evaluation_root")),
        docker_namespace=str(evaluation.get("swebench_docker_namespace")),
        timeout_seconds=int(evaluation.get("swebench_timeout_seconds")),
        conda_executable=_resolve(root, runtime.get("conda_executable")),
        conda_envs_dir=_resolve(root, runtime.get("conda_envs_dir")),
        environment_repository_root=_resolve(
            root, runtime.get("environment_repository_root")
        ),
    )
    try:
        task_environment_population = dict(
            harness.preflight_task_environments(tasks)
        )
    except Exception as exc:
        task_environment_population = {
            "all_ready": False,
            "total": len(tasks),
            "ready": 0,
            "unavailable": len(tasks),
            "rows": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        harness_receipt = dict(harness.preflight(tasks))
        harness_receipt["status"] = "passed"
    except SWEbenchHarnessUnavailable as exc:
        harness_receipt = {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "formal_evaluator_called": False,
            "resolved_labels_assigned": 0,
            "proxy_metric_used": False,
        }

    receipt = {
        "schema_version": "flowsteer.swebench.initial-adapter-preflight.v1",
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "training_started": False,
        "model_calls": 0,
        "selected_population": {
            "split": "test",
            "tasks": len(tasks),
            "unique_instance_ids": len({identity.instance_id for identity in identities}),
        },
        "repository_coverage": {
            "task_repository_base_state_ready": len(available_indices),
            "task_repository_base_state_unavailable": len(missing_indices),
            "available_unique_repositories": len(
                {identities[index].repo for index in available_indices}
            ),
            "missing_unique_repositories": len(
                {identities[index].repo for index in missing_indices}
            ),
            "missing_task_ids_preview": [
                tasks[index].task_id for index in missing_indices[:10]
            ],
            "scope": "every_selected_task_setup/base-state/cleanup operational preflight",
        },
        "repository_population": repository_population,
        "repository_smoke": repository_smoke,
        "task_environment_population": task_environment_population,
        "official_harness_preflight": harness_receipt,
        "resolved_rate": None,
        "resolved_rate_status": (
            "not_measured_preflight_only"
            if harness_receipt.get("status") == "passed"
            else "unmeasurable_until_official_harness_preflight_passes"
        ),
    }
    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else root
        / str(config["experiment"]["output_dir"])
        / "adapter_preflight_receipt.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**receipt, "receipt_path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    receipt = run(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "receipt_path": receipt["receipt_path"],
                "repository_smoke": receipt["repository_smoke"]["status"],
                "official_harness_preflight": receipt[
                    "official_harness_preflight"
                ]["status"],
                "resolved_rate": receipt["resolved_rate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
