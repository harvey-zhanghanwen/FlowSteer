#!/usr/bin/env python3
"""Prepare the task environments used by SkillFlow's SWE-bench runtime.

The environment name comes from SkillFlow's deployed ``_env_name`` helper.
All dependency and repository installation commands come from the official
SWE-bench ``make_test_spec`` result.  This script only relocates the harness'
container paths/names to the selected local Conda installation and persistent
source checkout; it contains no repository-specific dependency table.

``SWE_BENCH_ENV_SOURCE_ROOT`` is the shared runtime contract for locating the
persistent editable-install source at ``<root>/<environment_name>``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_DATASET = Path("data/swebench_skillflow_v3/test.jsonl")
DEFAULT_SKILLFLOW_EVALUATOR = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/swe_bench_eval.py"
)
DEFAULT_HARNESS = Path(
    "/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83"
)
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "SWE_BENCH_ENV_SOURCE_ROOT",
        "/ssd1/iclr/.private/skillflow-resources/swe-env-sources",
    )
)
DEFAULT_STATE_ROOT = Path(
    "/ssd1/iclr/.private/skillflow-resources/swe-env-build-state"
)
DEFAULT_RECEIPT = Path(
    "artifacts/swebench_skillflow_v3_initial_v1/"
    "task_environment_preparation_receipt.json"
)
RECEIPT_SCHEMA = "flowsteer.swebench.skillflow-task-environments.v1"


@dataclass(frozen=True, slots=True)
class EnvironmentPlan:
    repo: str
    version: str
    environment_name: str
    instance_ids: tuple[str, ...]
    representative_instance_id: str
    representative_base_commit: str
    setup_env_script: str
    install_repo_script: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_module(path: Path, name: str) -> Any:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("SWE-bench task population is empty")
    return rows


def _official_instance(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("SWE-bench row has no metadata")
    payload = metadata.get("evaluator_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("SWE-bench row has no evaluator_payload")
    required = ("instance_id", "repo", "version", "base_commit", "test_patch")
    missing = [key for key in required if not isinstance(payload.get(key), str)]
    if missing:
        raise ValueError("SWE-bench evaluator_payload is missing: " + ", ".join(missing))
    return {
        "instance_id": payload["instance_id"],
        "repo": payload["repo"],
        "version": payload["version"],
        "base_commit": payload["base_commit"],
        "environment_setup_commit": payload.get("environment_setup_commit", ""),
        "problem_statement": row.get("question", ""),
        "test_patch": payload["test_patch"],
        "FAIL_TO_PASS": payload.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": payload.get("PASS_TO_PASS", []),
    }


def plan_environments(
    rows: Iterable[Mapping[str, Any]],
    *,
    env_name: Callable[[str, str], str],
    make_test_spec: Callable[..., Any],
) -> list[EnvironmentPlan]:
    """Group the fixed population by SkillFlow's repo/version environment key."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        instance = _official_instance(row)
        key = (str(instance["repo"]), str(instance["version"]))
        groups.setdefault(key, []).append(instance)

    plans: list[EnvironmentPlan] = []
    seen_names: dict[str, tuple[str, str]] = {}
    for (repo, version), instances in groups.items():
        name = str(env_name(repo, version)).strip()
        if not name:
            raise ValueError(f"SkillFlow returned an empty environment name for {repo}@{version}")
        prior = seen_names.setdefault(name, (repo, version))
        if prior != (repo, version):
            raise ValueError(f"SkillFlow environment-name collision: {name}")
        specs = [make_test_spec(instance, namespace="swebench") for instance in instances]
        setup_scripts = {str(spec.setup_env_script) for spec in specs}
        if len(setup_scripts) != 1:
            raise ValueError(
                f"official setup_env_script differs inside {repo}@{version}; "
                "cannot share one SkillFlow environment"
            )
        representative = instances[0]
        plans.append(
            EnvironmentPlan(
                repo=repo,
                version=version,
                environment_name=name,
                instance_ids=tuple(str(item["instance_id"]) for item in instances),
                representative_instance_id=str(representative["instance_id"]),
                representative_base_commit=str(representative["base_commit"]),
                setup_env_script=setup_scripts.pop(),
                install_repo_script=str(specs[0].install_repo_script),
            )
        )
    return sorted(plans, key=lambda plan: (plan.repo, plan.version))


def render_official_scripts(
    plan: EnvironmentPlan,
    *,
    conda_executable: Path,
    envs_dir: Path,
    source_path: Path,
) -> tuple[str, str]:
    """Relocate only harness container paths and its fixed ``testbed`` name."""

    conda_root = conda_executable.expanduser().resolve().parent.parent
    activate = conda_root / "etc/profile.d/conda.sh"
    setup = plan.setup_env_script.replace(
        "source /opt/miniconda3/bin/activate",
        f"source {shlex.quote(str(activate))}",
    )
    install = plan.install_repo_script.replace(
        "source /opt/miniconda3/bin/activate",
        f"source {shlex.quote(str(activate))}",
    )
    setup = re.sub(r"\btestbed\b", plan.environment_name, setup)
    # NECESSARY_ADAPTATION: this host has not accepted the default
    # repo.anaconda.com channel terms, while conda-forge is available for all
    # Python versions in the fixed population.  Preserve the official package
    # list and only route Conda environment creation through conda-forge.
    setup = setup.replace(
        "conda create -n ",
        "conda create --override-channels -c conda-forge -n ",
    )
    setup = setup.replace(
        "conda create -c conda-forge -n ",
        "conda create --override-channels -c conda-forge -n ",
    )
    setup = setup.replace(
        "conda env create ",
        "conda env create --override-channels -c conda-forge ",
    )
    quoted_source = shlex.quote(str(source_path.expanduser().resolve()))
    install = install.replace("/testbed", quoted_source)
    install = re.sub(r"\btestbed\b", plan.environment_name, install)
    # The checkout persists for editable installs.  Retrying a partial build
    # reuses it and resets to the same official representative base commit.
    clone_prefix = f"git clone -o origin https://github.com/{plan.repo} {quoted_source}"
    resume_clone = (
        f"if [ -d {quoted_source}/.git ]; then "
        f"git -C {quoted_source} reset --hard {shlex.quote(plan.representative_base_commit)}; "
        f"else {clone_prefix}; fi"
    )
    install = install.replace(clone_prefix, resume_clone, 1)
    install = install.replace("git remote remove origin", "git remote remove origin || true")
    exports = (
        f"export CONDA_EXE={shlex.quote(str(conda_executable.expanduser().resolve()))}\n"
        f"export CONDA_ENVS_DIR={shlex.quote(str(envs_dir.expanduser().resolve()))}\n"
        f"export CONDA_ENVS_PATH={shlex.quote(str(envs_dir.expanduser().resolve()))}\n"
        f"export SWE_BENCH_ENV_SOURCE_ROOT={shlex.quote(str(source_path.parent.resolve()))}\n"
        "export CUDA_VISIBLE_DEVICES=\n"
    )
    return exports + setup, exports + install


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _ready_on_disk(
    previous: Mapping[str, Any] | None, *, python_path: Path, source_path: Path
) -> bool:
    return bool(
        previous
        and previous.get("status") == "ready"
        and python_path.is_file()
        and os.access(python_path, os.X_OK)
        and (source_path / ".git").is_dir()
    )


def _run_script(path: Path, *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(path)],
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )


def prepare_environment(
    plan: EnvironmentPlan,
    *,
    conda_executable: Path,
    envs_dir: Path,
    source_root: Path,
    state_root: Path,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run_script,
) -> dict[str, Any]:
    source_path = source_root / plan.environment_name
    python_path = envs_dir / plan.environment_name / "bin/python"
    env_state = state_root / plan.environment_name
    receipt_path = env_state / "receipt.json"
    previous = _read_json(receipt_path)
    previous_phases = {
        str(value)
        for value in ((previous or {}).get("completed_phases") or [])
        if isinstance(value, str)
    }
    base = {
        "repo": plan.repo,
        "version": plan.version,
        "environment_name": plan.environment_name,
        "instance_count": len(plan.instance_ids),
        "representative_instance_id": plan.representative_instance_id,
        "source_path": str(source_path.resolve()),
        "python_path": str(python_path.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "source": "SkillFlow _env_name + official SWE-bench make_test_spec scripts",
    }
    if _ready_on_disk(previous, python_path=python_path, source_path=source_path):
        return {
            **base,
            "status": "ready",
            "ready": True,
            "resumed": True,
            "completed_phases": list(previous.get("completed_phases", [])),
        }

    env_state.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    envs_dir.mkdir(parents=True, exist_ok=True)
    setup, install = render_official_scripts(
        plan,
        conda_executable=conda_executable,
        envs_dir=envs_dir,
        source_path=source_path,
    )
    setup_path = env_state / "setup_env.sh"
    install_path = env_state / "install_repo.sh"
    setup_path.write_text(setup, encoding="utf-8")
    install_path.write_text(install, encoding="utf-8")
    completed_phases = set(previous_phases)
    row: dict[str, Any] = {
        **base,
        "status": "building",
        "resumed": bool(previous),
        "completed_phases": sorted(completed_phases),
        "started_at": _utc_now(),
    }
    receipt_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        for phase, script_path in (("setup_env", setup_path), ("install_repo", install_path)):
            if phase == "setup_env" and phase in completed_phases and python_path.is_file():
                continue
            result = runner(script_path, timeout_seconds=timeout_seconds)
            (env_state / f"{phase}.stdout.log").write_text(result.stdout or "", encoding="utf-8")
            (env_state / f"{phase}.stderr.log").write_text(result.stderr or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"{phase} exited with status {result.returncode}")
            completed_phases.add(phase)
            row["completed_phases"] = sorted(completed_phases)
            receipt_path.write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not python_path.is_file() or not os.access(python_path, os.X_OK):
            raise RuntimeError("SkillFlow environment Python is unavailable after setup")
        if not (source_path / ".git").is_dir():
            raise RuntimeError("persistent repository source checkout is unavailable after setup")
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "ready": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": _utc_now(),
            }
        )
    else:
        row.update({"status": "ready", "ready": True, "finished_at": _utc_now()})
    receipt_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--skillflow-evaluator", type=Path, default=DEFAULT_SKILLFLOW_EVALUATOR)
    parser.add_argument("--harness-path", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--conda-executable", type=Path, required=True)
    parser.add_argument("--conda-envs-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="persistent source root (also exported as SWE_BENCH_ENV_SOURCE_ROOT)",
    )
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--receipt-path", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--limit-environments", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    conda = args.conda_executable.expanduser().resolve()
    harness = args.harness_path.expanduser().resolve()
    if not conda.is_file() or not os.access(conda, os.X_OK):
        raise SystemExit(f"Conda executable is unavailable: {conda}")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.limit_environments is not None and args.limit_environments <= 0:
        raise SystemExit("--limit-environments must be positive")
    module = _load_module(args.skillflow_evaluator, "_flowsteer_swe_env_skillflow")
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    from swebench.harness.test_spec.test_spec import make_test_spec

    rows = _load_jsonl(args.dataset_jsonl)
    plans = plan_environments(rows, env_name=module._env_name, make_test_spec=make_test_spec)
    if args.limit_environments is not None:
        plans = plans[: args.limit_environments]
    os.environ["CONDA_EXE"] = str(conda)
    os.environ["CONDA_ENVS_DIR"] = str(args.conda_envs_dir.expanduser().resolve())
    os.environ["SWE_BENCH_ENV_SOURCE_ROOT"] = str(args.source_root.expanduser().resolve())
    if args.plan_only:
        results = [
            {
                "repo": plan.repo,
                "version": plan.version,
                "environment_name": plan.environment_name,
                "instance_count": len(plan.instance_ids),
                "source_path": str((args.source_root / plan.environment_name).resolve()),
                "status": "planned",
                "ready": False,
            }
            for plan in plans
        ]
    else:
        kwargs = {
            "conda_executable": conda,
            "envs_dir": args.conda_envs_dir.expanduser().resolve(),
            "source_root": args.source_root.expanduser().resolve(),
            "state_root": args.state_root.expanduser().resolve(),
            "timeout_seconds": args.timeout_seconds,
        }
        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(prepare_environment, plan, **kwargs): plan for plan in plans}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: (str(row["repo"]), str(row["version"])))
    ready = sum(row.get("status") == "ready" for row in results)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "dataset_path": str(args.dataset_jsonl.expanduser().resolve()),
        "task_count": len(rows),
        "environment_count": len(results),
        "ready": ready,
        "failed": sum(row.get("status") == "failed" for row in results),
        "planned": sum(row.get("status") == "planned" for row in results),
        "all_ready": bool(results) and ready == len(results),
        "max_concurrency": args.jobs,
        "runtime_environment": {
            "CONDA_EXE": str(conda),
            "CONDA_ENVS_DIR": str(args.conda_envs_dir.expanduser().resolve()),
            "SWE_BENCH_ENV_SOURCE_ROOT": str(args.source_root.expanduser().resolve()),
        },
        "rows": results,
    }
    receipt_path = args.receipt_path.expanduser().resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("task_count", "environment_count", "ready", "failed", "planned", "all_ready")}, sort_keys=True))
    return 0 if args.plan_only or receipt["all_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
