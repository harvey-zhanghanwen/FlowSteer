from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/prepare_swebench_skillflow_task_environments.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_swebench_skillflow_task_environments", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


def _row(instance_id: str, *, repo: str = "owner/repo", version: str = "1.2"):
    return {
        "question": "fix it",
        "metadata": {
            "evaluator_payload": {
                "instance_id": instance_id,
                "repo": repo,
                "version": version,
                "base_commit": f"base-{instance_id}",
                "environment_setup_commit": "env-base",
                "test_patch": "diff --git a/test.py b/test.py\n",
                "FAIL_TO_PASS": ["test.py::test_fix"],
                "PASS_TO_PASS": [],
            }
        },
    }


class _Spec:
    setup_env_script = (
        "#!/bin/bash\nset -euxo pipefail\n"
        "source /opt/miniconda3/bin/activate\n"
        "conda create -n testbed python=3.9 -y\n"
        "conda activate testbed\n"
    )

    def __init__(self, instance):
        self.install_repo_script = (
            "#!/bin/bash\nset -euxo pipefail\n"
            f"git clone -o origin https://github.com/{instance['repo']} /testbed\n"
            "cd /testbed\n"
            f"git reset --hard {instance['base_commit']}\n"
            "git remote remove origin\n"
            "source /opt/miniconda3/bin/activate\n"
            "conda activate testbed\n"
            "python -m pip install -e .\n"
        )


def _plan():
    return module.plan_environments(
        [_row("one"), _row("two")],
        env_name=lambda repo, version: f"swe_{repo.replace('/', '_')}_{version.replace('.', '')}",
        make_test_spec=lambda instance, namespace: _Spec(instance),
    )[0]


def test_plan_groups_fixed_population_by_repo_version_and_skillflow_name():
    plans = module.plan_environments(
        [_row("one"), _row("two"), _row("other", repo="other/repo", version="3.0")],
        env_name=lambda repo, version: f"swe_{repo.replace('/', '_')}_{version.replace('.', '')}",
        make_test_spec=lambda instance, namespace: _Spec(instance),
    )

    assert len(plans) == 2
    assert plans[0].environment_name == "swe_other_repo_30"
    assert plans[1].instance_ids == ("one", "two")
    assert plans[1].representative_instance_id == "one"


def test_plan_fails_closed_when_official_env_specs_conflict():
    class ConflictingSpec(_Spec):
        def __init__(self, instance):
            super().__init__(instance)
            self.setup_env_script = _Spec.setup_env_script + instance["instance_id"]

    with pytest.raises(ValueError, match="setup_env_script differs"):
        module.plan_environments(
            [_row("one"), _row("two")],
            env_name=lambda repo, version: "swe_owner_repo_12",
            make_test_spec=lambda instance, namespace: ConflictingSpec(instance),
        )


def test_render_uses_local_conda_and_persistent_source_without_dependency_table(tmp_path):
    conda = tmp_path / "miniconda/bin/conda"
    source = tmp_path / "sources/swe_owner_repo_12"
    setup, install = module.render_official_scripts(
        _plan(),
        conda_executable=conda,
        envs_dir=tmp_path / "envs",
        source_path=source,
    )

    assert f"source {tmp_path}/miniconda/etc/profile.d/conda.sh" in setup
    assert (
        "conda create --override-channels -c conda-forge "
        "-n swe_owner_repo_12 python=3.9 -y"
    ) in setup
    assert f"export SWE_BENCH_ENV_SOURCE_ROOT={source.parent}" in install
    assert f"git clone -o origin https://github.com/owner/repo {source}" in install
    assert f"git -C {source} reset --hard base-one" in install
    assert "python -m pip install -e ." in install
    assert "/testbed" not in setup + install


def test_prepare_records_ready_and_resume_skips_completed_environment(tmp_path):
    conda = tmp_path / "miniconda/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\n", encoding="utf-8")
    conda.chmod(0o755)
    envs = tmp_path / "envs"
    sources = tmp_path / "sources"
    state = tmp_path / "state"
    calls = []

    def runner(script_path, *, timeout_seconds):
        calls.append(script_path.name)
        if script_path.name == "setup_env.sh":
            python = envs / _plan().environment_name / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
        else:
            (sources / _plan().environment_name / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(["bash"], 0, "ok", "")

    kwargs = dict(
        conda_executable=conda,
        envs_dir=envs,
        source_root=sources,
        state_root=state,
        timeout_seconds=10,
    )
    first = module.prepare_environment(_plan(), runner=runner, **kwargs)
    second = module.prepare_environment(
        _plan(),
        runner=lambda *args, **kwargs: pytest.fail("ready environment was rebuilt"),
        **kwargs,
    )

    assert calls == ["setup_env.sh", "install_repo.sh"]
    assert first["status"] == "ready"
    assert second["status"] == "ready" and second["resumed"] is True
    assert first["source_path"] == str((sources / _plan().environment_name).resolve())
    setup_script = (state / _plan().environment_name / "setup_env.sh").read_text(
        encoding="utf-8"
    )
    assert f"export HOME={state / _plan().environment_name / 'home'}" in setup_script
    assert f"cd {state / _plan().environment_name / 'work'}" in setup_script


def test_prepare_persists_failure_and_never_marks_it_ready(tmp_path):
    conda = tmp_path / "miniconda/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\n", encoding="utf-8")
    conda.chmod(0o755)

    row = module.prepare_environment(
        _plan(),
        conda_executable=conda,
        envs_dir=tmp_path / "envs",
        source_root=tmp_path / "sources",
        state_root=tmp_path / "state",
        timeout_seconds=10,
        runner=lambda path, timeout_seconds: subprocess.CompletedProcess(
            ["bash"], 7, "", "setup failed"
        ),
    )

    receipt = json.loads(Path(row["receipt_path"]).read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["ready"] is False
    assert receipt["error"] == "setup_env exited with status 7"


def test_retry_continues_after_completed_environment_setup(tmp_path):
    conda = tmp_path / "miniconda/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\n", encoding="utf-8")
    conda.chmod(0o755)
    plan = _plan()
    envs = tmp_path / "envs"
    sources = tmp_path / "sources"
    state = tmp_path / "state"
    python = envs / plan.environment_name / "bin/python"
    calls = []

    def first_runner(script_path, *, timeout_seconds):
        calls.append(script_path.name)
        if script_path.name == "setup_env.sh":
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            return subprocess.CompletedProcess(["bash"], 0, "", "")
        return subprocess.CompletedProcess(["bash"], 2, "", "install failed")

    kwargs = dict(
        conda_executable=conda,
        envs_dir=envs,
        source_root=sources,
        state_root=state,
        timeout_seconds=10,
    )
    failed = module.prepare_environment(plan, runner=first_runner, **kwargs)

    def retry_runner(script_path, *, timeout_seconds):
        calls.append(script_path.name)
        assert script_path.name == "install_repo.sh"
        (sources / plan.environment_name / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(["bash"], 0, "", "")

    ready = module.prepare_environment(plan, runner=retry_runner, **kwargs)

    assert failed["status"] == "failed"
    assert calls == ["setup_env.sh", "install_repo.sh", "install_repo.sh"]
    assert ready["status"] == "ready"
    assert ready["completed_phases"] == ["install_repo", "setup_env"]


def test_cli_caps_parallelism_at_two():
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--conda-executable",
                "/conda",
                "--conda-envs-dir",
                "/envs",
                "--jobs",
                "3",
            ]
        )


def test_cli_accepts_repeatable_exact_environment_filter():
    args = module._parser().parse_args(
        [
            "--conda-executable",
            "/conda",
            "--conda-envs-dir",
            "/envs",
            "--environment-name",
            "swe_owner_repo_12",
            "--environment-name",
            "swe_other_repo_30",
        ]
    )
    assert args.environment_name == ["swe_owner_repo_12", "swe_other_repo_30"]
