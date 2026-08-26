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


class _EnvironmentFileSpec(_Spec):
    setup_env_script = (
        "#!/bin/bash\nset -euxo pipefail\n"
        "source /opt/miniconda3/bin/activate\n"
        "cat <<'EOF' > environment.yml\n"
        "# conda env create --file environment.yml\n"
        "name: testbed\n"
        "channels:\n"
        "  - defaults\n"
        "dependencies:\n"
        "  - python=3.9\n"
        "EOF\n"
        "conda env create --file environment.yml\n"
        "conda activate testbed\n"
    )


class _RequirementsSpec(_Spec):
    setup_env_script = (
        "#!/bin/bash\nset -euxo pipefail\n"
        "source /opt/miniconda3/bin/activate\n"
        "conda create -n testbed python=3.9 -y\n"
        "cat <<'EOF' > $HOME/requirements.txt\npytest\nEOF\n"
        "conda activate testbed && python -m pip install -r $HOME/requirements.txt\n"
        "rm $HOME/requirements.txt\n"
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
    assert "PYTHONSAFEPATH=1 conda activate swe_owner_repo_12" in setup
    assert "PYTHONSAFEPATH=1 conda activate swe_owner_repo_12" in install
    assert subprocess.run(["bash", "-n"], input=setup, text=True).returncode == 0
    assert subprocess.run(["bash", "-n"], input=install, text=True).returncode == 0


def test_render_env_file_uses_nodefaults_without_unsupported_channel_flags(tmp_path):
    base = _plan()
    plan = module.EnvironmentPlan(
        repo=base.repo,
        version=base.version,
        environment_name=base.environment_name,
        instance_ids=base.instance_ids,
        representative_instance_id=base.representative_instance_id,
        representative_base_commit=base.representative_base_commit,
        setup_env_script=_EnvironmentFileSpec.setup_env_script,
        install_repo_script=base.install_repo_script,
    )
    setup, _ = module.render_official_scripts(
        plan,
        conda_executable=tmp_path / "miniconda/bin/conda",
        envs_dir=tmp_path / "envs",
        source_path=tmp_path / "sources" / plan.environment_name,
    )

    assert "conda env create --override-channels" not in setup
    assert "conda env update --override-channels" not in setup
    assert "export CONDA_CHANNELS=conda-forge" in setup
    assert "export CONDA_DEFAULT_CHANNELS=conda-forge" in setup
    assert "sed -i" in setup and "nodefaults" in setup
    assert "  - defaults" in setup  # preserved in the here-doc, rewritten at runtime
    assert "# conda env create --file environment.yml" in setup
    assert "if [ -x " in setup
    assert "conda env update --file environment.yml" in setup
    assert subprocess.run(["bash", "-n"], input=setup, text=True).returncode == 0

    conda = tmp_path / "miniconda/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\n", encoding="utf-8")
    conda.chmod(0o755)
    conda_init = tmp_path / "miniconda/etc/profile.d/conda.sh"
    conda_init.parent.mkdir(parents=True)
    conda_init.write_text("conda() { return 0; }\n", encoding="utf-8")
    runtime_setup, _ = module.render_official_scripts(
        plan,
        conda_executable=conda,
        envs_dir=tmp_path / "envs",
        source_path=tmp_path / "sources" / plan.environment_name,
    )
    result = subprocess.run(
        ["bash"],
        input=runtime_setup,
        text=True,
        capture_output=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    environment_file = (tmp_path / "environment.yml").read_text(encoding="utf-8")
    assert "  - nodefaults" in environment_file
    assert "  - defaults" not in environment_file


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
    assert f"cd {state / _plan().environment_name / 'work'}" in setup_script


def test_prepare_relocates_shared_requirements_file_and_makes_cleanup_idempotent(
    tmp_path,
):
    base = _plan()
    plan = module.EnvironmentPlan(
        repo=base.repo,
        version=base.version,
        environment_name=base.environment_name,
        instance_ids=base.instance_ids,
        representative_instance_id=base.representative_instance_id,
        representative_base_commit=base.representative_base_commit,
        setup_env_script=_RequirementsSpec.setup_env_script,
        install_repo_script=base.install_repo_script,
    )
    conda = tmp_path / "miniconda/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\n", encoding="utf-8")
    conda.chmod(0o755)
    envs = tmp_path / "envs"
    sources = tmp_path / "sources"
    state = tmp_path / "state"

    def runner(script_path, *, timeout_seconds):
        if script_path.name == "setup_env.sh":
            python = envs / plan.environment_name / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
        else:
            (sources / plan.environment_name / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(["bash"], 0, "", "")

    module.prepare_environment(
        plan,
        conda_executable=conda,
        envs_dir=envs,
        source_root=sources,
        state_root=state,
        timeout_seconds=10,
        runner=runner,
    )
    setup = (state / plan.environment_name / "setup_env.sh").read_text(
        encoding="utf-8"
    )
    private_requirements = state / plan.environment_name / "work/requirements.txt"
    assert "$HOME/requirements.txt" not in setup
    assert str(private_requirements) in setup
    assert f"rm -f {private_requirements}" in setup


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


def test_retry_reenters_failed_setup_without_recreating_existing_prefix(tmp_path):
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

    def failed_setup(script_path, *, timeout_seconds):
        calls.append(script_path.name)
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        return subprocess.CompletedProcess(["bash"], 1, "", "cleanup failed")

    kwargs = dict(
        conda_executable=conda,
        envs_dir=envs,
        source_root=sources,
        state_root=state,
        timeout_seconds=10,
    )
    failed = module.prepare_environment(plan, runner=failed_setup, **kwargs)

    def retry(script_path, *, timeout_seconds):
        calls.append(script_path.name)
        if script_path.name == "setup_env.sh":
            generated = script_path.read_text(encoding="utf-8")
            assert f"if [ ! -x {python.resolve()} ]; then" in generated
        else:
            (sources / plan.environment_name / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(["bash"], 0, "", "")

    ready = module.prepare_environment(plan, runner=retry, **kwargs)

    assert failed["failed_phase"] == "setup_env"
    assert calls == ["setup_env.sh", "setup_env.sh", "install_repo.sh"]
    assert ready["status"] == "ready"


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
