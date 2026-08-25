from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prepare_alfworld_protocol_v10.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_alfworld_protocol_v10",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _catalog(tmp_path: Path, output_name: str = "protocol") -> Path:
    catalog = {
        "schema_version": "flowsteer.agentgraph.datasets.v1",
        "protocol_version": "skillflow.protocol.v10",
        "aligned_dir": str(tmp_path / output_name),
        "aligned_files": {
            "preflight_train": str(tmp_path / output_name / "preflight_train.jsonl"),
            "valid_seen": str(tmp_path / output_name / "valid_seen.jsonl"),
            "valid_unseen": str(tmp_path / output_name / "valid_unseen.jsonl"),
            "manifest": str(tmp_path / output_name / "manifest.json"),
        },
        "action_policy_budget": 20,
        "simulator_step_cap": 50,
        "official_populations": {
            "valid_seen": {
                "mode": "eval_in_distribution",
                "output_file": "valid_seen.jsonl",
            },
            "valid_unseen": {
                "mode": "eval_out_of_distribution",
                "output_file": "valid_unseen.jsonl",
            },
        },
        "evaluator_preflight": {
            "mode": "train",
            "output_file": "preflight_train.jsonl",
            "sample_count": 1,
        },
        "train_heldout_smoke": {
            "task_file": "data/alfworld_v2/validation.jsonl",
            "manifest_file": "data/alfworld_v2/manifest.json",
            "project_split": "validation",
            "scope": "train_inventory_heldout_smoke_only",
        },
        "sources": {
            "alfworld": {
                "enabled": True,
                "display_name": "ALFWorld",
                "task_type": "interactive_agent",
                "metric": "success_rate",
                "ragen_adapter_path": str(tmp_path / "ragen_adapter.py"),
                "alfworld_path": str(tmp_path / "repo"),
                "alfworld_data": str(tmp_path / "data"),
                "config_file": str(tmp_path / "base_config.yaml"),
            }
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def _tasks(official_split: str, mode: str, count: int) -> list[dict[str, Any]]:
    families = _MODULE.OFFICIAL_TASK_FAMILIES
    return [
        {
            "official_split": official_split,
            "mode": mode,
            "task_family": families[index % len(families)],
            "game_index": index,
            "game_file": str(
                Path(
                    f"/official/{official_split}/"
                    f"{families[index % len(families)]}-Obj-None-Target-{index}/"
                    "trial/game.tw-pddl"
                )
            ),
            "canonical_instruction": (
                f"canonical {official_split} instruction {index}."
            ),
            "task_directory": f"task-{index}",
        }
        for index in range(count)
    ]


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_prepare_keeps_official_populations_and_budget_boundaries(
    tmp_path: Path,
) -> None:
    catalog_path = _catalog(tmp_path)
    requests: list[Mapping[str, Any]] = []

    def provider(request: Mapping[str, Any]):
        requests.append(dict(request))
        return _tasks(
            str(request["official_split"]),
            str(request["mode"]),
            int(request.get("sample_count", 3)),
        )

    output = _MODULE.prepare(catalog_path, task_provider=provider)

    assert [request["official_split"] for request in requests] == [
        "valid_seen",
        "valid_unseen",
        "preflight_train",
    ]
    assert all(request["action_policy_budget"] == 20 for request in requests)
    assert all(request["simulator_step_cap"] == 50 for request in requests)

    for official_split, mode in _MODULE.OFFICIAL_POPULATIONS.items():
        records = _records(output / f"{official_split}.jsonl")
        assert len(records) == 3
        assert {record["split"] for record in records} == {"test"}
        first = records[0]
        assert first["task_id"] == f"alfworld:{official_split}:00000"
        assert first["question"] == (
            f"canonical {official_split} instruction 0."
        )
        assert first["extra"]["official_split"] == official_split
        assert first["extra"]["mode"] == mode
        assert first["extra"]["game_index"] == 0
        assert first["extra"]["game_file"] == first["env_config"]["game_file"]
        assert first["extra"]["canonical_instruction"] == first["question"]
        assert first["extra"]["action_policy_budget"] == 20
        assert first["extra"]["simulator_step_cap"] == 50
        assert first["env_config"]["max_steps"] == 20
        assert first["env_config"]["simulator_step_cap"] == 50
        assert first["metadata"]["protocol"] == {
            "protocol_version": "skillflow.protocol.v10",
            "official_split": official_split,
            "mode": mode,
            "action_policy_budget": 20,
            "simulator_step_cap": 50,
            "terminal_evaluator": "official_environment_success",
        }

    preflight = _records(output / "preflight_train.jsonl")
    assert len(preflight) == 1
    assert preflight[0]["split"] == "train"
    assert preflight[0]["task_id"] == "alfworld:preflight_train:00000"

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts_by_official_split"] == {
        "valid_seen": 3,
        "valid_unseen": 3,
    }
    assert manifest["task_project_split"] == "test"
    assert manifest["action_policy_budget"] == 20
    assert manifest["simulator_step_cap"] == 50
    assert manifest["training_started"] is False
    assert manifest["evaluator_preflight"] == {
        "mode": "train",
        "file": "preflight_train.jsonl",
        "sample_count": 1,
    }
    assert manifest["train_heldout_smoke_reference"]["task_file"] == (
        "data/alfworld_v2/validation.jsonl"
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("action_policy_budget", 50, "action_policy_budget"),
        ("simulator_step_cap", 20, "simulator_step_cap"),
    ],
)
def test_prepare_rejects_conflated_policy_and_simulator_budgets(
    tmp_path: Path,
    field: str,
    value: int,
    match: str,
) -> None:
    catalog_path = _catalog(tmp_path)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog[field] = value
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _MODULE.prepare(catalog_path, task_provider=lambda _: ())


def test_default_provider_reuses_ragen_modes_and_resets_every_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_path = tmp_path / "ragen_adapter.py"
    adapter_path.touch()
    alfworld_path = tmp_path / "repo"
    alfworld_path.mkdir()
    alfworld_data = tmp_path / "data"
    alfworld_data.mkdir()
    config_file = tmp_path / "base_config.yaml"
    config_file.touch()
    families = _MODULE.OFFICIAL_TASK_FAMILIES
    inventories = {
        mode: [
            str(
                tmp_path
                / mode
                / f"{families[index]}-Obj-None-Target-{index}"
                / "trial"
                / "game.tw-pddl"
            )
            for index in range(2)
        ]
        for mode in _MODULE.OFFICIAL_POPULATIONS.values()
    }
    reset_receipts: list[tuple[str, int]] = []
    closed: list[tuple[str, int]] = []

    class AlfredEnvConfig:
        def __init__(self, config_file="") -> None:
            self.config_file = config_file

    class ALFWorldEnv:
        def __init__(self, config, mode) -> None:
            self.game_files = inventories[mode]

    class LiveEnvironment:
        def __init__(self, mode: str, seed: int) -> None:
            self.current_game_index = seed
            self.current_game_file = inventories[mode][seed]
            self.current_task_dir = Path(self.current_game_file).parent.parent.name
            self.task_description = f"canonical {mode} {seed}."
            self.alfred_env = SimpleNamespace(
                close=lambda: closed.append((mode, seed))
            )

    class RAGENAdapter:
        def __init__(self) -> None:
            self._env = None

        def reset(self, env_type, config, *, question, extra):
            assert env_type == "alfworld"
            assert question == ""
            assert extra == {}
            mode = str(config["mode"])
            seed = int(config["seed"])
            reset_receipts.append((mode, seed))
            self._env = LiveEnvironment(mode, seed)
            return f"Room.\nYour task is to: canonical {mode} {seed}."

    fake_module = SimpleNamespace(
        RAGENAdapter=RAGENAdapter,
        AlfredEnvConfig=AlfredEnvConfig,
        ALFWorldEnv=ALFWorldEnv,
        _check_alfworld=lambda: True,
    )
    monkeypatch.setattr(
        _MODULE._LEGACY,
        "_load_python_module",
        lambda _name, _path: fake_module,
    )
    source = {
        "ragen_adapter_path": str(adapter_path),
        "alfworld_path": str(alfworld_path),
        "alfworld_data": str(alfworld_data),
        "config_file": str(config_file),
    }
    request = {
        "official_split": "valid_unseen",
        "mode": "eval_out_of_distribution",
    }

    tasks = _MODULE._default_task_provider(
        request,
        source=source,
        base=tmp_path,
    )

    assert len(tasks) == 2
    assert reset_receipts == [
        ("eval_out_of_distribution", 0),
        ("eval_out_of_distribution", 1),
    ]
    assert closed == reset_receipts
    assert tasks[0]["official_split"] == "valid_unseen"
    assert tasks[0]["mode"] == "eval_out_of_distribution"
    assert tasks[0]["game_file"] == str(
        Path(inventories["eval_out_of_distribution"][0]).resolve()
    )
    assert tasks[0]["canonical_instruction"] == (
        "canonical eval_out_of_distribution 0."
    )


def test_repository_catalog_matches_skillflow_protocol_v10() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog_path = root / "config" / "datasets_alfworld_protocol_v10.yaml"
    catalog = _MODULE._catalog_mapping(catalog_path)

    assert _MODULE._population_specs(catalog) == (
        {
            "official_split": "valid_seen",
            "mode": "eval_in_distribution",
            "output_file": "valid_seen.jsonl",
        },
        {
            "official_split": "valid_unseen",
            "mode": "eval_out_of_distribution",
            "output_file": "valid_unseen.jsonl",
        },
    )
    assert catalog["aligned_dir"] == "data/alfworld_protocol_v10"
    assert catalog["action_policy_budget"] == 20
    assert catalog["simulator_step_cap"] == 50
    assert _MODULE._preflight_spec(catalog) == {
        "official_split": "preflight_train",
        "mode": "train",
        "output_file": "preflight_train.jsonl",
        "sample_count": 1,
    }
