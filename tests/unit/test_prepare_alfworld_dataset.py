from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml


_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "prepare_alfworld_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_alfworld_dataset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _catalog(tmp_path: Path, output_name: str = "aligned") -> Path:
    catalog = {
        "schema_version": "flowsteer.agentgraph.datasets.v1",
        "aligned_dir": str(tmp_path / output_name),
        "alignment_recipe": {
            "heldout_split": "validation",
            "heldout_count_per_dataset": 128,
            "train_count_per_dataset": 512,
            "selection": "sequential",
            "cycle_training_only": True,
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
                "candidate_selection": "family_round_robin",
                "task_families": list(_MODULE.OFFICIAL_TASK_FAMILIES),
                "env_config": {
                    "config_file": str(tmp_path / "base_config.yaml"),
                    "mode": "train",
                    "max_steps": 50,
                },
            }
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def _tasks(count: int) -> list[dict[str, Any]]:
    families = _MODULE.OFFICIAL_TASK_FAMILIES
    return [
        {
            "task_family": families[index % len(families)],
            "game_index": index,
            "game_file": f"/official/{families[index % len(families)]}-Obj-None-Target-{index}/trial/game.tw-pddl",
            "game_seed": index,
            "canonical_instruction": f"canonical instruction {index}.",
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


def test_round_robin_uses_official_family_order_and_inventory_indices() -> None:
    families = _MODULE.OFFICIAL_TASK_FAMILIES
    # Inventory order is deliberately grouped and unbalanced.
    game_files = [
        f"/data/{family}-Object-None-Target-{ordinal}/trial/game.tw-pddl"
        for family in reversed(families)
        for ordinal in range(2)
    ]

    ordered = _MODULE._round_robin_inventory(game_files, families)

    assert [item[0] for item in ordered[:6]] == list(families)
    assert [item[0] for item in ordered[6:12]] == list(families)
    assert [item[1] for item in ordered] != list(range(len(game_files)))
    for family, index, game_file in ordered:
        assert family == _MODULE._game_family(game_file)
        assert Path(game_file).resolve() == Path(game_files[index]).resolve()


def test_exact_canonical_instruction_identity_and_split_are_preserved(
    tmp_path: Path,
) -> None:
    catalog_path = _catalog(tmp_path)
    tasks = _tasks(700)
    received_configs: list[Mapping[str, Any]] = []

    def fake_provider(env_config: Mapping[str, Any]):
        received_configs.append(dict(env_config))
        return tasks

    output = _MODULE.prepare(catalog_path, task_provider=fake_provider)

    validation = _records(output / "validation.jsonl")
    train = _records(output / "train.jsonl")
    assert _records(output / "test.jsonl") == []
    assert len(received_configs) == 1
    assert "seed" not in received_configs[0]
    assert "game_file" not in received_configs[0]
    assert len(validation) == 128
    assert len(train) == 512
    assert [row["env_config"]["seed"] for row in validation] == list(range(128))
    assert [row["env_config"]["seed"] for row in train] == list(range(128, 640))

    first = validation[0]
    assert first["question"] == "canonical instruction 0."
    assert first["extra"]["canonical_instruction"] == first["question"]
    assert first["env_config"]["game_file"] == tasks[0]["game_file"]
    assert first["env_config"]["max_steps"] == 50
    assert first["extra"]["max_steps"] == 50
    assert first["metadata"]["environment"]["env_config"] == first["env_config"]
    assert first["metadata"]["sampling"]["selection_index"] == 0
    assert train[0]["task_id"] == "alfworld:train:00128"
    assert {
        row["env_config"]["game_file"] for row in validation
    }.isdisjoint(row["env_config"]["game_file"] for row in train)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["single_game_identity_verified"] is True
    assert manifest["candidate_selection"] == "family_round_robin"
    assert manifest["max_steps"] == 50


def test_short_inventory_cycles_only_remaining_training_games(tmp_path: Path) -> None:
    catalog_path = _catalog(tmp_path)
    tasks = _tasks(131)

    output = _MODULE.prepare(catalog_path, task_provider=lambda _: tasks)

    validation = _records(output / "validation.jsonl")
    train = _records(output / "train.jsonl")
    validation_files = [row["env_config"]["game_file"] for row in validation]
    train_files = [row["env_config"]["game_file"] for row in train]
    assert len(validation_files) == 128
    assert train_files[:9] == [
        tasks[index]["game_file"] for index in (128, 129, 130, 128, 129, 130, 128, 129, 130)
    ]
    assert set(validation_files).isdisjoint(train_files)
    assert train[3]["task_id"] == "alfworld:train:00128:cycle-0001"
    assert train[3]["metadata"]["sampling"]["cycled_training_sample"] is True

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    status = manifest["sources"]["alfworld"]
    assert status["unique_train_candidates"] == 3
    assert status["cycled_train_records"] == 509


def test_default_provider_resets_each_selected_game_and_checks_identity(
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
    game_files = [
        str(
            tmp_path
            / f"{family}-Object-None-Target-{ordinal}"
            / "trial"
            / "game.tw-pddl"
        )
        for ordinal in range(22)
        for family in families
    ]
    reset_seeds: list[int] = []
    closed: list[int] = []

    class AlfredEnvConfig:
        def __init__(self, config_file="") -> None:
            self.config_file = config_file

    class ALFWorldEnv:
        def __init__(self, config, mode) -> None:
            self.game_files = game_files

    class LiveGame:
        def __init__(self, seed: int) -> None:
            self.current_game_index = seed
            self.current_game_file = game_files[seed]
            self.current_task_dir = Path(game_files[seed]).parent.parent.name
            self.task_description = f"canonical task {seed}."
            self.alfred_env = SimpleNamespace(close=lambda: closed.append(seed))

    class RAGENAdapter:
        def __init__(self) -> None:
            self._env = None

        def reset(self, env_type, config, *, question, extra):
            assert env_type == "alfworld"
            assert question == ""
            assert extra == {}
            seed = int(config["seed"])
            reset_seeds.append(seed)
            self._env = LiveGame(seed)
            return f"Room.\nYour task is to: canonical task {seed}."

    fake_module = SimpleNamespace(
        RAGENAdapter=RAGENAdapter,
        AlfredEnvConfig=AlfredEnvConfig,
        ALFWorldEnv=ALFWorldEnv,
        _check_alfworld=lambda: True,
    )
    monkeypatch.setattr(_MODULE, "_load_python_module", lambda _name, _path: fake_module)
    source = {
        "ragen_adapter_path": str(adapter_path),
        "alfworld_path": str(alfworld_path),
        "alfworld_data": str(alfworld_data),
        "candidate_selection": "family_round_robin",
        "task_families": list(families),
    }
    env_config = {
        "config_file": str(config_file),
        "mode": "train",
        "max_steps": 50,
    }

    loaded = _MODULE._default_task_provider(source, env_config, base=tmp_path)

    assert len(loaded) == len(game_files)
    assert [task["task_family"] for task in loaded[:6]] == list(families)
    assert reset_seeds == list(range(len(game_files)))
    assert closed == reset_seeds
    assert loaded[0]["canonical_instruction"] == "canonical task 0."
    assert loaded[0]["game_file"] == str(Path(game_files[0]).resolve())


def test_default_provider_fails_closed_when_reset_falls_to_another_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_path = tmp_path / "ragen_adapter.py"
    adapter_path.touch()
    (tmp_path / "repo").mkdir()
    (tmp_path / "data").mkdir()
    config_file = tmp_path / "base_config.yaml"
    config_file.touch()
    families = _MODULE.OFFICIAL_TASK_FAMILIES
    game_files = [
        str(tmp_path / f"{family}-Obj-None-Target-0" / "trial" / "game.tw-pddl")
        for family in families
        for _ in range(22)
    ]

    class AlfredEnvConfig:
        def __init__(self, config_file="") -> None:
            self.config_file = config_file

    class ALFWorldEnv:
        def __init__(self, config, mode) -> None:
            self.game_files = game_files

    class RAGENAdapter:
        def __init__(self) -> None:
            self._env = None

        def reset(self, env_type, config, *, question, extra):
            seed = int(config["seed"])
            actual = (seed + 1) % len(game_files)
            self._env = SimpleNamespace(
                current_game_index=actual,
                current_game_file=game_files[actual],
                current_task_dir="wrong",
                task_description="wrong task.",
                alfred_env=SimpleNamespace(close=lambda: None),
            )
            return "Room.\nYour task is to: wrong task."

    fake_module = SimpleNamespace(
        RAGENAdapter=RAGENAdapter,
        AlfredEnvConfig=AlfredEnvConfig,
        ALFWorldEnv=ALFWorldEnv,
        _check_alfworld=lambda: True,
    )
    monkeypatch.setattr(_MODULE, "_load_python_module", lambda _name, _path: fake_module)
    source = {
        "ragen_adapter_path": str(adapter_path),
        "alfworld_path": str(tmp_path / "repo"),
        "alfworld_data": str(tmp_path / "data"),
        "candidate_selection": "family_round_robin",
        "task_families": list(families),
    }
    env_config = {
        "config_file": str(config_file),
        "mode": "train",
        "max_steps": 50,
    }

    with pytest.raises(RuntimeError, match="single-game identity"):
        _MODULE._default_task_provider(source, env_config, base=tmp_path)


def test_repository_catalog_uses_official_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALFWORLD_CONFIG_FILE", raising=False)
    catalog_path = Path(__file__).resolve().parents[2] / "config" / "datasets_alfworld.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    source = catalog["sources"]["alfworld"]

    env_config = _MODULE._environment_config(source, base=catalog_path.parent.parent)

    assert env_config["mode"] == "train"
    assert env_config["max_steps"] == 50
    assert source["candidate_selection"] == "family_round_robin"
    assert tuple(source["task_families"]) == _MODULE.OFFICIAL_TASK_FAMILIES
    assert catalog["aligned_dir"] == "data/alfworld_v2"


@pytest.mark.parametrize(
    "config_name,expected_split,expected_count",
    [
        ("development_alfworld_round_01.yaml", "train", 16),
        ("evaluation_alfworld_round_01.yaml", "validation", 128),
    ],
)
def test_round_configs_use_official_aligned_data_and_50_step_cap(
    config_name: str,
    expected_split: str,
    expected_count: int,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config" / config_name).read_text(encoding="utf-8"))

    assert config["data"]["catalog_path"] == "config/datasets_alfworld.yaml"
    assert config["data"]["train_path"] == "data/alfworld_v2/train.jsonl"
    assert config["data"]["validation_path"] == "data/alfworld_v2/validation.jsonl"
    assert config["alfworld_evaluation"]["split"] == expected_split
    assert config["alfworld_evaluation"]["sample_count"] == expected_count
    assert config["evaluation"]["max_environment_steps"] == 50
    assert config["evaluation"]["max_environment_steps_by_source"]["alfworld"] == 50
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
