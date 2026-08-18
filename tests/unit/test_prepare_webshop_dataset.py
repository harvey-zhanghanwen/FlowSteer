from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "prepare_webshop_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_webshop_dataset", _SCRIPT)
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
            "webshop": {
                "enabled": True,
                "display_name": "WebShop",
                "task_type": "interactive_agent",
                "metric": "success_rate",
                # Fake providers make these inert; keeping them here exercises
                # the exact config that is persisted into every record.
                "ragen_adapter_path": str(tmp_path / "ragen_adapter.py"),
                "webshop_path": str(tmp_path / "webshop"),
                "search_index_path": str(tmp_path / "index"),
                "env_config": {
                    "observation_mode": "text",
                    "human_goals": True,
                    "use_small": False,
                    "num_products": None,
                    "goal_split": "all",
                    "file_path": str(tmp_path / "items_shuffle.json"),
                    "attr_path": str(tmp_path / "items_ins_v2.json"),
                    "human_attr_path": str(tmp_path / "items_human_ins.json"),
                    "env_seed": 731,
                },
            }
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    return path


def _goals(count: int) -> list[dict[str, str]]:
    return [
        {
            "instruction_text": (
                f"official live instruction {index}, "
                f"and price lower than {index + 10}.00 dollars"
            ),
            "category": f"category-{index % 3}",
        }
        for index in range(count)
    ]


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_live_goal_order_and_exact_instruction_are_preserved(tmp_path: Path) -> None:
    catalog_path = _catalog(tmp_path)
    live_goals = _goals(700)
    received_configs: list[Mapping[str, Any]] = []

    def fake_provider(env_config: Mapping[str, Any]):
        received_configs.append(dict(env_config))
        return live_goals

    output = _MODULE.prepare(catalog_path, goal_provider=fake_provider)

    validation = _records(output / "validation.jsonl")
    train = _records(output / "train.jsonl")
    assert _records(output / "test.jsonl") == []
    assert len(received_configs) == 1
    assert "goal_index" not in received_configs[0]
    assert received_configs[0]["env_seed"] == 731
    assert len(validation) == 128
    assert len(train) == 512
    assert [row["env_config"]["goal_index"] for row in validation] == list(
        range(128)
    )
    assert [row["env_config"]["goal_index"] for row in train] == list(
        range(128, 640)
    )

    first = validation[0]
    instruction = live_goals[0]["instruction_text"]
    assert first["question"] == instruction
    assert first["extra"]["goal"] == instruction
    assert first["extra"]["instruction_text"] == instruction
    assert first["extra"]["goal_index"] == 0
    assert first["extra"]["env_seed"] == 731
    assert first["env_config"]["env_seed"] == 731
    assert first["env_config"]["human_goals"] is True
    assert first["env_config"]["use_small"] is False
    assert first["env_config"]["num_products"] is None
    assert first["env_config"]["goal_split"] == "all"
    assert first["metadata"]["environment"]["env_config"] == first["env_config"]
    assert first["metadata"]["sampling"]["selection_index"] == 0
    assert train[0]["task_id"] == "webshop:00128"


def test_short_inventory_cycles_only_the_remaining_training_goals(
    tmp_path: Path,
) -> None:
    catalog_path = _catalog(tmp_path)
    live_goals = _goals(131)

    output = _MODULE.prepare(catalog_path, goal_provider=lambda _: live_goals)

    validation = _records(output / "validation.jsonl")
    train = _records(output / "train.jsonl")
    validation_indices = [row["env_config"]["goal_index"] for row in validation]
    train_indices = [row["env_config"]["goal_index"] for row in train]
    assert validation_indices == list(range(128))
    assert train_indices[:9] == [128, 129, 130, 128, 129, 130, 128, 129, 130]
    assert set(train_indices) == {128, 129, 130}
    assert set(validation_indices).isdisjoint(train_indices)
    assert train[3]["task_id"] == "webshop:00128:cycle-0001"
    assert train[3]["metadata"]["sampling"]["cycled_training_sample"] is True

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    status = manifest["sources"]["webshop"]
    assert status["unique_train_candidates"] == 3
    assert status["cycled_train_records"] == 509


def test_fake_provider_outputs_are_deterministic(tmp_path: Path) -> None:
    goals = _goals(641)
    first_catalog = _catalog(tmp_path, "first")
    second_catalog = _catalog(tmp_path, "second")

    first = _MODULE.prepare(first_catalog, goal_provider=lambda _: goals)
    second = _MODULE.prepare(second_catalog, goal_provider=lambda _: goals)

    for split in ("validation.jsonl", "train.jsonl", "test.jsonl"):
        assert (first / split).read_bytes() == (second / split).read_bytes()


def test_repository_catalog_expands_the_default_environment_seed(
    monkeypatch,
) -> None:
    monkeypatch.delenv("WEBSHOP_ENV_SEED", raising=False)
    catalog_path = Path(__file__).resolve().parents[2] / "config" / "datasets_webshop.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    source = catalog["sources"]["webshop"]

    env_config = _MODULE._environment_config(
        source,
        base=catalog_path.parent.parent,
    )

    assert env_config["env_seed"] == 1000
    assert env_config["human_goals"] is True
    assert env_config["use_small"] is False
    assert env_config["goal_split"] == "all"
    assert catalog["aligned_dir"] == "data/webshop_v2"


def test_default_provider_seeds_before_the_deployed_reset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    adapter_path = tmp_path / "ragen_adapter.py"
    adapter_path.touch()
    webshop_path = tmp_path / "webshop"
    webshop_path.mkdir()
    search_index_path = tmp_path / "index"
    search_index_path.mkdir()
    env_config = {
        "observation_mode": "text",
        "human_goals": True,
        "use_small": False,
        "num_products": None,
        "goal_split": "all",
        "file_path": str(tmp_path / "items_shuffle.json"),
        "attr_path": str(tmp_path / "items_ins_v2.json"),
        "human_attr_path": str(tmp_path / "items_human_ins.json"),
        "env_seed": 731,
    }
    for field_name in ("file_path", "attr_path", "human_attr_path"):
        Path(env_config[field_name]).touch()

    reset_calls: list[dict[str, Any]] = []
    lifecycle: list[str] = []
    goals = _goals(129)

    class FakeRAGENAdapter:
        def __init__(self) -> None:
            self._env = None

        def reset(self, env_type, config, *, question, extra):
            lifecycle.append("reset")
            reset_calls.append(
                {
                    "env_type": env_type,
                    "config": dict(config),
                    "question": question,
                    "extra": dict(extra),
                    "random_value": random.random(),
                    "webshop_path": os.environ.get("SKILLRL_WEBSHOP_PATH"),
                    "search_index_path": os.environ.get(
                        "WEBSHOP_SEARCH_INDEX_PATH"
                    ),
                }
            )
            server = SimpleNamespace(goals=goals)
            self._env = SimpleNamespace(env=SimpleNamespace(server=server))
            return "official WebShop observation"

    def check_webshop() -> bool:
        lifecycle.append("check_webshop")
        random.seed(999)
        return True

    monkeypatch.setattr(
        _MODULE,
        "_load_python_module",
        lambda _name, _path: SimpleNamespace(
            RAGENAdapter=FakeRAGENAdapter,
            _check_webshop=check_webshop,
        ),
    )
    source = {
        "ragen_adapter_path": str(adapter_path),
        "webshop_path": str(webshop_path),
        "search_index_path": str(search_index_path),
    }

    loaded = _MODULE._default_goal_provider(source, env_config, base=tmp_path)

    assert loaded == tuple(goals)
    assert lifecycle == ["check_webshop", "reset"]
    assert len(reset_calls) == 1
    call = reset_calls[0]
    assert call["env_type"] == "webshop"
    assert call["config"] == env_config
    assert call["question"] == ""
    assert call["extra"] == {}
    assert call["random_value"] == random.Random(731).random()
    assert call["webshop_path"] == str(webshop_path)
    assert call["search_index_path"] == str(search_index_path)
