#!/usr/bin/env python3
"""Prepare official live ALFWorld tasks for the AgentGraph task contract.

Task identity is resolved through the deployed SkillFlow ``RAGENAdapter``.
The candidate inventory is the adapter's ALFWorld ``game_files`` list, grouped
by the six task families enabled by the official ALFWorld configuration and
interleaved in task-type order.  Every selected game is reset once; the exact
canonical instruction after ``Your task is to:`` and the resolved game file
are persisted only when both agree with the requested inventory entry.

This command prepares JSONL only.  It never starts a model service, calls an
LLM API, or performs training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_PREPARER_PATH = _SCRIPT_DIR / "prepare_agentgraph_datasets.py"
DEFAULT_RAGEN_ADAPTER_PATH = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py"
)
HELDOUT_COUNT = 128
TRAIN_COUNT = 512
MAX_EPISODE_STEPS = 50
OFFICIAL_TASK_FAMILIES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def _load_python_module(name: str, path: Path) -> Any:
    """Load one source file while retaining its process-local module caches."""

    source = path.expanduser().resolve()
    loaded = sys.modules.get(name)
    loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_path and Path(str(loaded_path)).expanduser().resolve() == source:
        return loaded
    if not source.is_file():
        raise FileNotFoundError(f"Python module not found: {source}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


_SHARED = _load_python_module(
    "_flowsteer_prepare_agentgraph_datasets",
    _SHARED_PREPARER_PATH,
)
TASK_SCHEMA_VERSION = _SHARED.TASK_SCHEMA_VERSION
CATALOG_SCHEMA_VERSION = _SHARED.CATALOG_SCHEMA_VERSION


TaskProvider = Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]]


def _expanded_path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("configured path must be non-empty")
    return _SHARED._path(str(value), base=base)


def _source_config(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("catalog.sources must be a mapping")
    source = sources.get("alfworld")
    if not isinstance(source, Mapping):
        raise ValueError("catalog.sources.alfworld must be a mapping")
    if source.get("enabled", True) is not True:
        raise ValueError("catalog.sources.alfworld must be enabled")
    return source


def _alignment_recipe(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    recipe = catalog.get("alignment_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("catalog.alignment_recipe must be a mapping")
    expected = {
        "heldout_split": "validation",
        "heldout_count_per_dataset": HELDOUT_COUNT,
        "train_count_per_dataset": TRAIN_COUNT,
        "selection": "sequential",
        "cycle_training_only": True,
    }
    mismatches = {
        key: {"expected": value, "actual": recipe.get(key)}
        for key, value in expected.items()
        if recipe.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "ALFWorld alignment recipe must remain held-out-first and deterministic: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return recipe


def _environment_config(source: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    configured = source.get("env_config")
    if not isinstance(configured, Mapping):
        raise ValueError("catalog.sources.alfworld.env_config must be a mapping")
    config = dict(configured)
    config["config_file"] = str(_expanded_path(config.get("config_file"), base=base))
    expected = {
        "mode": "train",
        "max_steps": MAX_EPISODE_STEPS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "ALFWorld must use the deployed official train inventory and 50-step cap: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if "seed" in config or "game_file" in config:
        raise ValueError("base ALFWorld env_config must not preselect a game")
    return config


def _task_families(source: Mapping[str, Any]) -> tuple[str, ...]:
    raw = source.get("task_families")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("catalog.sources.alfworld.task_families must be a sequence")
    families = tuple(str(item) for item in raw)
    if families != OFFICIAL_TASK_FAMILIES:
        raise ValueError(
            "ALFWorld task_families must follow official task-type order 1..6"
        )
    if source.get("candidate_selection") != "family_round_robin":
        raise ValueError("ALFWorld candidate_selection must be family_round_robin")
    return families


def _game_family(game_file: Any) -> str:
    task_directory = Path(str(game_file)).parent.parent.name
    return task_directory.split("-", 1)[0]


def _round_robin_inventory(
    game_files: Sequence[Any],
    families: Sequence[str],
) -> tuple[tuple[str, int, str], ...]:
    """Interleave each official family while preserving inventory order within it."""

    buckets: dict[str, list[tuple[int, str]]] = {family: [] for family in families}
    unknown: Counter[str] = Counter()
    for index, game_file in enumerate(game_files):
        path = str(Path(str(game_file)).expanduser().resolve())
        family = _game_family(path)
        if family in buckets:
            buckets[family].append((index, path))
        else:
            unknown[family] += 1
    missing = [family for family, entries in buckets.items() if not entries]
    if missing or unknown:
        raise ValueError(
            "ALFWorld inventory does not match the configured official six families: "
            + json.dumps(
                {"missing": missing, "unknown": dict(unknown)}, sort_keys=True
            )
        )

    offsets = {family: 0 for family in families}
    ordered: list[tuple[str, int, str]] = []
    while True:
        added = False
        for family in families:
            offset = offsets[family]
            entries = buckets[family]
            if offset < len(entries):
                game_index, game_file = entries[offset]
                ordered.append((family, game_index, game_file))
                offsets[family] = offset + 1
                added = True
        if not added:
            break
    return tuple(ordered)


def _canonical_instruction(observation: str, task_description: Any) -> str:
    marker = "Your task is to:"
    if marker not in observation:
        raise RuntimeError("ALFWorld reset observation has no canonical task marker")
    instruction = observation.split(marker, 1)[1].strip()
    deployed_instruction = str(task_description).strip()
    if not instruction or instruction != deployed_instruction:
        raise RuntimeError(
            "ALFWorld canonical instruction disagrees with deployed task_description"
        )
    return instruction


def _close_live_game(live_environment: Any) -> None:
    game = getattr(live_environment, "alfred_env", None)
    close = getattr(game, "close", None)
    if callable(close):
        close()


def _default_task_provider(
    source: Mapping[str, Any],
    env_config: Mapping[str, Any],
    *,
    base: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve canonical tasks through the deployed SkillFlow environment."""

    adapter_path = _expanded_path(
        source.get("ragen_adapter_path", DEFAULT_RAGEN_ADAPTER_PATH), base=base
    )
    alfworld_path = _expanded_path(source.get("alfworld_path"), base=base)
    alfworld_data = _expanded_path(source.get("alfworld_data"), base=base)
    config_file = Path(str(env_config["config_file"]))
    for label, path in {
        "deployed RAGEN adapter": adapter_path,
        "ALFWorld config": config_file,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    for label, path in {
        "ALFWorld repository": alfworld_path,
        "ALFWorld data": alfworld_data,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")

    families = _task_families(source)
    environment_overrides = {
        "SKILLRL_ALFWORLD_PATH": str(alfworld_path),
        "ALFWORLD_DATA": str(alfworld_data),
        "ALFWORLD_CONFIG_FILE": str(config_file),
        "SKILLEV_FORMAL_RUNTIME": "1",
    }
    previous_environment = {
        key: os.environ.get(key) for key in environment_overrides
    }
    try:
        os.environ.update(environment_overrides)
        module = _load_python_module(
            "_flowsteer_deployed_ragen_adapter_alfworld", adapter_path
        )
        adapter_class = getattr(module, "RAGENAdapter", None)
        config_class = getattr(module, "AlfredEnvConfig", None)
        environment_class = getattr(module, "ALFWorldEnv", None)
        if adapter_class is None or config_class is None or environment_class is None:
            raise RuntimeError(
                "deployed module does not expose the ALFWorld RAGENAdapter boundary"
            )
        check_alfworld = getattr(module, "_check_alfworld", None)
        if not callable(check_alfworld) or not bool(check_alfworld()):
            raise RuntimeError("deployed ALFWorld dependencies are unavailable")

        inventory_config = config_class(config_file=str(config_file))
        inventory = environment_class(
            config=inventory_config,
            mode=str(env_config["mode"]),
        )
        game_files = list(getattr(inventory, "game_files", ()) or ())
        ordered = _round_robin_inventory(game_files, families)
        required = HELDOUT_COUNT + TRAIN_COUNT
        if len(ordered) < HELDOUT_COUNT + 1:
            raise RuntimeError(
                "official ALFWorld inventory cannot provide held-out and training tasks"
            )

        resolved: list[Mapping[str, Any]] = []
        for family, game_index, expected_game_file in ordered[:required]:
            adapter = adapter_class()
            live_environment = None
            try:
                observation = str(
                    adapter.reset(
                        "alfworld",
                        {
                            "config_file": str(config_file),
                            "mode": str(env_config["mode"]),
                            "seed": game_index,
                        },
                        question="",
                        extra={},
                    )
                )
                live_environment = getattr(adapter, "_env", None)
                if observation.startswith("[ENV_UNAVAILABLE]") or live_environment is None:
                    raise RuntimeError(
                        f"deployed ALFWorld environment unavailable at seed {game_index}"
                    )
                actual_game_file = str(
                    Path(str(getattr(live_environment, "current_game_file", "")))
                    .expanduser()
                    .resolve()
                )
                actual_game_index = int(
                    getattr(live_environment, "current_game_index", -1)
                )
                if (
                    actual_game_index != game_index
                    or actual_game_file != expected_game_file
                ):
                    raise RuntimeError(
                        "ALFWorld reset did not preserve the requested single-game identity: "
                        + json.dumps(
                            {
                                "requested_index": game_index,
                                "actual_index": actual_game_index,
                                "requested_game_file": expected_game_file,
                                "actual_game_file": actual_game_file,
                            },
                            sort_keys=True,
                        )
                    )
                actual_family = _game_family(actual_game_file)
                if actual_family != family:
                    raise RuntimeError("ALFWorld reset changed the task family")
                instruction = _canonical_instruction(
                    observation,
                    getattr(live_environment, "task_description", ""),
                )
                resolved.append(
                    {
                        "task_family": family,
                        "game_index": game_index,
                        "game_file": actual_game_file,
                        "game_seed": game_index,
                        "canonical_instruction": instruction,
                        "task_directory": str(
                            getattr(live_environment, "current_task_dir", "")
                        ),
                    }
                )
                if len(resolved) % 32 == 0 or len(resolved) == min(required, len(ordered)):
                    print(
                        f"verified ALFWorld task identity {len(resolved)}/"
                        f"{min(required, len(ordered))}",
                        flush=True,
                    )
            finally:
                if live_environment is not None:
                    _close_live_game(live_environment)
        return tuple(resolved)
    finally:
        for key, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _validated_tasks(
    raw_tasks: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_tasks, (str, bytes, Mapping)):
        raise TypeError("ALFWorld task provider must return an ordered task iterable")
    tasks = tuple(raw_tasks)
    if not tasks:
        raise ValueError("ALFWorld live task inventory is empty")
    identities: set[str] = set()
    required = {
        "task_family",
        "game_index",
        "game_file",
        "game_seed",
        "canonical_instruction",
    }
    for ordinal, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise TypeError(f"ALFWorld live task {ordinal} must be a mapping")
        missing = sorted(required.difference(task))
        if missing:
            raise ValueError(f"ALFWorld live task {ordinal} is missing {missing}")
        family = str(task["task_family"])
        if family not in OFFICIAL_TASK_FAMILIES:
            raise ValueError(f"ALFWorld live task {ordinal} has invalid family {family}")
        instruction = str(task["canonical_instruction"])
        if not instruction.strip():
            raise ValueError(f"ALFWorld live task {ordinal} has no instruction")
        game_file = str(Path(str(task["game_file"])).expanduser().resolve())
        if game_file in identities:
            raise ValueError(f"duplicate ALFWorld game identity: {game_file}")
        identities.add(game_file)
    return tasks


def _candidate_records(
    tasks: Iterable[Mapping[str, Any]],
    source: Mapping[str, Any],
    env_config: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    display_name = str(source.get("display_name", "ALFWorld"))
    task_type = str(source.get("task_type", "interactive_agent"))
    metric = str(source.get("metric", "success_rate"))
    for task in tasks:
        game_index = int(task["game_index"])
        game_seed = int(task["game_seed"])
        game_file = str(Path(str(task["game_file"])).expanduser().resolve())
        instruction = str(task["canonical_instruction"])
        family = str(task["task_family"])
        record_env_config = dict(env_config)
        record_env_config.update(
            {
                "seed": game_seed,
                "game_file": game_file,
            }
        )
        extra = {
            "task_family": family,
            "canonical_instruction": instruction,
            "game_index": game_index,
            "game_seed": game_seed,
            "game_file": game_file,
            "max_steps": MAX_EPISODE_STEPS,
        }
        if str(task.get("task_directory", "")).strip():
            extra["task_directory"] = str(task["task_directory"])
        record = _SHARED._compat_record(
            dataset_key="alfworld",
            source=display_name,
            task_id=f"alfworld:train:{game_index:05d}",
            question=instruction,
            ground_truth="environment_success",
            split="train",
            task_type=task_type,
            metric=metric,
            env_type="alfworld",
            env_config=record_env_config,
            extra=extra,
            evaluator_payload={"target_won": True},
        )
        # The canonical instruction returned by reset is the task identity.
        # Retain it byte-for-byte rather than the shared builder's trim.
        record["question"] = instruction
        yield record


def prepare(
    catalog_path: Path,
    *,
    task_provider: TaskProvider | None = None,
) -> Path:
    """Publish balanced official-live validation/train ALFWorld records."""

    catalog_path = catalog_path.expanduser().resolve()
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("dataset catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported dataset catalog schema")
    recipe = _alignment_recipe(catalog)
    source = _source_config(catalog)
    _task_families(source)
    repo_root = catalog_path.parent.parent
    env_config = _environment_config(source, base=repo_root)
    raw_tasks = (
        task_provider(dict(env_config))
        if task_provider is not None
        else _default_task_provider(source, env_config, base=repo_root)
    )
    tasks = _validated_tasks(raw_tasks)
    heldout, train, unique_train_count = _SHARED._uniform_sample(
        _candidate_records(tasks, source, env_config),
        heldout_split="validation",
        heldout_count=HELDOUT_COUNT,
        train_count=TRAIN_COUNT,
    )

    output_dir = _expanded_path(catalog.get("aligned_dir"), base=repo_root)
    writers = _SHARED.SplitWriters(output_dir)
    for record in (*heldout, *train):
        writers.write(record)

    counts_by_split = {
        split: sum(
            count
            for (record_split, _source), count in writers.counts.items()
            if record_split == split
        )
        for split in _SHARED.SPLITS
    }
    family_counts = Counter(str(task["task_family"]) for task in tasks)
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "catalog": str(catalog_path),
        "training_started": False,
        "official_live_inventory": True,
        "single_game_identity_verified": True,
        "candidate_selection": "family_round_robin",
        "task_families": list(OFFICIAL_TASK_FAMILIES),
        "max_steps": MAX_EPISODE_STEPS,
        "alignment_recipe": dict(recipe),
        "counts_by_split": counts_by_split,
        "counts_by_source": {
            str(source.get("display_name", "ALFWorld")): counts_by_split,
        },
        "sources": {
            "alfworld": {
                "resolved_task_count": len(tasks),
                "resolved_family_counts": dict(family_counts),
                "heldout_split": "validation",
                "heldout_count": len(heldout),
                "train_count": len(train),
                "unique_train_candidates": unique_train_count,
                "cycled_train_records": len(train) - unique_train_count,
                "mode": str(env_config["mode"]),
            }
        },
        "files": {split: f"{split}.jsonl" for split in _SHARED.SPLITS},
    }
    writers.publish(manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default=str(_SCRIPT_DIR.parent / "config" / "datasets_alfworld.yaml"),
        help="ALFWorld dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare(Path(args.catalog))
    print(f"published official ALFWorld data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
