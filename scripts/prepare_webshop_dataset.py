#!/usr/bin/env python3
"""Prepare official live WebShop goals for the AgentGraph task contract.

The goal inventory comes from the deployed SkillFlow ``RAGENAdapter`` and its
live ``server.goals`` list.  Record construction, held-out-first sampling, and
atomic split publication are delegated to ``prepare_agentgraph_datasets.py``
so preparation and evaluation use one environment and one record protocol.

This command prepares JSONL only.  It never starts a model service, calls an
API, or performs training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Mapping

import yaml


_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_PREPARER_PATH = _SCRIPT_DIR / "prepare_agentgraph_datasets.py"
DEFAULT_RAGEN_ADAPTER_PATH = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py"
)
HELDOUT_COUNT = 128
TRAIN_COUNT = 512


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


GoalProvider = Callable[
    [Mapping[str, Any]],
    Iterable[Mapping[str, Any]],
]


def _expanded_path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("configured path must be non-empty")
    return _SHARED._path(str(value), base=base)


def _source_config(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("catalog.sources must be a mapping")
    source = sources.get("webshop")
    if not isinstance(source, Mapping):
        raise ValueError("catalog.sources.webshop must be a mapping")
    if source.get("enabled", True) is not True:
        raise ValueError("catalog.sources.webshop must be enabled")
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
            "WebShop alignment recipe must remain held-out-first and deterministic: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return recipe


def _environment_config(source: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    configured = source.get("env_config")
    if not isinstance(configured, Mapping):
        raise ValueError("catalog.sources.webshop.env_config must be a mapping")
    config = dict(configured)
    for field_name in ("file_path", "attr_path", "human_attr_path"):
        config[field_name] = str(_expanded_path(config.get(field_name), base=base))

    raw_env_seed = config.get("env_seed")
    if isinstance(raw_env_seed, str):
        raw_env_seed = _SHARED._expand_env_defaults(raw_env_seed)
    try:
        env_seed = int(raw_env_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("WebShop env_config.env_seed must be an integer") from exc
    if env_seed < 0:
        raise ValueError("WebShop env_config.env_seed must be non-negative")
    config["env_seed"] = env_seed

    required_values = {
        "observation_mode": "text",
        "human_goals": True,
        "use_small": False,
        "num_products": None,
        "goal_split": "all",
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required_values.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "WebShop must use the deployed full-catalog live-goal protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if "goal_index" in config:
        raise ValueError("base WebShop env_config must not preselect a goal_index")
    return config


def _default_goal_provider(
    source: Mapping[str, Any],
    env_config: Mapping[str, Any],
    *,
    base: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Read live goals through the same deployed reset used by evaluation."""

    adapter_path = _expanded_path(
        source.get("ragen_adapter_path", DEFAULT_RAGEN_ADAPTER_PATH),
        base=base,
    )
    webshop_path = _expanded_path(source.get("webshop_path"), base=base)
    search_index_path = _expanded_path(source.get("search_index_path"), base=base)
    if not adapter_path.is_file():
        raise FileNotFoundError(f"deployed RAGEN adapter not found: {adapter_path}")
    if not webshop_path.is_dir():
        raise FileNotFoundError(f"WebShop repository not found: {webshop_path}")
    if not search_index_path.is_dir():
        raise FileNotFoundError(f"WebShop search index not found: {search_index_path}")
    for field_name in ("file_path", "attr_path", "human_attr_path"):
        value = Path(str(env_config[field_name]))
        if not value.is_file():
            raise FileNotFoundError(f"WebShop {field_name} not found: {value}")

    environment_overrides = {
        "SKILLRL_WEBSHOP_PATH": str(webshop_path),
        "WEBSHOP_SEARCH_INDEX_PATH": str(search_index_path),
    }
    previous_environment = {
        key: os.environ.get(key) for key in environment_overrides
    }
    previous_random_state = random.getstate()
    try:
        os.environ.update(environment_overrides)
        module = _load_python_module(
            "_flowsteer_deployed_ragen_adapter",
            adapter_path,
        )
        adapter_class = getattr(module, "RAGENAdapter", None)
        if adapter_class is None:
            raise RuntimeError("deployed module does not expose RAGENAdapter")
        check_webshop = getattr(module, "_check_webshop", None)
        if callable(check_webshop) and not bool(check_webshop()):
            raise RuntimeError("deployed WebShop dependencies are unavailable")
        adapter = adapter_class()
        # The dependency import itself mutates Python's global RNG.  Complete
        # it first, then seed immediately before upstream goal generation
        # samples price constraints and applies its fixed goal shuffle.
        random.seed(int(env_config["env_seed"]))
        observation = str(
            adapter.reset(
                "webshop",
                dict(env_config),
                question="",
                extra={},
            )
        )
        live_environment = getattr(adapter, "_env", None)
        if observation.startswith("[ENV_UNAVAILABLE]") or live_environment is None:
            raise RuntimeError(f"deployed WebShop environment unavailable: {observation}")
        upstream_env = getattr(live_environment, "env", None)
        server = getattr(upstream_env, "server", None)
        goals = getattr(server, "goals", None)
        if not isinstance(goals, (list, tuple)) or not goals:
            raise RuntimeError("deployed WebShop server.goals is empty or incompatible")
        # Copy the list before leaving the live provider boundary.  Enumeration
        # below deliberately preserves this exact server index order.
        return tuple(dict(goal) if isinstance(goal, Mapping) else goal for goal in goals)
    finally:
        random.setstate(previous_random_state)
        for key, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _validated_goals(
    raw_goals: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_goals, (str, bytes, Mapping)):
        raise TypeError("WebShop goal provider must return an ordered goal iterable")
    goals = tuple(raw_goals)
    if not goals:
        raise ValueError("WebShop live goal inventory is empty")
    for goal_index, goal in enumerate(goals):
        if not isinstance(goal, Mapping):
            raise TypeError(f"WebShop live goal {goal_index} must be a mapping")
        instruction = goal.get("instruction_text")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"WebShop live goal {goal_index} has no non-empty instruction_text"
            )
    return goals


def _candidate_records(
    goals: Iterable[Mapping[str, Any]],
    source: Mapping[str, Any],
    env_config: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    display_name = str(source.get("display_name", "WebShop"))
    task_type = str(source.get("task_type", "interactive_agent"))
    metric = str(source.get("metric", "success_rate"))
    env_seed = int(env_config["env_seed"])
    for goal_index, goal in enumerate(goals):
        instruction = str(goal["instruction_text"])
        record_env_config = dict(env_config)
        record_env_config["goal_index"] = goal_index
        extra: dict[str, Any] = {
            "goal": instruction,
            "instruction_text": instruction,
            "goal_index": goal_index,
            "env_seed": env_seed,
        }
        category = goal.get("category")
        if isinstance(category, str) and category.strip():
            extra["category"] = category
        record = _SHARED._compat_record(
            dataset_key="webshop",
            source=display_name,
            task_id=f"webshop:{goal_index:05d}",
            question=instruction,
            ground_truth="environment_success",
            split="train",
            task_type=task_type,
            metric=metric,
            env_type="webshop",
            env_config=record_env_config,
            extra=extra,
            evaluator_payload={"target_reward": 1.0},
        )
        # The shared builder trims ordinary benchmark questions.  The live
        # simulator instruction is the WebShop task identity, so retain it
        # byte-for-byte in the canonical question as well as SkillFlow extra.
        record["question"] = instruction
        yield record


def prepare(
    catalog_path: Path,
    *,
    goal_provider: GoalProvider | None = None,
) -> Path:
    """Publish deterministic validation/train records from live goal order."""

    catalog_path = catalog_path.expanduser().resolve()
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("dataset catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported dataset catalog schema")
    recipe = _alignment_recipe(catalog)
    source = _source_config(catalog)
    repo_root = catalog_path.parent.parent
    env_config = _environment_config(source, base=repo_root)
    raw_goals = (
        goal_provider(dict(env_config))
        if goal_provider is not None
        else _default_goal_provider(source, env_config, base=repo_root)
    )
    goals = _validated_goals(raw_goals)
    heldout, train, unique_train_count = _SHARED._uniform_sample(
        _candidate_records(goals, source, env_config),
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
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "catalog": str(catalog_path),
        "training_started": False,
        "deterministic_live_goal_order": True,
        "alignment_recipe": dict(recipe),
        "counts_by_split": counts_by_split,
        "counts_by_source": {
            str(source.get("display_name", "WebShop")): counts_by_split,
        },
        "sources": {
            "webshop": {
                "live_goal_count": len(goals),
                "heldout_split": "validation",
                "heldout_count": len(heldout),
                "train_count": len(train),
                "unique_train_candidates": unique_train_count,
                "cycled_train_records": len(train) - unique_train_count,
                "env_seed": int(env_config["env_seed"]),
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
        default=str(_SCRIPT_DIR.parent / "config" / "datasets_webshop.yaml"),
        help="WebShop dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare(Path(args.catalog))
    print(f"published official WebShop data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
