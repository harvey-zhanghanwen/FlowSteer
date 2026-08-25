#!/usr/bin/env python3
"""Materialize SkillFlow protocol-v10 ALFWorld evaluation TaskRecords.

This is a thin data adapter over the deployed SkillFlow ``RAGENAdapter``.  It
enumerates the adapter's official ``valid_seen`` and ``valid_unseen`` game
inventories and resets each game once to retain the exact public instruction.
No simulator transition, reward, or terminal logic is implemented here.

The command prepares ignored JSONL data only.  It does not start a model,
perform evaluation, or enable training, GRPO, MACE, Bayesian inference, or
Skill evolution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping

import yaml


_SCRIPT_DIR = Path(__file__).resolve().parent
_LEGACY_PREPARER_PATH = _SCRIPT_DIR / "prepare_alfworld_dataset.py"


def _load_legacy_preparer() -> Any:
    """Import the existing ALFWorld identity and TaskRecord helpers."""

    import importlib.util
    import sys

    name = "_flowsteer_prepare_alfworld_dataset_protocol_v10"
    source = _LEGACY_PREPARER_PATH.resolve()
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load existing ALFWorld preparer: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


_LEGACY = _load_legacy_preparer()
_SHARED = _LEGACY._SHARED
TASK_SCHEMA_VERSION = _LEGACY.TASK_SCHEMA_VERSION
CATALOG_SCHEMA_VERSION = _LEGACY.CATALOG_SCHEMA_VERSION
OFFICIAL_TASK_FAMILIES = _LEGACY.OFFICIAL_TASK_FAMILIES

PROTOCOL_VERSION = "skillflow.protocol.v10"
ACTION_POLICY_BUDGET = 20
SIMULATOR_STEP_CAP = 50
OFFICIAL_POPULATIONS = {
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}
PREFLIGHT_POPULATION = {
    "official_split": "preflight_train",
    "mode": "train",
    "output_file": "preflight_train.jsonl",
    "sample_count": 1,
}

TaskProvider = Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]]


def _catalog_mapping(catalog_path: Path) -> Mapping[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("dataset catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported dataset catalog schema")
    if catalog.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
    if catalog.get("action_policy_budget") != ACTION_POLICY_BUDGET:
        raise ValueError(
            f"action_policy_budget must remain {ACTION_POLICY_BUDGET}"
        )
    if catalog.get("simulator_step_cap") != SIMULATOR_STEP_CAP:
        raise ValueError(
            f"simulator_step_cap must remain {SIMULATOR_STEP_CAP}"
        )
    return catalog


def _source_config(catalog: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("catalog.sources must be a mapping")
    source = sources.get("alfworld")
    if not isinstance(source, Mapping) or source.get("enabled", True) is not True:
        raise ValueError("catalog.sources.alfworld must be an enabled mapping")
    return source


def _population_specs(catalog: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = catalog.get("official_populations")
    if not isinstance(raw, Mapping) or tuple(raw) != tuple(OFFICIAL_POPULATIONS):
        raise ValueError(
            "official_populations must preserve valid_seen, valid_unseen order"
        )
    specs: list[dict[str, str]] = []
    for official_split, expected_mode in OFFICIAL_POPULATIONS.items():
        value = raw.get(official_split)
        if not isinstance(value, Mapping):
            raise ValueError(f"official_populations.{official_split} must be a mapping")
        mode = value.get("mode")
        output_file = value.get("output_file")
        if mode != expected_mode:
            raise ValueError(
                f"{official_split} must use deployed mode {expected_mode!r}"
            )
        if output_file != f"{official_split}.jsonl":
            raise ValueError(
                f"{official_split}.output_file must be {official_split}.jsonl"
            )
        specs.append(
            {
                "official_split": official_split,
                "mode": expected_mode,
                "output_file": str(output_file),
            }
        )
    return tuple(specs)


def _smoke_reference(catalog: Mapping[str, Any]) -> dict[str, str]:
    raw = catalog.get("train_heldout_smoke")
    if not isinstance(raw, Mapping):
        raise ValueError("train_heldout_smoke must be a mapping")
    required = {
        "task_file",
        "manifest_file",
        "project_split",
        "scope",
    }
    if set(raw) != required:
        raise ValueError(
            "train_heldout_smoke must contain only task/manifest/split/scope references"
        )
    if raw.get("project_split") != "validation":
        raise ValueError("train-heldout smoke must retain project split validation")
    if raw.get("scope") != "train_inventory_heldout_smoke_only":
        raise ValueError("train-heldout smoke scope must remain explicit")
    return {key: str(raw[key]) for key in sorted(required)}


def _preflight_spec(catalog: Mapping[str, Any]) -> dict[str, Any]:
    raw = catalog.get("evaluator_preflight")
    if not isinstance(raw, Mapping):
        raise ValueError("evaluator_preflight must be a mapping")
    expected = {
        key: value for key, value in PREFLIGHT_POPULATION.items()
        if key != "official_split"
    }
    if dict(raw) != expected:
        raise ValueError("evaluator_preflight must remain one pinned train task")
    return dict(PREFLIGHT_POPULATION)


def _resolved_source_paths(
    source: Mapping[str, Any], *, base: Path
) -> dict[str, Path]:
    paths = {
        "ragen_adapter_path": _LEGACY._expanded_path(
            source.get("ragen_adapter_path", _LEGACY.DEFAULT_RAGEN_ADAPTER_PATH),
            base=base,
        ),
        "alfworld_path": _LEGACY._expanded_path(
            source.get("alfworld_path"), base=base
        ),
        "alfworld_data": _LEGACY._expanded_path(
            source.get("alfworld_data"), base=base
        ),
        "config_file": _LEGACY._expanded_path(
            source.get("config_file"), base=base
        ),
    }
    for label in ("ragen_adapter_path", "config_file"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"configured {label} not found: {paths[label]}")
    for label in ("alfworld_path", "alfworld_data"):
        if not paths[label].is_dir():
            raise FileNotFoundError(f"configured {label} not found: {paths[label]}")
    return paths


def _default_task_provider(
    request: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    base: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve one complete official population through SkillFlow RAGENAdapter."""

    official_split = str(request["official_split"])
    mode = str(request["mode"])
    expected_mode = (
        PREFLIGHT_POPULATION["mode"]
        if official_split == PREFLIGHT_POPULATION["official_split"]
        else OFFICIAL_POPULATIONS.get(official_split)
    )
    if expected_mode != mode:
        raise ValueError("ALFWorld official split/mode pair is not protocol-v10")
    paths = _resolved_source_paths(source, base=base)
    environment_overrides = {
        "SKILLRL_ALFWORLD_PATH": str(paths["alfworld_path"]),
        "ALFWORLD_DATA": str(paths["alfworld_data"]),
        "ALFWORLD_CONFIG_FILE": str(paths["config_file"]),
        "SKILLEV_FORMAL_RUNTIME": "1",
    }
    previous_environment = {
        key: os.environ.get(key) for key in environment_overrides
    }
    try:
        os.environ.update(environment_overrides)
        module = _LEGACY._load_python_module(
            "_flowsteer_deployed_ragen_adapter_alfworld_protocol_v10",
            paths["ragen_adapter_path"],
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

        environment_config = config_class(config_file=str(paths["config_file"]))
        inventory = environment_class(config=environment_config, mode=mode)
        game_files = tuple(
            str(Path(str(value)).expanduser().resolve())
            for value in (getattr(inventory, "game_files", ()) or ())
        )
        if not game_files:
            raise RuntimeError(f"official ALFWorld {official_split} inventory is empty")

        resolved: list[Mapping[str, Any]] = []
        sample_count = request.get("sample_count", len(game_files))
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or not 1 <= sample_count <= len(game_files)
        ):
            raise ValueError("ALFWorld population sample_count is invalid")
        selected_game_files = game_files[:sample_count]
        for game_index, expected_game_file in enumerate(selected_game_files):
            adapter = adapter_class()
            live_environment = None
            try:
                observation = str(
                    adapter.reset(
                        "alfworld",
                        {
                            "config_file": str(paths["config_file"]),
                            "mode": mode,
                            "seed": game_index,
                        },
                        question="",
                        extra={},
                    )
                )
                live_environment = getattr(adapter, "_env", None)
                if observation.startswith("[ENV_UNAVAILABLE]") or live_environment is None:
                    raise RuntimeError(
                        f"deployed ALFWorld environment unavailable at {official_split} "
                        f"index {game_index}"
                    )
                actual_game_index = int(
                    getattr(live_environment, "current_game_index", -1)
                )
                actual_game_file = str(
                    Path(str(getattr(live_environment, "current_game_file", "")))
                    .expanduser()
                    .resolve()
                )
                if (
                    actual_game_index != game_index
                    or actual_game_file != expected_game_file
                ):
                    raise RuntimeError(
                        "ALFWorld reset did not preserve the requested official game "
                        "identity"
                    )
                canonical_instruction = _LEGACY._canonical_instruction(
                    observation,
                    getattr(live_environment, "task_description", ""),
                )
                resolved.append(
                    {
                        "official_split": official_split,
                        "mode": mode,
                        "task_family": _LEGACY._game_family(actual_game_file),
                        "game_index": game_index,
                        "game_file": actual_game_file,
                        "canonical_instruction": canonical_instruction,
                        "task_directory": str(
                            getattr(live_environment, "current_task_dir", "")
                        ),
                    }
                )
                if len(resolved) % 32 == 0 or len(resolved) == len(selected_game_files):
                    print(
                        f"verified ALFWorld {official_split} identity "
                        f"{len(resolved)}/{len(selected_game_files)}",
                        flush=True,
                    )
            finally:
                if live_environment is not None:
                    _LEGACY._close_live_game(live_environment)
        return tuple(resolved)
    finally:
        for key, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _validated_tasks(
    raw_tasks: Iterable[Mapping[str, Any]],
    *,
    official_split: str,
    mode: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_tasks, (str, bytes, Mapping)):
        raise TypeError("ALFWorld task provider must return an ordered iterable")
    tasks = tuple(raw_tasks)
    if not tasks:
        raise ValueError(f"ALFWorld {official_split} inventory is empty")
    required = {
        "task_family",
        "game_index",
        "game_file",
        "canonical_instruction",
    }
    identities: set[str] = set()
    indices: set[int] = set()
    for ordinal, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise TypeError(f"ALFWorld task {ordinal} must be a mapping")
        missing = sorted(required.difference(task))
        if missing:
            raise ValueError(f"ALFWorld task {ordinal} is missing {missing}")
        if "official_split" in task and task["official_split"] != official_split:
            raise ValueError(f"ALFWorld task {ordinal} changed official_split")
        if "mode" in task and task["mode"] != mode:
            raise ValueError(f"ALFWorld task {ordinal} changed mode")
        family = str(task["task_family"])
        if family not in OFFICIAL_TASK_FAMILIES:
            raise ValueError(f"ALFWorld task {ordinal} has invalid family {family}")
        instruction = str(task["canonical_instruction"])
        if not instruction.strip():
            raise ValueError(f"ALFWorld task {ordinal} has no canonical instruction")
        game_index = int(task["game_index"])
        game_file = str(Path(str(task["game_file"])).expanduser().resolve())
        if game_index in indices or game_file in identities:
            raise ValueError(f"duplicate ALFWorld identity in {official_split}")
        indices.add(game_index)
        identities.add(game_file)
    return tasks


def _record(
    task: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    config_file: Path,
    official_split: str,
    mode: str,
    project_split: str = "test",
) -> dict[str, Any]:
    game_index = int(task["game_index"])
    game_file = str(Path(str(task["game_file"])).expanduser().resolve())
    canonical_instruction = str(task["canonical_instruction"])
    env_config = {
        "config_file": str(config_file),
        "mode": mode,
        "seed": game_index,
        "game_file": game_file,
        # The project evaluator consumes ``max_steps`` as the policy action
        # budget.  SkillFlow's RAGENAdapter ignores this field and registers
        # the TextWorld simulator with its own fixed 50-step cap.
        "max_steps": ACTION_POLICY_BUDGET,
        "simulator_step_cap": SIMULATOR_STEP_CAP,
    }
    extra = {
        "official_split": official_split,
        "mode": mode,
        "task_family": str(task["task_family"]),
        "game_index": game_index,
        "game_seed": game_index,
        "game_file": game_file,
        "canonical_instruction": canonical_instruction,
        "action_policy_budget": ACTION_POLICY_BUDGET,
        "simulator_step_cap": SIMULATOR_STEP_CAP,
    }
    if str(task.get("task_directory", "")).strip():
        extra["task_directory"] = str(task["task_directory"])
    record = _SHARED._compat_record(
        dataset_key="alfworld",
        source=str(source.get("display_name", "ALFWorld")),
        task_id=f"alfworld:{official_split}:{game_index:05d}",
        question=canonical_instruction,
        ground_truth="environment_success",
        split=project_split,
        task_type=str(source.get("task_type", "interactive_agent")),
        metric=str(source.get("metric", "success_rate")),
        env_type="alfworld",
        env_config=env_config,
        extra=extra,
        evaluator_payload={"target_won": True},
        preserve_question_text=True,
    )
    record["metadata"]["protocol"] = {
        "protocol_version": PROTOCOL_VERSION,
        "official_split": official_split,
        "mode": mode,
        "action_policy_budget": ACTION_POLICY_BUDGET,
        "simulator_step_cap": SIMULATOR_STEP_CAP,
        "terminal_evaluator": "official_environment_success",
    }
    return record


def _publish(
    output_dir: Path,
    *,
    records_by_split: Mapping[str, tuple[Mapping[str, Any], ...]],
    manifest: Mapping[str, Any],
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"protocol output already exists and is non-empty: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix="alfworld-protocol-v10-", dir=output_dir.parent)
    )
    try:
        for official_split, records in records_by_split.items():
            destination = temp_dir / f"{official_split}.jsonl"
            with destination.open("x", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        with (temp_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(temp_dir.iterdir()):
            source_file.replace(output_dir / source_file.name)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def prepare(
    catalog_path: Path,
    *,
    task_provider: TaskProvider | None = None,
) -> Path:
    """Publish both full official protocol-v10 ALFWorld populations."""

    catalog_path = catalog_path.expanduser().resolve()
    catalog = _catalog_mapping(catalog_path)
    source = _source_config(catalog)
    population_specs = _population_specs(catalog)
    preflight_spec = _preflight_spec(catalog)
    smoke_reference = _smoke_reference(catalog)
    repo_root = catalog_path.parent.parent
    config_file = _LEGACY._expanded_path(source.get("config_file"), base=repo_root)

    records_by_split: dict[str, tuple[Mapping[str, Any], ...]] = {}
    family_counts: dict[str, Mapping[str, int]] = {}
    for spec in population_specs:
        request = {
            **spec,
            "config_file": str(config_file),
            "action_policy_budget": ACTION_POLICY_BUDGET,
            "simulator_step_cap": SIMULATOR_STEP_CAP,
        }
        raw_tasks = (
            task_provider(request)
            if task_provider is not None
            else _default_task_provider(request, source=source, base=repo_root)
        )
        tasks = _validated_tasks(
            raw_tasks,
            official_split=spec["official_split"],
            mode=spec["mode"],
        )
        records = tuple(
            _record(
                task,
                source=source,
                config_file=config_file,
                official_split=spec["official_split"],
                mode=spec["mode"],
            )
            for task in tasks
        )
        records_by_split[spec["official_split"]] = records
        family_counts[spec["official_split"]] = dict(
            Counter(str(task["task_family"]) for task in tasks)
        )

    preflight_request = {
        **preflight_spec,
        "config_file": str(config_file),
        "action_policy_budget": ACTION_POLICY_BUDGET,
        "simulator_step_cap": SIMULATOR_STEP_CAP,
    }
    raw_preflight_tasks = (
        task_provider(preflight_request)
        if task_provider is not None
        else _default_task_provider(preflight_request, source=source, base=repo_root)
    )
    preflight_tasks = _validated_tasks(
        raw_preflight_tasks,
        official_split=str(preflight_spec["official_split"]),
        mode=str(preflight_spec["mode"]),
    )
    if len(preflight_tasks) != 1:
        raise ValueError("evaluator preflight population must contain one task")
    records_by_split["preflight_train"] = tuple(
        _record(
            task,
            source=source,
            config_file=config_file,
            official_split="preflight_train",
            mode="train",
            project_split="train",
        )
        for task in preflight_tasks
    )
    family_counts["preflight_train"] = dict(
        Counter(str(task["task_family"]) for task in preflight_tasks)
    )

    output_dir = _LEGACY._expanded_path(catalog.get("aligned_dir"), base=repo_root)
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "catalog": str(catalog_path),
        "training_started": False,
        "official_live_inventory": True,
        "single_game_identity_verified": task_provider is None,
        "task_project_split": "test",
        "action_policy_budget": ACTION_POLICY_BUDGET,
        "simulator_step_cap": SIMULATOR_STEP_CAP,
        "terminal_evaluator": "official_environment_success",
        "counts_by_official_split": {
            key: len(records_by_split[key]) for key in OFFICIAL_POPULATIONS
        },
        "family_counts_by_official_split": {
            key: family_counts[key] for key in OFFICIAL_POPULATIONS
        },
        "official_populations": {
            spec["official_split"]: {
                "mode": spec["mode"],
                "file": spec["output_file"],
            }
            for spec in population_specs
        },
        "evaluator_preflight": {
            "mode": "train",
            "file": "preflight_train.jsonl",
            "sample_count": 1,
        },
        "train_heldout_smoke_reference": smoke_reference,
        "files": {
            **{
                spec["official_split"]: spec["output_file"]
                for spec in population_specs
            },
            "preflight_train": "preflight_train.jsonl",
            "manifest": "manifest.json",
        },
    }
    _publish(output_dir, records_by_split=records_by_split, manifest=manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default=str(
            _SCRIPT_DIR.parent / "config" / "datasets_alfworld_protocol_v10.yaml"
        ),
        help="SkillFlow protocol-v10 ALFWorld catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare(Path(args.catalog))
    print(f"published ALFWorld protocol-v10 data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
