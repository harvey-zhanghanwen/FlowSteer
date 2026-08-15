#!/usr/bin/env python3
"""Align the seven project benchmarks into FlowSteer/SkillFlow JSONL.

The field layout and QA/environment prompt construction are ported from
SkillFlow ``data/prepare_v3.py``.  The required adaptations are:

* emit the design-note ``TaskRecord`` fields and FlowSteer aliases;
* use real ALFWorld tasks instead of the old synthetic templates;
* replace MedQA with the requested eval-only HealthBench Professional;
* keep AIME 2026 and SWE-bench Verified as untouched test sets; and
* preserve evaluator-only payloads without putting them in ``question``.

This command only prepares data.  It never starts training, an LLM service, or
an interactive benchmark environment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import yaml


TASK_SCHEMA_VERSION = "flowsteer.agentgraph.task.v1"
CATALOG_SCHEMA_VERSION = "flowsteer.agentgraph.datasets.v1"
SPLITS = ("train", "validation", "test")


def _expand_env_defaults(value: str) -> str:
    """Expand ``${NAME:-default}`` plus ordinary environment variables."""

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(2))

    return os.path.expanduser(os.path.expandvars(pattern.sub(replace, value)))


def _path(value: str, *, base: Path | None = None) -> Path:
    resolved = Path(_expand_env_defaults(str(value)))
    if not resolved.is_absolute() and base is not None:
        resolved = base / resolved
    return resolved.resolve()


def _plain(value: Any) -> Any:
    """Convert pandas/numpy containers into JSON-native values."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_list(value: Any) -> list[Any]:
    native = _plain(value)
    if native is None:
        return []
    if isinstance(native, list):
        return native
    return [native]


def _iter_parquet_rows(pattern: Path) -> Iterator[dict[str, Any]]:
    files = sorted(Path(item) for item in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no parquet files matched {pattern}")
    for source in files:
        frame = pd.read_parquet(source)
        for row in frame.to_dict(orient="records"):
            yield _plain(row)


def _compat_record(
    *,
    dataset_key: str,
    source: str,
    task_id: str,
    question: str,
    ground_truth: Any,
    split: str,
    task_type: str,
    metric: str,
    context: Sequence[Any] = (),
    extra: Mapping[str, Any] | None = None,
    evaluator_payload: Mapping[str, Any] | None = None,
    env_type: str | None = None,
    env_config: Mapping[str, Any] | None = None,
    code_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the shared design-note, FlowSteer, and SkillFlow record."""

    if split not in SPLITS:
        raise ValueError(f"invalid project split: {split}")
    question = str(question).strip()
    if not question:
        raise ValueError(f"empty question for {task_id}")
    metadata: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "source": source,
        "task_type": task_type,
        "metric": metric,
    }
    if evaluator_payload:
        metadata["evaluator_payload"] = _plain(evaluator_payload)
    if env_type:
        metadata["environment"] = {
            "env_type": env_type,
            "env_config": _plain(dict(env_config or {})),
        }

    skillflow_extra = {"source": source, "metric": metric}
    skillflow_extra.update(_plain(dict(extra or {})))
    record: dict[str, Any] = {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": str(task_id),
        "question": question,
        "ground_truth": _plain(ground_truth),
        "split": split,
        "metadata": metadata,
        # FlowSteer compatibility.
        "source": source,
        "dataset": dataset_key,
        # SkillFlow compatibility.
        "answer": _plain(ground_truth),
        "task_type": task_type,
        "context": _plain(list(context)),
        "extra": skillflow_extra,
    }
    if env_type:
        record["env_type"] = env_type
        record["env_config"] = _plain(dict(env_config or {}))
    if code_files is not None:
        record["code_files"] = _plain(dict(code_files))
    return record


class SplitWriters:
    """Write all splits once and publish them only after complete conversion."""

    def __init__(self, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"aligned output already exists and is non-empty: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix="agentgraph-data-", dir=str(output_dir.parent))
        )
        self.handles = {
            split: (self.temp_dir / f"{split}.jsonl").open("x", encoding="utf-8")
            for split in SPLITS
        }
        self.counts: Counter[tuple[str, str]] = Counter()
        self.ids: dict[str, set[str]] = defaultdict(set)

    def write(self, record: Mapping[str, Any]) -> None:
        split = str(record["split"])
        task_id = str(record["task_id"])
        if task_id in self.ids[split]:
            raise ValueError(f"duplicate task_id within {split}: {task_id}")
        self.ids[split].add(task_id)
        self.handles[split].write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.counts[(split, str(record["source"]))] += 1

    def publish(self, manifest: Mapping[str, Any]) -> None:
        for handle in self.handles.values():
            handle.close()
        with (self.temp_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (*SPLITS, "manifest"):
            source = (
                self.temp_dir / f"{name}.jsonl"
                if name in SPLITS
                else self.temp_dir / "manifest.json"
            )
            source.replace(self.output_dir / source.name)
        self.temp_dir.rmdir()


def _hotpot_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    base = _path(str(config["path"]))
    order = config.get("candidate_sequence", config["files"].keys())
    for split in order:
        pattern = config["files"][split]
        for row in _iter_parquet_rows(base / str(pattern)):
            context = row.get("context") or {}
            titles = (
                _as_list(context.get("title")) if isinstance(context, Mapping) else []
            )
            sentence_groups = (
                _as_list(context.get("sentences"))
                if isinstance(context, Mapping)
                else []
            )
            passages = []
            for index, title in enumerate(titles):
                sentences = (
                    _as_list(sentence_groups[index])
                    if index < len(sentence_groups)
                    else []
                )
                passages.append(
                    f"[{title}] " + " ".join(str(item) for item in sentences)
                )

            prompt_parts = ["Based on the following passages, answer the question."]
            prompt_parts.extend(f"[{passage[:300]}]" for passage in passages[:10])
            prompt_parts.append(f"Question: {row['question']}")
            yield _compat_record(
                dataset_key="hotpotqa",
                source=str(config["display_name"]),
                task_id=f"hotpotqa:{row['id']}",
                question="\n\n".join(prompt_parts),
                ground_truth=str(row["answer"]),
                split=str(split),
                task_type=str(config["task_type"]),
                metric=str(config["metric"]),
                context=passages[:10],
                extra={"type": row.get("type", ""), "level": row.get("level", "")},
                evaluator_payload={
                    "supporting_facts": row.get("supporting_facts", {}),
                },
            )


def _trivia_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    base = _path(str(config["path"]))
    order = config.get("candidate_sequence", config["files"].keys())
    for split in order:
        pattern = config["files"][split]
        for row in _iter_parquet_rows(base / str(pattern)):
            answer = row.get("answer") or {}
            value = (
                str(answer.get("value", ""))
                if isinstance(answer, Mapping)
                else str(answer)
            )
            aliases = (
                _as_list(answer.get("aliases")) if isinstance(answer, Mapping) else []
            )
            accepted = []
            for item in [value, *aliases]:
                text = str(item).strip()
                if text and text not in accepted:
                    accepted.append(text)
            if not accepted:
                continue
            yield _compat_record(
                dataset_key="triviaqa",
                source=str(config["display_name"]),
                task_id=f"triviaqa:{row['question_id']}",
                question=str(row["question"]),
                ground_truth=" | ".join(accepted),
                split=str(split),
                task_type=str(config["task_type"]),
                metric=str(config["metric"]),
                extra={"question_source": row.get("question_source", "")},
                evaluator_payload={"accepted_answers": accepted},
            )


def _aime_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for row in _iter_parquet_rows(_path(str(config["path"]))):
        index = int(row["problem_idx"])
        answer = int(row["answer"])
        yield _compat_record(
            dataset_key="aime_2026",
            source=str(config["display_name"]),
            task_id=f"aime-2026:{index:02d}",
            question=str(row["problem"]),
            ground_truth=str(answer),
            split="test",
            task_type=str(config["task_type"]),
            metric=str(config["metric"]),
            extra={
                "problem_index": index,
                "answer_format": "integer-000-to-999",
                "benchmark_slice": "official_aime_2026",
                "native_split": "train",
            },
            evaluator_payload={"accepted_answers": [str(answer)]},
        )

    historical_path = _path(str(config["historical_path"]))
    with historical_path.open("r", encoding="utf-8") as handle:
        historical = json.load(handle)
    ordinal = 0
    for item in historical:
        extra = item.get("extra", {}) or {}
        if str(extra.get("source", "")).strip().lower() != "aime":
            continue
        ordinal += 1
        answer = str(item.get("answer", ""))
        yield _compat_record(
            dataset_key="aime_2026",
            source=str(config["display_name"]),
            task_id=f"aime-historical:{ordinal:04d}",
            question=str(item.get("question", "")),
            ground_truth=answer,
            split="train",
            task_type=str(config["task_type"]),
            metric=str(config["metric"]),
            context=_as_list(item.get("context", [])),
            extra={
                "benchmark_slice": "historical_aime_1983_2025",
                "native_source": "SkillFlow AIME historical pool",
            },
            evaluator_payload={"accepted_answers": [answer]},
        )


def _conversation_prompt(conversation: Any) -> str:
    if isinstance(conversation, Mapping):
        messages = _as_list(conversation.get("messages"))
    else:
        messages = _as_list(conversation)
    lines = ["Conversation:"]
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"[{role}] {content}")
    lines.append("[assistant]")
    return "\n\n".join(lines)


def _healthbench_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    source = _path(str(config["path"]))
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("id") or f"row-{line_number}")
            yield _compat_record(
                dataset_key="healthbench_professional",
                source=str(config["display_name"]),
                task_id=f"healthbench-professional:{task_id}",
                question=_conversation_prompt(row.get("conversation", {})),
                ground_truth=str(row.get("physician_response", "")),
                split="test",
                task_type=str(config["task_type"]),
                metric=str(config["metric"]),
                extra={
                    "use_case": row.get("use_case", ""),
                    "type": row.get("type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "specialty": row.get("specialty", ""),
                },
                evaluator_payload={
                    "rubric_items": row.get("rubric_items", []),
                    "physician_response": row.get("physician_response", ""),
                },
            )


def _webshop_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    root = _path(str(config["path"]))
    goals_path = root / str(config["goals_file"])
    with goals_path.open("r", encoding="utf-8") as handle:
        goals = json.load(handle)
    ranges = config["split_ranges"]
    file_path = str((root / str(config["product_file"])).resolve())
    attr_path = str((root / str(config["attribute_file"])).resolve())
    for split in ("test", "validation", "train"):
        start, stop = ranges[split]
        resolved_stop = len(goals) if stop is None else min(int(stop), len(goals))
        for index in range(int(start), resolved_stop):
            goal = goals[index]
            goal_text = (
                str(goal.get("goal", goal.get("instruction", goal)))
                if isinstance(goal, Mapping)
                else str(goal)
            )
            env_config = {
                "observation_mode": "text",
                "human_goals": True,
                "use_small": False,
                "goal_index": index,
                "goal_split": split,
                "file_path": file_path,
                "attr_path": attr_path,
            }
            yield _compat_record(
                dataset_key="webshop",
                source=str(config["display_name"]),
                task_id=f"webshop:{index:05d}",
                question=(
                    "You are shopping online. Find and buy the following item:\n\n"
                    f"{goal_text}\n\n"
                    "Available actions: search[query], click[element], buy"
                ),
                ground_truth="environment_success",
                split=split,
                task_type=str(config["task_type"]),
                metric=str(config["metric"]),
                env_type="webshop",
                env_config=env_config,
                extra={"goal_index": index, "goal": goal_text[:200]},
                evaluator_payload={"target_reward": 1.0},
            )


def _alfworld_task_text(row: Mapping[str, Any], fallback: str) -> str:
    annotations = row.get("turk_annotations") or {}
    anns = _as_list(annotations.get("anns")) if isinstance(annotations, Mapping) else []
    for annotation in anns:
        if (
            isinstance(annotation, Mapping)
            and str(annotation.get("task_desc", "")).strip()
        ):
            return str(annotation["task_desc"]).strip()
    return fallback.replace("_", " ").replace("-", " ")


def _alfworld_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    root = _path(str(config["path"]))
    data_root = root / str(config["data_dir"])
    config_file = str((root / str(config["config_file"])).resolve())
    modes = {
        "train": "train",
        "validation": "eval_in_distribution",
        "test": "eval_out_of_distribution",
    }
    for split in SPLITS:
        split_dir = data_root / str(config["split_dirs"][split])
        playable: list[tuple[Path, Mapping[str, Any]]] = []
        for trajectory_path in sorted(split_dir.rglob("traj_data.json")):
            game_path = trajectory_path.with_name("game.tw-pddl")
            if not game_path.exists():
                continue
            with game_path.open("r", encoding="utf-8") as handle:
                game = json.load(handle)
            if not bool(game.get("solvable", False)):
                continue
            with trajectory_path.open("r", encoding="utf-8") as handle:
                playable.append((trajectory_path, json.load(handle)))

        for seed, (trajectory_path, row) in enumerate(playable):
            task_id = str(row.get("task_id") or trajectory_path.parent.name)
            task_text = _alfworld_task_text(row, trajectory_path.parent.parent.name)
            relative_game = str(
                trajectory_path.with_name("game.tw-pddl").relative_to(root)
            )
            env_config = {
                "config_file": config_file,
                "mode": modes[split],
                "seed": seed,
                "game_file": str(trajectory_path.with_name("game.tw-pddl").resolve()),
            }
            yield _compat_record(
                dataset_key="alfworld",
                source=str(config["display_name"]),
                task_id=f"alfworld:{split}:{task_id}",
                question=(
                    "You are in a household environment. Complete this task:\n\n"
                    f"{task_text}\n\n"
                    "Use only the admissible actions returned by the environment."
                ),
                ground_truth="environment_success",
                split=split,
                task_type=str(config["task_type"]),
                metric=str(config["metric"]),
                env_type="alfworld",
                env_config=env_config,
                extra={
                    "task": row.get("task_type", ""),
                    "game_file": relative_game,
                },
                evaluator_payload={"target_won": True},
            )


def _swe_record(
    row: Mapping[str, Any], config: Mapping[str, Any], split: str
) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    return _compat_record(
        dataset_key="swe_bench",
        source=str(config["display_name"]),
        task_id=f"swe-bench:{instance_id}",
        question=f"Fix the following software issue:\n\n{row['problem_statement']}",
        ground_truth=str(row.get("patch", "")),
        split=split,
        task_type=str(config["task_type"]),
        metric=str(config["metric"]),
        code_files={},
        extra={
            "repo": row.get("repo", ""),
            "instance_id": instance_id,
            "base_commit": row.get("base_commit", ""),
        },
        evaluator_payload={
            "instance_id": instance_id,
            "repo": row.get("repo", ""),
            "base_commit": row.get("base_commit", ""),
            "test_patch": row.get("test_patch", ""),
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
            "version": row.get("version", ""),
            "environment_setup_commit": row.get("environment_setup_commit", ""),
        },
    )


def _swe_records(config: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    base = _path(str(config["path"]))
    if config.get("candidate_sequence") != ["swe_bench_verified"]:
        for split, pattern in config["files"].items():
            for row in _iter_parquet_rows(base / str(pattern)):
                yield _swe_record(row, config, str(split))

    verified_path = _path(str(config["verified_path"]))
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to load SWE-bench Verified") from exc
    verified = load_from_disk(str(verified_path))
    for row in verified:
        yield _swe_record(_plain(row), config, "test")


def _retag_record(
    record: Mapping[str, Any],
    *,
    split: str,
    selection_index: int,
    cycle_index: int | None = None,
) -> dict[str, Any]:
    """Assign the project split without changing the underlying task payload."""

    result = copy.deepcopy(dict(record))
    base_task_id = str(result["task_id"])
    native_split = str(result.get("split", ""))
    result["split"] = split
    sampling = {
        "selection": "sequential",
        "selection_index": selection_index,
        "base_task_id": base_task_id,
        "cycled_training_sample": cycle_index is not None,
    }
    if cycle_index is not None:
        result["task_id"] = f"{base_task_id}:cycle-{cycle_index:04d}"
        sampling["cycle_index"] = cycle_index
    metadata = dict(result.get("metadata", {}))
    metadata.setdefault("native_split", native_split)
    metadata["sampling"] = sampling
    result["metadata"] = metadata
    extra = dict(result.get("extra", {}))
    extra["sampling"] = sampling
    result["extra"] = extra
    return result


def _uniform_sample(
    records: Iterable[Mapping[str, Any]],
    *,
    heldout_split: str,
    heldout_count: int,
    train_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Apply the user-specified held-out-first sequential sampling rule."""

    iterator = iter(records)
    heldout_base: list[Mapping[str, Any]] = []
    for _ in range(heldout_count):
        try:
            heldout_base.append(next(iterator))
        except StopIteration as exc:
            raise ValueError(
                f"candidate sequence has only {len(heldout_base)} items; "
                f"cannot form held-out set of {heldout_count} without reusing it"
            ) from exc

    heldout_ids = {str(item["task_id"]) for item in heldout_base}
    if len(heldout_ids) != len(heldout_base):
        raise ValueError("held-out candidate sequence contains duplicate task IDs")

    train_base: list[Mapping[str, Any]] = []
    while len(train_base) < train_count:
        try:
            candidate = next(iterator)
        except StopIteration:
            break
        if str(candidate["task_id"]) in heldout_ids:
            continue
        train_base.append(candidate)
    if not train_base:
        raise ValueError("no training candidate remains after held-out selection")

    heldout = [
        _retag_record(item, split=heldout_split, selection_index=index)
        for index, item in enumerate(heldout_base)
    ]
    train = [
        _retag_record(item, split="train", selection_index=index)
        for index, item in enumerate(train_base)
    ]
    unique_train_count = len(train)
    cycle_index = 1
    while len(train) < train_count:
        source = train_base[(len(train) - unique_train_count) % unique_train_count]
        train.append(
            _retag_record(
                source,
                split="train",
                selection_index=len(train),
                cycle_index=cycle_index,
            )
        )
        cycle_index += 1
    return heldout, train, unique_train_count


CONVERTERS = {
    "hotpotqa": _hotpot_records,
    "triviaqa": _trivia_records,
    "aime_2026": _aime_records,
    "healthbench_professional": _healthbench_records,
    "webshop": _webshop_records,
    "alfworld": _alfworld_records,
    "swe_bench": _swe_records,
}


def prepare(catalog_path: Path, *, selected: set[str] | None = None) -> Path:
    repo_root = catalog_path.resolve().parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported dataset catalog schema")

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    recipe = catalog.get("alignment_recipe", {})
    heldout_split = str(recipe.get("heldout_split", "validation"))
    heldout_count = int(recipe.get("heldout_count_per_dataset", 128))
    train_count = int(recipe.get("train_count_per_dataset", 512))
    if heldout_split not in {"validation", "test"}:
        raise ValueError("alignment_recipe.heldout_split must be validation or test")
    if heldout_count <= 0 or train_count <= 0:
        raise ValueError("uniform recipe counts must be positive")
    if recipe.get("selection") != "sequential":
        raise ValueError("only sequential dataset selection is supported")
    if recipe.get("cycle_training_only") is not True:
        raise ValueError("cycle_training_only must remain true")

    source_status: Dict[str, Any] = {}
    for dataset_key, config in catalog["sources"].items():
        if not bool(config.get("enabled", True)):
            continue
        if selected is not None and dataset_key not in selected:
            continue
        converter = CONVERTERS.get(dataset_key)
        if converter is None:
            raise ValueError(f"no converter registered for {dataset_key}")
        heldout, train, unique_train_count = _uniform_sample(
            converter(config),
            heldout_split=heldout_split,
            heldout_count=heldout_count,
            train_count=train_count,
        )
        for record in [*heldout, *train]:
            writers.write(record)
        source_status[dataset_key] = {
            "heldout_split": heldout_split,
            "heldout_count": len(heldout),
            "train_count": len(train),
            "unique_train_candidates": unique_train_count,
            "cycled_train_records": len(train) - unique_train_count,
        }
        print(f"aligned {dataset_key}: {source_status[dataset_key]}", flush=True)

    counts_by_split = {
        split: sum(
            count
            for (record_split, _source), count in writers.counts.items()
            if record_split == split
        )
        for split in SPLITS
    }
    counts_by_source = {
        source: {split: writers.counts.get((split, source), 0) for split in SPLITS}
        for source in sorted({source for _, source in writers.counts})
    }
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path.resolve()),
        "training_started": False,
        "alignment_recipe": {
            "selection": "sequential",
            "heldout_split": heldout_split,
            "heldout_count_per_dataset": heldout_count,
            "train_count_per_dataset": train_count,
            "cycle_training_only": True,
        },
        "counts_by_split": counts_by_split,
        "counts_by_source": counts_by_source,
        "sources": source_status,
        "files": {split: f"{split}.jsonl" for split in SPLITS},
    }
    writers.publish(manifest)
    print(f"published aligned data to {output_dir}", flush=True)
    print(json.dumps(counts_by_split, sort_keys=True), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="config/datasets_agentgraph.yaml",
        help="dataset catalog path",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="optional comma-separated converter keys",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {
        item.strip() for item in args.datasets.split(",") if item.strip()
    } or None
    prepare(Path(args.catalog), selected=selected)


if __name__ == "__main__":
    main()
