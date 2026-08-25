#!/usr/bin/env python3
"""Prepare SkillFlow's v3 SWE-bench task populations for AgentGraph.

This adapter preserves the exact, source-ordered ``code_generation`` rows in
SkillFlow's ``train_v3.json`` and ``test_iid_v3.json``.  In the paper data,
the 500 training rows contain 372 unique SWE-bench Verified instances (128
rows are exact repeats), while the 128 IID held-out rows are unique and
instance-disjoint from training.

Repository/evaluator identity is joined from the official SWE-bench Verified
dataset by ``instance_id``.  Gold patches and test-harness fields remain in
the evaluator boundary; the task question, context, code-files fallback, and
public ``extra`` contain no gold patch or hidden test payload.  This command
only writes aligned JSONL and never starts a model, Docker, evaluation, or
training process.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


_SHARED_PREPARER_PATH = (
    Path(__file__).resolve().parent / "prepare_agentgraph_datasets.py"
)


def _load_shared_preparer() -> Any:
    module_name = "_flowsteer_prepare_agentgraph_datasets_for_swebench_skillflow_v3"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, _SHARED_PREPARER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load shared dataset preparer: {_SHARED_PREPARER_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SHARED = _load_shared_preparer()
TASK_SCHEMA_VERSION = _SHARED.TASK_SCHEMA_VERSION
SplitWriters = _SHARED.SplitWriters
_compat_record = _SHARED._compat_record
_path = _SHARED._path
_plain = _SHARED._plain


CATALOG_SCHEMA_VERSION = (
    "flowsteer.agentgraph.swebench.skillflow-v3.dataset.v1"
)
DATASET_KEY = "swe_bench"
DISPLAY_NAME = "SWE-bench"
TASK_TYPE = "code_generation"
TRAIN_COUNT = 500
TRAIN_UNIQUE_INSTANCE_IDS = 372
TRAIN_REPEATED_ROWS = 128
TEST_COUNT = 128
TEST_UNIQUE_INSTANCE_IDS = 128

RowProvider = Callable[[Path], Iterable[Mapping[str, Any]]]


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} has invalid {field}")
    return value.strip()


def _required_string(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} has invalid {field}")
    return value


def _json_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"SkillFlow v3 source must contain a JSON array: {path}")
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"SkillFlow v3 row {index} is not a mapping: {path}")
        yield row


def _code_generation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    source_name: str,
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if row.get("task_type") == TASK_TYPE]
    if len(selected) != expected_count:
        raise ValueError(
            f"{source_name} provides {len(selected)} {TASK_TYPE} rows; "
            f"expected {expected_count}"
        )
    return selected


def _source_identity(row: Mapping[str, Any], *, source_name: str) -> str:
    extra = row.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError(f"{source_name} row has no public extra mapping")
    expected = {
        "source": DISPLAY_NAME,
        "subset": "SWE-bench Verified",
        "metric": "resolved_rate",
    }
    drift = {
        key: {"expected": expected_value, "actual": extra.get(key)}
        for key, expected_value in expected.items()
        if extra.get(key) != expected_value
    }
    if drift:
        raise ValueError(
            f"{source_name} row metadata drift: "
            + json.dumps(drift, sort_keys=True)
        )
    return _required_text(extra, "instance_id", label=f"{source_name} extra")


def _official_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        instance_id = _required_text(row, "instance_id", label="Verified row")
        if instance_id in result:
            raise ValueError(
                f"official SWE-bench Verified contains duplicate instance_id: "
                f"{instance_id}"
            )
        result[instance_id] = row
    if not result:
        raise ValueError("official SWE-bench Verified population is empty")
    return result


def _verified_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to load SWE-bench Verified") from exc
    return load_from_disk(str(path))


def _canonical_source_row(row: Mapping[str, Any]) -> str:
    return json.dumps(_plain(dict(row)), ensure_ascii=False, sort_keys=True)


def _assert_train_repeat_protocol(rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        instance_id = _source_identity(row, source_name="skillflow_train_v3")
        grouped[instance_id].append(row)
    repeated_rows = len(rows) - len(grouped)
    if len(grouped) != TRAIN_UNIQUE_INSTANCE_IDS:
        raise ValueError(
            "skillflow_train_v3 unique instance count drift: "
            f"{len(grouped)} != {TRAIN_UNIQUE_INSTANCE_IDS}"
        )
    if repeated_rows != TRAIN_REPEATED_ROWS:
        raise ValueError(
            "skillflow_train_v3 repeated row count drift: "
            f"{repeated_rows} != {TRAIN_REPEATED_ROWS}"
        )
    for instance_id, group in grouped.items():
        if len({_canonical_source_row(row) for row in group}) != 1:
            raise ValueError(
                "skillflow_train_v3 repeated instance has conflicting rows: "
                f"{instance_id}"
            )


def _instance_ids(
    rows: Sequence[Mapping[str, Any]], *, source_name: str
) -> list[str]:
    return [_source_identity(row, source_name=source_name) for row in rows]


def _assert_split_protocol(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> None:
    _assert_train_repeat_protocol(train_rows)
    train_ids = set(_instance_ids(train_rows, source_name="skillflow_train_v3"))
    test_id_list = _instance_ids(test_rows, source_name="skillflow_test_iid_v3")
    test_ids = set(test_id_list)
    if len(test_ids) != TEST_UNIQUE_INSTANCE_IDS:
        raise ValueError(
            "skillflow_test_iid_v3 unique instance count drift: "
            f"{len(test_ids)} != {TEST_UNIQUE_INSTANCE_IDS}"
        )
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            "SkillFlow v3 train/test instance_id overlap: " + sorted(overlap)[0]
        )


def _record(
    source_row: Mapping[str, Any],
    official_row: Mapping[str, Any],
    *,
    split: str,
    source_name: str,
    selection_index: int,
    occurrence_index: int,
) -> dict[str, Any]:
    instance_id = _source_identity(source_row, source_name=source_name)
    official_instance_id = _required_text(
        official_row, "instance_id", label="Verified row"
    )
    if official_instance_id != instance_id:
        raise ValueError(
            f"Verified join changed instance identity: {instance_id} -> "
            f"{official_instance_id}"
        )

    source_extra = source_row["extra"]
    assert isinstance(source_extra, Mapping)
    repo = _required_text(official_row, "repo", label=f"Verified {instance_id}")
    base_commit = _required_text(
        official_row, "base_commit", label=f"Verified {instance_id}"
    )
    if _required_text(
        source_extra, "repo", label=f"{source_name} {instance_id} extra"
    ) != repo:
        raise ValueError(f"{source_name} repository identity drift: {instance_id}")
    if _required_text(
        source_extra, "base_commit", label=f"{source_name} {instance_id} extra"
    ) != base_commit:
        raise ValueError(f"{source_name} base commit drift: {instance_id}")

    question = _required_string(
        source_row, "question", label=f"{source_name} {instance_id}"
    )
    source_answer = _required_string(
        source_row, "answer", label=f"{source_name} {instance_id}"
    )
    gold_patch = _required_string(
        official_row, "patch", label=f"Verified {instance_id}"
    )
    if source_answer != gold_patch:
        raise ValueError(f"SkillFlow/Verified gold patch mismatch: {instance_id}")
    test_patch = _required_string(
        official_row, "test_patch", label=f"Verified {instance_id}"
    )
    if gold_patch in question or test_patch in question:
        raise ValueError(
            f"{source_name} question exposes evaluator-only patch: {instance_id}"
        )

    evaluator_payload = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "test_patch": test_patch,
        "FAIL_TO_PASS": _plain(official_row.get("FAIL_TO_PASS", [])),
        "PASS_TO_PASS": _plain(official_row.get("PASS_TO_PASS", [])),
        "version": official_row.get("version", ""),
        "environment_setup_commit": official_row.get(
            "environment_setup_commit", ""
        ),
        "dataset_source": "verified",
    }
    public_extra = {
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "dataset_source": "verified",
        "benchmark_slice": source_name,
        "selection_index": selection_index,
        "occurrence_index": occurrence_index,
    }
    task_id = (
        f"swe-bench:{instance_id}:skillflow-v3-train-row-{selection_index:03d}"
        if split == "train"
        else f"swe-bench:{instance_id}"
    )
    result = _compat_record(
        dataset_key=DATASET_KEY,
        source=DISPLAY_NAME,
        task_id=task_id,
        question=question,
        ground_truth=gold_patch,
        split=split,
        task_type=TASK_TYPE,
        metric="resolved_rate",
        context=(),
        extra=public_extra,
        evaluator_payload=evaluator_payload,
        code_files={},
        preserve_question_text=True,
    )
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "benchmark_slice": source_name,
            "dataset_source": "verified",
            "native_split": "test",
            "project_evaluation_role": (
                "training" if split == "train" else "iid-held-out-evaluation"
            ),
        }
    )
    result["metadata"] = metadata
    return result


def _records(
    rows: Sequence[Mapping[str, Any]],
    official: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    source_name: str,
) -> list[dict[str, Any]]:
    occurrences: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    for selection_index, row in enumerate(rows):
        instance_id = _source_identity(row, source_name=source_name)
        official_row = official.get(instance_id)
        if official_row is None:
            raise ValueError(
                f"{source_name} instance is absent from official Verified: "
                f"{instance_id}"
            )
        occurrence_index = occurrences[instance_id]
        occurrences[instance_id] += 1
        result.append(
            _record(
                row,
                official_row,
                split=split,
                source_name=source_name,
                selection_index=selection_index,
                occurrence_index=occurrence_index,
            )
        )
    return result


def prepare(
    catalog_path: Path,
    *,
    train_provider: RowProvider | None = None,
    iid_test_provider: RowProvider | None = None,
    verified_provider: RowProvider | None = None,
) -> Path:
    catalog_path = catalog_path.expanduser().resolve()
    repo_root = catalog_path.parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("SkillFlow v3 SWE-bench catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported SkillFlow v3 SWE-bench catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported AgentGraph task schema")

    expected_policy = {
        "filter_task_type": TASK_TYPE,
        "selection": "preserve_source_order",
        "train_source": "skillflow_train_v3",
        "train_count": TRAIN_COUNT,
        "train_unique_instance_ids": TRAIN_UNIQUE_INSTANCE_IDS,
        "train_repeated_rows": TRAIN_REPEATED_ROWS,
        "validation_population": "none",
        "test_source": "skillflow_test_iid_v3",
        "test_count": TEST_COUNT,
        "test_unique_instance_ids": TEST_UNIQUE_INSTANCE_IDS,
        "require_train_test_disjoint_instance_ids": True,
        "evaluator_join": "official_verified_by_instance_id",
    }
    policy = catalog.get("split_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("SkillFlow v3 SWE-bench split_policy must be a mapping")
    mismatches = {
        key: {"expected": expected, "actual": policy.get(key)}
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "SkillFlow v3 SWE-bench split policy drift: "
            + json.dumps(mismatches, sort_keys=True)
        )

    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("SkillFlow v3 SWE-bench sources must be a mapping")
    skillflow_source = sources.get("skillflow_v3")
    verified_source = sources.get("verified")
    if not isinstance(skillflow_source, Mapping) or not isinstance(
        verified_source, Mapping
    ):
        raise ValueError("SkillFlow v3 and official Verified sources are required")
    skillflow_root = _path(str(skillflow_source["path"]), base=repo_root)
    train_path = skillflow_root / str(skillflow_source["train_file"])
    iid_test_path = skillflow_root / str(skillflow_source["iid_test_file"])
    verified_path = _path(str(verified_source["path"]), base=repo_root)

    read_train = train_provider or _json_rows
    read_iid_test = iid_test_provider or _json_rows
    read_verified = verified_provider or _verified_rows
    train_rows = _code_generation_rows(
        read_train(train_path),
        expected_count=TRAIN_COUNT,
        source_name="skillflow_train_v3",
    )
    test_rows = _code_generation_rows(
        read_iid_test(iid_test_path),
        expected_count=TEST_COUNT,
        source_name="skillflow_test_iid_v3",
    )
    _assert_split_protocol(train_rows, test_rows)
    official = _official_index(read_verified(verified_path))
    train = _records(
        train_rows,
        official,
        split="train",
        source_name="skillflow_train_v3",
    )
    test = _records(
        test_rows,
        official,
        split="test",
        source_name="skillflow_test_iid_v3",
    )

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    for record in (*train, *test):
        writers.write(record)
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path),
        "training_started": False,
        "split_policy": dict(expected_policy),
        "counts_by_split": {
            "train": len(train),
            "validation": 0,
            "test": len(test),
        },
        "instance_id_protocol": {
            "status": "train_repeats_preserved_test_unique_train_test_disjoint",
            "train_rows": len(train),
            "train_unique_instance_ids": len(
                set(
                    _instance_ids(
                        train_rows, source_name="skillflow_train_v3"
                    )
                )
            ),
            "train_repeated_rows": len(train) - TRAIN_UNIQUE_INSTANCE_IDS,
            "test_rows": len(test),
            "test_unique_instance_ids": len(
                set(
                    _instance_ids(
                        test_rows, source_name="skillflow_test_iid_v3"
                    )
                )
            ),
            "train_test_overlap": 0,
        },
        "evaluator_join": {
            "source": "SWE-bench Verified",
            "key": "instance_id",
            "matched_train_rows": len(train),
            "matched_test_rows": len(test),
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
    }
    writers.publish(manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="config/datasets_swebench_skillflow_v3.yaml",
        help="SkillFlow v3 SWE-bench dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
