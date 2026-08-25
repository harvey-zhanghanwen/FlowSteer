#!/usr/bin/env python3
"""Prepare the official AIME 2026 evaluation view for AgentGraph.

The record shape and atomic split writer are reused from
``prepare_agentgraph_datasets.py``. The default catalog follows downstream
SkillEval's fixed MathArena AIME 2026 source and writes only its official
evaluation population. The legacy AIME-family catalog remains supported for
existing project conditions and may additionally materialize:

* the local year-labelled AIME 2000--2024 subset supplies 512 training tasks;
* AIME 2025 is development/validation, optionally followed by a disjoint
  historical development supplement requested by the dataset catalog; and
* all 30 MathArena AIME 2026 problems are final-evaluation-only and retain the
  exact downstream SkillEval row schema and source order.

This command only aligns data.  It does not start a model, an API request,
training, backward, an optimizer update, or Skill publication.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Sequence, cast

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_agentgraph_datasets import (  # noqa: E402
    TASK_SCHEMA_VERSION,
    SplitWriters,
    _compat_record,
    _path,
)
from src.interactive.aime2026_adapter import (  # noqa: E402
    AIME2026_ANSWER_FORMAT,
    AIME2026_DATASET_KEY,
    AIME2026_EVALUATOR_VERSION,
    AIME2026_TASK_FAMILY,
    canonical_aime_integer,
)


AIME2026_CATALOG_SCHEMA_VERSION = "flowsteer.agentgraph.aime2026.dataset.v1"


def _read_official_parquet_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Thin port of SkillEval ``PyArrowParquetRowReader.read_rows``.

    The source must be one explicitly named absolute shard. ``to_pylist``
    preserves Arrow row order and scalar values; benchmark validation remains
    in ``_final_records``.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("official AIME Parquet source must be an absolute Path")
    if not path.is_file():
        raise FileNotFoundError(path)

    import pyarrow.parquet as parquet

    table = parquet.read_table(path)
    raw_rows = table.to_pylist()
    if type(raw_rows) is not list or not raw_rows:
        raise ValueError("official AIME Parquet source must contain records")
    if any(
        type(row) is not dict
        or any(type(field_name) is not str for field_name in row)
        for row in raw_rows
    ):
        raise TypeError("official AIME rows must be exact string-keyed objects")
    return tuple(cast(dict[str, object], row) for row in raw_rows)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield dict(value)


def _statement_identity(value: str) -> str:
    """Normalized statement identity used only to enforce split isolation."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _part_slug(value: object) -> str:
    text = " ".join(str(value).upper().replace("-", " ").split())
    if text.endswith(" I"):
        return "i"
    if text.endswith(" II"):
        return "ii"
    raise ValueError(f"unsupported AIME part: {value!r}")


def _record(
    *,
    task_id: str,
    question: str,
    answer: object,
    split: str,
    benchmark_year: int,
    benchmark_slice: str,
    evaluation_role: str,
    problem_index: int,
    part: str | None,
    source_name: str = "AIME",
    source_split: str | None = None,
    source_format: str | None = None,
    benchmark_id: str | None = None,
    dataset_revision: str | None = None,
    preserve_question_text: bool = False,
) -> dict[str, Any]:
    canonical = canonical_aime_integer(answer)
    extra: dict[str, Any] = {
        "answer_format": AIME2026_ANSWER_FORMAT,
        "benchmark_year": benchmark_year,
        "benchmark_slice": benchmark_slice,
        "evaluation_role": evaluation_role,
        "problem_index": problem_index,
        "evaluator_version": AIME2026_EVALUATOR_VERSION,
    }
    if part is not None:
        extra["part"] = part
    if source_split is not None:
        extra["source_split"] = source_split
    if source_format is not None:
        extra["source_format"] = source_format
    if benchmark_id is not None:
        extra["benchmark_id"] = benchmark_id
    if dataset_revision is not None:
        extra["dataset_revision"] = dataset_revision
    result = _compat_record(
        dataset_key=AIME2026_DATASET_KEY,
        source=source_name,
        task_id=task_id,
        question=question,
        ground_truth=canonical,
        split=split,
        task_type="math_reasoning",
        metric="accuracy",
        extra=extra,
        evaluator_payload={"accepted_answers": [canonical]},
        preserve_question_text=preserve_question_text,
    )
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "task_family": AIME2026_TASK_FAMILY,
            "benchmark_year": benchmark_year,
            "benchmark_slice": benchmark_slice,
            "evaluation_role": evaluation_role,
            "answer_format": AIME2026_ANSWER_FORMAT,
            "evaluator_version": AIME2026_EVALUATOR_VERSION,
            "problem_index": problem_index,
            **({"source_split": source_split} if source_split is not None else {}),
            **({"source_format": source_format} if source_format is not None else {}),
            **({"benchmark_id": benchmark_id} if benchmark_id is not None else {}),
            **(
                {"dataset_revision": dataset_revision}
                if dataset_revision is not None
                else {}
            ),
        }
    )
    result["metadata"] = metadata
    return result


def _flowsteer_development_answer(value: object) -> str:
    """Canonicalize FlowSteer's trusted AIME 2025 target field.

    FlowSteer's checked-in AIME 2025 row II-5 stores the official integer 336
    as ``336^\\circ`` because the problem asks for an arc measured in degrees.
    This source-only adapter removes that unit before the private target is
    built; it does not relax prediction scoring.
    """

    try:
        return canonical_aime_integer(value)
    except ValueError:
        if not isinstance(value, str):
            raise
        match = re.fullmatch(r"\s*([0-9]{1,3})\s*\^?\s*\\circ\s*", value)
        if match is None:
            raise
        return canonical_aime_integer(match.group(1))


def _historical_records(
    path: Path,
    *,
    maximum_year: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    for row in _read_jsonl(path):
        year = row.get("year")
        index = row.get("index")
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError("historical AIME row has invalid year")
        if year > maximum_year:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 15:
            raise ValueError("historical AIME index must lie in [1, 15]")
        part = _part_slug(row.get("part"))
        question = str(row.get("problem", "")).strip()
        if not question:
            raise ValueError("historical AIME problem must be non-empty")
        identity = _statement_identity(question)
        if identity in identities:
            raise ValueError("historical AIME contains a duplicate statement")
        identities.add(identity)
        records.append(
            _record(
                task_id=f"aime-historical:{year}:{part}:{index:02d}",
                question=question,
                answer=row.get("answer"),
                split="train",
                benchmark_year=year,
                benchmark_slice="historical_aime_2000_2024",
                evaluation_role="training",
                problem_index=index,
                part=part,
            )
        )
    return records


def _historical_development_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Relabel one training-pool record as a held-out development record.

    This is used only by an explicit development-128 catalog.  The record is
    selected before the 512 training candidates, so statement identity and
    task identity remain disjoint from both training and official AIME 2026.
    """

    result = dict(record)
    result["split"] = "validation"
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "benchmark_slice": "heldout_historical_aime_2000_2024",
            "evaluation_role": "development",
        }
    )
    result["metadata"] = metadata
    extra = dict(result["extra"])
    extra.update(
        {
            "benchmark_slice": "heldout_historical_aime_2000_2024",
            "evaluation_role": "development",
        }
    )
    result["extra"] = extra
    return result


def _development_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    part_counts: Counter[str] = Counter()
    for row in _read_jsonl(path):
        part = _part_slug(row.get("source"))
        part_counts[part] += 1
        index = part_counts[part]
        records.append(
            _record(
                task_id=f"aime-2025:{part}:{index:02d}",
                question=str(row.get("problem", "")).strip(),
                answer=_flowsteer_development_answer(row.get("ground_truth")),
                split="validation",
                benchmark_year=2025,
                benchmark_slice="development_aime_2025",
                evaluation_role="development",
                problem_index=index,
                part=part,
            )
        )
    return records


def _final_records(
    source_path: Path,
    *,
    dataset_revision: str = "d2de22f3c656b4f56cf8981212186377d1e23bc3",
) -> list[dict[str, Any]]:
    required_fields = {"answer", "problem", "problem_idx"}
    raw_rows = list(_read_official_parquet_rows(source_path))
    for row_index, row in enumerate(raw_rows):
        if set(row) != required_fields:
            raise ValueError(
                "official AIME 2026 row "
                f"{row_index} fields differ from {sorted(required_fields)}"
            )
        if isinstance(row["problem_idx"], bool) or not isinstance(
            row["problem_idx"], int
        ):
            raise ValueError("official AIME 2026 problem_idx must be int64")
        if isinstance(row["answer"], bool) or not isinstance(row["answer"], int):
            raise ValueError("official AIME 2026 answer must be int64")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError("official AIME 2026 problem must be non-empty text")
    rows = raw_rows
    indices = [row["problem_idx"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("official AIME 2026 problem_idx values must be unique")
    records: list[dict[str, Any]] = []
    for row in rows:
        index = row["problem_idx"]
        if not 1 <= index <= 30:
            raise ValueError("official AIME 2026 problem_idx must lie in [1, 30]")
        records.append(
            _record(
                task_id=f"aime-2026/{index:02d}",
                question=row["problem"],
                answer=row.get("answer"),
                split="test",
                benchmark_year=2026,
                benchmark_slice="official_aime_2026",
                evaluation_role="final-evaluation",
                problem_index=index,
                part=None,
                source_name="MathArena/aime_2026",
                source_split="train",
                source_format="matharena-aime-2026-parquet@1",
                benchmark_id="aime-2026",
                dataset_revision=dataset_revision,
                preserve_question_text=True,
            )
        )
    return records


def _assert_split_isolation(
    train: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
) -> None:
    groups = {
        "train": {_statement_identity(str(item["question"])) for item in train},
        "development": {
            _statement_identity(str(item["question"])) for item in development
        },
        "final": {_statement_identity(str(item["question"])) for item in final},
    }
    if len(groups["development"]) != len(development):
        raise ValueError("AIME 2025 development contains duplicate statements")
    if len(groups["final"]) != len(final):
        raise ValueError("AIME 2026 final evaluation contains duplicate statements")
    for left, right in (("train", "development"), ("train", "final"), ("development", "final")):
        if groups[left] & groups[right]:
            raise ValueError(f"AIME statement overlap between {left} and {right}")


def prepare(catalog_path: Path) -> Path:
    repo_root = catalog_path.resolve().parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if catalog.get("schema_version") != AIME2026_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported AIME 2026 dataset catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported AgentGraph task schema")

    sources = catalog["sources"]
    split_policy = catalog["split_policy"]
    mode = str(split_policy.get("mode", "family_compatibility"))
    if mode == "official_evaluation_only":
        train: list[dict[str, Any]] = []
        development: list[dict[str, Any]] = []
        historical_development: list[dict[str, Any]] = []
        dataset_revision = str(sources["dataset_revision"])
        final = _final_records(
            _path(str(sources["final_path"])),
            dataset_revision=dataset_revision,
        )
        expected_development = 0
    elif mode == "family_compatibility":
        historical = _historical_records(
            _path(str(sources["historical_path"])),
            maximum_year=int(split_policy["training_maximum_year"]),
        )
        train_count = int(split_policy["train_count"])
        development_historical_count = int(
            split_policy.get("development_historical_count", 0)
        )
        if development_historical_count < 0:
            raise ValueError("development_historical_count must be non-negative")
        if len(historical) < development_historical_count + train_count:
            raise ValueError(
                "historical AIME provides "
                f"{len(historical)} tasks, expected at least "
                f"{development_historical_count + train_count} for disjoint "
                "development and training"
            )
        historical_development = [
            _historical_development_record(item)
            for item in historical[:development_historical_count]
        ]
        train = historical[
            development_historical_count : development_historical_count + train_count
        ]
        development = _development_records(
            _path(str(sources["development_path"]))
        ) + historical_development
        final = _final_records(_path(str(sources["final_path"])))
        expected_development = int(split_policy["development_count"])
        dataset_revision = None
    else:
        raise ValueError(f"unsupported AIME dataset mode: {mode!r}")
    expected_final = int(split_policy["final_count"])
    if len(development) != expected_development:
        raise ValueError(
            f"AIME 2025 provides {len(development)} development tasks, "
            f"expected {expected_development}"
        )
    if len(final) != expected_final:
        raise ValueError(
            f"AIME 2026 provides {len(final)} final tasks, expected {expected_final}"
        )
    _assert_split_isolation(train, development, final)

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    for item in train:
        writers.write(item)
    for item in development:
        writers.write(item)
    for item in final:
        writers.write(item)
    manifest = {
        "schema_version": AIME2026_CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path.resolve()),
        "training_started": False,
        "evaluator_version": AIME2026_EVALUATOR_VERSION,
        "task_family": AIME2026_TASK_FAMILY,
        "dataset_mode": mode,
        "dataset_source": (
            "MathArena/aime_2026" if mode == "official_evaluation_only" else None
        ),
        "dataset_revision": dataset_revision,
        "source_split": "train" if mode == "official_evaluation_only" else None,
        "split_policy": {
            "train": (
                "empty: official AIME 2026 is evaluation-only"
                if mode == "official_evaluation_only"
                else (
                    "historical AIME 2000--2024 after the held-out historical "
                    f"development prefix, next {len(train)} in source order"
                )
            ),
            "validation": (
                "empty: official AIME 2026 is evaluation-only"
                if mode == "official_evaluation_only"
                else (
                    "complete AIME 2025 development population plus "
                    f"{len(historical_development)} held-out historical tasks"
                )
            ),
            "test": "complete official AIME 2026 final-evaluation population",
        },
        "counts_by_split": {
            "train": len(train),
            "validation": len(development),
            "test": len(final),
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
        default="config/datasets_aime2026_official_v1.yaml",
        help="AIME 2026 dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
