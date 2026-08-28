#!/usr/bin/env python3
"""Materialize paraphrases for all 640 paired TriviaQA QA memories.

This is an explicitly transductive companion to
``generate_triviaqa_qa_memory_paraphrases.py``.  It reuses that script's local
Qwen3.5 paraphraser and strict semantic-preservation parser, but includes the
128 frozen development-validation questions and their canonical answers.
Consequently, any later score using this corpus is transductive retrieval
accuracy and is not an official held-out result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_triviaqa_qa_memory_paraphrases import (  # noqa: E402
    GENERATOR_PROVIDER,
    PARAPHRASE_METHOD,
    PROMPT_TEMPLATE_VERSION,
    LocalQwen35Paraphraser,
)
from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    TriviaQAQAMemoryRecord,
    load_materialized_qa_memory,
    validate_qa_memory_against_sources,
    write_materialized_qa_memory,
)
from src.interactive.triviaqa_transductive_qa_memory import (  # noqa: E402
    EVALUATION_REGIME,
    load_triviaqa_transductive_qa_memory_sources,
)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--train-only-paraphrases",
        default="data/triviaqa_qa_memory_v1/train_qa_memory.jsonl",
        help="Existing train-only materialization; this command refuses to overwrite it.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--model-id", default="supervisor_theta")
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--paraphrase-version", required=True)
    parser.add_argument("--base-seed", type=_nonnegative_integer, default=20260828)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=_nonnegative_integer, default=2)
    parser.add_argument("--expected-train-count", type=_positive_integer, default=512)
    parser.add_argument(
        "--expected-validation-count", type=_positive_integer, default=128
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = Path(args.output)
    train_only_path = Path(args.train_only_paraphrases)
    if output_path.resolve() == train_only_path.resolve():
        print(
            "transductive materialization must not overwrite the train-only corpus",
            file=sys.stderr,
        )
        return 1
    try:
        source_set = load_triviaqa_transductive_qa_memory_sources(
            args.train_tasks,
            args.validation_tasks,
            expected_train_count=args.expected_train_count,
            expected_validation_count=args.expected_validation_count,
        )
        existing = (
            load_materialized_qa_memory(output_path) if output_path.is_file() else ()
        )
        train_only = load_materialized_qa_memory(train_only_path)
        validate_qa_memory_against_sources(
            train_only,
            source_set.sources[: source_set.train_count],
            require_complete=True,
        )
        validate_qa_memory_against_sources(
            existing,
            source_set.sources,
            require_complete=False,
        )
        train_only_by_id = {
            record.source_train_task_id: record for record in train_only
        }
        for record in existing:
            bootstrap = train_only_by_id.get(record.source_train_task_id)
            if bootstrap is not None:
                if record != bootstrap:
                    raise ValueError(
                        "output train record differs from frozen train-only memory"
                    )
                continue
            admitted_seeds = range(
                args.base_seed + record.selection_index,
                args.base_seed + record.selection_index + args.max_retries + 1,
            )
            if (
                record.paraphrase_version != args.paraphrase_version
                or record.paraphrase_method != PARAPHRASE_METHOD
                or record.generator_provider != GENERATOR_PROVIDER
                or record.model_id != args.model_id
                or record.model_revision != args.model_revision
                or record.prompt_template_version != PROMPT_TEMPLATE_VERSION
                or record.generation_seed not in admitted_seeds
            ):
                raise ValueError(
                    "existing transductive paraphrases have incompatible provenance"
                )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"transductive source/materialization check failed: {exc}", file=sys.stderr)
        return 1

    try:
        client = LocalQwen35Paraphraser(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=os.environ.get(args.api_key_env, ""),
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        records = dict(train_only_by_id)
        records.update(
            {record.source_train_task_id: record for record in existing}
        )
        for source in source_set.sources:
            if source.source_train_task_id in records:
                continue
            paraphrase, accepted_seed = client.generate(
                source,
                seed=args.base_seed + source.selection_index,
            )
            records[source.source_train_task_id] = TriviaQAQAMemoryRecord.create(
                source=source,
                paraphrase_question=paraphrase,
                paraphrase_version=args.paraphrase_version,
                paraphrase_method=PARAPHRASE_METHOD,
                generator_provider=GENERATOR_PROVIDER,
                model_id=args.model_id,
                model_revision=args.model_revision,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                generation_seed=accepted_seed,
            )
            write_materialized_qa_memory(output_path, tuple(records.values()))
        completed = tuple(records.values())
        validate_qa_memory_against_sources(
            completed,
            source_set.sources,
            require_complete=True,
        )
        write_materialized_qa_memory(output_path, completed)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"transductive paraphrase generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": (
                    "flowsteer.triviaqa.qa_memory.transductive.materialization.v1"
                ),
                "output": str(output_path.resolve()),
                "record_count": len(completed),
                "bootstrap_train_count": len(train_only),
                "source_counts": {
                    "train": source_set.train_count,
                    "frozen_development_validation": source_set.evaluation_count,
                    "total": source_set.total_count,
                },
                "contains_evaluation_answers": True,
                "evaluation_memory_overlap_count": source_set.evaluation_count,
                "evaluation_regime": EVALUATION_REGIME,
                "official_heldout_eligible": False,
                "paraphrase_version": args.paraphrase_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
