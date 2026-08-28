#!/usr/bin/env python3
"""Build the explicitly transductive 640-pair TriviaQA BGE index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_qa_memory import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from src.interactive.triviaqa_transductive_qa_memory import (  # noqa: E402
    build_triviaqa_transductive_qa_memory_index,
)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paraphrases", required=True)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-only-index",
        default="data/triviaqa_qa_memory_v1/index",
        help="Existing train-only index; this command refuses to overwrite it.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-model-revision", required=True)
    parser.add_argument("--frozen-top-k", type=_positive_integer, default=3)
    parser.add_argument(
        "--max-tool-calls-per-agent-call", type=_positive_integer, default=4
    )
    parser.add_argument(
        "--max-turns-per-agent-call", type=_positive_integer, default=7
    )
    parser.add_argument("--batch-size", type=_positive_integer, default=64)
    parser.add_argument("--snippet-characters", type=_positive_integer, default=512)
    parser.add_argument("--expected-train-count", type=_positive_integer, default=512)
    parser.add_argument(
        "--expected-validation-count", type=_positive_integer, default=128
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == Path(args.train_only_index).resolve():
        print(
            "transductive index must not overwrite the train-only index",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = build_triviaqa_transductive_qa_memory_index(
            paraphrases_path=args.paraphrases,
            train_tasks_path=args.train_tasks,
            validation_tasks_path=args.validation_tasks,
            output_dir=output_dir,
            embedding_model=args.embedding_model,
            embedding_model_revision=args.embedding_model_revision,
            frozen_top_k=args.frozen_top_k,
            max_tool_calls_per_agent_call=args.max_tool_calls_per_agent_call,
            max_turns_per_agent_call=args.max_turns_per_agent_call,
            batch_size=args.batch_size,
            snippet_characters=args.snippet_characters,
            expected_train_count=args.expected_train_count,
            expected_validation_count=args.expected_validation_count,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"transductive TriviaQA index build failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            manifest.to_value(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
