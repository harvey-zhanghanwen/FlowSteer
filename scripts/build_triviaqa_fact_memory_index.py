#!/usr/bin/env python3
"""Build the TriviaQA fact-only embedding index.

The command consumes only a materialized four-field fact JSONL.  Original
questions, canonical answers, accepted aliases, source task IDs, evaluator
metadata, and the external provenance manifest are neither accepted nor read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    build_triviaqa_fact_memory_index,
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
    parser.add_argument(
        "--facts",
        required=True,
        help=(
            "Fact JSONL containing only schema_version, memory_id, tool_id, "
            "and fact_text."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-model-revision", required=True)
    parser.add_argument("--frozen-top-k", type=_positive_integer, default=3)
    parser.add_argument(
        "--max-tool-calls-per-agent-call",
        type=_positive_integer,
        default=4,
    )
    parser.add_argument(
        "--max-turns-per-agent-call",
        type=_positive_integer,
        default=6,
    )
    parser.add_argument("--batch-size", type=_positive_integer, default=64)
    parser.add_argument("--snippet-characters", type=_positive_integer, default=512)
    parser.add_argument(
        "--expected-count",
        type=_positive_integer,
        default=None,
        help="Optional frozen fact count; mismatch fails closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_triviaqa_fact_memory_index(
            facts_path=args.facts,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            embedding_model_revision=args.embedding_model_revision,
            frozen_top_k=args.frozen_top_k,
            max_tool_calls_per_agent_call=args.max_tool_calls_per_agent_call,
            max_turns_per_agent_call=args.max_turns_per_agent_call,
            batch_size=args.batch_size,
            snippet_characters=args.snippet_characters,
            expected_count=args.expected_count,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TriviaQA fact-memory index build failed: {exc}", file=sys.stderr)
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
