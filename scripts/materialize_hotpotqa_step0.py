#!/usr/bin/env python3
"""Preflight or materialize the untrained HotpotQA policy_step_000000 LoRA."""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

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

from src.interactive.hotpot_step0 import (
    HotpotStep0Config,
    materialize_hotpot_step0,
    preflight_hotpot_step0,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "QWEN35_9B_MODEL_PATH",
            "/home/test/SKILLEV/skillev-new-b2-temp/model/"
            "Qwen3.5-9B-modelscope",
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_multiagent_skill/policy_step_000000"
        ),
    )
    parser.add_argument(
        "--revision",
        default="74be52bb6bd9f0e9e68dacb72636b75649197983",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="load Qwen3.5 on CPU and write the fresh theta adapter",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    config = HotpotStep0Config(
        model_path=args.model_path,
        output_dir=str(output),
        revision=args.revision,
    )
    receipt = (
        materialize_hotpot_step0(config)
        if args.materialize
        else preflight_hotpot_step0(config)
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
