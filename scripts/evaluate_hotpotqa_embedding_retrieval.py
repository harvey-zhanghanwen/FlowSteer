#!/usr/bin/env python3
"""Run the frozen HotpotQA multi-hop embedding-retrieval condition."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_hotpotqa_round import (  # noqa: E402
    ConfigurationError,
    HotpotRoundError,
    _safe_error,
    run_hotpot_round,
)


CONFIG = PROJECT_ROOT / "config/evaluation_hotpotqa_embedding_retrieval_v4.yaml"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    try:
        manifest = asyncio.run(
            run_hotpot_round(
                args.config,
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
            )
        )
    except (ConfigurationError, HotpotRoundError, ValueError, RuntimeError) as exc:
        print(f"HotpotQA embedding retrieval failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
