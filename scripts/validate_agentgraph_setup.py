#!/usr/bin/env python3
"""Validate strict AgentGraph configuration, catalog aliases, and GPU mapping."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training_agent_graph.yaml")
    parser.add_argument("--catalog", default="config/model_catalog.yaml")
    parser.add_argument("--allow-example-catalog", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.interactive.config_loader import (
        ConfigurationError,
        load_model_registry,
        load_yaml,
        validate_agent_graph_config,
    )

    config_path = root / args.config
    catalog_path = root / args.catalog
    if not catalog_path.exists() and args.allow_example_catalog:
        catalog_path = root / "config/model_catalog.yaml.example"
    try:
        config = load_yaml(config_path)
        validate_agent_graph_config(config)
        registry = load_model_registry(catalog_path)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"configuration: OK ({config_path})")
    print(f"model catalog: OK ({len(registry)} aliases, id={registry.catalog_id[:12]})")
    print("catalog model IDs: " + ", ".join(registry.model_ids))
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [sys.executable, str(root / "scripts/check_gpu_plan.py")],
            cwd=root,
            check=False,
        )
        if result.returncode:
            return result.returncode
    else:
        print("GPU check: skipped (nvidia-smi unavailable)")
    print("setup validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
