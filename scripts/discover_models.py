#!/usr/bin/env python3
"""List model IDs from an OpenAI-compatible provider without exposing its key."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


PREFERRED_MARKERS = (
    "qwen3.5",
    "deepseek-v4",
    "gpt-4o-mini",
    "gemini",
    "grok",
    "minimax",
)


def models_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/console"):
        path = path[: -len("/console")]
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunparse(parsed._replace(path=path + "/models", query="", fragment=""))


def fetch_models(endpoint: str, api_key: str, timeout: float) -> list[dict[str, Any]]:
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("provider response does not contain an OpenAI-compatible data list")
    return [item for item in data if isinstance(item, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VECTOR_ENGINE_BASE_URL", "https://api.vectorengine.ai/v1"),
    )
    parser.add_argument("--api-key-env", default="VECTOR_ENGINE_API_KEY")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="emit the provider objects as JSON")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"missing required environment variable: {args.api_key_env}", file=sys.stderr)
        return 2
    try:
        endpoint = models_url(args.base_url)
        models = fetch_models(endpoint, api_key, args.timeout)
    except HTTPError as exc:
        print(f"model discovery failed with HTTP {exc.code} at {exc.url}", file=sys.stderr)
        return 3
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"model discovery failed: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(models, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    ids = sorted(str(item.get("id")) for item in models if item.get("id"))
    print(f"Discovered {len(ids)} models from {endpoint}")
    for model_id in ids:
        preferred = any(marker in model_id.lower() for marker in PREFERRED_MARKERS)
        print(f"{'*' if preferred else ' '} {model_id}")
    print("* = matches the configured cheap/fast preference markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
