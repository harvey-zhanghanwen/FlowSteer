#!/usr/bin/env python3
"""List model IDs from an OpenAI-compatible provider without exposing its key."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.agent_graph import AgentNode  # noqa: E402
from src.interactive.agent_runtime import (  # noqa: E402
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec  # noqa: E402
from src.interactive.openai_gateway import OpenAICompatibleGateway  # noqa: E402


PREFERRED_MARKERS = (
    "qwen3.5",
    "deepseek-v4",
    "gpt-4o-mini",
    "gemini",
    "grok",
    "minimax",
)

CANARY_PROBLEM = (
    "Context: Ada's destination country is France. The capital of France is Paris.\n"
    "Question: Which city is the capital of Ada's destination country?"
)
CANARY_EXPECTED = "<answer>Paris</answer>"


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


def fetch_models_with_receipt(
    endpoint: str,
    api_key: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        response_metadata = {
            "http_status": int(response.status),
            "request_id": (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
                or response.headers.get("x-amzn-requestid")
            ),
        }
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("provider response does not contain an OpenAI-compatible data list")
    return [item for item in data if isinstance(item, dict)], response_metadata


def fetch_models(endpoint: str, api_key: str, timeout: float) -> list[dict[str, Any]]:
    """Backward-compatible model-list helper used by existing callers."""

    models, _ = fetch_models_with_receipt(endpoint, api_key, timeout)
    return models


def write_receipt(
    output_path: str | os.PathLike[str],
    *,
    endpoint: str,
    models: list[dict[str, Any]],
    response_metadata: dict[str, Any],
) -> Path:
    """Persist the provider response without credentials or request headers."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "flowsteer.model_catalog.discovery_receipt.v1",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model_count": len(models),
        "response": response_metadata,
        "models": models,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


async def probe_models(
    model_ids: list[str],
    *,
    endpoint: str,
    api_key_env: str,
    timeout: float,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run one Output-protocol text-QA canary per exact provider model ID."""

    if concurrency < 1:
        raise ValueError("canary concurrency must be positive")
    provider = ProviderSpec(
        provider_id="vectorengine-canary",
        kind="openai-compatible",
        endpoint=endpoint.removesuffix("/models"),
        max_concurrency=concurrency,
        api_key_env=api_key_env,
    )
    gateway = OpenAICompatibleGateway(
        timeout_seconds=timeout,
        max_retries=0,
        default_temperature=0.0,
        default_top_p=1.0,
        default_max_tokens=128,
        default_seed=20260816,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int, model_id: str) -> dict[str, Any]:
        model = ModelSpec(
            model_id=model_id,
            provider_id=provider.provider_id,
            model_name=model_id,
            metadata={"max_tokens": "128"},
        )
        request = AgentRequest(
            request_id=f"model-catalog-canary-{index:04d}",
            run_id="model-catalog-audit",
            graph_revision=0,
            problem=CANARY_PROBLEM,
            agent=AgentNode(
                "output",
                model_id,
                "Answer the supplied two-hop text question using one concise answer span.",
            ),
            model=model,
            provider=provider,
            phase=ExecutionPhase.SINGLE,
            is_output_agent=True,
        )
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            async with semaphore:
                response = await gateway.generate(request)
        except Exception as exc:
            return {
                "model_id": model_id,
                "provider": "vectorengine",
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "compatible": False,
                "error": f"{type(exc).__name__}: {exc}",
                "response": None,
            }
        metadata = dict(response.metadata)
        compatible = response.text.strip() == CANARY_EXPECTED
        return {
            "model_id": model_id,
            "provider": "vectorengine",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if compatible else "incompatible_output",
            "compatible": compatible,
            "expected": CANARY_EXPECTED,
            "response": response.text,
            "request_id": metadata.get("provider_request_id"),
            "provider_model": metadata.get("provider_model"),
            "prompt_tokens": metadata.get("prompt_tokens"),
            "completion_tokens": metadata.get("completion_tokens"),
            "total_tokens": metadata.get("total_tokens"),
            "latency_ms": metadata.get("latency_ms"),
            "attempt_count": metadata.get("attempt_count"),
            "finish_reason": metadata.get("finish_reason"),
        }

    return list(await asyncio.gather(*(run(index, value) for index, value in enumerate(model_ids))))


def write_canary_receipt(
    output_path: str | os.PathLike[str],
    *,
    endpoint: str,
    canaries: list[dict[str, Any]],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "flowsteer.model_catalog.canary_receipt.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint.removesuffix("/models") + "/chat/completions",
        "prompt": CANARY_PROBLEM,
        "expected": CANARY_EXPECTED,
        "canaries": canaries,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VECTOR_ENGINE_BASE_URL", "https://api.vectorengine.ai/v1"),
    )
    parser.add_argument("--api-key-env", default="VECTOR_ENGINE_API_KEY")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="emit the provider objects as JSON")
    parser.add_argument(
        "--receipt-output",
        help="write the complete non-secret model-list receipt to this path",
    )
    parser.add_argument(
        "--canary-model",
        action="append",
        default=[],
        help="exact model ID to probe once; repeat for multiple models",
    )
    parser.add_argument("--canary-output", help="write non-secret canary receipts here")
    parser.add_argument("--canary-concurrency", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"missing required environment variable: {args.api_key_env}", file=sys.stderr)
        return 2
    try:
        endpoint = models_url(args.base_url)
        models, response_metadata = fetch_models_with_receipt(
            endpoint,
            api_key,
            args.timeout,
        )
    except HTTPError as exc:
        print(f"model discovery failed with HTTP {exc.code} at {exc.url}", file=sys.stderr)
        return 3
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"model discovery failed: {exc}", file=sys.stderr)
        return 3

    if args.receipt_output:
        saved = write_receipt(
            args.receipt_output,
            endpoint=endpoint,
            models=models,
            response_metadata=response_metadata,
        )
        print(f"Saved {len(models)} models to {saved}")

    if args.canary_model:
        available_ids = {str(item.get("id")) for item in models if item.get("id")}
        unknown = [model_id for model_id in args.canary_model if model_id not in available_ids]
        if unknown:
            print(
                "canary model IDs are absent from the discovered list: " + ", ".join(unknown),
                file=sys.stderr,
            )
            return 4
        canaries = asyncio.run(
            probe_models(
                list(dict.fromkeys(args.canary_model)),
                endpoint=endpoint,
                api_key_env=args.api_key_env,
                timeout=args.timeout,
                concurrency=args.canary_concurrency,
            )
        )
        if args.canary_output:
            saved = write_canary_receipt(
                args.canary_output,
                endpoint=endpoint,
                canaries=canaries,
            )
            print(f"Saved {len(canaries)} canary receipts to {saved}")
        for item in canaries:
            print(f"CANARY {item['model_id']}: {item['status']}")

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
