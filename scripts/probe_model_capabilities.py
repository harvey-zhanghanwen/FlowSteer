#!/usr/bin/env python3
"""Probe exact provider model IDs for text, ReAct, and coding compatibility.

The script deliberately separates model discovery from capability admission:
every requested ID must first occur verbatim in the provider's current
``/v1/models`` response.  Listing is the default operation.  Completion calls
are made only with the explicit ``--run-probes`` flag and never fall back to a
different model.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Optional
from urllib.error import HTTPError, URLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from discover_models import (  # noqa: E402
    fetch_models_with_receipt,
    models_url,
)
from src.interactive.agent_graph import AgentExecutionMode, AgentNode  # noqa: E402
from src.interactive.agent_runtime import (  # noqa: E402
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec  # noqa: E402
from src.interactive.openai_gateway import (  # noqa: E402
    OpenAICompatibleGateway,
    OpenAICompatibleGatewayError,
    build_agent_messages,
)
from src.interactive.tool_runtime import ActionKind, StructuredAction  # noqa: E402


SCHEMA_VERSION = "flowsteer.model_catalog.capability_canary.v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/model_capability_canary/capability_receipt.json"
)

TEXT_PROBLEM = (
    "Context: Ada's destination country is France. The capital of France is Paris.\n"
    "Question: Which city is the capital of Ada's destination country?"
)
TEXT_EXPECTED = "<answer>Paris</answer>"

REACT_EXPECTED = {
    "arguments": {"text": "ping"},
    "kind": "tool",
    "name": "echo",
    "resource_id": "capability.echo",
    "skill_id": None,
}
REACT_PROBLEM = (
    "Capability canary only. Request the admitted echo resource once with the text "
    "ping. Do not answer the text task and do not select another resource."
)

CODING_DIFF = (
    "diff --git a/canary.txt b/canary.txt\n"
    "--- a/canary.txt\n"
    "+++ b/canary.txt\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)
CODING_EXPECTED = {
    "arguments": {"value": CODING_DIFF},
    "kind": "complete",
    "name": "complete",
    "resource_id": None,
    "skill_id": None,
}
CODING_PROBLEM = (
    "Capability canary only. Return the requested one-file patch as the value of one "
    "complete StructuredAction. The patch must replace the sole line 'old' with 'new' "
    "in canary.txt and use unified-diff format."
)


class ModelSelectionError(ValueError):
    """A requested ID was not present verbatim in the discovery response."""


ModelFetcher = Callable[
    [str, str, float],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: object) -> object:
    """Return receipt-safe JSON without introducing provider-specific objects."""

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        return str(value)


def discovered_model_ids(models: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Extract the exact, non-empty IDs returned by ``/v1/models``."""

    return tuple(
        sorted(
            {
                item["id"]
                for item in models
                if isinstance(item.get("id"), str) and item["id"]
            }
        )
    )


def select_exact_model_ids(
    discovered_ids: Sequence[str],
    requested_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return an ordered exact intersection, rejecting aliases and case changes."""

    available = set(discovered_ids)
    selected: list[str] = []
    missing: list[str] = []
    for model_id in requested_ids:
        if not isinstance(model_id, str) or not model_id:
            raise ModelSelectionError("requested model IDs must be non-empty strings")
        if model_id not in available:
            missing.append(model_id)
        elif model_id not in selected:
            selected.append(model_id)
    if missing:
        raise ModelSelectionError(
            "requested model IDs absent from the current /v1/models response: "
            + ", ".join(missing)
        )
    return tuple(selected)


def _probe_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "probe_id": "text_answer",
            "capability": "text",
            "problem": TEXT_PROBLEM,
            "contract": "Answer the supplied closed-context question with one concise span.",
            "execution_mode": AgentExecutionMode.REASONING,
            "allowed_tools": (),
            "artifact_type": "answer",
            "completion_condition": None,
            "is_output_agent": True,
            "expected": TEXT_EXPECTED,
        },
        {
            "probe_id": "structured_action",
            "capability": "react",
            "problem": REACT_PROBLEM,
            "contract": (
                "Return exactly this tool request as one StructuredAction: "
                + json.dumps(
                    REACT_EXPECTED,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "execution_mode": AgentExecutionMode.REACT,
            "allowed_tools": ("capability.echo",),
            "artifact_type": "tool_request",
            "completion_condition": "request capability.echo exactly once",
            "is_output_agent": False,
            "expected": REACT_EXPECTED,
        },
        {
            "probe_id": "coding_unified_diff",
            "capability": "coding",
            "problem": CODING_PROBLEM,
            "contract": (
                "Return exactly this iterative coding completion action: "
                + json.dumps(
                    CODING_EXPECTED,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "execution_mode": AgentExecutionMode.CODING,
            "allowed_tools": (),
            "artifact_type": "unified_diff",
            "completion_condition": "emit the requested non-empty unified diff",
            "is_output_agent": False,
            "expected": CODING_EXPECTED,
        },
    )


def _build_request(
    *,
    model_id: str,
    provider: ProviderSpec,
    definition: Mapping[str, object],
    request_index: int,
    chat_template_enable_thinking: Optional[bool] = None,
) -> AgentRequest:
    execution_mode = definition["execution_mode"]
    if not isinstance(execution_mode, AgentExecutionMode):
        raise TypeError("probe execution_mode is incompatible")
    allowed_tools = definition["allowed_tools"]
    if not isinstance(allowed_tools, tuple):
        raise TypeError("probe allowed_tools are incompatible")
    model_metadata = {"max_tokens": "256"}
    if chat_template_enable_thinking is not None:
        model_metadata["chat_template_enable_thinking"] = (
            "true" if chat_template_enable_thinking else "false"
        )
    model = ModelSpec(
        model_id=model_id,
        provider_id=provider.provider_id,
        model_name=model_id,
        metadata=model_metadata,
    )
    return AgentRequest(
        request_id=f"capability-canary-{request_index:04d}",
        run_id="model-capability-canary",
        graph_revision=0,
        problem=str(definition["problem"]),
        agent=AgentNode(
            str(definition["probe_id"]),
            model_id,
            str(definition["contract"]),
            allowed_tools=allowed_tools,
            execution_mode=execution_mode,
            artifact_type=str(definition["artifact_type"]),
            completion_condition=(
                str(definition["completion_condition"])
                if definition["completion_condition"] is not None
                else None
            ),
        ),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        is_output_agent=bool(definition["is_output_agent"]),
    )


def _request_receipt(request: AgentRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "model_id": request.model.model_id,
        "provider_id": request.provider.provider_id,
        "execution_mode": request.agent.execution_mode.value,
        "allowed_tools": list(request.agent.allowed_tools),
        "messages": build_agent_messages(request),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "seed": 20260820,
            "chat_template_enable_thinking": request.model.metadata.get(
                "chat_template_enable_thinking"
            ),
        },
    }


def _validate_response(
    definition: Mapping[str, object],
    response_text: str,
) -> tuple[bool, Optional[str]]:
    probe_id = definition["probe_id"]
    expected = definition["expected"]
    if probe_id == "text_answer":
        if response_text.strip() == expected:
            return True, None
        return False, "text_answer_mismatch"
    try:
        decoded = json.loads(response_text)
        action = StructuredAction.from_value(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "structured_action_invalid"
    if action.to_value() != expected:
        return False, "structured_action_mismatch"
    if probe_id == "structured_action" and action.kind is not ActionKind.TOOL:
        return False, "react_action_kind_invalid"
    if probe_id == "coding_unified_diff":
        if action.kind is not ActionKind.COMPLETE:
            return False, "coding_action_kind_invalid"
        arguments = action.arguments
        patch = arguments.get("value") if isinstance(arguments, dict) else None
        required_markers = ("diff --git ", "--- ", "+++ ", "@@ ")
        if not isinstance(patch, str) or not all(
            marker in patch for marker in required_markers
        ):
            return False, "unified_diff_invalid"
    return True, None


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    result: list[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in result:
        result.append(current)
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    return tuple(result)


def classify_probe_exception(
    exc: BaseException,
    *,
    secret_values: Sequence[str] = (),
) -> dict[str, object]:
    """Separate HTTP, transport, provider-response, and client failures."""

    chain = _exception_chain(exc)
    http_error = next((item for item in chain if isinstance(item, HTTPError)), None)
    if isinstance(http_error, HTTPError):
        category = "http_error"
        status: Optional[int] = int(http_error.code)
    elif any(
        isinstance(item, (URLError, TimeoutError, socket.timeout)) for item in chain
    ):
        category = "transport_error"
        status = None
    elif any(isinstance(item, OpenAICompatibleGatewayError) for item in chain):
        category = "provider_response_error"
        status = None
    else:
        category = "client_error"
        status = None
    message = str(exc)
    for secret in secret_values:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return {
        "category": category,
        "error_type": type(exc).__name__,
        "http_status": status,
        "message": message,
    }


async def _run_probe(
    *,
    gateway: object,
    request: AgentRequest,
    definition: Mapping[str, object],
    semaphore: asyncio.Semaphore,
    secret_values: Sequence[str],
) -> dict[str, object]:
    started_at = _utc_now()
    receipt: dict[str, object] = {
        "probe_id": definition["probe_id"],
        "capability": definition["capability"],
        "model_id": request.model.model_id,
        "started_at": started_at,
        "request": _request_receipt(request),
        "expected": _json_value(definition["expected"]),
    }
    try:
        async with semaphore:
            generate = getattr(gateway, "generate")
            generated = await generate(request)
        response = (
            generated if isinstance(generated, AgentResponse) else AgentResponse(generated)
        )
    except Exception as exc:
        error = classify_probe_exception(exc, secret_values=secret_values)
        receipt.update(
            {
                "completed_at": _utc_now(),
                "status": error["category"],
                "compatible": False,
                "raw_response": None,
                "validation_error": None,
                "error": error,
            }
        )
        return receipt

    compatible, validation_error = _validate_response(definition, response.text)
    receipt.update(
        {
            "completed_at": _utc_now(),
            "status": "passed" if compatible else "model_output_error",
            "compatible": compatible,
            "raw_response": {
                "text": response.text,
                "metadata": _json_value(dict(response.metadata)),
            },
            "validation_error": validation_error,
            "error": None,
        }
    )
    return receipt


async def run_probe_matrix(
    model_ids: Sequence[str],
    *,
    endpoint: str,
    api_key_env: str,
    api_key: str,
    timeout: float,
    concurrency: int,
    gateway: Optional[object] = None,
    chat_template_enable_thinking: Optional[bool] = None,
) -> list[dict[str, object]]:
    """Run exactly three independent, no-fallback probes per selected ID."""

    if concurrency < 1:
        raise ValueError("probe concurrency must be positive")
    provider = ProviderSpec(
        provider_id="model-capability-canary",
        kind="openai-compatible",
        endpoint=endpoint.removesuffix("/models"),
        max_concurrency=concurrency,
        api_key_env=api_key_env,
    )
    client = gateway or OpenAICompatibleGateway(
        timeout_seconds=timeout,
        max_retries=0,
        default_temperature=0.0,
        default_top_p=1.0,
        default_max_tokens=256,
        default_seed=20260820,
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks: list[Awaitable[dict[str, object]]] = []
    request_index = 0
    for model_id in model_ids:
        for definition in _probe_definitions():
            request_index += 1
            request = _build_request(
                model_id=model_id,
                provider=provider,
                definition=definition,
                request_index=request_index,
                chat_template_enable_thinking=chat_template_enable_thinking,
            )
            tasks.append(
                _run_probe(
                    gateway=client,
                    request=request,
                    definition=definition,
                    semaphore=semaphore,
                    secret_values=(api_key,),
                )
            )
    return list(await asyncio.gather(*tasks))


async def audit_model_capabilities(
    *,
    base_url: str,
    api_key_env: str,
    api_key: str,
    requested_model_ids: Sequence[str],
    mode: str,
    timeout: float,
    concurrency: int,
    fetcher: ModelFetcher = fetch_models_with_receipt,
    gateway: Optional[object] = None,
    chat_template_enable_thinking: Optional[bool] = None,
) -> dict[str, object]:
    """Discover first, then list, plan, or probe exact discovered IDs."""

    if mode not in {"list_only", "dry_run", "run_probes"}:
        raise ValueError("mode must be list_only, dry_run, or run_probes")
    endpoint = models_url(base_url)
    models, discovery_response = fetcher(endpoint, api_key, timeout)
    actual_ids = discovered_model_ids(models)
    selected_ids = select_exact_model_ids(actual_ids, requested_model_ids)
    if mode in {"dry_run", "run_probes"} and not selected_ids:
        raise ModelSelectionError(f"{mode} requires at least one exact --model ID")

    plan = [
        {
            "model_id": model_id,
            "probe_ids": [
                str(definition["probe_id"]) for definition in _probe_definitions()
            ],
            "completion_requests": 3,
        }
        for model_id in selected_ids
    ]
    probes: list[dict[str, object]] = []
    if mode == "run_probes":
        probes = await run_probe_matrix(
            selected_ids,
            endpoint=endpoint,
            api_key_env=api_key_env,
            api_key=api_key,
            timeout=timeout,
            concurrency=concurrency,
            gateway=gateway,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )

    passed = sum(item.get("status") == "passed" for item in probes)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "mode": mode,
        "models_endpoint": endpoint,
        "discovery": {
            "response": _json_value(discovery_response),
            "model_count": len(models),
            "models": _json_value(models),
            "actual_model_ids": list(actual_ids),
        },
        "requested_model_ids": list(requested_model_ids),
        "selected_model_ids": list(selected_ids),
        "probe_plan": plan,
        "probes": probes,
        "summary": {
            "completion_requests_planned": len(selected_ids) * 3,
            "completion_requests_executed": len(probes),
            "passed": passed,
            "failed": len(probes) - passed,
            "fallback_requests": 0,
        },
    }


def write_capability_receipt(
    output_path: str | os.PathLike[str],
    receipt: Mapping[str, object],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _mode_from_args(args: argparse.Namespace) -> str:
    if args.run_probes:
        return "run_probes"
    if args.dry_run:
        return "dry_run"
    return "list_only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("VECTOR_ENGINE_BASE_URL", "https://api.vectorengine.ai/v1"),
    )
    parser.add_argument("--api-key-env", default="VECTOR_ENGINE_API_KEY")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="exact ID from the current /v1/models response; repeat as needed",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-only", action="store_true", help="discover only (default)")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="discover, validate exact IDs, and write the three-probe plan",
    )
    mode.add_argument(
        "--run-probes",
        action="store_true",
        help="make three completion calls per selected exact model ID",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--chat-template-enable-thinking",
        choices=("true", "false"),
        default=None,
        help=(
            "optional Qwen chat-template setting; use the same value as the "
            "frozen model catalog"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="non-secret discovery/probe receipt path",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"missing required environment variable: {args.api_key_env}", file=sys.stderr)
        return 2
    selected_mode = _mode_from_args(args)
    try:
        receipt = asyncio.run(
            audit_model_capabilities(
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                api_key=api_key,
                requested_model_ids=args.model,
                mode=selected_mode,
                timeout=args.timeout,
                concurrency=args.concurrency,
                chat_template_enable_thinking=(
                    args.chat_template_enable_thinking == "true"
                    if args.chat_template_enable_thinking is not None
                    else None
                ),
            )
        )
    except ModelSelectionError as exc:
        print(f"model selection failed: {exc}", file=sys.stderr)
        return 4
    except HTTPError as exc:
        print(f"model discovery failed with HTTP {exc.code}", file=sys.stderr)
        return 3
    except (URLError, TimeoutError, ValueError) as exc:
        print(f"model discovery/probe setup failed: {type(exc).__name__}", file=sys.stderr)
        return 3

    saved = write_capability_receipt(args.output, receipt)
    actual_ids = receipt["discovery"]["actual_model_ids"]
    print(f"Discovered {len(actual_ids)} exact model IDs; saved receipt to {saved}")
    if selected_mode == "list_only":
        for model_id in actual_ids:
            print(model_id)
    elif selected_mode == "dry_run":
        print(
            f"Dry run only: planned {receipt['summary']['completion_requests_planned']} "
            "completion requests; executed 0"
        )
    else:
        for item in receipt["probes"]:
            print(f"{item['model_id']} {item['probe_id']}: {item['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
