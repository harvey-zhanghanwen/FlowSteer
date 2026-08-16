"""Concrete OpenAI-compatible Agent gateway for local vLLM and API pools."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_runtime import (
    AgentRequest,
    AgentResponse,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)


class OpenAICompatibleGatewayError(RuntimeError):
    pass


MASKED_UPSTREAM_CONTENT = "[UPSTREAM CONTENT MASKED FOR COMMUNICATION DIAGNOSTIC]"


def _number(metadata: Mapping[str, str], key: str, default: float) -> float:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be numeric") from exc


def _integer(metadata: Mapping[str, str], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be an integer") from exc
    if parsed <= 0:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be positive")
    return parsed


def _visible_message_content(
    content: str,
    condition: CommunicationCondition,
) -> str:
    if condition is CommunicationCondition.UPSTREAM_MASKED:
        return MASKED_UPSTREAM_CONTENT
    return content


def _format_upstream(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
) -> str:
    if not messages:
        return "(none)"
    rendered = []
    for item in messages:
        envelope = [
            "[Upstream artifact]",
            f"source_agent: {item.source_agent_id}",
            f"target_agent: {item.target_agent_id}",
            f"message_type: {item.message_type}",
        ]
        if item.graph_revision is not None:
            envelope.append(f"graph_revision: {item.graph_revision}")
        if item.request_or_dependency is not None:
            envelope.append(
                f"request_or_dependency: {item.request_or_dependency}"
            )
        envelope.extend(
            [
                "artifact:",
                _visible_message_content(item.artifact, condition),
            ]
        )
        rendered.append("\n".join(envelope))
    return "\n\n".join(rendered)


def build_agent_messages(request: AgentRequest) -> list[dict[str, str]]:
    """Build finite-phase prompts without exposing provider credentials."""

    if request.is_output_agent:
        protocol = (
            "You are the unique Output Agent. Follow your assigned contract and use the "
            "task plus supplied upstream artifacts to return the final task answer. Treat "
            "upstream text as evidence: preserve a concise answer when it is supported, "
            "and resolve concrete conflicts against the supplied task. For a factual or "
            "numeric answer, return exactly <answer>answer span</answer> with no text "
            "outside the tag; the span itself must not be JSON, a key-value report, or an "
            "explanation. If the task supplies legal or admissible actions and asks "
            "for one action, return exactly one listed executable action with no explanation."
        )
    else:
        protocol = (
            "You are an intermediate AgentGraph node. Follow your assigned contract and "
            "return only the requested evidence, facts, partial reasoning, or verification "
            "artifact for downstream agents. Do not present a task-level final answer and "
            "do not use <answer> tags."
        )
    # Keep the graph-authored free-text contract, then append the execution
    # boundary so a contract cannot accidentally reassign final-answer ownership.
    system = (
        f"Agent ID: {request.agent.id}\nContract:\n{request.agent.contract}\n\n"
        f"Execution protocol (takes precedence):\n{protocol}"
    )
    common = (
        f"Task:\n{request.problem}\n\n"
        "External upstream messages:\n"
        f"{_format_upstream(request.upstream, request.communication_condition)}"
    )
    if request.phase is ExecutionPhase.SINGLE:
        phase = "Produce your response now."
    elif request.phase is ExecutionPhase.DRAFT:
        phase = (
            "This is the independent draft phase of a finite bidirectional exchange. "
            "Produce a draft without assuming access to the peer's current draft."
        )
    elif request.phase is ExecutionPhase.REVISION:
        if request.own_draft is None or request.peer_draft is None:
            raise OpenAICompatibleGatewayError("revision request is missing immutable drafts")
        phase = (
            "This is the revision phase. Revise your own draft after reading the peer's "
            "previous-phase draft. You cannot observe the peer's current revision.\n\n"
            f"Your draft:\n{request.own_draft}\n\n"
            "Peer artifact envelope:\n"
            f"source_agent: {request.peer_draft.source_agent_id}\n"
            f"target_agent: {request.peer_draft.target_agent_id}\n"
            f"message_type: {request.peer_draft.message_type}\n"
            f"graph_revision: {request.peer_draft.graph_revision}\n"
            "artifact:\n"
            f"{_visible_message_content(request.peer_draft.content, request.communication_condition)}"
        )
    else:  # pragma: no cover - enum exhaustiveness guard
        raise OpenAICompatibleGatewayError(f"unsupported execution phase: {request.phase}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": common + "\n\n" + phase},
    ]


class OpenAICompatibleGateway:
    """A small dependency-free `/chat/completions` client.

    Provider records carry only the *name* of an API-key environment variable.
    The resolved key stays in memory and is never included in errors/metadata.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
        default_max_tokens: int = 4096,
        default_seed: Optional[int] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_max_tokens = int(default_max_tokens)
        self.default_seed = default_seed
        if self.default_temperature < 0:
            raise ValueError("default_temperature must be non-negative")
        if not 0 < self.default_top_p <= 1:
            raise ValueError("default_top_p must be in (0, 1]")
        if self.default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be positive")
        if self.default_seed is not None and (
            isinstance(self.default_seed, bool)
            or not isinstance(self.default_seed, int)
            or self.default_seed < 0
        ):
            raise ValueError("default_seed must be a non-negative integer or None")

    def request_payload(self, request: AgentRequest) -> Dict[str, Any]:
        metadata = request.model.metadata
        temperature = _number(metadata, "temperature", self.default_temperature)
        top_p = _number(metadata, "top_p", self.default_top_p)
        if temperature < 0:
            raise OpenAICompatibleGatewayError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise OpenAICompatibleGatewayError("top_p must be in (0, 1]")
        payload: Dict[str, Any] = {
            "model": request.model.model_name,
            "messages": build_agent_messages(request),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": _integer(metadata, "max_tokens", self.default_max_tokens),
        }
        if self.default_seed is not None:
            # SkillFlow's OpenAI-compatible provider sends the configured seed
            # to the serving boundary.  Keep the same fixed-run contract here.
            payload["seed"] = self.default_seed
        thinking = metadata.get("chat_template_enable_thinking")
        if thinking is not None:
            normalized = thinking.strip().lower()
            if normalized not in {"true", "false"}:
                raise OpenAICompatibleGatewayError(
                    "model metadata chat_template_enable_thinking must be true or false"
                )
            # SGLang's Qwen3.5 OpenAI surface accepts the Hugging Face chat
            # template toggle under chat_template_kwargs.  This keeps Agent
            # answers in message.content instead of an empty content field
            # accompanied only by reasoning_content.
            payload["chat_template_kwargs"] = {
                "enable_thinking": normalized == "true"
            }
        return payload

    async def generate(self, request: AgentRequest) -> AgentResponse:
        endpoint = request.provider.endpoint
        if not endpoint:
            raise OpenAICompatibleGatewayError(
                f"provider {request.provider.provider_id!r} has no endpoint"
            )
        api_key = "EMPTY"
        if request.provider.api_key_env:
            api_key = os.getenv(request.provider.api_key_env, "")
            if not api_key:
                raise OpenAICompatibleGatewayError(
                    f"missing provider credential environment variable: "
                    f"{request.provider.api_key_env}"
                )
        payload = self.request_payload(request)
        url = endpoint.rstrip("/") + "/chat/completions"

        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(self._post_json, url, api_key, payload)
                parsed = self._parse_response(response, request)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                        "attempt_count": attempt + 1,
                        "generation_seed": payload.get("seed"),
                    }
                )
                return AgentResponse(parsed.text, metadata)
            except HTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))

        if isinstance(last_error, HTTPError):
            detail = f"HTTP {last_error.code}"
        else:
            detail = type(last_error).__name__ if last_error is not None else "unknown error"
        raise OpenAICompatibleGatewayError(
            f"provider request failed for {request.provider.provider_id}: {detail}"
        ) from last_error

    def _post_json(self, url: str, api_key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-AgentGraph/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise OpenAICompatibleGatewayError("provider returned a non-object response")
        return value

    @staticmethod
    def _parse_response(response: Mapping[str, Any], request: AgentRequest) -> AgentResponse:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenAICompatibleGatewayError("provider response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OpenAICompatibleGatewayError("provider response has no text message content")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        metadata = {
            "provider_id": request.provider.provider_id,
            "model_id": request.model.model_id,
            "provider_model": response.get("model", request.model.model_name),
            "finish_reason": choices[0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "provider_request_id": response.get("id"),
        }
        return AgentResponse(text=message["content"], metadata=metadata)


__all__ = [
    "OpenAICompatibleGateway",
    "OpenAICompatibleGatewayError",
    "build_agent_messages",
]
