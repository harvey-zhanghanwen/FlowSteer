"""LLM-based clustering of free-form Agent contracts.

The dynamic combination ledger uses a small, fixed set of role clusters while
the Flow-Director remains free to write Agent contracts in ordinary text.  This
module keeps that boundary explicit: the cluster is taken only from an LLM
completion that satisfies the response schema.  There is deliberately no
keyword or regular-expression fallback.

The OpenAI-compatible client follows the same dependency-free HTTP boundary as
the project's Director and Agent clients.  Constructing either the classifier
or the client performs no network I/O.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


ROLE_CLASSIFIER_SCHEMA_VERSION = "role-cluster-schema-v1"
ROLE_CLASSIFIER_PROMPT_VERSION = "role-cluster-prompt-v1"
ROLE_CLASSIFIER_VERSION = "llm-role-classifier-v1"
CONTRACT_REWRITE_SCHEMA_VERSION = "role-contract-rewrite-schema-v1"
CONTRACT_REWRITE_PROMPT_VERSION = "role-contract-rewrite-prompt-v1"
CONTRACT_REWRITER_VERSION = "llm-role-contract-rewriter-v1"

RoleCluster = Literal[
    "solve",
    "verify",
    "plan",
    "summarize",
    "arbitrate",
    "retrieve",
    "code",
    "test",
]

ROLE_CLUSTERS: Tuple[RoleCluster, ...] = (
    "solve",
    "verify",
    "plan",
    "summarize",
    "arbitrate",
    "retrieve",
    "code",
    "test",
)

ROLE_CLASSIFIER_SYSTEM_PROMPT = """Classify the primary function of the supplied free-text Agent contract.
Choose exactly one label: solve (derive an answer), verify (check correctness or evidence), plan (decompose or order work), summarize (condense information), arbitrate (reconcile alternatives), retrieve (obtain information), code (write or modify code), or test (evaluate behavior).
Return exactly one JSON object with this schema: {"role_cluster":"<label>"}. Do not rewrite the contract or design a workflow."""

CONTRACT_REWRITE_SYSTEM_PROMPT = """Revise the supplied free-text Agent contract so its primary function belongs to the supplied target role cluster.
Change only that function. Preserve the task scope, constraints, allowed tools, information boundaries, output protocol, and all other details. Do not add a workflow or fixed role title.
Return exactly one JSON object with this schema: {"contract":"<revised free-text contract>"}."""


class RoleClassificationError(RuntimeError):
    """Classification failed without selecting a fallback cluster."""

    def __init__(self, message: str, receipt: "RoleClassificationReceipt") -> None:
        super().__init__(message)
        self.receipt = receipt


class RoleCompletionClientError(RuntimeError):
    """The OpenAI-compatible completion boundary failed."""


class ContractRewriteError(RuntimeError):
    """Contract rewriting failed without fabricating a replacement contract."""

    def __init__(self, message: str, receipt: "ContractRewriteReceipt") -> None:
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class RoleCompletionRequest:
    """One versioned request sent to the injected classification model."""

    request_id: str
    messages: Tuple[Mapping[str, str], ...]
    seed: Optional[int]
    classifier_version: str
    prompt_version: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "messages": [dict(message) for message in self.messages],
            "seed": self.seed,
            "classifier_version": self.classifier_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ContractRewriteRequest:
    """One versioned role-only contract intervention request."""

    request_id: str
    messages: Tuple[Mapping[str, str], ...]
    seed: Optional[int]
    rewriter_version: str
    prompt_version: str
    schema_version: str
    task_family: str
    target_role_cluster: RoleCluster
    original_contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "messages": [dict(message) for message in self.messages],
            "seed": self.seed,
            "rewriter_version": self.rewriter_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "task_family": self.task_family,
            "target_role_cluster": self.target_role_cluster,
            "original_contract": self.original_contract,
        }


@dataclass(frozen=True, slots=True)
class RoleCompletionResponse:
    """Text completion plus provider receipt fields."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AsyncRoleCompletion(Protocol):
    """Injectable asynchronous completion boundary used by the classifier."""

    async def complete(self, request: RoleCompletionRequest) -> RoleCompletionResponse:
        ...


class AsyncContractRewriteCompletion(Protocol):
    """Injectable asynchronous completion boundary used by contract rewriting."""

    async def complete(self, request: ContractRewriteRequest) -> RoleCompletionResponse:
        ...


@dataclass(frozen=True, slots=True)
class RoleClassificationReceipt:
    """Auditable request/result receipt; failures never contain a chosen role."""

    request: RoleCompletionRequest
    status: Literal["classified", "failed"]
    role_cluster: Optional[RoleCluster]
    raw_completion: Optional[str]
    completion_metadata: Mapping[str, Any]
    latency_ms: float
    failure_stage: Optional[Literal["completion", "parse"]] = None
    error_type: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "classified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status,
            "role_cluster": self.role_cluster,
            "raw_completion": self.raw_completion,
            "completion_metadata": dict(self.completion_metadata),
            "latency_ms": self.latency_ms,
            "failure_stage": self.failure_stage,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class RoleClassification:
    role_cluster: RoleCluster
    receipt: RoleClassificationReceipt


@dataclass(frozen=True, slots=True)
class ContractRewriteReceipt:
    """Request/result receipt for one role-only paired intervention."""

    request: ContractRewriteRequest
    status: Literal["rewritten", "failed"]
    rewritten_contract: Optional[str]
    raw_completion: Optional[str]
    completion_metadata: Mapping[str, Any]
    latency_ms: float
    failure_stage: Optional[Literal["completion", "parse"]] = None
    error_type: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "rewritten"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status,
            "rewritten_contract": self.rewritten_contract,
            "raw_completion": self.raw_completion,
            "completion_metadata": dict(self.completion_metadata),
            "latency_ms": self.latency_ms,
            "failure_stage": self.failure_stage,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class ContractRewrite:
    contract: str
    receipt: ContractRewriteReceipt


def build_role_classification_messages(
    agent_contract: str,
) -> Tuple[Mapping[str, str], ...]:
    """Render the short neutral prompt without constraining contract wording."""

    if not isinstance(agent_contract, str) or not agent_contract.strip():
        raise ValueError("agent_contract must be a non-empty string")
    user_payload = json.dumps(
        {"agent_contract": agent_contract},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        {"role": "system", "content": ROLE_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    )


def build_contract_rewrite_messages(
    original_contract: str,
    *,
    target_role_cluster: RoleCluster,
    task_family: str,
) -> Tuple[Mapping[str, str], ...]:
    """Render a neutral request for a role-only paired intervention."""

    if not isinstance(original_contract, str) or not original_contract.strip():
        raise ValueError("original_contract must be a non-empty string")
    if target_role_cluster not in ROLE_CLUSTERS:
        raise ValueError("target_role_cluster is not in the fixed role catalog")
    if not isinstance(task_family, str) or not task_family.strip():
        raise ValueError("task_family must be a non-empty string")
    user_payload = json.dumps(
        {
            "original_contract": original_contract,
            "target_role_cluster": target_role_cluster,
            "task_family": task_family,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        {"role": "system", "content": CONTRACT_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_role_classification(text: str) -> RoleCluster:
    """Parse the exact one-field JSON schema; no text recovery is attempted."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("classification completion must be non-empty text")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("classification completion is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"role_cluster"}:
        raise ValueError("classification completion must contain only role_cluster")
    role = value["role_cluster"]
    if not isinstance(role, str) or role not in ROLE_CLUSTERS:
        raise ValueError("role_cluster is not in the fixed role catalog")
    return role  # type: ignore[return-value]


def parse_rewritten_contract(text: str) -> str:
    """Parse the exact contract rewrite schema without text recovery."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("contract rewrite completion must be non-empty text")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("contract rewrite completion is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"contract"}:
        raise ValueError("contract rewrite completion must contain only contract")
    contract = value["contract"]
    if not isinstance(contract, str) or not contract.strip():
        raise ValueError("rewritten contract must be a non-empty string")
    return contract


class RoleClassifier:
    """Classify a free-form contract exclusively from an injected LLM response."""

    def __init__(
        self,
        completion: AsyncRoleCompletion,
        *,
        classifier_version: str = ROLE_CLASSIFIER_VERSION,
        request_id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        if not classifier_version.strip():
            raise ValueError("classifier_version must be non-empty")
        self.completion = completion
        self.classifier_version = classifier_version
        self.request_id_factory = request_id_factory

    def build_request(
        self,
        agent_contract: str,
        *,
        seed: Optional[int] = None,
    ) -> RoleCompletionRequest:
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        request_id = self.request_id_factory()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id_factory must return a non-empty string")
        return RoleCompletionRequest(
            request_id=request_id,
            messages=build_role_classification_messages(agent_contract),
            seed=seed,
            classifier_version=self.classifier_version,
            prompt_version=ROLE_CLASSIFIER_PROMPT_VERSION,
            schema_version=ROLE_CLASSIFIER_SCHEMA_VERSION,
        )

    async def classify(
        self,
        agent_contract: str,
        *,
        seed: Optional[int] = None,
    ) -> RoleClassification:
        request = self.build_request(agent_contract, seed=seed)
        started_at = time.monotonic()
        try:
            response = await self.completion.complete(request)
            if not isinstance(response, RoleCompletionResponse):
                raise TypeError("completion client returned an invalid response type")
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            receipt = RoleClassificationReceipt(
                request=request,
                status="failed",
                role_cluster=None,
                raw_completion=None,
                completion_metadata={},
                latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
                failure_stage="completion",
                error_type=type(exc).__name__,
            )
            raise RoleClassificationError(
                "role classification completion failed; no cluster was selected",
                receipt,
            ) from exc

        try:
            role_cluster = parse_role_classification(response.text)
        except (TypeError, ValueError) as exc:
            receipt = RoleClassificationReceipt(
                request=request,
                status="failed",
                role_cluster=None,
                raw_completion=response.text,
                completion_metadata=dict(response.metadata),
                latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
                failure_stage="parse",
                error_type=type(exc).__name__,
            )
            raise RoleClassificationError(
                "role classification response failed schema validation; no cluster was selected",
                receipt,
            ) from exc

        receipt = RoleClassificationReceipt(
            request=request,
            status="classified",
            role_cluster=role_cluster,
            raw_completion=response.text,
            completion_metadata=dict(response.metadata),
            latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
        )
        return RoleClassification(role_cluster=role_cluster, receipt=receipt)


class ContractRoleRewriter:
    """Use an LLM to change only the role field represented by a free contract."""

    def __init__(
        self,
        completion: AsyncContractRewriteCompletion,
        *,
        rewriter_version: str = CONTRACT_REWRITER_VERSION,
        request_id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        if not rewriter_version.strip():
            raise ValueError("rewriter_version must be non-empty")
        self.completion = completion
        self.rewriter_version = rewriter_version
        self.request_id_factory = request_id_factory

    def build_request(
        self,
        original_contract: str,
        *,
        target_role_cluster: RoleCluster,
        task_family: str,
        seed: Optional[int] = None,
    ) -> ContractRewriteRequest:
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        messages = build_contract_rewrite_messages(
            original_contract,
            target_role_cluster=target_role_cluster,
            task_family=task_family,
        )
        request_id = self.request_id_factory()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id_factory must return a non-empty string")
        return ContractRewriteRequest(
            request_id=request_id,
            messages=messages,
            seed=seed,
            rewriter_version=self.rewriter_version,
            prompt_version=CONTRACT_REWRITE_PROMPT_VERSION,
            schema_version=CONTRACT_REWRITE_SCHEMA_VERSION,
            task_family=task_family,
            target_role_cluster=target_role_cluster,
            original_contract=original_contract,
        )

    async def rewrite(
        self,
        original_contract: str,
        *,
        target_role_cluster: RoleCluster,
        task_family: str,
        seed: Optional[int] = None,
    ) -> ContractRewrite:
        request = self.build_request(
            original_contract,
            target_role_cluster=target_role_cluster,
            task_family=task_family,
            seed=seed,
        )
        started_at = time.monotonic()
        try:
            response = await self.completion.complete(request)
            if not isinstance(response, RoleCompletionResponse):
                raise TypeError("completion client returned an invalid response type")
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            receipt = ContractRewriteReceipt(
                request=request,
                status="failed",
                rewritten_contract=None,
                raw_completion=None,
                completion_metadata={},
                latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
                failure_stage="completion",
                error_type=type(exc).__name__,
            )
            raise ContractRewriteError(
                "contract rewrite completion failed; no replacement was produced",
                receipt,
            ) from exc
        try:
            rewritten_contract = parse_rewritten_contract(response.text)
        except (TypeError, ValueError) as exc:
            receipt = ContractRewriteReceipt(
                request=request,
                status="failed",
                rewritten_contract=None,
                raw_completion=response.text,
                completion_metadata=dict(response.metadata),
                latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
                failure_stage="parse",
                error_type=type(exc).__name__,
            )
            raise ContractRewriteError(
                "contract rewrite response failed schema validation; no replacement was produced",
                receipt,
            ) from exc
        receipt = ContractRewriteReceipt(
            request=request,
            status="rewritten",
            rewritten_contract=rewritten_contract,
            raw_completion=response.text,
            completion_metadata=dict(response.metadata),
            latency_ms=max((time.monotonic() - started_at) * 1000.0, 0.0),
        )
        return ContractRewrite(contract=rewritten_contract, receipt=receipt)


class OpenAICompatibleRoleCompletionClient:
    """Small OpenAI-compatible/SGLang chat-completion adapter.

    Network access occurs only when :meth:`complete` is awaited.  The default
    deterministic payload disables Qwen thinking for this constrained
    classification call; it does not alter the Flow-Director policy.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8015/v1",
        model: str = "supervisor_theta",
        api_key_env: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 48,
        disable_thinking: bool = True,
        request_json_object: bool = True,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if timeout_seconds <= 0 or max_retries < 0 or max_tokens <= 0:
            raise ValueError("completion limits are invalid")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        self.disable_thinking = bool(disable_thinking)
        self.request_json_object = bool(request_json_object)

    def request_payload(
        self,
        request: RoleCompletionRequest | ContractRewriteRequest,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in request.messages],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.request_json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(
        self,
        request: RoleCompletionRequest | ContractRewriteRequest,
    ) -> RoleCompletionResponse:
        api_key = "EMPTY"
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env, "")
            if not api_key:
                raise RoleCompletionClientError(
                    f"missing completion credential environment variable: {self.api_key_env}"
                )
        payload = self.request_payload(request)
        url = self.base_url + "/chat/completions"
        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._post_json,
                    url,
                    api_key,
                    payload,
                    request.request_id,
                )
                parsed = self._parse_response(response)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "classifier_request_id": request.request_id,
                        "attempt_count": attempt + 1,
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                    }
                )
                return RoleCompletionResponse(parsed.text, metadata)
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
        detail = f"HTTP {last_error.code}" if isinstance(last_error, HTTPError) else (
            type(last_error).__name__ if last_error is not None else "unknown error"
        )
        raise RoleCompletionClientError(f"role completion request failed: {detail}") from last_error

    def _post_json(
        self,
        url: str,
        api_key: str,
        payload: Mapping[str, Any],
        request_id: str,
    ) -> Mapping[str, Any]:
        http_request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-Role-Classifier/1",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        with urlopen(http_request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RoleCompletionClientError("role completion returned a non-object response")
        return value

    def _parse_response(self, response: Mapping[str, Any]) -> RoleCompletionResponse:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RoleCompletionClientError("role completion response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RoleCompletionClientError("role completion response has no text content")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        return RoleCompletionResponse(
            text=message["content"],
            metadata={
                "provider_request_id": response.get("id"),
                "provider_model": response.get("model", self.model),
                "finish_reason": choices[0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )


__all__ = [
    "AsyncRoleCompletion",
    "AsyncContractRewriteCompletion",
    "CONTRACT_REWRITE_PROMPT_VERSION",
    "CONTRACT_REWRITE_SCHEMA_VERSION",
    "CONTRACT_REWRITE_SYSTEM_PROMPT",
    "CONTRACT_REWRITER_VERSION",
    "ContractRewrite",
    "ContractRewriteError",
    "ContractRewriteReceipt",
    "ContractRewriteRequest",
    "ContractRoleRewriter",
    "OpenAICompatibleRoleCompletionClient",
    "ROLE_CLASSIFIER_PROMPT_VERSION",
    "ROLE_CLASSIFIER_SCHEMA_VERSION",
    "ROLE_CLASSIFIER_SYSTEM_PROMPT",
    "ROLE_CLASSIFIER_VERSION",
    "ROLE_CLUSTERS",
    "RoleClassification",
    "RoleClassificationError",
    "RoleClassificationReceipt",
    "RoleClassifier",
    "RoleCluster",
    "RoleCompletionClientError",
    "RoleCompletionRequest",
    "RoleCompletionResponse",
    "build_role_classification_messages",
    "build_contract_rewrite_messages",
    "parse_rewritten_contract",
    "parse_role_classification",
]
