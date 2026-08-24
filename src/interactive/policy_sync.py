"""Transactional publication of trained Director LoRA adapters to SGLang.

Source classification for this smoke-training module:

* **Direct reuse** -- request retry, managed-adapter draining, candidate load,
  ``/v1/models`` verification, chat canary, route/generation switch ordering,
  old-adapter unload, and failed candidate cleanup follow SkillFlow's
  ``training/external_sglang.py::publish_external_adapter`` and
  ``runtime/sglang_gateway.py::_swap_supervisor_adapter_sync``.
* **Necessary adaptation** -- SkillFlow owns its generation route inside the
  gateway, while FlowSteer keeps it in ``SGLangReceiptDirectorClient``.  An
  optional callback bridges that split inside the same pause/drain transaction;
  receipts also distinguish trained publication from untrained Step0 activation.
* **Project algorithm addition** -- none; this is a runtime publication
  boundary, not a learning algorithm.
* **Not implemented here** -- distributed worker rendezvous and concurrent
  rollout cancellation.  The smoke runner awaits its rollout batch before
  calling this publisher.

No checkpoint hash is computed.  Logical policy version, checkpoint version,
and SGLang adapter name stay separate, matching the project's version
contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@runtime_checkable
class PolicySyncGate(Protocol):
    """Optional admission gate around a policy publication transaction."""

    def pause(self) -> None:
        """Prevent new rollout requests from entering."""

    def drain(self) -> None:
        """Wait until already-admitted rollout requests have completed."""

    def resume(self) -> None:
        """Allow rollout requests to enter again."""


@dataclass(frozen=True)
class PolicySyncConfig:
    """SGLang control-plane settings for one smoke-training publisher."""

    api_base: str = "http://127.0.0.1:8015"
    api_key: str = "EMPTY"
    adapter_name_prefix: str = "theta_smoke_step_"
    request_timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    max_response_bytes: int = 16 * 1024 * 1024
    canary_prompt: str = "Reply with OK."
    canary_max_tokens: int = 1

    def __post_init__(self) -> None:
        parsed = urlsplit(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("api_base must be an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("api_base must not contain credentials")
        if not self.adapter_name_prefix.strip():
            raise ValueError("adapter_name_prefix must be non-empty")
        if any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in self.adapter_name_prefix
        ):
            raise ValueError("adapter_name_prefix contains an unsupported character")
        if self.request_timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise ValueError("timeout must be positive and backoff non-negative")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if not self.canary_prompt.strip() or self.canary_max_tokens <= 0:
            raise ValueError("canary settings must be non-empty and positive")

    @property
    def control_root(self) -> str:
        base = self.api_base.rstrip("/")
        return base.removesuffix("/v1")

    @property
    def openai_root(self) -> str:
        return self.control_root + "/v1"


@dataclass(frozen=True)
class PolicySyncReceipt:
    """Immutable evidence for one attempted behavior-policy transition."""

    behavior_policy_version: str
    candidate_policy_version: str
    new_policy_version: Optional[str]
    adapter_name: str
    previous_adapter: Optional[str]
    checkpoint_version: str
    checkpoint_path: str
    models_before: tuple[str, ...]
    models_after: tuple[str, ...]
    success: bool
    status: str
    canary_succeeded: bool
    rollback_succeeded: Optional[bool]
    gate_used: bool
    gate_drained: bool
    route_switch_requested: bool = False
    route_switch_succeeded: bool = False
    route_rollback_succeeded: Optional[bool] = None
    training_performed: bool = True
    policy_published: bool = True
    request_attempts: Mapping[str, int] = field(default_factory=dict)
    error: str = ""
    started_at: str = field(default_factory=_utc_now)
    completed_at: str = field(default_factory=_utc_now)
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        required = {
            "behavior_policy_version": self.behavior_policy_version,
            "candidate_policy_version": self.candidate_policy_version,
            "adapter_name": self.adapter_name,
            "checkpoint_version": self.checkpoint_version,
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.success and self.new_policy_version != self.candidate_policy_version:
            raise ValueError(
                "successful sync must publish the candidate policy version"
            )
        if not self.success and self.new_policy_version is not None:
            raise ValueError("failed sync cannot claim a new active policy version")
        if (
            self.success
            and self.route_switch_requested
            and not self.route_switch_succeeded
        ):
            raise ValueError("successful sync must complete its requested route switch")
        if self.route_switch_succeeded and not self.route_switch_requested:
            raise ValueError("route switch success requires a requested route switch")
        if (
            type(self.training_performed) is not bool
            or type(self.policy_published) is not bool
        ):
            raise ValueError("training_performed and policy_published must be booleans")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicySyncError(RuntimeError):
    """Publication failed; ``receipt`` describes the rollback outcome."""

    def __init__(self, message: str, receipt: PolicySyncReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class _RequestFailure(RuntimeError):
    pass


class SGLangPolicyPublisher:
    """Publish a versioned LoRA only after SGLang proves it can serve it."""

    def __init__(
        self,
        config: PolicySyncConfig,
        *,
        http: Any = requests,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(config, PolicySyncConfig):
            raise TypeError("config must be PolicySyncConfig")
        self.config = config
        self._http = http
        self._sleep = sleeper

    def adapter_name(self, step: int) -> str:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        return f"{self.config.adapter_name_prefix}{step:06d}"

    def ensure_loaded_adapter(
        self,
        *,
        checkpoint_path: str | Path,
        adapter_name: str,
    ) -> Mapping[str, Any]:
        """Load and canary an existing inference adapter without a policy update.

        This is the evaluation-only subset of SkillFlow's external SGLang
        publication boundary: model list, adapter load, model-list verification,
        and chat canary.  It deliberately does not unload another policy or
        claim that any weights were trained or published.
        """

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("checkpoint_path must be an existing adapter directory")
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise ValueError("adapter_name must be non-empty")
        adapter_name = adapter_name.strip()
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        attempts: dict[str, int] = {}
        before, after, loaded_now = self._load_and_validate_adapter(
            checkpoint=checkpoint,
            adapter_name=adapter_name,
            attempts=attempts,
        )
        return {
            "status": "ready",
            "success": True,
            "training_performed": False,
            "policy_published": False,
            "adapter_name": adapter_name,
            "checkpoint_path": str(checkpoint),
            "models_before": list(before),
            "models_after": list(after),
            "loaded_now": loaded_now,
            "canary_succeeded": True,
            "request_attempts": dict(attempts),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "duration_seconds": max(time.monotonic() - started_monotonic, 0.0),
        }

    def server_runtime_receipt(self) -> Mapping[str, Any]:
        """Read the deployed SGLang batch/sampling boundary.

        SGLang exposes these resolved values through its public
        ``GET /server_info`` endpoint.  Evaluation records only the fields that
        can change request scheduling or deterministic decoding; it does not
        copy the full server configuration into a trajectory.
        """

        attempts: dict[str, int] = {}
        response = self._request(
            "get",
            "/server_info",
            operation="server_info",
            attempts=attempts,
        )
        payload = self._json_object(response, "SGLang server-info response")
        required_integer_fields = (
            "context_length",
            "max_running_requests",
            "max_total_num_tokens",
        )
        for field_name in required_integer_fields:
            value = payload.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _RequestFailure(
                    f"SGLang server-info {field_name} must be a positive integer"
                )
        deterministic = payload.get("enable_deterministic_inference")
        if not isinstance(deterministic, bool):
            raise _RequestFailure(
                "SGLang server-info enable_deterministic_inference must be boolean"
            )
        for field_name in ("attention_backend", "sampling_backend"):
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise _RequestFailure(
                    f"SGLang server-info {field_name} must be non-empty"
                )
        return {
            "schema_version": "flowsteer.sglang.server-runtime-receipt.v1",
            "context_length": int(payload["context_length"]),
            "max_running_requests": int(payload["max_running_requests"]),
            "max_total_num_tokens": int(payload["max_total_num_tokens"]),
            "enable_deterministic_inference": deterministic,
            "sampling_backend": str(payload["sampling_backend"]),
            "attention_backend": str(payload["attention_backend"]),
            "cuda_graph_backend_decode": payload.get(
                "cuda_graph_backend_decode"
            ),
            "weight_version": payload.get("weight_version"),
            "request_attempts": dict(attempts),
            "recorded_at": _utc_now(),
        }

    def activate_existing_policy(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_version: str,
        behavior_policy_version: str,
        active_policy_version: str,
        adapter_name: str,
        previous_adapter: Optional[str],
        gate: PolicySyncGate,
        route_switch: Callable[[str, str], None],
        route_rollback: Callable[[], None],
    ) -> PolicySyncReceipt:
        """Activate an already-materialized, untrained adapter transactionally.

        Formal Step0 needs the same SkillFlow Supervisor swap boundary as a
        trained update without claiming that an optimizer or policy publication
        occurred.  Candidate load, model-list verification, canary, Director
        route switch, and old-adapter unload all happen under one pause/drain.
        """

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("checkpoint_path must be an existing adapter directory")
        versions = {
            "checkpoint_version": checkpoint_version,
            "behavior_policy_version": behavior_policy_version,
            "active_policy_version": active_policy_version,
            "adapter_name": adapter_name,
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in versions.values()
        ):
            raise ValueError("checkpoint, policy, and adapter names must be non-empty")
        if active_policy_version == behavior_policy_version:
            raise ValueError("active policy version must differ from behavior policy")
        adapter_name = adapter_name.strip()
        if previous_adapter is not None:
            previous_adapter = previous_adapter.strip()
            if not previous_adapter:
                raise ValueError("previous_adapter must be non-empty when provided")
        if previous_adapter == adapter_name:
            raise ValueError("active adapter must differ from the previous adapter")

        started_at = _utc_now()
        started_monotonic = time.monotonic()
        attempts: dict[str, int] = {}
        models_before: tuple[str, ...] = ()
        models_after: tuple[str, ...] = ()
        loaded_now = False
        canary_succeeded = False
        rollback_succeeded: Optional[bool] = None
        route_switch_attempted = False
        route_switch_succeeded = False
        route_rollback_succeeded: Optional[bool] = None
        gate_paused = False
        gate_drained = False

        try:
            gate.pause()
            gate_paused = True
            gate.drain()
            gate_drained = True
            models_before = self._model_ids(
                attempts,
                "activation_models_before",
            )
            self._drain_stale_managed_adapters(
                models_before,
                candidate=adapter_name,
                previous_adapter=previous_adapter,
                attempts=attempts,
                remove_candidate_collision=False,
            )
            _, candidate_models, loaded_now = self._load_and_validate_adapter(
                checkpoint=checkpoint,
                adapter_name=adapter_name,
                attempts=attempts,
                operation_prefix="candidate_",
            )
            canary_succeeded = True

            route_switch_attempted = True
            route_switch(active_policy_version, adapter_name)
            route_switch_succeeded = True
            if previous_adapter is not None and previous_adapter in candidate_models:
                self._unload(previous_adapter, attempts, "unload_previous")
                models_after = tuple(
                    model for model in candidate_models if model != previous_adapter
                )
            else:
                models_after = candidate_models

            return PolicySyncReceipt(
                behavior_policy_version=behavior_policy_version,
                candidate_policy_version=active_policy_version,
                new_policy_version=active_policy_version,
                adapter_name=adapter_name,
                previous_adapter=previous_adapter,
                checkpoint_version=checkpoint_version,
                checkpoint_path=str(checkpoint),
                models_before=models_before,
                models_after=models_after,
                success=True,
                status="activated_existing",
                canary_succeeded=True,
                rollback_succeeded=None,
                gate_used=True,
                gate_drained=gate_drained,
                route_switch_requested=True,
                route_switch_succeeded=True,
                route_rollback_succeeded=None,
                training_performed=False,
                policy_published=False,
                request_attempts=dict(attempts),
                started_at=started_at,
                completed_at=_utc_now(),
                duration_seconds=max(time.monotonic() - started_monotonic, 0.0),
            )
        except Exception as error:
            rollback_error = ""
            if route_switch_attempted:
                try:
                    route_rollback()
                    route_rollback_succeeded = True
                except Exception as route_error:  # pragma: no cover - rare double fault
                    route_rollback_succeeded = False
                    rollback_error = f"; route rollback failed: {route_error}"
                    logger.exception("failed to restore previous Director route")
            if loaded_now and route_rollback_succeeded is not False:
                try:
                    self._unload(
                        adapter_name,
                        attempts,
                        "rollback_candidate",
                        retry=False,
                    )
                    rollback_succeeded = True
                except Exception as candidate_error:  # pragma: no cover
                    rollback_succeeded = False
                    rollback_error += f"; candidate rollback failed: {candidate_error}"
                    logger.exception("failed to unload rejected initial adapter")
            elif loaded_now:
                rollback_succeeded = False
                rollback_error += "; candidate retained because route rollback failed"
            try:
                models_after = self._model_ids(
                    attempts,
                    "models_after_rollback",
                    retry=False,
                )
            except Exception as model_error:
                rollback_error += f"; post-rollback model query failed: {model_error}"

            receipt = PolicySyncReceipt(
                behavior_policy_version=behavior_policy_version,
                candidate_policy_version=active_policy_version,
                new_policy_version=None,
                adapter_name=adapter_name,
                previous_adapter=previous_adapter,
                checkpoint_version=checkpoint_version,
                checkpoint_path=str(checkpoint),
                models_before=models_before,
                models_after=models_after,
                success=False,
                status=(
                    "rollback_failed"
                    if (
                        rollback_succeeded is False or route_rollback_succeeded is False
                    )
                    else "rolled_back"
                    if (rollback_succeeded is True or route_rollback_succeeded is True)
                    else "failed_before_load"
                ),
                canary_succeeded=canary_succeeded,
                rollback_succeeded=rollback_succeeded,
                gate_used=True,
                gate_drained=gate_drained,
                route_switch_requested=True,
                route_switch_succeeded=route_switch_succeeded,
                route_rollback_succeeded=route_rollback_succeeded,
                training_performed=False,
                policy_published=False,
                request_attempts=dict(attempts),
                error=f"{type(error).__name__}: {error}{rollback_error}",
                started_at=started_at,
                completed_at=_utc_now(),
                duration_seconds=max(time.monotonic() - started_monotonic, 0.0),
            )
            raise PolicySyncError(
                "existing SGLang policy activation failed",
                receipt,
            ) from error
        finally:
            if gate_paused:
                gate.resume()

    def _load_and_validate_adapter(
        self,
        *,
        checkpoint: Path,
        adapter_name: str,
        attempts: dict[str, int],
        operation_prefix: str = "",
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        """Reuse SkillFlow's load/models/canary candidate validation boundary."""

        before = self._model_ids(attempts, f"{operation_prefix}models_before")
        loaded_now = adapter_name not in before
        try:
            if loaded_now:
                self._expect_success_flag(
                    self._request(
                        "post",
                        "/load_lora_adapter",
                        operation=f"{operation_prefix}load_adapter",
                        attempts=attempts,
                        json={
                            "lora_name": adapter_name,
                            "lora_path": str(checkpoint),
                        },
                    ),
                    "external SGLang rejected the inference adapter",
                )
            after = self._model_ids(attempts, f"{operation_prefix}models_after")
            if adapter_name not in after:
                raise _RequestFailure("inference adapter is absent from /v1/models")
            validation = self._request(
                "post",
                "/v1/chat/completions",
                operation=f"{operation_prefix}canary",
                attempts=attempts,
                json={
                    "max_tokens": self.config.canary_max_tokens,
                    "messages": [
                        {"content": self.config.canary_prompt, "role": "user"}
                    ],
                    "model": adapter_name,
                    "temperature": 0,
                },
            )
            choices = self._json_object(validation, "canary response").get("choices")
            if not isinstance(choices, list) or not choices:
                raise _RequestFailure("inference adapter failed its chat canary")
        except Exception:
            if loaded_now:
                try:
                    self._unload(
                        adapter_name,
                        attempts,
                        f"{operation_prefix}rollback_adapter",
                        retry=False,
                    )
                except Exception:  # pragma: no cover - best-effort failure cleanup
                    logger.exception("failed to unload rejected inference adapter")
            raise
        return before, after, loaded_now

    def publish(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_version: str,
        behavior_policy_version: str,
        candidate_policy_version: str,
        step: int,
        previous_adapter: Optional[str] = None,
        gate: Optional[PolicySyncGate] = None,
        route_switch: Optional[Callable[[str, str], None]] = None,
        route_rollback: Optional[Callable[[], None]] = None,
    ) -> PolicySyncReceipt:
        """Load, verify, canary, and commit one trained adapter.

        When ``route_switch`` and ``route_rollback`` are supplied, the
        publisher mirrors SkillFlow's internal generation swap while the same
        gate remains paused and drained: validate the candidate, switch the
        client route, unload the previous adapter, and only then resume.  On
        failure the route is restored before the rejected candidate is
        removed.  Omitting both callbacks preserves the standalone publisher
        interface used by existing callers.
        """

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("checkpoint_path must be an existing adapter directory")
        versions = {
            "checkpoint_version": checkpoint_version,
            "behavior_policy_version": behavior_policy_version,
            "candidate_policy_version": candidate_policy_version,
        }
        if any(not value.strip() for value in versions.values()):
            raise ValueError("checkpoint and policy versions must be non-empty")
        if candidate_policy_version == behavior_policy_version:
            raise ValueError(
                "candidate policy version must differ from behavior policy"
            )
        if previous_adapter is not None and not previous_adapter.strip():
            raise ValueError("previous_adapter must be non-empty when provided")
        if (route_switch is None) != (route_rollback is None):
            raise ValueError(
                "route_switch and route_rollback must be supplied together"
            )
        if route_switch is not None and gate is None:
            raise ValueError("a route switch requires a publication gate")

        candidate = self.adapter_name(step)
        if previous_adapter == candidate:
            raise ValueError("candidate adapter must differ from the previous adapter")

        started_at = _utc_now()
        started_monotonic = time.monotonic()
        attempts: dict[str, int] = {}
        models_before: tuple[str, ...] = ()
        models_after: tuple[str, ...] = ()
        loaded = False
        canary_succeeded = False
        rollback_succeeded: Optional[bool] = None
        route_switch_attempted = False
        route_switch_succeeded = False
        route_rollback_succeeded: Optional[bool] = None
        gate_drained = False
        gate_paused = False

        try:
            if gate is not None:
                gate.pause()
                gate_paused = True
                gate.drain()
                gate_drained = True

            models_before = self._model_ids(attempts, "models_before")
            self._drain_stale_managed_adapters(
                models_before,
                candidate=candidate,
                previous_adapter=previous_adapter,
                attempts=attempts,
            )

            self._expect_success_flag(
                self._request(
                    "post",
                    "/load_lora_adapter",
                    operation="load_candidate",
                    attempts=attempts,
                    json={"lora_name": candidate, "lora_path": str(checkpoint)},
                ),
                "external SGLang rejected the candidate adapter",
            )
            loaded = True

            candidate_models = self._model_ids(attempts, "models_candidate")
            if candidate not in candidate_models:
                raise _RequestFailure("candidate adapter is absent from /v1/models")

            validation = self._request(
                "post",
                "/v1/chat/completions",
                operation="canary",
                attempts=attempts,
                json={
                    "max_tokens": self.config.canary_max_tokens,
                    "messages": [
                        {"content": self.config.canary_prompt, "role": "user"}
                    ],
                    "model": candidate,
                    "temperature": 0,
                },
            )
            payload = self._json_object(validation, "canary response")
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise _RequestFailure("candidate adapter failed its chat canary")
            canary_succeeded = True

            # SkillFlow updates its in-gateway AdapterGeneration at this exact
            # boundary: after load/models/canary and before unloading the old
            # adapter.  FlowSteer's Director route lives in a separate client,
            # so inject that same state transition without opening the gate.
            if route_switch is not None:
                route_switch_attempted = True
                route_switch(candidate_policy_version, candidate)
                route_switch_succeeded = True

            # The previous behavior adapter remains available until the new
            # candidate has appeared, answered canary, and become the Director
            # route.  No rollout can enter until the finally block resumes.
            if previous_adapter is not None and previous_adapter in candidate_models:
                self._unload(previous_adapter, attempts, "unload_previous")
                # All fallible verification happens before this irreversible
                # commit boundary.  SGLang has acknowledged the old unload;
                # derive the final set from the immediately preceding queried
                # model list so a later diagnostic request cannot strand the
                # service with both the old and candidate adapters removed.
                models_after = tuple(
                    model for model in candidate_models if model != previous_adapter
                )
            else:
                models_after = candidate_models

            return PolicySyncReceipt(
                behavior_policy_version=behavior_policy_version,
                candidate_policy_version=candidate_policy_version,
                new_policy_version=candidate_policy_version,
                adapter_name=candidate,
                previous_adapter=previous_adapter,
                checkpoint_version=checkpoint_version,
                checkpoint_path=str(checkpoint),
                models_before=models_before,
                models_after=models_after,
                success=True,
                status="published",
                canary_succeeded=True,
                rollback_succeeded=None,
                gate_used=gate is not None,
                gate_drained=gate_drained,
                route_switch_requested=route_switch is not None,
                route_switch_succeeded=route_switch_succeeded,
                route_rollback_succeeded=None,
                request_attempts=dict(attempts),
                started_at=started_at,
                completed_at=_utc_now(),
                duration_seconds=max(time.monotonic() - started_monotonic, 0.0),
            )
        except Exception as error:
            rollback_error = ""
            # Match SkillFlow's rollback order: restore the previous logical
            # generation first, then remove the failed candidate.  If route
            # restoration itself fails, leave the candidate resident so the
            # client cannot be resumed while pointing at an unloaded adapter.
            if route_switch_attempted and route_rollback is not None:
                try:
                    route_rollback()
                    route_rollback_succeeded = True
                except Exception as route_error:  # pragma: no cover - rare double fault
                    route_rollback_succeeded = False
                    rollback_error = f"; route rollback failed: {route_error}"
                    logger.exception("failed to restore previous Director route")
            if loaded and route_rollback_succeeded is not False:
                try:
                    self._unload(candidate, attempts, "rollback_candidate", retry=False)
                    rollback_succeeded = True
                except (
                    Exception
                ) as candidate_error:  # pragma: no cover - rare double fault
                    rollback_succeeded = False
                    rollback_error = f"; candidate rollback failed: {candidate_error}"
                    logger.exception("failed to unload rejected SGLang adapter")
            elif loaded:
                rollback_succeeded = False
                rollback_error += "; candidate retained because route rollback failed"
            try:
                models_after = self._model_ids(
                    attempts,
                    "models_after_rollback",
                    retry=False,
                )
            except Exception as model_error:
                rollback_error += f"; post-rollback model query failed: {model_error}"

            message = f"{type(error).__name__}: {error}{rollback_error}"
            receipt = PolicySyncReceipt(
                behavior_policy_version=behavior_policy_version,
                candidate_policy_version=candidate_policy_version,
                new_policy_version=None,
                adapter_name=candidate,
                previous_adapter=previous_adapter,
                checkpoint_version=checkpoint_version,
                checkpoint_path=str(checkpoint),
                models_before=models_before,
                models_after=models_after,
                success=False,
                status=(
                    "rollback_failed"
                    if (
                        rollback_succeeded is False or route_rollback_succeeded is False
                    )
                    else "rolled_back"
                    if (rollback_succeeded is True or route_rollback_succeeded is True)
                    else "failed_before_load"
                ),
                canary_succeeded=canary_succeeded,
                rollback_succeeded=rollback_succeeded,
                gate_used=gate is not None,
                gate_drained=gate_drained,
                route_switch_requested=route_switch is not None,
                route_switch_succeeded=route_switch_succeeded,
                route_rollback_succeeded=route_rollback_succeeded,
                request_attempts=dict(attempts),
                error=message,
                started_at=started_at,
                completed_at=_utc_now(),
                duration_seconds=max(time.monotonic() - started_monotonic, 0.0),
            )
            raise PolicySyncError(
                "SGLang policy publication failed", receipt
            ) from error
        finally:
            if gate is not None and gate_paused:
                gate.resume()

    def _drain_stale_managed_adapters(
        self,
        models: tuple[str, ...],
        *,
        candidate: str,
        previous_adapter: Optional[str],
        attempts: dict[str, int],
        remove_candidate_collision: bool = True,
    ) -> None:
        stale = sorted(
            model
            for model in models
            if model.startswith(self.config.adapter_name_prefix)
            and model not in {candidate, previous_adapter}
        )
        for index, adapter in enumerate(stale):
            self._unload(adapter, attempts, f"unload_stale_{index}")
        # A retried step may have left an uncommitted name behind.  The caller
        # cannot also declare it as the previous behavior adapter (checked
        # above), so removing the collision is safe and follows SkillFlow.
        if remove_candidate_collision and candidate in models:
            self._unload(candidate, attempts, "unload_candidate_collision")

    def _model_ids(
        self,
        attempts: dict[str, int],
        operation: str,
        *,
        retry: bool = True,
    ) -> tuple[str, ...]:
        response = self._request(
            "get",
            "/v1/models",
            operation=operation,
            attempts=attempts,
            retry=retry,
        )
        payload = self._json_object(response, "model-list response")
        data = payload.get("data")
        if not isinstance(data, list):
            raise _RequestFailure("SGLang model list is malformed")
        values: set[str] = set()
        for item in data:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                raise _RequestFailure("SGLang model list contains a malformed entry")
            values.add(str(item["id"]))
        return tuple(sorted(values))

    def _unload(
        self,
        adapter_name: str,
        attempts: dict[str, int],
        operation: str,
        *,
        retry: bool = True,
    ) -> None:
        response = self._request(
            "post",
            "/unload_lora_adapter",
            operation=operation,
            attempts=attempts,
            retry=retry,
            json={"lora_name": adapter_name},
        )
        self._expect_success_flag(
            response,
            f"external SGLang rejected unload of {adapter_name}",
        )

    def _request(
        self,
        method_name: str,
        path: str,
        *,
        operation: str,
        attempts: dict[str, int],
        retry: bool = True,
        **kwargs: object,
    ) -> Any:
        method = getattr(self._http, method_name)
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        limit = self.config.max_retries + 1 if retry else 1
        response: Any = None
        for index in range(limit):
            attempts[operation] = index + 1
            try:
                response = method(
                    self.config.control_root + path,
                    headers=headers,
                    timeout=self.config.request_timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                content = getattr(response, "content", b"")
                if content and len(content) > self.config.max_response_bytes:
                    raise _RequestFailure("SGLang response exceeded its byte limit")
                return response
            except requests.RequestException:
                status = getattr(response, "status_code", None)
                final = index + 1 >= limit
                if final or (isinstance(status, int) and status < 500):
                    raise
                self._sleep(self.config.retry_backoff_seconds * (2**index))
        raise AssertionError("unreachable")

    @staticmethod
    def _json_object(response: Any, label: str) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            raise _RequestFailure(f"{label} is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise _RequestFailure(f"{label} is not a JSON object")
        return payload

    @classmethod
    def _expect_success_flag(cls, response: Any, message: str) -> None:
        if (
            cls._json_object(response, "adapter control response").get("success")
            is not True
        ):
            raise _RequestFailure(message)


__all__ = [
    "PolicySyncConfig",
    "PolicySyncError",
    "PolicySyncGate",
    "PolicySyncReceipt",
    "SGLangPolicyPublisher",
]
