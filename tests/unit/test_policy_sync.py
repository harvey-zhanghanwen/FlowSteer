from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any

import requests

from src.interactive.policy_sync import (
    PolicySyncConfig,
    PolicySyncError,
    SGLangPolicyPublisher,
)


class _Response:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = b"{}"

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _SGLangControl:
    def __init__(
        self,
        adapters: set[str],
        *,
        max_running_requests: int | None = 4,
    ) -> None:
        self.adapters = set(adapters)
        self.max_running_requests = max_running_requests
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_canary = False
        self.models_failures_remaining = 0
        self.rejected_unloads: set[str] = set()

    def get(self, url: str, **kwargs: Any) -> _Response:
        operation = url.removesuffix("/").rsplit("/", 1)[-1]
        self.calls.append(("get", operation, kwargs))
        if operation == "server_info":
            return _Response(
                {
                    "context_length": 32768,
                    "max_running_requests": self.max_running_requests,
                    "max_total_num_tokens": 717868,
                    "enable_deterministic_inference": True,
                    "sampling_backend": "pytorch",
                    "attention_backend": "fa3",
                    "cuda_graph_backend_decode": "disabled",
                    "weight_version": "default",
                }
            )
        if self.models_failures_remaining:
            self.models_failures_remaining -= 1
            return _Response({}, status_code=503)
        return _Response(
            {
                "data": [
                    {"id": "supervisor_theta"},
                    *({"id": name} for name in sorted(self.adapters)),
                ]
            }
        )

    def post(self, url: str, **kwargs: Any) -> _Response:
        operation = url.removesuffix("/").rsplit("/", 1)[-1]
        self.calls.append(("post", operation, kwargs))
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        if operation == "load_lora_adapter":
            self.adapters.add(str(payload["lora_name"]))
            return _Response({"success": True})
        if operation == "unload_lora_adapter":
            adapter = str(payload["lora_name"])
            if adapter in self.rejected_unloads:
                return _Response({"success": False})
            self.adapters.discard(adapter)
            return _Response({"success": True})
        assert operation == "completions"
        return _Response({"choices": [] if self.fail_canary else [{"message": {}}]})


class _Gate:
    def __init__(self) -> None:
        self.events: list[str] = []

    def pause(self) -> None:
        self.events.append("pause")

    def drain(self) -> None:
        self.events.append("drain")

    def resume(self) -> None:
        self.events.append("resume")


class PolicySyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.tempdir.name) / "supervisor_lora"
        self.checkpoint.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def publisher(
        self,
        control: _SGLangControl,
        *,
        sleeps: list[float] | None = None,
    ) -> SGLangPolicyPublisher:
        config = PolicySyncConfig(
            api_base="http://127.0.0.1:8015/v1",
            max_retries=2,
            retry_backoff_seconds=0.5,
        )
        return SGLangPolicyPublisher(
            config,
            http=control,
            sleeper=(sleeps.append if sleeps is not None else lambda _: None),
        )

    def test_candidate_is_canaried_before_previous_adapter_is_unloaded(self) -> None:
        previous = "theta_smoke_step_000000"
        stale = "theta_smoke_step_999999"
        control = _SGLangControl({previous, stale})
        gate = _Gate()
        route = {"policy": previous, "adapter": previous}

        def switch_route(policy_version: str, adapter_name: str) -> None:
            self.assertEqual(gate.events, ["pause", "drain"])
            self.assertIn(previous, control.adapters)
            self.assertIn(adapter_name, control.adapters)
            route.update(policy=policy_version, adapter=adapter_name)

        def restore_route() -> None:
            route.update(policy=previous, adapter=previous)

        receipt = self.publisher(control).publish(
            checkpoint_path=self.checkpoint,
            checkpoint_version="checkpoint-step-1",
            behavior_policy_version=previous,
            candidate_policy_version="qwen35-9b-smoke-step-0001",
            step=1,
            previous_adapter=previous,
            gate=gate,
            route_switch=switch_route,
            route_rollback=restore_route,
        )

        self.assertTrue(receipt.success)
        self.assertEqual(receipt.new_policy_version, "qwen35-9b-smoke-step-0001")
        self.assertEqual(
            receipt.candidate_policy_version,
            "qwen35-9b-smoke-step-0001",
        )
        self.assertEqual(receipt.adapter_name, "theta_smoke_step_000001")
        self.assertEqual(receipt.checkpoint_version, "checkpoint-step-1")
        self.assertEqual(receipt.models_before, ("supervisor_theta", previous, stale))
        self.assertEqual(
            receipt.models_after,
            ("supervisor_theta", "theta_smoke_step_000001"),
        )
        self.assertEqual(control.adapters, {"theta_smoke_step_000001"})
        self.assertEqual(gate.events, ["pause", "drain", "resume"])
        self.assertEqual(
            route,
            {
                "policy": "qwen35-9b-smoke-step-0001",
                "adapter": "theta_smoke_step_000001",
            },
        )
        self.assertTrue(receipt.route_switch_requested)
        self.assertTrue(receipt.route_switch_succeeded)
        self.assertIsNone(receipt.route_rollback_succeeded)
        self.assertTrue(receipt.training_performed)
        self.assertTrue(receipt.policy_published)

        operations = [(method, operation) for method, operation, _ in control.calls]
        canary_index = operations.index(("post", "completions"))
        previous_unload_index = max(
            index
            for index, (method, operation, kwargs) in enumerate(control.calls)
            if method == "post"
            and operation == "unload_lora_adapter"
            and kwargs["json"]["lora_name"] == previous
        )
        self.assertLess(canary_index, previous_unload_index)
        canary_payload = control.calls[canary_index][2]["json"]
        self.assertEqual(canary_payload["model"], "theta_smoke_step_000001")

    def test_existing_evaluation_adapter_load_has_no_policy_publication(self) -> None:
        control = _SGLangControl(set())
        receipt = self.publisher(control).ensure_loaded_adapter(
            checkpoint_path=self.checkpoint,
            adapter_name="theta_smoke_step_000001",
        )

        self.assertTrue(receipt["success"])
        self.assertTrue(receipt["loaded_now"])
        self.assertFalse(receipt["training_performed"])
        self.assertFalse(receipt["policy_published"])
        self.assertIn("theta_smoke_step_000001", control.adapters)
        operations = [(method, operation) for method, operation, _ in control.calls]
        self.assertEqual(
            operations,
            [
                ("get", "models"),
                ("post", "load_lora_adapter"),
                ("get", "models"),
                ("post", "completions"),
            ],
        )

    def test_server_runtime_receipt_records_batch_invariant_boundary(self) -> None:
        control = _SGLangControl(set())

        receipt = self.publisher(control).server_runtime_receipt()

        self.assertEqual(
            receipt["schema_version"],
            "flowsteer.sglang.server-runtime-receipt.v1",
        )
        self.assertEqual(receipt["max_running_requests"], 4)
        self.assertEqual(receipt["max_total_num_tokens"], 717868)
        self.assertTrue(receipt["enable_deterministic_inference"])
        self.assertEqual(receipt["sampling_backend"], "pytorch")
        self.assertEqual(receipt["attention_backend"], "fa3")
        self.assertEqual(receipt["request_attempts"], {"server_info": 1})

    def test_server_runtime_receipt_accepts_automatic_request_scheduling(
        self,
    ) -> None:
        control = _SGLangControl(set(), max_running_requests=None)

        receipt = self.publisher(control).server_runtime_receipt()

        self.assertEqual(
            receipt["schema_version"],
            "flowsteer.sglang.server-runtime-receipt.v2",
        )
        self.assertIsNone(receipt["max_running_requests"])
        self.assertEqual(receipt["max_running_requests_mode"], "auto")

    def test_failed_canary_unloads_candidate_and_preserves_behavior_adapter(
        self,
    ) -> None:
        previous = "theta_smoke_step_000000"
        control = _SGLangControl({previous})
        control.fail_canary = True

        with self.assertRaises(PolicySyncError) as caught:
            self.publisher(control).publish(
                checkpoint_path=self.checkpoint,
                checkpoint_version="checkpoint-step-1",
                behavior_policy_version=previous,
                candidate_policy_version="qwen35-9b-smoke-step-0001",
                step=1,
                previous_adapter=previous,
            )

        receipt = caught.exception.receipt
        self.assertFalse(receipt.success)
        self.assertIsNone(receipt.new_policy_version)
        self.assertFalse(receipt.canary_succeeded)
        self.assertTrue(receipt.rollback_succeeded)
        self.assertEqual(receipt.status, "rolled_back")
        self.assertEqual(control.adapters, {previous})
        self.assertEqual(receipt.models_after, ("supervisor_theta", previous))

    def test_transient_model_query_uses_configured_exponential_backoff(self) -> None:
        control = _SGLangControl(set())
        control.models_failures_remaining = 2
        sleeps: list[float] = []

        receipt = self.publisher(control, sleeps=sleeps).publish(
            checkpoint_path=self.checkpoint,
            checkpoint_version="checkpoint-step-1",
            behavior_policy_version="supervisor_theta_initial",
            candidate_policy_version="qwen35-9b-smoke-step-0001",
            step=1,
        )

        self.assertTrue(receipt.success)
        self.assertEqual(sleeps, [0.5, 1.0])
        self.assertEqual(receipt.request_attempts["models_before"], 3)

    def test_old_adapter_unload_failure_rolls_back_validated_candidate(self) -> None:
        previous = "theta_smoke_step_000000"
        control = _SGLangControl({previous})
        control.rejected_unloads.add(previous)
        gate = _Gate()
        route = {"policy": "qwen35-9b-smoke-step-0000", "adapter": previous}

        def switch_route(policy_version: str, adapter_name: str) -> None:
            route.update(policy=policy_version, adapter=adapter_name)

        def restore_route() -> None:
            self.assertEqual(gate.events, ["pause", "drain"])
            self.assertIn(previous, control.adapters)
            route.update(policy="qwen35-9b-smoke-step-0000", adapter=previous)

        with self.assertRaises(PolicySyncError) as caught:
            self.publisher(control).publish(
                checkpoint_path=self.checkpoint,
                checkpoint_version="checkpoint-step-1",
                behavior_policy_version="qwen35-9b-smoke-step-0000",
                candidate_policy_version="qwen35-9b-smoke-step-0001",
                step=1,
                previous_adapter=previous,
                gate=gate,
                route_switch=switch_route,
                route_rollback=restore_route,
            )

        receipt = caught.exception.receipt
        self.assertTrue(receipt.canary_succeeded)
        self.assertTrue(receipt.rollback_succeeded)
        self.assertIsNone(receipt.new_policy_version)
        self.assertEqual(control.adapters, {previous})
        self.assertTrue(receipt.route_switch_succeeded)
        self.assertTrue(receipt.route_rollback_succeeded)
        self.assertEqual(
            route,
            {"policy": "qwen35-9b-smoke-step-0000", "adapter": previous},
        )
        self.assertEqual(gate.events, ["pause", "drain", "resume"])

    def test_existing_initial_adapter_activation_is_not_training_publication(
        self,
    ) -> None:
        previous = "theta_smoke_step_000001"
        initial = "theta_initial_step_000000"
        # Both slots may already be occupied when formal Step0 attaches the
        # deterministic initial adapter; activation must not load it twice.
        control = _SGLangControl({previous, initial})
        gate = _Gate()
        route = {"policy": "warm-policy", "adapter": previous}

        def switch_route(policy_version: str, adapter_name: str) -> None:
            self.assertEqual(gate.events, ["pause", "drain"])
            self.assertEqual(control.adapters, {previous, initial})
            route.update(policy=policy_version, adapter=adapter_name)

        receipt = self.publisher(control).activate_existing_policy(
            checkpoint_path=self.checkpoint,
            checkpoint_version="initial-checkpoint-v1",
            behavior_policy_version="warm-policy",
            active_policy_version="qwen35-9b-initial-step-0000",
            adapter_name=initial,
            previous_adapter=previous,
            gate=gate,
            route_switch=switch_route,
            route_rollback=lambda: route.update(
                policy="warm-policy",
                adapter=previous,
            ),
        )

        self.assertTrue(receipt.success)
        self.assertEqual(receipt.status, "activated_existing")
        self.assertFalse(receipt.training_performed)
        self.assertFalse(receipt.policy_published)
        self.assertTrue(receipt.route_switch_succeeded)
        self.assertEqual(control.adapters, {initial})
        self.assertEqual(
            route,
            {"policy": "qwen35-9b-initial-step-0000", "adapter": initial},
        )
        self.assertEqual(gate.events, ["pause", "drain", "resume"])
        candidate_loads = [
            kwargs["json"]["lora_name"]
            for method, operation, kwargs in control.calls
            if method == "post" and operation == "load_lora_adapter"
        ]
        self.assertEqual(candidate_loads, [])

    def test_existing_initial_checkpoint_loads_then_releases_warm_adapter(
        self,
    ) -> None:
        previous = "theta_smoke_step_000001"
        initial = "theta_hotpot_step_000000"
        control = _SGLangControl({previous})
        gate = _Gate()
        route = {"policy": "warm-policy", "adapter": previous}

        receipt = self.publisher(control).activate_existing_policy(
            checkpoint_path=self.checkpoint,
            checkpoint_version="initial-checkpoint-v1",
            behavior_policy_version="warm-policy",
            active_policy_version="qwen35-9b-hotpot-step-000000",
            adapter_name=initial,
            previous_adapter=previous,
            gate=gate,
            route_switch=lambda policy, adapter: route.update(
                policy=policy,
                adapter=adapter,
            ),
            route_rollback=lambda: route.update(
                policy="warm-policy",
                adapter=previous,
            ),
        )

        self.assertTrue(receipt.success)
        self.assertFalse(receipt.training_performed)
        self.assertFalse(receipt.policy_published)
        self.assertEqual(control.adapters, {initial})
        self.assertEqual(
            route,
            {"policy": "qwen35-9b-hotpot-step-000000", "adapter": initial},
        )
        loads = [
            kwargs["json"]["lora_name"]
            for method, operation, kwargs in control.calls
            if method == "post" and operation == "load_lora_adapter"
        ]
        self.assertEqual(loads, [initial])
        self.assertEqual(gate.events, ["pause", "drain", "resume"])

    def test_existing_initial_activation_rolls_route_back_before_candidate_cleanup(
        self,
    ) -> None:
        previous = "theta_smoke_step_000001"
        initial = "theta_initial_step_000000"
        control = _SGLangControl({previous})
        control.rejected_unloads.add(previous)
        gate = _Gate()
        route = {"policy": "warm-policy", "adapter": previous}

        def switch_route(policy_version: str, adapter_name: str) -> None:
            route.update(policy=policy_version, adapter=adapter_name)

        def restore_route() -> None:
            self.assertIn(initial, control.adapters)
            route.update(policy="warm-policy", adapter=previous)

        with self.assertRaises(PolicySyncError) as caught:
            self.publisher(control).activate_existing_policy(
                checkpoint_path=self.checkpoint,
                checkpoint_version="initial-checkpoint-v1",
                behavior_policy_version="warm-policy",
                active_policy_version="qwen35-9b-initial-step-0000",
                adapter_name=initial,
                previous_adapter=previous,
                gate=gate,
                route_switch=switch_route,
                route_rollback=restore_route,
            )

        receipt = caught.exception.receipt
        self.assertFalse(receipt.success)
        self.assertFalse(receipt.training_performed)
        self.assertFalse(receipt.policy_published)
        self.assertTrue(receipt.route_rollback_succeeded)
        self.assertTrue(receipt.rollback_succeeded)
        self.assertEqual(control.adapters, {previous})
        self.assertEqual(route, {"policy": "warm-policy", "adapter": previous})
        self.assertEqual(gate.events, ["pause", "drain", "resume"])

    def test_receipt_contains_versions_and_times_but_no_checkpoint_hash(self) -> None:
        control = _SGLangControl(set())
        receipt = self.publisher(control).publish(
            checkpoint_path=self.checkpoint,
            checkpoint_version="adapter-checkpoint-v1",
            behavior_policy_version="behavior-v0",
            candidate_policy_version="qwen35-9b-smoke-step-0007",
            step=7,
        )

        value = receipt.to_dict()
        self.assertEqual(value["candidate_policy_version"], "qwen35-9b-smoke-step-0007")
        self.assertEqual(value["adapter_name"], "theta_smoke_step_000007")
        self.assertEqual(value["checkpoint_path"], str(self.checkpoint.resolve()))
        self.assertTrue(value["started_at"])
        self.assertTrue(value["completed_at"])
        self.assertGreaterEqual(value["duration_seconds"], 0.0)
        self.assertNotIn("hash", " ".join(value))


if __name__ == "__main__":
    unittest.main()
