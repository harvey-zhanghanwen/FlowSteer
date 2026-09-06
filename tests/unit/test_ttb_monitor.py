from __future__ import annotations

from copy import deepcopy

import pytest

from src.interactive.ttb_monitor import (
    TTB_MONITOR_PROVENANCE,
    TTB_MONITOR_SCHEMA_VERSION,
    TTBMonitoringError,
    WandbTTBMonitor,
    WandbTTBMonitorConfig,
    metrics_from_ttb_step_receipt,
    ttb_monitor_run_config,
)


class _FakeRun:
    def __init__(self, *, fail_log: bool = False, run_id: str = "ttb-run") -> None:
        self.id = run_id
        self.fail_log = fail_log
        self.logged: list[tuple[dict, int, bool]] = []
        self.exit_codes: list[int] = []

    def log(self, payload, *, step: int, commit: bool) -> None:
        if self.fail_log:
            raise RuntimeError("network unavailable")
        self.logged.append((dict(payload), step, commit))

    def finish(self, *, exit_code: int) -> None:
        self.exit_codes.append(exit_code)


class _FakeWandb:
    def __init__(self, run: _FakeRun | None = None, *, fail_init: bool = False) -> None:
        self.run = run
        self.fail_init = fail_init
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        if self.fail_init:
            raise RuntimeError("not authenticated")
        return self.run


def _receipt() -> dict:
    return {
        "schema_version": TTB_MONITOR_SCHEMA_VERSION,
        "status": "completed",
        "step": 7,
        "versions": {
            "behavior": {
                "theta": "theta-step-000006",
                "phi": "phi-step-000006",
                "z": "z-step-000006",
            },
            "updated": {
                "theta": "theta-step-000007",
                "phi": "phi-step-000007",
                "z": "z-step-000007",
            },
        },
        "ttb": {"loss": 0.125, "delta_squared_mean": 2.0},
        "rollout": {
            "reward_mean": 0.5,
            "reward_sum": 12.0,
            "trajectory_length_mean": 5.25,
            "trajectory_length_max": 12,
            "valid_count": 24,
            "filtered_count": 4,
        },
        "runtime": {
            "gpu_device_count": 3,
            "gpu_peak_allocated_bytes": 60_000_000_000,
            "gpu_peak_reserved_bytes": 65_000_000_000,
            "step_seconds": 900.0,
            "rollouts_per_second": 28.0 / 900.0,
            "action_tokens_per_second": 8.5,
            "error_count": 1,
            "errors": [
                {"type": "TransientProviderError", "recovered": True}
            ],
        },
        "updates": {
            "theta": {"grad_norm": 1.5, "update_l2": 0.02, "nonzero": True},
            "phi": {"grad_norm": 0.8, "update_l2": 0.01, "nonzero": True},
            "z": {"grad_norm": 0.4, "update_l2": 0.003, "nonzero": True},
        },
        "publication": {
            "success": True,
            "canary_success": True,
            "theta_version": "theta-step-000007",
            "adapter_name": "theta_mbppplus_step_000007",
            "server_weight_version": "server-weight-000007",
            "checkpoint_version": "checkpoint:theta-step-000007",
            "sync_seconds": 12.5,
        },
        "checkpoints": {
            "official": {"due": False, "saved": False, "path": None},
            "project_recovery": {
                "saved": True,
                "path": "outputs/recovery/checkpoint_step_000007",
            },
        },
        "skill": {
            "enabled": True,
            "phase": "evidence_collection",
            "events": [{"event": "phase_started", "step": 7}],
        },
    }


def test_step_schema_covers_ttb_versions_updates_and_runtime() -> None:
    metrics = metrics_from_ttb_step_receipt(_receipt())
    payload = metrics.to_wandb()

    assert metrics.step == 7
    assert metrics.valid_rollout_count == 24
    assert metrics.filtered_rollout_count == 4
    assert metrics.ttb_loss == pytest.approx(0.125)
    assert metrics.delta_squared_mean == pytest.approx(2.0)
    assert payload["policy/theta/behavior_version"] == "theta-step-000006"
    assert payload["policy/phi/updated_version"] == "phi-step-000007"
    assert payload["policy/z/updated_version"] == "z-step-000007"
    assert payload["update/theta/nonzero"] is True
    assert payload["update/phi/l2"] == pytest.approx(0.01)
    assert payload["gradient/z/norm"] == pytest.approx(0.4)
    assert payload["publication/sync_success"] is True
    assert payload["publication/server_weight_version"] == "server-weight-000007"
    assert payload["publication/checkpoint_version"] == (
        "checkpoint:theta-step-000007"
    )
    assert payload["checkpoint/official_saved"] is False
    assert payload["checkpoint/project_recovery_saved"] is True
    assert payload["skill/event_count"] == 1
    assert payload["monitor/provenance"] == TTB_MONITOR_PROVENANCE


def test_due_official_checkpoint_and_project_recovery_are_required() -> None:
    missing_official = _receipt()
    missing_official["checkpoints"]["official"]["due"] = True
    with pytest.raises(TTBMonitoringError, match="official checkpoint"):
        metrics_from_ttb_step_receipt(missing_official)

    missing_recovery = _receipt()
    missing_recovery["checkpoints"]["project_recovery"]["saved"] = False
    with pytest.raises(TTBMonitoringError, match="project recovery"):
        metrics_from_ttb_step_receipt(missing_recovery)


def test_versions_publication_and_update_evidence_are_consistent() -> None:
    stale_version = _receipt()
    stale_version["versions"]["updated"]["phi"] = "phi-step-000006"
    with pytest.raises(TTBMonitoringError, match="phi updated version"):
        metrics_from_ttb_step_receipt(stale_version)

    wrong_publication = _receipt()
    wrong_publication["publication"]["theta_version"] = "theta-step-stale"
    with pytest.raises(TTBMonitoringError, match="published theta"):
        metrics_from_ttb_step_receipt(wrong_publication)

    false_nonzero = _receipt()
    false_nonzero["updates"]["z"]["nonzero"] = False
    with pytest.raises(TTBMonitoringError, match="z.nonzero disagrees"):
        metrics_from_ttb_step_receipt(false_nonzero)


def test_reward_runtime_and_error_aggregates_are_validated() -> None:
    inconsistent_reward = _receipt()
    inconsistent_reward["rollout"]["reward_sum"] = 11.0
    with pytest.raises(TTBMonitoringError, match="reward_sum is inconsistent"):
        metrics_from_ttb_step_receipt(inconsistent_reward)

    inconsistent_errors = _receipt()
    inconsistent_errors["runtime"]["error_count"] = 0
    with pytest.raises(TTBMonitoringError, match="error_count does not match"):
        metrics_from_ttb_step_receipt(inconsistent_errors)

    impossible_gpu_peak = _receipt()
    impossible_gpu_peak["runtime"]["gpu_peak_reserved_bytes"] = 1
    with pytest.raises(TTBMonitoringError, match="reserved_bytes"):
        metrics_from_ttb_step_receipt(impossible_gpu_peak)


def test_online_wandb_is_mandatory_and_no_test_uses_the_network() -> None:
    run = _FakeRun()
    wandb = _FakeWandb(run)
    monitor = WandbTTBMonitor.start(
        WandbTTBMonitorConfig(
            project="flowsteer-mbppplus",
            run_name="ttb-test",
            run_id="resume-ttb",
        ),
        run_config=ttb_monitor_run_config(
            dataset="mbpp_plus",
            total_steps=250,
            start_step=0,
            condition_id="mbppplus-ttb-v1",
        ),
        wandb_module=wandb,
    )
    metrics = monitor.log_completed_step(_receipt())
    monitor.finish(exit_code=0)

    assert monitor.run_id == "ttb-run"
    assert wandb.kwargs["mode"] == "online"
    assert wandb.kwargs["id"] == "resume-ttb"
    assert wandb.kwargs["resume"] == "allow"
    assert (
        wandb.kwargs["config"]["monitor_provenance"]
        == TTB_MONITOR_PROVENANCE
    )
    assert run.logged[0][1:] == (7, True)
    assert run.logged[0][0]["ttb/loss"] == pytest.approx(0.125)
    assert run.exit_codes == [0]
    assert metrics.step == 7


def test_wandb_init_and_log_fail_closed() -> None:
    with pytest.raises(TTBMonitoringError, match="must not continue unmonitored"):
        WandbTTBMonitor.start(
            WandbTTBMonitorConfig(project="project", run_name="run"),
            run_config={},
            wandb_module=_FakeWandb(fail_init=True),
        )

    monitor = WandbTTBMonitor(_FakeRun(fail_log=True))
    with pytest.raises(TTBMonitoringError, match="wandb.log failed"):
        monitor.log_completed_step(_receipt())


def test_schema_and_completed_publication_are_fail_closed() -> None:
    wrong_schema = _receipt()
    wrong_schema["schema_version"] = "legacy"
    with pytest.raises(TTBMonitoringError, match="schema_version"):
        metrics_from_ttb_step_receipt(wrong_schema)

    failed_sync = deepcopy(_receipt())
    failed_sync["publication"]["success"] = False
    with pytest.raises(TTBMonitoringError, match="publication and canary"):
        metrics_from_ttb_step_receipt(failed_sync)
