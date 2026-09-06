from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts import train_mbppplus_ttb as runner
from src.interactive.config_loader import load_yaml
from src.interactive.records import TaskRecord
from src.interactive.ttb_monitor import metrics_from_ttb_step_receipt


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "training_mbppplus_ttb_v1.yaml"


def _task(index: int) -> TaskRecord:
    return TaskRecord(
        task_id=f"mbpp-training:{index}",
        question=f"problem {index}",
        ground_truth=None,
        split="train",
        metadata={
            "dataset_key": "mbpp_plus",
            "training_population": "flowsteer_mbpp_train",
        },
    )


def test_formal_config_selects_ttb_without_grpo() -> None:
    config = load_yaml(CONFIG)
    formal = runner.validate_mbppplus_ttb_config(config)
    assert formal.optimizer_steps == 250
    assert formal.effective_batch_size == 28
    assert formal.max_trajectory_steps == 12
    assert formal.grpo_enabled is False
    assert formal.skill_evolution_enabled is False
    assert config["ttb"]["losses_mixed"] is False


def test_formal_config_rejects_mixed_grpo() -> None:
    config = load_yaml(CONFIG)
    config["grpo"]["enabled"] = True
    with pytest.raises(runner.MBPPPlusTTBRunError, match="grpo.enabled"):
        runner.validate_mbppplus_ttb_config(config)


def test_step_selection_is_distinct_and_rotates() -> None:
    tasks = tuple(_task(index) for index in range(10))
    first = runner._select_step_tasks(tasks, step=1, questions_per_batch=7)
    second = runner._select_step_tasks(tasks, step=2, questions_per_batch=7)
    assert [task.task_id for task in first] == [
        f"mbpp-training:{index}" for index in range(7)
    ]
    assert [task.task_id for task in second] == [
        "mbpp-training:7",
        "mbpp-training:8",
        "mbpp-training:9",
        "mbpp-training:0",
        "mbpp-training:1",
        "mbpp-training:2",
        "mbpp-training:3",
    ]
    assert len({task.task_id for task in second}) == 7


def test_gpu_resource_gate_requires_free_exclusive_devices(monkeypatch) -> None:
    config = load_yaml(CONFIG)
    gpu_rows = "\n".join(
        f"{index}, GPU-{index}, H800, 81920, 1024, 80896"
        for index in range(8)
    )

    def output(command, **kwargs):
        del kwargs
        if any("--query-gpu=" in value for value in command):
            return gpu_rows
        return "GPU-4, 4321, 2048, python\n"

    monkeypatch.setattr(runner.subprocess, "check_output", output)
    receipt = runner.gpu_resource_gate(config)
    assert receipt["ready"] is False
    assert any("rollout_supervisor" in value for value in receipt["blockers"])


def test_completed_step_receipt_satisfies_wandb_monitor_contract() -> None:
    summary = SimpleNamespace(
        update_step=1,
        reward_mean=0.5,
        ttb_loss=2.0,
        delta_squared_mean=3.0,
        trajectory_edges_mean=2.5,
        trajectory_edges_max=4,
        action_token_count=280,
        theta_grad_norm=1.0,
        theta_update_l2=0.2,
        phi_grad_norm=1.1,
        phi_update_l2=0.3,
        z_grad_norm=1.2,
        z_update_l2=0.4,
        behavior_policy_version=runner._theta_version(0),
        updated_policy_version=runner._theta_version(1),
        phi_behavior_version=runner._phi_version(0),
        phi_updated_version=runner._phi_version(1),
        z_behavior_version=runner._z_version(0),
        z_updated_version=runner._z_version(1),
        official_checkpoint_saved=False,
        official_checkpoint_dir="",
        recovery_checkpoint_dir="/tmp/recovery",
    )
    publication = SimpleNamespace(
        success=True,
        canary_succeeded=True,
        new_policy_version=runner._theta_version(1),
        adapter_name="theta_mbppplus_ttb_step_000001",
        checkpoint_version="checkpoint:qwen35-9b-mbppplus-ttb-theta-step-000001",
        duration_seconds=1.0,
    )
    receipt = runner._step_receipt(
        summary,
        publication,
        valid_count=28,
        filtered_count=2,
        step_seconds=20.0,
        gpu_metrics={
            "device_count": 3,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        server_weight_version="server-weight-step-000001",
    )
    metrics = metrics_from_ttb_step_receipt(receipt)
    assert metrics.valid_rollout_count == 28
    assert metrics.filtered_rollout_count == 2
    assert metrics.theta_update_nonzero is True
    assert metrics.published_server_weight_version == "server-weight-step-000001"


def test_continuation_server_weight_version_comes_from_committed_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "publication_receipt.json").write_text(
        json.dumps({"server_weight_version": "server-step-7"}),
        encoding="utf-8",
    )

    assert runner._continuation_server_weight_version("default", checkpoint) == (
        "server-step-7"
    )


def test_exact_serving_route_is_bound_inside_drained_gate() -> None:
    events: list[object] = []

    class Gate:
        def pause(self) -> None:
            events.append("pause")

        def drain(self) -> None:
            events.append("drain")

        def resume(self) -> None:
            events.append("resume")

    class Client:
        def update_policy_route(self, **values: object) -> None:
            events.append(dict(values))

    backend = SimpleNamespace(rollout_gate=Gate(), director_client=Client())
    runner._bind_exact_serving_route(
        backend,
        policy_version="theta-step-1",
        adapter_name="theta_adapter_1",
        server_weight_version="server-step-1",
    )

    assert events == [
        "pause",
        "drain",
        {
            "policy_version": "theta-step-1",
            "adapter_name": "theta_adapter_1",
            "expected_server_weight_version": "server-step-1",
        },
        "resume",
    ]
