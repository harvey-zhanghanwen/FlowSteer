from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.interactive.config_loader import (
    ConfigurationError,
    expand_environment,
    load_yaml,
    validate_agent_graph_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/training_triviaqa_v16_onepass_grpo_1step_prepare.yaml"


def _raw_config() -> dict:
    return load_yaml(CONFIG_PATH, expand_env=False)


def _resolved_config() -> dict:
    return expand_environment(
        _raw_config(),
        {
            "TRIVIAQA_LEARNER_GPU": "0",
            "TRIVIAQA_ROLLOUT_GPU": "1",
            "TRIVIAQA_GRADIENT_REPLICA_GPU": "2",
            "FLOWSTEER_TRIVIAQA_SUPERVISOR_PORT": "18015",
            "TRIVIAQA_GPU_ALLOCATION_RECEIPT": (
                "artifacts/triviaqa_v16_onepass_grpo/gpu_allocation.json"
            ),
        },
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_config_is_fail_closed_until_runner_and_resource_preflight() -> None:
    config = _raw_config()
    gate = config["execution_gate"]

    assert config["experiment"]["phase"] == "triviaqa_micro_training"
    assert config["experiment"]["status"] == "prepare_only_resource_blocked"
    assert gate["execution_enabled"] is False
    assert gate["allowed_mode"] == "prepare_only"
    assert gate["resource_status"] == "resource_blocked"
    assert gate["runner_support_status"] == (
        "triviaqa_micro_training_prepare_only_verified"
    )
    assert gate["gpu_allocation_frozen"] is False
    assert config["acceptance_gates"]["phase_0_accepted"] is False
    assert config["acceptance_gates"]["selected_loss_and_token_mask_verified"] is False

    with pytest.raises(
        ConfigurationError,
        match="TRIVIAQA_LEARNER_GPU",
    ):
        expand_environment(config, {})


def test_config_selects_only_md_one_pass_grpo_and_one_real_group() -> None:
    config = _resolved_config()
    validate_agent_graph_config(config)

    method = config["method"]
    grpo = config["grpo"]
    selection = config["data"]["triviaqa_micro"]

    assert method["primary_objective"] == "action_masked_one_pass_grpo"
    assert method["ttb_enabled"] is False
    assert method["loss_mixing_allowed"] is False
    assert method["trainable_parameters"] == ["director_theta_lora"]
    assert method["executor_frozen"] is True
    assert grpo["enabled"] is True
    assert grpo["objective"] == "action_masked_one_pass"
    assert grpo["group_key"] == ["task_id", "condition_id", "policy_version"]
    assert grpo["samples_per_problem"] == 4
    assert grpo["expected_rollout_count"] == 4
    assert grpo["optimization_passes_per_rollout_batch"] == 1
    assert grpo["max_optimizer_updates"] == 1
    assert grpo["terminal_task_reward_only"] is True
    assert grpo["structural_reward"] == 0.0
    assert grpo["exploration_reward"] == 0.0
    assert grpo["skill_usage_reward"] == 0.0
    assert selection["dataset_key"] == "triviaqa"
    assert selection["expected_total_tasks"] == 1
    assert selection["rollout_ordinals"] == [0, 1, 2, 3]
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_config_retains_v16_agentgraph_and_train_only_memory_contract() -> None:
    config = _resolved_config()
    director = config["director"]
    graph = config["agent_graph"]
    runtime = config["qa_tool_runtime"]

    assert "qwen3.5-9b" in director["base_model"].lower()
    assert director["backend"] == "sglang"
    assert director["served_model_name"] == "supervisor_theta"
    assert director["prompt_profile"] == "minimal"
    assert director["action_decoding"] == "unconstrained"
    assert "sampling_action_profile" not in director
    assert director["execute_on_edit"] is True
    assert graph["actions"][0] == "add_subgraph"
    assert graph["max_agents_per_subgraph"] == 3
    assert graph["director_feedback_mode"] == "control_plane"
    assert graph["semantic_protocol_by_source"] == {
        "triviaqa": "qa_verified_answer_lineage_v2"
    }
    assert graph["required_evidence_tool_id"] == "triviaqa.qa_memory"
    assert graph["terminal_protocol_by_source"] == {
        "triviaqa": "exact_single_answer_tag"
    }
    assert runtime["dataset_scope"] == ["triviaqa"]
    assert runtime["passage_source"] == "external_corpus"
    assert runtime["corpus_kind"] == "fact_memory"
    assert runtime["index_path"] == (
        "data/triviaqa_fact_memory_train512_v1/train_only/index"
    )

    index_manifest = json.loads(
        (ROOT / runtime["index_manifest_path"]).read_text(encoding="utf-8")
    )
    assert index_manifest["memory_count"] == 512
    assert index_manifest["fact_only"] is True
    assert index_manifest["provenance_loaded_by_index"] is False
    assert index_manifest["embedding_input_field"] == "fact_text"
    assert index_manifest["frozen_top_k"] == runtime["frozen_top_k"]


def test_frozen_split_is_exactly_512_train_and_128_heldout() -> None:
    config = _raw_config()
    data = config["data"]
    train = _read_jsonl(ROOT / data["train_path"])
    validation = _read_jsonl(ROOT / data["validation_path"])

    assert data["enforce_split_isolation"] is True
    assert len(train) == data["expected_train_count"] == 512
    assert len(validation) == data["expected_heldout_validation_count"] == 128
    assert {row["metadata"]["dataset_key"] for row in train} == {"triviaqa"}
    assert {row["metadata"]["dataset_key"] for row in validation} == {"triviaqa"}
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in validation} == {"validation"}

    train_base_ids = {
        row["metadata"]["sampling"]["base_task_id"] for row in train
    }
    validation_base_ids = {
        row["metadata"]["sampling"]["base_task_id"] for row in validation
    }
    assert train_base_ids.isdisjoint(validation_base_ids)


def test_gpu_roles_and_supervisor_port_are_launch_frozen_not_hard_coded() -> None:
    raw = _raw_config()
    gpu = raw["gpu"]

    assert gpu["learner_physical"] == "${TRIVIAQA_LEARNER_GPU}"
    assert gpu["rollout_physical"] == "${TRIVIAQA_ROLLOUT_GPU}"
    assert gpu["gradient_replica_physical"] == "${TRIVIAQA_GRADIENT_REPLICA_GPU}"
    assert gpu["learner_device"] == "cuda:${TRIVIAQA_LEARNER_GPU}"
    assert gpu["gradient_replica_device"] == (
        "cuda:${TRIVIAQA_GRADIENT_REPLICA_GPU}"
    )
    assert "${FLOWSTEER_TRIVIAQA_SUPERVISOR_PORT:-18015}" in (
        raw["director"]["api_base"]
    )
    assert "${FLOWSTEER_TRIVIAQA_SUPERVISOR_PORT:-18015}" in (
        raw["policy_sync"]["api_base"]
    )
    assert gpu["allocation_receipt_path"] == (
        "${TRIVIAQA_GPU_ALLOCATION_RECEIPT}"
    )

    resolved = _resolved_config()
    assert resolved["gpu"]["learner_physical"] == "0"
    assert resolved["gpu"]["rollout_physical"] == "1"
    assert resolved["gpu"]["gradient_replica_physical"] == "2"
    assert resolved["director"]["api_base"] == "http://127.0.0.1:18015/v1"
    assert resolved["policy_sync"]["api_base"] == "http://127.0.0.1:18015"


def test_wandb_binding_is_fixed_and_contains_no_credential() -> None:
    binding = _raw_config()["wandb_binding"]

    assert binding["required"] is True
    assert binding["mode"] == "online"
    assert binding["entity"] == "zhanghanwen6660909-dut"
    assert binding["project"] == "flowsteer-triviaqa"
    assert binding["credential_source"] == "wandb_sdk_standard"
    assert binding["require_run_url_before_training_started_status"] is True
    assert binding["api_key_in_config_allowed"] is False
    assert "api_key" not in binding
