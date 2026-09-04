from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]

V227_HELDOUT = ROOT / (
    "config/evaluation_healthbench_professional_mixed_all_thinking_v2_27_"
    "heldout20_bestbase_provenance_scope_quality.yaml"
)
V228_HELDOUT = ROOT / (
    "config/evaluation_healthbench_professional_mixed_all_thinking_v2_28_"
    "heldout20_declared_dependency_structured_evidence.yaml"
)
V227_FULL = ROOT / (
    "config/evaluation_healthbench_professional_mixed_all_thinking_v2_27_"
    "full525_bestbase_provenance_scope_quality.yaml"
)
V228_FULL = ROOT / (
    "config/evaluation_healthbench_professional_mixed_all_thinking_v2_28_"
    "full525_declared_dependency_structured_evidence.yaml"
)

CASES = (
    (
        V227_HELDOUT,
        V228_HELDOUT,
        "healthbench_professional_mixed_all_thinking_v2_27_heldout20_"
        "bestbase_provenance_scope_quality",
        "healthbench_professional_mixed_all_thinking_v2_28_heldout20_"
        "declared_dependency_structured_evidence",
    ),
    (
        V227_FULL,
        V228_FULL,
        "healthbench_professional_mixed_all_thinking_v2_27_full525_"
        "bestbase_provenance_scope_quality",
        "healthbench_professional_mixed_all_thinking_v2_28_full525_"
        "declared_dependency_structured_evidence",
    ),
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v228_preserves_each_v227_population_direct_generation_and_evaluator() -> None:
    for old_path, new_path, _, _ in CASES:
        old = _load(old_path)
        new = _load(new_path)

        assert new["schema_version"] == old["schema_version"]
        assert new["execution_timeout"] == old["execution_timeout"]
        assert new["data"] == old["data"]
        assert (
            new["healthbench_professional_evaluation"]
            == old["healthbench_professional_evaluation"]
        )
        assert new["director"] == old["director"]
        assert new["evaluation"] == old["evaluation"]
        for section in ("grpo", "exploration", "skills", "gpu", "deployment"):
            assert new[section] == old[section]

        old_tool = deepcopy(old["healthbench_tool_runtime"])
        new_tool = deepcopy(new["healthbench_tool_runtime"])
        old_tool.pop("condition_id")
        new_tool.pop("condition_id")
        assert new_tool == old_tool


def test_v228_changes_only_the_two_declared_agentgraph_treatments() -> None:
    for old_path, new_path, _, _ in CASES:
        old_graph = deepcopy(_load(old_path)["agent_graph"])
        new_graph = deepcopy(_load(new_path)["agent_graph"])

        assert new_graph.pop("require_declared_dependency_relations") is True
        assert (
            new_graph["artifact_communication_profile"]
            == "producer_context_structured_evidence_v2"
        )
        new_graph["artifact_communication_profile"] = old_graph[
            "artifact_communication_profile"
        ]
        assert old_graph["artifact_communication_profile"] == (
            "producer_context_exact_dedup_v1"
        )
        assert new_graph == old_graph


def test_v228_keeps_sampling_identity_and_uses_fresh_artifact_namespaces() -> None:
    for old_path, new_path, old_namespace, new_namespace in CASES:
        old = _load(old_path)
        new = _load(new_path)

        old_experiment = deepcopy(old["experiment"])
        new_experiment = deepcopy(new["experiment"])
        for field in ("name", "condition_id", "output_dir"):
            old_experiment.pop(field)
            new_experiment.pop(field)
        assert new_experiment == old_experiment
        assert new["experiment"]["name"] == new_namespace
        assert new["experiment"]["condition_id"] == new_namespace
        assert new["healthbench_tool_runtime"]["condition_id"] == new_namespace
        assert new["experiment"]["output_dir"] == (
            f"artifacts/{new_namespace}/evaluation"
        )

        assert new["storage"]["schema_version"] == old["storage"][
            "schema_version"
        ]
        for field, old_value in old["storage"].items():
            if field == "schema_version":
                continue
            new_value = new["storage"][field]
            assert isinstance(old_value, str)
            assert isinstance(new_value, str)
            assert new_namespace in new_value
            assert new_value.replace(new_namespace, old_namespace) == old_value

        old_policy = deepcopy(old["policy_sync"])
        new_policy = deepcopy(new["policy_sync"])
        old_policy.pop("adapter_name_prefix")
        new_policy.pop("adapter_name_prefix")
        assert new_policy == old_policy
        assert new["policy_sync"]["adapter_name_prefix"] == (
            "unused_healthbench_professional_mixed_all_thinking_v2_28_"
        )


def test_v228_heldout_and_full_retain_their_frozen_sampling_schedules() -> None:
    old_heldout = _load(V227_HELDOUT)
    new_heldout = _load(V228_HELDOUT)
    old_full = _load(V227_FULL)
    new_full = _load(V228_FULL)

    assert new_heldout["experiment"]["sampling_schedule_purpose"] == (
        old_heldout["experiment"]["sampling_schedule_purpose"]
    )
    assert new_full["experiment"]["sampling_schedule_purpose"] == (
        old_full["experiment"]["sampling_schedule_purpose"]
    )
    assert new_heldout["healthbench_professional_evaluation"]["sample_count"] == 20
    assert new_heldout["healthbench_professional_evaluation"]["selection"] == (
        "task_ids"
    )
    assert new_full["healthbench_professional_evaluation"]["sample_count"] == 525
    assert new_full["healthbench_professional_evaluation"]["selection"] == (
        "sequential"
    )


def test_v228_keeps_every_training_and_skill_path_disabled() -> None:
    for _, new_path, _, _ in CASES:
        config = _load(new_path)
        assert config["experiment"]["training_enabled"] is False
        assert config["director"]["lora"]["enabled"] is False
        assert config["grpo"]["enabled"] is False
        assert config["grpo"]["max_optimizer_updates"] == 0
        assert config["policy_sync"]["enabled"] is False
        assert config["exploration"]["enabled"] is False
        assert config["skills"]["enabled"] is False
        assert config["gpu"]["training_enabled"] is False
        assert config["deployment"]["allow_forced_probes"] is False
