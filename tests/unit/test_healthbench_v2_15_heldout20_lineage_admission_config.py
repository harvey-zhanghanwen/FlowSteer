from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V214_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_14_heldout20_runtime_sink_closure"
)
V215_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_15_heldout20_lineage_admission"
)
V214_CONFIG_PATH = ROOT / "config" / f"evaluation_{V214_NAMESPACE}.yaml"
V215_CONFIG_PATH = ROOT / "config" / f"evaluation_{V215_NAMESPACE}.yaml"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _expected_v215_from_v214(v214: dict) -> dict:
    expected = deepcopy(v214)
    expected["experiment"]["name"] = V215_NAMESPACE
    expected["experiment"]["condition_id"] = V215_NAMESPACE
    expected["experiment"]["output_dir"] = (
        f"artifacts/{V215_NAMESPACE}/evaluation"
    )
    expected["storage"] = {
        key: (
            value.replace(V214_NAMESPACE, V215_NAMESPACE)
            if isinstance(value, str)
            else value
        )
        for key, value in expected["storage"].items()
    }
    expected["policy_sync"]["adapter_name_prefix"] = (
        f"unused_{V215_NAMESPACE}_"
    )
    return expected


def test_v215_is_an_exact_namespace_only_thin_copy_of_v214_fixed20() -> None:
    v214 = _load(V214_CONFIG_PATH)
    v215 = _load(V215_CONFIG_PATH)

    assert v215 == _expected_v215_from_v214(v214)


def test_v215_keeps_the_exact_fixed20_direct_and_agentgraph_protocol() -> None:
    v214 = _load(V214_CONFIG_PATH)
    v215 = _load(V215_CONFIG_PATH)
    previous = v214["healthbench_professional_evaluation"]
    candidate = v215["healthbench_professional_evaluation"]

    assert candidate == previous
    assert candidate["selection"] == "task_ids"
    assert candidate["sample_count"] == 20
    assert len(candidate["task_ids"]) == 20
    assert len(set(candidate["task_ids"])) == 20
    assert candidate["direct_reused_from"] == (
        "artifacts/healthbench_professional_mixed_all_thinking_v2_6/"
        "evaluation/direct_predictions.jsonl"
    )
    assert v215["experiment"]["seed"] == v214["experiment"]["seed"]
    assert v215["director"] == v214["director"]
    assert v215["director"]["max_rounds"] == 20
    assert v215["agent_graph"] == v214["agent_graph"]
    assert v215["healthbench_tool_runtime"] == v214["healthbench_tool_runtime"]
    assert v215["evaluation"] == v214["evaluation"]


def test_v215_has_fully_isolated_artifact_report_and_adapter_namespaces() -> None:
    config = _load(V215_CONFIG_PATH)

    assert config["experiment"]["name"] == V215_NAMESPACE
    assert config["experiment"]["condition_id"] == V215_NAMESPACE
    assert config["experiment"]["output_dir"] == (
        f"artifacts/{V215_NAMESPACE}/evaluation"
    )
    for key, path in config["storage"].items():
        if key == "schema_version":
            continue
        expected_root = "reports" if key.startswith("report_") else "artifacts"
        assert path.startswith(f"{expected_root}/{V215_NAMESPACE}/"), key
        assert V214_NAMESPACE not in path, key
    assert config["policy_sync"]["adapter_name_prefix"] == (
        f"unused_{V215_NAMESPACE}_"
    )


def test_v215_keeps_training_grpo_sync_exploration_and_skills_disabled() -> None:
    config = _load(V215_CONFIG_PATH)

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["optimization_passes_per_rollout_batch"] == 0
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["exploration"]["forced_probe_rollouts"] == 0
    assert config["skills"] == {
        "enabled": False,
        "initial_library": [],
        "retrieval_top_k": 0,
        "library_version": "none",
    }
    assert config["gpu"]["training_enabled"] is False
