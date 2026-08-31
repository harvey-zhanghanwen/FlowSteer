from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V212_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_12_heldout20_output_closure"
)
V214_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_14_heldout20_runtime_sink_closure"
)
V212_CONFIG_PATH = ROOT / "config" / f"evaluation_{V212_NAMESPACE}.yaml"
V214_CONFIG_PATH = ROOT / "config" / f"evaluation_{V214_NAMESPACE}.yaml"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _expected_v214_from_v212(v212: dict) -> dict:
    expected = deepcopy(v212)
    expected["experiment"]["name"] = V214_NAMESPACE
    expected["experiment"]["condition_id"] = V214_NAMESPACE
    expected["experiment"]["output_dir"] = (
        f"artifacts/{V214_NAMESPACE}/evaluation"
    )
    expected["storage"] = {
        key: (
            value.replace(V212_NAMESPACE, V214_NAMESPACE)
            if isinstance(value, str)
            else value
        )
        for key, value in expected["storage"].items()
    }
    expected["policy_sync"]["adapter_name_prefix"] = (
        f"unused_{V214_NAMESPACE}_"
    )
    return expected


def test_v214_is_an_exact_namespace_only_thin_copy_of_v212_fixed20() -> None:
    v212 = _load(V212_CONFIG_PATH)
    v214 = _load(V214_CONFIG_PATH)

    assert v214 == _expected_v214_from_v212(v212)


def test_v214_keeps_the_exact_fixed20_direct_and_agentgraph_protocol() -> None:
    v212 = _load(V212_CONFIG_PATH)
    v214 = _load(V214_CONFIG_PATH)
    previous = v212["healthbench_professional_evaluation"]
    candidate = v214["healthbench_professional_evaluation"]

    assert candidate == previous
    assert candidate["selection"] == "task_ids"
    assert candidate["sample_count"] == 20
    assert len(candidate["task_ids"]) == 20
    assert len(set(candidate["task_ids"])) == 20
    assert candidate["direct_reused_from"] == (
        "artifacts/healthbench_professional_mixed_all_thinking_v2_6/"
        "evaluation/direct_predictions.jsonl"
    )
    assert v214["experiment"]["seed"] == v212["experiment"]["seed"]
    assert v214["director"] == v212["director"]
    assert v214["director"]["max_rounds"] == 20
    assert v214["agent_graph"] == v212["agent_graph"]
    assert v214["healthbench_tool_runtime"] == v212["healthbench_tool_runtime"]
    assert v214["evaluation"] == v212["evaluation"]


def test_v214_has_fully_isolated_artifact_report_and_adapter_namespaces() -> None:
    config = _load(V214_CONFIG_PATH)

    assert config["experiment"]["name"] == V214_NAMESPACE
    assert config["experiment"]["condition_id"] == V214_NAMESPACE
    assert config["experiment"]["output_dir"] == (
        f"artifacts/{V214_NAMESPACE}/evaluation"
    )
    for key, path in config["storage"].items():
        if key == "schema_version":
            continue
        expected_root = "reports" if key.startswith("report_") else "artifacts"
        assert path.startswith(f"{expected_root}/{V214_NAMESPACE}/"), key
        assert V212_NAMESPACE not in path, key
    assert config["policy_sync"]["adapter_name_prefix"] == (
        f"unused_{V214_NAMESPACE}_"
    )


def test_v214_keeps_training_grpo_sync_exploration_and_skills_disabled() -> None:
    config = _load(V214_CONFIG_PATH)

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
