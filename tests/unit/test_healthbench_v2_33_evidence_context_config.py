"""The new treatment changes orchestration, not the frozen evaluation task."""

from copy import deepcopy
from pathlib import Path

import yaml

from src.interactive.config_loader import validate_agent_graph_config


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_32_full525_receipt_bound_completion.yaml"
NEW = ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context.yaml"


def _profiles():
    return tuple(yaml.safe_load(p.read_text()) for p in (OLD, NEW))


def test_frozen_population_generation_and_reference_evaluator():
    old, new = _profiles()
    for key in ("data", "evaluation", "execution_timeout"):
        assert new[key] == old[key]
    for key in ("seed", "sampling_schedule_purpose", "catalog_order_namespace", "tool_version"):
        assert new["experiment"][key] == old["experiment"][key]
    old_eval = old["healthbench_professional_evaluation"]
    new_eval = new["healthbench_professional_evaluation"]
    for key, value in old_eval.items():
        assert new_eval[key] == value
    assert new_eval["sample_count"] == 525
    assert new_eval["direct_artifact_communication_profile"] == old["agent_graph"]["artifact_communication_profile"]
    assert new_eval["direct_reference_config"].endswith(OLD.name)
    assert new_eval["direct_reused_from"].endswith("direct_predictions.jsonl")
    for key in ("direct_reference_config", "direct_reference_manifest", "direct_reused_from", "direct_failures_reused_from"):
        assert "v2_32" in new_eval[key]


def test_only_versioned_director_and_graph_communication_treatments():
    old, new = _profiles()
    graph = deepcopy(new["agent_graph"])
    assert graph.pop("artifact_communication_profile") == "producer_context_structured_evidence_v3"
    old_graph = deepcopy(old["agent_graph"])
    old_graph.pop("artifact_communication_profile")
    assert graph == old_graph
    director = deepcopy(new["director"])
    old_director = deepcopy(old["director"])
    assert director.pop("behavior_policy_version").endswith("v2-33")
    old_director.pop("behavior_policy_version")
    assert director == old_director
    assert new["experiment"]["prompt_version"] == "agentgraph.director.minimal-neutral.v19"
    assert new["experiment"]["condition_id"] != old["experiment"]["condition_id"]
    tool = deepcopy(new["healthbench_tool_runtime"])
    old_tool = deepcopy(old["healthbench_tool_runtime"])
    tool.pop("condition_id")
    old_tool.pop("condition_id")
    assert tool == old_tool


def test_open_graph_and_no_training_or_skill_updates():
    _, new = _profiles()
    validate_agent_graph_config(new)
    assert new["agent_graph"]["contract_type"] == "free_text"
    assert new["agent_graph"]["require_format_agent"] is False
    assert new["agent_graph"]["max_bidirectional_block_size"] == 2
    assert new["agent_graph"]["semantic_protocol_by_source"]["healthbench_professional"] == "none"
    assert new["director"]["execute_on_edit"] is True
    assert new["experiment"]["training_enabled"] is False
    for section in ("grpo", "policy_sync", "exploration", "skills"):
        assert new[section]["enabled"] is False
    assert new["grpo"]["max_optimizer_updates"] == 0
