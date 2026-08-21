from pathlib import Path

import yaml

from src.interactive.director import AgentGraphOrchestrator
from src.interactive.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[2]


class _UnusedDirectorClient:
    async def propose(self, prompt: str, *, seed: int | None = None):
        raise AssertionError("catalog rendering must not call the Director")


def test_v9_catalog_keeps_capacity_receipts_out_of_director_observation() -> None:
    catalog_path = ROOT / "config/model_catalog_hotpotqa_dynamic_v9.yaml"
    registry = ModelRegistry.from_yaml(catalog_path)
    orchestrator = AgentGraphOrchestrator(registry, _UnusedDirectorClient())

    rendered = {
        item["model_id"]: item for item in orchestrator._model_catalog()
    }
    for model_id in ("qwen3.5-flash", "qwen3.5-plus", "deepseek-v4-pro"):
        stored = registry.require_model(model_id).metadata
        assert "hotpotqa_development_diagnostic" in stored
        visible = rendered[model_id]["routing_metadata"]
        assert "hotpotqa_development_diagnostic" not in visible
        assert "hotpotqa_development_diagnostic_scope" not in visible
        assert "hotpotqa_development_diagnostic_source" not in visible
        profile = visible["profile"]
        assert "hotpotqa_development_diagnostic" not in profile
        assert "strict_em" not in profile
        assert "n6" not in profile


def test_round8_capacity_prior_is_treatment_only() -> None:
    preregistration = yaml.safe_load(
        (ROOT / "config/joint_qa_round8_evidence.yaml").read_text(
            encoding="utf-8"
        )
    )
    baseline = preregistration["baseline_action"]
    capacity = preregistration["candidate_actions"][
        "receipt_backed_executor_capacity"
    ]["instruction"]

    assert baseline["prompt_prior"] is None
    assert "No additional prompt prior" in baseline["instruction"]
    for receipt_fact in (
        "Qwen3.5-Plus obtained 6/6",
        "Qwen3.5-Flash obtained 5/6",
        "DeepSeek-V4-Pro obtained 5/6",
        "one operational failure",
        "rejectable prior",
    ):
        assert receipt_fact in capacity
    assert "6/6" not in baseline["instruction"]
    assert "5/6" not in baseline["instruction"]


def test_round8_records_reuse_adaptation_and_project_addition_boundaries() -> None:
    preregistration = yaml.safe_load(
        (ROOT / "config/joint_qa_round8_evidence.yaml").read_text(
            encoding="utf-8"
        )
    )
    classification = preregistration["source_mapping"][
        "implementation_classification"
    ]

    assert classification["direct_reuse"]
    assert classification["necessary_adaptation"]
    assert classification["project_algorithm_addition"]
    assert classification["not_implemented"]
