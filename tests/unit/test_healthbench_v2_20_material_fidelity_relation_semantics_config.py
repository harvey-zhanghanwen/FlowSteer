from copy import deepcopy
from pathlib import Path
import json

import yaml

from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V16,
    director_system_prompt_for_version,
    encode_director_transcript,
)


ROOT = Path(__file__).resolve().parents[2]
V219_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_19_heldout20_executable_domain_query_guard.yaml"
)
V220_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_20_heldout20_material_fidelity_relation_semantics.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v220_preserves_fixed20_models_generation_tools_and_evaluator() -> None:
    v219 = _load(V219_CONFIG_PATH)
    v220 = _load(V220_CONFIG_PATH)

    assert v220["data"] == v219["data"]
    old_eval = deepcopy(v219["healthbench_professional_evaluation"])
    new_eval = deepcopy(v220["healthbench_professional_evaluation"])
    new_eval["direct_protocol"] = old_eval["direct_protocol"]
    assert new_eval == old_eval
    assert v220["director"] == v219["director"]
    assert v220["agent_graph"] == v219["agent_graph"]
    assert v220["evaluation"] == v219["evaluation"]
    old_tool = dict(v219["healthbench_tool_runtime"])
    new_tool = dict(v220["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    assert new_tool == old_tool


def test_v220_uses_compact_neutral_relation_semantics_prompt() -> None:
    config = _load(V220_CONFIG_PATH)
    assert config["experiment"]["prompt_version"] == (
        DIRECTOR_PROMPT_VERSION_V16
    )
    prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V16)
    assert "directed producer-to-consumer relation" in prompt
    assert "bounded peer revision" in prompt
    assert "material conflict" in prompt
    for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
        assert fixed_role not in prompt
    assert "HealthBench" not in prompt
    transcript = encode_director_transcript(
        (
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Task and live Canvas state"},
        )
    )
    assert transcript.startswith("Flow-Director chat transcript\n\n")
    payload = json.loads(transcript.split("\n\n", 1)[1])
    assert payload["messages"][0]["content"] == prompt


def test_v220_has_independent_namespace_and_no_training_or_skills() -> None:
    config = _load(V220_CONFIG_PATH)
    assert "v2_20" in config["experiment"]["condition_id"]
    for path_value in config["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_20" in path_value
    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
