"""Environment-aware YAML loading without ever persisting resolved secrets."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping

import yaml

from .model_registry import ModelRegistry


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-(.*?))?\}")


class ConfigurationError(ValueError):
    pass


def expand_environment(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${NAME}`` and ``${NAME:-default}`` placeholders."""

    source = os.environ if environ is None else environ
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = source.get(name)
            if resolved is not None and resolved != "":
                return resolved
            if default is not None:
                return default
            raise ConfigurationError(f"missing required environment variable: {name}")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_environment(item, source) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item, source) for item in value)
    if isinstance(value, dict):
        return {key: expand_environment(item, source) for key, item in value.items()}
    return value


def load_yaml(path: str | os.PathLike[str], *, expand_env: bool = True) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be a mapping")
    return expand_environment(value) if expand_env else value


def load_model_registry(path: str | os.PathLike[str]) -> ModelRegistry:
    value = load_yaml(path, expand_env=True)
    try:
        registry = ModelRegistry.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid model catalog: {exc}") from exc
    if not len(registry):
        raise ConfigurationError("model catalog must contain at least one model")
    return registry


def validate_agent_graph_config(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment",
        "data",
        "storage",
        "director",
        "agent_graph",
        "grpo",
        "exploration",
        "skills",
        "gpu",
    }
    missing = required - set(value)
    if missing:
        raise ConfigurationError("missing top-level sections: " + ", ".join(sorted(missing)))
    grpo = value["grpo"]
    if grpo.get("objective") != "action_masked_one_pass":
        raise ConfigurationError("AgentGraph path requires action_masked_one_pass objective")
    if grpo.get("group_key") != ["task_id", "condition_id", "policy_version"]:
        raise ConfigurationError("GRPO group key must preserve task, condition, and policy version")
    forbidden_positive = ("structural_reward", "exploration_reward", "skill_usage_reward")
    for name in forbidden_positive:
        if float(grpo.get(name, 0.0)) != 0.0:
            raise ConfigurationError(f"{name} must remain zero in the strict AgentGraph path")

    director = value["director"]
    if director.get("backend") != "sglang":
        raise ConfigurationError("Qwen3.5 AgentGraph path requires the SkillFlow SGLang backend")
    if director.get("served_model_name") != "supervisor_theta":
        raise ConfigurationError("SGLang served_model_name must be supervisor_theta")
    if director.get("prompt_profile") != "minimal":
        raise ConfigurationError("the architecture baseline requires the minimal Director prompt")

    graph = value["agent_graph"]
    expected_actions = [
        "add_agent",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    if graph.get("actions") != expected_actions:
        raise ConfigurationError("AgentGraph search space must contain the six atomic actions")
    if graph.get("contract_type") != "free_text" or graph.get("relation_encoding") != "two_bit":
        raise ConfigurationError("AgentGraph requires free-text contracts and two-bit relations")

    experiment = value["experiment"]
    if experiment.get("phase") == "architecture_only":
        enabled_flags = {
            "experiment.training_enabled": experiment.get("training_enabled"),
            "grpo.enabled": grpo.get("enabled"),
            "exploration.enabled": value["exploration"].get("enabled"),
            "skills.enabled": value["skills"].get("enabled"),
            "gpu.training_enabled": value["gpu"].get("training_enabled"),
        }
        active = [name for name, enabled in enabled_flags.items() if enabled is not False]
        if active:
            raise ConfigurationError(
                "architecture_only phase must keep training features disabled: " + ", ".join(active)
            )

    gpu = value["gpu"]
    physical = [
        int(gpu["learner_physical"]),
        int(gpu["rollout_physical"]),
        int(gpu["gradient_replica_physical"]),
    ]
    if len(set(physical)) != 3:
        raise ConfigurationError("the three GPU roles must use distinct physical devices")
    if gpu.get("rollout_engine") != "sglang":
        raise ConfigurationError("the rollout GPU must use SGLang")
    oom = gpu.get("oom_policy", {})
    micro_batch = int(oom.get("micro_batch_size", 0))
    minimum = int(oom.get("minimum_micro_batch_size", 0))
    if minimum < 1 or micro_batch < minimum:
        raise ConfigurationError("OOM micro-batch bounds are invalid")
