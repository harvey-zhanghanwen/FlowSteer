"""Unvalidated, rejectable HealthBench orchestration conditions.

Thin adaptation of ``run_joint_qa_mace_skill._prompt_condition``. The existing
``LiveSmokeBackend.collect(..., prompt_priors=..., forced_probe=True)`` owns
execution and exposure receipts. This module does not load a SkillStore,
publish ACTIVE Skills, infer effects, or call a model/evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.interactive.config_loader import ConfigurationError, load_yaml


PROFILE_SCHEMA = "flowsteer.healthbench-candidate-skill-profile.v1"
DATASET_KEY = "healthbench_professional"
_STAGES = {"*", "empty_graph", "construction", "before_final_answer"}


def _object(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"{name} has an incompatible field set")
    return value


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ConfigurationError(f"{name} must be non-empty text up to {limit} characters")
    if value != value.strip():
        raise ConfigurationError(f"{name} must not have surrounding whitespace")
    return value


def validate_candidate_skill_run_config(config: Mapping[str, Any]) -> None:
    """Require the existing inference-only boundary; do not relax ACTIVE gates."""

    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping) or experiment.get("training_enabled") is not False:
        raise ConfigurationError("candidate evaluation requires training_enabled=false")
    for name in ("grpo", "exploration", "skills", "policy_sync"):
        section = config.get(name)
        if not isinstance(section, Mapping) or section.get("enabled") is not False:
            raise ConfigurationError(f"candidate evaluation requires {name}.enabled=false")
    updates = config["grpo"].get("max_optimizer_updates")
    if type(updates) is not int or updates != 0:
        raise ConfigurationError("candidate evaluation requires max_optimizer_updates=0")
    director = config.get("director", {})
    lora = director.get("lora") if isinstance(director, Mapping) else None
    if not isinstance(lora, Mapping) or lora.get("enabled") is not False:
        raise ConfigurationError("candidate evaluation requires director.lora.enabled=false")
    for name in ("mace", "bayesian", "skill_evolution"):
        section = config.get(name)
        if section is not None and (
            not isinstance(section, Mapping) or section.get("enabled") is not False
        ):
            raise ConfigurationError(f"candidate evaluation cannot enable {name}")


def validate_candidate_skill_profile(
    profile: Mapping[str, Any], *, dataset_key: str = DATASET_KEY
) -> None:
    """Validate bounded candidate metadata without manufacturing effect evidence."""

    if dataset_key != DATASET_KEY:
        raise ConfigurationError("candidate profile supports only healthbench_professional")
    root = _object(profile, "candidate profile", {
        "schema_version", "profile_version", "dataset_key", "status",
        "training_enabled", "skill_publication_enabled", "candidates",
    })
    if root["schema_version"] != PROFILE_SCHEMA:
        raise ConfigurationError("unsupported candidate profile schema")
    _text(root["profile_version"], "profile_version", 128)
    if root["dataset_key"] != dataset_key or root["status"] != "candidate":
        raise ConfigurationError("profile must be candidate for the requested dataset")
    if root["training_enabled"] is not False or root["skill_publication_enabled"] is not False:
        raise ConfigurationError("candidate profile cannot enable training or publication")
    candidates = root["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ConfigurationError("candidate profile must contain exactly three candidates")
    identities: set[str] = set()
    for index, candidate in enumerate(candidates):
        item = _object(candidate, f"candidate[{index}]", {
            "condition_id", "status", "condition", "trigger", "action",
        })
        identity = _text(item["condition_id"], "condition_id", 128)
        if identity in identities:
            raise ConfigurationError("candidate condition IDs must be unique")
        identities.add(identity)
        if item["status"] != "candidate":
            raise ConfigurationError("unvalidated conditions must remain candidate")
        condition = _object(item["condition"], "condition", {
            "task_family", "graph_stage", "required_tools",
        })
        if condition["task_family"] != dataset_key:
            raise ConfigurationError("candidate task_family differs from dataset")
        if not isinstance(condition["graph_stage"], str) or condition["graph_stage"] not in _STAGES:
            raise ConfigurationError("unsupported candidate graph_stage")
        if condition["required_tools"] != []:
            raise ConfigurationError("generic orchestration candidates require no fixed tool")
        _text(item["trigger"], "trigger", 600)
        action = _object(item["action"], "action", {"instruction"})
        _text(action["instruction"], "instruction", 800)


def load_candidate_skill_profile(
    path: str | Path,
    *,
    run_config: Mapping[str, Any],
    dataset_key: str = DATASET_KEY,
) -> dict[str, Any]:
    """Load only a declared profile using the existing YAML loader."""

    validate_candidate_skill_run_config(run_config)
    profile = load_yaml(path, expand_env=False)
    validate_candidate_skill_profile(profile, dataset_key=dataset_key)
    return profile


def build_candidate_prompt_priors(
    profile: Mapping[str, Any], *, dataset_key: str = DATASET_KEY
) -> tuple[dict[str, Any], ...]:
    """Use the existing forced-probe condition wire, not ACTIVE Skill IDs."""

    validate_candidate_skill_profile(profile, dataset_key=dataset_key)
    priors: list[dict[str, Any]] = []
    for item in profile["candidates"]:
        condition = dict(item["condition"])
        condition["required_tools"] = list(condition["required_tools"])
        action = dict(item["action"])
        content = (
            "Predeclared unvalidated candidate condition, not an ACTIVE Skill. "
            "This is an optional, rejectable prompt prior. "
            f"Applicable graph stage: {condition['graph_stage']}. "
            f"Only when: {item['trigger']} "
            "Otherwise ignore this suggestion. "
            f"Suggested action: {action['instruction']}"
        )
        if len(content) > 1800:
            raise ConfigurationError("candidate prompt prior exceeds the bounded content limit")
        priors.append({
            "condition_id": item["condition_id"],
            "application_mode": "forced_probe_condition",
            "condition": condition,
            "action": action,
            "content": content,
            "rejectable": True,
        })
    if sum(len(prior["content"]) for prior in priors) > 4000:
        raise ConfigurationError("candidate prompt priors exceed the combined content limit")
    return tuple(priors)
