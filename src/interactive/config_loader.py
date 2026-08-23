"""Environment-aware YAML loading without ever persisting resolved secrets."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit

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
    base_model = str(director.get("base_model", "")).lower().replace("_", "-")
    if not (("qwen3.5" in base_model or "qwen35" in base_model) and "9b" in base_model):
        raise ConfigurationError("the Flow-Director base model must be Qwen3.5-9B")
    director_host = urlsplit(str(director.get("api_base", ""))).hostname
    if director_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError("the Qwen3.5-9B Flow-Director endpoint must be local")
    if director.get("execute_on_edit") is not True:
        raise ConfigurationError("the progressive Canvas requires execute_on_edit=true")
    history_window = director.get("history_window")
    if (
        isinstance(history_window, bool)
        or not isinstance(history_window, int)
        or history_window < 1
    ):
        raise ConfigurationError("director.history_window must be a positive integer")

    graph = value["agent_graph"]
    legacy_actions = [
        "add_agent",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    subgraph_actions = [
        "add_subgraph",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    action_profile = graph.get("actions")
    if action_profile != legacy_actions and action_profile != subgraph_actions:
        raise ConfigurationError(
            "AgentGraph actions must use the legacy scalar profile or the "
            "FlowSteer-compatible add_subgraph profile"
        )
    if action_profile == subgraph_actions:
        if graph.get("max_agents_per_subgraph") != 3:
            raise ConfigurationError(
                "add_subgraph profile requires max_agents_per_subgraph=3"
            )
    if graph.get("contract_type") != "free_text" or graph.get("relation_encoding") != "two_bit":
        raise ConfigurationError("AgentGraph requires free-text contracts and two-bit relations")
    require_format_agent = graph.get("require_format_agent")
    if require_format_agent is not None and type(require_format_agent) is not bool:
        raise ConfigurationError(
            "agent_graph.require_format_agent must be boolean when configured"
        )
    max_agents = graph.get("max_agents")
    if isinstance(max_agents, bool) or not isinstance(max_agents, int) or max_agents < 1:
        raise ConfigurationError("agent_graph.max_agents must be a positive integer")
    if graph.get("executor_selection") not in {
        "seeded_weighted_random",
        "director_catalog_choice",
    }:
        raise ConfigurationError(
            "Agent executor selection must be seeded_weighted_random or "
            "director_catalog_choice"
        )
    terminal_protocols = graph.get("terminal_protocol_by_source", {})
    if not isinstance(terminal_protocols, Mapping):
        raise ConfigurationError(
            "agent_graph.terminal_protocol_by_source must be a mapping"
        )
    invalid_terminal_protocols = {
        str(source): protocol
        for source, protocol in terminal_protocols.items()
        if protocol not in {"none", "exact_single_answer_tag"}
    }
    if invalid_terminal_protocols:
        raise ConfigurationError(
            "terminal protocols must be none or exact_single_answer_tag"
        )
    semantic_protocols = graph.get("semantic_protocol_by_source", {})
    if not isinstance(semantic_protocols, Mapping):
        raise ConfigurationError(
            "agent_graph.semantic_protocol_by_source must be a mapping"
        )
    invalid_semantic_protocols = {
        str(source): protocol
        for source, protocol in semantic_protocols.items()
        if protocol
        not in {
            "none",
            "hotpotqa_verified_answer_slot_v1",
            "qa_verified_answer_lineage_v2",
        }
    }
    if invalid_semantic_protocols:
        raise ConfigurationError(
            "semantic protocols must be none, "
            "hotpotqa_verified_answer_slot_v1, or "
            "qa_verified_answer_lineage_v2"
        )
    recovery_policy = graph.get("recovery_policy", "default")
    if recovery_policy not in {
        "default",
        "preserve_diagnose_repair_augment",
    }:
        raise ConfigurationError(
            "agent_graph.recovery_policy must be default or "
            "preserve_diagnose_repair_augment"
        )
    required_evidence_tool_id = graph.get("required_evidence_tool_id")
    if required_evidence_tool_id is not None and (
        not isinstance(required_evidence_tool_id, str)
        or not required_evidence_tool_id.strip()
    ):
        raise ConfigurationError(
            "agent_graph.required_evidence_tool_id must be non-empty text or null"
        )
    hotpot_semantic_protocol = semantic_protocols.get("hotpotqa", "none")
    if any(
        source != "hotpotqa"
        and protocol == "hotpotqa_verified_answer_slot_v1"
        for source, protocol in semantic_protocols.items()
    ):
        raise ConfigurationError(
            "hotpotqa_verified_answer_slot_v1 is scoped only to hotpotqa"
        )
    invalid_shared_qa_sources = {
        str(source): protocol
        for source, protocol in semantic_protocols.items()
        if protocol == "qa_verified_answer_lineage_v2"
        and source not in {"hotpotqa", "triviaqa"}
    }
    if invalid_shared_qa_sources:
        raise ConfigurationError(
            "qa_verified_answer_lineage_v2 is scoped to hotpotqa and triviaqa"
        )
    if hotpot_semantic_protocol == "hotpotqa_verified_answer_slot_v1":
        if (
            value["experiment"].get("prompt_version")
            != "agentgraph.director.hotpotqa-semantic-recovery.v22"
        ):
            raise ConfigurationError(
                "HotpotQA verified answer-slot protocol requires the exact "
                "HotpotQA Director v22 observation contract"
            )
        if recovery_policy != "preserve_diagnose_repair_augment":
            raise ConfigurationError(
                "HotpotQA verified answer-slot protocol requires "
                "preserve_diagnose_repair_augment recovery"
            )
        if required_evidence_tool_id != "qa-retrieval":
            raise ConfigurationError(
                "HotpotQA verified answer-slot protocol requires the "
                "qa-retrieval evidence tool"
            )
        if terminal_protocols.get("hotpotqa") != "exact_single_answer_tag":
            raise ConfigurationError(
                "HotpotQA verified answer-slot protocol requires the exact "
                "single-answer terminal protocol"
            )
        qa_runtime = value.get("qa_tool_runtime")
        if (
            not isinstance(qa_runtime, Mapping)
            or qa_runtime.get("enabled") is not True
            or qa_runtime.get("completion_policy") != "required_evidence"
            or "hotpotqa" not in qa_runtime.get("dataset_scope", ())
        ):
            raise ConfigurationError(
                "HotpotQA verified answer-slot protocol requires the enabled "
                "qa_tool_runtime with required_evidence completion"
            )
    shared_qa_sources = tuple(
        str(source)
        for source, protocol in semantic_protocols.items()
        if protocol == "qa_verified_answer_lineage_v2"
    )
    if shared_qa_sources:
        if (
            value["experiment"].get("prompt_version")
            != "agentgraph.director.qa-semantic-recovery.v1"
        ):
            raise ConfigurationError(
                "qa_verified_answer_lineage_v2 requires the exact shared QA "
                "Director prompt"
            )
        if recovery_policy != "preserve_diagnose_repair_augment":
            raise ConfigurationError(
                "qa_verified_answer_lineage_v2 requires "
                "preserve_diagnose_repair_augment recovery"
            )
        if required_evidence_tool_id != "qa-retrieval":
            raise ConfigurationError(
                "qa_verified_answer_lineage_v2 requires the qa-retrieval "
                "evidence tool"
            )
        if any(
            terminal_protocols.get(source) != "exact_single_answer_tag"
            for source in shared_qa_sources
        ):
            raise ConfigurationError(
                "qa_verified_answer_lineage_v2 requires the exact "
                "single-answer terminal protocol"
            )
        qa_runtime = value.get("qa_tool_runtime")
        runtime_scope = (
            ()
            if not isinstance(qa_runtime, Mapping)
            else qa_runtime.get("dataset_scope", ())
        )
        if (
            not isinstance(qa_runtime, Mapping)
            or qa_runtime.get("enabled") is not True
            or qa_runtime.get("completion_policy") != "required_evidence"
            or any(source not in runtime_scope for source in shared_qa_sources)
        ):
            raise ConfigurationError(
                "qa_verified_answer_lineage_v2 requires an enabled "
                "qa_tool_runtime with required_evidence completion for every "
                "configured QA source"
            )
    if graph.get("max_bidirectional_block_size") != 2:
        raise ConfigurationError("AgentGraph v1 supports bidirectional blocks of size two")
    if graph.get("require_unique_output") is not True:
        raise ConfigurationError("AgentGraph requires exactly one output Agent")
    if graph.get("require_all_agents_reach_output") is not True:
        raise ConfigurationError("every Agent must reach the output Agent")

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
