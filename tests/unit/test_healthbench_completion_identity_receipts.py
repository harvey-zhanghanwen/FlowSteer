from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from src.interactive.config_loader import load_model_registry, load_yaml
from src.interactive.records import TaskRecord


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_identity_test",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _config() -> dict:
    value = deepcopy(
        load_yaml(
            _ROOT
            / "config"
            / (
                "evaluation_healthbench_professional_mixed_all_thinking_v2_22_"
                "heldout20_scope_preserving_all_model_react.yaml"
            )
        )
    )
    value["healthbench_professional_evaluation"][
        "protocol_equivalent_to_direct"
    ] = False
    value["healthbench_tool_runtime"]["execution_profile_allowlist"] = [
        {"execution_mode": "reasoning", "allowed_tools": []},
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-authoritative.search"],
        },
    ]
    return value


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="healthbench-professional:identity-receipt-test",
        question=(
            "Conversation messages (respond to the final user message):\n"
            '{"messages":[{"role":"user","content":"What should I discuss?"}]}'
        ),
        ground_truth=None,
        split="test",
        metadata={"dataset_key": "healthbench_professional"},
    )


def _identity(config: dict, task: TaskRecord):
    graph = config["agent_graph"]
    registry = load_model_registry(_ROOT / graph["model_catalog_path"])
    bounded = config["healthbench_professional_evaluation"]
    base_seed = int(config["experiment"]["seed"])
    coordinate = _MODULE._direct_scientific_sampling_coordinate(
        config,
        task,
        base_seed=base_seed,
    )
    identity = _MODULE._healthbench_direct_generation_identity(
        SimpleNamespace(config=config, registry=registry),
        task,
        model_id=bounded["direct_model_id"],
        protocol=bounded["direct_protocol"],
        contract=bounded["direct_contract"],
        seed=base_seed,
        coordinate=coordinate,
    )
    return registry, coordinate, identity


def _react_model_calls(identity: dict, coordinate) -> list[dict]:
    scientific = identity["scientific_sampling"]
    requested = scientific["requested_sampling"]
    base_seed = scientific["base_seed"]
    turn = 1
    generation_seed = _MODULE.derive_generation_seed(
        base_seed=base_seed,
        coordinate=coordinate,
        step_index=turn,
        phase=_MODULE.GenerationPhase.ACTION,
    )
    receipt_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": requested["top_k"],
        "max_tokens": requested["max_tokens"],
        "seed": generation_seed,
        "chat_template_enable_thinking": requested[
            "chat_template_enable_thinking"
        ],
    }
    provider_sampling = dict(receipt_sampling)
    thinking_budget = requested.get("thinking_budget")
    if thinking_budget is not None:
        provider_sampling.update(
            max_tokens=requested["max_tokens"] + thinking_budget,
            visible_max_tokens=requested["max_tokens"],
            thinking_budget=thinking_budget,
        )
    provider_sampling["repetition_penalty"] = requested["repetition_penalty"]
    return [
        {
            "turn": turn,
            "request_id": "identity-receipt-test:react:1",
            "request_status": "completed",
            "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
            "scientific_sampling": {
                "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
                "base_seed": base_seed,
                "coordinate": coordinate.to_value(),
                "phase": _MODULE.GenerationPhase.ACTION.value,
                "step_index": turn,
                "generation_seed": generation_seed,
                "requested_sampling": receipt_sampling,
            },
            "requested_sampling": provider_sampling,
            "metadata": {
                "generation_seed": generation_seed,
                "requested_sampling": provider_sampling,
            },
        }
    ]


def _direct_value(identity: dict, coordinate) -> dict:
    model_calls = _react_model_calls(identity, coordinate)
    requested = identity["scientific_sampling"]["requested_sampling"]
    sampling_receipt = _MODULE._react_scientific_sampling_receipt(
        model_calls,
        base_seed=identity["scientific_sampling"]["base_seed"],
        coordinate=coordinate,
        max_action_tokens=requested["max_tokens"],
        expected_top_k=requested["top_k"],
        expected_repetition_penalty=requested["repetition_penalty"],
        expected_chat_template_enable_thinking=requested[
            "chat_template_enable_thinking"
        ],
        expected_thinking_budget=requested.get("thinking_budget"),
    )
    return {
        "direct_generation_identity": identity,
        "generation_identity_verified": True,
        "execution": {"metadata": {"response": {"model_calls": model_calls}}},
        "scientific_sampling_receipt": sampling_receipt,
    }


def _execution(agent_id: str, execution_mode: str, response: dict) -> dict:
    allowed_tools = (
        ["healthbench-authoritative.search"]
        if execution_mode == "react"
        else []
    )
    return {
        "agent_id": agent_id,
        "metadata": {
            "request": {
                "agent": {
                    "id": agent_id,
                    "execution_mode": execution_mode,
                    "allowed_tools": allowed_tools,
                }
            },
            "response": response,
        },
    }


def _mixed_trajectory(task: TaskRecord, registry, identity, coordinate) -> dict:
    base_seed = identity["scientific_sampling"]["base_seed"]
    return {
        "task": task.to_dict(),
        "condition_id": identity["condition_id"],
        "versions": {
            "model_catalog": registry.catalog_id,
            "tool": identity["tool"]["tool_version"],
        },
        "director_sampling": {
            "algorithm": _MODULE.SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": _MODULE.GenerationPhase.ACTION.value,
        },
        "sampling_receipt_verified": True,
        "turns": [
            {
                "executions": [
                    _execution(
                        "reasoner",
                        "reasoning",
                        {
                            "request_status": "completed",
                            "generation_seed": base_seed,
                            "requested_sampling": {"seed": base_seed},
                        },
                    ),
                    _execution(
                        "retriever",
                        "react",
                        {
                            "execution_mode": "react",
                            "model_calls": _react_model_calls(identity, coordinate),
                        },
                    ),
                ]
            }
        ],
        "explicit_finish": True,
        "termination_reason": "finish",
    }


def test_authoritative_retry_settings_are_resume_identity_with_legacy_defaults():
    config = _config()
    web = config["healthbench_tool_runtime"]["authoritative_web_search"]
    web.pop("max_retries", None)
    web.pop("retry_backoff_seconds", None)
    task = _task()
    _, coordinate, identity = _identity(config, task)
    retrieval = identity["authoritative_retrieval"]
    assert retrieval["web_max_retries"] == 0
    assert retrieval["web_retry_backoff_seconds"] == 1.0

    value = _direct_value(identity, coordinate)
    historical_identity = deepcopy(identity)
    historical = historical_identity["authoritative_retrieval"]
    historical.pop("web_max_retries")
    historical.pop("web_retry_backoff_seconds")
    value["direct_generation_identity"] = historical_identity
    assert _MODULE._persisted_healthbench_direct_identity_matches(value, identity)

    retry_config = deepcopy(config)
    retry_config["healthbench_tool_runtime"]["authoritative_web_search"].update(
        max_retries=2,
        retry_backoff_seconds=0.25,
    )
    _, _, retry_identity = _identity(retry_config, task)
    assert not _MODULE._persisted_healthbench_direct_identity_matches(
        value,
        retry_identity,
    )


def test_mixed_executor_profiles_verify_receipts_but_remain_descriptive():
    config = _config()
    task = _task()
    registry, coordinate, identity = _identity(config, task)
    direct_value = _direct_value(identity, coordinate)
    trajectory = _mixed_trajectory(task, registry, identity, coordinate)

    graph_check = _MODULE._graph_generation_identity_check(trajectory, identity)
    assert graph_check["verified"] is True
    assert graph_check["executor_profile_receipt_status"] == "verified"
    assert graph_check["executor_execution_receipt_count"] == 2
    assert graph_check["executor_react_sampling_status"] == "verified"

    row = {
        "task_id": task.task_id,
        "failure_type": "none",
        "direct": {
            **direct_value,
            "available": True,
            "valid": True,
            "overall_score": 0.4,
            "overall_score_length_adjusted": 0.4,
            "evaluation": {"valid": True, "details": {}},
            "telemetry": {},
        },
        "agentgraph": {
            "available": True,
            "valid": True,
            "overall_score": 0.5,
            "overall_score_length_adjusted": 0.5,
            "evaluation": {"valid": True, "details": {}},
            "explicit_finish": True,
            "termination_reason": "finish",
            "telemetry": {},
            "graph_diagnostic": None,
        },
    }
    report = _MODULE._report([row], config, [trajectory])
    assert report["paired_generation_identity"]["verified"] is True
    assert report["protocol_equivalent_to_direct"] is False
    assert report["comparison_interpretation"] == (
        "separate_protocol_descriptive_comparison"
    )

    reasoning_only = deepcopy(trajectory)
    reasoning_only["turns"][0]["executions"] = reasoning_only["turns"][0][
        "executions"
    ][:1]
    graph_check = _MODULE._graph_generation_identity_check(
        reasoning_only,
        identity,
    )
    assert graph_check["verified"] is True
    assert graph_check["executor_react_sampling_status"] == (
        "not_applicable_no_react_execution"
    )

    disallowed = deepcopy(trajectory)
    disallowed["turns"][0]["executions"][0]["metadata"]["request"]["agent"][
        "allowed_tools"
    ] = ["healthbench-authoritative.search"]
    graph_check = _MODULE._graph_generation_identity_check(disallowed, identity)
    assert graph_check["verified"] is False
    assert graph_check["executor_profile_receipt_status"] == "profile_not_admitted"
