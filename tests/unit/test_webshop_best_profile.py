from __future__ import annotations

import json
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _ROOT / "config" / "webshop_best_profile_v1.yaml"
_CURRENT_POINTER_PATH = _ROOT / "config" / "webshop_best_profile.yaml"


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_webshop_best_profile_points_to_exact_executable_condition() -> None:
    profile = _yaml(_PROFILE_PATH)
    target = _yaml(_ROOT / profile["selection"]["executable_config_path"])

    assert profile["dataset_key"] == "webshop"
    assert profile["selection"]["eligible_condition_count"] == 1
    assert target["experiment"]["condition_id"] == profile["selection"][
        "selected_condition_id"
    ]
    assert target["webshop_evaluation"]["split"] == "validation"
    assert target["webshop_evaluation"]["sample_count"] == 128
    assert target["experiment"]["training_enabled"] is False
    assert target["environment_runtime"]["enabled"] is True
    assert target["grpo"]["enabled"] is False
    assert target["skills"]["enabled"] is False


def test_webshop_best_profile_metrics_match_tracked_formal_report() -> None:
    profile = _yaml(_PROFILE_PATH)
    report = json.loads(
        (_ROOT / profile["selection"]["evaluation_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    selected = profile["verified_metrics"]

    assert report["sample_count"] == profile["selection_scope"]["sample_count"]
    assert report["primary_metric"] == profile["selection_scope"]["primary_metric"]
    assert report["agentgraph"]["strict_average_score"] == selected[
        "agentgraph"
    ]["strict_average_score"]
    assert report["agentgraph"]["strict_success_rate"] == selected[
        "agentgraph"
    ]["strict_success_rate"]
    assert report["agentgraph"]["evaluator_valid"] == selected["agentgraph"][
        "evaluator_valid"
    ]
    assert report["direct_local_baseline"]["strict_average_score"] == selected[
        "direct"
    ]["strict_average_score"]
    assert report["agentgraph_minus_direct"] == {
        "average_score": selected["agentgraph_minus_direct"][
            "strict_average_score"
        ],
        "success": selected["agentgraph_minus_direct"]["strict_success_rate"],
        "success_rate": selected["agentgraph_minus_direct"][
            "strict_success_rate"
        ],
    }


def test_webshop_next_run_pointer_is_explicit() -> None:
    profile = _yaml(_PROFILE_PATH)
    current = _yaml(_CURRENT_POINTER_PATH)

    assert current["selected_profile_path"] == str(_PROFILE_PATH.relative_to(_ROOT))
    assert current["selected_condition_id"] == profile["selection"][
        "selected_condition_id"
    ]
    assert current["next_run_config_path"] == profile["selection"][
        "executable_config_path"
    ]
    assert profile["next_run"]["selected"] is True
    assert profile["next_run"]["runner_requires_explicit_config_argument"] is True
    assert profile["next_run"]["automatic_global_default_resolver"] is False
    assert profile["next_run"]["full_evaluation_rerun_authorized"] is False


def test_webshop_higher_leaking_condition_is_explicitly_ineligible() -> None:
    profile = _yaml(_PROFILE_PATH)
    excluded = {
        item["condition"]: item["reason"]
        for item in profile["excluded_conditions"]
    }

    reason = excluded["webshop_ragen_environment_native_action_v4_stable_zero"]
    assert "evaluator-private" in reason
    assert "26 of 128" in reason

