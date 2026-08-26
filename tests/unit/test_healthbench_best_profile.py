from __future__ import annotations

import json
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_POINTER_PATH = _ROOT / "config" / "healthbench_professional_best_profile_v1.yaml"
_CURRENT_POINTER_PATH = (
    _ROOT / "config" / "healthbench_professional_best_profile.yaml"
)


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_healthbench_best_profile_points_to_exact_executable_condition():
    pointer = _yaml(_POINTER_PATH)
    target = _yaml(_ROOT / pointer["selection"]["executable_config_path"])

    assert pointer["dataset_key"] == "healthbench_professional"
    assert pointer["selection"]["eligible_condition_count"] == 1
    assert target["experiment"]["condition_id"] == pointer["selection"][
        "selected_condition_id"
    ]
    assert target["healthbench_professional_evaluation"]["split"] == "test"
    assert target["healthbench_professional_evaluation"]["sample_count"] == 525
    assert target["experiment"]["training_enabled"] is False
    assert target["healthbench_tool_runtime"]["enabled"] is False
    assert target["grpo"]["enabled"] is False
    assert target["skills"]["enabled"] is False


def test_healthbench_best_profile_metrics_match_tracked_formal_report():
    pointer = _yaml(_POINTER_PATH)
    report_path = _ROOT / pointer["selection"]["evaluation_report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = pointer["verified_metrics"]

    assert report["sample_count"] == pointer["selection_scope"]["sample_count"]
    assert report["primary_metric"] == pointer["selection_scope"]["primary_metric"]
    assert report["agentgraph"] == {
        "completed": selected["agentgraph"]["completed"],
        "completed_only_overall_score": selected["agentgraph"][
            "valid_only_overall_score"
        ],
        "completed_only_overall_score_length_adjusted": selected["agentgraph"][
            "valid_only_overall_score_length_adjusted"
        ],
        "denominator": selected["agentgraph"]["denominator"],
        "evaluator_valid": selected["agentgraph"]["evaluator_valid"],
        "strict_overall_score": selected["agentgraph"]["strict_overall_score"],
        "strict_overall_score_length_adjusted": selected["agentgraph"][
            "strict_overall_score_length_adjusted"
        ],
    }
    assert report["direct_local_baseline"][
        "strict_overall_score_length_adjusted"
    ] == selected["direct"]["strict_overall_score_length_adjusted"]
    assert report["agentgraph_minus_direct"] == {
        "overall_score": selected["agentgraph_minus_direct"][
            "strict_overall_score"
        ],
        "overall_score_length_adjusted": selected["agentgraph_minus_direct"][
            "strict_overall_score_length_adjusted"
        ],
    }


def test_healthbench_next_run_pointer_is_explicit_and_not_a_fabricated_default():
    pointer = _yaml(_POINTER_PATH)
    current = _yaml(_CURRENT_POINTER_PATH)
    next_run = pointer["next_run"]

    assert current["selected_profile_path"] == str(
        _POINTER_PATH.relative_to(_ROOT)
    )
    assert current["selected_condition_id"] == pointer["selection"][
        "selected_condition_id"
    ]
    assert current["next_run_config_path"] == pointer["selection"][
        "executable_config_path"
    ]
    assert next_run["selected"] is True
    assert next_run["executable_config_path"] == pointer["selection"][
        "executable_config_path"
    ]
    assert next_run["runner_requires_explicit_config_argument"] is True
    assert next_run["automatic_global_default_resolver"] is False
    assert next_run["full_evaluation_rerun_authorized"] is False
    assert current["runner_requires_explicit_config_argument"] is True
    assert current["automatic_global_default_resolver"] is False


def test_local_manifest_matches_pointer_when_private_receipts_are_present():
    pointer = _yaml(_POINTER_PATH)
    manifest_path = _ROOT / pointer["selection"]["run_manifest_path"]
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["fixed_split"] == pointer["selection_scope"]["split"]
    assert manifest["sample_count"] == pointer["selection_scope"]["sample_count"]
    assert manifest["agentgraph_progress"]["completed"] == 525
    assert manifest["agentgraph_progress"]["pending_evaluator_retries"] == 0
    assert manifest["metrics"]["agentgraph"][
        "strict_overall_score_length_adjusted"
    ] == pointer["verified_metrics"]["agentgraph"][
        "strict_overall_score_length_adjusted"
    ]
