"""Offline regressions for the runner's full ReAct-only config validation.

These exercise validate_completion_benchmark_config, beyond the general YAML
checks in test_healthbench_v246_config; no runtime or evaluator is constructed.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import re

import pytest

from src.interactive.config_loader import ConfigurationError, load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_v246_settings",
    _ROOT / "scripts" / "evaluate_completion_benchmark_round.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_TOOLS = [
    "healthbench-authoritative.search",
    "healthbench-medrag.search",
    "healthbench-source.read",
    "healthbench-drug.lookup",
    "healthbench-computation.calculator",
    "healthbench-knowledge.search",
    "healthbench-literature.search",
    "healthbench-trials.search",
    "healthbench-bookshelf.search",
    "healthbench-pdq.search",
    "healthbench-ahrq.search",
    "healthbench-terminology.search",
]


def _config(suffix="dev5"):
    return load_yaml(
        _ROOT
        / f"config/evaluation_healthbench_professional_candidate_skill_v2_46_{suffix}.yaml",
        expand_env=False,
    )


def _assert_rejected(config, check):
    with pytest.raises(ConfigurationError, match=re.escape(check)):
        _MODULE.validate_completion_benchmark_config(config)


@pytest.mark.parametrize("suffix", ("dev5", "full525"))
def test_actual_v246_full_toolset_react_only_settings_are_accepted(suffix):
    config = _config(suffix)
    bounded = config["healthbench_professional_evaluation"]
    runtime = config["healthbench_tool_runtime"]
    assert bounded["direct_execution_mode"] == "react"
    assert bounded["protocol_equivalent_to_direct"] is False
    assert bounded["direct_allowed_tools"] == _TOOLS
    assert runtime["clinical_reference_sources_enabled"] is True
    assert runtime["execution_profile_allowlist"] == [
        {"execution_mode": "react", "allowed_tools": _TOOLS}
    ]

    _MODULE.validate_completion_benchmark_config(config)


@pytest.mark.parametrize("missing_tool", _TOOLS)
@pytest.mark.parametrize("omit_from_direct_also", (False, True))
def test_react_only_settings_reject_any_missing_tool(missing_tool, omit_from_direct_also):
    config = _config()
    profile = config["healthbench_tool_runtime"]["execution_profile_allowlist"][0]
    profile["allowed_tools"].remove(missing_tool)
    if omit_from_direct_also:
        # Matching partial Direct/Graph profiles still cannot evade the registry check.
        config["healthbench_professional_evaluation"]["direct_allowed_tools"].remove(
            missing_tool
        )
        expected_check = "healthbench.direct_allowed_tools"
    else:
        expected_check = "healthbench_tool_runtime.execution_profile_allowlist"

    _assert_rejected(config, expected_check)


@pytest.mark.parametrize("execution_mode", ("reasoning", "coding", ""))
def test_react_only_settings_reject_wrong_graph_execution_mode(execution_mode):
    config = _config()
    config["healthbench_tool_runtime"]["execution_profile_allowlist"][0][
        "execution_mode"
    ] = execution_mode

    _assert_rejected(config, "healthbench_tool_runtime.execution_profile_allowlist")


@pytest.mark.parametrize(
    ("execution_mode", "expected_check"),
    (
        ("reasoning", "healthbench_tool_runtime.disabled"),
        ("coding", "healthbench.direct_execution_mode"),
    ),
)
def test_react_only_settings_reject_wrong_direct_execution_mode(execution_mode, expected_check):
    config = _config()
    config["healthbench_professional_evaluation"]["direct_execution_mode"] = execution_mode

    _assert_rejected(config, expected_check)


@pytest.mark.parametrize("suffix", ("dev5", "full525"))
@pytest.mark.parametrize("source_specific_profiles", (False, True))
def test_legacy_clinical_reasoning_and_react_allowlists_remain_accepted(
    suffix, source_specific_profiles
):
    config = _config(suffix)
    runtime = config["healthbench_tool_runtime"]
    full_react_profile = deepcopy(runtime["execution_profile_allowlist"][0])
    profiles = [{"execution_mode": "reasoning", "allowed_tools": []}]
    if source_specific_profiles:
        profiles.extend(
            {"execution_mode": "react", "allowed_tools": [tool]}
            for tool in _TOOLS
        )
    profiles.append(full_react_profile)
    runtime["execution_profile_allowlist"] = profiles

    _MODULE.validate_completion_benchmark_config(config)


def test_clinical_react_only_settings_still_reject_protocol_equivalence_claim():
    config = _config()
    config["healthbench_professional_evaluation"]["protocol_equivalent_to_direct"] = True

    _assert_rejected(config, "healthbench_tool_runtime.execution_profile_allowlist")


def test_clinical_react_only_settings_still_reject_unrecognized_toolset():
    config = _config()
    config["healthbench_tool_runtime"]["toolset"] = "unsupported_clinical_v1"

    _assert_rejected(config, "healthbench.direct_allowed_tools")
