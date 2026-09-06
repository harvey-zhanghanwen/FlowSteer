"""SkillFlow public-test reward adapter for MBPP+ TTB training.

The frozen MBPP+ fixed-100 population remains evaluator-only.  Training uses
the disjoint public MBPP tasks selected by the project data adapter and only
the assertions exposed in each task prompt.  Reward calculation is delegated
directly to ``SkillFlow/training/reward.py::code_test_pass_rate``; this module
only bridges that upstream scalar into the unified ``EvaluationOutcome``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

from .mbppplus_execution import extract_mbppplus_public_assertions
from .records import TaskRecord
from .task_evaluator import EvaluationOutcome


DEFAULT_SKILLFLOW_REWARD = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/reward.py"
)
MBPP_TRAINING_EVALUATOR_VERSION = (
    "skillflow.training.reward.code_test_pass_rate"
)


class SkillFlowRewardUnavailable(RuntimeError):
    """The deployed SkillFlow public-test reward cannot be loaded."""


def _load_skillflow_reward(path: Path) -> ModuleType:
    """Load the deployed SkillFlow reward module without copying its logic."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise SkillFlowRewardUnavailable(
            f"SkillFlow reward module is unavailable: {source}"
        )
    module_name = "_flowsteer_skillflow_training_reward"
    loaded = sys.modules.get(module_name)
    loaded_source = getattr(loaded, "__file__", None) if loaded else None
    if loaded_source and Path(str(loaded_source)).expanduser().resolve() == source:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise SkillFlowRewardUnavailable(
            f"cannot load SkillFlow reward module: {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_mbpp_public_tests(
    task: TaskRecord,
    prediction: str,
    *,
    timeout_seconds: float,
    skillflow_reward_path: str | Path = DEFAULT_SKILLFLOW_REWARD,
) -> EvaluationOutcome:
    """Return SkillFlow's exact public-test pass rate for a terminal program.

    ``timeout_seconds`` is retained at the project evaluator boundary for
    caller compatibility.  The upstream SkillFlow function has no timeout
    parameter, so the adapter does not claim process-isolation semantics.
    """

    if task.metadata.get("training_population") not in {
        "flowsteer_mbpp_train",
        "flowsteer_mbpp_validation",
    }:
        raise ValueError("task is not in the disjoint MBPP training population")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    public_test_program = task.metadata.get("public_test_program")
    try:
        tests = (
            tuple(
                line.strip()
                for line in public_test_program.splitlines()
                if line.strip()
            )
            if isinstance(public_test_program, str) and public_test_program.strip()
            else extract_mbppplus_public_assertions(task.question)
        )
    except ValueError as exc:
        return EvaluationOutcome(
            valid=False,
            reward=None,
            metrics={},
            reason="public_tests_unavailable",
            details={"error": str(exc)},
            evaluator_version=MBPP_TRAINING_EVALUATOR_VERSION,
        )

    module = _load_skillflow_reward(Path(skillflow_reward_path))
    reward_fn = getattr(module, "code_test_pass_rate", None)
    if not callable(reward_fn):
        raise SkillFlowRewardUnavailable(
            "SkillFlow reward module has no callable code_test_pass_rate"
        )

    # Public test cases come only from model-visible task fields.  Ground truth
    # and evaluator-private metadata are deliberately never passed upstream.
    source = prediction if isinstance(prediction, str) else ""
    # SkillFlow's code-generation reward converts its ``extra.test`` string to
    # one test case per non-empty line before calling code_test_pass_rate.
    raw_pass_rate = reward_fn(source, list(tests))
    if (
        isinstance(raw_pass_rate, bool)
        or not isinstance(raw_pass_rate, (int, float))
        or not math.isfinite(float(raw_pass_rate))
        or not 0.0 <= float(raw_pass_rate) <= 1.0
    ):
        raise ValueError("SkillFlow code_test_pass_rate returned an invalid score")

    pass_rate = float(raw_pass_rate)
    total = len(tests)
    passed = int(round(pass_rate * total))
    return EvaluationOutcome(
        valid=True,
        reward=pass_rate,
        metrics={
            "pass_at_1": float(pass_rate == 1.0),
            "public_test_pass_rate": pass_rate,
        },
        reason="evaluated",
        details={
            "passed": passed,
            "total": total,
            "evaluator_scope": "training_public_tests_only",
            "reward_function": MBPP_TRAINING_EVALUATOR_VERSION,
            "timeout_seconds_requested": float(timeout_seconds),
            "upstream_timeout_parameter": False,
        },
        evaluator_version=MBPP_TRAINING_EVALUATOR_VERSION,
    )


__all__ = [
    "DEFAULT_SKILLFLOW_REWARD",
    "MBPP_TRAINING_EVALUATOR_VERSION",
    "SkillFlowRewardUnavailable",
    "evaluate_mbpp_public_tests",
]
