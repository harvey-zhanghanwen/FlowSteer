"""Offline profile and collector wiring; no model, Tool HTTP or grading."""
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from scripts.healthbench_candidate_skill_profile import load_candidate_skill_profile, build_candidate_prompt_priors
from src.interactive.config_loader import load_yaml
from src.interactive.records import EvaluationReceipt
from tests.unit.test_completion_benchmark_round import _MODULE as completion
from tests.unit.test_hotpotqa_round import _MODULE as collector

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/evaluation_healthbench_professional_v233_all_sources_dev5.yaml"


def test_base_orchestration_and_evaluator_preserved_with_declared_tool_adaptations():
    current = load_yaml(CONFIG, expand_env=False)
    base = load_yaml(ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context.yaml", expand_env=False)
    completion.validate_completion_benchmark_config(current)
    graph = deepcopy(current["agent_graph"])
    graph.pop("local_agent_context_budget")
    graph["model_catalog_path"] = base["agent_graph"]["model_catalog_path"]
    assert graph == base["agent_graph"]
    director = deepcopy(current["director"])
    director["api_base"] = base["director"]["api_base"]
    assert director == base["director"]
    assert current["evaluation"] == base["evaluation"]
    assert current["execution_timeout"] == base["execution_timeout"]
    assert current["experiment"]["prompt_version"] == base["experiment"]["prompt_version"]
    bounded = current["healthbench_professional_evaluation"]
    assert bounded["sample_count"] == len(bounded["task_ids"]) == 5
    assert len(set(bounded["direct_allowed_tools"])) == 12
    assert current["healthbench_tool_runtime"]["execution_profile_allowlist"] == [
        {"execution_mode": "reasoning", "allowed_tools": []},
        {"execution_mode": "react", "allowed_tools": bounded["direct_allowed_tools"]},
    ]
    assert not any(k.startswith("direct_reference") or k == "direct_reused_from" for k in bounded)
    assert current["healthbench_tool_runtime"]["require_initial_search"] is False
    assert current["healthbench_tool_runtime"]["require_refinement_on_insufficient_evidence"] is False


def test_three_rejectable_candidates_reach_existing_collect_without_active_publication(tmp_path):
    config = load_yaml(CONFIG)
    profile = load_candidate_skill_profile(ROOT / config["candidate_skill_evaluation"]["profile_path"], run_config=config)
    priors = build_candidate_prompt_priors(profile)
    task = collector.TaskRecord(task_id="healthbench-professional:synthetic", question="Synthetic conversation",
        ground_truth="", split="test", metadata={"dataset_key": "healthbench_professional"})
    received = []

    class Backend:
        model_catalog_version = "synthetic-catalog"
        evidence_store = SimpleNamespace(trajectories=SimpleNamespace(payloads=lambda: ()))

        async def collect(self, task, rollout_index, versions, **kwargs):
            received.append(kwargs)
            return collector.TrajectoryRecord(
                trajectory_id="synthetic-trajectory", task=task, group_id="synthetic-group",
                condition_id=config["experiment"]["condition_id"], rollout_id="synthetic-rollout",
                versions=versions, turns=(), final_answer="Synthetic answer",
                evaluation=EvaluationReceipt(versions.evaluator, True, 0.5,
                    metrics={"overall_score": 0.5, "overall_score_length_adjusted": 0.5}),
                termination_reason="finish", explicit_finish=True,
            )

    compatibility = completion._compatibility_config(config, config["healthbench_professional_evaluation"])
    compatibility["storage"] = {}
    result = asyncio.run(collector._collect_graph(
        Backend(), (task,), compatibility, tmp_path / "trajectories.jsonl", [], {},
        tmp_path / "manifest.json", prompt_priors=priors,
    ))
    assert set(result) == {task.task_id}
    assert received == [{"expected_task_split": "test", "prompt_priors": priors, "forced_probe": True}]
    assert len(priors) == 3 and all(p["rejectable"] for p in priors)
    assert not config["skills"]["enabled"] and not config["exploration"]["enabled"]
