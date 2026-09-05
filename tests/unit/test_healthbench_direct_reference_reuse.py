"""Offline reference-control tests; no model, retrieval or grader calls."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hb_reference_receipt_fixtures",
    Path(__file__).with_name("test_healthbench_completion_identity_receipts.py"),
)
assert SPEC and SPEC.loader
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)
R = FIXTURES._MODULE


def _write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _case(tmp_path, monkeypatch, valid_count=2, failure_count=1):
    for key, value in {
        "FLOWSTEER_ROLLOUT_GPU": "5", "FLOWSTEER_SUPERVISOR_PORT": "8025",
        "FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH": "32768",
        "FLOWSTEER_SUPERVISOR_MEM_FRACTION": "0.82",
    }.items():
        monkeypatch.setenv(key, value)
    source = R.load_yaml(ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_32_full525_receipt_bound_completion.yaml")
    section = "healthbench_professional_evaluation"
    source[section]["sample_count"] = valid_count + failure_count
    for key in ("train_path", "validation_path", "test_path"):
        source["data"][key] = str(tmp_path / (key + ".jsonl"))
    source["evaluation"]["healthbench_private_cases_path"] = str(tmp_path / "private.jsonl")
    for name, key in (("agent_graph", "model_catalog_path"), ("evaluation", "healthbench_judge_catalog_path")):
        source[name][key] = str(ROOT / source[name][key])
    for key, value in source["storage"].items():
        if key.endswith("_path"):
            source["storage"][key] = str(tmp_path / "source" / Path(value).name)
    source_path = tmp_path / "source/config/baseline.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(yaml.safe_dump(source))
    source_paths = R._paths(source, source_path.parent.parent)
    tasks = tuple(R.TaskRecord(task_id=f"healthbench-professional:reference-{index}",
                              question=f"Conversation question {index}", ground_truth=None,
                              split="test", metadata={"dataset_key": "healthbench_professional"})
                  for index in range(valid_count + failure_count))
    _write_jsonl(source_paths["selected"], [{"schema_version": R.TASK_SCHEMA_VERSION, **task.to_dict()} for task in tasks])
    rows = []
    for task in tasks[:valid_count]:
        _, coordinate, identity = FIXTURES._identity(source, task)
        rows.append({**FIXTURES._direct_value(identity, coordinate),
                     "task_id": task.task_id, "task": task.to_dict(),
                     "runtime_condition_id": source["experiment"]["condition_id"],
                     "simple_baseline_topology": "single_react_agent",
                     "model_id": source[section]["direct_model_id"],
                     "protocol": source[section]["direct_protocol"],
                     "generation_seed": source[section]["direct_generation_seed"],
                     "tool_version": source["experiment"]["tool_version"],
                     "tool_resource_ids": source[section]["direct_allowed_tools"],
                     "final_answer": "An unchanged assistant response.",
                     "evaluation": {"valid": True, "evaluator_version": R.evaluator_version_for(task),
                                    "metrics": {"overall_score": 0.4, "overall_score_length_adjusted": 0.5},
                                    "details": {"judge_model": "gpt-5.4-2026-03-05"}}})
    _write_jsonl(source_paths["direct"], rows)
    failures = [{"task_id": task.task_id, "condition": "direct_local_qwen35_9b",
                 "stage": "generation_or_evaluator", "recorded_at": "2026-09-05T00:00:00Z",
                 "error": "AgentRuntimeError: react agent 'direct_react_agent' exhausted 6 turns without a valid completion"}
                for task in tasks[valid_count:]]
    _write_jsonl(source_paths["failures"], failures + [
        {"task_id": tasks[0].task_id, "condition": "agentgraph", "stage": "generation_or_evaluator",
         "error": "react agent 'node_1' exhausted 6 turns without a valid completion"}])
    manifest = {"config_path": str(source_path), "artifacts": {key: str(value) for key, value in source_paths.items()},
                "status": "interrupted_for_v233", "sample_count": len(tasks),
                "selected_task_ids": [task.task_id for task in tasks],
                "runtime_resource": {"effective_rollout_physical": 5, "supervisor_port": 8025,
                                     "context_length": 32768, "mem_fraction_static": 0.82,
                                     "evaluation_concurrency": 4, "task_timeout_seconds": 900,
                                     "sglang_server_runtime": {"weight_version": "default", "context_length": 32768,
                                          "attention_backend": "fa3", "sampling_backend": "flashinfer",
                                          "enable_deterministic_inference": False, "max_running_requests": 8}},
                "direct_progress": {"completed": valid_count, "frozen_react_terminal_failures": failure_count,
                                    "strict_zero_terminal_failures": failure_count, "pending_evaluator_retries": 0}}
    source_paths["manifest"].write_text(json.dumps(manifest))
    candidate = deepcopy(source)
    candidate["experiment"].update(name="new-graph", condition_id="new-graph", prompt_version="agentgraph.director.minimal-neutral.v19")
    candidate["healthbench_tool_runtime"]["condition_id"] = "new-graph"
    candidate["agent_graph"]["artifact_communication_profile"] = "producer_context_structured_evidence_v3"
    candidate[section].update(direct_reference_config=str(source_path),
                              direct_reference_manifest=str(source_paths["manifest"]),
                              direct_reused_from=str(source_paths["direct"]),
                              direct_failures_reused_from=str(source_paths["failures"]),
                              direct_artifact_communication_profile="producer_context_structured_evidence_v2")
    for key, value in candidate["storage"].items():
        if key.endswith("_path"):
            candidate["storage"][key] = str(tmp_path / "target" / Path(value).name)
    return candidate, tasks, rows, source_paths


def test_exact_459_valid_66_frozen_control_no_generation_or_regrading(tmp_path, monkeypatch):
    config, tasks, originals, paths = _case(tmp_path, monkeypatch, 459, 66)
    reference = R._healthbench_direct_reference(config, tmp_path, tasks)
    assert reference["receipt"]["sample_count"] == 525
    assert len(reference["records"]) == 459
    assert len(reference["failures"]) == 66
    assert R._read_jsonl(paths["direct"]) == originals
    for original in originals:
        copy = reference["records"][original["task_id"]]
        assert {key: value for key, value in copy.items() if key != "reuse_receipt"} == original
        assert copy["runtime_condition_id"] != "new-graph"

    async def forbidden(*args, **kwargs):
        raise AssertionError("No generation or regrading is allowed for the frozen control")
    monkeypatch.setattr(R, "_direct_one", forbidden)
    monkeypatch.setattr(R, "_evaluate_prediction", forbidden)
    manifest, failures = {}, []
    target_paths = R._paths(config, tmp_path)
    values = asyncio.run(R._collect_direct(SimpleNamespace(), tasks, config, tmp_path,
                         target_paths["direct"], failures, manifest, target_paths["manifest"],
                         direct_reference=reference))
    assert manifest["direct_progress"]["newly_collected_records"] == 0
    assert len(failures) == 66
    report = R._report(R._paired_rows(tasks, values, {}, "healthbench_professional"), config,
                       collection_failures=failures)
    assert report["direct_local_baseline"]["denominator"] == 525
    assert report["direct_local_baseline"]["evaluator_valid"] == 459
    assert report["direct_local_baseline"]["strict_overall_score"] == pytest.approx(459 * 0.4 / 525)


@pytest.mark.parametrize("section,key,value", [
    ("healthbench_professional_evaluation", "direct_model_id", "qwen3.5-flash"),
    ("healthbench_professional_evaluation", "direct_contract", "Changed prompt"),
    ("healthbench_professional_evaluation", "direct_generation_seed", 7),
    ("healthbench_professional_evaluation", "direct_artifact_communication_profile", "producer_context_structured_evidence_v3"),
    ("healthbench_tool_runtime", "max_successful_queries", 3),
    ("director", "max_action_tokens", 8192),
    ("evaluation", "healthbench_reasoning_effort", "high"),
])
def test_changed_generation_condition_rejected_before_calls(tmp_path, monkeypatch, section, key, value):
    config, tasks, _, _ = _case(tmp_path, monkeypatch)
    config[section][key] = value
    with pytest.raises(R.CompletionBenchmarkRoundError, match="Direct reference rejected"):
        R._healthbench_direct_reference(config, tmp_path, tasks)


@pytest.mark.parametrize("failure", ["pending", "overlap", "wrong_horizon", "wrong_manifest", "invalid_grade"])
def test_incomplete_or_misattributed_control_rejected(tmp_path, monkeypatch, failure):
    config, tasks, rows, paths = _case(tmp_path, monkeypatch)
    manifest = json.loads(paths["manifest"].read_text())
    if failure == "pending":
        manifest["direct_progress"]["pending_evaluator_retries"] = 1
    elif failure == "wrong_manifest":
        manifest["config_path"] = "/wrong/source/config.yaml"
    elif failure == "invalid_grade":
        rows[0]["evaluation"]["valid"] = False
        _write_jsonl(paths["direct"], rows)
    else:
        failures = R._read_jsonl(paths["failures"])
        if failure == "overlap":
            failures[0]["task_id"] = tasks[0].task_id
        else:
            failures[0]["error"] = failures[0]["error"].replace("6 turns", "5 turns")
        _write_jsonl(paths["failures"], failures)
    paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(R.CompletionBenchmarkRoundError, match="Direct reference rejected"):
        R._healthbench_direct_reference(config, tmp_path, tasks)


def test_explicit_cross_condition_pairing_keeps_raw_condition_and_common_receipts(tmp_path, monkeypatch):
    config, tasks, _, _ = _case(tmp_path, monkeypatch)
    reference = R._healthbench_direct_reference(config, tmp_path, tasks)
    value = reference["records"][tasks[0].task_id]
    source = R.load_yaml(Path(config["healthbench_professional_evaluation"]["direct_reference_config"]))
    registry, coordinate, identity = FIXTURES._identity(source, tasks[0])
    trajectory = FIXTURES._mixed_trajectory(tasks[0], registry, identity, coordinate)
    trajectory["condition_id"] = "new-graph"
    assert not R._graph_generation_identity_check(trajectory, identity)["verified"]
    assert R._graph_generation_identity_check(trajectory, identity, graph_condition_id="new-graph")["verified"]
    row = {"task_id": tasks[0].task_id, "direct": value}
    assert R._paired_generation_identity_receipt([row], [trajectory], config)["verified"]
    del value["reuse_receipt"]
    assert not R._paired_generation_identity_receipt([row], [trajectory], config)["verified"]


def test_opt_in_cannot_silently_fall_back_to_generation(tmp_path, monkeypatch):
    config, tasks, _, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(R.CompletionBenchmarkRoundError, match="before API preflight"):
        asyncio.run(R._collect_direct(SimpleNamespace(), tasks, config, tmp_path,
                    tmp_path / "direct.jsonl", [], {}, tmp_path / "manifest.json"))


def test_legacy_default_does_not_load_reference():
    assert R._healthbench_direct_reference(FIXTURES._config(), ROOT, []) is None


def test_direct_profile_override_is_task_local_and_in_generation_identity(tmp_path, monkeypatch):
    config, tasks, _, _ = _case(tmp_path, monkeypatch)
    registry, _, identity = FIXTURES._identity(config, tasks[0])
    assert identity["artifact_communication_profile"] == "producer_context_structured_evidence_v2"
    shared = SimpleNamespace(artifact_communication_profile="producer_context_structured_evidence_v3")
    owned = SimpleNamespace(artifact_communication_profile="producer_context_structured_evidence_v3")
    async def stop_before_generation(*args, **kwargs):
        assert owned.artifact_communication_profile == "producer_context_structured_evidence_v2"
        assert shared.artifact_communication_profile == "producer_context_structured_evidence_v3"
        raise RuntimeError("observed owned profile without generation")
    owned.execute = stop_before_generation
    backend = SimpleNamespace(config=config, registry=registry, runtime=shared,
        _runtime_for_task=lambda *args, **kwargs: (
            owned, SimpleNamespace(resource_ids=("healthbench-authoritative.search",)), lambda: None))
    bounded = config["healthbench_professional_evaluation"]
    with pytest.raises(RuntimeError, match="observed owned profile"):
        asyncio.run(R._direct_one(backend, tasks[0], 0, model_id=bounded["direct_model_id"],
                    protocol=bounded["direct_protocol"], contract=bounded["direct_contract"],
                    seed=bounded["direct_generation_seed"], run_label="offline-profile-test"))
