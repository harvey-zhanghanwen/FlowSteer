from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.interactive.config_loader import load_yaml
from src.interactive.director import DIRECTOR_SYSTEM_PROMPT, encode_director_transcript
from src.interactive.records import TaskRecord
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.versioning import VersionBundle


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_hotpotqa_tool_availability_pair.py"
SPEC = importlib.util.spec_from_file_location(
    "run_hotpotqa_tool_availability_pair", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _temporary_storage(config: dict, tmp_path: Path) -> dict:
    copied = deepcopy(config)
    names = {
        "selected_tasks_path": "selected.jsonl",
        "direct_predictions_path": "direct.jsonl",
        "trajectories_path": "trajectories.jsonl",
        "paired_results_path": "pairs.jsonl",
        "manifest_path": "manifest.json",
    }
    for field, name in names.items():
        copied["storage"][field] = str(tmp_path / name)
    return copied


def test_prepare_only_locks_v4_tasks_and_reuses_exactly_two_v3_direct_records(
    tmp_path, monkeypatch
):
    config = _temporary_storage(
        load_yaml(ROOT / "config" / "development_hotpotqa_tool_availability_pair_v1.yaml"),
        tmp_path,
    )

    def forbidden_backend(*args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("prepare-only constructed the live backend")

    monkeypatch.setattr(MODULE.LiveSmokeBackend, "from_config", forbidden_backend)
    selected, manifest = asyncio.run(MODULE.prepare(config, ROOT))

    assert [task.task_id for task in selected] == [
        "hotpotqa:5a7a06935542990198eaf050",
        "hotpotqa:5a879ab05542996e4f30887e",
    ]
    assert [task.metadata["native_candidate_position"] for task in selected] == [0, 1]
    assert all(task.metadata["joint_qa_partition"] == "development" for task in selected)
    assert manifest["prepare_only_constructed_backend"] is False
    assert manifest["model_api_calls"] == 0
    assert manifest["direct_reuse"]["reused_records"] == 2
    assert manifest["direct_reuse"]["newly_collected_records"] == 0
    copied_direct = MODULE._read_jsonl(tmp_path / "direct.jsonl")
    assert len(copied_direct) == 2
    assert all(value["reuse_receipt"]["reused"] is True for value in copied_direct)


def test_direct_reuse_fails_closed_when_embedded_question_changes(tmp_path):
    config = _temporary_storage(
        load_yaml(ROOT / "config" / "development_hotpotqa_tool_availability_pair_v1.yaml"),
        tmp_path,
    )
    source = MODULE._resolve(
        ROOT, config["hotpotqa_evaluation"]["direct_reused_from"]
    )
    values = MODULE._read_jsonl(source)
    values[0]["task"]["question"] += " changed"
    altered = tmp_path / "altered_direct.jsonl"
    altered.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    config["hotpotqa_evaluation"]["direct_reused_from"] = str(altered)
    selected = tuple(
        MODULE._select_tasks(config, ROOT, tmp_path / "selected_for_failure.jsonl")
    )

    with pytest.raises(MODULE.ToolAvailabilityPairError, match="semantics differ"):
        asyncio.run(
            MODULE._validated_direct_reuse(
                config, selected, ROOT, tmp_path / "must_not_be_written.jsonl"
            )
        )


def _fake_record(
    task: TaskRecord,
    versions: VersionBundle,
    *,
    arm: str,
    condition_id: str,
    director_sampling: dict,
    tool_receipts: list[dict],
):
    evaluation = SimpleNamespace(
        valid=True,
        reward=1.0,
        evaluator_version=versions.evaluator,
        metrics={"exact_match": 1.0, "token_f1": 1.0},
        to_dict=lambda: {
            "evaluator_version": versions.evaluator,
            "valid": True,
            "reward": 1.0,
            "metrics": {"exact_match": 1.0, "token_f1": 1.0},
            "reason": "evaluated",
            "details": {},
        },
    )
    observation = {
        "canvas": {"nodes": [], "relations": [], "output_agent_id": None},
        "task": task.question,
        "model_catalog": [{"model_id": "qwen3.5-9b-local"}],
    }
    if arm in {"on", "tool_on"}:
        observation["tool_catalog"] = [
            {"tool_id": "qa-retrieval.search"},
            {"tool_id": "qa-retrieval.read"},
        ]
    prompt = encode_director_transcript(
        (
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Canvas observation.\n\n"
                + json.dumps(observation, sort_keys=True, separators=(",", ":")),
            },
        )
    )
    return SimpleNamespace(
        trajectory_id=f"trajectory:{arm}",
        task=task,
        condition_id=condition_id,
        versions=versions,
        forced_probe=True,
        grpo_eligible=False,
        condition_satisfied=True,
        natural_policy_terminal=True,
        api_fallback_used=False,
        manual_repair_used=False,
        active_skill_ids=(),
        retrieved_skill_ids=(),
        invoked_skill_ids=(),
        sampling_receipt_verified=True,
        director_sampling=director_sampling,
        turns=(
            SimpleNamespace(
                prompt=prompt,
                executions=(
                    SimpleNamespace(
                        metadata={"response": {"tool_receipts": tool_receipts}}
                    ),
                ),
            ),
        ),
        evaluation=evaluation,
        final_answer="<answer>answer</answer>",
    )


def test_pair_persists_assignment_separately_from_actual_tool_invocation():
    task = TaskRecord(
        task_id="hotpotqa:pair",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    versions = VersionBundle(
        policy="policy-v1",
        model_catalog="catalog-v1",
        evaluator="hotpotqa.official.answer.v1",
        prompt="prompt-v1",
        tool="tool-v1",
        encoder="none",
        feature_schema="qa-tool-availability-itt.v1",
        posterior="none",
        skill_library="none",
    )
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
        schedule_purpose="paired-schedule",
        ordered_sequence_hash=stable_hash([task.task_id]),
        sequence_position=0,
        task_id=task.task_id,
        optimizer_step_or_anchor_ordinal=3,
    )
    sampling = {
        "algorithm": "sha256-coordinate-v1",
        "base_seed": 17,
        "coordinate": coordinate.to_value(),
        "phase": "action",
    }
    condition_ids = {"tool_off": "pair:off", "tool_on": "pair:on"}
    receipt = {
        "tool_id": "skillflow.qa.search",
        "tool_version": "v1",
        "request": {"action": "search", "arguments": {"query": "question"}},
        "result": {"status": "ok"},
        "started_at_monotonic": 1.0,
        "ended_at_monotonic": 2.0,
        "latency_ms": 1000.0,
        "error_type": None,
    }
    observed = {
        "tool_off": _fake_record(
            task,
            versions,
            arm="off",
            condition_id=condition_ids["tool_off"],
            director_sampling=sampling,
            tool_receipts=[],
        ),
        "tool_on": _fake_record(
            task,
            versions,
            arm="on",
            condition_id=condition_ids["tool_on"],
            director_sampling=sampling,
            tool_receipts=[receipt],
        ),
    }

    row = MODULE._admit_pair(
        task,
        observed,
        condition_ids=condition_ids,
        schedule_purpose="paired-schedule",
        anchor=3,
        branch_order=("tool_on", "tool_off"),
        versions=versions,
    )

    assert row["estimand"] == MODULE.ESTIMAND
    assert row["forced_probe"] is True
    assert row["grpo_eligible"] is False
    assert row["arms"]["tool_off"]["tool_available"] is False
    assert row["arms"]["tool_off"]["invoked_tool_ids"] == []
    assert row["arms"]["tool_on"]["tool_available"] is True
    assert row["arms"]["tool_on"]["tool_invoked"] is True
    assert row["arms"]["tool_on"]["invoked_tool_ids"] == [
        "skillflow.qa.search"
    ]
    assert row["arms"]["tool_on"]["tool_receipts"] == [receipt]
    assert row["tool_availability_is_not_invocation"] is True
    assert row["tool_availability_is_not_useful_skill"] is True
    assert row["treatment_exposure_receipt"] == {
        "tool_off_catalog_tool_ids": [],
        "tool_on_catalog_tool_ids": [
            "qa-retrieval.search",
            "qa-retrieval.read",
        ],
        "non_treatment_observation_projection_equal": True,
    }


def test_live_entry_requires_behavior_adapter_preflight_before_collection(
    tmp_path, monkeypatch
):
    config = _temporary_storage(
        load_yaml(ROOT / "config" / "development_hotpotqa_tool_availability_pair_v1.yaml"),
        tmp_path,
    )
    expected_checkpoint = str(
        ROOT / config["director"]["behavior_adapter_checkpoint"]
    )

    class ExpectedPreflight(RuntimeError):
        pass

    class Publisher:
        def ensure_loaded_adapter(self, *, checkpoint_path, adapter_name):
            assert checkpoint_path == expected_checkpoint
            assert adapter_name == config["director"]["behavior_adapter_name"]
            raise ExpectedPreflight("preflight reached")

    async def prepared(_config, _root):
        return (), {"versions": {"model_catalog": "catalog-v1"}}

    backend = SimpleNamespace(publisher=Publisher())
    monkeypatch.setattr(MODULE, "prepare", prepared)
    monkeypatch.setattr(
        MODULE.LiveSmokeBackend, "from_config", lambda *args, **kwargs: backend
    )

    with pytest.raises(ExpectedPreflight, match="preflight reached"):
        asyncio.run(MODULE.run_live(config, ROOT))


def test_pair_fails_closed_when_sampling_schedule_differs():
    task = TaskRecord(
        "hotpotqa:pair", "question", "answer", "validation", {"dataset_key": "hotpotqa"}
    )
    versions = VersionBundle(
        "policy-v1",
        "catalog-v1",
        "hotpotqa.official.answer.v1",
        "prompt-v1",
        "tool-v1",
        "none",
        "qa-tool-availability-itt.v1",
        "none",
        "none",
    )
    condition_ids = {"tool_off": "pair:off", "tool_on": "pair:on"}

    def sampling(purpose: str) -> dict:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose=purpose,
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=3,
        )
        return {
            "algorithm": "sha256-coordinate-v1",
            "base_seed": 17,
            "coordinate": coordinate.to_value(),
            "phase": "action",
        }

    observed = {
        arm: _fake_record(
            task,
            versions,
            arm=arm,
            condition_id=condition_ids[arm],
            director_sampling=sampling("paired-schedule" if arm == "tool_off" else "other"),
            tool_receipts=[],
        )
        for arm in ("tool_off", "tool_on")
    }
    with pytest.raises(MODULE.ToolAvailabilityPairError, match="sampling coordinate"):
        MODULE._admit_pair(
            task,
            observed,
            condition_ids=condition_ids,
            schedule_purpose="paired-schedule",
            anchor=3,
            branch_order=("tool_off", "tool_on"),
            versions=versions,
        )
