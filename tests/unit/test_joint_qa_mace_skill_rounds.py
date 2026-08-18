from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from scripts import run_joint_qa_mace_skill as runner
from src.interactive.task_dataset import iter_task_records


def _dataset_records(path: Path, split: str, dataset: str):
    return tuple(
        task
        for task in iter_task_records(path, expected_split=split)
        if task.metadata.get("dataset_key") == dataset
    )


def test_epoch0_default_paths_tasks_and_versions_remain_compatible() -> None:
    spec = runner._spec_for_round(0)
    assert inspect.signature(runner.run).parameters["round_id"].default == 0
    assert spec.output_root == runner.OUTPUT_ROOT
    assert spec.report_root == runner.REPORT_ROOT
    assert spec.evidence_root == runner.EVIDENCE_ROOT
    assert spec.skill_store_path == runner.SKILL_STORE_PATH
    assert spec.pair_path == runner.PAIR_PATH
    assert spec.selection_path == runner.SELECTION_PATH
    assert spec.evsi_path == runner.EVSI_PATH
    assert spec.publication_path == runner.PUBLICATION_PATH
    assert spec.manifest_path == runner.MANIFEST_PATH
    assert spec.candidate_actions is runner.CANDIDATE_ACTIONS
    assert spec.seed == runner.SEED
    assert spec.posterior_version == runner.POSTERIOR_VERSION
    assert spec.skill_library_version == runner.SKILL_LIBRARY_VERSION

    discovery, confirmation, natural = runner._selected_tasks()
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation.jsonl"
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        heldout = _dataset_records(confirmation_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[:3]
        ]
        assert natural[dataset].task_id == train[3].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in heldout[:20]
        ]

    manifest = runner._manifest(discovery, confirmation, natural)
    assert manifest["seed"] == 20260818
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation:first20_per_dataset"
    )
    assert manifest["epochs"] == {
        "discovery": 0,
        "validation": 1,
        "eligible_activation": 2,
    }
    assert "round_id" not in manifest
    assert "experiment_version" not in manifest


def test_epoch1_uses_disjoint_task_blocks_and_reserved_training_is_untouched() -> None:
    spec = runner._spec_for_round(1)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    epoch0_discovery, epoch0_confirmation, epoch0_natural = runner._selected_tasks()
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation.jsonl"
    dev_ids = {
        task.task_id
        for task in iter_task_records(
            runner.ROOT / "data/joint_qa_v2/development.jsonl",
            expected_split="validation",
        )
    }
    test_ids = {
        task.task_id
        for task in iter_task_records(
            runner.ROOT / "data/joint_qa_v2/test.jsonl",
            expected_split="test",
        )
    }

    epoch0_ids = {
        task.task_id
        for dataset in runner.DATASETS
        for task in (
            *epoch0_discovery[dataset],
            epoch0_natural[dataset],
            *epoch0_confirmation[dataset],
        )
    }
    epoch1_ids: set[str] = set()
    reserved_training_ids: set[str] = set()
    reserved_positions = runner._reserved_training_positions()
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        heldout = _dataset_records(confirmation_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[4:7]
        ]
        assert natural[dataset].task_id == train[7].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in heldout[20:40]
        ]
        epoch1_ids.update(task.task_id for task in discovery[dataset])
        epoch1_ids.add(natural[dataset].task_id)
        epoch1_ids.update(task.task_id for task in confirmation[dataset])
        reserved_training_ids.add(train[reserved_positions[dataset]].task_id)

    assert not epoch1_ids & epoch0_ids
    assert not epoch1_ids & dev_ids
    assert not epoch1_ids & test_ids
    assert not epoch1_ids & reserved_training_ids


def test_epoch1_spec_candidates_manifest_and_publication_scope() -> None:
    spec = runner._spec_for_round(1)
    assert spec.output_root.name == "skill_epoch_000001"
    assert spec.report_root.name == "skill_epoch_000001"
    assert spec.seed != runner.SEED
    assert spec.posterior_version != runner.POSTERIOR_VERSION
    assert spec.skill_library_version != runner.SKILL_LIBRARY_VERSION
    assert (
        spec.discovery_epoch,
        spec.validation_epoch,
        spec.activation_epoch,
    ) == (2, 3, 4)
    assert tuple(spec.candidate_actions) == (
        "answer_type_and_span_consistency",
        "relation_grounded_evidence_fan_in",
    )

    answer_instruction = spec.candidate_actions[
        "answer_type_and_span_consistency"
    ]["instruction"]
    assert "expected answer type" in answer_instruction
    assert "canonical, minimal complete extractive span" in answer_instruction
    assert "unconditional shortest-span" in answer_instruction
    topology_instruction = spec.candidate_actions[
        "relation_grounded_evidence_fan_in"
    ]["instruction"]
    for phrase in (
        "subject, relation, and qualifiers",
        "independent evidence subquestions",
        "parallel evidence branches",
        "semantic combiner",
        "one ADD_SUBGRAPH transaction",
        "Format Agent only to serialize",
        "directed serial dependencies",
        "Do not prescribe a fixed Agent count",
    ):
        assert phrase in topology_instruction

    discovery, confirmation, natural = runner._selected_tasks(spec)
    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["round_id"] == 1
    assert manifest["experiment_version"] == spec.experiment_version
    assert manifest["selection_coordinates"] == {
        "train_discovery": {
            "start": 4,
            "stop": 7,
            "zero_based_stop_exclusive": True,
        },
        "train_natural_candidate_position": 7,
        "skill_confirmation": {
            "partition": "skill_confirmation",
            "start": 20,
            "stop": 40,
            "zero_based_stop_exclusive": True,
        },
        "reserved_grpo_training_positions": {
            "hotpotqa": 9,
            "triviaqa": 12,
        },
    }
    assert manifest["causal_estimand"] == (
        "Skill prompt-prior visibility intent-to-treat effect"
    )
    assert manifest["not_a_prefix_topology_intervention"] is True
    assert manifest["topology_adoption_acceptance"] == {
        "verified_by_this_protocol": False,
        "requires_independent_evaluation": True,
        "reason": (
            "The paired intervention estimates full-trajectory Skill prompt-prior "
            "visibility intent-to-treat effect from an empty Canvas; terminal answer "
            "F1 does not verify Director adoption of parallel branches or semantic "
            "fan-in."
        ),
    }


def test_round1_output_guard_allows_matching_resume_and_rejects_foreign_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    spec = runner.EPOCH1_SPEC
    manifest = {
        key: value
        for key, value in {
            "policy_version": runner.POLICY_VERSION,
            "adapter_name": runner.BEHAVIOR_ADAPTER_NAME,
            "posterior_version": spec.posterior_version,
            "skill_library_version": spec.skill_library_version,
            "prompt_version": runner.PROMPT_VERSION,
            "tool_version": runner.TOOL_VERSION,
            "seed": spec.seed,
            "candidate_actions": {
                key: dict(value) for key, value in spec.candidate_actions.items()
            },
            "discovery_tasks": {dataset: [] for dataset in runner.DATASETS},
            "natural_candidate_tasks": {dataset: "task" for dataset in runner.DATASETS},
            "confirmation_tasks": {dataset: [] for dataset in runner.DATASETS},
            "round_id": 1,
            "experiment_version": spec.experiment_version,
            "selection_coordinates": {"round": 1},
        }.items()
    }
    spec.output_root.mkdir(parents=True)
    (spec.output_root / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without manifest"):
        runner._guard_output_identity(spec, manifest)

    (spec.output_root / "foreign.txt").unlink()
    spec.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    runner._guard_output_identity(spec, manifest)

    persisted = dict(manifest)
    persisted["seed"] = -1
    spec.manifest_path.write_text(
        json.dumps(persisted, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different evidence regime: seed"):
        runner._guard_output_identity(spec, manifest)


def test_epoch2_uses_fresh_evidence_and_conditional_topology_priors() -> None:
    spec = runner._spec_for_round(2)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation.jsonl"
    reserved_positions = runner._reserved_training_positions()
    prior_ids: set[str] = set()
    for prior_round in (0, 1):
        prior_discovery, prior_confirmation, prior_natural = runner._selected_tasks(
            runner._spec_for_round(prior_round)
        )
        prior_ids.update(
            task.task_id
            for dataset in runner.DATASETS
            for task in (
                *prior_discovery[dataset],
                prior_natural[dataset],
                *prior_confirmation[dataset],
            )
        )

    selected_ids: set[str] = set()
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        heldout = _dataset_records(confirmation_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[13:16]
        ]
        assert natural[dataset].task_id == train[16].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in heldout[40:60]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        assert train[reserved_positions[dataset]].task_id not in selected_ids

    assert not selected_ids & prior_ids
    assert tuple(spec.candidate_actions) == (
        "subject_relation_answer_type_grounding",
        "conditional_independence_evidence_fan_in",
    )
    semantic = spec.candidate_actions[
        "subject_relation_answer_type_grounding"
    ]["instruction"]
    for phrase in (
        "semantic verification only when",
        "subject, relation, answer, and qualifiers",
        "answer-type mismatch",
        "supporting span",
        "one <answer> tag",
    ):
        assert phrase in semantic
    topology = spec.candidate_actions[
        "conditional_independence_evidence_fan_in"
    ]["instruction"]
    for phrase in (
        "conditionally independent",
        "one ADD_SUBGRAPH transaction",
        "first fan-in Agent",
        "directed serial dependencies",
        "Do not prescribe a total Agent count",
    ):
        assert phrase in topology

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["round_id"] == 2
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation:[40:60]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert manifest["epochs"] == {
        "discovery": 4,
        "validation": 5,
        "eligible_activation": 6,
    }


def test_epoch3_uses_unopened_development_validation_and_component_prior() -> None:
    spec = runner._spec_for_round(3)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    development_path = runner.ROOT / "data/joint_qa_v2/development.jsonl"
    test_ids = {
        task.task_id
        for task in iter_task_records(
            runner.ROOT / "data/joint_qa_v2/test.jsonl", expected_split="test"
        )
    }
    prior_ids: set[str] = set()
    for prior_round in (0, 1, 2):
        prior_discovery, prior_confirmation, prior_natural = runner._selected_tasks(
            runner._spec_for_round(prior_round)
        )
        prior_ids.update(
            task.task_id
            for dataset in runner.DATASETS
            for task in (
                *prior_discovery[dataset],
                prior_natural[dataset],
                *prior_confirmation[dataset],
            )
        )

    selected_ids: set[str] = set()
    fixed_development_ids: set[str] = set()
    reserved_positions = runner._reserved_training_positions()
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        development = _dataset_records(development_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[17:20]
        ]
        assert natural[dataset].task_id == train[20].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in development[32:52]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        fixed_development_ids.update(task.task_id for task in development[:32])
        assert train[reserved_positions[dataset]].task_id not in selected_ids

    assert not selected_ids & prior_ids
    assert not selected_ids & fixed_development_ids
    assert not selected_ids & test_ids
    assert tuple(spec.candidate_actions) == (
        "evidence_grounded_component_transaction",
        "evidence_span_answer_contract",
    )
    component = spec.candidate_actions[
        "evidence_grounded_component_transaction"
    ]["instruction"]
    for phrase in (
        "conditionally independent",
        "one ADD_SUBGRAPH transaction",
        "first semantic fan-in Agent",
        "without setting output_agent_id",
        "after Canvas feedback",
        "directed fan-in-to-Format relation",
        "set Format as output_agent_id",
        "subject, relation, answer, qualifiers",
        "verbatim evidence span",
        "evidence surface form and unit",
    ):
        assert phrase in component

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/development:[32:52]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert (
        "reports/joint_qa_progressive/skill_epoch_000002/publication_results.json"
        in manifest["candidate_source_artifacts"]
    )
    assert "task IDs are read only for overlap exclusion" in manifest["final_test_block"]
    assert manifest["selection_coordinates"]["skill_confirmation"] == {
        "partition": "development",
        "start": 32,
        "stop": 52,
        "zero_based_stop_exclusive": True,
    }
    assert manifest["epochs"] == {
        "discovery": 6,
        "validation": 7,
        "eligible_activation": 8,
    }


def test_epoch4_uses_fresh_canonical_confirmation_and_atomic_priors() -> None:
    spec = runner._spec_for_round(4)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = (
        runner.ROOT / "data/joint_qa_v2/skill_confirmation_round4.jsonl"
    )
    development_ids = {
        task.task_id
        for task in iter_task_records(
            runner.ROOT / "data/joint_qa_v2/development.jsonl",
            expected_split="validation",
        )
    }
    test_ids = {
        task.task_id
        for task in iter_task_records(
            runner.ROOT / "data/joint_qa_v2/test.jsonl", expected_split="test"
        )
    }
    prior_ids: set[str] = set()
    for prior_round in (0, 1, 2, 3):
        prior_discovery, prior_confirmation, prior_natural = runner._selected_tasks(
            runner._spec_for_round(prior_round)
        )
        prior_ids.update(
            task.task_id
            for dataset in runner.DATASETS
            for task in (
                *prior_discovery[dataset],
                prior_natural[dataset],
                *prior_confirmation[dataset],
            )
        )

    selected_ids: set[str] = set()
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        validation = _dataset_records(confirmation_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[48:51]
        ]
        assert natural[dataset].task_id == train[51].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in validation[:40]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        assert all(
            task.metadata["joint_qa_partition"] == "skill_confirmation_round4"
            for task in confirmation[dataset]
        )

    assert not selected_ids & prior_ids
    assert not selected_ids & development_ids
    assert not selected_ids & test_ids
    assert tuple(spec.candidate_actions) == (
        "conditional_fan_in_deferred_format",
        "exact_answer_handoff",
    )
    fan_in = spec.candidate_actions["conditional_fan_in_deferred_format"][
        "instruction"
    ]
    for phrase in (
        "Only when two evidence subquestions are conditionally independent",
        "one ADD_SUBGRAPH",
        "Omit output_agent_id",
        "After execution",
        "next ADD_SUBGRAPH",
        "connect fan-in to Format",
        "set Format as output_agent_id",
    ):
        assert phrase in fan_in
    handoff = spec.candidate_actions["exact_answer_handoff"]["instruction"]
    for phrase in (
        "exactly one supported answer candidate",
        "semantic type",
        "names, modifiers, units, and evidence language",
        "only copies that candidate",
        "must not shorten, translate, expand, or add alternatives",
    ):
        assert phrase in handoff

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation_round4:[0:40]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert manifest["additional_confirmation_manifest"] == (
        "data/joint_qa_v2/skill_confirmation_round4_manifest.json"
    )
    assert (
        "reports/joint_qa_progressive/skill_epoch_000003/publication_results.json"
        in manifest["candidate_source_artifacts"]
    )
    assert manifest["selection_coordinates"]["skill_confirmation"] == {
        "partition": "skill_confirmation_round4",
        "start": 0,
        "stop": 40,
        "zero_based_stop_exclusive": True,
    }
    assert manifest["epochs"] == {
        "discovery": 9,
        "validation": 10,
        "eligible_activation": 11,
    }


def test_epoch5_revalidates_atomic_priors_after_terminal_feedback_alignment() -> None:
    spec = runner._spec_for_round(5)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = (
        runner.ROOT / "data/joint_qa_v2/skill_confirmation_round5.jsonl"
    )
    prior_ids: set[str] = set()
    for prior_round in range(5):
        prior_discovery, prior_confirmation, prior_natural = runner._selected_tasks(
            runner._spec_for_round(prior_round)
        )
        prior_ids.update(
            task.task_id
            for dataset in runner.DATASETS
            for task in (
                *prior_discovery[dataset],
                prior_natural[dataset],
                *prior_confirmation[dataset],
            )
        )

    selected_ids: set[str] = set()
    for dataset in runner.DATASETS:
        train = _dataset_records(train_path, "train", dataset)
        validation = _dataset_records(confirmation_path, "validation", dataset)
        assert [task.task_id for task in discovery[dataset]] == [
            task.task_id for task in train[52:55]
        ]
        assert natural[dataset].task_id == train[55].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in validation[:40]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        assert all(
            task.metadata["joint_qa_partition"] == "skill_confirmation_round5"
            for task in confirmation[dataset]
        )
    assert not selected_ids & prior_ids

    fan_in = spec.candidate_actions["conditional_fan_in_deferred_format"][
        "instruction"
    ]
    assert "without output_agent_id" in fan_in
    assert "After that component executes" in fan_in
    assert "empty Canvas reports" not in fan_in
    assert spec.candidate_actions["exact_answer_handoff"] == (
        runner.ROUND4_CANDIDATE_ACTIONS["exact_answer_handoff"]
    )

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation_round5:[0:40]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert manifest["additional_confirmation_manifest"] == (
        "data/joint_qa_v2/skill_confirmation_round5_manifest.json"
    )
    assert manifest["prompt_version"] == (
        "agentgraph.director.progressive_subgraph.intermediate-partial.v2"
    )
    assert manifest["selection_coordinates"]["skill_confirmation"] == {
        "partition": "skill_confirmation_round5",
        "start": 0,
        "stop": 40,
        "zero_based_stop_exclusive": True,
    }
    assert manifest["epochs"] == {
        "discovery": 12,
        "validation": 13,
        "eligible_activation": 14,
    }


def test_unknown_round_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Skill evidence round"):
        runner._spec_for_round(6)
