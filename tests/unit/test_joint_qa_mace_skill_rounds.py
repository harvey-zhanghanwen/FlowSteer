from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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

    answer_instruction = spec.candidate_actions["answer_type_and_span_consistency"][
        "instruction"
    ]
    assert "expected answer type" in answer_instruction
    assert "canonical, minimal complete extractive span" in answer_instruction
    assert "unconditional shortest-span" in answer_instruction
    topology_instruction = spec.candidate_actions["relation_grounded_evidence_fan_in"][
        "instruction"
    ]
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
    semantic = spec.candidate_actions["subject_relation_answer_type_grounding"][
        "instruction"
    ]
    for phrase in (
        "semantic verification only when",
        "subject, relation, answer, and qualifiers",
        "answer-type mismatch",
        "supporting span",
        "one <answer> tag",
    ):
        assert phrase in semantic
    topology = spec.candidate_actions["conditional_independence_evidence_fan_in"][
        "instruction"
    ]
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
    component = spec.candidate_actions["evidence_grounded_component_transaction"][
        "instruction"
    ]
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
        "joint_qa_v2/development:[32:52]_per_dataset (zero-based, stop-exclusive)"
    )
    assert (
        "reports/joint_qa_progressive/skill_epoch_000002/publication_results.json"
        in manifest["candidate_source_artifacts"]
    )
    assert (
        "task IDs are read only for overlap exclusion" in manifest["final_test_block"]
    )
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
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation_round4.jsonl"
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
    fan_in = spec.candidate_actions["conditional_fan_in_deferred_format"]["instruction"]
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
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation_round5.jsonl"
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

    fan_in = spec.candidate_actions["conditional_fan_in_deferred_format"]["instruction"]
    assert "without output_agent_id" in fan_in
    assert "After that component executes" in fan_in
    assert "empty Canvas reports" not in fan_in
    assert (
        spec.candidate_actions["exact_answer_handoff"]
        == (runner.ROUND4_CANDIDATE_ACTIONS["exact_answer_handoff"])
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


def test_epoch6_quarantines_aborted_round_and_uses_nullable_output_tool_version() -> (
    None
):
    spec = runner._spec_for_round(6)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation_round6.jsonl"
    prior_ids: set[str] = set()
    for prior_round in range(6):
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
            task.task_id for task in train[56:59]
        ]
        assert natural[dataset].task_id == train[59].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in validation[:40]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        assert all(
            task.metadata["joint_qa_partition"] == "skill_confirmation_round6"
            for task in confirmation[dataset]
        )
    assert not selected_ids & prior_ids

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation_round6:[0:40]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert manifest["additional_confirmation_manifest"] == (
        "data/joint_qa_v2/skill_confirmation_round6_manifest.json"
    )
    assert manifest["prompt_version"] == (
        "agentgraph.director.progressive_subgraph.intermediate-partial.v2"
    )
    assert manifest["tool_version"] == (
        "agentgraph.add-subgraph-nullable-output+skillflow-public-retrieval.v2"
    )
    assert manifest["selection_coordinates"]["skill_confirmation"] == {
        "partition": "skill_confirmation_round6",
        "start": 0,
        "stop": 40,
        "zero_based_stop_exclusive": True,
    }
    assert manifest["epochs"] == {
        "discovery": 15,
        "validation": 16,
        "eligible_activation": 17,
    }


def test_epoch7_uses_candidate_specific_graph_stages_and_fresh_confirmation() -> None:
    spec = runner._spec_for_round(7)
    discovery, confirmation, natural = runner._selected_tasks(spec)
    train_path = runner.ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = runner.ROOT / "data/joint_qa_v2/skill_confirmation_round7.jsonl"
    prior_ids: set[str] = set()
    for prior_round in range(7):
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
            task.task_id for task in train[60:63]
        ]
        assert natural[dataset].task_id == train[63].task_id
        assert [task.task_id for task in confirmation[dataset]] == [
            task.task_id for task in validation[:40]
        ]
        selected_ids.update(task.task_id for task in discovery[dataset])
        selected_ids.add(natural[dataset].task_id)
        selected_ids.update(task.task_id for task in confirmation[dataset])
        assert all(
            task.metadata["joint_qa_partition"] == "skill_confirmation_round7"
            for task in confirmation[dataset]
        )
    assert not selected_ids & prior_ids

    assert (
        spec.candidate_actions["subject_relation_answer_type_grounding"]
        == (runner.ROUND2_CANDIDATE_ACTIONS["subject_relation_answer_type_grounding"])
    )
    assert (
        spec.candidate_actions["exact_answer_handoff"]
        == (runner.ROUND4_CANDIDATE_ACTIONS["exact_answer_handoff"])
    )
    assert spec.candidate_graph_stages == {
        "subject_relation_answer_type_grounding": "construction",
        "exact_answer_handoff": "before_final_answer",
    }

    manifest = runner._manifest(discovery, confirmation, natural, spec)
    assert manifest["confirmation_block"] == (
        "joint_qa_v2/skill_confirmation_round7:[0:40]_per_dataset "
        "(zero-based, stop-exclusive)"
    )
    assert manifest["prompt_version"] == (
        "agentgraph.director.progressive-subgraph.stage-conditioned-skill.v3"
    )
    assert manifest["tool_version"] == (
        "agentgraph.add-subgraph-nullable-output+"
        "skillflow-stage-conditioned-forced-probe.v3"
    )
    assert manifest["candidate_conditions"]["hotpotqa"] == {
        "subject_relation_answer_type_grounding": {
            "task_family": "hotpotqa",
            "graph_stage": "construction",
            "tags": [],
        },
        "exact_answer_handoff": {
            "task_family": "hotpotqa",
            "graph_stage": "before_final_answer",
            "tags": [],
        },
    }
    assert manifest["causal_estimand"] == (
        "Stage-conditioned Skill prompt-prior assignment intent-to-treat effect"
    )
    assert manifest["epochs"] == {
        "discovery": 18,
        "validation": 19,
        "eligible_activation": 20,
    }
    preregistered = runner.load_yaml(
        runner.ROOT / "config/joint_qa_round7_evidence.yaml"
    )
    assert preregistered["experiment_version"] == spec.experiment_version
    assert preregistered["seed"] == spec.seed
    assert preregistered["posterior_version"] == spec.posterior_version
    assert preregistered["skill_library_version"] == spec.skill_library_version
    assert preregistered["prompt_version"] == spec.prompt_version
    assert preregistered["tool_version"] == spec.tool_version
    assert preregistered["candidate_actions"] == {
        key: dict(value) for key, value in spec.candidate_actions.items()
    }
    for candidate_id, stage in spec.candidate_graph_stages.items():
        for dataset in runner.DATASETS:
            assert preregistered["candidate_conditions"][
                "by_candidate_and_dataset"
            ][candidate_id][dataset] == {
                "task_family": dataset,
                "graph_stage": stage,
                "tags": [],
            }


def test_epoch7_backend_binds_the_preregistered_prompt_and_tool_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    sentinel = object()

    def capture(config, root, *, evaluation_only):
        captured["config"] = config
        captured["root"] = root
        captured["evaluation_only"] = evaluation_only
        return sentinel

    monkeypatch.setattr(
        runner.LiveSmokeBackend,
        "from_config",
        capture,
    )
    result = runner._backend(runner.EPOCH7_SPEC)

    assert result is sentinel
    assert captured["root"] == runner.ROOT
    assert captured["evaluation_only"] is True
    assert captured["config"]["experiment"]["prompt_version"] == (
        runner.EPOCH7_SPEC.prompt_version
    )
    assert captured["config"]["experiment"]["tool_version"] == (
        runner.EPOCH7_SPEC.tool_version
    )


def test_epoch8_registers_empty_graph_hotpot_prior_without_enabling_execution() -> None:
    spec = runner._spec_for_round(8)
    assert spec.datasets == ("hotpotqa",)
    assert spec.execution_status == "prepared_not_executed"
    assert "[1024, 1064)" in str(spec.execution_blocker)
    assert spec.prompt_version == "agentgraph.director.minimal-neutral.v9"
    assert spec.tool_version == "skillflow.qa-retrieval.provided-context.v6"
    assert spec.backend_config_path == (
        runner.ROOT / "config/development_hotpotqa_dynamic_stable_zero_v9_2.yaml"
    )
    assert tuple(spec.candidate_actions) == (
        "evidence_aligned_answer_span_preservation",
        "dependency_conditioned_agentgraph",
        "receipt_backed_executor_capacity",
    )
    assert spec.candidate_graph_stages == {
        "evidence_aligned_answer_span_preservation": "empty_graph",
        "dependency_conditioned_agentgraph": "empty_graph",
        "receipt_backed_executor_capacity": "empty_graph",
    }

    answer_span = spec.candidate_actions[
        "evidence_aligned_answer_span_preservation"
    ]["instruction"]
    for phrase in (
        "proper-name modifiers, plurality, date, unit, and qualifier",
        "alias normalization as candidate disagreement",
        "singleton Format Agent, which only copies it",
        "rejectable prior",
    ):
        assert phrase in answer_span
    topology = spec.candidate_actions["dependency_conditioned_agentgraph"][
        "instruction"
    ]
    for phrase in (
        "model-visible question",
        "directed path for sequential dependencies",
        "fan-out followed by semantic fan-in for independent dependencies",
        "bounded reciprocal exchange",
        "explicit comparison or an independent candidate conflict",
        "rejectable prior",
    ):
        assert phrase in topology
    capacity = spec.candidate_actions["receipt_backed_executor_capacity"][
        "instruction"
    ]
    for phrase in (
        "saved bounded development receipt only as a weak prior",
        "Qwen3.5-Plus obtained 6/6 strict exact matches",
        "Qwen3.5-Flash obtained 5/6",
        "DeepSeek-V4-Pro obtained 5/6 with one operational failure",
        "six preselected development examples do not establish generalization",
        "Director remains the local Qwen3.5-9B",
        "fixed global role or model recipe",
        "rejectable prior",
    ):
        assert phrase in capacity
    combined = answer_span + topology + capacity
    for prohibited in (
        "native question type",
        "reference answer",
        "supporting-fact label",
        "task ID",
    ):
        assert prohibited not in combined

    preregistered = runner.load_yaml(
        runner.ROOT / "config/joint_qa_round8_evidence.yaml"
    )
    assert preregistered["status"] == spec.execution_status
    assert preregistered["datasets"] == ["hotpotqa"]
    assert preregistered["experiment_version"] == spec.experiment_version
    assert preregistered["posterior_version"] == spec.posterior_version
    assert preregistered["skill_library_version"] == spec.skill_library_version
    assert preregistered["prompt_version"] == spec.prompt_version
    assert preregistered["tool_version"] == spec.tool_version
    assert preregistered["candidate_actions"] == {
        key: dict(value) for key, value in spec.candidate_actions.items()
    }
    assert preregistered["selection_coordinates"]["skill_confirmation"] == {
        "expected_path": "data/joint_qa_v2/skill_confirmation_round8.jsonl",
        "expected_manifest": (
            "data/joint_qa_v2/skill_confirmation_round8_manifest.json"
        ),
        "partition": "skill_confirmation_round8",
        "canonical_candidate_range": {
            "start": 1024,
            "stop": 1064,
            "zero_based_stop_exclusive": True,
        },
        "per_dataset_range": {
            "start": 0,
            "stop": 40,
            "zero_based_stop_exclusive": True,
        },
        "materialized": False,
        "overlap_check_required_before_execution": True,
        "read_only_precheck": {
            "candidate_count": 40,
            "unique_task_id_count": 40,
            "frozen_test_overlap_count": 0,
            "final_test_answers_or_metrics_used": False,
        },
    }
    assert preregistered["director_observation_contract"]["topology_is_forced"] is False
    assert preregistered["data_exclusion_contract"]["final_test_untouched"] is True
    assert preregistered["backend_config"] == (
        "config/development_hotpotqa_dynamic_stable_zero_v9_2.yaml"
    )
    assert preregistered["runner_compatibility"][
        "current_runner_supports_three_candidates"
    ] is True


def test_epoch8_prepare_only_never_selects_tasks_or_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("blocked Round 8 must not select tasks or call a backend")

    monkeypatch.setattr(runner, "_selected_tasks", forbidden)
    monkeypatch.setattr(runner, "_backend", forbidden)
    result = runner.asyncio.run(runner.run(prepare_only=True, round_id=8))
    assert result == {
        "status": "prepared_not_executed",
        "round_id": 8,
        "experiment_version": "hotpotqa.mace-skill-evidence.epoch8.v2",
        "datasets": ["hotpotqa"],
        "config": str(runner.ROOT / "config/joint_qa_round8_evidence.yaml"),
        "blocker": runner.EPOCH8_SPEC.execution_blocker,
        "model_or_api_calls": 0,
    }
    with pytest.raises(RuntimeError, match="prepared_not_executed"):
        runner.asyncio.run(runner.run(prepare_only=False, round_id=8))


def test_epoch8_backend_reuses_v9_json_schema_and_provided_context_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    sentinel = object()

    def capture(config, root, *, evaluation_only):
        captured["config"] = config
        captured["root"] = root
        captured["evaluation_only"] = evaluation_only
        return sentinel

    monkeypatch.setattr(runner.LiveSmokeBackend, "from_config", capture)
    result = runner._backend(runner.EPOCH8_SPEC)
    assert result is sentinel
    assert captured["root"] == runner.ROOT
    assert captured["evaluation_only"] is True
    assert captured["config"]["director"]["action_decoding"] == "json_schema"
    assert captured["config"]["director"]["action_schema_version"] == (
        "agentgraph.canvas-action-json-schema.v1"
    )
    assert captured["config"]["director"]["sampling_schema_version"] == (
        "agentgraph.state-conditioned-action-mask.v1"
    )
    assert captured["config"]["director"]["sampling_action_profile"] == (
        "progressive_add_subgraph_then_finish"
    )
    assert captured["config"]["experiment"]["prompt_version"] == (
        runner.EPOCH8_SPEC.prompt_version
    )
    assert captured["config"]["experiment"]["tool_version"] == (
        runner.EPOCH8_SPEC.tool_version
    )
    assert captured["config"]["qa_tool_runtime"]["passage_source"] == (
        "provided_context"
    )
    assert captured["config"]["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_hotpotqa_dynamic_v9.yaml"
    )
    assert captured["config"]["hotpotqa_evaluation"]["sample_count"] == 128
    assert captured["config"]["hotpotqa_evaluation"][
        "stable_zero_sample_count"
    ] == 2
    assert set(captured["config"]["data"]) == {
        "validation_path",
        "enforce_split_isolation",
        "task_schema_version",
    }
    assert captured["config"]["experiment"]["training_enabled"] is False
    assert captured["config"]["gpu"]["training_enabled"] is False
    assert captured["config"]["grpo"]["enabled"] is False
    assert captured["config"]["skills"]["enabled"] is False
    raw_backend_config = runner.load_yaml(runner.EPOCH8_SPEC.backend_config_path)
    assert "v9_2" in raw_backend_config["experiment"]["name"]
    assert "v9_2" in raw_backend_config["experiment"]["condition_id"]
    assert "v9_2" in raw_backend_config["experiment"]["output_dir"]
    assert captured["config"]["experiment"]["condition_id"] == (
        "joint_qa_progressive_skill_epoch8"
    )
    for path in raw_backend_config["storage"].values():
        if isinstance(path, str) and path.endswith((".json", ".jsonl", ".md")):
            assert "v9_2" in path


def test_later_round_overlap_guard_reads_only_opaque_manifest_test_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.iter_task_records
    opened_paths: list[Path] = []

    def guarded(path, *args, **kwargs):
        resolved = Path(path)
        opened_paths.append(resolved)
        if resolved.name == "test.jsonl":
            raise AssertionError("sealed test payload must not be opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner, "iter_task_records", guarded)
    runner._selected_tasks(runner.EPOCH1_SPEC)
    assert all(path.name != "test.jsonl" for path in opened_paths)

    manifest = json.loads(
        (runner.ROOT / "data/joint_qa_v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected = set(
        manifest["datasets"]["hotpotqa"]["ordered_task_ids"]["test"]
    )
    assert runner._frozen_test_task_ids(runner.EPOCH8_SPEC) == expected


def test_epoch8_manifest_freezes_director_action_identity() -> None:
    task = next(
        iter_task_records(
            runner.ROOT / "data/joint_qa_v2/train.jsonl",
            expected_split="train",
        )
    )
    manifest = runner._manifest(
        {"hotpotqa": (task,)},
        {"hotpotqa": (task,)},
        {"hotpotqa": task},
        runner.EPOCH8_SPEC,
    )
    assert manifest["action_schema_version"] == (
        "agentgraph.canvas-action-json-schema.v1"
    )
    assert manifest["sampling_schema_version"] == (
        "agentgraph.state-conditioned-action-mask.v1"
    )
    assert manifest["sampling_action_profile"] == (
        "progressive_add_subgraph_then_finish"
    )
    assert manifest["per_turn_action_receipt_required"] is True
    assert "test.jsonl" in manifest["final_test_block"]
    assert "excluded" in manifest["final_test_block"]


def test_epoch8_per_turn_action_receipt_fails_closed_on_regime_drift() -> None:
    good_turn = SimpleNamespace(
        round_index=0,
        runtime_summary={
            "director_action_schema_version": (
                "agentgraph.state-conditioned-action-mask.v1"
            ),
            "director_action_schema_branch": "add_subgraph",
        },
        action={"action": "add_subgraph"},
    )
    record = SimpleNamespace(trajectory_id="trajectory-good", turns=(good_turn,))
    receipts = runner._director_action_receipts(record, runner.EPOCH8_SPEC)
    assert receipts == (
        {
            "round_index": 0,
            "sampling_action_profile": "progressive_add_subgraph_then_finish",
            "sampling_schema_version": (
                "agentgraph.state-conditioned-action-mask.v1"
            ),
            "action_schema_version": "agentgraph.canvas-action-json-schema.v1",
            "action_schema_branch": "add_subgraph",
            "parsed_action": "add_subgraph",
        },
    )

    drifted = SimpleNamespace(
        trajectory_id="trajectory-drifted",
        turns=(
            SimpleNamespace(
                round_index=0,
                runtime_summary={
                    "director_action_schema_version": "foreign-schema",
                    "director_action_schema_branch": "add_subgraph",
                },
                action={"action": "add_subgraph"},
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="sampling schema differs"):
        runner._director_action_receipts(drifted, runner.EPOCH8_SPEC)


def test_epoch8_paired_exposure_requires_treatment_round0_and_clean_incumbent() -> None:
    incumbent = SimpleNamespace(name="incumbent")
    candidate = SimpleNamespace(name="candidate")

    class Backend:
        receipts = {
            "incumbent": {
                "current_observation_rounds": (),
                "transcript_retained_rounds": (),
            },
            "candidate": {
                "current_observation_rounds": (0,),
                "transcript_retained_rounds": (0, 1),
            },
        }

        @classmethod
        def _prompt_prior_exposure_receipt(cls, trajectory, condition_id):
            assert condition_id == "dependency_conditioned_agentgraph"
            return cls.receipts[trajectory.name]

    receipt = runner._paired_exposure_receipts(
        Backend(),
        incumbent,
        candidate,
        "dependency_conditioned_agentgraph",
        runner.EPOCH8_SPEC,
    )
    assert receipt["verified"] is True
    assert receipt["preregistered_graph_stage"] == "empty_graph"
    assert receipt["candidate"]["current_observation_rounds"] == [0]
    assert receipt["incumbent"]["transcript_retained_rounds"] == []

    Backend.receipts["candidate"] = {
        "current_observation_rounds": (1,),
        "transcript_retained_rounds": (1,),
    }
    with pytest.raises(RuntimeError, match="not exposed at round 0"):
        runner._paired_exposure_receipts(
            Backend(),
            incumbent,
            candidate,
            "dependency_conditioned_agentgraph",
            runner.EPOCH8_SPEC,
        )


def test_epoch8_publication_preflight_rejects_missing_or_control_exposure() -> None:
    action_receipt = {
        "round_index": 0,
        "sampling_action_profile": "progressive_add_subgraph_then_finish",
        "sampling_schema_version": "agentgraph.state-conditioned-action-mask.v1",
        "action_schema_version": "agentgraph.canvas-action-json-schema.v1",
        "action_schema_branch": "add_subgraph",
        "parsed_action": "add_subgraph",
    }
    row = {
        "candidate_id": "dependency_conditioned_agentgraph",
        "prompt_prior_exposure_receipt": {
            "preregistered_graph_stage": "empty_graph",
            "incumbent": {
                "current_observation_rounds": [],
                "transcript_retained_rounds": [],
            },
            "candidate": {
                "current_observation_rounds": [0],
                "transcript_retained_rounds": [0, 1],
            },
            "verified": True,
        },
        "director_action_identity": {
            "sampling_action_profile": "progressive_add_subgraph_then_finish",
            "sampling_schema_version": (
                "agentgraph.state-conditioned-action-mask.v1"
            ),
            "action_schema_version": "agentgraph.canvas-action-json-schema.v1",
        },
        "incumbent_director_action_receipts": [action_receipt],
        "candidate_director_action_receipts": [action_receipt],
    }
    runner._validate_paired_exposure_rows([row], runner.EPOCH8_SPEC)

    missing = dict(row)
    missing.pop("prompt_prior_exposure_receipt")
    with pytest.raises(RuntimeError, match="lacks a verified exposure"):
        runner._validate_paired_exposure_rows([missing], runner.EPOCH8_SPEC)

    contaminated = json.loads(json.dumps(row))
    contaminated["prompt_prior_exposure_receipt"]["incumbent"][
        "transcript_retained_rounds"
    ] = [0]
    with pytest.raises(RuntimeError, match="incumbent has candidate Skill exposure"):
        runner._validate_paired_exposure_rows(
            [contaminated], runner.EPOCH8_SPEC
        )


def test_unknown_round_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Skill evidence round"):
        runner._spec_for_round(9)
