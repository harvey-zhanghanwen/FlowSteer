#!/usr/bin/env python3
"""Run the bounded progressive joint-QA MACE -> posterior -> Skill protocol.

This is an experiment adapter over existing project components, not another
exploration or Skill implementation.  It reuses:

* FlowSteer's progressive Canvas execution through ``LiveSmokeBackend``;
* the existing paired-intervention records and randomized branch order;
* the existing joint Bayesian posterior and MACE-style UCB policy;
* the existing ``SkillEvidencePipeline`` and deterministic evidence gate; and
* SkillFlow's public TriviaQA retrieval observations.

Development, discovery, Skill confirmation, and final test come from the
disjoint ``joint_qa_v2`` manifest.  Skill confirmation never enters GRPO or
reported development/test EM/F1.
The paired intervention starts from an empty Canvas, freezes policy/model/
evaluator/tool versions and sampling coordinates, and regenerates downstream
execution for each arm.  Its primary outcome is official answer token F1;
normalized exact match is persisted as a companion outcome.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.evaluate_triviaqa_round import _prepare_retrieval
from scripts.train_agentgraph_smoke import (
    LiveSmokeBackend,
    _write_json,
    _write_jsonl,
    version_bundle_for,
)
from src.interactive.config_loader import load_yaml
from src.interactive.exploration import randomize_probe_order
from src.interactive.exploration.skill_experiment import (
    DATASETS,
    ENCODER_VERSION,
    FEATURE_SCHEMA_VERSION,
    POSTERIOR_VERSION,
    JointQAPosteriorScheduler,
    calibrate_skill_validation,
)
from src.interactive.persistence import EvidenceStore, stable_id
from src.interactive.qa_retrieval import augment_task_with_retrieval
from src.interactive.records import (
    ProbeRecord,
    SelectionReceipt,
    TaskRecord,
    TrajectoryRecord,
)
from src.interactive.skills import (
    SkillEvidencePipeline,
    SkillGateConfig,
    SkillProbeEvidence,
    SkillStatus,
    SkillStore,
    StructuredSkillCandidate,
)
from src.interactive.task_dataset import iter_task_records


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/joint_qa_progressive/skill_epoch_000000"
REPORT_ROOT = ROOT / "reports/joint_qa_progressive/skill_epoch_000000"
EVIDENCE_ROOT = OUTPUT_ROOT / "evidence"
SKILL_STORE_PATH = OUTPUT_ROOT / "skills.json"
PAIR_PATH = OUTPUT_ROOT / "paired_observations.jsonl"
SELECTION_PATH = OUTPUT_ROOT / "selection_receipts.jsonl"
EVSI_PATH = OUTPUT_ROOT / "evsi_receipts.jsonl"
PUBLICATION_PATH = OUTPUT_ROOT / "publication_results.json"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
RUNTIME_VERSION = "flowsteer.agentgraph.progressive-runtime.v1"
POLICY_VERSION = "qwen35-9b-hotpot-step-000000"
PROMPT_VERSION = "agentgraph.director.progressive_subgraph.v1"
TOOL_VERSION = "agentgraph.add-subgraph+skillflow-public-retrieval.v1"
SKILL_LIBRARY_VERSION = "jointqa.skill-library.progressive.epoch2.v1"
EVALUATION_CONFIG = ROOT / "config/evaluation_joint_qa_progressive_step0_hotpotqa.yaml"
TRAINING_CONFIG = ROOT / "config/training_joint_qa_progressive_skill_on_step1.yaml"
BEHAVIOR_ADAPTER_NAME = "theta_jointqa_progressive_step_000000"
BEHAVIOR_ADAPTER_CHECKPOINT = (
    ROOT / "artifacts/hotpotqa_multiagent_skill/policy_step_000000/theta"
)
SEED = 20260818

BASELINE_ACTION: Mapping[str, Any] = {
    "instruction": "No additional prompt prior; use the frozen Director policy."
}

# These bounded prompt priors encode the two failure classes supported by the
# persisted joint-QA wrong demos: fragile multi-action component construction
# and unsupported/overlong terminal answers.  They do not prescribe a fixed
# topology or model and never mutate the Canvas directly.
CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "dependency_aligned_topology": {
        "instruction": (
            "When the question contains independent evidence subproblems, represent "
            "them as parallel branches in one add_subgraph transaction and route them "
            "to the first Agent that must combine them. When a draft and critique need "
            "bounded revision, use one finite reciprocal pair; otherwise keep directed "
            "dependencies. Do not add branches when the dependencies are serial."
        )
    },
    "evidence_to_format_handoff": {
        "instruction": (
            "Before FINISH, ensure the terminal path carries the evidence and answer "
            "candidate needed by the Output Agent. When exact answer formatting is a "
            "separate responsibility, use a Format Agent only to extract one shortest "
            "supported answer span inside a single <answer> tag."
        )
    },
}


# Round 1 is a second, disjoint evidence epoch over the same upstream
# FlowSteer/SkillFlow execution and evidence pipeline.  The two prompt priors
# are predeclared from the persisted joint-QA wrong demos; they do not mutate
# the Canvas, prescribe a fixed Agent count, or add a second Skill framework.
ROUND1_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "answer_type_and_span_consistency": {
        "instruction": (
            "Preserve the question's expected answer type when handing an answer "
            "candidate to the Output Agent, and return a canonical, minimal complete "
            "extractive span supported by the evidence. Do not replace a named entity "
            "with a count, category, abbreviation, or expanded alias, and do not apply "
            "unconditional shortest-span compression."
        )
    },
    "relation_grounded_evidence_fan_in": {
        "instruction": (
            "Check the subject, relation, and qualifiers against retrieved evidence. "
            "Only when independent evidence subquestions exist, create parallel "
            "evidence branches and their semantic combiner in one ADD_SUBGRAPH "
            "transaction; use a Format Agent only to serialize the combined answer. "
            "Otherwise keep directed serial dependencies. Do not prescribe a fixed "
            "Agent count."
        )
    },
}


# Round 2 narrows the prompt priors to the two failure modes established by
# round-1 confirmation receipts: subject/relation drift and missing conditional
# fan-in adoption.  This is still the same paired-intervention caller chain;
# no Canvas mutation, topology reward, or second Skill implementation is added.
ROUND2_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "subject_relation_answer_type_grounding": {
        "instruction": (
            "Before FINISH, perform semantic verification only when evidence "
            "contains competing entities, a comparison, a qualifier, an alias, "
            "or an answer candidate whose semantic type may differ from the "
            "question. Represent the supported proposition as subject, relation, "
            "answer, and qualifiers; reject a different subject or relation, an "
            "intermediate value, or an answer-type mismatch. Preserve the evidence "
            "span's language and necessary proper-name modifiers. Route exactly one "
            "verified answer candidate and its supporting span to the Output Agent, "
            "which only serializes it in one <answer> tag."
        )
    },
    "conditional_independence_evidence_fan_in": {
        "instruction": (
            "Decompose into parallel evidence branches only when at least two "
            "evidence subquestions are conditionally independent. In one "
            "ADD_SUBGRAPH transaction, add those branches and their first fan-in "
            "Agent. Each branch emits a supported subject-relation-answer-qualifier "
            "proposition and evidence span; the fan-in Agent checks identity, "
            "relation, qualifier, and answer-type consistency before emitting one "
            "candidate. If conditional independence is absent, use directed serial "
            "dependencies. Do not prescribe a total Agent count; the Format Agent "
            "only serializes the verified fan-in output."
        )
    },
}


# Round 3 keeps the evidence-grounding behavior that produced positive,
# zero-harm effects in round 2 and makes the FlowSteer interaction boundary
# explicit: a semantic component executes before the terminal Format Agent is
# added.  The second candidate isolates the answer contract/refusal failure
# class.  Both remain rejectable prompt priors under the unchanged gate.
ROUND3_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "evidence_grounded_component_transaction": {
        "instruction": (
            "First identify the requested answer type and the evidence dependency "
            "structure. When two evidence subquestions are conditionally "
            "independent, use one ADD_SUBGRAPH transaction for two evidence "
            "branches feeding their first semantic fan-in Agent, then execute that "
            "component without setting output_agent_id. Only after Canvas feedback exposes the combined semantic "
            "answer, use the next transaction to add the Format Agent, add the "
            "directed fan-in-to-Format relation, set Format as output_agent_id, and "
            "execute it. For serial or "
            "single-hop dependencies, execute the smallest directed semantic "
            "component before adding Format. Each evidence artifact states subject, "
            "relation, answer, qualifiers, and a verbatim evidence span. The fan-in Agent "
            "rejects entity/relation drift, intermediate values, answer-type "
            "mismatch, and unsupported refusal; it preserves the evidence surface "
            "form and unit. The Format Agent only emits that verified candidate in "
            "one <answer> tag."
        )
    },
    "evidence_span_answer_contract": {
        "instruction": (
            "Before FINISH, bind one answer candidate to the question's subject, "
            "relation, qualifiers, and expected semantic type. Prefer the exact "
            "supported surface form, unit, and necessary proper-name modifiers; "
            "exclude an intermediate entity, a type noun already supplied by the "
            "question, translation, expanded alias, explanation, or multiple "
            "candidates. When retrieval is incomplete but a supported candidate can "
            "still be identified from the available evidence and model knowledge, "
            "verify that candidate instead of returning a generic refusal. Route it "
            "to a Format Agent that only emits one <answer> tag."
        )
    },
}


# Round 4 separates the two atomic prompt priors that round 3 combined.  The
# first addresses only FlowSteer's component-execution order; the second keeps
# semantic selection outside the Format Agent.  Both are short, conditional,
# and rejectable.  They use a new canonical source block beyond every existing
# joint_qa_v2 partition rather than reusing exposed development examples.
ROUND4_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "conditional_fan_in_deferred_format": {
        "instruction": (
            "Only when two evidence subquestions are conditionally independent, "
            "first use one ADD_SUBGRAPH for the two evidence Agents and their "
            "semantic fan-in Agent. Omit output_agent_id at this stage, even if the "
            "empty Canvas reports that no Format Output is selected. After "
            "execution, use the next ADD_SUBGRAPH to add one Format Agent, connect "
            "fan-in to Format, set Format as output_agent_id, and then FINISH."
        )
    },
    "exact_answer_handoff": {
        "instruction": (
            "Pass exactly one supported answer candidate to the Format Agent. The "
            "candidate must match the question's semantic type and preserve required "
            "names, modifiers, units, and evidence language. The Format Agent only "
            "copies that candidate into one <answer> tag; it must not shorten, "
            "translate, expand, or add alternatives."
        )
    },
}


# Round 5 reruns the same atomic hypotheses after aligning the Director's
# intermediate Canvas observation with FlowSteer's actual interaction boundary:
# terminal completeness and Format constraints are checked at FINISH, not on a
# valid output-free semantic component.  The obsolete workaround text from
# round 4 is removed; no topology reward or fixed role template is introduced.
ROUND5_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "conditional_fan_in_deferred_format": {
        "instruction": (
            "Only when two evidence subquestions are conditionally independent, "
            "first use one ADD_SUBGRAPH for the two evidence Agents and their "
            "semantic fan-in Agent, without output_agent_id. After that component "
            "executes, use the next ADD_SUBGRAPH to add one Format Agent, connect "
            "fan-in to Format, set Format as output_agent_id, and then FINISH."
        )
    },
    "exact_answer_handoff": deepcopy(ROUND4_CANDIDATE_ACTIONS["exact_answer_handoff"]),
}


# Round 7 reuses two previously predeclared semantic hypotheses verbatim and
# changes only their decision-time visibility boundary.  Grounding is queried
# while the Canvas is under construction; terminal answer handoff is queried
# only after a complete graph exists and immediately before FINISH.
ROUND7_CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "subject_relation_answer_type_grounding": deepcopy(
        ROUND2_CANDIDATE_ACTIONS["subject_relation_answer_type_grounding"]
    ),
    "exact_answer_handoff": deepcopy(ROUND4_CANDIDATE_ACTIONS["exact_answer_handoff"]),
}


@dataclass(frozen=True)
class SkillEvidenceRoundSpec:
    """Immutable coordinates for one bounded Skill evidence epoch."""

    round_id: int
    experiment_version: str
    candidate_actions: Mapping[str, Mapping[str, Any]]
    discovery_start: int
    discovery_stop: int
    natural_index: int
    confirmation_start: int
    confirmation_stop: int
    seed: int
    posterior_version: str
    skill_library_version: str
    discovery_epoch: int
    validation_epoch: int
    activation_epoch: int
    confirmation_source: str = "skill_confirmation"
    prompt_version: str = PROMPT_VERSION
    tool_version: str = TOOL_VERSION
    candidate_graph_stages: Mapping[str, str] = field(default_factory=dict)

    @property
    def output_root(self) -> Path:
        return ROOT / f"artifacts/joint_qa_progressive/skill_epoch_{self.round_id:06d}"

    @property
    def report_root(self) -> Path:
        return ROOT / f"reports/joint_qa_progressive/skill_epoch_{self.round_id:06d}"

    @property
    def evidence_root(self) -> Path:
        return self.output_root / "evidence"

    @property
    def skill_store_path(self) -> Path:
        return self.output_root / "skills.json"

    @property
    def pair_path(self) -> Path:
        return self.output_root / "paired_observations.jsonl"

    @property
    def selection_path(self) -> Path:
        return self.output_root / "selection_receipts.jsonl"

    @property
    def evsi_path(self) -> Path:
        return self.output_root / "evsi_receipts.jsonl"

    @property
    def publication_path(self) -> Path:
        return self.output_root / "publication_results.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "experiment_manifest.json"

    @property
    def anchor_offset(self) -> int:
        # Keep every epoch0 sampling coordinate byte-for-byte compatible.
        return self.round_id * 10_000


EPOCH0_SPEC = SkillEvidenceRoundSpec(
    round_id=0,
    experiment_version="jointqa.mace-skill-evidence.epoch0.v1",
    candidate_actions=CANDIDATE_ACTIONS,
    discovery_start=0,
    discovery_stop=3,
    natural_index=3,
    confirmation_start=0,
    confirmation_stop=20,
    seed=SEED,
    posterior_version=POSTERIOR_VERSION,
    skill_library_version=SKILL_LIBRARY_VERSION,
    discovery_epoch=0,
    validation_epoch=1,
    activation_epoch=2,
)

EPOCH1_SPEC = SkillEvidenceRoundSpec(
    round_id=1,
    experiment_version="jointqa.mace-skill-evidence.epoch1.v1",
    candidate_actions=ROUND1_CANDIDATE_ACTIONS,
    discovery_start=4,
    discovery_stop=7,
    natural_index=7,
    confirmation_start=20,
    confirmation_stop=40,
    seed=20260819,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch2.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch4.v1",
    discovery_epoch=2,
    validation_epoch=3,
    activation_epoch=4,
)

EPOCH2_SPEC = SkillEvidenceRoundSpec(
    round_id=2,
    experiment_version="jointqa.mace-skill-evidence.epoch2.v1",
    candidate_actions=ROUND2_CANDIDATE_ACTIONS,
    discovery_start=13,
    discovery_stop=16,
    natural_index=16,
    confirmation_start=40,
    confirmation_stop=60,
    seed=20260820,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch3.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch6.v1",
    discovery_epoch=4,
    validation_epoch=5,
    activation_epoch=6,
)

EPOCH3_SPEC = SkillEvidenceRoundSpec(
    round_id=3,
    experiment_version="jointqa.mace-skill-evidence.epoch3.v1",
    candidate_actions=ROUND3_CANDIDATE_ACTIONS,
    discovery_start=17,
    discovery_stop=20,
    natural_index=20,
    confirmation_start=32,
    confirmation_stop=52,
    seed=20260821,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch4.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch8.v1",
    discovery_epoch=6,
    validation_epoch=7,
    activation_epoch=8,
    confirmation_source="development",
)

EPOCH4_SPEC = SkillEvidenceRoundSpec(
    round_id=4,
    experiment_version="jointqa.mace-skill-evidence.epoch4.v1",
    candidate_actions=ROUND4_CANDIDATE_ACTIONS,
    discovery_start=48,
    discovery_stop=51,
    natural_index=51,
    confirmation_start=0,
    confirmation_stop=40,
    seed=20260822,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch5.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch11.v1",
    discovery_epoch=9,
    validation_epoch=10,
    activation_epoch=11,
    confirmation_source="skill_confirmation_round4",
)

EPOCH5_SPEC = SkillEvidenceRoundSpec(
    round_id=5,
    experiment_version="jointqa.mace-skill-evidence.epoch5.v1",
    candidate_actions=ROUND5_CANDIDATE_ACTIONS,
    discovery_start=52,
    discovery_stop=55,
    natural_index=55,
    confirmation_start=0,
    confirmation_stop=40,
    seed=20260823,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch6.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch13.v1",
    discovery_epoch=12,
    validation_epoch=13,
    activation_epoch=14,
    confirmation_source="skill_confirmation_round5",
    prompt_version="agentgraph.director.progressive_subgraph.intermediate-partial.v2",
)

EPOCH6_SPEC = SkillEvidenceRoundSpec(
    round_id=6,
    experiment_version="jointqa.mace-skill-evidence.epoch6.v1",
    candidate_actions=ROUND5_CANDIDATE_ACTIONS,
    discovery_start=56,
    discovery_stop=59,
    natural_index=59,
    confirmation_start=0,
    confirmation_stop=40,
    seed=20260824,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch7.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch15.v1",
    discovery_epoch=15,
    validation_epoch=16,
    activation_epoch=17,
    confirmation_source="skill_confirmation_round6",
    prompt_version="agentgraph.director.progressive_subgraph.intermediate-partial.v2",
    tool_version="agentgraph.add-subgraph-nullable-output+skillflow-public-retrieval.v2",
)

EPOCH7_SPEC = SkillEvidenceRoundSpec(
    round_id=7,
    experiment_version="jointqa.mace-skill-evidence.epoch7.v1",
    candidate_actions=ROUND7_CANDIDATE_ACTIONS,
    discovery_start=60,
    discovery_stop=63,
    natural_index=63,
    confirmation_start=0,
    confirmation_stop=40,
    seed=20260825,
    posterior_version="jointqa.bayesian-linear.progressive-subgraph.epoch8.v1",
    skill_library_version="jointqa.skill-library.progressive.epoch18.v1",
    discovery_epoch=18,
    validation_epoch=19,
    activation_epoch=20,
    confirmation_source="skill_confirmation_round7",
    prompt_version=(
        "agentgraph.director.progressive-subgraph.stage-conditioned-skill.v3"
    ),
    tool_version=(
        "agentgraph.add-subgraph-nullable-output+"
        "skillflow-stage-conditioned-forced-probe.v3"
    ),
    candidate_graph_stages={
        "subject_relation_answer_type_grounding": "construction",
        "exact_answer_handoff": "before_final_answer",
    },
)

ROUND_SPECS: Mapping[int, SkillEvidenceRoundSpec] = {
    0: EPOCH0_SPEC,
    1: EPOCH1_SPEC,
    2: EPOCH2_SPEC,
    3: EPOCH3_SPEC,
    4: EPOCH4_SPEC,
    5: EPOCH5_SPEC,
    6: EPOCH6_SPEC,
    7: EPOCH7_SPEC,
}


def _spec_for_round(round_id: int) -> SkillEvidenceRoundSpec:
    try:
        return ROUND_SPECS[round_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Skill evidence round: {round_id}") from exc


def _reserved_training_positions() -> dict[str, int]:
    """Read the formal GRPO task positions from the existing training config."""

    config = load_yaml(TRAINING_CONFIG)
    joint = config.get("data", {}).get("joint_qa_micro", {})
    configured = joint.get("task_positions", {})
    positions: dict[str, int] = {}
    for dataset in DATASETS:
        values = configured.get(dataset)
        if (
            not isinstance(values, list)
            or len(values) != 1
            or isinstance(values[0], bool)
            or not isinstance(values[0], int)
            or values[0] < 0
        ):
            raise RuntimeError(
                f"training config must declare one non-negative position for {dataset}"
            )
        positions[dataset] = values[0]
    return positions


def _dataset_key(task: TaskRecord) -> str:
    value = task.metadata.get("dataset_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task has no dataset key: {task.task_id}")
    return value


def _require_partition(task: TaskRecord, expected: str) -> None:
    observed = task.metadata.get("joint_qa_partition")
    if observed != expected:
        raise RuntimeError(
            f"task {task.task_id} belongs to partition {observed!r}, expected {expected!r}"
        )


def _condition(dataset: str, graph_stage: str = "*") -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if graph_stage not in {
        "*",
        "empty_graph",
        "construction",
        "before_final_answer",
    }:
        raise ValueError(f"unsupported graph stage: {graph_stage}")
    return {"task_family": dataset, "graph_stage": graph_stage, "tags": []}


def _candidate_condition(
    dataset: str,
    candidate_id: str,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> dict[str, Any]:
    if candidate_id not in spec.candidate_actions:
        raise ValueError(f"unknown candidate action: {candidate_id}")
    graph_stage = spec.candidate_graph_stages.get(candidate_id, "*")
    return _condition(dataset, graph_stage)


def _prompt_condition(
    dataset: str,
    candidate_id: str,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> dict[str, Any]:
    action = dict(spec.candidate_actions[candidate_id])
    return {
        "condition_id": candidate_id,
        "application_mode": "forced_probe_condition",
        "condition": _candidate_condition(dataset, candidate_id, spec),
        "action": action,
        "content": (
            "Predeclared paired-intervention condition. This is an optional, "
            "rejectable prompt prior; apply it only when it fits the current task. "
            f"Suggested action: {action['instruction']}"
        ),
        "rejectable": True,
    }


def _selected_tasks(
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> tuple[
    dict[str, tuple[TaskRecord, ...]],
    dict[str, tuple[TaskRecord, ...]],
    dict[str, TaskRecord],
]:
    train_path = ROOT / "data/joint_qa_v2/train.jsonl"
    confirmation_path = ROOT / "data/joint_qa_v2/skill_confirmation.jsonl"
    round4_confirmation_path = ROOT / "data/joint_qa_v2/skill_confirmation_round4.jsonl"
    round5_confirmation_path = ROOT / "data/joint_qa_v2/skill_confirmation_round5.jsonl"
    round6_confirmation_path = ROOT / "data/joint_qa_v2/skill_confirmation_round6.jsonl"
    round7_confirmation_path = ROOT / "data/joint_qa_v2/skill_confirmation_round7.jsonl"
    development_path = ROOT / "data/joint_qa_v2/development.jsonl"
    train = tuple(iter_task_records(train_path, expected_split="train"))
    validation_sources = {
        "skill_confirmation": tuple(
            iter_task_records(confirmation_path, expected_split="validation")
        ),
        "development": tuple(
            iter_task_records(development_path, expected_split="validation")
        ),
        "skill_confirmation_round4": tuple(
            iter_task_records(round4_confirmation_path, expected_split="validation")
        ),
        "skill_confirmation_round5": tuple(
            iter_task_records(round5_confirmation_path, expected_split="validation")
        ),
        "skill_confirmation_round6": tuple(
            iter_task_records(round6_confirmation_path, expected_split="validation")
        ),
        "skill_confirmation_round7": tuple(
            iter_task_records(round7_confirmation_path, expected_split="validation")
        ),
    }
    if spec.confirmation_source not in validation_sources:
        raise RuntimeError(
            f"unsupported confirmation source: {spec.confirmation_source}"
        )
    discovery: dict[str, tuple[TaskRecord, ...]] = {}
    confirmation: dict[str, tuple[TaskRecord, ...]] = {}
    natural: dict[str, TaskRecord] = {}
    reserved_positions = _reserved_training_positions() if spec.round_id > 0 else {}
    for dataset in DATASETS:
        dataset_train = tuple(task for task in train if _dataset_key(task) == dataset)
        dataset_validation = tuple(
            task
            for task in validation_sources[spec.confirmation_source]
            if _dataset_key(task) == dataset
        )
        required_train = max(
            spec.discovery_stop,
            spec.natural_index + 1,
            reserved_positions.get(dataset, -1) + 1,
        )
        if (
            len(dataset_train) < required_train
            or len(dataset_validation) < spec.confirmation_stop
        ):
            raise RuntimeError(f"insufficient aligned tasks for {dataset}")
        discovery[dataset] = dataset_train[spec.discovery_start : spec.discovery_stop]
        natural[dataset] = dataset_train[spec.natural_index]
        confirmation[dataset] = dataset_validation[
            spec.confirmation_start : spec.confirmation_stop
        ]
        for task in (*discovery[dataset], natural[dataset]):
            _require_partition(task, "train")
        for task in confirmation[dataset]:
            _require_partition(task, spec.confirmation_source)
    all_ids = [
        task.task_id
        for dataset in DATASETS
        for task in (
            *discovery[dataset],
            natural[dataset],
            *confirmation[dataset],
        )
    ]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(
            "discovery, natural-candidate, and confirmation tasks overlap"
        )

    # Every later evidence epoch is disjoint from all prior evidence, held-out
    # development and test, and the dataset-specific task positions reserved by
    # the formal GRPO training config.
    # Enforce this at selection time instead of relying only on the manifest.
    if spec.round_id > 0:
        prior_evidence_ids: set[str] = set()
        reserved_training_ids: set[str] = set()
        for dataset in DATASETS:
            dataset_train = tuple(
                task for task in train if _dataset_key(task) == dataset
            )
            for prior_round_id, prior_spec in ROUND_SPECS.items():
                if prior_round_id >= spec.round_id:
                    continue
                prior_validation = tuple(
                    task
                    for task in validation_sources[prior_spec.confirmation_source]
                    if _dataset_key(task) == dataset
                )
                prior_evidence_ids.update(
                    task.task_id
                    for task in (
                        *dataset_train[
                            prior_spec.discovery_start : prior_spec.discovery_stop
                        ],
                        dataset_train[prior_spec.natural_index],
                        *prior_validation[
                            prior_spec.confirmation_start : prior_spec.confirmation_stop
                        ],
                    )
                )
            reserved_training_ids.add(
                dataset_train[reserved_positions[dataset]].task_id
            )
        # Step0/Skill-on/Step1 development evaluation is fixed to the first 32
        # sequential tasks per dataset.  Later development tasks may serve as
        # independent validation, but the fixed evaluation block and all test
        # tasks remain protected.
        development = tuple(
            iter_task_records(development_path, expected_split="validation")
        )
        fixed_development_ids = {
            task.task_id
            for dataset in DATASETS
            for task in tuple(
                item for item in development if _dataset_key(item) == dataset
            )[:32]
        }
        final_test_ids = {
            task.task_id
            for task in iter_task_records(
                ROOT / "data/joint_qa_v2/test.jsonl", expected_split="test"
            )
        }
        selected_ids = set(all_ids)
        forbidden = selected_ids & (
            prior_evidence_ids
            | reserved_training_ids
            | fixed_development_ids
            | final_test_ids
        )
        if forbidden:
            raise RuntimeError(
                f"round{spec.round_id} evidence tasks overlap prior evidence, "
                "held-out, or reserved training tasks: " + ", ".join(sorted(forbidden))
            )
    return discovery, confirmation, natural


def _augment_trivia(
    discovery: dict[str, tuple[TaskRecord, ...]],
    confirmation: dict[str, tuple[TaskRecord, ...]],
    natural: dict[str, TaskRecord],
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> None:
    config = load_yaml(
        ROOT / "config/evaluation_joint_qa_progressive_step0_triviaqa.yaml"
    )
    ordered = (
        *discovery["triviaqa"],
        natural["triviaqa"],
        *confirmation["triviaqa"],
    )
    receipt_path = spec.output_root / "triviaqa_retrieval_receipts.jsonl"
    receipts = _prepare_retrieval(ordered, config, receipt_path)
    augmented = {
        task.task_id: augment_task_with_retrieval(task, receipts[task.task_id])
        for task in ordered
    }
    discovery["triviaqa"] = tuple(
        augmented[task.task_id] for task in discovery["triviaqa"]
    )
    confirmation["triviaqa"] = tuple(
        augmented[task.task_id] for task in confirmation["triviaqa"]
    )
    natural["triviaqa"] = augmented[natural["triviaqa"].task_id]


def _backend(spec: SkillEvidenceRoundSpec = EPOCH0_SPEC) -> LiveSmokeBackend:
    config = deepcopy(load_yaml(EVALUATION_CONFIG))
    config["storage"]["root"] = str(spec.evidence_root)
    config["skills"]["enabled"] = False
    config["experiment"]["prompt_version"] = spec.prompt_version
    config["experiment"]["tool_version"] = spec.tool_version
    config["experiment"]["condition_id"] = (
        "joint_qa_progressive_skill_epoch0"
        if spec.round_id == 0
        else f"joint_qa_progressive_skill_epoch{spec.round_id}"
    )
    config["experiment"]["sampling_schedule_purpose"] = (
        "joint_qa_progressive_skill_paired_v1"
        if spec.round_id == 0
        else f"joint_qa_progressive_skill_epoch{spec.round_id}_paired_v1"
    )
    return LiveSmokeBackend.from_config(config, ROOT, evaluation_only=True)


def _versions(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
):
    return version_bundle_for(
        task,
        policy_version=POLICY_VERSION,
        model_catalog_version=backend.model_catalog_version,
        prompt_version=spec.prompt_version,
        tool_version=spec.tool_version,
        encoder_version=ENCODER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        posterior_version=spec.posterior_version,
        skill_library_version=spec.skill_library_version,
    )


def _executor_versions(backend: LiveSmokeBackend) -> dict[str, str]:
    return {
        model_id: (
            f"{backend.registry.require_model(model_id).provider_id}:"
            f"{backend.registry.require_model(model_id).model_name}"
        )
        for model_id in backend.registry.model_ids
    }


def _resume_trajectories(
    store: EvidenceStore,
) -> dict[tuple[str, str], TrajectoryRecord]:
    result: dict[tuple[str, str], TrajectoryRecord] = {}
    for payload in store.trajectories.payloads():
        record = TrajectoryRecord.from_dict(payload)
        key = (record.task.task_id, record.condition_id)
        if key in result and result[key].trajectory_id != record.trajectory_id:
            raise RuntimeError(f"ambiguous persisted rollout for {key}")
        result[key] = record
    return result


def _append_posterior_once(store: EvidenceStore, record: Any) -> None:
    """Resume an immutable posterior snapshot without changing its timestamp."""

    persisted = store.posteriors.get(record.posterior_id)
    if persisted is None:
        store.append_posterior(record)
        return
    expected = record.to_dict()
    persisted_semantics = {
        key: value for key, value in persisted.items() if key != "created_at"
    }
    expected_semantics = {
        key: value for key, value in expected.items() if key != "created_at"
    }
    if json.dumps(
        persisted_semantics,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) != json.dumps(
        expected_semantics,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise RuntimeError(
            f"persisted posterior semantics differ: {record.posterior_id}"
        )


async def _arm(
    backend: LiveSmokeBackend,
    cache: dict[tuple[str, str], TrajectoryRecord],
    task: TaskRecord,
    *,
    condition_id: str,
    schedule_purpose: str,
    prompt_priors: Sequence[Mapping[str, Any]],
    stage_conditioned_prompt_prior: Mapping[str, Any] | None = None,
    forced_probe: bool,
    anchor: int,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> TrajectoryRecord:
    key = (task.task_id, condition_id)
    existing = cache.get(key)
    versions = _versions(backend, task, spec)
    if existing is not None:
        if existing.task.to_dict() != task.to_dict() or existing.versions != versions:
            raise RuntimeError(f"persisted rollout regime differs for {key}")
        return existing
    record = await backend.collect(
        task,
        0,
        versions,
        expected_task_split=task.split,
        condition_id=condition_id,
        sampling_schedule_purpose=schedule_purpose,
        prompt_priors=prompt_priors,
        stage_conditioned_prompt_prior=stage_conditioned_prompt_prior,
        forced_probe=forced_probe,
        condition_satisfied=True,
        sampling_anchor_ordinal=anchor,
    )
    cache[key] = record
    return record


def _valid_outcome(record: TrajectoryRecord) -> None:
    if not record.natural_policy_terminal or not record.evaluation.valid:
        raise RuntimeError(f"invalid terminal trajectory: {record.trajectory_id}")
    if record.evaluation.reward is None:
        raise RuntimeError(f"missing terminal reward: {record.trajectory_id}")
    if record.evaluation.evaluator_version != record.versions.evaluator:
        raise RuntimeError(f"evaluator/version mismatch: {record.trajectory_id}")
    if record.api_fallback_used or record.manual_repair_used:
        raise RuntimeError(
            f"fallback/manual repair in paired evidence: {record.trajectory_id}"
        )


def _empty_snapshot_id(task: TaskRecord, stage: str, anchor: int) -> str:
    return stable_id(
        "canvas_snapshot",
        {
            "task_id": task.task_id,
            "task_split": task.split,
            "stage": stage,
            "anchor": anchor,
            "graph": {"nodes": [], "relations": [], "output_agent_id": None},
        },
    )


async def _paired_probe(
    backend: LiveSmokeBackend,
    cache: dict[tuple[str, str], TrajectoryRecord],
    scheduler: JointQAPosteriorScheduler,
    task: TaskRecord,
    candidate_id: str,
    *,
    stage: str,
    anchor: int,
    order_rng: np.random.Generator,
    sampling_probability: float,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> tuple[SkillProbeEvidence, dict[str, Any]]:
    dataset = _dataset_key(task)
    if stage not in {"discovery", "confirmation"}:
        raise ValueError("stage must be discovery or confirmation")
    order = randomize_probe_order("incumbent", "candidate", order_rng)
    schedule = (
        f"joint_qa_progressive_skill_{stage}_paired_v1"
        if spec.round_id == 0
        else f"joint_qa_progressive_skill_epoch{spec.round_id}_{stage}_paired_v1"
    )
    condition_ids = {
        "incumbent": f"jointqa_skill_{stage}:incumbent",
        "candidate": f"jointqa_skill_{stage}:candidate:{candidate_id}",
    }
    candidate_prior = _prompt_condition(dataset, candidate_id, spec)
    candidate_stage = candidate_prior["condition"]["graph_stage"]
    stage_conditioned_candidate = candidate_stage != "*"
    priors = {
        "incumbent": (),
        "candidate": () if stage_conditioned_candidate else (candidate_prior,),
    }
    stage_conditioned_priors = {
        "incumbent": None,
        "candidate": candidate_prior if stage_conditioned_candidate else None,
    }
    observed: dict[str, TrajectoryRecord] = {}
    for arm_name in (order.presented_first, order.presented_second):
        observed[arm_name] = await _arm(
            backend,
            cache,
            task,
            condition_id=condition_ids[arm_name],
            schedule_purpose=schedule,
            prompt_priors=priors[arm_name],
            stage_conditioned_prompt_prior=stage_conditioned_priors[arm_name],
            forced_probe=True,
            anchor=anchor,
            spec=spec,
        )
        _valid_outcome(observed[arm_name])
        if not observed[arm_name].forced_probe or observed[arm_name].grpo_eligible:
            raise RuntimeError("paired arm did not preserve forced-probe isolation")
    incumbent = observed["incumbent"]
    candidate = observed["candidate"]
    if incumbent.director_sampling != candidate.director_sampling:
        raise RuntimeError("paired arms do not share the frozen sampling coordinate")
    snapshot_id = _empty_snapshot_id(task, stage, anchor)
    backend.evidence_store.append_snapshot(
        {
            "snapshot_id": snapshot_id,
            "problem_id": task.task_id,
            "task_split": task.split,
            "prefix_stage": "empty_canvas",
            "graph": {"nodes": [], "relations": [], "output_agent_id": None},
            "policy_version": POLICY_VERSION,
            "stage": stage,
            "sampling_anchor_ordinal": anchor,
        }
    )
    probe_id = stable_id(
        "probe",
        {
            "problem_id": task.task_id,
            "stage": stage,
            "candidate_id": candidate_id,
            "snapshot_id": snapshot_id,
            "policy_version": POLICY_VERSION,
        },
    )
    persisted_probe = backend.evidence_store.resolve_probe(probe_id)
    if persisted_probe is None:
        probe = ProbeRecord(
            probe_id=probe_id,
            problem_id=task.task_id,
            task_split=task.split,
            snapshot_id=snapshot_id,
            policy_version=POLICY_VERSION,
            state_features=scheduler.features.to_state_features(dataset, candidate_id),
            incumbent_action=dict(BASELINE_ACTION),
            candidate_action=dict(spec.candidate_actions[candidate_id]),
            sampling_probability=sampling_probability,
            incumbent_returns=(float(incumbent.evaluation.reward),),
            candidate_returns=(float(candidate.evaluation.reward),),
            executor_versions=_executor_versions(backend),
            evaluator_version=incumbent.versions.evaluator,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            branch_order=(order.presented_first, order.presented_second),
        )
        evidence = SkillProbeEvidence(
            probe=probe,
            condition=_candidate_condition(dataset, candidate_id, spec),
            runtime_version=RUNTIME_VERSION,
            model_catalog_version=backend.model_catalog_version,
        )
    else:
        record_fields = {
            key: persisted_probe[key]
            for key in (
                "probe_id",
                "problem_id",
                "task_split",
                "snapshot_id",
                "policy_version",
                "state_features",
                "incumbent_action",
                "candidate_action",
                "sampling_probability",
                "incumbent_returns",
                "candidate_returns",
                "executor_versions",
                "evaluator_version",
                "feature_schema_version",
                "branch_order",
                "created_at",
            )
        }
        probe = ProbeRecord(**record_fields)
        evidence = SkillProbeEvidence(
            probe=probe,
            condition=persisted_probe["condition"],
            runtime_version=str(persisted_probe["runtime_version"]),
            model_catalog_version=str(persisted_probe["model_catalog_version"]),
        )
    row = {
        "schema_version": "flowsteer.joint-qa.paired-observation.v1",
        "probe": probe.to_dict(),
        "dataset": dataset,
        "stage": stage,
        "candidate_id": candidate_id,
        "incumbent_trajectory_id": incumbent.trajectory_id,
        "candidate_trajectory_id": candidate.trajectory_id,
        "incumbent_exact_match": float(
            incumbent.evaluation.metrics.get("exact_match", 0.0)
        ),
        "candidate_exact_match": float(
            candidate.evaluation.metrics.get("exact_match", 0.0)
        ),
        "exact_match_effect": float(
            candidate.evaluation.metrics.get("exact_match", 0.0)
            - incumbent.evaluation.metrics.get("exact_match", 0.0)
        ),
        "incumbent_token_f1": float(incumbent.evaluation.reward),
        "candidate_token_f1": float(candidate.evaluation.reward),
        "token_f1_effect": probe.paired_effect,
        "condition": _candidate_condition(dataset, candidate_id, spec),
        "candidate_action": dict(spec.candidate_actions[candidate_id]),
        "estimand": (
            "stage_conditioned_skill_prompt_prior_assignment_"
            "intent_to_treat_effect_on_official_token_f1"
            if spec.candidate_graph_stages
            else "skill_prompt_prior_visibility_intent_to_treat_effect"
        ),
        "treatment_assigned": True,
        "prompt_prior_visible": bool(candidate.condition_satisfied),
        "prompt_prior_exposure_rounds": list(
            backend._prompt_prior_exposure_rounds(candidate, candidate_id)
        ),
        "director_adoption_verified": False,
        "condition_ids": condition_ids,
        "branch_order": [order.presented_first, order.presented_second],
        "shared_director_sampling": dict(incumbent.director_sampling),
        "versions": incumbent.versions.to_dict(),
        "executor_versions": _executor_versions(backend),
        "evaluator_version": incumbent.evaluation.evaluator_version,
        "forced_probe": True,
        "grpo_eligible": False,
    }
    return evidence, row


def _proposal(
    dataset: str,
    candidate_id: str,
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> StructuredSkillCandidate:
    return StructuredSkillCandidate(
        skill_id=f"jointqa.{dataset}.{candidate_id}",
        condition=_candidate_condition(dataset, candidate_id, spec),
        action=dict(spec.candidate_actions[candidate_id]),
        baseline_id="frozen_progressive_step0_no_skill",
        baseline_action=dict(BASELINE_ACTION),
        failure_scope=(),
    )


def _manifest(
    discovery: Mapping[str, Sequence[TaskRecord]],
    confirmation: Mapping[str, Sequence[TaskRecord]],
    natural: Mapping[str, TaskRecord],
    spec: SkillEvidenceRoundSpec = EPOCH0_SPEC,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "flowsteer.joint-qa.mace-bayesian-skill.v2",
        "seed": spec.seed,
        "policy_version": POLICY_VERSION,
        "adapter_name": BEHAVIOR_ADAPTER_NAME,
        "encoder_version": ENCODER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "posterior_version": spec.posterior_version,
        "skill_library_version": spec.skill_library_version,
        "prompt_version": spec.prompt_version,
        "tool_version": spec.tool_version,
        "runtime_version": RUNTIME_VERSION,
        "primary_outcome": "official_answer_token_f1",
        "companion_outcome": "normalized_exact_match",
        "partition_manifest": "data/joint_qa_v2/manifest.json",
        "additional_confirmation_manifest": (
            f"data/joint_qa_v2/{spec.confirmation_source}_manifest.json"
            if spec.confirmation_source.startswith("skill_confirmation_round")
            else None
        ),
        "development_block": "joint_qa_v2/development",
        "confirmation_block": (
            "joint_qa_v2/skill_confirmation:first20_per_dataset"
            if spec.round_id == 0
            else (
                f"joint_qa_v2/{spec.confirmation_source}:"
                f"[{spec.confirmation_start}:{spec.confirmation_stop}]_per_dataset "
                "(zero-based, stop-exclusive)"
            )
        ),
        "final_test_block": (
            "joint_qa_v2/test task IDs are read only for overlap exclusion; "
            "answers and metrics are excluded from evidence, posterior fitting, and training"
        ),
        "candidate_actions": {
            key: dict(value) for key, value in spec.candidate_actions.items()
        },
        "candidate_source_artifacts": [
            "reports/joint_qa_progressive/skill_epoch_000002/publication_results.json",
            *(
                [
                    "reports/joint_qa_progressive/skill_epoch_000003/publication_results.json"
                ]
                if spec.round_id >= 4
                else []
            ),
            *(
                [
                    "reports/joint_qa_progressive/skill_epoch_000004/publication_results.json"
                ]
                if spec.round_id >= 5
                else []
            ),
            *(
                ["reports/joint_qa_progressive/skill_epoch_000005/abort_report.json"]
                if spec.round_id >= 6
                else []
            ),
            *(
                [
                    "reports/joint_qa_progressive/skill_epoch_000006/"
                    "publication_results.json"
                ]
                if spec.round_id >= 7
                else []
            ),
            "reports/hotpotqa_multiagent_skill",
            "reports/joint_qa_curve",
            "reports/joint_qa_mace_skill",
            "artifacts/joint_qa_progressive/step_000000/*/agentgraph_trajectories.jsonl",
        ],
        "baseline_action": dict(BASELINE_ACTION),
        "skill_gate": SkillGateConfig().to_dict(),
        "evsi_budget": {
            "ucb_prefilter_top_k": 2,
            "posterior_particles": 1024,
            "observation_samples": 2048,
        },
        "epochs": {
            "discovery": spec.discovery_epoch,
            "validation": spec.validation_epoch,
            "eligible_activation": spec.activation_epoch,
        },
        "discovery_schedule": (
            "balanced cold start, posterior-UCB prefilter, then particle EVSI"
        ),
        "intervention_scope": "full trajectory from a shared empty-Canvas snapshot",
        "causal_estimand": "Skill prompt-prior visibility intent-to-treat effect",
        "not_a_prefix_topology_intervention": True,
        "forced_probe_excluded_from_grpo_and_benchmark": True,
        "natural_candidate_tasks": {key: task.task_id for key, task in natural.items()},
        "discovery_tasks": {
            key: [task.task_id for task in values] for key, values in discovery.items()
        },
        "confirmation_tasks": {
            key: [task.task_id for task in values]
            for key, values in confirmation.items()
        },
    }

    if spec.candidate_graph_stages:
        manifest.update(
            {
                "candidate_conditions": {
                    dataset: {
                        candidate_id: _candidate_condition(
                            dataset,
                            candidate_id,
                            spec,
                        )
                        for candidate_id in spec.candidate_actions
                    }
                    for dataset in DATASETS
                },
                "intervention_scope": (
                    "candidate-specific graph-stage prompt-prior assignment "
                    "from a shared empty-Canvas snapshot"
                ),
                "causal_estimand": (
                    "Stage-conditioned Skill prompt-prior assignment "
                    "intent-to-treat effect"
                ),
            }
        )

    # Epoch0 remains serialization-compatible with the already persisted
    # protocol.  Later rounds add explicit coordinates and the scope boundary
    # that answer-quality evidence cannot itself establish topology adoption.
    if spec.round_id > 0:
        manifest.update(
            {
                "round_id": spec.round_id,
                "experiment_version": spec.experiment_version,
                "selection_coordinates": {
                    "train_discovery": {
                        "start": spec.discovery_start,
                        "stop": spec.discovery_stop,
                        "zero_based_stop_exclusive": True,
                    },
                    "train_natural_candidate_position": spec.natural_index,
                    "skill_confirmation": {
                        "partition": spec.confirmation_source,
                        "start": spec.confirmation_start,
                        "stop": spec.confirmation_stop,
                        "zero_based_stop_exclusive": True,
                    },
                    "reserved_grpo_training_positions": (
                        _reserved_training_positions()
                    ),
                },
                "topology_adoption_acceptance": {
                    "verified_by_this_protocol": False,
                    "requires_independent_evaluation": True,
                    "reason": (
                        (
                            "The paired intervention estimates graph-stage-conditioned "
                            "Skill prompt-prior assignment intent-to-treat effect from "
                            "an empty Canvas; terminal answer F1 does not verify "
                            "Director adoption of parallel branches or semantic fan-in."
                        )
                        if spec.candidate_graph_stages
                        else (
                            "The paired intervention estimates full-trajectory Skill "
                            "prompt-prior visibility intent-to-treat effect from an "
                            "empty Canvas; terminal answer F1 does not verify Director "
                            "adoption of parallel branches or semantic fan-in."
                        )
                    ),
                },
            }
        )
    return manifest


def _guard_output_identity(
    spec: SkillEvidenceRoundSpec,
    manifest: Mapping[str, Any],
) -> None:
    """Permit an exact resume but refuse to claim a non-empty foreign root."""

    identity_keys = (
        "policy_version",
        "adapter_name",
        "posterior_version",
        "skill_library_version",
        "prompt_version",
        "tool_version",
        "seed",
        "candidate_actions",
        "discovery_tasks",
        "natural_candidate_tasks",
        "confirmation_tasks",
    )
    if spec.candidate_graph_stages:
        identity_keys += ("candidate_conditions",)
    roots = [(spec.output_root, spec.manifest_path)]
    if spec.round_id > 0:
        roots.append((spec.report_root, spec.report_root / "experiment_manifest.json"))
    for root, manifest_path in roots:
        if not root.exists() or not any(root.iterdir()):
            continue
        if not manifest_path.is_file():
            raise RuntimeError(
                f"refusing to overwrite non-empty output without manifest: {root}"
            )
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [
            key for key in identity_keys if persisted.get(key) != manifest.get(key)
        ]
        if spec.round_id > 0:
            for key in ("round_id", "experiment_version", "selection_coordinates"):
                if persisted.get(key) != manifest.get(key):
                    mismatched.append(key)
        if mismatched:
            raise RuntimeError(
                "refusing to overwrite output from a different evidence regime: "
                + ", ".join(sorted(set(mismatched)))
            )


def _write_manifest(
    spec: SkillEvidenceRoundSpec,
    manifest: Mapping[str, Any],
) -> None:
    _write_json(spec.manifest_path, manifest)
    if spec.round_id > 0:
        _write_json(spec.report_root / "experiment_manifest.json", manifest)


async def run(*, prepare_only: bool = False, round_id: int = 0) -> dict[str, Any]:
    spec = _spec_for_round(round_id)
    discovery_tasks, confirmation_tasks, natural_tasks = _selected_tasks(spec)
    manifest = _manifest(
        discovery_tasks,
        confirmation_tasks,
        natural_tasks,
        spec,
    )
    _guard_output_identity(spec, manifest)
    _write_manifest(spec, manifest)
    if prepare_only:
        return {"status": "prepared", "manifest": str(spec.manifest_path)}
    _augment_trivia(discovery_tasks, confirmation_tasks, natural_tasks, spec)

    backend = _backend(spec)
    behavior_preflight = await asyncio.to_thread(
        backend.publisher.ensure_loaded_adapter,
        checkpoint_path=str(BEHAVIOR_ADAPTER_CHECKPOINT),
        adapter_name=BEHAVIOR_ADAPTER_NAME,
    )
    manifest["behavior_policy_preflight"] = dict(behavior_preflight)
    manifest["frozen_model_catalog_version"] = backend.model_catalog_version
    manifest["frozen_executor_versions"] = _executor_versions(backend)
    _write_manifest(spec, manifest)
    cache = _resume_trajectories(backend.evidence_store)
    scheduler = JointQAPosteriorScheduler(
        tuple(spec.candidate_actions),
        seed=spec.seed,
        exploration_alpha=1.0,
        prior_precision=1.0,
        observation_variance=0.25,
    )
    order_rng = np.random.default_rng(spec.seed)
    pair_rows: list[dict[str, Any]] = []
    selection_rows: list[SelectionReceipt] = []
    evsi_rows: list[dict[str, Any]] = []
    discovery_evidence: dict[str, list[SkillProbeEvidence]] = {
        dataset: [] for dataset in DATASETS
    }

    # Interleave datasets so the shared posterior receives both task slices.
    for cycle in range(3):
        for dataset_index, dataset in enumerate(DATASETS):
            task = discovery_tasks[dataset][cycle]
            before = scheduler.posterior_record(
                epoch=spec.discovery_epoch,
                policy_version=POLICY_VERSION,
            )
            _append_posterior_once(backend.evidence_store, before)
            scheduled = scheduler.select(dataset)
            sampling_probability = 1.0
            selected_id = scheduled.candidate_id
            selection_mode = scheduled.selection_mode
            if scheduled.decision is not None:
                scores = np.asarray(scheduled.decision.scores, dtype=np.float64)
                prefilter_indices = tuple(
                    sorted(
                        range(len(scores)),
                        key=lambda index: (-float(scores[index]), index),
                    )[: min(2, len(scores))]
                )
                prefiltered = tuple(
                    tuple(spec.candidate_actions)[index] for index in prefilter_indices
                )
                evsi = scheduler.rank_probes_by_evsi(
                    dataset,
                    candidate_ids=prefiltered,
                    seed=spec.seed + cycle * len(DATASETS) + dataset_index,
                    posterior_particles=1024,
                    observation_samples=2048,
                )
                selected_id = evsi.selected_id
                selection_mode = "posterior_ucb_prefilter_particle_evsi"
                evsi_rows.append(
                    {
                        "schema_version": "flowsteer.joint-qa.evsi-selection.v1",
                        "problem_id": task.task_id,
                        "dataset": dataset,
                        "posterior_id": before.posterior_id,
                        "ucb_candidate_ids": list(tuple(spec.candidate_actions)),
                        "ucb_scores": [float(value) for value in scores],
                        "prefiltered_candidate_ids": list(prefiltered),
                        "evsi_ranked_candidate_ids": list(evsi.candidate_ids),
                        "evsi_values": list(evsi.values),
                        "selected_id": selected_id,
                        "posterior_particles": evsi.posterior_particles,
                        "observation_samples": evsi.observation_samples,
                        "observation_std": evsi.observation_std,
                    }
                )
            means = {
                candidate_id: scheduler.predict(dataset, candidate_id)[0]
                for candidate_id in spec.candidate_actions
            }
            stds = {
                candidate_id: scheduler.predict(dataset, candidate_id)[1]
                for candidate_id in spec.candidate_actions
            }
            snapshot_id = _empty_snapshot_id(
                task,
                "discovery",
                spec.anchor_offset + 3000 + cycle * len(DATASETS) + dataset_index,
            )
            selection_rows.append(
                SelectionReceipt(
                    selection_id=stable_id(
                        "selection",
                        {
                            "problem_id": task.task_id,
                            "posterior_id": before.posterior_id,
                            "selected_id": selected_id,
                        },
                    ),
                    snapshot_id=snapshot_id,
                    strategy=selection_mode,
                    candidate_ids=tuple(spec.candidate_actions),
                    selected_id=selected_id,
                    predicted_means=means,
                    predicted_stds=stds,
                    sampling_probability=sampling_probability,
                    posterior_id=before.posterior_id,
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                )
            )
            evidence, row = await _paired_probe(
                backend,
                cache,
                scheduler,
                task,
                selected_id,
                stage="discovery",
                anchor=(
                    spec.anchor_offset + 3000 + cycle * len(DATASETS) + dataset_index
                ),
                order_rng=order_rng,
                sampling_probability=sampling_probability,
                spec=spec,
            )
            scheduler.update(
                dataset,
                selected_id,
                evidence.probe.paired_effect,
                observation_id=evidence.probe.probe_id,
            )
            discovery_evidence[dataset].append(evidence)
            pair_rows.append(row)
            _write_jsonl(spec.pair_path, pair_rows)
            _write_jsonl(spec.selection_path, selection_rows)
            _write_jsonl(spec.evsi_path, evsi_rows)

    posterior = scheduler.posterior_record(
        epoch=spec.discovery_epoch,
        policy_version=POLICY_VERSION,
    )
    _append_posterior_once(backend.evidence_store, posterior)
    selected_candidates = {dataset: scheduler.exploit(dataset) for dataset in DATASETS}
    predicted = {
        dataset: {
            "candidate_id": selected_candidates[dataset],
            "mean": scheduler.predict(dataset, selected_candidates[dataset])[0],
            "epistemic_std": scheduler.predict(dataset, selected_candidates[dataset])[
                1
            ],
        }
        for dataset in DATASETS
    }

    pipeline = SkillEvidencePipeline(
        evidence_store=backend.evidence_store,
        skill_store=SkillStore(spec.skill_store_path),
        retrieval_top_k=1,
    )
    candidates = {}
    for dataset_index, dataset in enumerate(DATASETS):
        candidate_id = selected_candidates[dataset]
        proposal = _proposal(dataset, candidate_id, spec)
        existing = pipeline.skill_store.get(proposal.skill_id)
        if existing is None:
            natural = await _arm(
                backend,
                cache,
                natural_tasks[dataset],
                condition_id=f"jointqa_skill_candidate:{dataset}",
                schedule_purpose=(
                    "joint_qa_skill_natural_candidate_v1"
                    if spec.round_id == 0
                    else f"joint_qa_skill_epoch{spec.round_id}_natural_candidate_v1"
                ),
                prompt_priors=(),
                forced_probe=False,
                anchor=spec.anchor_offset + 2000 + dataset_index,
                spec=spec,
            )
            _valid_outcome(natural)
            candidates[dataset] = pipeline.discover(
                natural,
                proposal,
                created_epoch=spec.discovery_epoch,
                runtime_version=RUNTIME_VERSION,
                executor_versions=_executor_versions(backend),
            )
        else:
            candidates[dataset] = existing

    # Confirmation is independent of posterior fitting.  Different problems
    # run concurrently; the two branches of one problem remain sequential in
    # the pre-randomized order.
    semaphore = asyncio.Semaphore(4)

    async def confirm_one(
        dataset: str,
        index: int,
        task: TaskRecord,
    ) -> tuple[SkillProbeEvidence, dict[str, Any]]:
        async with semaphore:
            return await _paired_probe(
                backend,
                cache,
                scheduler,
                task,
                selected_candidates[dataset],
                stage="confirmation",
                anchor=(
                    spec.anchor_offset + 4000 + DATASETS.index(dataset) * 100 + index
                ),
                order_rng=np.random.default_rng(
                    spec.seed + DATASETS.index(dataset) * 100 + index
                ),
                sampling_probability=1.0,
                spec=spec,
            )

    confirmation_jobs = [
        (dataset, index, task)
        for dataset in DATASETS
        for index, task in enumerate(confirmation_tasks[dataset])
    ]
    confirmation_results = await asyncio.gather(
        *(
            confirm_one(dataset, index, task)
            for dataset, index, task in confirmation_jobs
        )
    )
    confirmation_evidence: dict[str, list[SkillProbeEvidence]] = {
        dataset: [] for dataset in DATASETS
    }
    for (dataset, _, _), (evidence, row) in zip(
        confirmation_jobs, confirmation_results, strict=True
    ):
        confirmation_evidence[dataset].append(evidence)
        pair_rows.append(row)
    _write_jsonl(spec.pair_path, pair_rows)

    publications: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        candidate = candidates[dataset]
        if candidate.version >= 2:
            decision = pipeline.gate.evaluate(candidate)
            publications[dataset] = {
                "selected_candidate": selected_candidates[dataset],
                "predicted_before_confirmation": predicted[dataset],
                "skill": candidate.to_dict(),
                "gate": asdict(decision),
                "resumed": True,
            }
            continue
        selected_discovery = [
            evidence
            for evidence in discovery_evidence[dataset]
            if evidence.probe.candidate_action
            == dict(spec.candidate_actions[selected_candidates[dataset]])
        ]
        effects = {
            evidence.probe.problem_id: evidence.probe.paired_effect
            for evidence in confirmation_evidence[dataset]
        }
        statistics, calibration = calibrate_skill_validation(
            effects,
            predicted_mean=float(predicted[dataset]["mean"]),
            predicted_std=float(predicted[dataset]["epistemic_std"]),
            task_family=dataset,
            seed=spec.seed + dataset_index,
        )
        result = pipeline.confirm_and_publish(
            candidate,
            discovery_probes=selected_discovery,
            validation_probes=confirmation_evidence[dataset],
            statistics=statistics,
            validation_epoch=spec.validation_epoch,
            activation_epoch=spec.activation_epoch,
        )
        publications[dataset] = {
            "selected_candidate": selected_candidates[dataset],
            "predicted_before_confirmation": predicted[dataset],
            "calibration": calibration,
            "skill": result.skill.to_dict(),
            "gate": asdict(result.gate_decision),
            "resumed": False,
        }
    result = {
        "schema_version": "flowsteer.joint-qa.skill-publication-result.v1",
        "posterior": posterior.to_dict(),
        "selected_candidates": selected_candidates,
        "publications": publications,
        "active_datasets": [
            dataset
            for dataset, value in publications.items()
            if value["skill"]["status"] == SkillStatus.ACTIVE.value
        ],
        "discovery_pair_count": sum(row["stage"] == "discovery" for row in pair_rows),
        "confirmation_pair_count": sum(
            row["stage"] == "confirmation" for row in pair_rows
        ),
        "forced_probe_trajectories_are_not_training_data": True,
    }
    if spec.round_id > 0:
        result.update(
            {
                "round_id": spec.round_id,
                "experiment_version": spec.experiment_version,
                "seed": spec.seed,
                "posterior_version": spec.posterior_version,
                "skill_library_version": spec.skill_library_version,
                "confirmation_block": {
                    "partition": f"joint_qa_v2/{spec.confirmation_source}",
                    "start": spec.confirmation_start,
                    "stop": spec.confirmation_stop,
                    "zero_based_stop_exclusive": True,
                },
                "causal_estimand": (
                    (
                        "Stage-conditioned Skill prompt-prior assignment "
                        "intent-to-treat effect on official answer token F1 "
                        "from a shared empty Canvas"
                    )
                    if spec.candidate_graph_stages
                    else (
                        "Skill prompt-prior visibility intent-to-treat effect on "
                        "official answer token F1 from a shared empty Canvas"
                    )
                ),
                "topology_adoption_acceptance": {
                    "verified_by_this_protocol": False,
                    "requires_independent_evaluation": True,
                },
            }
        )
    _write_json(spec.publication_path, result)
    if spec.round_id > 0:
        _write_json(spec.report_root / "publication_results.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--round",
        type=int,
        choices=tuple(ROUND_SPECS),
        default=0,
        help="bounded Skill evidence round (default: 0 for backward compatibility)",
    )
    args = parser.parse_args(argv)
    result = asyncio.run(run(prepare_only=args.prepare_only, round_id=args.round))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
