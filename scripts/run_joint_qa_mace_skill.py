#!/usr/bin/env python3
"""Run the bounded joint-QA MACE -> Bayesian posterior -> Skill protocol.

This is an experiment adapter over existing project components, not another
exploration or Skill implementation.  It reuses:

* FlowSteer's progressive Canvas execution through ``LiveSmokeBackend``;
* the existing paired-intervention records and randomized branch order;
* the existing joint Bayesian posterior and MACE-style UCB policy;
* the existing ``SkillEvidencePipeline`` and deterministic evidence gate; and
* SkillFlow's public TriviaQA retrieval observations.

The fixed final benchmark is validation[0:32].  Skill confirmation uses the
disjoint validation[32:52] block and never enters GRPO or benchmark EM/F1.
The paired intervention starts from an empty Canvas, freezes policy/model/
evaluator/tool versions and sampling coordinates, and regenerates downstream
execution for each arm.  Its primary outcome is official answer token F1;
normalized exact match is persisted as a companion outcome.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
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
from src.interactive.records import ProbeRecord, SelectionReceipt, TaskRecord, TrajectoryRecord
from src.interactive.skills import (
    SkillEvidencePipeline,
    SkillProbeEvidence,
    SkillStatus,
    SkillStore,
    StructuredSkillCandidate,
)
from src.interactive.task_dataset import iter_task_records


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/joint_qa_mace_skill"
REPORT_ROOT = ROOT / "reports/joint_qa_mace_skill"
EVIDENCE_ROOT = OUTPUT_ROOT / "evidence"
SKILL_STORE_PATH = OUTPUT_ROOT / "skills.json"
PAIR_PATH = OUTPUT_ROOT / "paired_observations.jsonl"
SELECTION_PATH = OUTPUT_ROOT / "selection_receipts.jsonl"
PUBLICATION_PATH = OUTPUT_ROOT / "publication_results.json"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
RUNTIME_VERSION = "flowsteer.agentgraph.progressive-runtime.v1"
POLICY_VERSION = "qwen35-9b-jointqa-step-000002"
SKILL_LIBRARY_VERSION = "jointqa.skill-library.epoch2.v1"
SEED = 20260818

BASELINE_ACTION: Mapping[str, Any] = {
    "instruction": "No additional prompt prior; use the frozen Director policy."
}

# These two bounded actions are derived from the already persisted Step-2
# wrong-demo categories.  They remain rejectable Director context and never
# mutate the Canvas directly.
CANDIDATE_ACTIONS: Mapping[str, Mapping[str, Any]] = {
    "independent_evidence_fan_in": {
        "instruction": (
            "When the question requires evidence reconciliation, construct a directed "
            "acyclic workflow with two independent Evidence Agents feeding one "
            "Reasoning or Verification Agent, followed by one terminal Format Agent. "
            "Route every required upstream artifact to the terminal path; use reciprocal "
            "communication only when the evidence artifacts conflict."
        )
    },
    "answer_span_verification": {
        "instruction": (
            "Before finish, verify the selected entity and answer granularity against "
            "the question and the cited evidence. Route the verified result to one "
            "terminal Format Agent, which emits exactly one shortest supported answer "
            "span inside a single <answer> tag without explanation."
        )
    },
}


def _dataset_key(task: TaskRecord) -> str:
    value = task.metadata.get("dataset_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task has no dataset key: {task.task_id}")
    return value


def _condition(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    return {"task_family": dataset, "graph_stage": "*", "tags": []}


def _prompt_condition(dataset: str, candidate_id: str) -> dict[str, Any]:
    action = dict(CANDIDATE_ACTIONS[candidate_id])
    return {
        "condition_id": candidate_id,
        "application_mode": "forced_probe_condition",
        "condition": _condition(dataset),
        "action": action,
        "content": (
            "Predeclared paired-intervention condition. This is an optional, "
            "rejectable prompt prior; apply it only when it fits the current task. "
            f"Suggested action: {action['instruction']}"
        ),
        "rejectable": True,
    }


def _selected_tasks() -> tuple[
    dict[str, tuple[TaskRecord, ...]],
    dict[str, tuple[TaskRecord, ...]],
    dict[str, TaskRecord],
]:
    train_path = ROOT / "data/agentgraph_v1/train.jsonl"
    validation_path = ROOT / "data/agentgraph_v1/validation.jsonl"
    train = tuple(iter_task_records(train_path, expected_split="train"))
    validation = tuple(
        iter_task_records(validation_path, expected_split="validation")
    )
    discovery: dict[str, tuple[TaskRecord, ...]] = {}
    confirmation: dict[str, tuple[TaskRecord, ...]] = {}
    natural: dict[str, TaskRecord] = {}
    for dataset in DATASETS:
        dataset_train = tuple(task for task in train if _dataset_key(task) == dataset)
        dataset_validation = tuple(
            task for task in validation if _dataset_key(task) == dataset
        )
        if len(dataset_train) < 4 or len(dataset_validation) < 52:
            raise RuntimeError(f"insufficient aligned tasks for {dataset}")
        discovery[dataset] = dataset_train[:3]
        natural[dataset] = dataset_train[3]
        confirmation[dataset] = dataset_validation[32:52]
    return discovery, confirmation, natural


def _augment_trivia(
    discovery: dict[str, tuple[TaskRecord, ...]],
    confirmation: dict[str, tuple[TaskRecord, ...]],
    natural: dict[str, TaskRecord],
) -> None:
    config = load_yaml(ROOT / "config/evaluation_joint_qa_step2_triviaqa.yaml")
    ordered = (
        *discovery["triviaqa"],
        natural["triviaqa"],
        *confirmation["triviaqa"],
    )
    receipt_path = OUTPUT_ROOT / "triviaqa_retrieval_receipts.jsonl"
    receipts = _prepare_retrieval(ordered, config, receipt_path)
    augmented = {
        task.task_id: augment_task_with_retrieval(task, receipts[task.task_id])
        for task in ordered
    }
    discovery["triviaqa"] = tuple(augmented[task.task_id] for task in discovery["triviaqa"])
    confirmation["triviaqa"] = tuple(
        augmented[task.task_id] for task in confirmation["triviaqa"]
    )
    natural["triviaqa"] = augmented[natural["triviaqa"].task_id]


def _backend() -> LiveSmokeBackend:
    config = deepcopy(
        load_yaml(ROOT / "config/evaluation_joint_qa_step2_hotpotqa.yaml")
    )
    config["storage"]["root"] = str(EVIDENCE_ROOT)
    config["skills"]["enabled"] = False
    config["experiment"]["condition_id"] = "joint_qa_mace_skill"
    config["experiment"]["sampling_schedule_purpose"] = (
        "joint_qa_mace_skill_paired_v1"
    )
    return LiveSmokeBackend.from_config(config, ROOT, evaluation_only=True)


def _versions(backend: LiveSmokeBackend, task: TaskRecord):
    return version_bundle_for(
        task,
        policy_version=POLICY_VERSION,
        model_catalog_version=backend.model_catalog_version,
        prompt_version="agentgraph.director.generic_dependency.v1",
        tool_version="agentgraph.atomic-actions+skillflow-public-retrieval.v1",
        encoder_version=ENCODER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        posterior_version=POSTERIOR_VERSION,
        skill_library_version=SKILL_LIBRARY_VERSION,
    )


def _executor_versions(backend: LiveSmokeBackend) -> dict[str, str]:
    return {
        model_id: (
            f"{backend.registry.require_model(model_id).provider_id}:"
            f"{backend.registry.require_model(model_id).model_name}"
        )
        for model_id in backend.registry.model_ids
    }


def _resume_trajectories(store: EvidenceStore) -> dict[tuple[str, str], TrajectoryRecord]:
    result: dict[tuple[str, str], TrajectoryRecord] = {}
    for payload in store.trajectories.payloads():
        record = TrajectoryRecord.from_dict(payload)
        key = (record.task.task_id, record.condition_id)
        if key in result and result[key].trajectory_id != record.trajectory_id:
            raise RuntimeError(f"ambiguous persisted rollout for {key}")
        result[key] = record
    return result


async def _arm(
    backend: LiveSmokeBackend,
    cache: dict[tuple[str, str], TrajectoryRecord],
    task: TaskRecord,
    *,
    condition_id: str,
    schedule_purpose: str,
    prompt_priors: Sequence[Mapping[str, Any]],
    forced_probe: bool,
    anchor: int,
) -> TrajectoryRecord:
    key = (task.task_id, condition_id)
    existing = cache.get(key)
    versions = _versions(backend, task)
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
        raise RuntimeError(f"fallback/manual repair in paired evidence: {record.trajectory_id}")


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
) -> tuple[SkillProbeEvidence, dict[str, Any]]:
    dataset = _dataset_key(task)
    if stage not in {"discovery", "confirmation"}:
        raise ValueError("stage must be discovery or confirmation")
    order = randomize_probe_order("incumbent", "candidate", order_rng)
    schedule = f"joint_qa_mace_skill_{stage}_paired_v1"
    condition_ids = {
        "incumbent": f"jointqa_skill_{stage}:incumbent",
        "candidate": f"jointqa_skill_{stage}:candidate:{candidate_id}",
    }
    priors = {
        "incumbent": (),
        "candidate": (_prompt_condition(dataset, candidate_id),),
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
            forced_probe=True,
            anchor=anchor,
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
            candidate_action=dict(CANDIDATE_ACTIONS[candidate_id]),
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
            condition=_condition(dataset),
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
        "condition": _condition(dataset),
        "candidate_action": dict(CANDIDATE_ACTIONS[candidate_id]),
        "forced_probe": True,
        "grpo_eligible": False,
    }
    return evidence, row


def _proposal(dataset: str, candidate_id: str) -> StructuredSkillCandidate:
    return StructuredSkillCandidate(
        skill_id=f"jointqa.{dataset}.{candidate_id}",
        condition=_condition(dataset),
        action=dict(CANDIDATE_ACTIONS[candidate_id]),
        baseline_id="frozen_step2_no_skill",
        baseline_action=dict(BASELINE_ACTION),
        failure_scope=(),
    )


def _manifest(
    discovery: Mapping[str, Sequence[TaskRecord]],
    confirmation: Mapping[str, Sequence[TaskRecord]],
    natural: Mapping[str, TaskRecord],
) -> dict[str, Any]:
    return {
        "schema_version": "flowsteer.joint-qa.mace-bayesian-skill.v1",
        "seed": SEED,
        "policy_version": POLICY_VERSION,
        "adapter_name": "theta_jointqa_step_000002",
        "encoder_version": ENCODER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "posterior_version": POSTERIOR_VERSION,
        "skill_library_version": SKILL_LIBRARY_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "primary_outcome": "official_answer_token_f1",
        "companion_outcome": "normalized_exact_match",
        "final_benchmark_block": "validation[0:32]",
        "confirmation_block": "validation[32:52]",
        "candidate_actions": {key: dict(value) for key, value in CANDIDATE_ACTIONS.items()},
        "baseline_action": dict(BASELINE_ACTION),
        "discovery_schedule": "two-candidate balanced cold start then posterior UCB",
        "intervention_scope": "full trajectory from a shared empty-Canvas snapshot",
        "forced_probe_excluded_from_grpo_and_benchmark": True,
        "natural_candidate_tasks": {
            key: task.task_id for key, task in natural.items()
        },
        "discovery_tasks": {
            key: [task.task_id for task in values]
            for key, values in discovery.items()
        },
        "confirmation_tasks": {
            key: [task.task_id for task in values]
            for key, values in confirmation.items()
        },
    }


async def run(*, prepare_only: bool = False) -> dict[str, Any]:
    discovery_tasks, confirmation_tasks, natural_tasks = _selected_tasks()
    _augment_trivia(discovery_tasks, confirmation_tasks, natural_tasks)
    manifest = _manifest(discovery_tasks, confirmation_tasks, natural_tasks)
    _write_json(MANIFEST_PATH, manifest)
    if prepare_only:
        return {"status": "prepared", "manifest": str(MANIFEST_PATH)}

    backend = _backend()
    cache = _resume_trajectories(backend.evidence_store)
    scheduler = JointQAPosteriorScheduler(
        tuple(CANDIDATE_ACTIONS),
        seed=SEED,
        exploration_alpha=1.0,
        prior_precision=1.0,
        observation_variance=0.25,
    )
    order_rng = np.random.default_rng(SEED)
    pair_rows: list[dict[str, Any]] = []
    selection_rows: list[SelectionReceipt] = []
    discovery_evidence: dict[str, list[SkillProbeEvidence]] = {
        dataset: [] for dataset in DATASETS
    }

    # Interleave datasets so the shared posterior receives both task slices.
    for cycle in range(3):
        for dataset_index, dataset in enumerate(DATASETS):
            task = discovery_tasks[dataset][cycle]
            before = scheduler.posterior_record(
                epoch=0,
                policy_version=POLICY_VERSION,
            )
            backend.evidence_store.append_posterior(before)
            scheduled = scheduler.select(dataset)
            sampling_probability = 1.0
            if scheduled.decision is not None:
                scores = np.asarray(scheduled.decision.scores, dtype=np.float64)
                tied = np.flatnonzero(np.isclose(scores, scores.max()))
                sampling_probability = 1.0 / float(len(tied))
            means = {
                candidate_id: scheduler.predict(dataset, candidate_id)[0]
                for candidate_id in CANDIDATE_ACTIONS
            }
            stds = {
                candidate_id: scheduler.predict(dataset, candidate_id)[1]
                for candidate_id in CANDIDATE_ACTIONS
            }
            snapshot_id = _empty_snapshot_id(
                task, "discovery", 3000 + cycle * len(DATASETS) + dataset_index
            )
            selection_rows.append(
                SelectionReceipt(
                    selection_id=stable_id(
                        "selection",
                        {
                            "problem_id": task.task_id,
                            "posterior_id": before.posterior_id,
                            "selected_id": scheduled.candidate_id,
                        },
                    ),
                    snapshot_id=snapshot_id,
                    strategy=scheduled.selection_mode,
                    candidate_ids=tuple(CANDIDATE_ACTIONS),
                    selected_id=scheduled.candidate_id,
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
                scheduled.candidate_id,
                stage="discovery",
                anchor=3000 + cycle * len(DATASETS) + dataset_index,
                order_rng=order_rng,
                sampling_probability=sampling_probability,
            )
            scheduler.update(
                dataset,
                scheduled.candidate_id,
                evidence.probe.paired_effect,
                observation_id=evidence.probe.probe_id,
            )
            discovery_evidence[dataset].append(evidence)
            pair_rows.append(row)
            _write_jsonl(PAIR_PATH, pair_rows)
            _write_jsonl(SELECTION_PATH, selection_rows)

    posterior = scheduler.posterior_record(epoch=0, policy_version=POLICY_VERSION)
    backend.evidence_store.append_posterior(posterior)
    selected_candidates = {
        dataset: scheduler.exploit(dataset) for dataset in DATASETS
    }
    predicted = {
        dataset: {
            "candidate_id": selected_candidates[dataset],
            "mean": scheduler.predict(dataset, selected_candidates[dataset])[0],
            "epistemic_std": scheduler.predict(dataset, selected_candidates[dataset])[1],
        }
        for dataset in DATASETS
    }

    pipeline = SkillEvidencePipeline(
        evidence_store=backend.evidence_store,
        skill_store=SkillStore(SKILL_STORE_PATH),
        retrieval_top_k=1,
    )
    candidates = {}
    for dataset_index, dataset in enumerate(DATASETS):
        candidate_id = selected_candidates[dataset]
        proposal = _proposal(dataset, candidate_id)
        existing = pipeline.skill_store.get(proposal.skill_id)
        if existing is None:
            natural = await _arm(
                backend,
                cache,
                natural_tasks[dataset],
                condition_id=f"jointqa_skill_candidate:{dataset}",
                schedule_purpose="joint_qa_skill_natural_candidate_v1",
                prompt_priors=(),
                forced_probe=False,
                anchor=2000 + dataset_index,
            )
            _valid_outcome(natural)
            candidates[dataset] = pipeline.discover(
                natural,
                proposal,
                created_epoch=0,
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
                anchor=4000 + DATASETS.index(dataset) * 100 + index,
                order_rng=np.random.default_rng(SEED + DATASETS.index(dataset) * 100 + index),
                sampling_probability=1.0,
            )

    confirmation_jobs = [
        (dataset, index, task)
        for dataset in DATASETS
        for index, task in enumerate(confirmation_tasks[dataset])
    ]
    confirmation_results = await asyncio.gather(
        *(confirm_one(dataset, index, task) for dataset, index, task in confirmation_jobs)
    )
    confirmation_evidence: dict[str, list[SkillProbeEvidence]] = {
        dataset: [] for dataset in DATASETS
    }
    for (dataset, _, _), (evidence, row) in zip(
        confirmation_jobs, confirmation_results, strict=True
    ):
        confirmation_evidence[dataset].append(evidence)
        pair_rows.append(row)
    _write_jsonl(PAIR_PATH, pair_rows)

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
            == dict(CANDIDATE_ACTIONS[selected_candidates[dataset]])
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
            seed=SEED + dataset_index,
        )
        result = pipeline.confirm_and_publish(
            candidate,
            discovery_probes=selected_discovery,
            validation_probes=confirmation_evidence[dataset],
            statistics=statistics,
            validation_epoch=1,
            activation_epoch=2,
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
        "discovery_pair_count": sum(
            row["stage"] == "discovery" for row in pair_rows
        ),
        "confirmation_pair_count": sum(
            row["stage"] == "confirmation" for row in pair_rows
        ),
        "forced_probe_trajectories_are_not_training_data": True,
    }
    _write_json(PUBLICATION_PATH, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    result = asyncio.run(run(prepare_only=args.prepare_only))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
