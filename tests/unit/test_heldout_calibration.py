"""CPU-only tests for independent held-out paired calibration receipts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from src.interactive.exploration.combination_ledger import (
    CombinationPosterior,
    DecisionKey,
)
from src.interactive.exploration.heldout_calibration import (
    HeldoutPairedCalibrationManifest,
    build_heldout_paired_calibration,
    load_heldout_paired_calibration,
    save_heldout_paired_calibration,
    validate_heldout_paired_calibration,
)
from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.records import ProbeRecord, TaskRecord
from src.interactive.versioning import VersionBundle


POLICY = "policy-e0"
CONDITION = "condition-e0"
EVALUATOR = "hotpotqa-em-f1-v1"
FEATURE_SCHEMA = "dynamic-combination-decision-key-v1"
CATALOG = "catalog-v1"


def _key(model: str) -> DecisionKey:
    return DecisionKey(
        task_family="hotpotqa",
        role_cluster="solve",
        model_id=model,
        edge_type="unidirectional",
        same_model_as_upstream=False,
        stage="other",
    )


def _versions() -> VersionBundle:
    return VersionBundle(
        policy=POLICY,
        model_catalog=CATALOG,
        evaluator=EVALUATOR,
        prompt="prompt-v1",
        tool="tool-v1",
        encoder="encoder-v1",
        feature_schema=FEATURE_SCHEMA,
    )


@dataclass(frozen=True)
class _Evaluation:
    valid: bool = True


@dataclass(frozen=True)
class _Branch:
    trajectory_id: str
    task: TaskRecord
    condition_id: str
    versions: VersionBundle
    forced_probe: bool = True
    grpo_eligible: bool = False
    evaluation: _Evaluation = _Evaluation()


def _manifest(*, count: int = 20) -> HeldoutPairedCalibrationManifest:
    return HeldoutPairedCalibrationManifest(
        dataset="hotpotqa",
        problem_ids=tuple(f"validation-{index:03d}" for index in range(count)),
        discovery_problem_ids=tuple(f"train-{index:03d}" for index in range(50)),
        discovery_probe_ids=tuple(f"discovery-probe-{index:03d}" for index in range(5)),
        policy_version=POLICY,
        condition_id=CONDITION,
        evaluator_version=EVALUATOR,
        feature_schema_version=FEATURE_SCHEMA,
        model_catalog_version=CATALOG,
        posterior_snapshot_id="ledger-e0",
        minimum_unique_problems=20,
    )


def _evidence(manifest: HeldoutPairedCalibrationManifest):
    probes: list[ProbeRecord] = []
    branches: list[_Branch] = []
    for index, problem_id in enumerate(manifest.problem_ids):
        probe_id = f"heldout-probe-{index:03d}"
        branch_ids = make_probe_branch_ids(probe_id, seed=1000 + index)
        if index % 3 == 0:
            keep, switch = (0, 0, 0), (1, 1, 1)
        elif index % 3 == 1:
            keep, switch = (1, 1, 1), (0, 0, 0)
        else:
            keep, switch = (0, 1, 0), (1, 0, 1)
        probes.append(
            ProbeRecord(
                probe_id=probe_id,
                problem_id=problem_id,
                task_split="validation",
                snapshot_id=f"snapshot-{index:03d}",
                policy_version=POLICY,
                state_features={"is_audit": False},
                incumbent_action=_key("model-a").to_dict(),
                candidate_action=_key("model-b").to_dict(),
                sampling_probability=1.0,
                incumbent_returns=keep,
                candidate_returns=switch,
                executor_versions={"model-a": CATALOG, "model-b": CATALOG},
                evaluator_version=EVALUATOR,
                feature_schema_version=FEATURE_SCHEMA,
                branch_order=branch_ids.execution_order,
            )
        )
        task = TaskRecord(
            task_id=problem_id,
            question="A held-out question",
            ground_truth="answer",
            split="validation",
        )
        branches.extend(
            _Branch(branch_id, task, CONDITION, _versions())
            for branch_id in branch_ids.all_ids
        )
    return tuple(probes), tuple(branches)


def _ledger() -> CombinationPosterior:
    ledger = CombinationPosterior(POLICY)
    ledger.register_key(_key("model-a"))
    ledger.register_key(_key("model-b"))
    return ledger


def test_build_save_and_load_complete_heldout_calibration(tmp_path) -> None:
    manifest = _manifest()
    probes, branches = _evidence(manifest)
    ledger = _ledger()
    before = ledger.dumps()

    receipt = build_heldout_paired_calibration(
        ledger, manifest, probes, branches
    )

    assert receipt.status == "complete"
    assert receipt.problem_count == 20
    assert receipt.probe_count == 20
    assert receipt.empirical_coverage >= 0.95
    assert receipt.to_dict()["brier_score"] is None
    assert receipt.to_dict()["heldout_calibration_gate_passed"] is True
    assert receipt.to_dict()["does_not_authorize_training_by_itself"] is True
    assert ledger.dumps() == before

    path = save_heldout_paired_calibration(receipt, tmp_path / "receipt.json")
    restored = load_heldout_paired_calibration(
        path, expected_manifest=manifest, ledger=ledger
    )
    assert restored.to_dict() == receipt.to_dict()


def test_manifest_requires_twenty_unique_disjoint_validation_problems() -> None:
    with pytest.raises(ValueError, match="fewer than minimum_unique_problems"):
        _manifest(count=19)
    value = _manifest().to_dict()
    value["discovery_problem_ids"][0] = value["problem_ids"][0]
    with pytest.raises(ValueError, match="problem IDs overlap"):
        HeldoutPairedCalibrationManifest.from_dict(value)
    value = _manifest().to_dict()
    value["task_split"] = "train"
    with pytest.raises(ValueError, match="validation split"):
        HeldoutPairedCalibrationManifest.from_dict(value)


def test_builder_rejects_discovery_probe_reuse_and_nonexact_problem_set() -> None:
    manifest = _manifest()
    probes, branches = _evidence(manifest)
    overlapping = replace(
        probes[0], probe_id=manifest.discovery_probe_ids[0]
    )
    replacement_ids = make_probe_branch_ids(overlapping.probe_id, seed=3)
    overlapping = replace(overlapping, branch_order=replacement_ids.execution_order)
    replacement_task = branches[0].task
    replacement_branches = tuple(
        _Branch(identifier, replacement_task, CONDITION, _versions())
        for identifier in replacement_ids.all_ids
    )
    old_ids = set(probes[0].branch_order)
    remaining = tuple(branch for branch in branches if branch.trajectory_id not in old_ids)
    with pytest.raises(ValueError, match="discovery probe IDs overlap"):
        build_heldout_paired_calibration(
            _ledger(),
            manifest,
            (overlapping, *probes[1:]),
            (*replacement_branches, *remaining),
        )
    with pytest.raises(ValueError, match="exactly one probe"):
        build_heldout_paired_calibration(
            _ledger(), manifest, probes[:-1], branches
        )


def test_builder_rejects_split_policy_condition_and_grpo_leakage() -> None:
    manifest = _manifest()
    probes, branches = _evidence(manifest)
    with pytest.raises(ValueError, match="validation split"):
        build_heldout_paired_calibration(
            _ledger(), manifest, (replace(probes[0], task_split="train"), *probes[1:]), branches
        )
    with pytest.raises(ValueError, match="policy version mismatch"):
        build_heldout_paired_calibration(
            _ledger(), manifest, (replace(probes[0], policy_version="wrong"), *probes[1:]), branches
        )
    wrong_condition = (replace(branches[0], condition_id="wrong"), *branches[1:])
    with pytest.raises(ValueError, match="condition ID mismatch"):
        build_heldout_paired_calibration(
            _ledger(), manifest, probes, wrong_condition
        )
    leaked = (replace(branches[0], forced_probe=False, grpo_eligible=True), *branches[1:])
    with pytest.raises(ValueError, match="marked natural"):
        build_heldout_paired_calibration(_ledger(), manifest, probes, leaked)


def test_builder_rejects_executor_and_feature_schema_version_mismatch() -> None:
    manifest = _manifest()
    probes, branches = _evidence(manifest)
    wrong_executor = replace(probes[0], executor_versions={"model-a": "wrong"})
    with pytest.raises(ValueError, match="executor model catalog mismatch"):
        build_heldout_paired_calibration(
            _ledger(), manifest, (wrong_executor, *probes[1:]), branches
        )
    wrong_schema = replace(probes[0], feature_schema_version="wrong")
    with pytest.raises(ValueError, match="feature schema version mismatch"):
        build_heldout_paired_calibration(
            _ledger(), manifest, (wrong_schema, *probes[1:]), branches
        )


def test_validator_recomputes_metrics_and_rejects_another_manifest() -> None:
    manifest = _manifest()
    probes, branches = _evidence(manifest)
    receipt = build_heldout_paired_calibration(
        _ledger(), manifest, probes, branches
    )
    changed = replace(receipt, gaussian_nll_mean=receipt.gaussian_nll_mean + 1.0)
    with pytest.raises(ValueError, match="metric mismatch"):
        validate_heldout_paired_calibration(
            changed, expected_manifest=manifest
        )
    another = replace(manifest, posterior_snapshot_id="another-ledger")
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_heldout_paired_calibration(
            receipt, expected_manifest=another
        )
