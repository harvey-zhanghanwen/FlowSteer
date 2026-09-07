"""CPU-only tests for the runner's committed probe-to-Skill boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.exploration.combination_ledger import (
    CombinationPosterior,
    DecisionKey,
)
from src.interactive.exploration.ledger_epoch import LedgerEpoch
from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.records import ProbeRecord
from src.interactive.versioning import VersionBundle


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_hotpotqa_grpo.py"
SPEC = importlib.util.spec_from_file_location(
    "test_hotpotqa_skill_runtime_runner_boundary_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


BEHAVIOR_POLICY = "qwen35-9b-smoke-step-0001"
PUBLICATION_POLICY = "qwen35-9b-hotpotqa-dynamic-ledger-grpo-step-000001"
EVALUATOR_VERSION = "skillflow.training.reward.v1"
FEATURE_SCHEMA_VERSION = "dynamic-combination-decision-key-v1"
CATALOG_VERSION = "catalog-current-step-fixture-v1"


def _action(model_id: str) -> dict[str, object]:
    return {
        "task_family": "hotpotqa",
        "role_cluster": "retrieve",
        "model_id": model_id,
        "edge_type": "independent",
        "same_model_as_upstream": False,
        "stage": "other",
    }


def _current_probe(
    ordinal: int,
    *,
    split: str = "train",
) -> ProbeRecord:
    """Mirror the two persisted step-1 probe shapes without using artifacts."""

    incumbent = (
        "qwen3.5-flash" if ordinal == 0 else "qwen3.5-9b-local"
    )
    probe_id = f"probe:trajectory_current_{ordinal:02d}:0"
    branches = make_probe_branch_ids(probe_id, seed=ordinal + 31)
    return ProbeRecord(
        probe_id=probe_id,
        problem_id=f"hotpotqa:current-problem-{ordinal:02d}",
        task_split=split,
        snapshot_id=f"current-snapshot-{ordinal:02d}",
        policy_version=BEHAVIOR_POLICY,
        state_features={"is_audit": False},
        incumbent_action=_action(incumbent),
        candidate_action=_action("gpt-4o-mini"),
        sampling_probability=0.1,
        incumbent_returns=(1, 1, 1),
        candidate_returns=(1, 1, 1),
        executor_versions={
            incumbent: CATALOG_VERSION,
            "gpt-4o-mini": CATALOG_VERSION,
        },
        evaluator_version=EVALUATOR_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        branch_order=branches.execution_order,
    )


def _write_probe_file(path: Path, probes: tuple[ProbeRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(probe.to_dict(), sort_keys=True) + "\n" for probe in probes
        ),
        encoding="utf-8",
    )


def _canonical_probe(probe: ProbeRecord) -> dict[str, object]:
    """Compare persisted JSON values rather than tuple/list container details."""

    return json.loads(json.dumps(probe.to_dict(), sort_keys=True))


def _write_step(
    run_root: Path,
    *,
    status: str,
    probes: tuple[ProbeRecord, ...],
) -> None:
    step_root = run_root / "steps" / "step_000001"
    step_root.mkdir(parents=True, exist_ok=True)
    (step_root / "step_manifest.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    _write_probe_file(step_root / "probe_records.jsonl", probes)


def _publication_epoch() -> LedgerEpoch:
    versions = VersionBundle(
        policy=PUBLICATION_POLICY,
        model_catalog=CATALOG_VERSION,
        evaluator=EVALUATOR_VERSION,
        prompt="agentgraph.director.minimal.v2",
        tool="agentgraph.executor-react.v2",
        encoder="none",
        feature_schema=FEATURE_SCHEMA_VERSION,
    )
    ledger = CombinationPosterior(PUBLICATION_POLICY, epoch=1)
    for model_id in ("qwen3.5-flash", "qwen3.5-9b-local", "gpt-4o-mini"):
        ledger.register_key(DecisionKey.from_dict(_action(model_id)))
    return LedgerEpoch.freeze(ledger, (), versions)


def _config() -> dict[str, object]:
    return {
        "skills": {
            "delta_min": 0.05,
            "minimum_discovery_pairs": 10,
            "minimum_confirmation_problems": 20,
            "calibration_alpha": 0.05,
            "benjamini_hochberg_fdr": 0.10,
            "maximum_harm_probability": 0.05,
            "retire_after_suspended_epochs": 3,
        }
    }


def test_read_and_materialize_current_two_committed_train_probes(
    tmp_path: Path,
) -> None:
    probes = (_current_probe(0), _current_probe(1))
    run_root = tmp_path / "run"
    _write_step(run_root, status="committed", probes=probes)

    restored = RUNNER._read_probe_records(
        run_root / "steps" / "step_000001" / "probe_records.jsonl"
    )
    assert [_canonical_probe(probe) for probe in restored] == [
        _canonical_probe(probe) for probe in probes
    ]
    assert [probe.task_split for probe in restored] == ["train", "train"]
    assert [probe.paired_effect for probe in restored] == [0.0, 0.0]

    committed = RUNNER._committed_discovery_probes(
        run_root,
        completed_steps=1,
    )
    assert [_canonical_probe(probe) for probe in committed] == [
        _canonical_probe(probe) for probe in probes
    ]

    update, output_path = RUNNER._materialize_skill_runtime_update(
        _config(),
        run_root=run_root,
        ledger_epoch=_publication_epoch(),
        completed_steps=1,
    )
    assert update.skills == ()
    assert update.receipt.status == "complete_no_qualified_candidate"
    assert update.receipt.producer_complete is True
    assert update.receipt.discovery_probe_count == 2
    assert update.receipt.discovery_qualified_rule_count == 0
    assert update.receipt.grpo_reward_contribution == 0.0
    assert update.receipt.ttb_enabled is False

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["discovery_probe_ids"] == [
        probe.probe_id for probe in probes
    ]
    assert persisted["confirmation_probe_ids"] == []
    assert persisted["grpo_reward_contribution"] == 0.0
    assert persisted["ttb_enabled"] is False
    assert persisted["receipt"]["status"] == "complete_no_qualified_candidate"
    assert persisted["receipt"]["grpo_reward_contribution"] == 0.0
    assert persisted["receipt"]["ttb_enabled"] is False


@pytest.mark.parametrize(
    ("status", "split", "match"),
    (
        ("collecting", "train", "is not committed Skill discovery evidence"),
        ("committed", "validation", "Skill discovery evidence is not train-only"),
    ),
)
def test_committed_discovery_probes_fail_closed(
    tmp_path: Path,
    status: str,
    split: str,
    match: str,
) -> None:
    run_root = tmp_path / f"run-{status}-{split}"
    _write_step(
        run_root,
        status=status,
        probes=(_current_probe(0, split=split),),
    )

    with pytest.raises(RUNNER.HotpotTrainingError, match=match):
        RUNNER._committed_discovery_probes(run_root, completed_steps=1)
