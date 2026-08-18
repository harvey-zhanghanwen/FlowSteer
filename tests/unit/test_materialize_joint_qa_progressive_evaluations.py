from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from scripts.materialize_joint_qa_progressive_evaluations import (
    DATASETS,
    DEFAULT_SKILL_ON_TEMPLATES,
    DEFAULT_STEP1_TEMPLATES,
    materialize_skill_on_evaluations,
    materialize_step1_skill_off_evaluations,
)
from scripts.materialize_joint_qa_progressive_skill_training import (
    DEFAULT_TEMPLATE as DEFAULT_TRAINING_TEMPLATE,
    materialize as materialize_skill_training,
)
from src.interactive.skills import SkillEvidence, SkillRecord, SkillStatus, SkillStore
from src.interactive.versioning import VersionBundle


POLICY = "qwen35-9b-hotpot-step-000000"
UPDATED_POLICY = "qwen35-9b-jointqa-progressive-step-000001"
ADAPTER = "theta_jointqa_progressive_step_000001"
LIBRARY = "jointqa.skill-library.progressive.epoch2.v1"
POSTERIOR = "jointqa.bayesian-linear.progressive-subgraph.v1"


def _active_skill(dataset: str, *, activated_epoch: int = 2) -> SkillRecord:
    evidence_ids = tuple(f"{dataset}-evidence-{index}" for index in range(4))
    problem_ids = tuple(f"{dataset}-problem-{index}" for index in range(4))
    return SkillRecord(
        skill_id=f"jointqa.{dataset}.bounded-skill",
        version=1,
        status=SkillStatus.ACTIVE,
        condition={"task_family": dataset, "graph_stage": "*", "tags": []},
        action={"instruction": "Use the bounded evidence-gated instruction."},
        evidence=SkillEvidence(
            baseline="frozen_progressive_step0_no_skill",
            paired_effect_mean=0.10,
            calibrated_lower=0.04,
            calibrated_upper=0.16,
            effective_pairs=4,
            independent_problem_ids=problem_ids,
            discovery_problem_ids=(f"{dataset}-discovery",),
            validation_problem_ids=problem_ids,
            validation_splits=("skill_confirmation",),
            heldout_task_families=(dataset,),
            empirical_coverage=1.0,
            harm_probability=0.01,
            slice_effects={dataset: 0.10},
            evidence_ids=evidence_ids,
        ),
        versions=VersionBundle(
            policy=POLICY,
            model_catalog="catalog-v1",
            evaluator=f"{dataset}.official.answer.v1",
            prompt="agentgraph.director.progressive_subgraph.v1",
            tool="agentgraph.add-subgraph+skillflow-public-retrieval.v1",
            encoder="jointqa.skill-condition.fixed.v1",
            feature_schema="jointqa.skill-candidate-dataset-interaction.v1",
            posterior=POSTERIOR,
            skill_library=LIBRARY,
        ),
        created_epoch=0,
        eligible_epoch=1,
        activated_epoch=activated_epoch,
        gate_config={"delta_min": 0.03},
        gate_receipt=f"gate-{dataset}",
    )


def _write_publication_and_store(root: Path) -> tuple[Path, Path, dict[str, SkillRecord]]:
    store_path = root / "skills.json"
    store = SkillStore(store_path)
    records = {dataset: _active_skill(dataset) for dataset in DATASETS}
    for record in records.values():
        store.upsert(record)
    publication = {
        "schema_version": "flowsteer.joint-qa.skill-publication-result.v1",
        "active_datasets": list(DATASETS),
        "publications": {
            dataset: {
                "skill": record.to_dict(),
                "gate": {"approved": True},
            }
            for dataset, record in records.items()
        },
    }
    publication_path = root / "publication_results.json"
    publication_path.write_text(json.dumps(publication), encoding="utf-8")
    return publication_path, store_path, records


def _output_paths(root: Path, phase: str) -> dict[str, Path]:
    return {dataset: root / phase / f"{dataset}.yaml" for dataset in DATASETS}


def _load_outputs(paths: dict[str, Path]) -> dict[str, dict]:
    return {
        dataset: yaml.safe_load(paths[dataset].read_text(encoding="utf-8"))
        for dataset in DATASETS
    }


def _assert_no_placeholder(value: object) -> None:
    assert "__REPLACE_" not in yaml.safe_dump(value, allow_unicode=True)


def test_skill_on_materialization_binds_two_active_skills_without_route_drift() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        publication, store, records = _write_publication_and_store(root)
        outputs = _output_paths(root, "skill-on")
        before = {
            dataset: yaml.safe_load(DEFAULT_SKILL_ON_TEMPLATES[dataset].read_text())
            for dataset in DATASETS
        }

        receipt = materialize_skill_on_evaluations(
            publication_path=publication,
            skill_store_path=store,
            output_paths=outputs,
        )
        resolved = _load_outputs(outputs)

        assert receipt["status"] == "materialized"
        assert receipt["evaluation_started"] is False
        assert receipt["policy_version"] == POLICY
        for dataset in DATASETS:
            _assert_no_placeholder(resolved[dataset])
            assert resolved[dataset]["skills"]["required_skill_ids"] == [
                records[dataset].skill_id
            ]
            assert resolved[dataset]["skills"]["library_version"] == LIBRARY
            assert resolved[dataset]["skills"]["posterior_version"] == POSTERIOR
            assert resolved[dataset]["exploration"]["posterior_version"] == POSTERIOR
            assert resolved[dataset]["experiment"]["seed"] == before[dataset]["experiment"]["seed"]
            assert (
                resolved[dataset]["agent_graph"]["model_catalog_path"]
                == before[dataset]["agent_graph"]["model_catalog_path"]
            )
            for field in (
                "behavior_policy_version",
                "behavior_adapter_name",
                "behavior_adapter_checkpoint",
                "expected_server_weight_version",
            ):
                assert resolved[dataset]["director"][field] == before[dataset]["director"][field]

        with pytest.raises(FileExistsError, match="write-once"):
            materialize_skill_on_evaluations(
                publication_path=publication,
                skill_store_path=store,
                output_paths=outputs,
            )


def test_skill_on_fails_closed_unless_both_store_records_are_active() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        publication, store_path, records = _write_publication_and_store(root)
        raw = json.loads(store_path.read_text(encoding="utf-8"))
        trivia = records["triviaqa"].to_dict()
        trivia["status"] = "suspended"
        trivia["suspended_reason"] = "policy drift"
        raw["current"][records["triviaqa"].skill_id] = trivia
        store_path.write_text(json.dumps(raw), encoding="utf-8")
        outputs = _output_paths(root, "skill-on")

        with pytest.raises(RuntimeError, match="matching ACTIVE Skill"):
            materialize_skill_on_evaluations(
                publication_path=publication,
                skill_store_path=store_path,
                output_paths=outputs,
            )
        assert not any(path.exists() for path in outputs.values())


def test_skill_on_uses_the_common_delayed_activation_epoch() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        store_path = root / "skills.json"
        store = SkillStore(store_path)
        records = {
            dataset: _active_skill(dataset, activated_epoch=4)
            for dataset in DATASETS
        }
        for record in records.values():
            store.upsert(record)
        publication_path = root / "publication_results.json"
        publication_path.write_text(
            json.dumps(
                {
                    "active_datasets": list(DATASETS),
                    "publications": {
                        dataset: {"skill": record.to_dict(), "gate": {"approved": True}}
                        for dataset, record in records.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        outputs = _output_paths(root, "skill-on-epoch4")

        receipt = materialize_skill_on_evaluations(
            publication_path=publication_path,
            skill_store_path=store_path,
            output_paths=outputs,
        )
        resolved = _load_outputs(outputs)

        assert receipt["current_epoch"] == 4
        assert all(config["skills"]["current_epoch"] == 4 for config in resolved.values())


def test_skill_training_uses_the_common_delayed_activation_epoch() -> None:
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
        root = Path(directory)
        store_path = root / "skills.json"
        store = SkillStore(store_path)
        records = {
            dataset: _active_skill(dataset, activated_epoch=4)
            for dataset in DATASETS
        }
        for record in records.values():
            store.upsert(record)
        publication_path = root / "publication_results.json"
        publication_path.write_text(
            json.dumps(
                {
                    "active_datasets": list(DATASETS),
                    "publications": {
                        dataset: {"skill": record.to_dict(), "gate": {"approved": True}}
                        for dataset, record in records.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        template = yaml.safe_load(DEFAULT_TRAINING_TEMPLATE.read_text(encoding="utf-8"))
        template["data"]["joint_qa_micro"]["schedule_path"] = str(root / "schedule.json")
        template["data"]["joint_qa_micro"]["cursor_path"] = str(root / "cursor.json")
        template_path = root / "training.yaml"
        template_path.write_text(
            yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        output_path = root / "resolved.yaml"

        receipt = materialize_skill_training(
            template_path=template_path,
            publication_path=publication_path,
            skill_store_path=store_path,
            output_path=output_path,
        )
        resolved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        assert receipt["current_epoch"] == 4
        assert resolved["skills"]["current_epoch"] == 4
        schedule = json.loads(
            Path(receipt["schedule"]["schedule"]).read_text(encoding="utf-8")
        )
        selected = {
            task["dataset_key"]: task for task in schedule["steps"][0]["tasks"]
        }
        assert selected["hotpotqa"]["task_position"] == 9
        assert selected["triviaqa"]["task_position"] == 12


def _write_training_receipts(root: Path) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoint" / "theta"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"lora")
    (checkpoint / "policy_version.json").write_text(
        json.dumps(
            {
                "behavior_policy_version": POLICY,
                "updated_policy_version": UPDATED_POLICY,
                "optimizer_updates": 1,
                "trainable_update_l2": 0.025,
            }
        ),
        encoding="utf-8",
    )
    sync = {
        "status": "published",
        "success": True,
        "training_performed": True,
        "policy_published": True,
        "canary_succeeded": True,
        "route_switch_requested": True,
        "route_switch_succeeded": True,
        "behavior_policy_version": POLICY,
        "candidate_policy_version": UPDATED_POLICY,
        "new_policy_version": UPDATED_POLICY,
        "adapter_name": ADAPTER,
        "checkpoint_version": f"checkpoint:{UPDATED_POLICY}",
        "checkpoint_path": str(checkpoint),
        "models_before": ["supervisor_theta"],
        "models_after": ["supervisor_theta", ADAPTER],
    }
    manifest = {
        "schema_version": "flowsteer.agentgraph.smoke_manifest.v1",
        "status": "completed",
        "training": {
            "optimizer_updates": 1,
            "trainable_update_l2": 0.025,
            "updated_policy_version": UPDATED_POLICY,
            "checkpoint_dir": str(checkpoint),
        },
        "policy_sync": dict(sync),
        "post_update_canaries": {
            "collected": 2,
            "adapter_name": ADAPTER,
            "policy_version": UPDATED_POLICY,
            "trajectory_ids": ["canary-hotpot", "canary-trivia"],
        },
    }
    sync_path = root / "sync_receipt.json"
    manifest_path = root / "training_manifest.json"
    sync_path.write_text(json.dumps(sync), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, sync_path, checkpoint


def test_step1_materialization_binds_real_update_adapter_and_default_server_weight() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, sync, checkpoint = _write_training_receipts(root)
        outputs = _output_paths(root, "step1")

        receipt = materialize_step1_skill_off_evaluations(
            training_manifest_path=manifest,
            sync_receipt_path=sync,
            output_paths=outputs,
        )
        resolved = _load_outputs(outputs)

        assert receipt["optimizer_updates"] == 1
        assert receipt["server_weight_version"] == "default"
        assert receipt["server_weight_source"] == "sglang_actual_default_receipt"
        assert receipt["checkpoint"] == str(checkpoint.resolve())
        for dataset in DATASETS:
            _assert_no_placeholder(resolved[dataset])
            director = resolved[dataset]["director"]
            assert director["behavior_policy_version"] == UPDATED_POLICY
            assert director["behavior_adapter_name"] == ADAPTER
            assert director["behavior_adapter_checkpoint"] == str(checkpoint.resolve())
            assert director["expected_server_weight_version"] == "default"
            assert resolved[dataset]["skills"]["enabled"] is False
            assert resolved[dataset]["skills"]["retrieval_top_k"] == 0

        hotpot_identity = (
            resolved["hotpotqa"]["experiment"]["seed"],
            resolved["hotpotqa"]["agent_graph"]["model_catalog_path"],
        )
        trivia_identity = (
            resolved["triviaqa"]["experiment"]["seed"],
            resolved["triviaqa"]["agent_graph"]["model_catalog_path"],
        )
        assert hotpot_identity == trivia_identity


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest, sync: manifest["training"].update(optimizer_updates=0), "optimizer update"),
        (lambda manifest, sync: sync.update(success=False), "successful trained adapter"),
        (lambda manifest, sync: sync.update(new_policy_version="wrong"), "consistent new policy"),
    ],
)
def test_step1_fails_closed_on_incomplete_training_or_sync(mutation, message: str) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, sync_path, _ = _write_training_receipts(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        mutation(manifest, sync)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        sync_path.write_text(json.dumps(sync), encoding="utf-8")
        outputs = _output_paths(root, "step1")

        with pytest.raises(RuntimeError, match=message):
            materialize_step1_skill_off_evaluations(
                training_manifest_path=manifest_path,
                sync_receipt_path=sync_path,
                output_paths=outputs,
            )
        assert not any(path.exists() for path in outputs.values())


def test_step1_checks_checkpoint_policy_metadata_before_writing() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, sync, checkpoint = _write_training_receipts(root)
        metadata = json.loads((checkpoint / "policy_version.json").read_text())
        metadata["optimizer_updates"] = 0
        (checkpoint / "policy_version.json").write_text(json.dumps(metadata))
        outputs = _output_paths(root, "step1")

        with pytest.raises(RuntimeError, match="checkpoint metadata"):
            materialize_step1_skill_off_evaluations(
                training_manifest_path=manifest,
                sync_receipt_path=sync,
                output_paths=outputs,
            )
        assert not any(path.exists() for path in outputs.values())
