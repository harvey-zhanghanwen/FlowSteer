from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from scripts.evaluate_hotpotqa_round import validate_hotpot_config
from scripts.evaluate_triviaqa_round import (
    _report as trivia_report,
    _report_markdown as trivia_report_markdown,
    validate_trivia_config,
)
from scripts.materialize_joint_qa_progressive_evaluations import (
    DEFAULT_SKILL_ON_TEMPLATES,
    FIXED_TEST_PATH,
    _apply_evaluation_scope,
    _evaluation_identity,
    materialize_step1_skill_off_evaluations,
)
from src.interactive.config_loader import load_yaml


POLICY = "qwen35-9b-jointqa-progressive-step-000001"
ADAPTER = "theta_jointqa_progressive_step_000001"


def _write_step1_receipts(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "checkpoint" / "theta"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"lora")
    (checkpoint / "policy_version.json").write_text(
        json.dumps(
            {
                "updated_policy_version": POLICY,
                "optimizer_updates": 1,
                "trainable_update_l2": 0.025,
            }
        ),
        encoding="utf-8",
    )
    sync = {
        "success": True,
        "training_performed": True,
        "policy_published": True,
        "canary_succeeded": True,
        "route_switch_requested": True,
        "route_switch_succeeded": True,
        "candidate_policy_version": POLICY,
        "new_policy_version": POLICY,
        "adapter_name": ADAPTER,
        "checkpoint_version": f"checkpoint:{POLICY}",
        "checkpoint_path": str(checkpoint),
        "models_after": ["supervisor_theta", ADAPTER],
    }
    manifest = {
        "status": "completed",
        "training": {
            "optimizer_updates": 1,
            "trainable_update_l2": 0.025,
            "updated_policy_version": POLICY,
            "checkpoint_dir": str(checkpoint),
        },
        "policy_sync": dict(sync),
        "post_update_canaries": {
            "collected": 2,
            "adapter_name": ADAPTER,
            "policy_version": POLICY,
        },
    }
    manifest_path = root / "training_manifest.json"
    sync_path = root / "sync_receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sync_path.write_text(json.dumps(sync), encoding="utf-8")
    return manifest_path, sync_path


@pytest.mark.parametrize(
    ("dataset", "validator"),
    (("hotpotqa", validate_hotpot_config), ("triviaqa", validate_trivia_config)),
)
def test_fixed_test_scope_reuses_evaluator_config_with_isolated_artifacts(
    dataset, validator
) -> None:
    template = load_yaml(DEFAULT_SKILL_ON_TEMPLATES[dataset])

    resolved = _apply_evaluation_scope(
        template,
        dataset=dataset,
        mode="step_000000_skill_on",
        evaluation_split="test",
        sample_count=128,
    )

    bounded = resolved[f"{dataset}_evaluation"]
    assert bounded["split"] == "test"
    assert bounded["sample_count"] == 128
    assert resolved["data"]["test_path"] == FIXED_TEST_PATH
    assert resolved["experiment"]["training_enabled"] is False
    assert (
        resolved["experiment"]["sampling_schedule_purpose"]
        == "joint_qa_progressive_fixed_test"
    )
    assert resolved["grpo"]["enabled"] is False
    assert resolved["grpo"]["max_optimizer_updates"] == 0
    assert resolved["exploration"]["enabled"] is False
    assert resolved["policy_sync"]["enabled"] is False
    assert resolved["gpu"]["training_enabled"] is False
    assert "/fixed_test/" in resolved["storage"]["selected_tasks_path"]
    assert "/fixed_test/" in resolved["storage"]["direct_predictions_path"]
    assert _evaluation_identity(resolved) == _evaluation_identity(template)
    validator(resolved)


def test_validation_scope_is_backward_compatible() -> None:
    template = load_yaml(DEFAULT_SKILL_ON_TEMPLATES["hotpotqa"])

    resolved = _apply_evaluation_scope(
        template,
        dataset="hotpotqa",
        mode="step_000000_skill_on",
        evaluation_split="validation",
        sample_count=None,
    )

    assert resolved == template
    assert resolved is not template


def test_step1_public_materializer_emits_formal_fixed_test_configs() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, sync = _write_step1_receipts(root)
        outputs = {
            dataset: root / "resolved" / f"{dataset}.yaml"
            for dataset in ("hotpotqa", "triviaqa")
        }

        receipt = materialize_step1_skill_off_evaluations(
            training_manifest_path=manifest,
            sync_receipt_path=sync,
            output_paths=outputs,
            evaluation_split="test",
            sample_count=128,
        )

        assert receipt["evaluation_split"] == "test"
        assert receipt["sample_count"] == 128
        assert receipt["formal_full_test"] is True
        assert receipt["test_excluded_from_training_and_skill_evidence"] is True
        for dataset, output in outputs.items():
            config = yaml.safe_load(output.read_text(encoding="utf-8"))
            assert config[f"{dataset}_evaluation"]["split"] == "test"
            assert config[f"{dataset}_evaluation"]["sample_count"] == 128
            assert config["data"]["test_path"] == FIXED_TEST_PATH
            assert "/fixed_test/" in config["storage"]["manifest_path"]


@pytest.mark.parametrize("sample_count", (None, 0, 129, True))
def test_fixed_test_requires_an_explicit_bounded_sample_count(sample_count) -> None:
    template = load_yaml(DEFAULT_SKILL_ON_TEMPLATES["hotpotqa"])

    with pytest.raises(ValueError, match="sample_count"):
        _apply_evaluation_scope(
            template,
            dataset="hotpotqa",
            mode="step_000000_skill_on",
            evaluation_split="test",
            sample_count=sample_count,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda config: config["experiment"].update(training_enabled=True),
        lambda config: config["grpo"].update(enabled=True),
        lambda config: config["grpo"].update(max_optimizer_updates=1),
        lambda config: config["exploration"].update(enabled=True),
        lambda config: config["exploration"].update(forced_probe_rollouts=1),
        lambda config: config["policy_sync"].update(enabled=True),
        lambda config: config["gpu"].update(training_enabled=True),
        lambda config: config["data"].update(test_path="data/joint_qa_v2/train.jsonl"),
    ),
)
def test_fixed_test_fails_closed_on_learning_or_non_test_inputs(mutation) -> None:
    template = deepcopy(load_yaml(DEFAULT_SKILL_ON_TEMPLATES["hotpotqa"]))
    mutation(template)

    with pytest.raises(RuntimeError, match="fixed test is not evaluation-only"):
        _apply_evaluation_scope(
            template,
            dataset="hotpotqa",
            mode="step_000000_skill_on",
            evaluation_split="test",
            sample_count=128,
        )


def test_trivia_report_and_markdown_use_the_configured_test_split() -> None:
    config = load_yaml(DEFAULT_SKILL_ON_TEMPLATES["triviaqa"])
    config = _apply_evaluation_scope(
        config,
        dataset="triviaqa",
        mode="step_000000_skill_on",
        evaluation_split="test",
        sample_count=128,
    )

    report = trivia_report((), config, ())
    markdown = trivia_report_markdown(report)

    assert report["project_split"] == "test"
    assert "固定项目 held-out test" in markdown
    assert "固定项目 validation" not in markdown
