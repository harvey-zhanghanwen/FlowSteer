#!/usr/bin/env python3
"""Materialize fail-closed joint-QA progressive evaluation configs.

The two materialization boundaries deliberately remain read-only with respect
to experiment evidence:

* Step 0 Skill-on requires matching evidence-gated ``ACTIVE`` Skills in both
  the publication result and the frozen :class:`SkillStore`.
* Step 1 Skill-off requires one completed optimizer update, a successful SGLang
  adapter publication, a matching post-update canary, and a locally verifiable
  LoRA checkpoint.

This script only binds existing receipts into the four checked-in evaluation
templates.  It does not call a model/API, collect a rollout, train, publish an
adapter, or mutate any input receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.skills import SkillStatus, SkillStore


DATASETS = ("hotpotqa", "triviaqa")
DEFAULT_PUBLICATION = (
    PROJECT_ROOT
    / "artifacts/joint_qa_progressive/skill_epoch_000000/publication_results.json"
)
DEFAULT_SKILL_STORE = (
    PROJECT_ROOT / "artifacts/joint_qa_progressive/skill_epoch_000000/skills.json"
)
DEFAULT_TRAINING_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/joint_qa_progressive/training/step_000001/training_manifest.json"
)
DEFAULT_SYNC_RECEIPT = (
    PROJECT_ROOT
    / "artifacts/joint_qa_progressive/training/step_000001/sync_receipt.json"
)
DEFAULT_SKILL_ON_TEMPLATES = {
    dataset: PROJECT_ROOT
    / f"config/evaluation_joint_qa_progressive_step0_skill_on_{dataset}.yaml"
    for dataset in DATASETS
}
DEFAULT_STEP1_TEMPLATES = {
    dataset: PROJECT_ROOT
    / f"config/evaluation_joint_qa_progressive_step1_skill_off_{dataset}.yaml"
    for dataset in DATASETS
}
DEFAULT_SKILL_ON_OUTPUTS = {
    dataset: PROJECT_ROOT
    / f"artifacts/joint_qa_progressive/evaluation_configs/step_000000_skill_on/{dataset}.yaml"
    for dataset in DATASETS
}
DEFAULT_STEP1_OUTPUTS = {
    dataset: PROJECT_ROOT
    / f"artifacts/joint_qa_progressive/evaluation_configs/step_000001_skill_off/{dataset}.yaml"
    for dataset in DATASETS
}


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _json_mapping(path: Path, field: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _paths(
    supplied: Mapping[str, Path] | None,
    defaults: Mapping[str, Path],
    field: str,
) -> dict[str, Path]:
    values = dict(defaults if supplied is None else supplied)
    if set(values) != set(DATASETS):
        raise ValueError(f"{field} must contain exactly {DATASETS}")
    return {dataset: Path(values[dataset]).resolve() for dataset in DATASETS}


def _config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _placeholder_locations(value: object, field: str = "config") -> list[str]:
    locations: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            locations.extend(_placeholder_locations(item, f"{field}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            locations.extend(_placeholder_locations(item, f"{field}[{index}]"))
    elif isinstance(value, str) and "__REPLACE_" in value:
        locations.append(field)
    return locations


def _require_no_placeholders(config: Mapping[str, Any], dataset: str) -> None:
    locations = _placeholder_locations(config)
    if locations:
        raise RuntimeError(
            f"{dataset} resolved config retains fail-closed placeholders at "
            + ", ".join(locations)
        )


def _ensure_outputs_absent(outputs: Mapping[str, Path]) -> None:
    existing = [str(outputs[dataset]) for dataset in DATASETS if outputs[dataset].exists()]
    if existing:
        raise FileExistsError(
            "write-once resolved evaluation config already exists: " + ", ".join(existing)
        )


def _write_once(configs: Mapping[str, Mapping[str, Any]], outputs: Mapping[str, Path]) -> None:
    _ensure_outputs_absent(outputs)
    rendered = {
        dataset: yaml.safe_dump(
            dict(configs[dataset]), sort_keys=False, allow_unicode=True
        )
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        output = outputs[dataset]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered[dataset])


def _evaluation_identity(config: Mapping[str, Any]) -> tuple[object, ...]:
    experiment = _mapping(config.get("experiment"), "experiment")
    director = _mapping(config.get("director"), "director")
    agent_graph = _mapping(config.get("agent_graph"), "agent_graph")
    return (
        experiment.get("seed"),
        director.get("behavior_policy_version"),
        director.get("behavior_adapter_name"),
        director.get("behavior_adapter_checkpoint"),
        director.get("expected_server_weight_version"),
        agent_graph.get("model_catalog_path"),
    )


def materialize_skill_on_evaluations(
    *,
    publication_path: Path = DEFAULT_PUBLICATION,
    skill_store_path: Path = DEFAULT_SKILL_STORE,
    template_paths: Mapping[str, Path] | None = None,
    output_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Bind the two evidence-gated ACTIVE Skills into matched Step 0 configs."""

    publication_path = Path(publication_path).resolve()
    skill_store_path = Path(skill_store_path).resolve()
    templates = _paths(template_paths, DEFAULT_SKILL_ON_TEMPLATES, "template_paths")
    outputs = _paths(output_paths, DEFAULT_SKILL_ON_OUTPUTS, "output_paths")
    _ensure_outputs_absent(outputs)

    publication = _json_mapping(publication_path, "publication")
    publications = _mapping(publication.get("publications"), "publications")
    if set(publication.get("active_datasets", ())) != set(DATASETS):
        raise RuntimeError(
            "Skill-on evaluation requires evidence-gated ACTIVE Skills for both datasets"
        )
    if not skill_store_path.is_file():
        raise FileNotFoundError(f"SkillStore does not exist: {skill_store_path}")
    store = SkillStore(skill_store_path)

    records = {}
    for dataset in DATASETS:
        payload = _mapping(publications.get(dataset), f"publications.{dataset}")
        skill_payload = _mapping(payload.get("skill"), f"publications.{dataset}.skill")
        if skill_payload.get("status") != SkillStatus.ACTIVE.value:
            raise RuntimeError(f"{dataset} publication is not ACTIVE")
        skill_id = str(skill_payload.get("skill_id", "")).strip()
        record = store.get(skill_id)
        if record is None or record.status is not SkillStatus.ACTIVE:
            raise RuntimeError(f"{dataset} publication has no matching ACTIVE Skill")
        if record.condition.get("task_family") != dataset:
            raise RuntimeError(f"{dataset} ACTIVE Skill has an incompatible task condition")
        if record.version != skill_payload.get("version"):
            raise RuntimeError(f"{dataset} SkillStore/publication version mismatch")
        if record.activated_epoch is None:
            raise RuntimeError(f"{dataset} ACTIVE Skill has no activation epoch")
        records[dataset] = record

    library_versions = {record.versions.skill_library for record in records.values()}
    posterior_versions = {record.versions.posterior for record in records.values()}
    policy_versions = {record.versions.policy for record in records.values()}
    if len(library_versions) != 1 or len(posterior_versions) != 1:
        raise RuntimeError("ACTIVE Skills do not share one frozen library/posterior regime")
    if len(policy_versions) != 1:
        raise RuntimeError("ACTIVE Skills do not share one frozen behavior policy")
    library_version = next(iter(library_versions))
    posterior_version = next(iter(posterior_versions))
    policy_version = next(iter(policy_versions))

    configs: dict[str, dict[str, Any]] = {}
    identities: dict[str, tuple[object, ...]] = {}
    for dataset in DATASETS:
        template = load_yaml(templates[dataset])
        identity = _evaluation_identity(template)
        config = deepcopy(template)
        director = _mapping(config.get("director"), "director")
        if director.get("behavior_policy_version") != policy_version:
            raise RuntimeError(
                f"{dataset} template policy differs from ACTIVE Skill evidence"
            )
        experiment = _mapping(config.get("experiment"), "experiment")
        record = records[dataset]
        if experiment.get("prompt_version") != record.versions.prompt:
            raise RuntimeError(f"{dataset} template prompt version differs from Skill")
        if experiment.get("tool_version") != record.versions.tool:
            raise RuntimeError(f"{dataset} template tool version differs from Skill")

        skills = dict(_mapping(config.get("skills"), "skills"))
        current_epoch = int(skills.get("current_epoch", 0))
        if record.activated_epoch is None or record.activated_epoch > current_epoch:
            raise RuntimeError(f"{dataset} ACTIVE Skill is not visible in this epoch")
        skills.update(
            store_path=_config_path(skill_store_path),
            library_version=library_version,
            posterior_version=posterior_version,
            required_skill_ids=[record.skill_id],
        )
        config["skills"] = skills
        exploration = dict(_mapping(config.get("exploration"), "exploration"))
        exploration["posterior_version"] = posterior_version
        config["exploration"] = exploration
        if _evaluation_identity(config) != identity:
            raise RuntimeError(
                f"{dataset} Skill materialization changed policy/adapter/catalog/seed"
            )
        _require_no_placeholders(config, dataset)
        configs[dataset] = config
        identities[dataset] = identity

    if identities["hotpotqa"] != identities["triviaqa"]:
        raise RuntimeError(
            "Step 0 Skill-on dataset configs do not share one policy/adapter/catalog/seed"
        )

    _write_once(configs, outputs)
    return {
        "status": "materialized",
        "mode": "step0_skill_on",
        "publication": str(publication_path),
        "skill_store": str(skill_store_path),
        "policy_version": policy_version,
        "library_version": library_version,
        "posterior_version": posterior_version,
        "required_skill_ids": {
            dataset: records[dataset].skill_id for dataset in DATASETS
        },
        "outputs": {dataset: str(outputs[dataset]) for dataset in DATASETS},
        "evaluation_started": False,
    }


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be a positive number")
    number = float(value)
    if number <= 0:
        raise RuntimeError(f"{field} must be a positive number")
    return number


def _resolve_checkpoint(
    training: Mapping[str, Any], sync: Mapping[str, Any]
) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = Path(str(training.get("checkpoint_dir", ""))).expanduser().resolve()
    sync_path = Path(str(sync.get("checkpoint_path", ""))).expanduser().resolve()
    if manifest_path != sync_path:
        raise RuntimeError("training manifest and sync receipt checkpoint paths differ")
    checkpoint = sync_path
    if not checkpoint.is_dir():
        raise RuntimeError(f"published LoRA checkpoint is not a directory: {checkpoint}")
    for name in ("adapter_config.json", "adapter_model.safetensors", "policy_version.json"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"published LoRA checkpoint is missing {name}")
    metadata = _json_mapping(checkpoint / "policy_version.json", "policy metadata")
    return checkpoint, metadata


def _server_weight_version(
    manifest: Mapping[str, Any], sync: Mapping[str, Any]
) -> tuple[str, str]:
    canaries = _mapping(manifest.get("post_update_canaries"), "post_update_canaries")
    candidates = (
        (sync.get("server_weight_version"), "sync_receipt.server_weight_version"),
        (sync.get("actual_server_weight_version"), "sync_receipt.actual_server_weight_version"),
        (canaries.get("server_weight_version"), "post_update_canaries.server_weight_version"),
    )
    for value, source in candidates:
        if value is not None:
            text = str(value).strip()
            if not text:
                raise RuntimeError(f"{source} is empty")
            return text, source
    # The current SGLang OpenAI-compatible trajectory receipt reports the
    # backend weight coordinate as the literal ``default`` while the logical
    # policy and loaded LoRA are carried separately.  Preserve that actual
    # runtime contract explicitly instead of leaving the template unresolved.
    return "default", "sglang_actual_default_receipt"


def materialize_step1_skill_off_evaluations(
    *,
    training_manifest_path: Path = DEFAULT_TRAINING_MANIFEST,
    sync_receipt_path: Path = DEFAULT_SYNC_RECEIPT,
    template_paths: Mapping[str, Path] | None = None,
    output_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Bind one real LoRA update and successful SGLang sync into Step 1."""

    training_manifest_path = Path(training_manifest_path).resolve()
    sync_receipt_path = Path(sync_receipt_path).resolve()
    templates = _paths(template_paths, DEFAULT_STEP1_TEMPLATES, "template_paths")
    outputs = _paths(output_paths, DEFAULT_STEP1_OUTPUTS, "output_paths")
    _ensure_outputs_absent(outputs)

    manifest = _json_mapping(training_manifest_path, "training_manifest")
    sync = _json_mapping(sync_receipt_path, "sync_receipt")
    if manifest.get("status") != "completed":
        raise RuntimeError("training manifest is not completed")
    training = _mapping(manifest.get("training"), "training_manifest.training")
    if training.get("optimizer_updates") != 1:
        raise RuntimeError("Step 1 evaluation requires exactly one optimizer update")
    _positive_number(training.get("trainable_update_l2"), "training.trainable_update_l2")

    required_sync_flags = (
        "success",
        "training_performed",
        "policy_published",
        "canary_succeeded",
    )
    if any(sync.get(field) is not True for field in required_sync_flags):
        raise RuntimeError("sync receipt is not a successful trained adapter publication")
    if sync.get("route_switch_requested") is True and sync.get("route_switch_succeeded") is not True:
        raise RuntimeError("sync receipt did not complete its requested route switch")

    policy_version = str(sync.get("new_policy_version", "")).strip()
    candidate_policy = str(sync.get("candidate_policy_version", "")).strip()
    adapter_name = str(sync.get("adapter_name", "")).strip()
    checkpoint_version = str(sync.get("checkpoint_version", "")).strip()
    if not policy_version or policy_version != candidate_policy:
        raise RuntimeError("sync receipt has no consistent new policy version")
    if not adapter_name or not checkpoint_version:
        raise RuntimeError("sync receipt has no adapter/checkpoint version")
    models_after = sync.get("models_after")
    if not isinstance(models_after, (list, tuple)) or adapter_name not in models_after:
        raise RuntimeError("published adapter is absent from sync receipt models_after")
    if training.get("updated_policy_version") != policy_version:
        raise RuntimeError("training manifest and sync receipt policy versions differ")

    manifest_sync = _mapping(manifest.get("policy_sync"), "training_manifest.policy_sync")
    for field, expected in (
        ("success", True),
        ("adapter_name", adapter_name),
        ("new_policy_version", policy_version),
        ("checkpoint_path", sync.get("checkpoint_path")),
    ):
        if manifest_sync.get(field) != expected:
            raise RuntimeError(f"training manifest policy_sync differs at {field}")

    canaries = _mapping(manifest.get("post_update_canaries"), "post_update_canaries")
    if not isinstance(canaries.get("collected"), int) or int(canaries["collected"]) < 1:
        raise RuntimeError("training manifest has no post-update canary")
    if canaries.get("adapter_name") != adapter_name:
        raise RuntimeError("post-update canary adapter differs from sync receipt")
    if canaries.get("policy_version") != policy_version:
        raise RuntimeError("post-update canary policy differs from sync receipt")

    checkpoint, metadata = _resolve_checkpoint(training, sync)
    if metadata.get("updated_policy_version") != policy_version:
        raise RuntimeError("checkpoint metadata has the wrong updated policy version")
    if metadata.get("optimizer_updates") != 1:
        raise RuntimeError("checkpoint metadata does not record exactly one optimizer update")
    _positive_number(metadata.get("trainable_update_l2"), "policy metadata.trainable_update_l2")
    server_weight_version, server_weight_source = _server_weight_version(manifest, sync)

    configs: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        config = deepcopy(load_yaml(templates[dataset]))
        director = dict(_mapping(config.get("director"), "director"))
        director.update(
            behavior_policy_version=policy_version,
            behavior_adapter_name=adapter_name,
            behavior_adapter_checkpoint=_config_path(checkpoint),
            expected_server_weight_version=server_weight_version,
        )
        config["director"] = director
        skills = _mapping(config.get("skills"), "skills")
        if skills.get("enabled") is not False or skills.get("retrieval_top_k") != 0:
            raise RuntimeError(f"{dataset} Step 1 Skill-off template enables Skill retrieval")
        _require_no_placeholders(config, dataset)
        configs[dataset] = config

    identity = {
        dataset: (
            configs[dataset]["experiment"]["seed"],
            configs[dataset]["agent_graph"]["model_catalog_path"],
            configs[dataset]["director"]["behavior_policy_version"],
            configs[dataset]["director"]["behavior_adapter_name"],
            configs[dataset]["director"]["behavior_adapter_checkpoint"],
            configs[dataset]["director"]["expected_server_weight_version"],
        )
        for dataset in DATASETS
    }
    if identity["hotpotqa"] != identity["triviaqa"]:
        raise RuntimeError("Step 1 dataset configs do not share one policy/adapter/catalog/seed")
    _write_once(configs, outputs)
    return {
        "status": "materialized",
        "mode": "step1_skill_off",
        "training_manifest": str(training_manifest_path),
        "sync_receipt": str(sync_receipt_path),
        "policy_version": policy_version,
        "adapter_name": adapter_name,
        "checkpoint_version": checkpoint_version,
        "checkpoint": str(checkpoint),
        "server_weight_version": server_weight_version,
        "server_weight_source": server_weight_source,
        "optimizer_updates": 1,
        "outputs": {dataset: str(outputs[dataset]) for dataset in DATASETS},
        "evaluation_started": False,
    }


def _dataset_paths(hotpotqa: str, triviaqa: str) -> dict[str, Path]:
    return {"hotpotqa": Path(hotpotqa), "triviaqa": Path(triviaqa)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    skill = subparsers.add_parser("skill-on", help="materialize matched Step 0 Skill-on configs")
    skill.add_argument("--publication", default=str(DEFAULT_PUBLICATION))
    skill.add_argument("--skill-store", default=str(DEFAULT_SKILL_STORE))
    skill.add_argument(
        "--hotpot-template", default=str(DEFAULT_SKILL_ON_TEMPLATES["hotpotqa"])
    )
    skill.add_argument(
        "--trivia-template", default=str(DEFAULT_SKILL_ON_TEMPLATES["triviaqa"])
    )
    skill.add_argument(
        "--hotpot-output", default=str(DEFAULT_SKILL_ON_OUTPUTS["hotpotqa"])
    )
    skill.add_argument(
        "--trivia-output", default=str(DEFAULT_SKILL_ON_OUTPUTS["triviaqa"])
    )

    step1 = subparsers.add_parser(
        "step1-skill-off", help="materialize matched Step 1 Skill-off configs"
    )
    step1.add_argument("--training-manifest", default=str(DEFAULT_TRAINING_MANIFEST))
    step1.add_argument("--sync-receipt", default=str(DEFAULT_SYNC_RECEIPT))
    step1.add_argument(
        "--hotpot-template", default=str(DEFAULT_STEP1_TEMPLATES["hotpotqa"])
    )
    step1.add_argument(
        "--trivia-template", default=str(DEFAULT_STEP1_TEMPLATES["triviaqa"])
    )
    step1.add_argument("--hotpot-output", default=str(DEFAULT_STEP1_OUTPUTS["hotpotqa"]))
    step1.add_argument("--trivia-output", default=str(DEFAULT_STEP1_OUTPUTS["triviaqa"]))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    templates = _dataset_paths(args.hotpot_template, args.trivia_template)
    outputs = _dataset_paths(args.hotpot_output, args.trivia_output)
    if args.mode == "skill-on":
        result = materialize_skill_on_evaluations(
            publication_path=Path(args.publication),
            skill_store_path=Path(args.skill_store),
            template_paths=templates,
            output_paths=outputs,
        )
    else:
        result = materialize_step1_skill_off_evaluations(
            training_manifest_path=Path(args.training_manifest),
            sync_receipt_path=Path(args.sync_receipt),
            template_paths=templates,
            output_paths=outputs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
