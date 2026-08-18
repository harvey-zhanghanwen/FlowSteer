#!/usr/bin/env python3
"""Materialize one evidence-gated joint-QA Skill-on micro-training step.

This is a thin transaction over the existing SkillStore, frozen joint-QA
schedule/cursor, YAML loader, and smoke-runner validator.  It does not collect
rollouts, train a model, or publish an adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_joint_qa_training_schedule import freeze_schedule_artifacts
from scripts.train_agentgraph_smoke import validate_smoke_bounds
from src.interactive.config_loader import load_yaml
from src.interactive.skills import SkillStatus, SkillStore


DEFAULT_TEMPLATE = PROJECT_ROOT / "config/training_joint_qa_progressive_skill_on_step1.yaml"
DEFAULT_PUBLICATION = (
    PROJECT_ROOT
    / "artifacts/joint_qa_progressive/skill_epoch_000000/publication_results.json"
)
DEFAULT_STORE = (
    PROJECT_ROOT / "artifacts/joint_qa_progressive/skill_epoch_000000/skills.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/joint_qa_progressive/training/step_000001/resolved_config.yaml"
)
DATASETS = ("hotpotqa", "triviaqa")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def materialize(
    *,
    template_path: Path = DEFAULT_TEMPLATE,
    publication_path: Path = DEFAULT_PUBLICATION,
    skill_store_path: Path = DEFAULT_STORE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"write-once resolved config already exists: {output_path}")
    publication = _mapping(
        json.loads(publication_path.read_text(encoding="utf-8")),
        "publication",
    )
    publications = _mapping(publication.get("publications"), "publications")
    if set(publication.get("active_datasets", ())) != set(DATASETS):
        raise RuntimeError(
            "Skill-on training requires evidence-gated ACTIVE Skills for both datasets"
        )

    store = SkillStore(skill_store_path)
    required_ids: list[str] = []
    records = []
    for dataset in DATASETS:
        payload = _mapping(publications.get(dataset), f"publications.{dataset}")
        skill_payload = _mapping(payload.get("skill"), f"publications.{dataset}.skill")
        skill_id = str(skill_payload.get("skill_id", ""))
        record = store.get(skill_id)
        if record is None or record.status is not SkillStatus.ACTIVE:
            raise RuntimeError(f"{dataset} publication has no matching ACTIVE Skill")
        if record.condition.get("task_family") != dataset:
            raise RuntimeError(f"{dataset} ACTIVE Skill has an incompatible task condition")
        if record.activated_epoch is None or record.activated_epoch > 2:
            raise RuntimeError(f"{dataset} ACTIVE Skill is not visible in epoch 2")
        required_ids.append(record.skill_id)
        records.append(record)

    library_versions = {record.versions.skill_library for record in records}
    posterior_versions = {record.versions.posterior for record in records}
    policy_versions = {record.versions.policy for record in records}
    if len(library_versions) != 1 or len(posterior_versions) != 1:
        raise RuntimeError("ACTIVE Skills do not share one frozen library/posterior regime")
    if len(policy_versions) != 1:
        raise RuntimeError("ACTIVE Skills do not share one frozen behavior policy")

    config = load_yaml(template_path)
    skills = dict(_mapping(config["skills"], "skills"))
    skills.update(
        store_path=_relative(skill_store_path),
        library_version=next(iter(library_versions)),
        posterior_version=next(iter(posterior_versions)),
        required_skill_ids=required_ids,
        current_epoch=2,
    )
    config["skills"] = skills
    director = _mapping(config["director"], "director")
    if director.get("behavior_policy_version") != next(iter(policy_versions)):
        raise RuntimeError("training template policy differs from ACTIVE Skill evidence")
    validate_smoke_bounds(config)

    data = _mapping(config["data"], "data")
    joint = _mapping(data.get("joint_qa_micro"), "data.joint_qa_micro")
    schedule_path = PROJECT_ROOT / str(joint["schedule_path"])
    cursor_path = PROJECT_ROOT / str(joint["cursor_path"])
    if schedule_path.exists() or cursor_path.exists():
        raise FileExistsError("write-once training schedule or cursor already exists")
    schedule = freeze_schedule_artifacts(
        train_path=PROJECT_ROOT / str(data["train_path"]),
        validation_path=PROJECT_ROOT / str(data["validation_path"]),
        skill_confirmation_path=PROJECT_ROOT / str(data["skill_confirmation_path"]),
        test_path=PROJECT_ROOT / str(data["test_path"]),
        schedule_path=schedule_path,
        cursor_path=cursor_path,
        step_count=1,
        rollouts_per_task=8,
        hotpotqa_task_positions=(8,),
        triviaqa_task_positions=(8,),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "status": "materialized",
        "config": str(output_path),
        "skill_store": str(skill_store_path),
        "required_skill_ids": required_ids,
        "policy_version": next(iter(policy_versions)),
        "library_version": next(iter(library_versions)),
        "posterior_version": next(iter(posterior_versions)),
        "schedule": schedule,
        "training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--publication", default=str(DEFAULT_PUBLICATION))
    parser.add_argument("--skill-store", default=str(DEFAULT_STORE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = materialize(
        template_path=Path(args.template).resolve(),
        publication_path=Path(args.publication).resolve(),
        skill_store_path=Path(args.skill_store).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
