import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.interactive.healthbench_professional_adapter import (
    HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
    HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION,
    evaluator_case_from_official_row,
    load_healthbench_professional_evaluator_cases,
    parse_model_visible_conversation,
    public_task_record_fields,
    render_model_visible_conversation,
    validate_official_healthbench_professional_row,
)
from src.interactive.task_dataset import load_task_records


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prepare_healthbench_professional_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_healthbench_professional_dataset",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_PREPARE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPARE_MODULE)


def _official_row(index: int = 0) -> dict[str, object]:
    return {
        "id": f"case-{index:03d}",
        "conversation": {
            "messages": [
                {
                    "role": "user",
                    "content": f"initial question {index}\nwith a second line",
                },
                {
                    "role": "assistant",
                    "content": "earlier assistant response with </tag> and quotes \"x\"",
                },
                {
                    "role": "user",
                    "content": "follow-up request with [brackets]",
                },
            ]
        },
        "rubric_items": [
            {"criterion_text": f"private criterion {index}", "points": 5},
            {"criterion_text": f"private penalty {index}", "points": -2},
        ],
        "physician_response": f"private physician response {index}",
        "use_case": f"private-use-case-{index}",
        "type": f"private-type-{index}",
        "difficulty": f"private-difficulty-{index}",
        "specialty": f"private-specialty-{index}",
        "canary_string": f"private-canary-{index}",
    }


def test_model_visible_conversation_round_trips_roles_and_content_exactly():
    messages = _official_row()["conversation"]["messages"]

    rendered = render_model_visible_conversation(messages)
    recovered = parse_model_visible_conversation(rendered)

    assert recovered == tuple(messages)
    assert "rubric" not in rendered.casefold()
    assert "physician" not in rendered.casefold()


def test_public_and_private_boundaries_are_disjoint():
    row = _official_row(7)

    public = public_task_record_fields(row)
    private = evaluator_case_from_official_row(row)

    assert public["task_id"] == private["task_id"]
    assert public["evaluator_route"] == HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE
    assert set(public) == {
        "task_id",
        "source_id",
        "question",
        "conversation",
        "evaluator_route",
    }
    serialized_public = json.dumps(public, ensure_ascii=False)
    for hidden in (
        "private criterion 7",
        "private physician response 7",
        "private-use-case-7",
        "private-type-7",
        "private-difficulty-7",
        "private-specialty-7",
        "private-canary-7",
    ):
        assert hidden not in serialized_public
    assert private["rubric_items"] == row["rubric_items"]
    assert private["physician_response"] == row["physician_response"]
    assert private["evaluator_metadata"]["canary_string"] == row["canary_string"]


def test_official_schema_validation_fails_closed_on_extra_field():
    row = _official_row()
    row["unexpected"] = "value"

    with pytest.raises(ValueError, match="official schema"):
        validate_official_healthbench_professional_row(row)


def test_prepare_writes_525_public_tests_and_task_id_joined_private_cases(tmp_path):
    source = tmp_path / "official.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for index in range(525):
            handle.write(json.dumps(_official_row(index), ensure_ascii=False) + "\n")
    catalog_path = tmp_path / "config" / "healthbench.yaml"
    catalog_path.parent.mkdir()
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": (
                    "flowsteer.agentgraph.healthbench-professional.dataset.v1"
                ),
                "task_schema_version": "flowsteer.agentgraph.task.v1",
                "aligned_dir": "data/healthbench",
                "sources": {
                    "dataset_id": "openai/healthbench-professional",
                    "source_split": "test",
                    "expected_count": 525,
                    "source_path": str(source),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = _PREPARE_MODULE.prepare(catalog_path)

    assert (output_dir / "train.jsonl").read_text(encoding="utf-8") == ""
    assert (output_dir / "validation.jsonl").read_text(encoding="utf-8") == ""
    public_lines = (output_dir / "test.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    private_lines = (output_dir / "private_cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(public_lines) == len(private_lines) == 525

    public_rows = [json.loads(line) for line in public_lines]
    private_rows = [json.loads(line) for line in private_lines]
    assert public_rows[0]["task_id"] == "healthbench-professional:case-000"
    assert public_rows[-1]["task_id"] == "healthbench-professional:case-524"
    assert [item["task_id"] for item in public_rows] == [
        item["task_id"] for item in private_rows
    ]
    for public in public_rows:
        assert public["ground_truth"] is None
        assert public["answer"] is None
        assert "evaluator_payload" not in public["metadata"]
        assert "rubric_items" not in public
        assert "physician_response" not in public

    loaded = load_task_records(output_dir / "test.jsonl", expected_split="test")
    assert len(loaded) == 525
    assert parse_model_visible_conversation(loaded[0].question) == tuple(
        _official_row(0)["conversation"]["messages"]
    )
    evaluator_cases = load_healthbench_professional_evaluator_cases(
        output_dir / "private_cases.jsonl"
    )
    assert len(evaluator_cases) == 525
    assert evaluator_cases["healthbench-professional:case-524"][
        "schema_version"
    ] == HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts_by_split"] == {
        "train": 0,
        "validation": 0,
        "test": 525,
    }
    assert manifest["evaluator_join"] == {
        "public_file": "test.jsonl",
        "private_file": "private_cases.jsonl",
        "key": "task_id",
    }
