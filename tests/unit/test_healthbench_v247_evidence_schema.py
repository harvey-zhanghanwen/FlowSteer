"""CPU grammar regressions with synthetic receipts; no model/tokenizer downloads."""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from tests.unit.test_healthbench_clinical_react import artifact, complete, evidence, request
from tests.unit.test_healthbench_v242_evidence_schema import (
    adapter, insufficient, source_observation,
)


METADATA = ("document_id", "source", "title", "date", "url")


@pytest.fixture
def sources():
    return [
        evidence("source-a"),
        {**evidence("source-b"), "source": "Other synthetic source",
         "title": "Other synthetic title", "date": None, "url": None,
         "excerpt": "A second source documents another synthetic relationship."},
    ]


def state_schema(sources, *, output=False):
    executor = adapter(max_completion_artifact_characters=6000)
    node = request(output=output)
    observed = [source_observation(row) for row in sources]
    return executor, node, executor._completion_arguments_schema_for_state(node, observed)


def items_schema(schema):
    return schema["properties"]["value"]["properties"]["evidence_items"]["items"]


def previous_schema(executor, node, sources):
    schema = deepcopy(executor._completion_arguments_schema(node))
    items_schema(schema)["anyOf"] = [
        {"properties": {field: {"const": row[field]} for field in METADATA}}
        for row in sources
    ]
    return schema


@pytest.fixture(scope="module")
def compiler():
    xgr = pytest.importorskip("xgrammar")
    # A complete ASCII byte vocabulary needs no model, tokenizer files or GPU.
    tokenizer = xgr.TokenizerInfo([bytes([index]) for index in range(128)])
    return xgr, xgr.GrammarCompiler(tokenizer, max_threads=1)


def compile_schema(compiler, schema):
    _, grammar_compiler = compiler
    # Match ToolReactExecutionAdapter's response_json_schema serialization.
    return grammar_compiler.compile_json_schema(json.dumps(schema, sort_keys=True))


def grammar_accepts(compiler, grammar, value):
    xgr, _ = compiler
    matcher = xgr.GrammarMatcher(grammar)
    accepted = matcher.accept_string(json.dumps(value, sort_keys=True))
    return accepted and matcher.is_completed()


def completion_schema(executor, arguments_schema):
    return executor._action_schema(
        arguments_schema=arguments_schema, kind="complete", name="complete", resource_id=None,
    )


def test_each_metadata_branch_preserves_the_complete_original_object(sources):
    executor, node, schema = state_schema(sources)
    original = executor._completion_arguments_schema(node)
    original_items = items_schema(original)
    items = items_schema(schema)
    assert items["required"] == original_items["required"]
    assert items["properties"] == original_items["properties"]
    assert len(items["required"]) == 8
    for branch, source in zip(items["anyOf"], sources, strict=True):
        assert branch["type"] == "object"
        assert branch["required"] == original_items["required"]
        assert branch["additionalProperties"] is False
        for field, field_schema in original_items["properties"].items():
            expected = {**field_schema, "const": source[field]} if field in METADATA else field_schema
            assert branch["properties"][field] == expected
    assert "anyOf" not in original_items
    # The branch construction must not mutate the base schema shared by calls.
    assert executor._completion_arguments_schema(node) == original


def test_jsonschema_semantics_equal_the_previous_sibling_intersection(sources):
    executor, node, schema = state_schema(sources)
    previous = previous_schema(executor, node, sources)
    validators = [Draft202012Validator(value) for value in (previous, schema)]
    for validator in validators:
        validator.check_schema(validator.schema)
    cases = [(artifact(source), True) for source in sources]
    cases.append((insufficient(), True))
    for field in items_schema(schema)["required"]:
        missing = artifact(sources[0])
        del missing["evidence_items"][0][field]
        cases.append((missing, False))
    for field in METADATA:
        mixed = artifact(sources[0])
        mixed["evidence_items"][0][field] = sources[1][field]
        cases.append((mixed, False))
    for field, value in [
        ("unexpected", "extra field"), ("supported_claim", ""),
        ("supported_claim", "x" * 1201), ("conditions_or_qualifiers", 1),
        ("conditions_or_qualifiers", "x" * 1001), ("evidence_span", ""),
        ("evidence_span", "x" * 601),
    ]:
        invalid = artifact(sources[0])
        invalid["evidence_items"][0][field] = value
        cases.append((invalid, False))
    for value, expected in cases:
        assert [validator.is_valid({"value": value}) for validator in validators] == [expected, expected]


def test_xgrammar_rejects_metadata_only_and_accepts_complete_objects(compiler, sources):
    executor, _, schema = state_schema(sources)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    for source in sources:
        value = artifact(source)
        metadata_only = deepcopy(value)
        metadata_only["evidence_items"] = [
            {field: source[field] for field in METADATA},
        ]
        assert not grammar_accepts(compiler, grammar, complete(metadata_only))
        assert grammar_accepts(compiler, grammar, complete(value))


@pytest.mark.parametrize("missing", ["supported_claim", "conditions_or_qualifiers", "evidence_span"])
def test_xgrammar_requires_each_non_metadata_field(compiler, sources, missing):
    executor, _, schema = state_schema(sources)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    value = artifact(sources[0])
    del value["evidence_items"][0][missing]
    assert not grammar_accepts(compiler, grammar, complete(value))


@pytest.mark.parametrize("field", METADATA)
def test_xgrammar_never_crosses_metadata_between_observed_sources(compiler, sources, field):
    executor, _, schema = state_schema(sources)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    mixed = artifact(sources[0])
    mixed["evidence_items"][0][field] = sources[1][field]
    assert not grammar_accepts(compiler, grammar, complete(mixed))


def test_xgrammar_preserves_closed_eight_field_object(compiler, sources):
    executor, _, schema = state_schema(sources)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    value = artifact(sources[0])
    value["evidence_items"][0]["extra"] = "not permitted"
    assert not grammar_accepts(compiler, grammar, complete(value))


def test_output_agent_schema_is_unchanged_and_accepts_the_original_text_form(compiler, sources):
    executor, node, schema = state_schema(sources, output=True)
    assert schema == executor._completion_arguments_schema(node)
    assert schema["properties"]["value"]["type"] == "string"
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    assert grammar_accepts(compiler, grammar, complete("A complete synthetic response."))
    assert not grammar_accepts(compiler, grammar, complete(artifact(sources[0])))
