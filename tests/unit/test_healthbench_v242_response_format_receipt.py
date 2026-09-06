"""Receipt tests against a captured fake transport payload; no API calls."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from src.interactive.openai_gateway import OpenAICompatibleGateway
from tests.unit.test_openai_gateway import request


OBJECT_SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}},
    "required": ["value"], "additionalProperties": False}


@pytest.mark.parametrize("schema,root,count", [
    (OBJECT_SCHEMA, "object", 1),
    ({"oneOf": [OBJECT_SCHEMA, {"type": "string"}]}, "oneOf", 2),
    ({"anyOf": [OBJECT_SCHEMA, {"type": "string"}]}, "anyOf", 2),
    ({"type": "object", "anyOf": [OBJECT_SCHEMA, OBJECT_SCHEMA]}, "object+anyOf", 2),
    (None, None, 0),
])
def test_success_receipt_matches_exact_payload_without_changing_request(schema, root, count):
    metadata = {} if schema is None else {"response_json_schema": json.dumps(schema)}
    req = request(model_metadata=metadata)
    gateway = OpenAICompatibleGateway(max_retries=0)
    expected_payload = gateway.request_payload(req)
    captured = []

    def fake_post(url, api_key, payload):
        captured.append(deepcopy(payload))
        return {"id": "fixture-request", "model": "remote-model-id", "choices": [{
            "message": {"content": "not JSON: provider output is not validated by this receipt"},
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    gateway._post_json = fake_post
    response = asyncio.run(gateway.generate(req))
    assert captured == [expected_payload]
    receipt = response.metadata["structured_output_request"]
    assert receipt["requested"] is (schema is not None)
    assert receipt["scope"] == "client_request_only"
    assert receipt["provider_enforcement_verified"] is False
    if schema is not None:
        actual_format = captured[0]["response_format"]
        assert receipt["type"] == actual_format["type"] == "json_schema"
        assert receipt["strict"] is actual_format["json_schema"]["strict"] is True
        assert receipt["schema_root"] == root
        assert receipt["schema_branch_count"] == count
    assert "schema" not in receipt and "messages" not in receipt and "headers" not in receipt
    assert len(json.dumps(receipt)) < 300
    assert response.text.startswith("not JSON")
