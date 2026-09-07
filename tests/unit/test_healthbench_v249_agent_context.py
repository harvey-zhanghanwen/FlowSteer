"""Exact local context receipts using synthetic tokenizers and fake transport."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts.train_agentgraph_smoke import _local_agent_context_clients
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.model_registry import ModelRegistry
from src.interactive.openai_gateway import OpenAICompatibleGateway, OpenAICompatibleGatewayError
from src.interactive.react_execution import ReactGenerationError, ToolReactExecutionAdapter
from src.interactive.rollout_collector import SGLangReceiptDirectorClient
from src.interactive.tool_runtime import ToolRegistry
from tests.unit.test_openai_gateway import request


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "http://127.0.0.1:8026/v1"
DIRECTOR = {"api_base": ENDPOINT, "served_model_name": "supervisor_theta", "max_context_tokens": 32768}


class SyntheticTokenizer:
    def __init__(self, count):
        self.count = count
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return [7] * self.count

    def decode(self, tokens, **kwargs):
        return "unused synthetic decoder"


def client(count=100, **kwargs):
    return SGLangReceiptDirectorClient(
        SyntheticTokenizer(count), base_url=ENDPOINT, policy_version="synthetic",
        max_context_tokens=32768, **kwargs,
    )


def local_request(*, thinking=True):
    metadata = {
        "max_tokens": "4096", "deployment_locality": "local", "sampling_backend": "sglang",
        "chat_template_enable_thinking": "true" if thinking else "false",
        "supports_top_k": "true", "top_k": "-1", "generation_seed": "17",
        **({"thinking_budget": "4096"} if thinking else {}),
    }
    value = request(execution_mode="react", model_metadata=metadata)
    return replace(value, provider=replace(value.provider, endpoint=ENDPOINT),
        model=replace(value.model, model_name="supervisor_theta", context_window=32768))


def registry(item=None):
    item = item or local_request()
    return ModelRegistry([item.provider], [item.model])


def fake_transport(gateway, *, body=None):
    calls = []

    def post(url, api_key, payload):
        calls.append(deepcopy(payload))
        if body is not None:
            raise HTTPError(url, 400, "Bad Request", {"unused-header": "not retained"}, BytesIO(body))
        return {"choices": [{"message": {"content": "Synthetic complete response."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    gateway._post_json = post
    return calls


@pytest.mark.parametrize("count,thinking,expected", [
    (100, True, 8192), (24576, True, 8192), (30000, True, 2768),
    (32767, True, 1), (100, False, 4096), (30000, False, 2768),
])
def test_exact_budget_preserves_messages_and_actual_template_kwargs(count, thinking, expected):
    item, local_client = local_request(thinking=thinking), client(count)
    gateway = OpenAICompatibleGateway(max_retries=0, local_context_clients={"model": local_client})
    original = deepcopy(gateway.request_payload(item))
    calls = fake_transport(gateway)
    response = asyncio.run(gateway.generate(item))
    assert len(calls) == len(local_client.tokenizer.calls) == 1
    assert calls[0] == {**original, "max_tokens": expected}
    assert local_client.tokenizer.calls[0] == (original["messages"], {
        "tokenize": True, "add_generation_prompt": True, **original["chat_template_kwargs"],
    })
    sampling = response.metadata["requested_sampling"]
    assert sampling["context_budget"] == local_client._context_budget(original["max_tokens"], count)
    assert sampling["max_tokens"] == expected and sampling["request_sent"] is True
    assert sampling["chat_template_enable_thinking"] is thinking
    if thinking:
        assert sampling["visible_max_tokens"] == sampling["thinking_budget"] == 4096
        assert sampling["context_budget"]["configured_max_new_tokens"] == 8192
    assert gateway.request_payload(item) == original


@pytest.mark.parametrize("count", [32768, 34000])
def test_exhausted_input_fails_before_transport_with_public_count_receipt(count):
    local_client = client(count)
    gateway = OpenAICompatibleGateway(local_context_clients={"model": local_client})
    calls = fake_transport(gateway)
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(local_request()))
    error = caught.value
    assert not calls and len(local_client.tokenizer.calls) == 1
    assert error.request_status == "failed" and error.http_status is None
    assert error.provider_id == "provider" and error.model_id == "model"
    assert f"input_tokens={count}" in str(error)
    assert "configured_output_tokens=8192" in str(error) and "context_limit=32768" in str(error)
    assert "not sent" in str(error)
    sampling = error.requested_sampling
    assert sampling["request_sent"] is False
    assert sampling["context_budget"] == {
        "profile": "sglang-exact-input-context-budget.v1", "max_context_tokens": 32768,
        "input_tokens": count, "configured_max_new_tokens": 8192,
        "effective_max_new_tokens": 0, "context_limited": True, "input_truncated": False,
    }


@pytest.mark.parametrize("stage", ["tokenization", "budget"])
def test_non_capacity_errors_are_not_reported_as_context_exhaustion(stage):
    local_client = client()

    def fail(*args, **kwargs):
        raise TypeError("synthetic incompatible interface")

    if stage == "tokenization":
        local_client.tokenizer.apply_chat_template = fail
    else:
        local_client._context_budget = fail
    gateway = OpenAICompatibleGateway(local_context_clients={"model": local_client})
    calls = fake_transport(gateway)
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(local_request()))
    sampling = caught.value.requested_sampling
    assert not calls and sampling["request_sent"] is False
    assert "context_budget" not in sampling
    assert sampling["context_preflight_error"]["stage"] == stage
    assert sampling["context_preflight_error"]["error_type"] == "TypeError"
    assert "TypeError" in str(caught.value) and "exhausted" not in str(caught.value)


def test_preflight_context_receipt_reaches_existing_react_failed_model_call():
    gateway = OpenAICompatibleGateway(local_context_clients={"model": client(32768)})
    calls = fake_transport(gateway)
    executor = ToolReactExecutionAdapter(gateway=gateway, tool_registry=ToolRegistry(()),
        max_turns=1, max_tool_calls=0)
    with pytest.raises(ReactGenerationError) as caught:
        asyncio.run(executor.execute(local_request()))
    assert not calls and not caught.value.tool_receipts
    model_call = caught.value.model_calls[0]
    assert model_call["request_status"] == "failed"
    assert model_call["requested_sampling"]["request_sent"] is False
    assert model_call["requested_sampling"]["context_budget"]["input_tokens"] == 32768


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_and_unmapped_remote_requests_keep_original_payload(enabled):
    local_client = client(100000)
    item = request() if enabled else local_request()
    gateway = OpenAICompatibleGateway(local_context_clients={"other-model": local_client} if enabled else None)
    original = gateway.request_payload(item)
    calls = fake_transport(gateway)
    response = asyncio.run(gateway.generate(item))
    assert calls == [original] and not local_client.tokenizer.calls
    assert "context_budget" not in response.metadata["requested_sampling"]
    assert "request_sent" not in response.metadata["requested_sampling"]


def test_http400_preserves_only_bounded_error_fields_and_effective_context():
    gateway = OpenAICompatibleGateway(local_context_clients={"model": client(30000)})
    message = "Requested context exceeds limit. " * 40
    body = json.dumps({"error": {"type": "invalid_request_error", "code": "context_length_exceeded",
        "message": message, "param": "not retained"}, "request_payload": "NEVER_RETAIN"}).encode()
    calls = fake_transport(gateway, body=body)
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(local_request()))
    error = caught.value
    assert error.http_status == 400 and error.request_status == "failed" and len(calls) == 1
    sampling = error.requested_sampling
    assert sampling["provider_error"] == {"type": "invalid_request_error",
        "code": "context_length_exceeded", "message": message[:512]}
    assert sampling["context_budget"]["effective_max_new_tokens"] == 2768
    assert sampling["request_sent"] is True
    assert "NEVER_RETAIN" not in json.dumps(sampling) and "not retained" not in json.dumps(sampling)


@pytest.mark.parametrize("body,expected", [
    (b'not json', None), (b'{"message":"outside error field"}', None),
    (b'{"error":"short public failure"}', {"message": "short public failure"}),
    (b'{"error":{"code":400,"type":[],"message":{}}}', {"code": 400}),
])
def test_malformed_or_nonstandard_error_body_does_not_hide_http_status(body, expected):
    gateway = OpenAICompatibleGateway(max_retries=0)
    calls = fake_transport(gateway, body=body)
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(request()))
    assert caught.value.http_status == 400 and len(calls) == 1
    assert caught.value.requested_sampling.get("provider_error") == expected


@pytest.mark.parametrize("invalid", ["true", "false", 1, 0, None])
def test_factory_flag_is_strict_boolean(invalid):
    with pytest.raises(ConfigurationError, match="must be boolean"):
        _local_agent_context_clients({"local_agent_context_budget": invalid}, DIRECTOR, registry(), client())


def test_factory_disabled_does_not_require_a_context_client():
    assert _local_agent_context_clients({}, {}, registry(), None) == {}


def test_factory_reuses_actual_catalog_client_only_for_the_matching_local_model(monkeypatch):
    monkeypatch.setenv("FLOWSTEER_SUPERVISOR_PORT", "8026")
    catalog = ModelRegistry.from_dict(load_yaml(
        ROOT / "config/model_catalog_healthbench_professional_clinical_reference_thinking_v11.yaml",
    ))
    local_client = client()
    bindings = _local_agent_context_clients({"local_agent_context_budget": True}, DIRECTOR, catalog, local_client)
    assert bindings == {"qwen3.5-9b-local": local_client}
    assert not local_client.tokenizer.calls


@pytest.mark.parametrize("mismatch", ["endpoint", "model_name", "locality", "backend"])
def test_factory_never_maps_another_deployment_or_remote_backend(mismatch):
    item = local_request()
    if mismatch == "endpoint":
        item = replace(item, provider=replace(item.provider, endpoint="http://127.0.0.1:9999/v1"))
    elif mismatch == "model_name":
        item = replace(item, model=replace(item.model, model_name="another-served-model"))
    else:
        key, value = ("deployment_locality", "remote") if mismatch == "locality" else ("sampling_backend", "other")
        item = replace(item, model=replace(item.model, metadata={**item.model.metadata, key: value}))
    assert _local_agent_context_clients({"local_agent_context_budget": True}, DIRECTOR, registry(item), client()) == {}


@pytest.mark.parametrize("window", [None, 16384])
def test_factory_rejects_conflicting_context_for_the_same_deployment(window):
    item = local_request()
    item = replace(item, model=replace(item.model, context_window=window))
    with pytest.raises(ConfigurationError, match="context_window"):
        _local_agent_context_clients({"local_agent_context_budget": True}, DIRECTOR, registry(item), client())


@pytest.mark.parametrize("change", [
    {"max_context_tokens": None}, {"max_context_tokens": True}, {"max_context_tokens": 16384},
    {"api_base": "http://127.0.0.1:9999/v1"}, {"served_model_name": ""},
])
def test_factory_rejects_mismatched_director_client_configuration(change):
    with pytest.raises(ConfigurationError):
        _local_agent_context_clients({"local_agent_context_budget": True}, {**DIRECTOR, **change}, registry(), client())
