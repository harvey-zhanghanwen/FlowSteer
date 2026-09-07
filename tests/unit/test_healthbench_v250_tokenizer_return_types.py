"""Gateway token counts across real/local and synthetic tokenizer containers."""

import asyncio
from collections import UserDict
from collections.abc import Mapping
from copy import deepcopy

import pytest

from src.interactive.openai_gateway import OpenAICompatibleGateway, OpenAICompatibleGatewayError
from tests.unit.test_healthbench_v249_agent_context import (
    SyntheticTokenizer, client, fake_transport, local_request,
)


LOCAL_TOKENIZER = "/home/test/SKILLEV/skillev-new-b2-temp/tokenizer/Qwen3.5-9B"


class ToListContainer:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return 2  # Container dimensions/fields are not the sequence length.

    def tolist(self):
        return self.values


class ReturnTypeTokenizer(SyntheticTokenizer):
    def __init__(self, encoded):
        super().__init__(0)
        self.encoded = encoded

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return self.encoded


def gateway_for(encoded):
    local_client = client()
    local_client.tokenizer = ReturnTypeTokenizer(encoded)
    gateway = OpenAICompatibleGateway(max_retries=0, local_context_clients={"model": local_client})
    return gateway, local_client, fake_transport(gateway)


@pytest.mark.parametrize("shape", [
    "mapping", "mapping_single_batch", "mapping_tolist", "user_mapping",
    "single_batch", "tolist", "tolist_single_batch",
])
@pytest.mark.parametrize("count", [3410, 30000])
def test_gateway_counts_token_ids_not_container_fields(shape, count):
    ids = [7] * count
    values = {
        "mapping": {"input_ids": ids, "attention_mask": [1] * count},
        "mapping_single_batch": {"input_ids": [ids], "attention_mask": [[1] * count]},
        "mapping_tolist": {"input_ids": ToListContainer([ids]), "attention_mask": [1] * count},
        "user_mapping": UserDict({"input_ids": ids, "attention_mask": [1] * count}),
        "single_batch": [ids],
        "tolist": ToListContainer(ids),
        "tolist_single_batch": ToListContainer([ids]),
    }
    encoded = values[shape]
    gateway, local_client, calls = gateway_for(encoded)
    original = gateway.request_payload(local_request())
    response = asyncio.run(gateway.generate(local_request()))
    budget = response.metadata["requested_sampling"]["context_budget"]
    assert budget["input_tokens"] == count != 2
    assert budget == local_client._context_budget(8192, count)
    assert len(calls) == len(local_client.tokenizer.calls) == 1
    assert calls[0] == {**original, "max_tokens": min(8192, 32768 - count)}
    assert budget["input_truncated"] is False


def test_mapping_with_full_input_window_rejects_before_any_transport():
    gateway, _, calls = gateway_for({"input_ids": [7] * 32768, "attention_mask": [1] * 32768})
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(local_request()))
    assert not calls
    sampling = caught.value.requested_sampling
    assert sampling["request_sent"] is False
    assert sampling["context_budget"]["input_tokens"] == 32768
    assert sampling["context_budget"]["effective_max_new_tokens"] == 0


@pytest.mark.parametrize("encoded", [
    {}, {"attention_mask": [1, 1]}, {"input_ids": [], "attention_mask": []},
    [], [[]], None, "[1, 2]", [True], [1, -1], [1, "2"],
    [[1, 2], [3, 4]], ToListContainer([]),
    {"input_ids": ToListContainer([[1, True]])},
])
def test_invalid_or_empty_encoding_fails_closed_without_fabricated_budget(encoded):
    gateway, local_client, calls = gateway_for(encoded)
    with pytest.raises(OpenAICompatibleGatewayError) as caught:
        asyncio.run(gateway.generate(local_request()))
    assert not calls and len(local_client.tokenizer.calls) == 1
    sampling = caught.value.requested_sampling
    assert sampling["request_sent"] is False
    assert "context_budget" not in sampling
    assert sampling["context_preflight_error"]["stage"] == "tokenization"
    assert sampling["context_preflight_error"]["error_type"] == "ReceiptValidationError"
    assert sampling["context_preflight_error"]["input_tokens"] is None


def test_installed_local_qwen_tokenizer_native_object_matches_gateway_count(monkeypatch):
    # This is the installed tokenizer only, not a model load or generation.
    # Offline flags supplement local_files_only: no missing asset is fetched.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_TOKENIZER, local_files_only=True, trust_remote_code=True,
    )
    item, local_client = local_request(), client()
    local_client.tokenizer = tokenizer
    gateway = OpenAICompatibleGateway(max_retries=0, local_context_clients={"model": local_client})
    original = gateway.request_payload(item)
    native = tokenizer.apply_chat_template(
        original["messages"], tokenize=True, add_generation_prompt=True,
        **original.get("chat_template_kwargs", {}),
    )
    # Independent native input_ids oracle; do not use the normalization helper
    # under test to calculate its own expected answer.
    assert isinstance(native, Mapping)
    assert "input_ids" in native and "attention_mask" in native
    native_ids = native["input_ids"]
    if hasattr(native_ids, "tolist"):
        native_ids = native_ids.tolist()
    if len(native_ids) == 1 and isinstance(native_ids[0], (list, tuple)):
        native_ids = native_ids[0]
    expected = len(native_ids)
    assert expected > len(native) == 2
    calls = fake_transport(gateway)
    response = asyncio.run(gateway.generate(item))
    budget = response.metadata["requested_sampling"]["context_budget"]
    assert budget["input_tokens"] == expected
    assert budget == local_client._context_budget(original["max_tokens"], expected)
    assert len(calls) == 1
    assert calls[0] == {**original, "max_tokens": budget["effective_max_new_tokens"]}
    print(f"local Qwen tokenizer: {type(native).__name__}, fields={len(native)}, input_tokens={expected}")
