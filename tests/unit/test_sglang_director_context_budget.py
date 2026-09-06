"""Exact-input context bounds using existing offline SGLang test doubles."""

from __future__ import annotations

import asyncio

import pytest

from src.interactive.director import (
    DIRECTOR_ACTION_JSON_SCHEMA_TEXT,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_schema_branch,
)
from src.interactive.rollout_collector import ReceiptValidationError
from tests.unit.test_rollout_collector import (
    CharacterTokenizer,
    POLICY_VERSION,
    ScriptedSGLangClient,
    TwoPhaseScriptedSGLangClient,
)


ACTION = '{"action":"finish"}'
REASONING = '<think>fixture</think>'


class ExactLengthTokenizer(CharacterTokenizer):
    """Materialize complete deterministic prompts at specified token counts."""

    def __init__(self, lengths):
        super().__init__()
        self.lengths = iter(lengths)
        self.encoded_prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        tokens = [100 + i % 100 for i in range(next(self.lengths))]
        self.encoded_prompts.append(tuple(tokens))
        return tokens


def single_client(length, context=32768, maximum=4096):
    client = ScriptedSGLangClient(
        [ACTION], policy_version=POLICY_VERSION,
        expected_server_weight_version='default',
        max_tokens=maximum, max_context_tokens=context,
    )
    client.tokenizer = ExactLengthTokenizer([length])
    return client


def two_phase_client(lengths, context=32768, action=4096, reasoning=2048, logical_calls=1):
    client = TwoPhaseScriptedSGLangClient(
        [REASONING, ACTION] * logical_calls,
        [len(REASONING)] * logical_calls,
        policy_version=POLICY_VERSION,
        expected_server_weight_version='default',
        action_json_schema=DIRECTOR_ACTION_JSON_SCHEMA_TEXT,
        action_json_schema_version='agentgraph.canvas-action-json-schema.v1',
        max_context_tokens=context,
        max_reasoning_tokens=reasoning,
        max_action_tokens=action,
    )
    client.tokenizer = ExactLengthTokenizer(lengths)
    return client


def test_29102_token_input_retained_and_output_capped_at_3666():
    client = single_client(29102)
    response = asyncio.run(client.propose('complete task and receipts', seed=43))
    payload = client.payloads[0]
    assert tuple(payload['input_ids']) == client.tokenizer.encoded_prompts[0]
    assert len(payload['input_ids']) == 29102
    assert payload['sampling_params']['max_new_tokens'] == 3666
    budget = response.metadata['context_budget']
    assert budget['configured_max_new_tokens'] == 4096
    assert budget['effective_max_new_tokens'] == 3666
    assert budget['input_tokens'] == 29102
    assert budget['input_truncated'] is False
    assert response.metadata['max_new_tokens'] == 3666
    assert response.metadata['receipt_verified'] is True


def test_sufficient_space_keeps_configured_limit():
    client = single_client(100)
    response = asyncio.run(client.propose('complete task'))
    assert client.payloads[0]['sampling_params']['max_new_tokens'] == 4096
    assert response.metadata['context_budget']['context_limited'] is False
    assert tuple(client.payloads[0]['input_ids']) == client.tokenizer.encoded_prompts[0]


def test_unconfigured_context_keeps_legacy_behavior():
    client = single_client(29102, context=None)
    response = asyncio.run(client.propose('complete task'))
    assert client.payloads[0]['sampling_params']['max_new_tokens'] == 4096
    assert 'context_budget' not in response.metadata
    assert 'context_budget' not in client.payloads[0]['_flowsteer_request_metadata']


@pytest.mark.parametrize('length', [32768, 32769])
def test_no_remaining_space_fails_locally_without_transport(length):
    client = single_client(length)
    with pytest.raises(ReceiptValidationError, match='context exhausted before generation'):
        asyncio.run(client.propose('complete task'))
    assert client.payloads == []
    assert len(client.tokenizer.encoded_prompts[0]) == length


def test_two_phase_action_counts_new_template_and_preserves_thinking():
    client = two_phase_client([28500, 29102])
    response = asyncio.run(client.propose('complete task and full receipts', seed=43))
    reasoning, action = client.payloads
    assert reasoning['sampling_params']['max_new_tokens'] == 2048
    assert action['sampling_params']['max_new_tokens'] == 3666
    assert reasoning['require_reasoning'] is True
    assert 'json_schema' not in reasoning['sampling_params']
    assert action['sampling_params']['json_schema'] == DIRECTOR_ACTION_JSON_SCHEMA_TEXT
    assert client.tokenizer.chat_calls[0][1]['enable_thinking'] is True
    assert client.tokenizer.chat_calls[1][0][-2]['content'] == REASONING
    for i, payload in enumerate(client.payloads):
        assert tuple(payload['input_ids']) == client.tokenizer.encoded_prompts[i]
        assert payload['logprob_start_len'] == len(payload['input_ids'])
    phases = response.metadata['generation_phase_receipts']
    assert phases['action']['max_new_tokens'] == 3666
    assert phases['action']['configured_max_new_tokens'] == 4096
    assert phases['reasoning']['max_new_tokens'] == 2048
    assert response.metadata['max_action_tokens'] == 3666
    assert response.metadata['configured_max_action_tokens'] == 4096
    assert response.metadata['configured_max_reasoning_tokens'] == 2048
    assert phases['action']['receipt_verified'] is True
    assert phases['reasoning']['receipt_verified'] is True
    assert response.metadata['receipt_verified'] is True


def test_both_phase_effective_budgets_are_measured_independently():
    client = two_phase_client([31768, 32368])
    response = asyncio.run(client.propose('complete task', seed=43))
    assert [p['sampling_params']['max_new_tokens'] for p in client.payloads] == [1000, 400]
    assert response.metadata['max_reasoning_tokens'] == 1000
    assert response.metadata['max_action_tokens'] == 400
    phases = response.metadata['generation_phase_receipts']
    assert phases['reasoning']['context_budget']['configured_max_new_tokens'] == 2048
    assert phases['action']['context_budget']['configured_max_new_tokens'] == 4096


def test_action_with_no_space_is_not_sent_after_reasoning():
    client = two_phase_client([31000, 32768])
    with pytest.raises(ReceiptValidationError, match='context exhausted before generation'):
        asyncio.run(client.propose('complete task'))
    assert len(client.payloads) == 1
    assert client.payloads[0]['_flowsteer_request_metadata']['generation_phase'] == 'reasoning'
    assert len(client.tokenizer.encoded_prompts[1]) == 32768


def test_reasoning_with_no_space_fails_before_either_phase_transport():
    client = two_phase_client([32768])
    with pytest.raises(ReceiptValidationError, match='context exhausted before generation'):
        asyncio.run(client.propose('complete task'))
    assert client.payloads == []


def test_hierarchical_selector_and_parameter_calls_retain_effective_receipts():
    client = two_phase_client([28500, 29102, 29000, 30000], logical_calls=2)
    actions = ('add_subgraph', 'finish')
    response = asyncio.run(client.propose(
        'complete task', seed=43,
        action_json_schema=director_model_admissible_sampling_json_schema_text(actions),
        action_json_schema_version=DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
        action_schema_branch=director_model_admissible_schema_branch(actions),
    ))
    assert len(client.payloads) == 4
    assert [p['sampling_params']['max_new_tokens'] for p in client.payloads] == [2048, 3666, 2048, 2768]
    selector = response.metadata['hierarchical_phase_receipts']['action_selection']
    assert selector['generation_phase_receipts']['action']['max_new_tokens'] == 3666
    assert selector['max_action_tokens'] == 3666
    assert selector['configured_max_action_tokens'] == 4096
    assert response.metadata['max_action_tokens'] == 2768
    assert response.metadata['generation_phase_receipts']['action']['max_new_tokens'] == 2768
    for i, payload in enumerate(client.payloads):
        assert tuple(payload['input_ids']) == client.tokenizer.encoded_prompts[i]


def test_context_receipt_must_match_effective_budget():
    client = single_client(29102)
    original = client._post_json

    def tampered(payload):
        result = original(payload)
        payload['_flowsteer_request_metadata']['context_budget']['effective_max_new_tokens'] = 4096
        return result

    client._post_json = tampered
    with pytest.raises(ReceiptValidationError, match='context budget receipt differs'):
        asyncio.run(client.propose('complete task'))


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_context_limit_must_be_positive_integer(value):
    with pytest.raises(ValueError, match='max_context_tokens'):
        single_client(10, context=value)
