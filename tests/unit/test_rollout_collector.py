from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_action_parser import AgentActionParseError, AgentActionParser
from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_ACTION_JSON_SCHEMA,
    DIRECTOR_ACTION_JSON_SCHEMA_TEXT,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
    DIRECTOR_SGLANG_SAMPLING_SCHEMA_VERSION,
    DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION,
    DIRECTOR_SYSTEM_PROMPT,
    director_actions_from_admissible_schema_branch,
    director_action_json_schema_text,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_sampling_json_schema_text_v1,
    director_model_admissible_sampling_json_schema_text_v3,
    director_model_admissible_schema_branch,
    director_model_admissible_schema_branch_v1,
    director_model_admissible_schema_branch_v3,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_action_parameter_json_schema_text,
    director_live_action_target_domains_json,
    director_live_modify_agent_selector_json_schema_text,
    director_live_relation_candidate_selector_json_schema_text,
    director_modify_agent_field_sampling_json_schema_text,
    director_modify_agent_field_selector_json_schema_text,
    director_sglang_sampling_json_schema_text,
    director_state_conditioned_sampling_json_schema_text,
    encode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.persistence import EvidenceStore
from src.interactive.records import TaskRecord
from src.interactive.rollout_collector import (
    AGENTGRAPH_SMOKE_SOURCES,
    AgentGraphRolloutCollector,
    ReceiptValidationError,
    RolloutGate,
    SGLangReceiptDirectorClient,
    _validate_v3_hierarchical_action_receipt,
    select_balanced_tasks,
)
from src.interactive.scientific_sampling import (
    GenerationPhase,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.versioning import VersionBundle


POLICY_VERSION = "qwen35-9b-smoke-step-0000"
EVALUATOR_VERSION = "smoke-evaluator-v1"


class CharacterTokenizer:
    def __init__(self) -> None:
        self.chat_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return [101, 102, 103]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(chr(token_id) for token_id in token_ids)


class ScriptedSGLangClient(SGLangReceiptDirectorClient):
    def __init__(self, actions, **kwargs):
        self.actions = list(actions)
        self.payloads = []
        super().__init__(CharacterTokenizer(), **kwargs)

    def _post_json(self, payload):
        self.payloads.append(payload)
        text = self.actions.pop(0)
        output_ids = [ord(character) for character in text]
        return {
            "text": text,
            "output_ids": output_ids,
            "meta_info": {
                "id": f"request-{len(self.payloads)}",
                "weight_version": "default",
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(output_ids),
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [
                    [-0.01 * (index + 1), token_id, None]
                    for index, token_id in enumerate(output_ids)
                ],
            },
        }


class MismatchedTokenClient(ScriptedSGLangClient):
    def _post_json(self, payload):
        value = dict(super()._post_json(payload))
        value["output_ids"] = list(value["output_ids"])
        value["output_ids"][-1] += 1
        return value


class FakeGateway:
    async def generate(self, request):
        return AgentResponse(
            "final answer",
            {
                "provider_request_id": request.request_id + ":provider",
                "provider_model": request.model.model_name,
                "finish_reason": "stop",
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            },
        )


class UnifiedMetadataGateway:
    async def generate(self, request):
        return AgentResponse(
            "final answer",
            {
                "provider_request_id": request.request_id + ":provider",
                "provider_model": request.model.model_name,
                "finish_reason": "stop",
                "provider_id": request.provider.provider_id,
                "model_id": request.model.model_id,
                "attempt_count": True,
                "generation_seed": 17,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
                "execution_mode": "react",
                "react_turns_used": 2,
                "tool_calls": 1,
                "tool_receipts": (
                    {
                        "tool_id": "qa-retrieval.search",
                        "tool_version": "retrieval-v1",
                        "latency_ms": 1.25,
                    },
                ),
                "react_trace": (
                    {"turn": 1, "observation_status": "success"},
                    {"turn": 2, "observation_status": "completed"},
                ),
                "model_calls": (
                    {
                        "turn": 1,
                        "request_id": request.request_id + ":react:1",
                        "metadata": {
                            "provider_request_id": "provider-react-1",
                            "finish_reason": "stop",
                            "prompt_tokens": 5,
                            "completion_tokens": 1,
                            "total_tokens": 6,
                            "latency_ms": 1.25,
                            "attempt_count": 1,
                        },
                    },
                    {
                        "turn": 2,
                        "request_id": request.request_id + ":react:2",
                        "metadata": {
                            "provider_request_id": "provider-react-2",
                            "finish_reason": "stop",
                            "prompt_tokens": 7,
                            "completion_tokens": 2,
                            "total_tokens": 9,
                            "latency_ms": 2.75,
                            "attempt_count": 2,
                        },
                    },
                ),
                "environment_id": "webshop:test-1",
                "task_family": "WebShop",
                "environment_revision": 1,
                "environment_reset_receipt": {
                    "environment_id": "webshop:test-1",
                    "observation": "initial observation",
                },
                "environment_receipts": (
                    {
                        "turn": 1,
                        "action": "search[query]",
                        "state_advanced": True,
                    },
                ),
                "environment_terminal": True,
                "environment_turns_used": 1,
                "environment_steps": 1,
                "evaluator_environment_trace": (
                    {
                        "step": 0,
                        "action": "search[query]",
                        "reward": 1.0,
                        "done": True,
                    },
                ),
                "opaque_runtime_object": object(),
            },
        )


class FailOnceReceiptGateway(FakeGateway):
    def __init__(self) -> None:
        self.failed = False

    async def generate(self, request):
        if not self.failed:
            self.failed = True
            raise ReactExecutionError(
                "bounded execution exhausted",
                react_trace=(
                    {"turn": 1, "observation_status": "success"},
                ),
                tool_receipts=(
                    {"tool_id": "qa.search", "success": True},
                ),
                model_calls=(
                    {"turn": 1, "request_id": request.request_id},
                ),
            )
        return await super().generate(request)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("vector", endpoint="http://provider.invalid/v1")],
        [ModelSpec("cheap-model", "vector", model_name="cheap-provider-model")],
    )


def _versions() -> VersionBundle:
    return VersionBundle(
        policy=POLICY_VERSION,
        model_catalog="catalog-v1",
        evaluator=EVALUATOR_VERSION,
        prompt="minimal-director-v1",
        tool="agentgraph-v1",
    )


def _task(
    task_id: str = "hotpotqa:first",
    source: str = "HotpotQA",
    split: str = "train",
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        question="What is the answer?",
        ground_truth="final answer",
        split=split,
        metadata={"source": source},
    )


def _orchestrator(
    registry: ModelRegistry,
    client: SGLangReceiptDirectorClient,
    *,
    max_rounds: int,
    base_seed: int = 7,
    rollout_ordinal: int = 0,
    sampling_action_profile: str | None = None,
    sampling_action_schema_version: str = (
        DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
    ),
) -> AgentGraphOrchestrator:
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=base_seed),
        schedule_purpose="exploit",
        ordered_sequence_hash=stable_hash(["hotpotqa:first"]),
        sequence_position=rollout_ordinal,
        task_id="hotpotqa:first",
        optimizer_step_or_anchor_ordinal=0,
    )
    return AgentGraphOrchestrator(
        registry,
        client,
        max_rounds=max_rounds,
        seed=base_seed,
        sampling_base_seed=base_seed,
        sampling_coordinate=coordinate,
        sampling_action_profile=sampling_action_profile,
        sampling_action_schema_version=sampling_action_schema_version,
    )


def test_native_sglang_receipt_uses_real_input_ids_and_separates_versions():
    text = 'preface {"action":"finish"}\nunused explanation'
    client = ScriptedSGLangClient(
        [text],
        policy_version=POLICY_VERSION,
        adapter_name="theta_live",
        expected_server_weight_version="default",
        base_url="http://127.0.0.1:8015/v1",
        action_json_schema=DIRECTOR_ACTION_JSON_SCHEMA_TEXT,
        action_json_schema_version="agentgraph.canvas-action-json-schema.v1",
    )

    response = asyncio.run(client.propose("ordinary prompt", seed=23))
    payload = client.payloads[0]
    messages, template_kwargs = client.tokenizer.chat_calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "ordinary prompt"}
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert client.generate_url == "http://127.0.0.1:8015/generate"
    assert payload["input_ids"] == [101, 102, 103]
    assert payload["return_logprob"] is True
    assert payload["lora_path"] == "theta_live"
    assert payload["sampling_params"]["sampling_seed"] == 23
    assert json.loads(payload["sampling_params"]["json_schema"]) == (
        DIRECTOR_ACTION_JSON_SCHEMA
    )
    assert response.metadata["policy_version"] == POLICY_VERSION
    assert response.metadata["server_weight_version"] == "default"
    assert response.metadata["adapter_name"] == "theta_live"
    assert response.metadata["latency_ms"] >= 0.0
    assert response.metadata["attempt_count"] == 1
    assert response.metadata["generation_seed"] == 23
    assert response.metadata["action_json_schema_version"] == (
        "agentgraph.canvas-action-json-schema.v1"
    )
    assert len(response.metadata["output_token_ids"]) == len(
        response.metadata["behavior_log_probs"]
    )

    action = AgentActionParser().parse(text)
    consumed = client.executed_prefix_tokens(response, action)
    assert consumed == action.consumed_end
    assert consumed < len(response.metadata["output_token_ids"])


def test_native_sglang_per_request_schema_does_not_mutate_client_default():
    client = ScriptedSGLangClient(
        ['{"action":"add_subgraph","agents":[],"relations":[]}', '{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
        action_json_schema=DIRECTOR_ACTION_JSON_SCHEMA_TEXT,
        action_json_schema_version="agentgraph.canvas-action-json-schema.v1",
    )
    add_schema = director_state_conditioned_sampling_json_schema_text("add_subgraph")

    overridden = asyncio.run(
        client.propose(
            "first prompt",
            action_json_schema=add_schema,
            action_json_schema_version=(
                DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch="add_subgraph",
        )
    )
    defaulted = asyncio.run(client.propose("second prompt"))

    assert client.payloads[0]["sampling_params"]["json_schema"] == add_schema
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        DIRECTOR_ACTION_JSON_SCHEMA_TEXT
    )
    assert overridden.metadata["action_json_schema_version"] == (
        DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
    )
    assert overridden.metadata["action_schema_branch"] == "add_subgraph"
    assert defaulted.metadata["action_json_schema_version"] == (
        "agentgraph.canvas-action-json-schema.v1"
    )
    assert defaulted.metadata["action_schema_branch"] is None

    with pytest.raises(ValueError, match="must be supplied together"):
        client.request_payload(
            "invalid override",
            action_json_schema=add_schema,
        )
    with pytest.raises(ValueError, match="does not match its declared branch"):
        client.request_payload(
            "mismatched override",
            action_json_schema=add_schema,
            action_json_schema_version=(
                DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch="finish",
        )


def test_native_sglang_validates_model_admissible_schema_receipt():
    client = ScriptedSGLangClient(
        [
            '{"action":"modify_agent"}',
            '{"action":"modify_agent","field":"contract"}',
            '{"action":"modify_agent","agent_id":"solver","contract":"repair"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    actions = ("add_subgraph", "modify_agent", "finish")
    schema = director_model_admissible_sampling_json_schema_text(actions)
    branch = director_model_admissible_schema_branch(actions)

    response = asyncio.run(
        client.propose(
            "current Canvas",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch=branch,
        )
    )

    assert client.payloads[0]["sampling_params"]["json_schema"] == schema
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_modify_agent_field_selector_json_schema_text()
    )
    assert client.payloads[2]["sampling_params"]["json_schema"] == (
        director_modify_agent_field_sampling_json_schema_text("contract")
    )
    assert response.metadata["action_json_schema_version"] == (
        DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
    )
    assert response.metadata["action_schema_branch"] == branch
    assert response.metadata["action_decoding_strategy"] == (
        "hierarchical_json_schema"
    )
    assert response.metadata["selected_action"] == "modify_agent"
    assert response.metadata["selected_modify_field"] == "contract"
    assert response.metadata["request_count"] == 3
    assert director_actions_from_admissible_schema_branch(branch) == actions

    with pytest.raises(ValueError, match="strict schema for its branch"):
        client.request_payload(
            "mismatched admissible receipt",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch="admissible:finish",
        )
    with pytest.raises(ValueError, match="strict schema for its branch"):
        client.request_payload(
            "malformed admissible receipt",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch="admissible:finish|unknown",
        )


def test_native_sglang_v3_uses_exact_live_relation_candidate_receipt():
    candidates = [
        {
            "source_id": "reasoner",
            "target_id": "verifier",
            "source_to_target": True,
            "target_to_source": False,
        },
        {
            "source_id": "verifier",
            "target_id": "format",
            "source_to_target": True,
            "target_to_source": False,
        },
    ]
    actions = ("set_relation", "finish")
    domains = {
        "set_relation": {"candidates": candidates},
        "finish": {"admissible": True},
    }
    schema = director_model_admissible_sampling_json_schema_text_v3(actions)
    branch = director_model_admissible_schema_branch_v3(actions)
    domains_json = director_live_action_target_domains_json(actions, domains)
    expected_relation = {
        "action": "set_relation",
        **candidates[1],
    }
    client = ScriptedSGLangClient(
        [
            '{"action":"set_relation"}',
            '{"action":"set_relation","candidate_index":1}',
            json.dumps(expected_relation, separators=(",", ":")),
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "current Canvas",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=branch,
            action_target_domains_json=domains_json,
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )

    assert client.payloads[0]["sampling_params"]["json_schema"] == schema
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_live_relation_candidate_selector_json_schema_text(domains)
    )
    assert client.payloads[2]["sampling_params"]["json_schema"] == (
        director_live_action_parameter_json_schema_text(
            "set_relation",
            domains,
            relation_candidate_index=1,
        )
    )
    assert response.text == json.dumps(expected_relation, separators=(",", ":"))
    assert response.metadata["action_target_domains_json"] == domains_json
    assert response.metadata["action_target_domain_version"] == (
        DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
    )
    assert response.metadata["selected_relation_candidate"] == 1
    assert response.metadata["request_count"] == 3
    assert set(response.metadata["hierarchical_phase_receipts"]) == {
        "action_selection",
        "relation_candidate_selection",
    }

    finish_client = ScriptedSGLangClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    finish_actions = ("finish",)
    finish_domains = {"finish": {"admissible": True}}
    finish_response = asyncio.run(
        finish_client.propose(
            "terminal Canvas",
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text_v3(
                    finish_actions
                )
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=director_model_admissible_schema_branch_v3(
                finish_actions
            ),
            action_target_domains_json=director_live_action_target_domains_json(
                finish_actions,
                finish_domains,
            ),
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )
    assert finish_client.payloads[0]["sampling_params"]["json_schema"] == (
        director_live_action_parameter_json_schema_text(
            "finish",
            finish_domains,
        )
    )
    assert finish_response.metadata["selected_action"] == "finish"
    assert finish_response.metadata["request_count"] == 1


def test_native_sglang_v3_samples_add_declarations_then_complete_exact_action():
    domains = {
        "add_subgraph": {
            "min_new_agents": 1,
            "max_new_agents": 2,
            "existing_agent_ids": ["incumbent"],
            "required_agent_fields": [
                "agent_id",
                "model_id",
                "contract",
                "role_family",
                "allowed_tools",
                "execution_mode",
            ],
            "model_ids": ["cheap-model", "other-model"],
            "role_constraints": {
                "reasoner": {
                    "execution_modes": ["react"],
                    "allowed_tools": [["qa-retrieval"]],
                },
                "verifier": {
                    "execution_modes": ["reasoning"],
                    "allowed_tools": [[]],
                },
            },
            "endpoint_scope": {
                "relation_endpoint_sources": [
                    "existing_agent_ids",
                    "same_action_agent_ids",
                ],
                "output_agent_id_sources": [
                    "existing_agent_ids",
                    "same_action_agent_ids",
                ],
            },
        }
    }
    declarations = {
        "action": "add_subgraph",
        "agents": [
            {
                "agent_id": "reasoner",
                "model_id": "cheap-model",
                "contract": "align evidence",
                "role_family": "reasoner",
                "allowed_tools": ["qa-retrieval"],
                "execution_mode": "react",
            }
        ],
    }
    final_action = {
        **declarations,
        "relations": [
            {
                "source_id": "incumbent",
                "target_id": "reasoner",
                "source_to_target": True,
                "target_to_source": False,
            }
        ],
        "output_agent_id": "reasoner",
    }
    actions = ("add_subgraph",)
    domains_json = director_live_action_target_domains_json(actions, domains)
    client = ScriptedSGLangClient(
        [
            json.dumps(declarations, separators=(",", ":")),
            json.dumps(final_action, separators=(",", ":")),
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "current Canvas",
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text_v3(actions)
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=director_model_admissible_schema_branch_v3(actions),
            action_target_domains_json=domains_json,
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )

    assert client.payloads[0]["sampling_params"]["json_schema"] == (
        director_live_add_subgraph_agent_declarations_json_schema_text(domains)
    )
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=declarations["agents"],
        )
    )
    assert response.text == json.dumps(final_action, separators=(",", ":"))
    assert response.metadata["selected_add_agent_ids"] == ["reasoner"]
    assert response.metadata["request_count"] == 2
    assert set(response.metadata["hierarchical_phase_receipts"]) == {
        "add_agent_declarations"
    }

    conflict_client = ScriptedSGLangClient(
        [
            json.dumps(
                {
                    **declarations,
                    "agents": [
                        {**declarations["agents"][0], "agent_id": "incumbent"}
                    ],
                },
                separators=(",", ":"),
            )
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    with pytest.raises(
        ReceiptValidationError,
        match="Agent declaration phase is invalid",
    ):
        asyncio.run(
            conflict_client.propose(
                "current Canvas",
                action_json_schema=(
                    director_model_admissible_sampling_json_schema_text_v3(actions)
                ),
                action_json_schema_version=(
                    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
                ),
                action_schema_branch=director_model_admissible_schema_branch_v3(
                    actions
                ),
                action_target_domains_json=domains_json,
                action_target_domain_version=(
                    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
                ),
            )
        )


def test_native_sglang_v3_binds_modify_agent_and_discrete_value():
    actions = ("modify_agent",)
    domains = {
        "modify_agent": {
            "mutable_fields": ["model_id", "contract"],
            "per_agent_candidates": [
                {
                    "agent_id": "reasoner",
                    "mutable_fields": ["model_id", "contract"],
                    "discrete_value_domains": {"model_id": ["other-model"]},
                },
                {
                    "agent_id": "verifier",
                    "mutable_fields": ["model_id"],
                    "discrete_value_domains": {"model_id": ["cheap-model"]},
                },
            ],
        }
    }
    domains_json = director_live_action_target_domains_json(actions, domains)
    final_action = {
        "action": "modify_agent",
        "agent_id": "reasoner",
        "model_id": "other-model",
    }
    client = ScriptedSGLangClient(
        [
            '{"action":"modify_agent","field":"model_id"}',
            '{"action":"modify_agent","agent_id":"reasoner"}',
            json.dumps(final_action, separators=(",", ":")),
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "repair Canvas",
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text_v3(actions)
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=director_model_admissible_schema_branch_v3(actions),
            action_target_domains_json=domains_json,
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )

    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_live_modify_agent_selector_json_schema_text(domains, "model_id")
    )
    assert client.payloads[2]["sampling_params"]["json_schema"] == (
        director_live_action_parameter_json_schema_text(
            "modify_agent",
            domains,
            modify_field="model_id",
            modify_agent_id="reasoner",
        )
    )
    assert response.metadata["selected_modify_agent_id"] == "reasoner"
    assert response.metadata["request_count"] == 3
    assert set(response.metadata["hierarchical_phase_receipts"]) == {
        "modify_field_selection",
        "modify_agent_selection",
    }


def test_v3_receipt_validation_fails_closed_on_phase_and_final_action_mismatch():
    actions = ("add_subgraph",)
    domains = {
        "add_subgraph": {
            "min_new_agents": 1,
            "max_new_agents": 1,
            "existing_agent_ids": [],
            "required_agent_fields": [
                "agent_id",
                "model_id",
                "contract",
                "role_family",
                "allowed_tools",
                "execution_mode",
            ],
            "model_ids": ["cheap-model"],
            "role_constraints": {
                "reasoner": {
                    "execution_modes": ["react"],
                    "allowed_tools": [["qa-retrieval"]],
                }
            },
            "endpoint_scope": {
                "relation_endpoint_sources": [
                    "existing_agent_ids",
                    "same_action_agent_ids",
                ],
                "output_agent_id_sources": [
                    "existing_agent_ids",
                    "same_action_agent_ids",
                ],
            },
        }
    }
    declaration = {
        "action": "add_subgraph",
        "agents": [
            {
                "agent_id": "reasoner",
                "model_id": "cheap-model",
                "contract": "align evidence",
                "role_family": "reasoner",
                "allowed_tools": ["qa-retrieval"],
                "execution_mode": "react",
            }
        ],
    }
    final_action = {
        **declaration,
        "relations": [],
        "output_agent_id": "reasoner",
    }
    action = AgentActionParser().parse(
        json.dumps(final_action, separators=(",", ":"))
    )
    domains_json = director_live_action_target_domains_json(actions, domains)
    schema_request = {
        "action_json_schema": (
            director_model_admissible_sampling_json_schema_text_v3(actions)
        ),
        "action_json_schema_version": (
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
        ),
        "action_schema_branch": director_model_admissible_schema_branch_v3(actions),
        "action_target_domains_json": domains_json,
        "action_target_domain_version": (
            DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
        ),
    }
    metadata = {
        "selected_action": "add_subgraph",
        "selected_add_agent_ids": ["reasoner"],
        "selected_modify_agent_id": None,
        "parameter_schema_branch": "add_subgraph",
        "request_count": 2,
        "hierarchical_phase_receipts": {
            "add_agent_declarations": {
                "text": json.dumps(declaration, separators=(",", ":"))
            }
        },
    }

    assert _validate_v3_hierarchical_action_receipt(
        action,
        metadata,
        schema_request,
    ) == {"add_agent_declarations"}
    with pytest.raises(ReceiptValidationError, match="request count"):
        _validate_v3_hierarchical_action_receipt(
            action,
            {**metadata, "request_count": 1},
            schema_request,
        )

    mismatched_action = AgentActionParser().parse(
        json.dumps(
            {
                **final_action,
                "agents": [
                    {**declaration["agents"][0], "contract": "changed contract"}
                ],
            },
            separators=(",", ":"),
        )
    )
    with pytest.raises(ReceiptValidationError, match="changed its sampled"):
        _validate_v3_hierarchical_action_receipt(
            mismatched_action,
            metadata,
            schema_request,
        )


def test_native_sglang_replays_legacy_v1_model_admissible_receipt():
    client = ScriptedSGLangClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    actions = ("modify_agent", "finish")
    schema = director_model_admissible_sampling_json_schema_text_v1(actions)
    branch = director_model_admissible_schema_branch_v1(actions)

    response = asyncio.run(
        client.propose(
            "legacy Canvas",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1
            ),
            action_schema_branch=branch,
        )
    )

    assert response.metadata["action_schema_branch"] == branch
    assert client.payloads[0]["sampling_params"]["json_schema"] == schema
    with pytest.raises(ValueError, match="strict schema for its branch"):
        client.request_payload(
            "cross-version mismatch",
            action_json_schema=schema,
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch=branch,
        )


def test_director_action_schema_preserves_legacy_add_and_relation_removal():
    branches = DIRECTOR_ACTION_JSON_SCHEMA["oneOf"]
    by_action = {
        branch["properties"]["action"]["const"]: branch
        for branch in branches
    }

    assert "add_agent" in by_action
    assert "anyOf" not in by_action["set_relation"]

    subgraph_profile = json.loads(
        director_action_json_schema_text(
            (
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            )
        )
    )
    admitted = {
        branch["properties"]["action"]["const"]
        for branch in subgraph_profile["oneOf"]
    }
    assert "add_subgraph" in admitted
    assert "add_agent" not in admitted


def test_sglang_sampling_schema_flattens_only_the_top_level_union():
    actions = (
        "add_subgraph",
        "modify_agent",
        "set_relation",
        "finish",
    )
    schema = json.loads(director_sglang_sampling_json_schema_text(actions))

    assert "oneOf" not in schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action"]
    assert schema["properties"]["action"] == {"enum": list(actions)}
    assert schema["properties"]["agents"]["items"]["additionalProperties"] is False
    assert schema["properties"]["relations"]["items"]["additionalProperties"] is False
    assert DIRECTOR_SGLANG_SAMPLING_SCHEMA_VERSION == (
        "agentgraph.sglang-flat-action-sampling-schema.v1"
    )

    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors({"action": "finish"}))
    assert list(
        validator.iter_errors({"action": "finish", "legacy_prompt": "forbidden"})
    )
    assert list(validator.iter_errors({"action": "delete_agent"}))
    assert list(
        validator.iter_errors(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": "solver",
                        "model_id": "model",
                        "contract": "solve",
                        "legacy_prompt": "forbidden",
                    }
                ],
                "relations": [],
            }
        )
    )


def test_sglang_sampling_schema_does_not_change_strict_action_schema_or_parser():
    strict_before = json.loads(DIRECTOR_ACTION_JSON_SCHEMA_TEXT)
    sampling = json.loads(
        director_sglang_sampling_json_schema_text(("add_subgraph", "finish"))
    )

    assert strict_before == DIRECTOR_ACTION_JSON_SCHEMA
    assert "oneOf" in strict_before
    assert sampling["properties"]["action"]["enum"] == [
        "add_subgraph",
        "finish",
    ]
    with pytest.raises(AgentActionParseError):
        AgentActionParser().parse('{"action":"finish","agents":[]}')


def test_native_sglang_receipt_submits_exact_transcript_messages():
    messages = (
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": "initial Canvas observation"},
        {"role": "assistant", "content": '{"action":"finish"}'},
        {"role": "user", "content": "current Canvas observation"},
    )
    client = ScriptedSGLangClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    asyncio.run(client.propose(encode_director_transcript(messages), seed=29))

    rendered_messages, template_kwargs = client.tokenizer.chat_calls[0]
    assert rendered_messages == list(messages)
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_sglang_client_rejects_disagreeing_token_receipts():
    client = MismatchedTokenClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    with pytest.raises(ReceiptValidationError, match="output_ids disagree"):
        asyncio.run(client.propose("prompt"))
    assert client.rollout_gate.in_flight == 0


def test_rollout_gate_pauses_drains_and_guards_policy_route():
    gate = RolloutGate(poll_interval_seconds=0.001)
    client = ScriptedSGLangClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        rollout_gate=gate,
    )
    with pytest.raises(RuntimeError, match="paused and drained"):
        client.update_policy_route(
            policy_version="next",
            adapter_name="theta_next",
            expected_server_weight_version="default",
        )

    asyncio.run(gate.acquire())
    gate.pause()
    completed = threading.Event()

    def drain():
        gate.drain(timeout_seconds=1.0)
        completed.set()

    worker = threading.Thread(target=drain)
    worker.start()
    time.sleep(0.02)
    assert not completed.is_set()
    gate.release()
    worker.join(timeout=1.0)
    assert completed.is_set()

    client.update_policy_route(
        policy_version="next",
        adapter_name="theta_next",
        expected_server_weight_version="default",
    )
    assert client.policy_version == "next"
    assert client.adapter_name == "theta_next"
    gate.resume()


def test_balanced_selector_takes_first_two_from_each_source():
    tasks = []
    for source in reversed(AGENTGRAPH_SMOKE_SOURCES):
        for index in range(3):
            tasks.append(_task(f"{source}:{index}", source))
    selected = select_balanced_tasks(tasks)
    assert len(selected) == 14
    assert [item.metadata["source"] for item in selected] == [
        source for source in AGENTGRAPH_SMOKE_SOURCES for _ in range(2)
    ]
    assert [item.task_id for item in selected[:2]] == ["HotpotQA:0", "HotpotQA:1"]


def test_balanced_selector_fails_instead_of_silently_underfilling():
    with pytest.raises(ValueError, match="insufficient smoke-training tasks"):
        select_balanced_tasks([_task()], per_source=2)


def test_collector_materializes_exact_finish_trajectory_and_evidence(tmp_path):
    registry = _registry()
    first_sample = (
        '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
        '"contract":"solve directly"}\n{"action":"finish"}'
    )
    second_sample = '{"action":"set_output","agent_id":"solver"} trailing'
    client = ScriptedSGLangClient(
        [
            first_sample,
            second_sample,
            '{"action":"finish"} trailing',
        ],
        policy_version=POLICY_VERSION,
        adapter_name="theta_live",
        expected_server_weight_version="default",
    )
    orchestrator = _orchestrator(registry, client, max_rounds=3)
    environment = AgentWorkflowEnv(registry, gateway=UnifiedMetadataGateway())
    evidence = EvidenceStore(tmp_path)
    collector = AgentGraphRolloutCollector(
        orchestrator,
        environment,
        _versions(),
        evidence,
    )

    def evaluator(task, final_answer, final_graph, runtime):
        assert task.task_id == "hotpotqa:first"
        assert final_answer == "final answer"
        assert final_graph["output_agent_id"] == "solver"
        assert runtime is not None
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
            "reason": "exact",
            "details": {"gold": "final answer", "trace_id": "eval-1"},
        }

    workflow_problem = (
        "What is the answer?\n\nExecution interface: return one admissible action."
    )
    trajectory = asyncio.run(
        collector.collect(
            _task(),
            0,
            evaluator,
            workflow_problem=workflow_problem,
        )
    )
    assert trajectory.explicit_finish is True
    assert trajectory.task.question == "What is the answer?"
    assert trajectory.termination_reason == "finish"
    assert trajectory.final_answer == "final answer"
    assert trajectory.grpo_eligible is True
    assert trajectory.group_id == f"hotpotqa:first:exploit:{POLICY_VERSION}"
    assert trajectory.evaluation.details["trace_id"] == "eval-1"
    assert len(trajectory.turns) == 3
    assert all(turn.receipt_verified for turn in trajectory.turns)
    assert all(turn.policy_version == POLICY_VERSION for turn in trajectory.turns)
    assert all(turn.policy_adapter == "theta_live" for turn in trajectory.turns)
    assert all(turn.server_weight_version == "default" for turn in trajectory.turns)
    assert all(turn.director_request_id for turn in trajectory.turns)
    assert all((turn.director_latency_ms or 0.0) >= 0.0 for turn in trajectory.turns)
    assert all(turn.director_attempt_count == 1 for turn in trajectory.turns)
    coordinate = ScientificSamplingCoordinate.from_value(
        trajectory.director_sampling["coordinate"]
    )
    assert [turn.director_generation_seed for turn in trajectory.turns] == [
        derive_generation_seed(
            base_seed=7,
            coordinate=coordinate,
            step_index=index,
            phase=GenerationPhase.ACTION,
        )
        for index in (1, 2, 3)
    ]
    assert trajectory.sampling_receipt_verified is True
    assert all(
        turn.executed_prefix_tokens < len(turn.output_token_ids)
        for turn in trajectory.turns
    )
    initial_user_message = client.tokenizer.chat_calls[0][0][1]["content"]
    assert "Execution interface: return one admissible action." in initial_user_message
    second_round_messages = client.tokenizer.chat_calls[1][0]
    third_round_messages = client.tokenizer.chat_calls[2][0]
    assert [item["role"] for item in second_round_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_round_messages[0]["content"] == DIRECTOR_SYSTEM_PROMPT
    assert second_round_messages[2]["content"] == first_sample.split("\n", 1)[0]
    assert [item["role"] for item in third_round_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert third_round_messages[4]["content"] == second_sample.removesuffix(" trailing")
    assert len(trajectory.turns[-1].executions) == 1
    assert trajectory.turns[-1].executions[0].output == "final answer"
    assert trajectory.turns[-1].runtime_summary["output_agent_id"] == "solver"
    assert trajectory.turns[-1].runtime_summary["outputs"] == {"solver": "final answer"}
    assert trajectory.turns[-1].runtime_summary["block_completion_order"] == [
        ["solver"]
    ]
    request_receipt = trajectory.turns[-1].executions[0].metadata["request"]
    assert request_receipt["is_output_agent"] is True
    assert request_receipt["communication_condition"] == "normal"
    assert request_receipt["rendered_messages"][0]["role"] == "system"
    assert request_receipt["rendered_messages"][1]["role"] == "user"
    response_receipt = trajectory.turns[-1].executions[0].metadata["response"]
    assert response_receipt["execution_mode"] == "react"
    assert response_receipt["react_turns_used"] == 2
    assert response_receipt["tool_calls"] == 1
    assert response_receipt["tool_receipts"][0]["tool_id"] == "qa-retrieval.search"
    assert response_receipt["react_trace"][1]["observation_status"] == "completed"
    assert response_receipt["model_calls"][0]["request_id"].endswith(":react:1")
    assert response_receipt["environment_id"] == "webshop:test-1"
    assert response_receipt["task_family"] == "WebShop"
    assert response_receipt["environment_revision"] == 1
    assert response_receipt["environment_reset_receipt"]["observation"] == (
        "initial observation"
    )
    assert response_receipt["environment_receipts"][0]["state_advanced"] is True
    assert response_receipt["environment_terminal"] is True
    assert response_receipt["environment_turns_used"] == 1
    assert response_receipt["environment_steps"] == 1
    assert response_receipt["evaluator_environment_trace"][0]["reward"] == 1.0
    assert response_receipt["provider_id"] == "vector"
    assert response_receipt["model_id"] == "cheap-model"
    assert response_receipt["prompt_tokens"] == 12
    assert response_receipt["completion_tokens"] == 3
    assert response_receipt["total_tokens"] == 15
    assert response_receipt["latency_ms"] == 4.0
    assert response_receipt["attempt_count"] == 3
    assert trajectory.turns[-1].executions[0].input_tokens == 12
    assert trajectory.turns[-1].executions[0].output_tokens == 3
    assert trajectory.turns[-1].executions[0].latency_ms == 4.0
    assert "opaque_runtime_object" not in response_receipt
    assert trajectory.turns[-1].runtime_summary["communication_condition"] == "normal"
    output_metadata = trajectory.turns[-1].runtime_summary["output_metadata"]["solver"]
    assert output_metadata["tool_receipts"] == response_receipt["tool_receipts"]
    assert output_metadata["evaluator_environment_trace"] == response_receipt[
        "evaluator_environment_trace"
    ]
    assert output_metadata["attempt_count"] == 3
    assert "opaque_runtime_object" not in output_metadata
    json.dumps(trajectory.to_dict())
    assert len(evidence.snapshots) == 3
    assert len(evidence.trajectories) == 1


def test_collector_uses_state_conditioned_schema_on_every_progressive_turn():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            (
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"solver","model_id":"cheap-model",'
                '"contract":"solve directly"}],"relations":[],'
                '"output_agent_id":"solver"}'
            ),
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(
            registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
        ),
        AgentWorkflowEnv(
            registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        ),
        _versions(),
    )

    def evaluator(task, final_answer, final_graph, runtime):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))

    assert trajectory.explicit_finish is True
    assert [
        turn.runtime_summary["director_action_schema_branch"]
        for turn in trajectory.turns
    ] == ["add_subgraph", "finish"]
    assert all(
        turn.runtime_summary["director_action_schema_version"]
        == DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
        for turn in trajectory.turns
    )
    assert client.payloads[0]["sampling_params"]["json_schema"] == (
        director_state_conditioned_sampling_json_schema_text("add_subgraph")
    )
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_state_conditioned_sampling_json_schema_text("finish")
    )


def test_collector_records_model_admissible_schema_on_every_canvas_turn():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            (
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"solver","model_id":"cheap-model",'
                '"contract":"solve directly"}],"relations":[],'
                '"output_agent_id":"solver"}'
            ),
            '{"action":"finish"}',
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(
            registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
        ),
        AgentWorkflowEnv(
            registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        ),
        _versions(),
    )

    def evaluator(task, final_answer, final_graph, runtime):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))

    assert trajectory.explicit_finish is True
    for turn in trajectory.turns:
        branch = turn.runtime_summary["director_action_schema_branch"]
        actions = director_actions_from_admissible_schema_branch(branch)
        assert turn.runtime_summary["director_action_schema_version"] == (
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
        )
    assert client.payloads[0]["sampling_params"]["json_schema"] == (
        director_state_conditioned_sampling_json_schema_text("add_subgraph")
    )
    second_actions = director_actions_from_admissible_schema_branch(
        trajectory.turns[1].runtime_summary["director_action_schema_branch"]
    )
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_model_admissible_sampling_json_schema_text(second_actions)
    )
    assert client.payloads[2]["sampling_params"]["json_schema"] == (
        director_state_conditioned_sampling_json_schema_text("finish")
    )
    assert director_actions_from_admissible_schema_branch(
        trajectory.turns[0].runtime_summary["director_action_schema_branch"]
    ) == ("add_subgraph",)
    assert "finish" in director_actions_from_admissible_schema_branch(
        trajectory.turns[1].runtime_summary["director_action_schema_branch"]
    )
    assert trajectory.turns[0].runtime_summary["director_action_decoding"] == {
        "strategy": "hierarchical_json_schema",
        "selected_action": "add_subgraph",
        "selected_modify_field": None,
        "parameter_schema_branch": "add_subgraph",
        "request_count": 1,
        "phase_receipts": {},
    }
    finish_decoding = trajectory.turns[1].runtime_summary[
        "director_action_decoding"
    ]
    assert finish_decoding["selected_action"] == "finish"
    assert finish_decoding["request_count"] == 2
    assert set(finish_decoding["phase_receipts"]) == {"action_selection"}


def test_collector_does_not_duplicate_reused_progressive_execution():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
            '"contract":"solve directly"}',
            '{"action":"set_output","agent_id":"solver"}',
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(
            registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        ),
        _versions(),
    )

    def evaluator(task, final_answer, final_graph, runtime):
        assert final_answer == "final answer"
        assert runtime is not None
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
            "reason": "exact",
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))

    assert len(trajectory.turns[1].executions) == 1
    assert trajectory.turns[1].execution_reused is False
    assert trajectory.turns[2].executions == ()
    assert trajectory.turns[2].execution_reused is True


def test_collector_persists_public_failure_receipts_before_canvas_repair():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            (
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"solver","model_id":"cheap-model",'
                '"contract":"solve directly"}],"relations":[],'
                '"output_agent_id":"solver"}'
            ),
            (
                '{"action":"modify_agent","agent_id":"solver",'
                '"contract":"solve after execution feedback"}'
            ),
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(
            registry,
            gateway=FailOnceReceiptGateway(),
            execute_on_edit=True,
        ),
        _versions(),
    )

    def evaluator(task, final_answer, final_graph, runtime):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))

    failed_turn = trajectory.turns[0]
    self_receipt = failed_turn.runtime_summary["failure_records"][0]
    assert failed_turn.runtime_summary["execution_status"] == "failed"
    assert self_receipt["agent_id"] == "solver"
    assert self_receipt["metadata"]["tool_receipts"][0]["tool_id"] == "qa.search"
    assert failed_turn.runtime_summary["unresolved_dirty_agent_ids"] == ["solver"]
    assert trajectory.explicit_finish is True


def test_collector_refreshes_skill_priors_at_each_graph_stage():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
            '"contract":"solve directly"}',
            '{"action":"set_output","agent_id":"solver"}',
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    observed_stages = []

    def skill_provider(task, environment, versions):
        assert task.task_id == "hotpotqa:first"
        assert versions == _versions()
        stage = (
            "empty_graph"
            if not environment.graph.nodes
            else (
                "before_final_answer"
                if environment.graph.output_agent_id is not None
                else "construction"
            )
        )
        observed_stages.append(stage)
        return ({"skill_id": f"validated-{stage}"},)

    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(registry, gateway=FakeGateway(), execute_on_edit=True),
        _versions(),
        skill_provider=skill_provider,
        active_skill_provider=lambda task, environment, versions: (
            "validated-before_final_answer",
            "validated-construction",
            "validated-empty_graph",
        ),
    )

    def evaluator(task, final_answer, final_graph, runtime):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"f1": 1.0},
        }

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))

    assert trajectory.explicit_finish is True
    assert observed_stages == [
        "empty_graph",
        "construction",
        "before_final_answer",
    ]
    assert "validated-empty_graph" in client.tokenizer.chat_calls[0][0][-1]["content"]
    assert "validated-construction" in client.tokenizer.chat_calls[1][0][-1]["content"]
    assert "validated-before_final_answer" in client.tokenizer.chat_calls[2][0][-1]["content"]
    assert [turn.retrieved_skill_ids for turn in trajectory.turns] == [
        ("validated-empty_graph",),
        ("validated-construction",),
        ("validated-before_final_answer",),
    ]
    assert [turn.visible_skill_ids for turn in trajectory.turns] == [
        ("validated-empty_graph",),
        ("validated-construction",),
        ("validated-before_final_answer",),
    ]
    assert trajectory.active_skill_ids == (
        "validated-before_final_answer",
        "validated-construction",
        "validated-empty_graph",
    )
    assert trajectory.retrieved_skill_ids == ("validated-empty_graph",)
    assert trajectory.invoked_skill_ids == ()


def test_collector_returns_complete_max_rounds_trajectory():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
            '"contract":"solve"}'
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=1, rollout_ordinal=1),
        AgentWorkflowEnv(registry, gateway=FakeGateway()),
        _versions(),
    )
    observed = {}

    class Outcome:
        evaluator_version = EVALUATOR_VERSION
        valid = True
        reward = 0.0
        metrics = {"finished": 0.0}
        reason = "maximum rounds"
        details = {"terminal_state": "max_rounds"}

    async def evaluator(task, final_answer, final_graph, runtime):
        observed.update(
            task=task,
            final_answer=final_answer,
            final_graph=final_graph,
            runtime=runtime,
        )
        return Outcome()

    trajectory = asyncio.run(collector.collect(_task(), 1, evaluator))
    assert trajectory.explicit_finish is False
    assert trajectory.termination_reason == "max_rounds"
    assert trajectory.final_answer is None
    assert trajectory.evaluation.reward == 0.0
    assert trajectory.evaluation.details == {"terminal_state": "max_rounds"}
    assert trajectory.terminal_failure is True
    assert trajectory.grpo_eligible is True
    assert len(trajectory.turns) == 1
    assert observed["runtime"] is None
    assert observed["final_graph"]["nodes"][0]["id"] == "solver"


def test_collector_allows_explicit_heldout_split_without_grpo_admission():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
            '"contract":"solve"}',
            '{"action":"set_output","agent_id":"solver"}',
            '{"action":"finish"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(registry, gateway=FakeGateway()),
        _versions(),
        expected_task_split="validation",
    )

    def evaluator(task, final_answer, final_graph, runtime):
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "valid": True,
            "reward": 1.0,
            "metrics": {"token_f1": 1.0},
            "reason": "exact",
        }

    trajectory = asyncio.run(collector.collect(_task(split="validation"), 0, evaluator))

    assert trajectory.task.split == "validation"
    assert trajectory.explicit_finish is True
    assert trajectory.grpo_eligible is False
