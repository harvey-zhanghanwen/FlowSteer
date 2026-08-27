from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from src.interactive.agent_action_parser import AgentActionParser
from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    director_live_action_target_domains_json,
    director_live_action_parameter_json_schema_text,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_add_subgraph_relation_candidates,
    director_live_add_subgraph_role_selection_from_text,
    director_live_add_subgraph_role_selection_json_schema_text,
    director_live_modify_agent_field_selector_json_schema_text,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_schema_branch,
    director_model_admissible_schema_branch_v3,
    director_state_conditioned_sampling_json_schema_text,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.persistence import EvidenceStore
from src.interactive.react_execution import ReactExecutionError
from src.interactive.records import TaskRecord
from src.interactive.rollout_collector import (
    AGENTGRAPH_SMOKE_SOURCES,
    AgentGraphRolloutCollector,
    ReceiptValidationError,
    RolloutGate,
    SGLangReceiptDirectorClient,
    select_balanced_tasks,
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


class FailOnceReceiptGateway(FakeGateway):
    def __init__(self) -> None:
        self.failed = False

    async def generate(self, request):
        if not self.failed:
            self.failed = True
            raise ReactExecutionError(
                "bounded execution exhausted",
                metadata={
                    "react_trace": (
                        {"turn": 1, "observation_status": "success"},
                    ),
                    "tool_receipts": (
                        {"tool_id": "qa.search", "success": True},
                    ),
                    "model_calls": (
                        {"turn": 1, "request_id": request.request_id},
                    ),
                },
            )
        return await super().generate(request)


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
                "opaque_runtime_object": object(),
            },
        )


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


def test_native_sglang_receipt_uses_real_input_ids_and_separates_versions():
    text = 'preface {"action":"finish"}\nunused explanation'
    client = ScriptedSGLangClient(
        [text],
        policy_version=POLICY_VERSION,
        adapter_name="theta_live",
        expected_server_weight_version="default",
        base_url="http://127.0.0.1:8015/v1",
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
    assert response.metadata["policy_version"] == POLICY_VERSION
    assert response.metadata["server_weight_version"] == "default"
    assert response.metadata["adapter_name"] == "theta_live"
    assert response.metadata["latency_ms"] >= 0.0
    assert response.metadata["attempt_count"] == 1
    assert response.metadata["generation_seed"] == 23
    assert len(response.metadata["output_token_ids"]) == len(
        response.metadata["behavior_log_probs"]
    )

    action = AgentActionParser().parse(text)
    consumed = client.executed_prefix_tokens(response, action)
    assert consumed == action.consumed_end
    assert consumed < len(response.metadata["output_token_ids"])


def test_sglang_client_rejects_disagreeing_token_receipts():
    client = MismatchedTokenClient(
        ['{"action":"finish"}'],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    with pytest.raises(ReceiptValidationError, match="output_ids disagree"):
        asyncio.run(client.propose("prompt"))
    assert client.rollout_gate.in_flight == 0


def test_sglang_client_uses_two_stage_model_admissible_action_schema():
    actions = ("add_subgraph", "add_agent")
    client = ScriptedSGLangClient(
        [
            '{"action":"add_subgraph"}',
            '{"action":"add_subgraph","agents":[{"agent_id":"searcher",'
            '"model_id":"cheap-model","contract":"retrieve evidence",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]}],'
            '"relations":[]}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "prompt",
            seed=23,
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text(actions)
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
            action_schema_branch=director_model_admissible_schema_branch(actions),
        )
    )

    assert len(client.payloads) == 2
    assert client.payloads[0]["sampling_params"]["json_schema"] == (
        director_model_admissible_sampling_json_schema_text(actions)
    )
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_state_conditioned_sampling_json_schema_text("add_subgraph")
    )
    assert response.metadata["selected_action"] == "add_subgraph"
    assert response.metadata["action_decoding_strategy"] == (
        "hierarchical_json_schema"
    )
    assert response.metadata["base_prompt_text"] == "prompt"
    assert response.metadata["request_count"] == 2
    assert response.metadata["hierarchical_phase_receipts"][
        "action_selection"
    ]["receipt_verified"] is True
    parsed = AgentActionParser().parse(response.text)
    assert parsed.action_type.value == "add_subgraph"


def test_sglang_client_v3_binds_add_subgraph_to_live_domains():
    actions = ("add_subgraph", "add_agent")
    domains = {
        "registered_execution_profiles": [
            {"execution_mode": "reasoning", "allowed_tools": []},
            {
                "execution_mode": "react",
                "allowed_tools": ["hotpotqa.qa_memory"],
            },
        ],
        "finish_admissibility": {
            "admissible": False,
            "reason": "graph has no Output Agent",
        },
        "add_subgraph": {
            "model_ids": ["cheap-model"],
            "execution_profiles": [
                {"execution_mode": "reasoning", "allowed_tools": []},
                {
                    "execution_mode": "react",
                    "allowed_tools": ["hotpotqa.qa_memory"],
                },
            ],
            "existing_agent_ids": [],
            "max_new_agents": 3,
        },
        "add_agent": {
            "model_ids": ["cheap-model"],
            "execution_profiles": [
                {"execution_mode": "reasoning", "allowed_tools": []},
                {
                    "execution_mode": "react",
                    "allowed_tools": ["hotpotqa.qa_memory"],
                },
            ],
        },
    }
    client = ScriptedSGLangClient(
        [
            '{"action":"add_subgraph"}',
            '{"action":"add_subgraph","agents":[{"agent_id":"searcher",'
            '"model_id":"cheap-model","contract":"retrieve evidence",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]},'
            '{"agent_id":"answerer","model_id":"cheap-model",'
            '"contract":"answer from upstream evidence",'
            '"execution_mode":"reasoning"}]}',
            '{"action":"add_subgraph","agents":[{"agent_id":"searcher",'
            '"model_id":"cheap-model","contract":"retrieve evidence",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]},'
            '{"agent_id":"answerer","model_id":"cheap-model",'
            '"contract":"answer from upstream evidence",'
            '"execution_mode":"reasoning","allowed_tools":[]}],"relations":['
            '{"source_id":"searcher","target_id":"answerer",'
            '"source_to_target":true,"target_to_source":false}],'
            '"output_agent_id":"answerer"}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "prompt",
            seed=23,
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text(actions)
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=director_model_admissible_schema_branch_v3(
                actions
            ),
            action_target_domains_json=director_live_action_target_domains_json(
                actions,
                domains,
            ),
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )

    assert len(client.payloads) == 3
    assert client.payloads[1]["sampling_params"]["json_schema"] == (
        director_live_add_subgraph_agent_declarations_json_schema_text(domains)
    )
    assert response.metadata["selected_add_agent_ids"] == [
        "searcher",
        "answerer",
    ]
    assert response.metadata["action_target_domain_version"] == (
        DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
    )
    parsed = AgentActionParser().parse(response.text)
    assert parsed.relations[0].source_id == "searcher"
    assert parsed.relations[0].target_id == "answerer"


def test_hotpotqa_v3_parameter_schemas_use_exact_relation_and_modify_deltas():
    profiles = [
        {"execution_mode": "reasoning", "allowed_tools": []},
        {
            "execution_mode": "react",
            "allowed_tools": ["hotpotqa.qa_memory"],
        },
    ]
    role_constraints = {
        "evidence_retriever": {
            "execution_profiles": [profiles[1]],
        },
        "reasoner": {"execution_profiles": [profiles[0]]},
        "verifier": {"execution_profiles": [profiles[0]]},
        "format": {"execution_profiles": [profiles[0]]},
        "repair": {"execution_profiles": [profiles[1]]},
    }
    add_domain = {
        "semantic_protocol": "hotpotqa.qa_memory.worker_lineage.v1",
        "model_ids": ["cheap-model"],
        "execution_profiles": profiles,
        "existing_agent_ids": [],
        "existing_agents": [],
        "min_new_agents": 1,
        "max_new_agents": 3,
        "required_agent_fields": [
            "agent_id",
            "model_id",
            "contract",
            "role_family",
            "execution_mode",
            "allowed_tools",
        ],
        "output_role_family": "format",
        "role_constraints": role_constraints,
        "admitted_new_role_families": [
            "evidence_retriever",
            "reasoner",
            "verifier",
            "format",
        ],
    }
    add_agents = [
        {
            "agent_id": "node_1",
            "model_id": "cheap-model",
            "contract": "retrieve evidence",
            "role_family": "evidence_retriever",
            "execution_mode": "react",
            "allowed_tools": ["hotpotqa.qa_memory"],
        },
        {
            "agent_id": "node_2",
            "model_id": "cheap-model",
            "contract": "derive answer",
            "role_family": "reasoner",
            "execution_mode": "reasoning",
            "allowed_tools": [],
        },
        {
            "agent_id": "node_3",
            "model_id": "cheap-model",
            "contract": "format answer",
            "role_family": "format",
            "execution_mode": "reasoning",
            "allowed_tools": [],
        },
    ]
    domains = {
        "add_subgraph": add_domain,
        "set_relation": {
            "source_agent_ids": ["verifier", "formatter"],
            "target_agent_ids": ["verifier", "formatter"],
            "endpoints_must_differ": True,
            "candidates": [
                {
                    "source_id": "verifier",
                    "target_id": "formatter",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            ],
        },
        "modify_agent": {
            "mutable_fields": [
                "model_id",
                "contract",
                "artifact_type",
                "completion_condition",
            ],
            "per_agent_candidates": [
                {
                    "agent_id": "retriever",
                    "mutable_fields": ["model_id", "contract"],
                    "current_values": {
                        "model_id": "cheap-model",
                        "contract": "retrieve evidence",
                    },
                    "discrete_value_domains": {
                        "model_id": ["alternate-model"],
                    },
                }
            ],
        },
    }

    role_schema = json.loads(
        director_live_add_subgraph_role_selection_json_schema_text(domains)
    )
    first_role_branch = role_schema["properties"]["agents"]["oneOf"][0]
    assert first_role_branch["prefixItems"][0]["properties"]["agent_id"] == {
        "const": "node_1"
    }
    selected_roles = director_live_add_subgraph_role_selection_from_text(
        '{"action":"add_subgraph","agents":['
        '{"agent_id":"node_1","role_family":"evidence_retriever"},'
        '{"agent_id":"node_2","role_family":"reasoner"},'
        '{"agent_id":"node_3","role_family":"format"}]}<|endoftext|>',
        domains,
    )
    declaration_schema = json.loads(
        director_live_add_subgraph_agent_declarations_json_schema_text(
            domains,
            selected_agent_roles=selected_roles,
        )
    )
    declaration_agents = declaration_schema["properties"]["agents"]["oneOf"][0]
    assert declaration_agents["prefixItems"][0]["anyOf"][0]["properties"][
        "agent_id"
    ] == {"const": "node_1"}
    domains_with_existing = json.loads(json.dumps(domains))
    domains_with_existing["add_subgraph"]["existing_agent_ids"] = ["node_1"]
    domains_with_existing["add_subgraph"]["existing_agents"] = [
        {"agent_id": "node_1", "role_family": "evidence_retriever"}
    ]
    with pytest.raises(ValueError, match="reused or changed"):
        director_live_add_subgraph_role_selection_from_text(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","role_family":"reasoner"}]}',
            domains_with_existing,
        )
    with pytest.raises(ValueError, match="outside the live domain"):
        director_live_add_subgraph_agent_declarations_from_text(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","model_id":"cheap-model",'
            '"contract":"retrieve","role_family":"evidence_retriever",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]},'
            '{"agent_id":"node_1","model_id":"cheap-model",'
            '"contract":"reason","role_family":"reasoner",'
            '"execution_mode":"reasoning","allowed_tools":[]}]}',
            domains,
        )

    candidates = director_live_add_subgraph_relation_candidates(
        domains,
        add_agents,
    )
    assert candidates
    assert all(candidate["source_id"] != "node_3" for candidate in candidates)
    assert not any(candidate["target_id"] == "node_3" for candidate in candidates)
    add_schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "add_subgraph",
            domains,
            add_agents=add_agents,
        )
    )
    assert add_schema["properties"]["relations"]["maxItems"] == 1
    assert add_schema["properties"]["output_agent_id"] == {"const": None}
    relation_branches = add_schema["properties"]["relations"]["items"]["anyOf"]
    assert all(
        branch["properties"]["source_to_target"].get("const") is not None
        for branch in relation_branches
    )

    relation_schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "set_relation",
            domains,
        )
    )
    assert relation_schema["oneOf"][0]["properties"] == {
        "action": {"const": "set_relation"},
        "source_id": {"const": "verifier"},
        "target_id": {"const": "formatter"},
        "source_to_target": {"const": True},
        "target_to_source": {"const": False},
    }

    field_schema = json.loads(
        director_live_modify_agent_field_selector_json_schema_text(domains)
    )
    fields = field_schema["properties"]["field"]["enum"]
    assert "role_family" not in fields
    assert "execution_mode" not in fields
    assert "allowed_tools" not in fields
    modify_schema = json.loads(
        director_live_action_parameter_json_schema_text(
            "modify_agent",
            domains,
            modify_field="model_id",
        )
    )
    assert modify_schema["oneOf"][0]["properties"]["agent_id"] == {
        "const": "retriever"
    }
    assert modify_schema["oneOf"][0]["properties"]["model_id"] == {
        "enum": ["alternate-model"]
    }


def test_sglang_client_hotpotqa_add_uses_role_first_canvas_assigned_ids():
    actions = ("add_subgraph",)
    domains = {
        "add_subgraph": {
            "semantic_protocol": "hotpotqa.qa_memory.worker_lineage.v1",
            "model_ids": ["cheap-model"],
            "execution_profiles": [
                {"execution_mode": "reasoning", "allowed_tools": []},
                {
                    "execution_mode": "react",
                    "allowed_tools": ["hotpotqa.qa_memory"],
                },
            ],
            "existing_agent_ids": [],
            "existing_agents": [],
            "min_new_agents": 1,
            "max_new_agents": 1,
            "required_agent_fields": [
                "agent_id",
                "model_id",
                "contract",
                "role_family",
                "execution_mode",
                "allowed_tools",
            ],
            "output_role_family": "format",
            "role_constraints": {
                "evidence_retriever": {
                    "execution_profiles": [
                        {
                            "execution_mode": "react",
                            "allowed_tools": ["hotpotqa.qa_memory"],
                        }
                    ]
                },
                "format": {
                    "execution_profiles": [
                        {"execution_mode": "reasoning", "allowed_tools": []}
                    ]
                },
            },
            "admitted_new_role_families": ["evidence_retriever"],
        }
    }
    client = ScriptedSGLangClient(
        [
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","role_family":"evidence_retriever"}]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","model_id":"cheap-model",'
            '"contract":"retrieve evidence","role_family":"evidence_retriever",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]}]}',
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","model_id":"cheap-model",'
            '"contract":"retrieve evidence","role_family":"evidence_retriever",'
            '"execution_mode":"react","allowed_tools":["hotpotqa.qa_memory"]}],'
            '"relations":[],"output_agent_id":null}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )

    response = asyncio.run(
        client.propose(
            "prompt",
            seed=23,
            action_json_schema=(
                director_model_admissible_sampling_json_schema_text(actions)
            ),
            action_json_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            action_schema_branch=director_model_admissible_schema_branch_v3(
                actions
            ),
            action_target_domains_json=director_live_action_target_domains_json(
                actions,
                domains,
            ),
            action_target_domain_version=(
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        )
    )

    assert len(client.payloads) == 3
    role_schema = json.loads(client.payloads[0]["sampling_params"]["json_schema"])
    assert role_schema["properties"]["agents"]["oneOf"][0]["prefixItems"][0][
        "properties"
    ]["agent_id"] == {"const": "node_1"}
    declaration_schema = json.loads(
        client.payloads[1]["sampling_params"]["json_schema"]
    )
    assert declaration_schema["properties"]["agents"]["oneOf"][0][
        "prefixItems"
    ][0]["anyOf"][0]["properties"]["agent_id"] == {"const": "node_1"}
    assert response.metadata["selected_add_agent_ids"] == ["node_1"]
    assert response.metadata["selected_add_agent_roles"] == [
        {"agent_id": "node_1", "role_family": "evidence_retriever"}
    ]
    assert response.metadata["action_decoding_strategy"] == (
        "hierarchical_json_schema_role_first_add_v1"
    )


def test_collector_forwards_v3_live_action_schema_and_persists_receipt():
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent"}',
            '{"action":"add_agent","execution_mode":"reasoning"}',
            '{"action":"add_agent","agent_id":"solver",'
            '"model_id":"cheap-model","contract":"solve directly",'
            '"execution_mode":"reasoning","allowed_tools":[]}',
        ],
        policy_version=POLICY_VERSION,
        expected_server_weight_version="default",
    )
    orchestrator = AgentGraphOrchestrator(
        registry,
        client,
        max_rounds=1,
        seed=7,
        sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
        sampling_action_schema_version=(
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
        ),
    )
    collector = AgentGraphRolloutCollector(
        orchestrator,
        AgentWorkflowEnv(registry, gateway=FakeGateway()),
        _versions(),
    )

    trajectory = asyncio.run(
        collector.collect(
            _task(),
            0,
            lambda *args: {
                "evaluator_version": EVALUATOR_VERSION,
                "valid": True,
                "reward": 0.0,
                "metrics": {"finished": 0.0},
                "reason": "maximum rounds",
            },
        )
    )

    assert len(client.payloads) == 3
    assert all(
        "json_schema" in payload["sampling_params"]
        for payload in client.payloads
    )
    receipt = trajectory.turns[0].runtime_summary
    assert receipt["director_action_schema_version"] == (
        DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
    )
    assert receipt["director_action_target_domain_version"] == (
        DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
    )
    assert receipt["director_action_decoding"]["strategy"] == (
        "hierarchical_json_schema"
    )
    assert receipt["director_action_decoding"]["selected_action"] == (
        "add_agent"
    )
    assert receipt["director_action_decoding"]["selected_execution_mode"] == (
        "reasoning"
    )
    assert receipt["director_action_decoding"]["selected_allowed_tools"] == []


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
    client = ScriptedSGLangClient(
        [
            '{"action":"add_agent","agent_id":"solver","model_id":"cheap-model",'
            '"contract":"solve directly"}\n{"action":"finish"}',
            '{"action":"set_output","agent_id":"solver"} trailing',
            '{"action":"finish"} trailing',
        ],
        policy_version=POLICY_VERSION,
        adapter_name="theta_live",
        expected_server_weight_version="default",
    )
    orchestrator = AgentGraphOrchestrator(registry, client, max_rounds=3, seed=7)
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

    trajectory = asyncio.run(collector.collect(_task(), 0, evaluator))
    assert trajectory.explicit_finish is True
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
    assert [turn.director_generation_seed for turn in trajectory.turns] == [7, 8, 9]
    assert all(
        turn.executed_prefix_tokens < len(turn.output_token_ids)
        for turn in trajectory.turns
    )
    assert len(trajectory.turns[-1].executions) == 1
    assert trajectory.turns[-1].executions[0].output == "final answer"
    assert trajectory.turns[-1].runtime_summary["output_agent_id"] == "solver"
    assert trajectory.turns[-1].runtime_summary["outputs"] == {
        "solver": "final answer"
    }
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
    assert response_receipt["prompt_tokens"] == 12
    assert response_receipt["completion_tokens"] == 3
    assert response_receipt["total_tokens"] == 15
    assert response_receipt["latency_ms"] == 4.0
    assert response_receipt["attempt_count"] == 3
    execution = trajectory.turns[-1].executions[0]
    assert execution.input_tokens == 12
    assert execution.output_tokens == 3
    assert execution.latency_ms == 4.0
    assert "opaque_runtime_object" not in response_receipt
    output_metadata = trajectory.turns[-1].runtime_summary["output_metadata"]["solver"]
    assert output_metadata["tool_receipts"] == response_receipt["tool_receipts"]
    assert output_metadata["attempt_count"] == 3
    assert "opaque_runtime_object" not in output_metadata
    json.dumps(trajectory.to_dict())
    assert len(evidence.snapshots) == 3
    assert len(evidence.trajectories) == 1


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
        AgentGraphOrchestrator(registry, client, max_rounds=3),
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
        AgentGraphOrchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(
            registry,
            gateway=FailOnceReceiptGateway(),
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

    failed_turn = trajectory.turns[0]
    failure = failed_turn.runtime_summary["failure_records"][0]
    assert failed_turn.runtime_summary["execution_status"] == "failed"
    assert failure["agent_id"] == "solver"
    assert failure["error_type"] == "ReactExecutionError"
    assert failure["metadata"]["react_trace"][0]["observation_status"] == (
        "success"
    )
    assert failure["metadata"]["tool_receipts"][0]["tool_id"] == "qa.search"
    assert failure["metadata"]["model_calls"][0]["request_id"].endswith(
        ":solver:single"
    )
    assert trajectory.explicit_finish is True
    json.dumps(trajectory.to_dict())


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
        AgentGraphOrchestrator(registry, client, max_rounds=1),
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
        AgentGraphOrchestrator(registry, client, max_rounds=3),
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

    trajectory = asyncio.run(
        collector.collect(_task(split="validation"), 0, evaluator)
    )

    assert trajectory.task.split == "validation"
    assert trajectory.explicit_finish is True
    assert trajectory.grpo_eligible is False
