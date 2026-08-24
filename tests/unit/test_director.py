from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentResponse, AgentRuntime
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION,
    _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
)
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
    DIRECTOR_PROMPT_VERSION,
    DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION,
    DIRECTOR_SYSTEM_PROMPT,
    HOTPOTQA_DIRECTOR_PROMPT_VERSION,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,
    HOTPOTQA_SEMANTIC_PROTOCOL,
    QA_DIRECTOR_PROMPT_VERSION,
    QA_DIRECTOR_SYSTEM_PROMPT_V1,
    QA_DIRECTOR_SYSTEM_PROMPT_V2,
    QA_DIRECTOR_SYSTEM_PROMPT_V3,
    QA_DIRECTOR_SYSTEM_PROMPT_V4,
    QA_DIRECTOR_SYSTEM_PROMPT_V5,
    QA_DIRECTOR_SYSTEM_PROMPT_V6,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V4,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V5,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V2,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V3,
    LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
    LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
    PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
    DirectorResponse,
    OpenAIDirectorClient,
    decode_director_transcript,
    director_actions_from_admissible_schema_branch,
    director_action_json_schema_text,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_sampling_json_schema_text_v1,
    director_model_admissible_sampling_json_schema_text_v3,
    director_model_admissible_schema_branch,
    director_model_admissible_schema_branch_v1,
    director_model_admissible_schema_branch_v3,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_add_subgraph_role_selection_from_text,
    director_live_add_subgraph_role_selection_json_schema_text,
    director_live_add_subgraph_relation_candidates,
    director_live_action_parameter_json_schema_text,
    director_live_action_target_domains_json,
    director_live_modify_agent_selector_json_schema_text,
    director_live_relation_candidate_selector_json_schema_text,
    director_modify_agent_field_sampling_json_schema_text,
    director_modify_agent_field_selector_json_schema_text,
    director_system_prompt_for_version,
    director_state_conditioned_sampling_json_schema_text,
    encode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)


class ScriptedDirector:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []
        self.seeds = []
        self.schema_requests = []

    async def propose(
        self,
        prompt,
        *,
        seed=None,
        action_json_schema=None,
        action_json_schema_version=None,
        action_schema_branch=None,
        action_target_domains_json=None,
        action_target_domain_version=None,
    ):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        self.schema_requests.append(
            {
                "action_json_schema": action_json_schema,
                "action_json_schema_version": action_json_schema_version,
                "action_schema_branch": action_schema_branch,
                "action_target_domains_json": action_target_domains_json,
                "action_target_domain_version": action_target_domain_version,
            }
        )
        return DirectorResponse(self.actions.pop(0), {"policy_version": "test"})


class FakeGateway:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(f"answer from {request.agent.id}")


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", endpoint="http://local/v1")],
        [
            ModelSpec(
                "qwen",
                "provider",
                cheap_weight=10,
                fast_weight=10,
                metadata={
                    "family": "qwen",
                    "profile": "text_qa",
                    "text_qa_canary": "passed",
                    "max_tokens": "512",
                },
            ),
            ModelSpec("other", "provider", cheap_weight=1, fast_weight=1),
        ],
    )


def transcript_messages(prompt: str) -> tuple[dict[str, str], ...]:
    messages = decode_director_transcript(prompt)
    if messages is None:
        raise AssertionError("expected a canonical Director transcript")
    return tuple(dict(message) for message in messages)


def observation_payload(message: dict[str, str]) -> dict[str, object]:
    if message["role"] != "user":
        raise AssertionError("Canvas observation must be a user message")
    heading, separator, raw_payload = message["content"].partition("\n\n")
    if not separator or not (
        heading.startswith("Canvas observation.")
        or heading.startswith("Historical Canvas public feedback.")
    ):
        raise AssertionError("Canvas observation message has no JSON payload")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise AssertionError("Canvas observation payload must be an object")
    return payload


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_canonical_transcript_remains_decodable(self) -> None:
        legacy = encode_director_transcript(
            (
                {
                    "role": "system",
                    "content": LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
                },
                {"role": "user", "content": "legacy Canvas observation"},
            )
        )

        decoded = decode_director_transcript(legacy)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(LEGACY_DIRECTOR_SYSTEM_PROMPT_V8, decoded[0]["content"])
        self.assertEqual(
            "agentgraph.director.minimal-neutral.v10",
            DIRECTOR_PROMPT_VERSION,
        )

    async def test_versioned_prompt_resolver_preserves_legacy_and_default(self) -> None:
        self.assertIs(
            DIRECTOR_SYSTEM_PROMPT,
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION),
        )
        self.assertIs(
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
            director_system_prompt_for_version(
                "agentgraph.director.minimal-neutral.v9"
            ),
        )
        self.assertIs(
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
            director_system_prompt_for_version(
                "agentgraph.director.constrained-action.skillflow-qa.v8"
            ),
        )
        self.assertIs(
            DIRECTOR_SYSTEM_PROMPT,
            director_system_prompt_for_version("prompt-v1"),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,
            director_system_prompt_for_version(HOTPOTQA_DIRECTOR_PROMPT_VERSION),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V6,
            director_system_prompt_for_version(QA_DIRECTOR_PROMPT_VERSION),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V5,
            director_system_prompt_for_version(
                LEGACY_QA_DIRECTOR_PROMPT_VERSION_V5
            ),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V4,
            director_system_prompt_for_version(
                LEGACY_QA_DIRECTOR_PROMPT_VERSION_V4
            ),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V3,
            director_system_prompt_for_version(
                LEGACY_QA_DIRECTOR_PROMPT_VERSION_V3
            ),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V2,
            director_system_prompt_for_version(
                LEGACY_QA_DIRECTOR_PROMPT_VERSION_V2
            ),
        )
        self.assertIs(
            QA_DIRECTOR_SYSTEM_PROMPT_V1,
            director_system_prompt_for_version(
                "agentgraph.director.qa-semantic-recovery.v1"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v21"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v20"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v18"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v17"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v16"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v15"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v14"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v13"
            ),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v12"
            ),
        )
        with self.assertRaises(ValueError):
            director_system_prompt_for_version(" ")

    async def test_hotpot_v22_prompt_encodes_semantic_and_recovery_policy(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="Which player won more titles?",
            semantic_protocol=HOTPOTQA_SEMANTIC_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            system_prompt=HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,
            prompt_version=HOTPOTQA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=HOTPOTQA_SEMANTIC_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )

        messages = transcript_messages(orchestrator.build_prompt(env, 0, ()))
        state = observation_payload(messages[-1])

        self.assertEqual(HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22, messages[0]["content"])
        self.assertIn(
            "must not predict or embed a concrete candidate answer",
            messages[0]["content"],
        )
        self.assertIn(
            "repair the responsible Agent's contract or completion condition",
            messages[0]["content"],
        )
        self.assertIn("public repair_instruction", messages[0]["content"])
        self.assertIn(
            "generic output-schema obligation",
            messages[0]["content"],
        )
        self.assertEqual(HOTPOTQA_SEMANTIC_PROTOCOL, state["semantic_protocol"])
        self.assertEqual(
            PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            state["recovery_policy"],
        )
        self.assertEqual(
            "qa-retrieval",
            state["terminal_constraints"]["required_evidence_tool_id"],
        )
        self.assertEqual(
            "reasoner_or_direct_reasoner_predecessor",
            state["semantic_lineage_constraints"][
                "required_evidence_tool_owner"
            ],
        )

    async def test_shared_qa_prompt_uses_neutral_live_canvas_policy(self) -> None:
        model_registry = registry()
        runtime = AgentRuntime(
            model_registry,
            FakeGateway(),
            dataset_id="triviaqa",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        env = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="Who wrote Lord of the Flies?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=QA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )

        messages = transcript_messages(orchestrator.build_prompt(env, 0, ()))
        state = observation_payload(messages[-1])
        self.assertEqual(QA_DIRECTOR_SYSTEM_PROMPT_V6, messages[0]["content"])
        self.assertNotIn("For HotpotQA,", messages[0]["content"])
        self.assertEqual(
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            state["semantic_protocol"],
        )
        self.assertEqual(
            "qa-retrieval",
            state["terminal_constraints"]["required_evidence_tool_id"],
        )
        self.assertNotIn("semantic_lineage_constraints", state)
        self.assertNotIn("optional_role_capabilities", state)
        prompt = messages[0]["content"]
        self.assertIn("admissible_action_types", prompt)
        self.assertIn("action_target_domains", prompt)
        self.assertIn("model_catalog", prompt)
        self.assertIn("tool_catalog", prompt)
        self.assertIn("ReAct is a bounded Tool-execution mode", prompt)
        self.assertIn("requested relation", prompt)
        self.assertIn("never invent either", prompt)
        self.assertIn("Preserve valid evidence", prompt)
        self.assertIn("typed failure attribution", prompt)
        self.assertIn("finish_admissibility is present and admissible", prompt)
        self.assertIn("Do not assume a fixed Agent count", prompt)
        self.assertIn("retrieval recipe", prompt)
        for fixed_recipe in (
            "Exactly one Reasoner",
            "The Verifier consumes only",
            "The terminal Formatter consumes only",
            "Reasoner--Verifier",
            "An evidence_retriever produces",
            "A reasoner binds",
            "A verifier checks",
            "A format Agent only copies",
            "optional_role_capabilities",
            "Which-comparison returns",
            "who-question returns",
            "spelling normalization",
            "alias expansion",
            "entity disambiguation",
            "query rewriting",
            "expand top-k",
        ):
            self.assertNotIn(fixed_recipe.casefold(), prompt.casefold())
        add_domain = state["action_target_domains"]["add_subgraph"]
        self.assertEqual(1, add_domain["min_new_agents"])
        self.assertEqual(3, add_domain["max_new_agents"])
        self.assertEqual([], add_domain["existing_agent_ids"])
        self.assertEqual([], add_domain["existing_agents"])
        self.assertIsNone(add_domain["current_output_agent_id"])
        self.assertEqual(
            {"format", "output"},
            set(add_domain["output_role_families"]),
        )
        self.assertEqual(
            [
                "agent_id",
                "model_id",
                "contract",
                "role_family",
                "allowed_tools",
                "execution_mode",
            ],
            add_domain["required_agent_fields"],
        )
        self.assertEqual(
            {"provider"},
            {item["provider_id"] for item in state["model_catalog"]},
        )

        strict_env = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="Who wrote Lord of the Flies?",
            require_format_agent=True,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        strict_messages = transcript_messages(
            orchestrator.build_prompt(strict_env, 0, ())
        )
        strict_state = observation_payload(strict_messages[-1])
        self.assertEqual(QA_DIRECTOR_SYSTEM_PROMPT_V6, strict_messages[0]["content"])
        self.assertNotIn("semantic_lineage_constraints", strict_state)
        strict_domain = strict_state["action_target_domains"]["add_subgraph"]
        self.assertTrue(strict_domain["require_format_agent"])
        self.assertEqual("format", strict_domain["output_role_family"])
        self.assertNotIn("output_role_families", strict_domain)
        self.assertNotIn("output", strict_domain["role_constraints"])
        director_live_add_subgraph_agent_declarations_json_schema_text(
            {"add_subgraph": strict_domain}
        )

        default = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        default_state = observation_payload(
            transcript_messages(default.build_prompt(env, 0, ()))[-1]
        )
        self.assertNotIn("semantic_protocol", default_state)
        self.assertNotIn("semantic_lineage_constraints", default_state)
        self.assertNotIn("recovery_policy", default_state)

    def test_shared_qa_system_prompt_is_workflow_neutral(self) -> None:
        prompt = QA_DIRECTOR_SYSTEM_PROMPT_V4.casefold()
        legacy_sentence = (
            "These artifact contracts do not require a separate Retriever, a "
            "fixed Agent count, a role order, or any particular edge or "
            "topology; follow only the responsibilities and relations admitted "
            "by the current state."
        )
        current_sentence = (
            "Whether a semantic responsibility is currently required is "
            "determined only by action_target_domains; do not infer a fixed "
            "Agent count, role order, edge, topology, or retrieval recipe."
        )

        self.assertIn(legacy_sentence, QA_DIRECTOR_SYSTEM_PROMPT_V3)
        self.assertNotIn(legacy_sentence, QA_DIRECTOR_SYSTEM_PROMPT_V4)
        self.assertEqual(
            QA_DIRECTOR_SYSTEM_PROMPT_V3.replace(
                legacy_sentence,
                current_sentence,
            ),
            QA_DIRECTOR_SYSTEM_PROMPT_V4,
        )

        self.assertIn("do not assume a fixed number or sequence", prompt)
        self.assertIn("fixed graph topology", prompt)
        self.assertIn("retrieval-strategy recipe", prompt)
        self.assertIn(
            "semantic responsibility is currently required is determined only "
            "by action_target_domains",
            prompt,
        )
        self.assertIn(
            "do not infer a fixed agent count, role order, edge, topology, or "
            "retrieval recipe",
            prompt,
        )
        self.assertIn("non-destructive recovery", prompt)
        self.assertIn("matching successful read tool receipt", prompt)
        self.assertIn("a format agent only copies", prompt)
        for prohibited in (
            "exactly one reasoner",
            "reasoner -> verifier",
            "verifier -> format",
            "evidence_retriever -> reasoner",
            "do not require a separate retriever",
            "linear workflow",
            "chain workflow",
            "required_direct_role_edges",
            "semantic_answer_owner_count",
            "preferred_actions",
            "preferred_action_order",
            "spelling normalization",
            "alias expansion",
            "entity disambiguation",
            "query rewriting",
            "expand top-k",
            "lord of the flies",
            "william golding",
        ):
            self.assertNotIn(prohibited, prompt)

    async def test_shared_qa_complete_transcript_is_workflow_neutral(self) -> None:
        model_registry = registry()
        runtime = AgentRuntime(
            model_registry,
            FakeGateway(),
            dataset_id="triviaqa",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        env = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="Who wrote Lord of the Flies?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=QA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )

        complete_prompt = orchestrator.build_prompt(env, 0, ())
        state = observation_payload(transcript_messages(complete_prompt)[-1])

        self.assertIn("admissible_action_types", state)
        self.assertIn("action_target_domains", state)
        self.assertIn("finish_admissibility", state)
        self.assertIn(
            "responsible_constraint",
            state["finish_admissibility"]["failure_attribution"],
        )
        self.assertNotIn("semantic_lineage_constraints", state)
        self.assertNotIn("optional_role_capabilities", state)
        self.assertEqual(
            {"format", "output"},
            set(
                state["action_target_domains"]["add_subgraph"][
                    "output_role_families"
                ]
            ),
        )
        for prohibited in (
            "required_direct_role_edges",
            "semantic_answer_owner_count",
            "preferred_actions",
            "preferred_action_order",
            "preferred_repair",
            "action_order",
        ):
            self.assertNotIn(prohibited, complete_prompt)

        rejected = await env.step('{"action":"finish"}')
        self.assertFalse(rejected.accepted)
        continued_prompt = orchestrator.continue_prompt(
            complete_prompt,
            '{"action":"finish"}',
            env,
            (),
        )
        continued_state = observation_payload(
            transcript_messages(continued_prompt)[-1]
        )
        self.assertIn("canvas_feedback", continued_state)
        self.assertIn("recent_rejected_actions", continued_state)
        for prohibited in (
            "required_direct_role_edges",
            "semantic_answer_owner_count",
            "preferred_actions",
            "preferred_action_order",
            "preferred_repair",
            "action_order",
        ):
            self.assertNotIn(prohibited, continued_prompt)

    async def test_rejected_qa_answer_precommit_is_not_replayed_to_director(self) -> None:
        model_registry = registry()
        runtime = AgentRuntime(
            model_registry,
            FakeGateway(),
            dataset_id="triviaqa",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        env = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="Where in England was Dame Judi Dench born?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=QA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )
        initial_prompt = orchestrator.build_prompt(env, 0, ())
        sampled_action = json.dumps(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": "reasoner",
                        "model_id": "qwen",
                        "contract": "Output ONLY the word 'Shirley'.",
                        "role_family": "reasoner",
                        "allowed_tools": [],
                        "execution_mode": "reasoning",
                    }
                ],
                "relations": [],
            }
        )

        rejected = await env.step(sampled_action)
        self.assertFalse(rejected.accepted)
        continued = orchestrator.continue_prompt(
            initial_prompt,
            sampled_action,
            env,
            (),
        )

        self.assertNotIn("Shirley", continued)
        messages = transcript_messages(continued)
        self.assertEqual(2, len(messages))
        state = observation_payload(messages[-1])
        self.assertEqual(1, len(state["recent_rejected_actions"]))
        self.assertIn(
            "pre-execution obligations only",
            state["recent_rejected_actions"][0]["reason"],
        )

    async def test_orchestrator_rejects_prompt_and_version_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            AgentGraphOrchestrator(
                registry(),
                ScriptedDirector([]),
                system_prompt=HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
                prompt_version=DIRECTOR_PROMPT_VERSION,
            )

    async def test_openai_director_boundary_is_local_qwen_supervisor(self) -> None:
        OpenAIDirectorClient(
            base_url="http://127.0.0.1:8015/v1",
            model="supervisor_theta",
        )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="https://provider.example/v1",
                model="supervisor_theta",
            )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="http://127.0.0.1:8015/v1",
                model="gpt-4o-mini",
            )

    async def test_openai_director_submits_exact_transcript_messages(self) -> None:
        messages = (
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": "initial Canvas observation"},
            {"role": "assistant", "content": '{"action":"finish"}'},
            {"role": "user", "content": "current Canvas observation"},
        )
        client = OpenAIDirectorClient(max_retries=0)
        captured = {}

        def fake_post(api_key, payload):
            captured.update(api_key=api_key, payload=payload)
            return {
                "id": "director-request",
                "model": "supervisor_theta",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }

        client._post = fake_post  # type: ignore[method-assign]
        response = await client.propose(encode_director_transcript(messages), seed=9)

        self.assertEqual(list(messages), captured["payload"]["messages"])
        self.assertEqual(9, captured["payload"]["seed"])
        self.assertEqual('{"action":"finish"}', response.text)

    async def test_end_to_end_scripted_canvas_and_execution(self) -> None:
        model_registry = registry()
        add_action = (
            '{"action":"add_agent","agent_id":"solver","model_id":"qwen",'
            '"contract":"solve"}'
        )
        add_response = "preface\n" + add_action + "\ndiscarded suffix"
        client = ScriptedDirector(
            [
                add_response,
                '{"action":"set_output","agent_id":"solver"}',
                '{"action":"finish"}',
            ]
        )
        gateway = FakeGateway()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=gateway,
            execute_on_edit=True,
            max_agents=10,
        )
        result = await AgentGraphOrchestrator(model_registry, client, seed=1).run(
            env, "2+2?"
        )
        self.assertEqual(result.final_answer, "answer from solver")
        self.assertEqual(len(result.turns), 3)
        self.assertNotIn("weighted_preferred_model", client.prompts[0])
        self.assertNotIn("available_skills", client.prompts[0])
        self.assertEqual(result.final_graph["output_agent_id"], "solver")
        self.assertEqual(client.seeds, [1, 2, 3])

        # Each completed Agent configuration executes immediately.  Changing
        # the node to the Output role re-executes that dirty node.
        self.assertIsNotNone(result.turns[0].canvas_result.execution)
        self.assertIsNotNone(result.turns[1].canvas_result.execution)
        self.assertIn("execution_result=", result.turns[0].canvas_result.feedback)
        self.assertIn("execution_result=", result.turns[1].canvas_result.feedback)
        self.assertIn("answer from solver", result.turns[1].canvas_result.feedback)
        self.assertNotIn("output_format", result.turns[1].canvas_result.feedback)
        self.assertNotIn('"final_answer"', result.turns[1].canvas_result.feedback)
        self.assertIn("output_inbox", result.turns[1].canvas_result.feedback)
        # Finish reuses the successful result for the unchanged graph revision.
        self.assertEqual(len(gateway.requests), 2)

        initial_messages = transcript_messages(client.prompts[0])
        continued_messages = transcript_messages(client.prompts[1])
        complete_messages = transcript_messages(client.prompts[2])
        self.assertEqual(
            ["system", "user"], [item["role"] for item in initial_messages]
        )
        self.assertEqual(
            ["system", "user", "assistant", "user"],
            [item["role"] for item in continued_messages],
        )
        self.assertEqual(
            ["system", "user", "assistant", "user", "assistant", "user"],
            [item["role"] for item in complete_messages],
        )
        self.assertEqual(initial_messages, complete_messages[:2])
        first_action = result.turns[0].canvas_result.action
        self.assertIsNotNone(first_action)
        assert first_action is not None
        self.assertEqual(
            add_response[: first_action.consumed_end],
            continued_messages[2]["content"],
        )
        self.assertNotIn("discarded suffix", continued_messages[2]["content"])

        initial_state = observation_payload(initial_messages[-1])
        complete_state = observation_payload(complete_messages[-1])
        self.assertEqual(initial_state["max_agents"], 10)
        self.assertEqual(initial_state["task"], "2+2?")
        self.assertNotIn("directed_edges", initial_state)
        self.assertNotIn("structural_issues", initial_state)
        self.assertNotIn("terminal_format_issue", initial_state)
        qwen_catalog = next(
            item
            for item in initial_state["model_catalog"]
            if item["model_id"] == "qwen"
        )
        self.assertEqual(
            {
                "family": "qwen",
                "profile": "text_qa",
                "text_qa_canary": "passed",
            },
            qwen_catalog["routing_metadata"],
        )
        self.assertEqual("solver", complete_state["current_graph"]["output_agent_id"])
        self.assertNotIn("directed_edges", complete_state)
        self.assertNotIn("structural_issues", complete_state)
        self.assertNotIn("task", complete_state)
        self.assertNotIn("model_catalog", complete_state)
        self.assertIn("execution_result=", complete_state["canvas_feedback"])
        self.assertEqual("empty", initial_state["topology_statistics"]["topology_family"])
        self.assertEqual("single", complete_state["topology_statistics"]["topology_family"])
        for state in (initial_state, complete_state):
            for removed_cue in (
                "max_rounds",
                "remaining_rounds",
                "graph_validation",
                "structurally_complete",
                "recent_canvas_history",
                "construction_progress",
                "output_format",
            ):
                self.assertNotIn(removed_cue, state)
        # The latest progressive result occurs only in the current user
        # observation, rather than in a reconstructed history field.
        self.assertEqual(2, client.prompts[2].count("execution_result="))

    async def test_intermediate_component_defers_terminal_constraints_until_finish(
        self,
    ) -> None:
        model_registry = registry()
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="two-hop question",
            require_exact_answer_tag=True,
            require_format_agent=True,
        )
        component_action = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"left","model_id":"qwen","contract":"left evidence"},'
            '{"agent_id":"right","model_id":"other","contract":"right evidence"},'
            '{"agent_id":"merge","model_id":"qwen","contract":"merge evidence"}'
            '],"relations":['
            '{"source_id":"left","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"right","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            ']}'
        )

        initial_prompt = orchestrator.build_prompt(env, 0, ())
        initial_state = observation_payload(transcript_messages(initial_prompt)[-1])
        self.assertNotIn("structural_issues", initial_state)
        self.assertNotIn("terminal_format_issue", initial_state)
        self.assertEqual(
            {
                "explicit_finish_required": True,
                "require_exact_answer_tag": True,
                "require_format_agent": True,
                "required_tool_id": None,
            },
            initial_state["terminal_constraints"],
        )

        component = await env.step(component_action)
        self.assertTrue(component.accepted)
        self.assertIsNone(env.graph.output_agent_id)
        component_prompt = orchestrator.continue_prompt(
            initial_prompt,
            component_action,
            env,
            (),
        )
        component_state = observation_payload(
            transcript_messages(component_prompt)[-1]
        )
        self.assertEqual(
            "fan_in",
            component_state["topology_statistics"]["topology_family"],
        )
        self.assertNotIn("structural_issues", component_state)
        self.assertNotIn("terminal_format_issue", component_state)

        finish_action = '{"action":"finish"}'
        premature_finish = await env.step(finish_action)
        self.assertFalse(premature_finish.accepted)
        self.assertIn(
            "output_agent_count",
            {issue.code for issue in premature_finish.validation_issues},
        )
        finish_prompt = orchestrator.continue_prompt(
            component_prompt,
            finish_action,
            env,
            (),
        )
        finish_state = observation_payload(transcript_messages(finish_prompt)[-1])
        self.assertIn("cannot finish", finish_state["canvas_feedback"])
        self.assertIn("output_agent_count", finish_state["canvas_feedback"])
        self.assertNotIn("structural_issues", finish_state)
        self.assertNotIn("terminal_format_issue", finish_state)

    async def test_selected_invalid_output_reports_terminal_format_issue(self) -> None:
        model_registry = registry()
        graph = AgentGraph(
            nodes=(
                AgentNode(
                    "solver",
                    "qwen",
                    "compute the semantic answer",
                    role_family="reasoning",
                ),
            ),
            output_agent_id="solver",
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="question",
            graph=graph,
            require_format_agent=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )

        state = observation_payload(
            transcript_messages(orchestrator.build_prompt(env, 0, ()))[-1]
        )
        self.assertNotIn("structural_issues", state)
        self.assertIn("terminal_format_issue", state)
        self.assertIn("distinct Format Agent", state["terminal_format_issue"])

    async def test_canvas_exposes_only_positive_finish_admission_without_semantic_protocol(
        self,
    ) -> None:
        model_registry = registry()

        class FormatGateway(FakeGateway):
            async def generate(self, request):
                self.requests.append(request)
                if request.is_format_agent:
                    return AgentResponse("<answer>Paris</answer>")
                return AgentResponse("Candidate answer: Paris\nEvidence: supplied fact")

        env = AgentWorkflowEnv(
            model_registry,
            gateway=FormatGateway(),
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=HOTPOTQA_DIRECTOR_PROMPT_VERSION,
        )
        initial = observation_payload(
            transcript_messages(orchestrator.build_prompt(env, 0, ()))[-1]
        )
        self.assertNotIn("finish_admissibility", initial)
        self.assertEqual(
            list(env.allowed_action_types),
            initial["admissible_action_types"],
        )

        action = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"solver","model_id":"qwen",'
            '"contract":"compute one semantic answer"},'
            '{"agent_id":"format","model_id":"other",'
            '"contract":"serialize one candidate","role_family":"format"}'
            '],"relations":['
            '{"source_id":"solver","target_id":"format",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"format"}'
        )
        accepted = await env.step(action)
        self.assertTrue(accepted.accepted)
        prompt = orchestrator.continue_prompt(
            orchestrator.build_prompt(env.fork(), 0, ()),
            action,
            env,
            (),
        )
        state = observation_payload(transcript_messages(prompt)[-1])
        self.assertEqual(
            {
                "admissible": True,
                "graph_revision": env.revision,
                "submission_semantics": "explicit_finish",
            },
            state["finish_admissibility"],
        )
        self.assertNotIn("remaining_rounds", state)
        self.assertIn("finish", state["admissible_action_types"])

    async def test_director_terminal_policy_is_issue_driven_without_role_template(
        self,
    ) -> None:
        self.assertIn("exactly one valid JSON action each turn", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn(
            "action types listed in admissible_action_types",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "add_subgraph adds one functional subgraph of one to three Agents as one transaction",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "A directed relation routes the source artifact to the target",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Each accepted edit is executed once",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Use finish only when finish_admissibility is present and admissible",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not assume a fixed workflow topology or an unlisted Skill",
            DIRECTOR_SYSTEM_PROMPT,
        )
        for prohibited in (
            "answer span",
            "answer type",
            "source-grounded evidence",
            "qualifier",
            "Format Agent",
            "fan-in",
            "Researcher",
            "Critic",
            "must use three",
            "singleton",
        ):
            self.assertNotIn(prohibited.casefold(), DIRECTOR_SYSTEM_PROMPT.casefold())

    async def test_history_window_keeps_actions_and_full_prior_observations_for_neutral_v10(
        self,
    ) -> None:
        model_registry = registry()
        actions = [
            '{"action":"add_agent","agent_id":"source","model_id":"qwen","contract":"produce evidence"}',
            '{"action":"add_agent","agent_id":"sink","model_id":"other","contract":"consume source evidence"}',
            '{"action":"set_relation","source_id":"source","target_id":"sink","source_to_target":true,"target_to_source":false}',
            '{"action":"set_output","agent_id":"sink"}',
        ]
        client = ScriptedDirector(actions)
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())

        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=4,
            history_window=2,
        ).run(env, "same task")

        messages = transcript_messages(client.prompts[3])
        self.assertEqual(
            ["system", "user", "assistant", "user", "assistant", "user"],
            [item["role"] for item in messages],
        )
        self.assertEqual(actions[1], messages[2]["content"])
        self.assertEqual(actions[2], messages[4]["content"])
        self.assertNotIn(actions[0], [item["content"] for item in messages[2:]])
        self.assertNotIn("recent_canvas_history", client.prompts[3])
        historical_state = observation_payload(messages[3])
        current_state = observation_payload(messages[5])
        self.assertIn("current_graph", historical_state)
        self.assertIn("admissible_action_types", historical_state)
        self.assertEqual(
            result.turns[2].canvas_result.snapshot.graph.to_dict(),
            current_state["current_graph"],
        )
        self.assertEqual(
            list(env.allowed_action_types),
            current_state["admissible_action_types"],
        )

    async def test_compact_history_preserves_failure_and_tool_receipts_exactly(
        self,
    ) -> None:
        model_registry = registry()
        runtime = AgentRuntime(
            model_registry,
            FakeGateway(),
            dataset_id="triviaqa",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        )
        env = AgentWorkflowEnv(
            model_registry,
            runtime=runtime,
            problem="Who wrote Lord of the Flies?",
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            history_window=2,
            prompt_version=QA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )
        initial_prompt = orchestrator.build_prompt(env, 0, ())
        initial_messages = transcript_messages(initial_prompt)
        canvas_feedback = (
            'accepted modify_agent at revision 3; execution_error={'
            '"failure_category":"tool_execution","retryability":"retryable",'
            '"react_public_error_summary":{"successful_tool_receipt_count":2,'
            '"successful_evidence_read_count":1,"last_public_error":"timeout"}}'
        )
        failure_attribution = {
            "responsible_agent_id": "reasoner",
            "responsible_constraint": "evidence_receipt_lineage",
            "retryability": "retryable",
        }
        tool_receipts = [
            {
                "tool_id": "qa-retrieval",
                "request_id": "receipt-1",
                "result": {"passage": "public evidence"},
            }
        ]
        historical_payload = {
            "current_graph": {"large_stale_graph": "x" * 4000},
            "topology_statistics": {"node_count": 8},
            "admissible_action_types": ["modify_agent"],
            "action_target_domains": {"modify_agent": "y" * 4000},
            "terminal_constraints": {"explicit_finish_required": True},
            "semantic_lineage_constraints": {"output_role_family": "format"},
            "canvas_feedback": canvas_feedback,
            "recent_rejected_actions": [
                {
                    "revision": 2,
                    "action": "finish",
                    "reason": "missing receipt",
                }
            ],
            "finish_admissibility": {
                "admissible": False,
                "stage": "semantic_protocol",
                "reason": "missing receipt-backed evidence lineage",
                "issues": [{"code": "missing_receipt"}],
                "failure_attribution": failure_attribution,
                "recovery_state": {"large_duplicate_state": "z" * 4000},
            },
            "tool_receipts": tool_receipts,
            "execution_receipt": {"executed_agent_ids": ["reasoner"]},
        }
        previous_prompt = encode_director_transcript(
            (
                initial_messages[0],
                initial_messages[1],
                {"role": "assistant", "content": '{"action":"modify_agent"}'},
                {
                    "role": "user",
                    "content": orchestrator._observation_message(
                        historical_payload
                    ),
                },
            )
        )
        expected_current = orchestrator._canvas_observation(
            env,
            include_task_context=False,
            skills=(),
        )

        continued = orchestrator.continue_prompt(
            previous_prompt,
            '{"action":"set_output","agent_id":"reasoner"}',
            env,
            (),
        )

        messages = transcript_messages(continued)
        compact_history = observation_payload(messages[3])
        current_state = observation_payload(messages[-1])
        self.assertEqual(canvas_feedback, compact_history["canvas_feedback"])
        self.assertEqual(tool_receipts, compact_history["tool_receipts"])
        self.assertEqual(
            {"executed_agent_ids": ["reasoner"]},
            compact_history["execution_receipt"],
        )
        self.assertEqual(
            failure_attribution,
            compact_history["finish_admissibility"]["failure_attribution"],
        )
        self.assertEqual(
            historical_payload["recent_rejected_actions"],
            compact_history["recent_rejected_actions"],
        )
        self.assertNotIn("recovery_state", compact_history["finish_admissibility"])
        for stale_key in (
            "current_graph",
            "topology_statistics",
            "admissible_action_types",
            "action_target_domains",
            "terminal_constraints",
            "semantic_lineage_constraints",
        ):
            self.assertNotIn(stale_key, compact_history)
        self.assertEqual(expected_current, current_state)
        self.assertEqual(
            env.model_admissible_action_targets(),
            current_state["action_target_domains"],
        )

    async def test_legacy_qa_v4_keeps_full_historical_observation(self) -> None:
        model_registry = registry()
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=LEGACY_QA_DIRECTOR_PROMPT_VERSION_V4,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )
        messages = [
            {"role": "system", "content": QA_DIRECTOR_SYSTEM_PROMPT_V4},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"action":"finish"}'},
            {
                "role": "user",
                "content": orchestrator._observation_message(
                    {"current_graph": {"revision": 1}}
                ),
            },
            {"role": "assistant", "content": '{"action":"finish"}'},
            {
                "role": "user",
                "content": orchestrator._observation_message(
                    {"current_graph": {"revision": 2}}
                ),
            },
        ]

        replay = orchestrator._compact_historical_messages(messages)

        self.assertEqual(messages, replay)

    async def test_legacy_qa_v5_keeps_compact_historical_observation(self) -> None:
        model_registry = registry()
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=LEGACY_QA_DIRECTOR_PROMPT_VERSION_V5,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )
        messages = [
            {"role": "system", "content": QA_DIRECTOR_SYSTEM_PROMPT_V5},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"action":"finish"}'},
            {
                "role": "user",
                "content": orchestrator._observation_message(
                    {
                        "current_graph": {"revision": 1},
                        "canvas_feedback": "typed failure",
                    }
                ),
            },
            {"role": "assistant", "content": '{"action":"finish"}'},
            {
                "role": "user",
                "content": orchestrator._observation_message(
                    {"current_graph": {"revision": 2}}
                ),
            },
        ]

        replay = orchestrator._compact_historical_messages(messages)

        historical = observation_payload(replay[3])
        current = observation_payload(replay[-1])
        self.assertEqual("typed failure", historical["canvas_feedback"])
        self.assertNotIn("current_graph", historical)
        self.assertEqual({"revision": 2}, current["current_graph"])

    async def test_catalog_order_is_decoupled_from_rollout_sampling_seed(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )
        first = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=101,
            catalog_order_seed="condition:same-task",
        )
        second = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=202,
            catalog_order_seed="condition:same-task",
        )

        first_state = observation_payload(
            transcript_messages(first.build_prompt(env, 0, ()))[-1]
        )
        second_state = observation_payload(
            transcript_messages(second.build_prompt(env, 0, ()))[-1]
        )
        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.seed, second.seed)

    async def test_scientific_rollout_ordinal_changes_sampling_not_catalog(
        self,
    ) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )

        def orchestrator(rollout_ordinal: int) -> AgentGraphOrchestrator:
            coordinate = ScientificSamplingCoordinate(
                sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
                schedule_purpose="architecture-dev",
                ordered_sequence_hash=stable_hash(["hotpotqa:one"]),
                sequence_position=rollout_ordinal,
                task_id="hotpotqa:one",
                optimizer_step_or_anchor_ordinal=0,
            )
            return AgentGraphOrchestrator(
                model_registry,
                ScriptedDirector([]),
                seed=17,
                catalog_order_seed="architecture-dev:hotpotqa:one",
                sampling_base_seed=17,
                sampling_coordinate=coordinate,
            )

        first = orchestrator(0)
        second = orchestrator(1)
        first_state = observation_payload(
            transcript_messages(first.build_prompt(env, 0, ()))[-1]
        )
        second_state = observation_payload(
            transcript_messages(second.build_prompt(env, 0, ()))[-1]
        )

        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.generation_seed(0), second.generation_seed(0))
        self.assertEqual(
            "skillev-scientific-sampling@1",
            first.sampling_receipt["algorithm"],
        )

    async def test_forced_probe_condition_is_separate_from_active_skill_prior(
        self,
    ) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        active_prior = {
            "skill_id": "skill-active",
            "application_mode": "rejectable_prompt_prior",
            "content": "Optional evidence-gated instruction.",
        }
        forced_condition = {
            "condition_id": "candidate-a",
            "application_mode": "forced_probe_condition",
            "content": "Apply the predeclared intervention for this probe arm.",
        }

        messages = transcript_messages(
            orchestrator.build_prompt(
                env,
                0,
                (active_prior, forced_condition),
            )
        )
        state = observation_payload(messages[-1])

        self.assertEqual([active_prior], state["available_skills"])
        self.assertEqual([forced_condition], state["exploration_conditions"])
        self.assertNotIn("validated Skill facts", messages[-1]["content"])

    async def test_round_limit_is_explicit_failure(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                '{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}'
            ]
        )
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())
        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=1,
        ).run(env, "task")

        self.assertIsNone(result.final_answer)
        self.assertFalse(result.explicit_finish)
        self.assertEqual("max_rounds", result.termination_reason)
        self.assertEqual(1, len(result.turns))

    async def test_verified_qa_empty_canvas_domain_is_natural_terminal(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                (
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"solver","model_id":"qwen",'
                    '"contract":"solve"}],"relations":[],'
                    '"output_agent_id":"solver"}'
                )
            ]
        )

        class DeadEndAfterOneTurnEnv(AgentWorkflowEnv):
            def model_admissible_action_types(self):
                if self.history:
                    return ()
                return super().model_admissible_action_types()

        env = DeadEndAfterOneTurnEnv(model_registry, gateway=FakeGateway())
        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=3,
            semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            sampling_action_profile=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE
            ),
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
        ).run(env, "task")

        self.assertEqual(1, len(client.prompts))
        self.assertEqual(1, len(result.turns))
        self.assertEqual("max_rounds", result.termination_reason)
        self.assertFalse(result.explicit_finish)
        self.assertIsNone(result.final_answer)
        self.assertEqual(
            "canvas_action_domain_exhausted",
            result.terminal_canvas_diagnosis["public_error_code"],
        )
        self.assertEqual(
            result.final_graph["revision"],
            result.terminal_canvas_diagnosis["graph_revision"],
        )
        self.assertIn(
            "finish_admissibility",
            result.terminal_canvas_diagnosis,
        )
        self.assertIn("recovery_state", result.terminal_canvas_diagnosis)

    async def test_progressive_action_mask_switches_only_after_finish_admission(
        self,
    ) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                (
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"solver","model_id":"qwen",'
                    '"contract":"solve"}],"relations":[],'
                    '"output_agent_id":"solver"}'
                ),
                '{"action":"finish"}',
            ]
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        )

        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
        ).run(env, "task")

        self.assertTrue(result.explicit_finish)
        self.assertEqual(
            ["add_subgraph", "finish"],
            [item["action_schema_branch"] for item in client.schema_requests],
        )
        self.assertEqual(
            director_state_conditioned_sampling_json_schema_text("add_subgraph"),
            client.schema_requests[0]["action_json_schema"],
        )
        self.assertEqual(
            director_state_conditioned_sampling_json_schema_text("finish"),
            client.schema_requests[1]["action_json_schema"],
        )
        self.assertEqual(
            [DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION] * 2,
            [
                item["action_json_schema_version"]
                for item in client.schema_requests
            ],
        )

    async def test_model_admissible_action_mask_tracks_current_canvas_domain(
        self,
    ) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                (
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"solver","model_id":"qwen",'
                    '"contract":"solve"}],"relations":[],'
                    '"output_agent_id":"solver"}'
                ),
                '{"action":"finish"}',
            ]
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ),
        )

        result = await orchestrator.run(env, "task")

        self.assertTrue(result.explicit_finish)
        first_actions = director_actions_from_admissible_schema_branch(
            client.schema_requests[0]["action_schema_branch"]
        )
        second_actions = director_actions_from_admissible_schema_branch(
            client.schema_requests[1]["action_schema_branch"]
        )
        self.assertEqual(("add_subgraph",), first_actions)
        self.assertIn("finish", second_actions)
        self.assertEqual(
            director_model_admissible_sampling_json_schema_text(first_actions),
            client.schema_requests[0]["action_json_schema"],
        )
        self.assertEqual(
            director_model_admissible_sampling_json_schema_text(second_actions),
            client.schema_requests[1]["action_json_schema"],
        )
        self.assertEqual(
            [DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION] * 2,
            [
                item["action_json_schema_version"]
                for item in client.schema_requests
            ],
        )

    def test_model_admissible_schema_branch_round_trip_is_exact(self) -> None:
        actions = (
            "add_subgraph",
            "modify_agent",
            "set_relation",
            "finish",
        )
        branch = director_model_admissible_schema_branch(actions)

        self.assertEqual(
            "admissible-v2:add_subgraph|modify_agent|set_relation|finish",
            branch,
        )
        self.assertEqual(
            actions,
            director_actions_from_admissible_schema_branch(branch),
        )
        singleton_selector = json.loads(
            director_model_admissible_sampling_json_schema_text(("finish",))
        )
        self.assertEqual(["finish"], singleton_selector["properties"]["action"]["enum"])
        with self.assertRaises(ValueError):
            director_actions_from_admissible_schema_branch("finish")
        with self.assertRaises(ValueError):
            director_actions_from_admissible_schema_branch("admissible:")
        with self.assertRaises(ValueError):
            director_model_admissible_schema_branch(("finish", "finish"))

        legacy_branch = director_model_admissible_schema_branch_v1(actions)
        self.assertEqual(
            "admissible:add_subgraph|modify_agent|set_relation|finish",
            legacy_branch,
        )
        self.assertEqual(
            actions,
            director_actions_from_admissible_schema_branch(legacy_branch),
        )

    def test_model_admissible_v2_factorizes_action_and_exact_parameters(self) -> None:
        actions = ("modify_agent", "set_relation", "finish")
        strict_schema = json.loads(
            director_model_admissible_sampling_json_schema_text(actions)
        )
        legacy_schema = json.loads(
            director_model_admissible_sampling_json_schema_text_v1(actions)
        )

        self.assertEqual(["action"], strict_schema["required"])
        self.assertEqual(
            list(actions),
            strict_schema["properties"]["action"]["enum"],
        )
        self.assertEqual(["action"], legacy_schema["required"])
        self.assertNotIn("agent_id", strict_schema["properties"])
        self.assertIn("agent_id", legacy_schema["properties"])

        field_selector = json.loads(
            director_modify_agent_field_selector_json_schema_text()
        )
        self.assertEqual(
            ["action", "field"],
            field_selector["required"],
        )
        contract_branch = json.loads(
            director_modify_agent_field_sampling_json_schema_text("contract")
        )
        self.assertEqual(
            {"action", "agent_id", "contract"},
            set(contract_branch["required"]),
        )
        self.assertEqual(
            {"action", "agent_id", "contract"},
            set(contract_branch["properties"]),
        )

    def test_model_admissible_v3_binds_live_parameter_domains(self) -> None:
        actions = (
            "add_subgraph",
            "modify_agent",
            "set_relation",
            "set_output",
            "finish",
        )
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 3,
                "existing_agent_ids": ["reasoner"],
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ],
                "model_ids": ["qwen", "other"],
                "role_constraints": {
                    "reasoner": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                    },
                    "verifier": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                    "format": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                        "contracts": [
                            "copy the supported Verifier candidate "
                            "character-for-character into the required answer wrapper"
                        ],
                    },
                },
                "admitted_new_role_families": [
                    "reasoner",
                    "verifier",
                    "format",
                ],
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
            },
            "modify_agent": {
                "mutable_fields": ["contract", "model_id"],
                "per_agent_candidates": [
                    {
                        "agent_id": "reasoner",
                        "mutable_fields": ["contract", "model_id"],
                        "discrete_value_domains": {"model_id": ["other"]},
                    },
                    {
                        "agent_id": "verifier",
                        "mutable_fields": ["model_id"],
                        "discrete_value_domains": {"model_id": ["qwen"]},
                    },
                ],
            },
            "set_relation": {
                "candidates": [
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
            },
            "set_output": {"agent_ids": ["format"]},
            "finish": {"admissible": True},
        }

        self.assertEqual(
            "admissible-v3:" + "|".join(actions),
            director_model_admissible_schema_branch_v3(actions),
        )
        canonical_domains = director_live_action_target_domains_json(
            actions,
            domains,
        )
        self.assertEqual(domains, json.loads(canonical_domains))
        self.assertEqual(
            json.loads(director_model_admissible_sampling_json_schema_text(actions)),
            json.loads(director_model_admissible_sampling_json_schema_text_v3(actions)),
        )

        declaration_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(domains)
        )
        agent_declaration = declaration_schema["properties"]["agents"]
        count_branches = agent_declaration["oneOf"]
        self.assertEqual([1, 2, 3], [item["minItems"] for item in count_branches])
        self.assertEqual([1, 2, 3], [item["maxItems"] for item in count_branches])
        self.assertTrue(all(item["items"] is False for item in count_branches))
        reasoner_branch = count_branches[0]["prefixItems"][0]["anyOf"][0]
        self.assertEqual(
            {"const": "node_1"},
            reasoner_branch["properties"]["agent_id"],
        )
        self.assertIn("role_family", reasoner_branch["required"])
        self.assertEqual(
            {"const": "reasoner"},
            reasoner_branch["properties"]["role_family"],
        )
        self.assertEqual(
            {"enum": [["qa-retrieval"]]},
            reasoner_branch["properties"]["allowed_tools"],
        )

        self.assertEqual(
            {"enum": ["qwen", "other"]},
            reasoner_branch["properties"]["model_id"],
        )
        self.assertEqual(
            {"const": "node_2"},
            count_branches[1]["prefixItems"][1]["anyOf"][0]["properties"][
                "agent_id"
            ],
        )
        occupied_id_domains = json.loads(json.dumps(domains))
        occupied_id_domains["add_subgraph"]["existing_agent_ids"] = [
            "reasoner",
            "node_1",
        ]
        occupied_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                occupied_id_domains
            )
        )
        self.assertEqual(
            {"const": "node_2"},
            occupied_schema["properties"]["agents"]["oneOf"][0][
                "prefixItems"
            ][0]["anyOf"][0]["properties"]["agent_id"],
        )
        role_selection_schema = json.loads(
            director_live_add_subgraph_role_selection_json_schema_text(domains)
        )
        self.assertEqual(
            [1, 2, 3],
            [
                item["minItems"]
                for item in role_selection_schema["properties"]["agents"][
                    "oneOf"
                ]
            ],
        )
        selected_roles = director_live_add_subgraph_role_selection_from_text(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","role_family":"reasoner"}]}',
            domains,
        )
        conditioned_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains,
                selected_agent_roles=selected_roles,
            )
        )
        conditioned_agents = conditioned_schema["properties"]["agents"]["oneOf"]
        self.assertEqual(1, len(conditioned_agents))
        conditioned_role_branches = conditioned_agents[0]["prefixItems"][0][
            "anyOf"
        ]
        self.assertEqual(1, len(conditioned_role_branches))
        self.assertEqual(
            {"const": "reasoner"},
            conditioned_role_branches[0]["properties"]["role_family"],
        )
        sampled_agents = director_live_add_subgraph_agent_declarations_from_text(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","model_id":"qwen",'
            '"contract":"align evidence","role_family":"reasoner",'
            '"allowed_tools":["qa-retrieval"],"execution_mode":"react"}]}',
            domains,
            selected_agent_roles=selected_roles,
        )
        sampled_with_qwen_eos = (
            director_live_add_subgraph_agent_declarations_from_text(
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"node_1","model_id":"qwen",'
                '"contract":"align evidence","role_family":"reasoner",'
                '"allowed_tools":["qa-retrieval"],"execution_mode":"react"}]}'
                '<|endoftext|>',
                domains,
                selected_agent_roles=selected_roles,
            )
        )
        self.assertEqual(sampled_agents, sampled_with_qwen_eos)
        with self.assertRaisesRegex(ValueError, "changed their selected roles"):
            director_live_add_subgraph_agent_declarations_from_text(
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"node_1","model_id":"qwen",'
                '"contract":"verify","role_family":"verifier",'
                '"allowed_tools":[],"execution_mode":"reasoning"}]}',
                domains,
                selected_agent_roles=selected_roles,
            )
        with self.assertRaisesRegex(ValueError, "unique neutral IDs"):
            director_live_add_subgraph_agent_declarations_from_text(
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"node_1","model_id":"qwen",'
                '"contract":"align evidence","role_family":"reasoner",'
                '"allowed_tools":["qa-retrieval"],"execution_mode":"react"},'
                '{"agent_id":"node_1","model_id":"qwen",'
                '"contract":"verify","role_family":"verifier",'
                '"allowed_tools":[],"execution_mode":"reasoning"}]}',
                domains,
            )
        with self.assertRaisesRegex(ValueError, "contain trailing text"):
            director_live_add_subgraph_agent_declarations_from_text(
                '{"action":"add_subgraph","agents":['
                '{"agent_id":"node_1","model_id":"qwen",'
                '"contract":"align evidence","role_family":"reasoner",'
                '"allowed_tools":["qa-retrieval"],"execution_mode":"react"}]}'
                'unconstrained suffix',
                domains,
            )
        add_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=sampled_agents,
            )
        )
        self.assertEqual(
            list(sampled_agents),
            add_schema["properties"]["agents"]["const"],
        )
        relation_branch = add_schema["properties"]["relations"]["items"]["anyOf"][0]
        self.assertEqual(
            ["reasoner", "node_1"],
            relation_branch["properties"]["source_id"]["enum"],
        )
        self.assertEqual(
            ["reasoner", "node_1"],
            add_schema["properties"]["output_agent_id"]["anyOf"][0]["enum"],
        )
        modify_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "modify_agent",
                domains,
                modify_field="model_id",
                modify_agent_id="reasoner",
            )
        )
        self.assertEqual(
            {"const": "reasoner"},
            modify_schema["properties"]["agent_id"],
        )
        self.assertEqual(
            {"enum": ["other"]},
            modify_schema["properties"]["model_id"],
        )
        modify_selector = json.loads(
            director_live_modify_agent_selector_json_schema_text(
                domains,
                "model_id",
            )
        )
        self.assertEqual(
            ["reasoner", "verifier"],
            modify_selector["properties"]["agent_id"]["enum"],
        )
        relation_selector = json.loads(
            director_live_relation_candidate_selector_json_schema_text(domains)
        )
        self.assertEqual(
            [0, 1],
            relation_selector["properties"]["candidate_index"]["enum"],
        )
        relation_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "set_relation",
                domains,
                relation_candidate_index=0,
            )
        )
        self.assertEqual(
            {"const": "reasoner"},
            relation_schema["properties"]["source_id"],
        )
        self.assertEqual(
            {"enum": ["format"]},
            json.loads(
                director_live_action_parameter_json_schema_text(
                    "set_output",
                    domains,
                )
            )["properties"]["agent_id"],
        )
        self.assertEqual(
            {"const": "finish"},
            json.loads(
                director_live_action_parameter_json_schema_text(
                    "finish",
                    domains,
                )
            )["properties"]["action"],
        )
        with self.assertRaises(ValueError):
            director_live_relation_candidate_selector_json_schema_text(
                {"set_relation": {"candidates": []}}
            )

        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            semantic_protocol=HOTPOTQA_SEMANTIC_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        request = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
        ).action_schema_request(env)
        self.assertEqual(
            DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
            request["action_target_domain_version"],
        )
        self.assertEqual(
            "agentgraph.live-action-target-domains.v10",
            DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
        )
        initial_retriever_domain = env.model_admissible_action_targets()[
            "add_subgraph"
        ]["role_constraints"]["evidence_retriever"]
        self.assertNotIn("contracts", initial_retriever_domain)
        self.assertNotIn(
            "completion_conditions",
            initial_retriever_domain,
        )
        self.assertEqual(
            env.model_admissible_action_targets(),
            json.loads(request["action_target_domains_json"]),
        )
        self.assertEqual(
            "admissible-v3:add_subgraph",
            request["action_schema_branch"],
        )

    def test_role_conditional_qa_schema_preserves_execution_profile_pairs(
        self,
    ) -> None:
        profiles = [
            {"execution_mode": "reasoning", "allowed_tools": []},
            {
                "execution_mode": "react",
                "allowed_tools": ["qa-retrieval"],
            },
        ]
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 1,
                "existing_agent_ids": [],
                "existing_agents": [],
                "current_output_agent_id": None,
                "output_role_families": ["format", "output"],
                "semantic_protocol": QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ],
                "model_ids": ["qwen"],
                "registered_execution_profiles": profiles,
                "role_constraints": {
                    "output": {
                        "execution_modes": ["reasoning", "react"],
                        "allowed_tools": [[], ["qa-retrieval"]],
                        "execution_profiles": profiles,
                    }
                },
                "admitted_new_role_families": ["output"],
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

        schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains
            )
        )
        branches = schema["properties"]["agents"]["oneOf"][0][
            "prefixItems"
        ][0]["anyOf"]
        sampled_profiles = {
            (
                branch["properties"]["execution_mode"]["const"],
                tuple(branch["properties"]["allowed_tools"]["const"]),
            )
            for branch in branches
        }
        self.assertEqual(
            {("reasoning", ()), ("react", ("qa-retrieval",))},
            sampled_profiles,
        )
        with self.assertRaisesRegex(ValueError, "registered profile"):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {
                        "action": "add_subgraph",
                        "agents": [
                            {
                                "agent_id": "node_1",
                                "model_id": "qwen",
                                "contract": "answer from public evidence",
                                "role_family": "output",
                                "allowed_tools": ["qa-retrieval"],
                                "execution_mode": "reasoning",
                            }
                        ],
                    }
                ),
                domains,
            )

    def test_hotpotqa_v3_binds_semantic_relation_directions_and_format_output(
        self,
    ) -> None:
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 3,
                "existing_agent_ids": [],
                "existing_agents": [],
                "current_output_agent_id": None,
                "output_role_family": "format",
                "semantic_protocol": HOTPOTQA_SEMANTIC_PROTOCOL,
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ],
                "model_ids": ["qwen"],
                "role_constraints": {
                    "reasoner": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                    },
                    "verifier": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                    "format": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                },
                "admitted_new_role_families": [
                    "reasoner",
                    "verifier",
                    "format",
                ],
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
        agents = [
            {
                "agent_id": "node_1",
                "model_id": "qwen",
                "contract": "align evidence to the answer slot",
                "role_family": "reasoner",
                "allowed_tools": ["qa-retrieval"],
                "execution_mode": "react",
            },
            {
                "agent_id": "node_2",
                "model_id": "qwen",
                "contract": "verify the semantic candidate",
                "role_family": "verifier",
                "allowed_tools": [],
                "execution_mode": "reasoning",
            },
            {
                "agent_id": "node_3",
                "model_id": "qwen",
                "contract": (
                    "copy the supported Verifier candidate character-for-character "
                    "into the required answer wrapper"
                ),
                "role_family": "format",
                "allowed_tools": [],
                "execution_mode": "reasoning",
            },
        ]
        candidates = director_live_add_subgraph_relation_candidates(
            domains,
            agents,
        )
        self.assertEqual(
            (
                {
                    "source_id": "node_1",
                    "target_id": "node_2",
                    "source_to_target": True,
                    "target_to_source": False,
                },
                {
                    "source_id": "node_1",
                    "target_id": "node_2",
                    "source_to_target": True,
                    "target_to_source": True,
                },
                {
                    "source_id": "node_2",
                    "target_id": "node_3",
                    "source_to_target": True,
                    "target_to_source": False,
                },
            ),
            candidates,
        )
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=agents,
            )
        )
        relation_branches = schema["properties"]["relations"]["items"][
            "anyOf"
        ]
        self.assertEqual(1, schema["properties"]["relations"]["maxItems"])
        sampled_domains = tuple(
            {
                key: branch["properties"][key]["const"]
                for key in (
                    "source_id",
                    "target_id",
                    "source_to_target",
                    "target_to_source",
                )
            }
            for branch in relation_branches
        )
        self.assertEqual(candidates, sampled_domains)
        self.assertEqual(
            {"enum": ["node_3"]},
            schema["properties"]["output_agent_id"]["anyOf"][0],
        )
        malformed = json.loads(json.dumps(domains))
        malformed["add_subgraph"].pop("existing_agents")
        with self.assertRaisesRegex(ValueError, "existing-Agent role domain"):
            director_live_add_subgraph_agent_declarations_json_schema_text(
                malformed
            )

        reordered_agents = [
            {**agents[1], "agent_id": "node_1"},
            {**agents[0], "agent_id": "node_2"},
            agents[2],
        ]
        reordered_candidates = director_live_add_subgraph_relation_candidates(
            domains,
            reordered_agents,
        )
        self.assertIn(
            {
                "source_id": "node_2",
                "target_id": "node_1",
                "source_to_target": True,
                "target_to_source": False,
            },
            reordered_candidates,
        )
        self.assertNotIn(
            {
                "source_id": "node_1",
                "target_id": "node_2",
                "source_to_target": False,
                "target_to_source": True,
            },
            reordered_candidates,
        )

        existing_output_domains = json.loads(json.dumps(domains))
        existing_output_domain = existing_output_domains["add_subgraph"]
        existing_output_domain["existing_agent_ids"] = ["existing_format"]
        existing_output_domain["existing_agents"] = [
            {"agent_id": "existing_format", "role_family": "format"}
        ]
        existing_output_domain["current_output_agent_id"] = "existing_format"
        existing_output_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                existing_output_domains,
                add_agents=agents[:2],
            )
        )
        self.assertEqual(
            {"type": "null"},
            existing_output_schema["properties"]["output_agent_id"],
        )

        missing_output_receipt = json.loads(json.dumps(domains))
        missing_output_receipt["add_subgraph"].pop("current_output_agent_id")
        with self.assertRaisesRegex(ValueError, "current Output receipt"):
            director_live_add_subgraph_agent_declarations_json_schema_text(
                missing_output_receipt
            )

    def test_replacement_add_schema_binds_artifact_and_isolated_execution(
        self,
    ) -> None:
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 1,
                "existing_agent_ids": [
                    "failed_reader",
                    "reasoner",
                    "verifier",
                    "formatter",
                ],
                "existing_agents": [
                    {
                        "agent_id": "failed_reader",
                        "role_family": "evidence_retriever",
                    },
                    {"agent_id": "reasoner", "role_family": "reasoner"},
                    {"agent_id": "verifier", "role_family": "verifier"},
                    {"agent_id": "formatter", "role_family": "format"},
                ],
                "current_output_agent_id": "formatter",
                "output_role_family": "format",
                "semantic_protocol": HOTPOTQA_SEMANTIC_PROTOCOL,
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                    "artifact_type",
                ],
                "model_ids": ["qwen"],
                "role_constraints": {
                    "evidence_retriever": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                        "artifact_types": ["retrieval_evidence"],
                        "contracts": [
                            _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                        ],
                        "completion_conditions": [
                            _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                        ],
                    },
                    "reasoner": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                    },
                    "verifier": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                    "format": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                },
                "admitted_new_role_families": ["evidence_retriever"],
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
                "relations": [],
                "output_agent_id": None,
            }
        }
        declarations_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains
            )
        )
        agent_schema = declarations_schema["properties"]["agents"]["oneOf"][
            0
        ]["prefixItems"][0]["anyOf"][0]
        self.assertEqual(
            {"enum": ["retrieval_evidence"]},
            agent_schema["properties"]["artifact_type"],
        )
        self.assertEqual(
            {"enum": [_QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT]},
            agent_schema["properties"]["contract"],
        )
        self.assertEqual(
            {"enum": [_QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION]},
            agent_schema["properties"]["completion_condition"],
        )
        self.assertIn("completion_condition", agent_schema["required"])

        agent = {
            "agent_id": "node_1",
            "model_id": "qwen",
            "contract": _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
            "completion_condition": (
                _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
            ),
            "role_family": "evidence_retriever",
            "allowed_tools": ["qa-retrieval"],
            "execution_mode": "react",
            "artifact_type": "retrieval_evidence",
        }
        wrong_artifact = {**agent, "artifact_type": "repair_evidence"}
        with self.assertRaisesRegex(
            ValueError,
            "artifact_type violates its role",
        ):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {"action": "add_subgraph", "agents": [wrong_artifact]}
                ),
                domains,
            )
        self.assertEqual(
            (agent,),
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps({"action": "add_subgraph", "agents": [agent]}),
                domains,
            ),
        )
        wrong_contract = {**agent, "contract": "continue evidence retrieval"}
        with self.assertRaisesRegex(ValueError, "contract violates its role"):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {"action": "add_subgraph", "agents": [wrong_contract]}
                ),
                domains,
            )
        wrong_completion = {
            **agent,
            "completion_condition": "return the best available answer",
        }
        with self.assertRaisesRegex(
            ValueError,
            "completion_condition violates its role",
        ):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {"action": "add_subgraph", "agents": [wrong_completion]}
                ),
                domains,
            )
        missing_completion = dict(agent)
        missing_completion.pop("completion_condition")
        with self.assertRaisesRegex(
            ValueError,
            "completion_condition violates its role",
        ):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {"action": "add_subgraph", "agents": [missing_completion]}
                ),
                domains,
            )
        self.assertEqual(
            (),
            director_live_add_subgraph_relation_candidates(domains, [agent]),
        )
        final_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=[agent],
            )
        )
        self.assertEqual(
            {"type": "array", "maxItems": 0},
            final_schema["properties"]["relations"],
        )
        self.assertEqual(
            {"type": "null"},
            final_schema["properties"]["output_agent_id"],
        )

    def test_add_relation_candidates_are_prospectively_canvas_valid(
        self,
    ) -> None:
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 2,
                "existing_agent_ids": ["node_1", "node_2"],
                "existing_agents": [
                    {"agent_id": "node_1", "role_family": "reasoner"},
                    {"agent_id": "node_2", "role_family": "verifier"},
                ],
                "current_output_agent_id": None,
                "output_role_family": "format",
                "semantic_protocol": HOTPOTQA_SEMANTIC_PROTOCOL,
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ],
                "model_ids": ["qwen"],
                "role_constraints": {
                    "evidence_retriever": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                    },
                    "reasoner": {
                        "execution_modes": ["react"],
                        "allowed_tools": [["qa-retrieval"]],
                    },
                    "verifier": {
                        "execution_modes": ["reasoning"],
                        "allowed_tools": [[]],
                    },
                },
                "admitted_new_role_families": [
                    "evidence_retriever",
                    "reasoner",
                ],
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
        agents = [
            {
                "agent_id": "node_3",
                "model_id": "qwen",
                "contract": "retrieve evidence for the requested relation",
                "role_family": "evidence_retriever",
                "allowed_tools": ["qa-retrieval"],
                "execution_mode": "react",
            },
            {
                "agent_id": "node_4",
                "model_id": "qwen",
                "contract": "bind grounded evidence to the answer slot",
                "role_family": "reasoner",
                "allowed_tools": ["qa-retrieval"],
                "execution_mode": "react",
            },
        ]

        candidates = director_live_add_subgraph_relation_candidates(
            domains,
            agents,
        )

        self.assertFalse(
            any(
                {candidate["source_id"], candidate["target_id"]}
                == {"node_1", "node_2"}
                for candidate in candidates
            ),
            "ADD must not rewrite a relation between two existing Agents",
        )
        self.assertIn(
            {
                "source_id": "node_3",
                "target_id": "node_1",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        self.assertNotIn(
            {
                "source_id": "node_1",
                "target_id": "node_3",
                "source_to_target": True,
                "target_to_source": True,
            },
            candidates,
            "ADD cannot prove a reciprocal new/existing edge executable",
        )
        self.assertIn(
            {
                "source_id": "node_3",
                "target_id": "node_4",
                "source_to_target": True,
                "target_to_source": True,
            },
            candidates,
            "a reciprocal pair declared wholly inside one ADD remains sampled",
        )

        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=agents,
            )
        )
        sampled_candidates = tuple(
            {
                key: branch["properties"][key]["const"]
                for key in (
                    "source_id",
                    "target_id",
                    "source_to_target",
                    "target_to_source",
                )
            }
            for branch in schema["properties"]["relations"]["items"][
                "anyOf"
            ]
        )
        self.assertEqual(candidates, sampled_candidates)

    async def test_model_admissible_v1_receipts_remain_replayable(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                (
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"solver","model_id":"qwen",'
                    '"contract":"solve"}],"relations":[],'
                    '"output_agent_id":"solver"}'
                ),
                '{"action":"finish"}',
            ]
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
            sampling_action_schema_version=(
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1
            ),
        )

        result = await orchestrator.run(env, "task")

        self.assertTrue(result.explicit_finish)
        for request in client.schema_requests:
            actions = director_actions_from_admissible_schema_branch(
                request["action_schema_branch"]
            )
            self.assertTrue(request["action_schema_branch"].startswith("admissible:"))
            self.assertEqual(
                director_model_admissible_sampling_json_schema_text_v1(actions),
                request["action_json_schema"],
            )

    def test_state_conditioned_relation_schema_repeats_full_object_contract(self) -> None:
        schema = json.loads(
            director_state_conditioned_sampling_json_schema_text("add_subgraph")
        )
        relation = schema["properties"]["relations"]["items"]
        expected_required = [
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        ]
        self.assertEqual(2, len(relation["anyOf"]))
        for branch in relation["anyOf"]:
            self.assertEqual(expected_required, branch["required"])
            self.assertEqual(set(expected_required), set(branch["properties"]))
            self.assertFalse(branch["additionalProperties"])
        self.assertEqual(
            {"const": True},
            relation["anyOf"][0]["properties"]["source_to_target"],
        )
        self.assertEqual(
            {"const": True},
            relation["anyOf"][1]["properties"]["target_to_source"],
        )


if __name__ == "__main__":
    unittest.main()
