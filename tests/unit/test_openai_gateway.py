from __future__ import annotations

from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.healthbench_professional_adapter import (
    render_model_visible_conversation,
)
from src.interactive.openai_gateway import (
    OpenAICompatibleGateway,
    OpenAICompatibleGatewayError,
    MASKED_UPSTREAM_CONTENT,
    build_agent_messages,
)


def request(
    phase: ExecutionPhase = ExecutionPhase.SINGLE,
    *,
    keyed: bool = False,
    is_output_agent: bool = True,
    is_format_agent: bool = False,
    is_format_predecessor: bool = False,
    execution_mode: str = "reasoning",
    role_family: str | None = None,
    problem: str = "Solve the task",
    upstream_artifact: str = "evidence",
    semantic_protocol: str = "none",
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL,
) -> AgentRequest:
    provider = ProviderSpec(
        "provider",
        kind="openai-compatible",
        endpoint="https://example.invalid/v1",
        api_key_env="TEST_FLOWSTEER_KEY" if keyed else None,
    )
    model = ModelSpec(
        "model",
        "provider",
        model_name="remote-model-id",
        metadata={"temperature": "0.2", "max_tokens": "512"},
    )
    return AgentRequest(
        request_id="run:1:agent:single",
        run_id="run",
        graph_revision=1,
        problem=problem,
        agent=AgentNode(
            "agent",
            "model",
            "verify carefully",
            role_family=role_family,
            execution_mode=execution_mode,
        ),
        model=model,
        provider=provider,
        phase=phase,
        is_output_agent=is_output_agent,
        is_format_agent=is_format_agent,
        is_format_predecessor=is_format_predecessor,
        communication_condition=communication_condition,
        semantic_protocol=semantic_protocol,
        upstream=(
            UpstreamMessage(
                "source",
                "agent",
                upstream_artifact,
                graph_revision=1,
                request_or_dependency="verify carefully",
            ),
        ),
        own_draft="own" if phase is ExecutionPhase.REVISION else None,
        peer_draft=(
            UpstreamMessage(
                "peer",
                "agent",
                "peer draft",
                message_type="candidate",
                graph_revision=1,
                request_or_dependency="verify carefully",
            )
            if phase is ExecutionPhase.REVISION
            else None
        ),
    )


class MessageTests(unittest.TestCase):
    def test_healthbench_conversation_preserves_native_roles_and_content(self) -> None:
        problem = render_model_visible_conversation(
            (
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "Prior response."},
                {"role": "user", "content": "Follow-up question."},
            )
        )
        messages = build_agent_messages(request(problem=problem))

        self.assertEqual("system", messages[0]["role"])
        self.assertEqual(
            [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "Prior response."},
                {"role": "user", "content": "Follow-up question."},
            ],
            messages[1:4],
        )
        self.assertEqual("user", messages[4]["role"])
        self.assertIn("AgentGraph execution context", messages[4]["content"])
        self.assertNotIn("rubric", "\n".join(item["content"] for item in messages))

    def test_semantic_lineage_projects_only_artifact_referenced_read_receipts(
        self,
    ) -> None:
        artifact = json.dumps(
            {
                "question_scope": "Where was the person born?",
                "entity_identity": {
                    "question_surface": "the person",
                    "evidence_surface": "The person",
                },
                "target_relation": "was born",
                "answer_type_constraint": "location",
                "evidence_proposition": {
                    "subject": "The person",
                    "predicate": "was born in",
                    "object_or_attribute_value": "East Ward",
                },
                "evidence_span": "The person was born in East Ward.",
                "passage_id": "p2",
            }
        )
        search_receipt = {
            "tool_id": "qa-retrieval",
            "error_type": None,
            "request": {
                "action": "search",
                "arguments": {"query": "unrelated search", "limit": 5},
            },
            "result": {"value": {"operation": "search", "passage_ids": ["p1"]}},
        }

        def read_receipt(passage_id: str, text: str) -> dict[str, object]:
            return {
                "tool_id": "qa-retrieval",
                "error_type": None,
                "request": {
                    "action": "read",
                    "arguments": {"passage_id": passage_id},
                },
                "result": {
                    "value": {
                        "operation": "read",
                        "passage_id": passage_id,
                        "passage": {
                            "passage_id": passage_id,
                            "title": passage_id,
                            "text": text,
                        },
                    }
                },
            }

        upstream = UpstreamMessage(
            "retriever",
            "agent",
            artifact,
            artifact_type="evidence",
            tool_receipts=(
                search_receipt,
                read_receipt("p1", "Unrelated public passage."),
                read_receipt("p2", "The person was born in East Ward."),
            ),
        )
        agent_request = replace(
            request(
                is_output_agent=False,
                role_family="reasoner",
                semantic_protocol="qa_verified_answer_lineage_v2",
            ),
            upstream=(upstream,),
        )

        messages = build_agent_messages(agent_request)
        visible = messages[1]["content"]

        self.assertIn(
            "tool_receipt_projection: artifact-referenced-successful-reads",
            visible,
        )
        self.assertIn('"passage_id":"p2"', visible)
        self.assertNotIn('"passage_id":"p1"', visible)
        self.assertNotIn("unrelated search", visible)
        self.assertEqual(3, len(agent_request.upstream[0].tool_receipts))

    def test_revision_prompt_uses_immutable_own_and_peer_drafts(self) -> None:
        messages = build_agent_messages(request(ExecutionPhase.REVISION))
        text = messages[1]["content"]
        self.assertIn("Your draft:\nown", text)
        self.assertIn("Peer artifact envelope", text)
        self.assertIn("source_agent: peer", text)
        self.assertIn("message_type: candidate", text)
        self.assertIn("artifact:\npeer draft", text)
        self.assertIn("External upstream messages", text)
        self.assertIn("source_agent: source", text)
        self.assertIn("target_agent: agent", text)
        self.assertIn("request_or_dependency: verify carefully", text)
        self.assertIn(
            "Preserve the output form and level of detail",
            messages[0]["content"],
        )
        self.assertNotIn("<answer>", messages[0]["content"])
        self.assertIn("exactly one listed executable action", messages[0]["content"])

    def test_intermediate_contract_forbids_task_level_answer_tag(self) -> None:
        messages = build_agent_messages(request(is_output_agent=False))
        system = messages[0]["content"]
        self.assertIn("intermediate AgentGraph node", system)
        self.assertIn("do not use <answer> tags", system)
        self.assertIn("original relation, qualifiers, comparison criterion", system)
        self.assertIn("independently reconstruct that evidence", system)
        self.assertNotIn("direct semantic predecessor", system)
        self.assertNotIn("unique Output Agent", system)

    def test_format_predecessor_has_explicit_semantic_handoff_contract(self) -> None:
        messages = build_agent_messages(
            request(is_output_agent=False, is_format_predecessor=True)
        )
        system = messages[0]["content"]

        self.assertIn("direct semantic predecessor", system)
        self.assertIn("Candidate answer: ...", system)
        self.assertIn("only the answer value", system)
        self.assertIn("Evidence: ...", system)
        self.assertIn("Do not use <answer> tags", system)
        for treatment_only_rule in (
            "expected answer type and granularity",
            "minimal sufficient answer span",
            "full proper name",
            "date, number expression, unit",
            "qualifier",
            "alias",
            "abbreviation",
            "only yes or no",
        ):
            self.assertNotIn(treatment_only_rule, system)

    def test_react_format_predecessor_applies_handoff_to_complete_action(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                is_format_predecessor=True,
                execution_mode="react",
            )
        )
        system = messages[0]["content"]

        self.assertIn("StructuredAction JSON object", system)
        self.assertIn("complete action", system)
        self.assertIn("Candidate answer: ...", system)
        self.assertIn("Evidence: ...", system)

    def test_react_output_turn_uses_structured_action_not_answer_wrapper(self) -> None:
        messages = build_agent_messages(request(execution_mode="react"))
        system = messages[0]["content"]

        self.assertIn("StructuredAction JSON object", system)
        self.assertIn("complete action", system)
        self.assertIn("state-conditioned action domain", system)
        self.assertIn("only its declared keys in arguments", system)
        self.assertIn("Do not emit <answer> tags", system)
        self.assertNotIn("unique Output Agent", system)

    def test_reasoner_aligns_fact_propositions_to_answer_slot(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                role_family="reasoner",
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )
        )
        system = messages[0]["content"]

        self.assertIn("semantic Reasoner", system)
        self.assertIn("subject/entity", system)
        self.assertIn("predicate/relation", system)
        self.assertIn("answer slot actually requested", system)
        self.assertIn("evidence_propositions", system)
        self.assertIn("answer_slot", system)
        self.assertIn("answer_cardinality", system)
        self.assertIn("proposition_index selects one item", system)
        self.assertIn("complete, evidence-aligned referential surface", system)
        self.assertIn("never permits truncating that entity mention", system)
        self.assertIn(
            "full possessor entity mention immediately before the possessive marker",
            system,
        )
        self.assertIn("title, honorific, or name suffix", system)
        self.assertIn("reclassified as an unrequested qualifier", system)
        self.assertIn("even when a shorter form is coreferential", system)
        self.assertIn("multi_hop_chain", system)
        self.assertIn("candidate_answer", system)
        self.assertIn("unexpectedly equal values", system)

    def test_verifier_checks_candidate_without_replacing_it(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                is_format_predecessor=True,
                role_family="verifier",
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )
        )
        system = messages[0]["content"]

        self.assertIn("semantic Verifier", system)
        self.assertIn("explicit database or retrieved evidence", system)
        self.assertIn("entity is bound to the correct attribute/value", system)
        self.assertIn("multi-hop bridge is complete", system)
        self.assertIn("scope was not narrowed", system)
        self.assertIn("answer type and cardinality", system)
        self.assertIn(
            "answer surface is one complete, evidence-aligned referential surface",
            system,
        )
        self.assertIn(
            "full possessor entity mention immediately before the possessive marker",
            system,
        )
        self.assertIn("title, honorific, or name suffix", system)
        self.assertIn("incomplete possessor entity mention", system)
        self.assertIn("any candidate that shortens the full possessor mention", system)
        self.assertIn("explicit identity binding", system)
        self.assertIn("must not select, replace, canonicalize", system)
        self.assertIn("Answer type cardinality correct:", system)
        self.assertIn("Minimal answer surface:", system)
        self.assertIn("Alias binding correct:", system)
        self.assertIn("Verification status:", system)
        self.assertIn("repair_required", system)
        self.assertNotIn("direct semantic predecessor", system)

    def test_react_is_execution_schedule_not_reasoner_role(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                execution_mode="react",
                role_family="reasoner",
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )
        )
        system = messages[0]["content"]

        self.assertIn("StructuredAction JSON object", system)
        self.assertIn("ReAct is only this node's execution schedule", system)
        self.assertIn("semantic Reasoner", system)
        self.assertIn("complete, evidence-aligned referential surface", system)
        self.assertIn(
            "full possessor entity mention immediately before the possessive marker",
            system,
        )
        self.assertIn("title, honorific, or name suffix", system)

    def test_unified_qa_reasoner_uses_grounded_entity_relation_protocol(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                execution_mode="react",
                role_family="reasoner",
                semantic_protocol="qa_verified_answer_lineage_v2",
                problem="Which city hosted the event?",
            )
        )
        system = messages[0]["content"]

        self.assertIn("semantic Reasoner", system)
        self.assertIn("ReAct is only this node's execution schedule", system)
        self.assertIn("entity identity", system)
        self.assertIn("requested relation", system)
        self.assertIn("Tool receipts", system)
        self.assertIn("explicit identity binding", system)
        self.assertIn(
            "copy each entity surface exactly as it occurs",
            system,
        )
        self.assertIn(
            "separate evidence-supported identity proposition",
            system,
        )
        self.assertNotIn("ReAct Agent", system)

    def test_unified_qa_evidence_retriever_owns_only_provenance(self) -> None:
        messages = build_agent_messages(
            request(
                is_output_agent=False,
                execution_mode="react",
                role_family="evidence_retriever",
                semantic_protocol="qa_verified_answer_lineage_v2",
                problem="Which city does David Soul come from?",
            )
        )
        system = messages[0]["content"]

        self.assertIn("ReAct is only this node's execution schedule", system)
        self.assertIn("You are the Evidence Retriever", system)
        self.assertIn("own only public retrieval provenance", system)
        self.assertIn("Cite a successful read receipt", system)
        self.assertIn(
            "do not select or emit candidate_answer, answer_slot, or final_answer",
            system,
        )
        self.assertNotIn("You are the semantic Reasoner", system)

    def test_unified_qa_verifier_and_formatter_preserve_candidate(self) -> None:
        verifier_messages = build_agent_messages(
            request(
                is_output_agent=False,
                is_format_predecessor=True,
                role_family="verifier",
                semantic_protocol="qa_verified_answer_lineage_v2",
            )
        )
        verifier_system = verifier_messages[0]["content"]
        self.assertIn("semantic Verifier", verifier_system)
        self.assertIn("entity-to-relation binding", verifier_system)
        self.assertIn("must not select, replace, canonicalize", verifier_system)
        self.assertIn(
            "exactly copies the selected proposition argument",
            verifier_system,
        )
        self.assertIn(
            "do not reject that complete mention merely because a shorter alias",
            verifier_system,
        )
        self.assertIn("Verification status:", verifier_system)

        original_question = "PRIVATE QUESTION SHOULD NOT REACH FORMATTER"
        formatter_messages = build_agent_messages(
            request(
                is_format_agent=True,
                role_family="format",
                problem=original_question,
                upstream_artifact=(
                    "Candidate answer: Florence\n"
                    "Evidence supported: true\n"
                    "Entity attribute binding correct: true\n"
                    "Multi-hop complete: true\nScope preserved: true\n"
                    "Answer type cardinality correct: true\n"
                    "Minimal answer surface: true\nAlias binding correct: true\n"
                    "Verification status: supported"
                ),
                semantic_protocol="qa_verified_answer_lineage_v2",
            )
        )
        formatter_system = formatter_messages[0]["content"]
        formatter_user = formatter_messages[1]["content"]
        self.assertIn("terminal FlowSteer Format Operator", formatter_system)
        self.assertIn("Copy character-for-character", formatter_user)
        self.assertIn("never change an alias, abbreviation", formatter_user)
        self.assertNotIn(original_question, formatter_system)
        self.assertNotIn(original_question, formatter_user)

    def test_format_agent_extracts_one_upstream_solution_without_resolving(self) -> None:
        original_question = "SECRET ORIGINAL QUESTION"
        messages = build_agent_messages(
            request(
                is_format_agent=True,
                role_family="format",
                problem=original_question,
                upstream_artifact=(
                    "Candidate answer: The Joshua Tree\n"
                    "Evidence supported: true\n"
                    "Entity attribute binding correct: true\nMulti-hop complete: true\n"
                    "Scope preserved: true\n"
                    "Answer type cardinality correct: true\n"
                    "Minimal answer surface: true\n"
                    "Alias binding correct: true\nVerification status: supported"
                ),
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )
        )
        system = messages[0]["content"]
        user = messages[1]["content"]

        self.assertIn("terminal FlowSteer Format Operator", system)
        self.assertIn("exactly one routed upstream artifact", system)
        self.assertIn("do not solve, verify, or extend", system)
        self.assertIn("has ALREADY been computed as: Candidate answer", user)
        self.assertIn("Your ONLY job is to extract the final answer", user)
        self.assertIn("OUTPUT ANSWER VALUE ONLY", user)
        self.assertIn("PRESERVE ORIGINAL NON-MATH FORMATS", user)
        self.assertIn("exactly one <answer>...</answer> wrapper", user)
        self.assertIn("return exactly <answer></answer>", user)
        self.assertIn("Copy character-for-character", user)
        self.assertIn("value following its single `Candidate answer:` label", user)
        self.assertIn("never change an alias, abbreviation", user)
        self.assertNotIn("possessor entity mention", system)
        self.assertNotIn("possessor entity mention", user)
        self.assertNotIn(original_question, system)
        self.assertNotIn(original_question, user)
        self.assertNotIn("Contract:\nverify carefully", system)
        self.assertNotIn("request_or_dependency: verify carefully", user)
        self.assertNotIn("External upstream messages", user)
        self.assertNotIn("source_agent:", user)

    def test_hotpot_semantic_roles_do_not_change_default_reasoner_prompt(self) -> None:
        messages = build_agent_messages(
            request(is_output_agent=False, role_family="reasoner")
        )
        self.assertNotIn("semantic Reasoner", messages[0]["content"])
        self.assertNotIn("Evidence propositions:", messages[0]["content"])

    def test_masked_condition_preserves_receipt_but_masks_visible_messages(self) -> None:
        item = request(
            ExecutionPhase.REVISION,
            communication_condition=CommunicationCondition.UPSTREAM_MASKED,
        )
        messages = build_agent_messages(item)
        visible = messages[1]["content"]
        self.assertEqual("evidence", item.upstream[0].content)
        self.assertEqual("peer draft", item.peer_draft.content)  # type: ignore[union-attr]
        self.assertNotIn("\nevidence", visible)
        self.assertNotIn("artifact:\npeer draft", visible)
        self.assertEqual(2, visible.count(MASKED_UPSTREAM_CONTENT))
        self.assertIn("source_agent: source", visible)
        self.assertIn("source_agent: peer", visible)
        self.assertIn("Your draft:\nown", visible)

    def test_revision_without_drafts_is_rejected(self) -> None:
        broken = request(ExecutionPhase.SINGLE)
        object.__setattr__(broken, "phase", ExecutionPhase.REVISION)
        with self.assertRaises(OpenAICompatibleGatewayError):
            build_agent_messages(broken)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def test_request_generation_seed_overrides_gateway_default(self) -> None:
        item = request()
        item = replace(
            item,
            model=replace(
                item.model,
                metadata={
                    **dict(item.model.metadata),
                    "generation_seed": "18446744073709551615",
                    "temperature": "1.0",
                    "top_p": "1.0",
                },
            ),
        )
        payload = OpenAICompatibleGateway(default_seed=17).request_payload(item)
        self.assertEqual(18446744073709551615, payload["seed"])
        self.assertEqual(1.0, payload["temperature"])
        self.assertEqual(1.0, payload["top_p"])

    def test_generation_seed_rejects_values_outside_uint64(self) -> None:
        item = request()
        item = replace(
            item,
            model=replace(
                item.model,
                metadata={
                    **dict(item.model.metadata),
                    "generation_seed": str(2**64),
                },
            ),
        )
        with self.assertRaisesRegex(
            OpenAICompatibleGatewayError,
            "unsigned 64-bit",
        ):
            OpenAICompatibleGateway().request_payload(item)
        with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
            OpenAICompatibleGateway(default_seed=2**64)

    async def test_local_sglang_signed_seed_preserves_scientific_receipt(
        self,
    ) -> None:
        scientific_seed = (1 << 63) + 29
        item = request()
        item = replace(
            item,
            provider=replace(
                item.provider,
                metadata={
                    "sampling_backend": "sglang",
                    "deployment_locality": "local",
                },
            ),
            model=replace(
                item.model,
                metadata={
                    **dict(item.model.metadata),
                    "generation_seed": str(scientific_seed),
                    "supports_top_k": "true",
                },
            ),
        )
        gateway = OpenAICompatibleGateway(max_retries=0)
        captured = {}

        def fake_post(url, api_key, payload):
            captured["payload"] = payload
            return {
                "id": "req-local",
                "model": "supervisor_theta",
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

        gateway._post_json = fake_post  # type: ignore[method-assign]
        response = await gateway.generate(item)

        self.assertEqual(29, captured["payload"]["seed"])
        self.assertEqual(scientific_seed, response.metadata["generation_seed"])
        self.assertEqual(29, response.metadata["backend_sampling_seed"])
        self.assertEqual(
            {"seed": scientific_seed, "backend_seed": 29},
            {
                key: response.metadata["requested_sampling"][key]
                for key in ("seed", "backend_seed")
            },
        )

    def test_top_k_is_forwarded_only_for_declared_local_sglang(self) -> None:
        item = request()
        declared_provider = replace(
            item.provider,
            metadata={
                "sampling_backend": "sglang",
                "deployment_locality": "local",
            },
        )
        declared_model = replace(
            item.model,
            metadata={
                **dict(item.model.metadata),
                "supports_top_k": "true",
                "top_k": "-1",
            },
        )
        local_payload = OpenAICompatibleGateway().request_payload(
            replace(item, provider=declared_provider, model=declared_model)
        )
        self.assertEqual(-1, local_payload["top_k"])

        remote_payload = OpenAICompatibleGateway().request_payload(
            replace(item, model=declared_model)
        )
        self.assertNotIn("top_k", remote_payload)

    def test_repetition_penalty_is_forwarded_only_for_declared_local_sglang(
        self,
    ) -> None:
        item = request()
        declared_provider = replace(
            item.provider,
            metadata={
                "sampling_backend": "sglang",
                "deployment_locality": "local",
            },
        )
        declared_model = replace(
            item.model,
            metadata={
                **dict(item.model.metadata),
                "supports_repetition_penalty": "true",
                "repetition_penalty": "1.05",
            },
        )
        local_payload = OpenAICompatibleGateway().request_payload(
            replace(item, provider=declared_provider, model=declared_model)
        )
        self.assertEqual(1.05, local_payload["repetition_penalty"])

        remote_payload = OpenAICompatibleGateway().request_payload(
            replace(item, model=declared_model)
        )
        self.assertNotIn("repetition_penalty", remote_payload)

    def test_declared_repetition_penalty_requires_supported_range(self) -> None:
        item = request()
        provider = replace(
            item.provider,
            metadata={
                "sampling_backend": "sglang",
                "deployment_locality": "local",
            },
        )
        for invalid_value in ("0", "-0.1", "2.01", "not-a-number"):
            with self.subTest(invalid_value=invalid_value):
                model = replace(
                    item.model,
                    metadata={
                        **dict(item.model.metadata),
                        "supports_repetition_penalty": "true",
                        "repetition_penalty": invalid_value,
                    },
                )
                with self.assertRaisesRegex(
                    OpenAICompatibleGatewayError,
                    "repetition_penalty",
                ):
                    OpenAICompatibleGateway().request_payload(
                        replace(item, provider=provider, model=model)
                    )

    async def test_repetition_penalty_is_persisted_in_requested_sampling(
        self,
    ) -> None:
        item = request()
        item = replace(
            item,
            provider=replace(
                item.provider,
                metadata={
                    "sampling_backend": "sglang",
                    "deployment_locality": "local",
                },
            ),
            model=replace(
                item.model,
                metadata={
                    **dict(item.model.metadata),
                    "supports_repetition_penalty": "true",
                    "repetition_penalty": "1.05",
                },
            ),
        )
        gateway = OpenAICompatibleGateway(max_retries=0)
        gateway._post_json = lambda *_: {  # type: ignore[method-assign]
            "id": "req-repeat",
            "model": "supervisor_theta",
            "choices": [
                {"message": {"content": "answer"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }

        response = await gateway.generate(item)

        self.assertEqual(
            1.05,
            response.metadata["requested_sampling"]["repetition_penalty"],
        )

    async def test_request_and_response_mapping(self) -> None:
        gateway = OpenAICompatibleGateway(max_retries=0, default_seed=17)
        captured = {}

        def fake_post(url, api_key, payload):
            captured.update(url=url, api_key=api_key, payload=payload)
            return {
                "id": "req-1",
                "model": "actual-model",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

        gateway._post_json = fake_post  # type: ignore[method-assign]
        response = await gateway.generate(request())
        self.assertEqual(response.text, "answer")
        self.assertEqual(captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(captured["api_key"], "EMPTY")
        self.assertEqual(captured["payload"]["model"], "remote-model-id")
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["payload"]["seed"], 17)
        self.assertEqual(response.metadata["total_tokens"], 12)
        self.assertGreaterEqual(response.metadata["latency_ms"], 0.0)
        self.assertEqual(response.metadata["attempt_count"], 1)
        self.assertEqual(response.metadata["generation_seed"], 17)
        self.assertEqual(response.metadata["request_status"], "completed")
        self.assertEqual(
            response.metadata["requested_sampling"],
            {
                "temperature": 0.2,
                "top_p": 1.0,
                "top_k": None,
                "repetition_penalty": None,
                "max_tokens": 512,
                "seed": 17,
            },
        )

    async def test_missing_credential_names_variable_without_printing_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OpenAICompatibleGatewayError, "TEST_FLOWSTEER_KEY"):
                await OpenAICompatibleGateway(max_retries=0).generate(request(keyed=True))

    async def test_malformed_provider_response_is_rejected(self) -> None:
        gateway = OpenAICompatibleGateway(max_retries=0)
        gateway._post_json = lambda *_: {"choices": []}  # type: ignore[method-assign]
        with self.assertRaises(OpenAICompatibleGatewayError):
            await gateway.generate(request())

    def test_qwen_chat_template_thinking_toggle_is_explicit(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"chat_template_enable_thinking": "false"},
            ),
        )
        payload = OpenAICompatibleGateway().request_payload(item)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_skillflow_response_schema_is_forwarded(self) -> None:
        item = request()
        schema = {
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"const": "complete"}},
            "additionalProperties": False,
        }
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"response_json_schema": json.dumps(schema)},
            ),
        )

        payload = OpenAICompatibleGateway().request_payload(item)

        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(
            payload["response_format"]["json_schema"],
            {"name": "skillev_action", "schema": schema, "strict": True},
        )


if __name__ == "__main__":
    unittest.main()
