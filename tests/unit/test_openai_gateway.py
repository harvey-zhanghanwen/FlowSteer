from __future__ import annotations

from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1,
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
    artifact_communication_profile: str = "legacy",
    model_metadata: dict[str, str] | None = None,
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
        metadata={
            "temperature": "0.2",
            "max_tokens": "512",
            **(model_metadata or {}),
        },
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
        artifact_communication_profile=artifact_communication_profile,
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
    def test_versioned_upstream_renders_producer_context_once(self) -> None:
        upstream = UpstreamMessage(
            "source",
            "agent",
            "evidence",
            graph_revision=2,
            artifact_version="source-v2",
            source_model_id="model-a",
            source_contract="Collect the evidence required by the task.",
            source_execution_mode="reasoning",
            source_role_family="evidence",
            source_completion_condition="Return one supported artifact.",
            source_finish_reason="stop",
            input_artifact_provenance=(
                {
                    "source_agent_id": "retriever",
                    "artifact_version": "retriever-v1",
                    "tool_receipts": [
                        {
                            "tool_id": "public_search",
                            "result": {"document_id": "doc-1"},
                        }
                    ],
                },
            ),
        )
        agent_request = replace(
            request(
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1
                )
            ),
            upstream=(upstream, upstream),
        )

        messages = build_agent_messages(agent_request)
        visible = messages[1]["content"]

        self.assertEqual(1, visible.count("[Upstream artifact]"))
        self.assertEqual(1, visible.count("artifact:\nevidence"))
        self.assertIn("artifact_version: source-v2", visible)
        self.assertIn("source_model_id: model-a", visible)
        self.assertIn("source_execution_mode: reasoning", visible)
        self.assertIn("source_finish_reason: stop", visible)
        self.assertIn("source_contract_provenance:", visible)
        self.assertIn("Collect the evidence required by the task.", visible)
        self.assertIn("input_artifact_provenance:", visible)
        self.assertIn('"artifact_version":"retriever-v1"', visible)
        self.assertIn('"document_id":"doc-1"', visible)
        self.assertIn("provenance describing why its artifact", messages[0]["content"])

    def test_versioned_revision_uses_same_peer_envelope(self) -> None:
        peer = UpstreamMessage(
            "peer",
            "agent",
            "peer draft",
            message_type="candidate",
            graph_revision=3,
            artifact_version="peer-v3",
            source_model_id="model-peer",
            source_contract="Review the current candidate.",
            source_execution_mode="reasoning",
            source_finish_reason="stop",
        )
        agent_request = replace(
            request(
                ExecutionPhase.REVISION,
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1
                ),
            ),
            peer_draft=peer,
        )

        visible = build_agent_messages(agent_request)[1]["content"]

        self.assertIn("Peer artifact envelope", visible)
        self.assertIn("artifact_version: peer-v3", visible)
        self.assertIn("source_model_id: model-peer", visible)
        self.assertIn("source_contract_provenance:\nReview the current candidate.", visible)

    def test_healthbench_structured_evidence_projects_only_cited_receipt_rows(
        self,
    ) -> None:
        evidence_span = "Supported guidance for the requested population."
        artifact = json.dumps(
            {
                "schema_version": "healthbench.structured-evidence.v1",
                "status": "supported",
                "summary": "One supported finding.",
                "evidence_items": [
                    {
                        "supported_claim": "The requested finding is supported.",
                        "conditions_or_qualifiers": "For the requested population.",
                        "document_id": "doc-cited",
                        "source": "NCBI PubMed",
                        "title": "Cited source",
                        "date": "2025",
                        "url": "https://example.invalid/cited",
                        "evidence_span": evidence_span,
                    }
                ],
                "uncertainties": [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        receipt = {
            "tool_id": "healthbench-authoritative.search",
            "tool_version": "large-version-string-that-must-not-be-rendered",
            "error_type": None,
            "started_at_monotonic": 10.0,
            "ended_at_monotonic": 11.0,
            "request": {
                "action": "search",
                "arguments": {"query": "requested clinical relation"},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "query": "requested clinical relation",
                    "evidence": [
                        {
                            "document_id": "doc-cited",
                            "source": "NCBI PubMed",
                            "title": "Cited source",
                            "date": "2025",
                            "url": "https://example.invalid/cited",
                            "excerpt": evidence_span + " Additional body text.",
                        },
                        {
                            "document_id": "doc-unrelated",
                            "source": "Other source",
                            "title": "Unrelated source",
                            "date": None,
                            "url": None,
                            "excerpt": "UNRELATED RAW RESULT BODY",
                        },
                    ],
                    "frozen_corpus": {"corpus_rows": 999999},
                },
            },
        }
        duplicate_with_different_timing = {
            **receipt,
            "started_at_monotonic": 20.0,
            "ended_at_monotonic": 21.0,
        }
        upstream = UpstreamMessage(
            "searcher",
            "agent",
            artifact,
            artifact_type="evidence",
            artifact_version="searcher-v1",
            source_execution_mode="react",
            tool_receipts=(receipt, duplicate_with_different_timing),
        )
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "What does the evidence support?"},)
        )
        agent_request = replace(
            request(
                problem=problem,
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2
                ),
            ),
            upstream=(upstream,),
        )

        visible = build_agent_messages(agent_request)[-1]["content"]

        self.assertIn(
            "tool_receipt_projection: "
            "artifact-referenced-healthbench-evidence-v1",
            visible,
        )
        self.assertIn("tool_receipt_projection_status: complete", visible)
        self.assertEqual(1, visible.count('"tool_id":"healthbench-authoritative.search"'))
        self.assertIn('"document_id":"doc-cited"', visible)
        self.assertIn('"query":"requested clinical relation"', visible)
        self.assertNotIn("doc-unrelated", visible)
        self.assertNotIn("UNRELATED RAW RESULT BODY", visible)
        self.assertNotIn("frozen_corpus", visible)
        self.assertNotIn("started_at_monotonic", visible)
        self.assertNotIn("large-version-string-that-must-not-be-rendered", visible)
        self.assertEqual(2, len(agent_request.upstream[0].tool_receipts))
        self.assertEqual(
            "UNRELATED RAW RESULT BODY",
            agent_request.upstream[0].tool_receipts[0]["result"]["value"][
                "evidence"
            ][1]["excerpt"],
        )

    def test_healthbench_structured_profile_compacts_nested_provenance(self) -> None:
        evidence_span = "A bounded exact evidence span."
        evidence_artifact = json.dumps(
            {
                "schema_version": "healthbench.structured-evidence.v1",
                "status": "supported",
                "summary": "Bound evidence.",
                "evidence_items": [
                    {
                        "supported_claim": "A supported claim.",
                        "conditions_or_qualifiers": "One qualifier.",
                        "document_id": "doc-nested",
                        "source": "Source",
                        "title": "Nested title",
                        "date": None,
                        "url": None,
                        "evidence_span": evidence_span,
                    }
                ],
                "uncertainties": [],
            },
            sort_keys=True,
        )
        receipt = {
            "tool_id": "healthbench-authoritative.search",
            "error_type": None,
            "latency_ms": 54321.0,
            "request": {
                "action": "search",
                "arguments": {"query": "nested relation"},
            },
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "query": "nested relation",
                    "evidence": [
                        {
                            "document_id": "doc-nested",
                            "source": "Source",
                            "title": "Nested title",
                            "date": None,
                            "url": None,
                            "excerpt": evidence_span,
                        },
                        {
                            "document_id": "nested-unrelated",
                            "source": "Source",
                            "title": "Noise",
                            "date": None,
                            "url": None,
                            "excerpt": "NESTED UNRELATED BODY",
                        },
                    ],
                },
            },
        }
        provenance = {
            "source_agent_id": "searcher",
            "target_agent_id": "synthesizer",
            "message_type": "artifact",
            "artifact_type": "evidence",
            "artifact_version": "searcher-v1",
            "source_execution_mode": "react",
            "artifact": evidence_artifact,
            "artifact_body": evidence_artifact,
            "content": evidence_artifact,
            "source_contract": "A long nested contract is backend provenance.",
            "request_or_dependency": "A repeated nested dependency.",
            "tool_receipts": [receipt],
            "input_artifact_provenance": [],
        }
        duplicate_provenance = {
            **provenance,
            "tool_receipts": [{**receipt, "latency_ms": 99999.0}],
        }
        upstream = UpstreamMessage(
            "synthesizer",
            "agent",
            "A downstream synthesis.",
            artifact_version="synthesizer-v1",
            source_execution_mode="reasoning",
            input_artifact_provenance=(provenance, duplicate_provenance),
        )
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Use the synthesis."},)
        )
        agent_request = replace(
            request(
                problem=problem,
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2
                ),
            ),
            upstream=(upstream,),
        )

        visible = build_agent_messages(agent_request)[-1]["content"]

        self.assertIn(
            "input_artifact_provenance_projection: compact-structured-evidence-v2",
            visible,
        )
        self.assertIn('"artifact_version":"searcher-v1"', visible)
        self.assertEqual(
            1,
            visible.count('"artifact_version":"searcher-v1"'),
        )
        self.assertIn('"document_id":"doc-nested"', visible)
        self.assertNotIn('"artifact_body"', visible)
        self.assertNotIn('"content"', visible)
        self.assertNotIn('"tool_receipts"', visible)
        self.assertNotIn("source_contract", visible)
        self.assertNotIn("request_or_dependency", visible)
        self.assertNotIn("latency_ms", visible)
        self.assertNotIn("nested-unrelated", visible)
        self.assertNotIn("NESTED UNRELATED BODY", visible)
        self.assertEqual(receipt, provenance["tool_receipts"][0])

    def test_structured_evidence_profile_is_healthbench_scoped(self) -> None:
        receipt = {
            "tool_id": "healthbench-authoritative.search",
            "error_type": None,
            "request": {"action": "search", "arguments": {"query": "q"}},
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "query": "q",
                    "evidence": [],
                },
            },
        }
        upstream = UpstreamMessage(
            "source",
            "agent",
            "plain non-HealthBench artifact",
            source_execution_mode="react",
            tool_receipts=(receipt,),
        )
        agent_request = replace(
            request(
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2
                )
            ),
            upstream=(upstream,),
        )

        visible = build_agent_messages(agent_request)[-1]["content"]

        self.assertIn('"tool_id":"healthbench-authoritative.search"', visible)
        self.assertNotIn("artifact-referenced-healthbench-evidence-v1", visible)

    def test_healthbench_projection_requires_explicit_completed_receipt(
        self,
    ) -> None:
        evidence_span = "Evidence from an incomplete receipt."
        artifact = json.dumps(
            {
                "schema_version": "healthbench.structured-evidence.v1",
                "status": "supported",
                "summary": "A candidate summary.",
                "evidence_items": [
                    {
                        "supported_claim": "A candidate claim.",
                        "conditions_or_qualifiers": "",
                        "document_id": "doc-incomplete",
                        "source": "Source",
                        "title": "Title",
                        "date": None,
                        "url": None,
                        "evidence_span": evidence_span,
                    }
                ],
                "uncertainties": [],
            }
        )
        incomplete_receipt = {
            "tool_id": "healthbench-authoritative.search",
            "error_type": None,
            "request": {"action": "search", "arguments": {"query": "q"}},
            "result": {
                "value": {
                    "operation": "search",
                    "query": "q",
                    "evidence": [
                        {
                            "document_id": "doc-incomplete",
                            "source": "Source",
                            "title": "Title",
                            "date": None,
                            "url": None,
                            "excerpt": evidence_span,
                        }
                    ],
                }
            },
        }
        upstream = UpstreamMessage(
            "searcher",
            "agent",
            artifact,
            source_execution_mode="react",
            tool_receipts=(incomplete_receipt,),
        )
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Use completed evidence only."},)
        )
        agent_request = replace(
            request(
                problem=problem,
                artifact_communication_profile=(
                    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2
                ),
            ),
            upstream=(upstream,),
        )

        visible = build_agent_messages(agent_request)[-1]["content"]

        self.assertIn(
            "tool_receipt_projection_status: unavailable-no-receipt-match",
            visible,
        )
        self.assertNotIn("evidence_receipts:", visible)
        self.assertNotIn('"operation":"search"', visible)

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
        self.assertIn(
            "HealthBench Professional execution protocol",
            messages[0]["content"],
        )
        self.assertIn("ambiguous shorthand", messages[0]["content"])
        self.assertIn("entity-property binding", messages[0]["content"])
        self.assertIn("do not introduce a new decisive claim", messages[0]["content"])
        self.assertIn("do not endorse unsafe content", messages[0]["content"])
        self.assertIn("must not expose Agent IDs", messages[0]["content"])
        self.assertNotIn("rubric", "\n".join(item["content"] for item in messages))

    def test_versioned_healthbench_protocol_treats_contract_as_instruction(self) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "What level should be used?"},)
        )
        messages = build_agent_messages(
            request(
                problem=problem,
                model_metadata={
                    "healthbench_execution_protocol": "contract-is-not-evidence.v2"
                },
            )
        )
        system = messages[0]["content"]

        self.assertIn("contract describes work to perform, not evidence", system)
        self.assertIn("answer every explicit part", system)
        self.assertNotIn("rubric", system.casefold())

    def test_v3_healthbench_react_output_complete_is_user_facing(self) -> None:
        problem = render_model_visible_conversation(
            (
                {"role": "user", "content": "Explain the options."},
                {"role": "assistant", "content": "Which aspect matters most?"},
                {
                    "role": "user",
                    "content": "Compare both options and give next steps.",
                },
            )
        )
        system = build_agent_messages(
            request(
                problem=problem,
                execution_mode="react",
                is_output_agent=True,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.v3"
                    )
                },
            )
        )[0]["content"]

        self.assertIn("StructuredAction JSON object", system)
        self.assertIn("contract describes work to perform, not evidence", system)
        self.assertIn("arguments must contain only the admitted `query` field", system)
        self.assertIn("`arguments.value` must be the complete, self-contained", system)
        self.assertIn("answers every explicit request", system)
        self.assertIn("final user message of the original conversation", system)
        self.assertIn("must not expose AgentGraph", system)
        self.assertIn("use a short query", system)
        self.assertIn("broaden it by removing restrictive terms", system)

    def test_v3_healthbench_output_suffix_is_strictly_version_and_role_scoped(
        self,
    ) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Compare both options."},)
        )
        v2_output = build_agent_messages(
            request(
                problem=problem,
                execution_mode="react",
                is_output_agent=True,
                model_metadata={
                    "healthbench_execution_protocol": "contract-is-not-evidence.v2"
                },
            )
        )[0]["content"]
        v3_non_output = build_agent_messages(
            request(
                problem=problem,
                execution_mode="react",
                is_output_agent=False,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.v3"
                    )
                },
            )
        )[0]["content"]
        qa_output = build_agent_messages(
            request(
                problem="Which city hosted the event?",
                execution_mode="react",
                is_output_agent=True,
                semantic_protocol="qa_verified_answer_lineage_v2",
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.v3"
                    )
                },
            )
        )[0]["content"]

        suffix = "`arguments.value` must be the complete, self-contained"
        self.assertNotIn(suffix, v2_output)
        self.assertNotIn(suffix, v3_non_output)
        self.assertNotIn(suffix, qa_output)
        self.assertNotIn("use a short query", v2_output)
        self.assertIn("use a short query", v3_non_output)
        self.assertIn("contract describes work to perform, not evidence", v3_non_output)

    def test_v4_healthbench_protocol_binds_every_explicit_answer_slot(self) -> None:
        problem = render_model_visible_conversation(
            (
                {
                    "role": "user",
                    "content": (
                        "Explain AC and distinguish the procedural locations."
                    ),
                },
            )
        )
        system = build_agent_messages(
            request(
                problem=problem,
                execution_mode="reasoning",
                is_output_agent=False,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.slot-binding.v4"
                    )
                },
            )
        )[0]["content"]

        self.assertIn("each explicit noun phrase and unresolved abbreviation", system)
        self.assertIn("exact entity, attribute, condition, and procedural stage", system)
        self.assertIn("never substitute a related but different property", system)
        self.assertIn("access or entry site", system)
        self.assertIn("target or tip position", system)
        self.assertIn("coverage or treatment level", system)
        self.assertNotIn("rubric", system.casefold())

    def test_v4_healthbench_react_output_retains_complete_action_suffix(self) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Address every requested item."},)
        )
        system = build_agent_messages(
            request(
                problem=problem,
                execution_mode="react",
                is_output_agent=True,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.slot-binding.v4"
                    )
                },
            )
        )[0]["content"]

        self.assertIn("`arguments.value` must be the complete, self-contained", system)
        self.assertIn("answers every explicit request", system)
        self.assertIn("each explicit noun phrase and unresolved abbreviation", system)

    def test_v4_slot_binding_is_versioned_and_healthbench_scoped(self) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Explain AC."},)
        )
        v3_system = build_agent_messages(
            request(
                problem=problem,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.v3"
                    )
                },
            )
        )[0]["content"]
        non_healthbench_system = build_agent_messages(
            request(
                problem="Explain AC.",
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.slot-binding.v4"
                    )
                },
            )
        )[0]["content"]

        slot_binding = "each explicit noun phrase and unresolved abbreviation"
        self.assertNotIn(slot_binding, v3_system)
        self.assertNotIn("access or entry site", v3_system)
        self.assertNotIn(slot_binding, non_healthbench_system)
        self.assertNotIn("access or entry site", non_healthbench_system)

    def test_v5_healthbench_protocol_closes_tool_and_output_semantics(self) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "Translate this treatment plan."},)
        )
        system = build_agent_messages(
            request(
                problem=problem,
                execution_mode="react",
                is_output_agent=True,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete."
                        "slot-binding.closure.v5"
                    )
                },
            )
        )[0]["content"]

        self.assertIn("evidence-derived finding", system)
        self.assertIn("never complete with a query, keyword list", system)
        self.assertIn("Resolve internal contradictions", system)
        self.assertIn("urgent or in-person evaluation", system)
        self.assertIn("adding the necessary safety correction", system)
        self.assertIn("apply the correction", system)
        self.assertIn("`arguments.value` must be the complete", system)
        self.assertNotIn("rubric", system.casefold())

        v4_system = build_agent_messages(
            request(
                problem=problem,
                model_metadata={
                    "healthbench_execution_protocol": (
                        "contract-is-not-evidence.output-complete.slot-binding.v4"
                    )
                },
            )
        )[0]["content"]
        self.assertNotIn("evidence-derived finding", v4_system)

    def test_legacy_healthbench_catalog_keeps_v1_execution_protocol(self) -> None:
        problem = render_model_visible_conversation(
            ({"role": "user", "content": "What level should be used?"},)
        )
        system = build_agent_messages(request(problem=problem))[0]["content"]

        self.assertNotIn("contract describes work to perform, not evidence", system)

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
        self.assertIn("unique Output Agent", messages[0]["content"])
        self.assertIn("return the final task answer", messages[0]["content"])
        self.assertNotIn("<answer>", messages[0]["content"])

    def test_generic_contract_separates_intermediate_and_output_protocols(self) -> None:
        messages = build_agent_messages(request(is_output_agent=False))
        system = messages[0]["content"]
        output_system = build_agent_messages(
            request(is_output_agent=True)
        )[0]["content"]
        self.assertIn("intermediate AgentGraph node", system)
        self.assertIn("Preserve the task's original relation", system)
        self.assertIn("unique Output Agent", output_system)
        self.assertIn("return the final task answer", output_system)
        self.assertIn("Do not expose AgentGraph identifiers", output_system)
        self.assertNotIn("direct semantic predecessor", system)
        self.assertNotIn("unique Output Agent", system)
        self.assertNotEqual(system, output_system)

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
                "chat_template_enable_thinking": None,
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

    def test_qwen_chat_template_thinking_can_be_enabled_explicitly(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"chat_template_enable_thinking": "true"},
            ),
        )
        payload = OpenAICompatibleGateway().request_payload(item)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True},
        )

    def test_qwen_thinking_budget_is_added_to_visible_budget(
        self,
    ) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={
                    "chat_template_enable_thinking": "true",
                    "max_tokens": "4096",
                    "thinking_budget": "4096",
                },
            ),
        )
        payload = OpenAICompatibleGateway().request_payload(item)

        self.assertEqual(8192, payload["max_tokens"])
        self.assertEqual(
            {"enable_thinking": True},
            payload["chat_template_kwargs"],
        )

    def test_qwen_thinking_budget_requires_thinking(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={
                    "chat_template_enable_thinking": "false",
                    "thinking_budget": "1024",
                },
            ),
        )
        with self.assertRaisesRegex(
            OpenAICompatibleGatewayError,
            "thinking_budget requires",
        ):
            OpenAICompatibleGateway().request_payload(item)

    async def test_qwen_thinking_response_records_counts_not_reasoning_body(self) -> None:
        item = request()
        object.__setattr__(
            item,
            "model",
            ModelSpec(
                "model",
                "provider",
                model_name="supervisor_theta",
                metadata={"chat_template_enable_thinking": "true"},
            ),
        )
        gateway = OpenAICompatibleGateway(max_retries=0)
        gateway._post_json = lambda *_: {  # type: ignore[method-assign]
            "id": "req-thinking",
            "model": "supervisor_theta",
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning_content": "private reasoning body",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 11},
            },
        }

        response = await gateway.generate(item)

        self.assertEqual(response.text, "final answer")
        self.assertIs(response.metadata["reasoning_content_present"], True)
        self.assertEqual(response.metadata["reasoning_content_characters"], 22)
        self.assertEqual(response.metadata["reasoning_tokens"], 11)
        self.assertNotIn(
            "private reasoning body",
            json.dumps(dict(response.metadata)),
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
