from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.qa_tool_adapter import (
    QARetrievalReactExecutionAdapter,
    QA_RETRIEVAL_TOOL_ID,
    QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    _factual_retrieval_attempt_records,
    _factual_strategy_proofs,
    _relation_surface_replacement_classes,
    build_qa_tool_registry,
)
from src.interactive.react_execution import ReactExecutionError


QUESTION = (
    "In which decade did Chart Weekly magazine first publish an American "
    "hit chart?"
)
QUERIES = (
    "Chart Weekly magazine first publish American hit chart decade",
    "Chart Weekly magazine first published American hit chart decade",
    "Chart Weekly magazine first publication American hit chart decade",
    "Chart Weekly magazine first publication American hit chart decade publish",
    "Chart Weekly magazine first publication American hit parade decade publish",
)


def search_observation(
    query: str,
    *,
    limit: int,
    passage_id: str,
    title: str,
    snippet: str,
) -> dict[str, object]:
    return {
        "observation_status": "success",
        "executed_action": {
            "kind": "tool",
            "name": "search",
            "resource_id": QA_RETRIEVAL_TOOL_ID,
            "arguments": {"query": query, "limit": limit},
        },
        "result": {
            "operation": "search",
            "query": query,
            "top_k": limit,
            "passage_ids": [passage_id],
            "hits": [
                {
                    "passage_id": passage_id,
                    "document_id": f"document-{passage_id}",
                    "title": title,
                    "snippet": snippet,
                    "rank": 1,
                }
            ],
        },
    }


OBSERVATIONS = (
    search_observation(
        QUERIES[0],
        limit=5,
        passage_id="public-0",
        title="Chart Weekly",
        snippet=(
            "Chart Weekly magazine first published an American hit chart."
        ),
    ),
    search_observation(
        QUERIES[1],
        limit=10,
        passage_id="public-1",
        title="Chart Weekly",
        snippet=(
            "Chart Weekly magazine made the first publication of an American "
            "hit chart."
        ),
    ),
    search_observation(
        QUERIES[2],
        limit=15,
        passage_id="public-2",
        title="Chart Weekly",
        snippet=(
            "Chart Weekly magazine first published an American hit chart."
        ),
    ),
    search_observation(
        QUERIES[3],
        limit=20,
        passage_id="public-3",
        title="Chart Weekly",
        snippet="Chart Weekly magazine first published an American hit chart.",
    ),
    search_observation(
        QUERIES[4],
        limit=25,
        passage_id="public-4",
        title="Chart Weekly",
        snippet=(
            "Chart Weekly magazine made the first publication of an American "
            "music hit parade."
        ),
    ),
)


@dataclass(frozen=True)
class Manifest:
    corpus_name: str = "synthetic-public-corpus"
    corpus_version: str = "v1"
    index_id: str = "synthetic-index"
    format: str = "skillev-public-retrieval-index@2"
    retrieval_backend: str = "sqlite-fts5-lexical"


@dataclass(frozen=True)
class Hit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    rank: int


class CountingIndex:
    manifest = Manifest()

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int) -> tuple[Hit, ...]:
        self.search_calls.append((query, limit))
        return (
            Hit(
                "public-0",
                "document-0",
                "Chart Weekly",
                "Chart Weekly magazine first published an American hit chart.",
                1,
            ),
        )

    def read(self, passage_id: str) -> object:
        raise AssertionError(f"unexpected read: {passage_id}")

    def close(self) -> None:
        return None


def request() -> AgentRequest:
    return AgentRequest(
        request_id="synthetic-proof-request",
        run_id="synthetic-proof-run",
        graph_revision=1,
        problem=QUESTION,
        agent=AgentNode(
            "retriever",
            "model",
            "retrieve public evidence for the requested entity and relation",
            role_family="evidence_retriever",
            allowed_tools=(QA_RETRIEVAL_TOOL_ID,),
            execution_mode="react",
        ),
        model=ModelSpec("model", "provider"),
        provider=ProviderSpec("provider", kind="test"),
        phase=ExecutionPhase.SINGLE,
        semantic_protocol=QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    )


class QAToolStrategyProofTests(unittest.TestCase):
    def test_prior_hit_support_requires_verified_action_observation_mirror(
        self,
    ) -> None:
        mismatched_prior = json.loads(json.dumps(OBSERVATIONS[0]))
        mismatched_prior["result"]["query"] = "different public query"

        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=QUERIES[:2],
            search_observations=(mismatched_prior,),
        )

        self.assertFalse(proofs[-1].verified)
        self.assertEqual("unverified_strategy_attempt", proofs[-1].proof_strength)
        self.assertEqual((), proofs[-1].source_passage_ids)

    def test_controlled_relation_replacement_rejects_added_subtopic(self) -> None:
        allowed = _relation_surface_replacement_classes(
            original_question=QUESTION,
            previous_query=QUERIES[1],
            query=QUERIES[2],
        )
        injected = _relation_surface_replacement_classes(
            original_question=QUESTION,
            previous_query=QUERIES[1],
            query=QUERIES[2] + " unrelated candidate",
        )

        self.assertTrue(allowed)
        self.assertEqual(frozenset(), injected)

    def test_search_mirror_rejects_malformed_missing_or_reordered_hits(
        self,
    ) -> None:
        malformed = json.loads(json.dumps(OBSERVATIONS[0]))
        del malformed["result"]["hits"][0]["document_id"]

        unequal = json.loads(json.dumps(OBSERVATIONS[0]))
        unequal["result"]["passage_ids"].append("public-extra")

        reordered = json.loads(json.dumps(OBSERVATIONS[0]))
        second_hit = json.loads(json.dumps(reordered["result"]["hits"][0]))
        second_hit.update(
            {
                "document_id": "document-public-second",
                "passage_id": "public-second",
                "rank": 2,
            }
        )
        reordered["result"]["passage_ids"] = [
            "public-second",
            "public-0",
        ]
        reordered["result"]["hits"].append(second_hit)

        for observation in (malformed, unequal, reordered):
            with self.subTest(observation=observation):
                (record,) = _factual_retrieval_attempt_records(
                    original_question=QUESTION,
                    search_observations=(observation,),
                )
                self.assertFalse(record.tool_transition_verified)
                self.assertFalse(record.verified)

    def test_recall_expansion_inherits_adjacent_transition_proof_and_zero_is_false(
        self,
    ) -> None:
        unsupported_initial = search_observation(
            QUERIES[0],
            limit=5,
            passage_id="irrelevant",
            title="Unrelated Topic",
            snippet="An unrelated public passage with no matching entity or relation.",
        )
        unsupported_spelling = search_observation(
            QUERIES[1],
            limit=10,
            passage_id="still-irrelevant",
            title="Another Topic",
            snippet="Another unrelated public passage.",
        )
        spelling_recall_expansion = search_observation(
            QUERIES[1],
            limit=15,
            passage_id="expanded-irrelevant",
            title="Third Topic",
            snippet="A larger top-k still does not verify the spelling query.",
        )
        records = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=(
                unsupported_initial,
                unsupported_spelling,
                spelling_recall_expansion,
            ),
        )

        self.assertTrue(records[1].query_variant_verified)
        self.assertFalse(records[2].strategy_advanced)
        self.assertTrue(records[2].query_variant_verified)
        self.assertTrue(records[2].verified)

        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda agent_request: None),
            tool_registry=build_qa_tool_registry(CountingIndex()),
            max_turns=8,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        empty_state = adapter._required_evidence_state(request(), [])
        self.assertFalse(empty_state.retrieval_attempts_verified)

    def test_every_search_variant_has_receipt_replayable_attempt_record(
        self,
    ) -> None:
        recall_expansion = search_observation(
            QUERIES[0],
            limit=10,
            passage_id="public-recall",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )
        shifted = tuple(
            search_observation(
                query,
                limit=limit,
                passage_id=f"shifted-{index}",
                title="Chart Weekly",
                snippet=OBSERVATIONS[index]["result"]["hits"][0]["snippet"],
            )
            for index, (query, limit) in enumerate(
                zip(QUERIES[1:], (15, 20, 25, 25)),
                start=1,
            )
        )
        records = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=(
                OBSERVATIONS[0],
                recall_expansion,
                *shifted,
            ),
        )

        self.assertEqual(6, len(records))
        self.assertEqual(
            (
                "initial_retrieval",
                None,
                "spelling_normalization",
                "alias_expansion",
                "query_rewriting",
                "alias_expansion",
            ),
            tuple(record.required_strategy for record in records),
        )
        self.assertEqual(
            (True, False, True, True, True, True),
            tuple(record.strategy_advanced for record in records),
        )
        self.assertEqual(
            (False, True, False, False, False, False),
            tuple(record.recall_expansion for record in records),
        )
        self.assertEqual(
            (5, 10, 15, 20, 25, 25),
            tuple(record.required_top_k for record in records),
        )
        self.assertEqual(
            (5, 10, 15, 20, 25, 25),
            tuple(record.observed_top_k for record in records),
        )
        self.assertTrue(all(record.verified for record in records))
        self.assertEqual(
            list(records[1].fts_term_set),
            records[1].to_value()["fts_term_set"],
        )

    def test_repeated_top_k_expansion_does_not_create_strategy_labels(
        self,
    ) -> None:
        observations = tuple(
            search_observation(
                QUERIES[0],
                limit=limit,
                passage_id=f"public-top-{limit}",
                title="Chart Weekly",
                snippet=(
                    "Chart Weekly magazine first published an American hit "
                    "chart."
                ),
            )
            for limit in (5, 10, 15)
        )
        records = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=observations,
        )

        self.assertEqual(
            ("initial_retrieval", None, None),
            tuple(record.required_strategy for record in records),
        )
        self.assertEqual(
            (False, True, True),
            tuple(record.recall_expansion for record in records),
        )
        self.assertEqual(
            (5, 10, 15),
            tuple(record.required_top_k for record in records),
        )
        self.assertTrue(all(record.verified for record in records))

    def test_attempt_record_rejects_action_observation_receipt_mismatch(
        self,
    ) -> None:
        mismatched = json.loads(json.dumps(OBSERVATIONS[0]))
        mismatched["result"]["top_k"] = 10

        (record,) = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=(mismatched,),
        )

        self.assertFalse(record.tool_transition_verified)
        self.assertFalse(record.verified)

    def test_five_stage_proofs_are_answer_free_and_receipt_derived(self) -> None:
        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=QUERIES,
            search_observations=OBSERVATIONS,
        )

        self.assertEqual(5, len(proofs))
        self.assertTrue(all(proof.verified for proof in proofs))
        self.assertEqual(
            (
                "question_invariant_strategy_attempt",
                "tool_receipt_conditioned_strategy_attempt",
                "deterministic_relation_invariant_strategy_attempt",
                "tool_receipt_conditioned_strategy_attempt",
                "deterministic_relation_invariant_strategy_attempt",
            ),
            tuple(proof.proof_strength for proof in proofs),
        )
        self.assertEqual(("public-0",), proofs[1].source_passage_ids)
        self.assertEqual(("public-1",), proofs[2].source_passage_ids)
        self.assertEqual(("public-2",), proofs[3].source_passage_ids)

        adapter = QARetrievalReactExecutionAdapter(
            gateway=SimpleNamespace(generate=lambda agent_request: None),
            tool_registry=build_qa_tool_registry(CountingIndex()),
            max_turns=8,
            max_tool_calls=12,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )
        public_state = adapter._public_retrieval_continuation_state(
            request(), list(OBSERVATIONS)
        )
        assert public_state is not None
        self.assertEqual(
            [proof.to_value() for proof in proofs],
            public_state["strategy_proofs"],
        )

    def test_receipts_upgrade_proof_without_narrowing_legal_attempts(self) -> None:
        question_invariant = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=QUERIES[:2],
        )
        receipt_conditioned = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=QUERIES[:2],
            search_observations=OBSERVATIONS[:2],
        )
        invalid_alias_append = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=(
                *QUERIES[:2],
                (
                    "Chart Weekly magazine first published American hit chart "
                    "music hit parade decade"
                ),
            ),
            search_observations=OBSERVATIONS[:2],
        )

        self.assertFalse(question_invariant[-1].verified)
        self.assertEqual(
            "unverified_strategy_attempt",
            question_invariant[-1].proof_strength,
        )
        self.assertEqual(
            "tool_receipt_conditioned_strategy_attempt",
            receipt_conditioned[-1].proof_strength,
        )
        self.assertFalse(invalid_alias_append[-1].verified)

    def test_relation_alias_transition_is_not_ordinal_spelling_stage(
        self,
    ) -> None:
        relation_alias_query = QUERIES[2]
        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=(QUERIES[1], relation_alias_query),
            search_observations=(OBSERVATIONS[1], OBSERVATIONS[2]),
        )
        records = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=(OBSERVATIONS[1], OBSERVATIONS[2]),
        )

        self.assertEqual("alias_expansion", proofs[1].strategy)
        self.assertTrue(proofs[1].verified)
        self.assertEqual("alias_expansion", records[1].required_strategy)
        self.assertNotEqual("spelling_normalization", proofs[1].strategy)

    def test_tc5_snippet_grounded_relation_context_is_query_rewriting(
        self,
    ) -> None:
        previous_query = (
            "Chart Weekly magazine first publish American hit chart decade"
        )
        rewritten_query = previous_query + " publication"
        prior = search_observation(
            previous_query,
            limit=5,
            passage_id="tc5-prior",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine made the first publication of an "
                "American hit chart."
            ),
        )
        current = search_observation(
            rewritten_query,
            limit=10,
            passage_id="tc5-current",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )

        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=(previous_query, rewritten_query),
            search_observations=(prior, current),
        )

        self.assertEqual(2, len(proofs))
        self.assertEqual("query_rewriting", proofs[1].strategy)
        self.assertTrue(proofs[1].verified)
        self.assertEqual(("tc5-prior",), proofs[1].source_passage_ids)

    def test_query_rewriting_rejects_snippet_proper_name_as_relation_context(
        self,
    ) -> None:
        previous_query = (
            "Chart Weekly magazine first publish American hit chart decade"
        )
        rewritten_query = previous_query + " Lewis"
        prior = search_observation(
            previous_query,
            limit=5,
            passage_id="rewrite-prior",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart; "
                "Lewis discussed it."
            ),
        )
        current = search_observation(
            rewritten_query,
            limit=10,
            passage_id="rewrite-current",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )

        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=(previous_query, rewritten_query),
            search_observations=(prior, current),
        )

        self.assertFalse(proofs[1].verified)
        self.assertEqual("unverified_strategy_attempt", proofs[1].proof_strength)

    def test_query_rewriting_can_only_delete_question_syntax_noise(self) -> None:
        noisy_query = (
            "In which decade did Chart Weekly magazine first publish an "
            "American hit chart"
        )
        concise_query = (
            "decade Chart Weekly magazine first publish American hit chart"
        )
        prior = search_observation(
            noisy_query,
            limit=5,
            passage_id="delete-prior",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )
        current = search_observation(
            concise_query,
            limit=10,
            passage_id="delete-current",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )

        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=(noisy_query, concise_query),
            search_observations=(prior, current),
        )

        self.assertEqual("query_rewriting", proofs[1].strategy)
        self.assertTrue(proofs[1].verified)

    def test_generic_relation_rewrite_requires_local_public_snippet_context(
        self,
    ) -> None:
        question = "When did Aurora premiere?"
        previous_query = "Aurora premiere"
        rewritten_query = "Aurora premiere debuted"
        prior = search_observation(
            previous_query,
            limit=5,
            passage_id="generic-rewrite-prior",
            title="Aurora",
            snippet="Aurora premiered and debuted at the festival.",
        )
        current = search_observation(
            rewritten_query,
            limit=10,
            passage_id="generic-rewrite-current",
            title="Aurora",
            snippet="Aurora debuted at the festival.",
        )

        proofs = _factual_strategy_proofs(
            original_question=question,
            distinct_queries=(previous_query, rewritten_query),
            search_observations=(prior, current),
        )

        self.assertEqual("query_rewriting", proofs[1].strategy)
        self.assertTrue(proofs[1].verified)
        self.assertEqual(
            ("generic-rewrite-prior",),
            proofs[1].source_passage_ids,
        )

    def test_local_snippet_proximity_cannot_replace_the_question_predicate(
        self,
    ) -> None:
        question = "When did Aurora premiere?"
        previous_query = "Aurora premiere"
        drifted_query = "Aurora festival"
        prior = search_observation(
            previous_query,
            limit=5,
            passage_id="generic-drift-prior",
            title="Aurora",
            snippet="Aurora premiered at the festival.",
        )
        current = search_observation(
            drifted_query,
            limit=10,
            passage_id="generic-drift-current",
            title="Aurora",
            snippet="Aurora appeared at the festival.",
        )

        proofs = _factual_strategy_proofs(
            original_question=question,
            distinct_queries=(previous_query, drifted_query),
            search_observations=(prior, current),
        )

        self.assertEqual("query_rewriting", proofs[1].strategy)
        self.assertFalse(proofs[1].verified)
        self.assertEqual(
            "unverified_strategy_attempt",
            proofs[1].proof_strength,
        )

    def test_recall_expansion_has_attempt_record_but_no_strategy_proof(
        self,
    ) -> None:
        expanded = search_observation(
            QUERIES[0],
            limit=10,
            passage_id="expanded",
            title="Chart Weekly",
            snippet=(
                "Chart Weekly magazine first published an American hit chart."
            ),
        )
        proofs = _factual_strategy_proofs(
            original_question=QUESTION,
            distinct_queries=QUERIES[:2],
            search_observations=(OBSERVATIONS[0], OBSERVATIONS[1]),
        )
        records = _factual_retrieval_attempt_records(
            original_question=QUESTION,
            search_observations=(OBSERVATIONS[0], expanded, OBSERVATIONS[1]),
        )

        self.assertEqual(2, len(proofs))
        self.assertEqual(3, len(records))
        self.assertIsNone(records[1].required_strategy)
        self.assertTrue(records[1].recall_expansion)

    def test_knowledge_base_coverage_requires_complete_valid_schedule(self) -> None:
        diagnose = QARetrievalReactExecutionAdapter._factual_exhaustion_diagnosis

        self.assertEqual(
            "retrieval_strategy_failure",
            diagnose(
                strategy_progress_count=5,
                strategy_semantics_verified=False,
                successful_search_hit_counts=(0, 0, 0, 0, 0),
                tool_error_count=0,
            ),
        )
        self.assertEqual(
            "knowledge_base_coverage_failure",
            diagnose(
                strategy_progress_count=5,
                strategy_semantics_verified=True,
                successful_search_hit_counts=(0, 0, 0, 0, 0),
                tool_error_count=0,
                verified_strategy_coverage=(
                    "initial_retrieval",
                    "spelling_normalization",
                    "alias_expansion",
                    "entity_disambiguation",
                    "query_rewriting",
                ),
            ),
        )
        self.assertEqual(
            "retrieval_strategy_failure",
            diagnose(
                strategy_progress_count=5,
                strategy_semantics_verified=True,
                successful_search_hit_counts=(0, 0, 0, 0, 0),
                tool_error_count=0,
                verified_strategy_coverage=(
                    "initial_retrieval",
                    "query_rewriting",
                    "query_rewriting",
                    "query_rewriting",
                    "query_rewriting",
                ),
            ),
        )


class QAToolInvalidActionDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_recall_expansion_matches_replay_for_five_ten_fifteen(
        self,
    ) -> None:
        query = QUERIES[0]

        def action(limit: int) -> str:
            return json.dumps(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": query, "limit": limit},
                }
            )

        class Gateway:
            def __init__(self) -> None:
                self.outputs = [action(5), action(10), action(15)]
                self.schema_limits: list[int] = []
                self.contracts: list[str] = []

            async def generate(self, agent_request: AgentRequest) -> AgentResponse:
                self.contracts.append(agent_request.agent.contract)
                schema = json.loads(
                    agent_request.model.metadata["response_json_schema"]
                )
                branches = schema.get("oneOf", [schema])
                search_branch = next(
                    branch
                    for branch in branches
                    if branch["properties"]["name"].get("const") == "search"
                )
                self.schema_limits.append(
                    search_branch["properties"]["arguments"]["properties"]
                    ["limit"]["const"]
                )
                return AgentResponse(self.outputs.pop(0))

        class EmptyCountingIndex(CountingIndex):
            def search(self, query: str, *, limit: int) -> tuple[Hit, ...]:
                self.search_calls.append((query, limit))
                return ()

        # Recall expansion is admissible only while the public search has no
        # relation-compatible unread candidate. Candidate -> read transition
        # is covered separately by the QA adapter action-domain tests.
        index = EmptyCountingIndex()
        gateway = Gateway()
        adapter = QARetrievalReactExecutionAdapter(
            gateway=gateway,
            tool_registry=build_qa_tool_registry(index),
            max_turns=3,
            max_tool_calls=8,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await adapter.execute(request())

        self.assertEqual([(query, 5), (query, 10), (query, 15)], index.search_calls)
        self.assertEqual([5, 10, 15], gateway.schema_limits)
        self.assertIn(
            "at each strictly larger required top_k",
            gateway.contracts[1],
        )
        self.assertNotIn("may be repeated once", gateway.contracts[1])
        records = caught.exception.react_trace[-1][
            "terminal_failure_diagnosis"
        ]["retrieval_attempts"]
        self.assertEqual([5, 10, 15], [record["required_top_k"] for record in records])
        self.assertEqual([False, True, True], [record["recall_expansion"] for record in records])

    async def test_invalid_spelling_action_does_not_dispatch_or_advance(self) -> None:
        def action(query: str, limit: int) -> str:
            return json.dumps(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": query, "limit": limit},
                }
            )

        class Gateway:
            outputs = [
                action(QUERIES[0], 5),
                action("Southbridge College publish American hit chart decade", 10),
            ]

            async def generate(self, agent_request: AgentRequest) -> AgentResponse:
                del agent_request
                return AgentResponse(self.outputs.pop(0))

        index = CountingIndex()
        adapter = QARetrievalReactExecutionAdapter(
            gateway=Gateway(),
            tool_registry=build_qa_tool_registry(index),
            max_turns=2,
            max_tool_calls=8,
            task_type="factual_qa",
            completion_policy="required_evidence",
        )

        with self.assertRaises(ReactExecutionError) as caught:
            await adapter.execute(request())

        self.assertEqual([(QUERIES[0], 5)], index.search_calls)
        self.assertEqual(1, len(caught.exception.tool_receipts))
        self.assertEqual(
            1,
            caught.exception.react_trace[-1]["terminal_failure_diagnosis"]
            ["retrieval_attempt_count"],
        )


if __name__ == "__main__":
    unittest.main()
