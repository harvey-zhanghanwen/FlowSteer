"""Unit tests for strict LLM-based free-contract role clustering."""

from __future__ import annotations

import asyncio
import json
import unittest

from src.interactive.exploration.role_classifier import (
    CONTRACT_REWRITE_PROMPT_VERSION,
    CONTRACT_REWRITE_SCHEMA_VERSION,
    ContractRewriteError,
    ContractRewriteRequest,
    ContractRoleRewriter,
    OpenAICompatibleRoleCompletionClient,
    ROLE_CLASSIFIER_PROMPT_VERSION,
    ROLE_CLASSIFIER_SCHEMA_VERSION,
    ROLE_CLUSTERS,
    RoleClassificationError,
    RoleClassifier,
    RoleCompletionRequest,
    RoleCompletionResponse,
    build_contract_rewrite_messages,
    build_role_classification_messages,
    parse_rewritten_contract,
    parse_role_classification,
)


class ScriptedCompletion:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[RoleCompletionRequest | ContractRewriteRequest] = []

    async def complete(
        self,
        request: RoleCompletionRequest | ContractRewriteRequest,
    ) -> RoleCompletionResponse:
        self.requests.append(request)
        return RoleCompletionResponse(
            self.text,
            {"provider_request_id": "provider-17", "provider_model": "classifier"},
        )


class FailingCompletion:
    async def complete(
        self,
        request: RoleCompletionRequest | ContractRewriteRequest,
    ) -> RoleCompletionResponse:
        del request
        raise TimeoutError("simulated timeout")


class RoleClassifierTests(unittest.TestCase):
    def test_free_contract_is_classified_only_from_llm_json(self) -> None:
        completion = ScriptedCompletion('{"role_cluster":"verify"}')
        classifier = RoleClassifier(
            completion,
            classifier_version="qwen35-role-cluster-20260907",
            request_id_factory=lambda: "role-request-1",
        )
        # The wording intentionally contains neither "verify" nor a fixed role name.
        contract = "Compare every claimed relation with the supplied records before output."
        result = asyncio.run(classifier.classify(contract, seed=23))

        self.assertEqual(result.role_cluster, "verify")
        self.assertTrue(result.receipt.succeeded)
        self.assertEqual(result.receipt.request.request_id, "role-request-1")
        self.assertEqual(
            result.receipt.request.classifier_version,
            "qwen35-role-cluster-20260907",
        )
        self.assertEqual(result.receipt.request.prompt_version, ROLE_CLASSIFIER_PROMPT_VERSION)
        self.assertEqual(result.receipt.request.schema_version, ROLE_CLASSIFIER_SCHEMA_VERSION)
        self.assertEqual(result.receipt.completion_metadata["provider_request_id"], "provider-17")
        self.assertEqual(result.receipt.to_dict()["role_cluster"], "verify")
        self.assertEqual(len(completion.requests), 1)

        rendered = completion.requests[0].messages
        self.assertEqual([message["role"] for message in rendered], ["system", "user"])
        self.assertEqual(json.loads(rendered[1]["content"]), {"agent_contract": contract})

    def test_contract_words_do_not_override_llm_classification(self) -> None:
        completion = ScriptedCompletion('{"role_cluster":"plan"}')
        classifier = RoleClassifier(completion, request_id_factory=lambda: "request-2")
        result = asyncio.run(
            classifier.classify("Retrieve sources, test claims, then summarize the result.")
        )
        self.assertEqual(result.role_cluster, "plan")

    def test_fixed_catalog_is_accepted_exhaustively(self) -> None:
        for role in ROLE_CLUSTERS:
            with self.subTest(role=role):
                self.assertEqual(
                    parse_role_classification(json.dumps({"role_cluster": role})),
                    role,
                )

    def test_parser_rejects_recovery_formats_and_schema_drift(self) -> None:
        rejected = (
            '```json\n{"role_cluster":"solve"}\n```',
            '{"role_cluster":"solve"} trailing',
            '{"role_cluster":"research"}',
            '{"role_cluster":"solve","confidence":1}',
            '{"role_cluster":"solve","role_cluster":"verify"}',
            '[{"role_cluster":"solve"}]',
            "",
        )
        for text in rejected:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_role_classification(text)

    def test_invalid_llm_output_fails_closed_with_receipt(self) -> None:
        classifier = RoleClassifier(
            ScriptedCompletion("The contract appears to retrieve information."),
            request_id_factory=lambda: "request-3",
        )
        with self.assertRaises(RoleClassificationError) as caught:
            asyncio.run(classifier.classify("Find the relevant source."))

        receipt = caught.exception.receipt
        self.assertFalse(receipt.succeeded)
        self.assertIsNone(receipt.role_cluster)
        self.assertEqual(receipt.failure_stage, "parse")
        self.assertEqual(receipt.error_type, "ValueError")
        self.assertEqual(receipt.request.request_id, "request-3")
        self.assertEqual(receipt.raw_completion, "The contract appears to retrieve information.")

    def test_completion_failure_fails_closed_without_default_role(self) -> None:
        classifier = RoleClassifier(
            FailingCompletion(),
            request_id_factory=lambda: "request-4",
        )
        with self.assertRaises(RoleClassificationError) as caught:
            asyncio.run(classifier.classify("Produce an answer."))
        receipt = caught.exception.receipt
        self.assertEqual(receipt.status, "failed")
        self.assertIsNone(receipt.role_cluster)
        self.assertEqual(receipt.failure_stage, "completion")
        self.assertEqual(receipt.error_type, "TimeoutError")
        self.assertIsNone(receipt.raw_completion)

    def test_input_validation_occurs_before_completion(self) -> None:
        completion = ScriptedCompletion('{"role_cluster":"solve"}')
        classifier = RoleClassifier(completion)
        with self.assertRaises(ValueError):
            asyncio.run(classifier.classify("  "))
        with self.assertRaises(ValueError):
            asyncio.run(classifier.classify("answer", seed=-1))
        self.assertEqual(completion.requests, [])

    def test_prompt_does_not_rewrite_or_require_fixed_contract_words(self) -> None:
        contract = "Inspect what arrived and return a careful judgment."
        messages = build_role_classification_messages(contract)
        self.assertIn("free-text Agent contract", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["agent_contract"], contract)


class ContractRoleRewriterTests(unittest.TestCase):
    def test_role_only_rewrite_uses_llm_and_records_versions(self) -> None:
        rewritten = (
            "Check each proposed answer against the supplied passages and return the "
            "supported answer in the existing output form."
        )
        completion = ScriptedCompletion(json.dumps({"contract": rewritten}))
        rewriter = ContractRoleRewriter(
            completion,
            rewriter_version="qwen35-contract-rewriter-20260907",
            request_id_factory=lambda: "rewrite-request-1",
        )
        original = "Derive the answer from the supplied passages and return it concisely."
        result = asyncio.run(
            rewriter.rewrite(
                original,
                target_role_cluster="verify",
                task_family="hotpotqa",
                seed=29,
            )
        )

        self.assertEqual(result.contract, rewritten)
        self.assertTrue(result.receipt.succeeded)
        request = result.receipt.request
        self.assertEqual(request.original_contract, original)
        self.assertEqual(request.target_role_cluster, "verify")
        self.assertEqual(request.task_family, "hotpotqa")
        self.assertEqual(request.rewriter_version, "qwen35-contract-rewriter-20260907")
        self.assertEqual(request.prompt_version, CONTRACT_REWRITE_PROMPT_VERSION)
        self.assertEqual(request.schema_version, CONTRACT_REWRITE_SCHEMA_VERSION)
        self.assertEqual(request.request_id, "rewrite-request-1")
        payload = json.loads(request.messages[1]["content"])
        self.assertEqual(
            payload,
            {
                "original_contract": original,
                "target_role_cluster": "verify",
                "task_family": "hotpotqa",
            },
        )
        self.assertIn("Change only that function", request.messages[0]["content"])

    def test_rewrite_is_not_implemented_by_a_template_or_keyword_rule(self) -> None:
        model_contract = "Inspect provenance before accepting the upstream artifact."
        rewriter = ContractRoleRewriter(
            ScriptedCompletion(json.dumps({"contract": model_contract})),
            request_id_factory=lambda: "rewrite-request-2",
        )
        result = asyncio.run(
            rewriter.rewrite(
                "Retrieve and code a test for the answer.",
                target_role_cluster="summarize",
                task_family="hotpotqa",
            )
        )
        self.assertEqual(result.contract, model_contract)

    def test_rewrite_parser_is_strict(self) -> None:
        self.assertEqual(
            parse_rewritten_contract('{"contract":"Produce the result."}'),
            "Produce the result.",
        )
        rejected = (
            '```json\n{"contract":"Produce the result."}\n```',
            '{"contract":"Produce the result.","role_cluster":"solve"}',
            '{"contract":""}',
            '{"contract":"a","contract":"b"}',
            "Produce the result.",
        )
        for text in rejected:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_rewritten_contract(text)

    def test_invalid_rewrite_fails_closed(self) -> None:
        rewriter = ContractRoleRewriter(
            ScriptedCompletion("A rewritten contract without JSON"),
            request_id_factory=lambda: "rewrite-request-3",
        )
        with self.assertRaises(ContractRewriteError) as caught:
            asyncio.run(
                rewriter.rewrite(
                    "Answer the question.",
                    target_role_cluster="retrieve",
                    task_family="hotpotqa",
                )
            )
        receipt = caught.exception.receipt
        self.assertFalse(receipt.succeeded)
        self.assertIsNone(receipt.rewritten_contract)
        self.assertEqual(receipt.failure_stage, "parse")
        self.assertEqual(receipt.request.target_role_cluster, "retrieve")

    def test_rewrite_completion_failure_does_not_copy_original_contract(self) -> None:
        rewriter = ContractRoleRewriter(
            FailingCompletion(),
            request_id_factory=lambda: "rewrite-request-4",
        )
        with self.assertRaises(ContractRewriteError) as caught:
            asyncio.run(
                rewriter.rewrite(
                    "Answer the question.",
                    target_role_cluster="plan",
                    task_family="hotpotqa",
                )
            )
        self.assertIsNone(caught.exception.receipt.rewritten_contract)
        self.assertEqual(caught.exception.receipt.failure_stage, "completion")

    def test_rewrite_input_validation_precedes_completion(self) -> None:
        completion = ScriptedCompletion('{"contract":"unused"}')
        rewriter = ContractRoleRewriter(completion)
        with self.assertRaises(ValueError):
            asyncio.run(
                rewriter.rewrite(
                    "Answer.",
                    target_role_cluster="unknown",  # type: ignore[arg-type]
                    task_family="hotpotqa",
                )
            )
        with self.assertRaises(ValueError):
            build_contract_rewrite_messages(
                "Answer.", target_role_cluster="solve", task_family=" "
            )
        self.assertEqual(completion.requests, [])


class OpenAICompatibleRoleCompletionClientTests(unittest.TestCase):
    def request(self) -> RoleCompletionRequest:
        return RoleCompletionRequest(
            request_id="request-http-1",
            messages=build_role_classification_messages("Arrange the work."),
            seed=7,
            classifier_version="classifier-v1",
            prompt_version=ROLE_CLASSIFIER_PROMPT_VERSION,
            schema_version=ROLE_CLASSIFIER_SCHEMA_VERSION,
        )

    def test_payload_matches_openai_sglang_chat_completion_without_network(self) -> None:
        client = OpenAICompatibleRoleCompletionClient(
            base_url="http://127.0.0.1:8016/v1/",
            model="supervisor_theta",
            max_tokens=32,
        )
        payload = client.request_payload(self.request())
        self.assertEqual(payload["model"], "supervisor_theta")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_openai_response_is_adapted_to_completion_receipt(self) -> None:
        client = OpenAICompatibleRoleCompletionClient()
        result = client._parse_response(
            {
                "id": "chatcmpl-1",
                "model": "supervisor_theta",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"role_cluster":"plan"}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            }
        )
        self.assertEqual(result.text, '{"role_cluster":"plan"}')
        self.assertEqual(result.metadata["provider_request_id"], "chatcmpl-1")
        self.assertEqual(result.metadata["total_tokens"], 25)

    def test_construction_has_no_network_side_effect(self) -> None:
        # If construction performed I/O this invalid port would fail here.
        client = OpenAICompatibleRoleCompletionClient(
            base_url="http://127.0.0.1:1/v1",
            max_retries=0,
        )
        self.assertEqual(client.base_url, "http://127.0.0.1:1/v1")


if __name__ == "__main__":
    unittest.main()
