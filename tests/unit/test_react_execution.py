from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import (
    ReactExecutionError,
    ReactGenerationError,
    ToolReactExecutionAdapter,
)
from src.interactive.scientific_sampling import (
    GenerationPhase,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


def action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "resource_id": resource_id,
            "skill_id": None,
        }
    )


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolRegistration(
                "wiki.search",
                FakeTool(
                    {
                        "search": lambda arguments: {
                            "passage_ids": ["p1"],
                            "query": arguments["query"],
                        }
                    }
                ),
                ToolCapability(
                    tool_id="wiki.search",
                    dataset_scope=("triviaqa",),
                    action_schemas={
                        "search": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {"query": {"type": "string"}},
                        }
                    },
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    side_effect="none",
                    timeout_seconds=1.0,
                    version="wiki-test-v1",
                ),
            ),
        )
    )


def request() -> AgentRequest:
    return AgentRequest(
        request_id="run:1:r:single",
        run_id="run",
        graph_revision=1,
        problem="Who wrote the first published algorithm?",
        agent=AgentNode(
            "r",
            "m",
            "retrieve evidence and answer",
            allowed_tools=("wiki.search",),
            execution_mode="react",
            artifact_type="evidence",
            completion_condition="return an answer supported by the observation",
        ),
        model=ModelSpec("m", "fake"),
        provider=ProviderSpec("fake", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


def provenance_assessment_request(
    *,
    execution_mode: str = "react",
) -> AgentRequest:
    item = request()
    return replace(
        item,
        problem=(
            "benchmark_id=aime-2026\n"
            "Find the requested integer from the stated mathematical problem."
        ),
        agent=replace(
            item.agent,
            execution_mode=execution_mode,
            contract=(
                "Assess every routed work product from its public derivation "
                "and return the declared artifact."
            ),
        ),
        upstream=(
            UpstreamMessage(
                source_agent_id="source_solver",
                target_agent_id="r",
                content=(
                    "The public recurrence leaves 65 admissible cases.\n"
                    "Final Answer: 65"
                ),
                graph_revision=1,
                artifact_version="artifact:source_solver:1",
                artifact_complete=True,
                tool_receipts=(
                    {
                        "tool_id": "python.compute",
                        "request": {"action": "run"},
                        "status": "completed",
                    },
                ),
            ),
        ),
        artifact_assessment_protocol=(
            "provenance_bound_candidate_assessment_v2"
        ),
    )


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(self.outputs.pop(0), {"provider_request_id": len(self.requests)})


class ResponseSequenceGateway:
    def __init__(self, responses: list[AgentResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class ToolReactExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_tool_or_complete_domain_uses_strict_one_of_schema(self) -> None:
        adapter = ToolReactExecutionAdapter(
            gateway=SequenceGateway([]),
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=1,
        )

        schema = adapter._state_conditioned_response_schema(request(), [])

        self.assertIsNotNone(schema)
        branches = schema["oneOf"]
        self.assertEqual(2, len(branches))
        self.assertEqual(
            {"complete", "tool"},
            {branch["properties"]["kind"]["const"] for branch in branches},
        )
        for branch in branches:
            self.assertEqual(
                ["arguments", "kind", "name", "resource_id", "skill_id"],
                branch["required"],
            )
            self.assertFalse(branch["additionalProperties"])

    async def test_provenance_bound_complete_rejects_scalar_then_preserves_receipt(
        self,
    ) -> None:
        valid_artifact = (
            "The routed recurrence enumerates the admissible cases and its "
            "public calculation gives 65.\n"
            "Final Answer: 65\n"
            "<artifact_assessments>"
            '[{"assessed_artifact_id":"artifact:source_solver:1",'
            '"candidate":"65","assessment":"supported",'
            '"basis":"The public recurrence explicitly derives the count.",'
            '"counterexample":null}]'
            "</artifact_assessments>"
        )
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={"query": "public calculation receipt"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "65"},
                    resource_id=None,
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": valid_artifact},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=1,
        ).execute(provenance_assessment_request())

        self.assertEqual(valid_artifact, response.text)
        scalar_rejection = response.metadata["react_trace"][1]
        self.assertEqual("schema_invalid", scalar_rejection["observation_status"])
        self.assertEqual(
            "provenance_completion_public_derivation_missing",
            scalar_rejection["public_error_code"],
        )
        self.assertIn(
            "A bare scalar candidate is not a complete assessment artifact",
            scalar_rejection["repair_instruction"],
        )
        receipt = response.metadata["artifact_assessment_completion_receipt"]
        self.assertEqual("admitted", receipt["admission_status"])
        self.assertTrue(receipt["public_derivation_preserved"])
        self.assertEqual("65", receipt["output_candidate"])
        self.assertEqual(1, receipt["local_tool_receipt_count"])
        self.assertEqual(
            {
                "source_agent": "source_solver",
                "artifact_id": "artifact:source_solver:1",
                "candidate": "65",
            },
            {
                key: receipt["candidate_sources"][0][key]
                for key in ("source_agent", "artifact_id", "candidate")
            },
        )
        self.assertEqual(
            "source_solver",
            receipt["assessment_bindings"][0]["source_agent"],
        )
        self.assertEqual(
            "artifact:source_solver:1",
            receipt["assessment_bindings"][0]["artifact_id"],
        )
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            receipt,
            response.metadata["react_trace"][-1][
                "artifact_assessment_completion_receipt"
            ],
        )

        first_schema = json.loads(
            gateway.requests[0].model.metadata["response_json_schema"]
        )
        complete_branch = next(
            branch
            for branch in first_schema["oneOf"]
            if branch["properties"]["kind"]["const"] == "complete"
        )
        value_schema = complete_branch["properties"]["arguments"][
            "properties"
        ]["value"]
        self.assertEqual("string", value_schema["type"])
        self.assertEqual(1, value_schema["minLength"])
        self.assertIn("bare candidate", value_schema["description"])
        self.assertIn(
            '"source_agent":"source_solver"',
            gateway.requests[0].agent.contract,
        )
        self.assertIn(
            '"artifact_id":"artifact:source_solver:1"',
            gateway.requests[0].agent.contract,
        )
        self.assertIn(
            "escape every literal backslash",
            gateway.requests[0].agent.contract,
        )

    async def test_provenance_bound_complete_rejects_wrong_candidate_binding(
        self,
    ) -> None:
        wrong_binding = (
            "A public derivation is present.\n"
            "<artifact_assessments>"
            '[{"assessed_artifact_id":"artifact:source_solver:1",'
            '"candidate":"64","assessment":"supported",'
            '"basis":"A stated public derivation.","counterexample":null}]'
            "</artifact_assessments>"
        )
        valid_artifact = (
            "The public recurrence derives 65 cases.\n"
            "Final Answer: 65\n"
            "<artifact_assessments>"
            '[{"assessed_artifact_id":"artifact:source_solver:1",'
            '"candidate":"65","assessment":"supported",'
            '"basis":"The recurrence derives the stated count.",'
            '"counterexample":null}]'
            "</artifact_assessments>"
        )
        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": wrong_binding},
                    resource_id=None,
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": valid_artifact},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=0,
        ).execute(provenance_assessment_request())

        self.assertEqual(valid_artifact, response.text)
        self.assertEqual(
            "provenance_completion_candidate_binding_mismatch",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertIn(
            '"candidate":"65"',
            gateway.requests[1].agent.contract,
        )

    async def test_provenance_bound_coding_uses_same_complete_admission(
        self,
    ) -> None:
        valid_artifact = (
            "The public computation derives the count 65.\n"
            "Final Answer: 65\n"
            "<artifact_assessments>"
            '[{"assessed_artifact_id":"artifact:source_solver:1",'
            '"candidate":"65","assessment":"supported",'
            '"basis":"The public computation derives the count.",'
            '"counterexample":null}]'
            "</artifact_assessments>"
        )
        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "65"},
                    resource_id=None,
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": valid_artifact},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=0,
            execution_mode="coding",
        ).execute(provenance_assessment_request(execution_mode="coding"))

        self.assertEqual(valid_artifact, response.text)
        self.assertEqual(
            "provenance_completion_public_derivation_missing",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertEqual(
            "provenance_bound_candidate_assessment_v2",
            response.metadata["artifact_assessment_completion_receipt"][
                "protocol"
            ],
        )

    async def test_ordinary_coding_complete_still_accepts_scalar_artifact(
        self,
    ) -> None:
        item = replace(
            request(),
            agent=replace(request().agent, execution_mode="coding"),
        )
        response = await ToolReactExecutionAdapter(
            gateway=SequenceGateway(
                [
                    action(
                        "complete",
                        name="complete",
                        arguments={"value": "65"},
                        resource_id=None,
                    )
                ]
            ),
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=0,
            execution_mode="coding",
        ).execute(item)

        self.assertEqual("65", response.text)
        self.assertNotIn(
            "artifact_assessment_completion_receipt",
            response.metadata,
        )

    async def test_empty_action_domain_fails_before_any_model_call(self) -> None:
        class ExhaustedAdapter(ToolReactExecutionAdapter):
            def _state_conditioned_action_domain(
                self,
                request: AgentRequest,
                observations: list[dict[str, object]],
            ) -> tuple[frozenset[tuple[str, str]], bool]:
                del request, observations
                return frozenset(), False

        gateway = SequenceGateway([])
        with self.assertRaises(ReactExecutionError) as caught:
            await ExhaustedAdapter(
                gateway=gateway,
                tool_registry=registry(),
                max_turns=4,
                max_tool_calls=2,
            ).execute(request())

        self.assertIs(True, caught.exception.tool_plan_exhausted)
        self.assertEqual((), caught.exception.react_trace)
        self.assertEqual((), caught.exception.tool_receipts)
        self.assertEqual((), caught.exception.model_calls)
        self.assertEqual([], gateway.requests)

    async def test_scientific_sampling_uses_absolute_action_turn(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(
                base_seed=17
            ),
            schedule_purpose="triviaqa-fixed-schedule",
            ordered_sequence_hash=stable_hash(["triviaqa:tc_5"]),
            sequence_position=0,
            task_id="triviaqa:tc_5",
            optimizer_step_or_anchor_ordinal=0,
        )
        completed = action(
            "complete",
            name="complete",
            arguments={"value": "answer"},
            resource_id=None,
        )
        gateway = SequenceGateway([completed])
        continued = request()
        continued = AgentRequest(
            request_id=continued.request_id,
            run_id=continued.run_id,
            graph_revision=2,
            problem=continued.problem,
            agent=continued.agent,
            model=continued.model,
            provider=continued.provider,
            phase=continued.phase,
            action_history=(
                {"turn": 1, "observation_status": "schema_invalid"},
                # Cross-Agent public-state projection may retain only the
                # Tool-bearing turns; the next seed follows the last explicit
                # turn, not the projected list length.
                {"turn": 4, "observation_status": "schema_invalid"},
            ),
            continuation_source_agent_id="reasoner",
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=2,
            sampling_base_seed=17,
            sampling_coordinate=coordinate,
        ).execute(continued)

        expected_seed = derive_generation_seed(
            base_seed=17,
            coordinate=coordinate,
            step_index=5,
            phase=GenerationPhase.ACTION,
        )
        metadata = gateway.requests[0].model.metadata
        self.assertEqual("1.0", metadata["temperature"])
        self.assertEqual("1.0", metadata["top_p"])
        self.assertEqual(str(expected_seed), metadata["generation_seed"])
        receipt = response.metadata["model_calls"][0]["scientific_sampling"]
        self.assertEqual(5, receipt["step_index"])
        self.assertEqual("action", receipt["phase"])
        self.assertEqual(expected_seed, receipt["generation_seed"])
        self.assertEqual(coordinate.to_value(), receipt["coordinate"])
        self.assertEqual(
            "skillev-scientific-sampling@1",
            response.metadata["model_calls"][0]["algorithm"],
        )
        self.assertEqual("completed", response.metadata["model_calls"][0]["request_status"])
        self.assertEqual(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "max_tokens": 512,
                "seed": expected_seed,
                "thinking_budget": None,
            },
            response.metadata["model_calls"][0]["requested_sampling"],
        )

    async def test_independent_action_and_thinking_budgets_are_injected_per_turn(
        self,
    ) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "answer"},
                    resource_id=None,
                )
            ]
        )
        item = request()
        item = replace(
            item,
            model=replace(
                item.model,
                metadata={
                    **dict(item.model.metadata),
                    "max_tokens": "256",
                    "chat_template_enable_thinking": "true",
                },
            ),
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
            max_action_tokens=8192,
            thinking_budget_tokens=3072,
        ).execute(item)

        turn_metadata = gateway.requests[0].model.metadata
        self.assertEqual("11264", turn_metadata["max_tokens"])
        self.assertEqual("3072", turn_metadata["thinking_budget_tokens"])
        self.assertEqual("true", turn_metadata["chat_template_enable_thinking"])
        self.assertEqual(
            {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_tokens": 11264,
                "seed": None,
                "thinking_budget": 3072,
            },
            response.metadata["model_calls"][0]["requested_sampling"],
        )
        self.assertEqual(
            8192,
            response.metadata["model_calls"][0]["max_action_tokens"],
        )
        self.assertEqual(
            3072,
            response.metadata["model_calls"][0]["max_reasoning_tokens"],
        )

    def test_thinking_budget_is_positive_and_independent_of_action_budget(
        self,
    ) -> None:
        ToolReactExecutionAdapter(
            gateway=SequenceGateway([]),
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
            max_action_tokens=512,
            thinking_budget_tokens=1024,
        )
        for budget in (0, True):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(
                    ValueError,
                    "positive integer or None",
                ):
                    ToolReactExecutionAdapter(
                        gateway=SequenceGateway([]),
                        tool_registry=registry(),
                        max_turns=1,
                        max_tool_calls=1,
                        max_action_tokens=512,
                        thinking_budget_tokens=budget,
                    )

    def test_coding_wire_contract_preserves_complete_source_in_json(self) -> None:
        coding_registry = ToolRegistry(
            (
                ToolRegistration(
                    "python.compute",
                    FakeTool({"run": lambda arguments: {"stdout": "1"}}),
                    ToolCapability(
                        tool_id="python.compute",
                        dataset_scope=("aime2026",),
                        action_schemas={
                            "run": {
                                "type": "object",
                                "required": ["code"],
                                "properties": {"code": {"type": "string"}},
                            }
                        },
                        input_schema={"type": "object"},
                        output_schema={"type": "object"},
                        side_effect="none",
                        timeout_seconds=1.0,
                        version="python-test-v1",
                    ),
                ),
            )
        )
        item = request()
        item = replace(
            item,
            agent=replace(
                item.agent,
                allowed_tools=("python.compute",),
                execution_mode="coding",
            ),
        )
        contract = ToolReactExecutionAdapter(
            gateway=SequenceGateway([]),
            tool_registry=coding_registry,
            max_turns=1,
            max_tool_calls=1,
            execution_mode="coding",
        )._contract(item, [])

        self.assertIn(
            "arguments.code must contain the complete executable source",
            contract,
        )
        self.assertIn("JSON \\n escapes", contract)
        self.assertIn("explicit print(...) statement", contract)
        self.assertIn("Do not put natural-language reasoning", contract)
        self.assertNotIn("Solver", contract)
        self.assertNotIn("Verifier", contract)

    async def test_scientific_sampling_rejects_invalid_continuation_turns(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="triviaqa-fixed-schedule",
            ordered_sequence_hash=stable_hash(["triviaqa:tc_5"]),
            sequence_position=0,
            task_id="triviaqa:tc_5",
            optimizer_step_or_anchor_ordinal=0,
        )
        histories = (
            ({"turn": 1}, {"turn": 1}),
            ({"turn": 2}, {"turn": 1}),
            ({"turn": 0},),
            ({"turn": True},),
            ({"observation_status": "schema_invalid"},),
        )
        for history in histories:
            with self.subTest(history=history):
                with self.assertRaises(ReactExecutionError):
                    await ToolReactExecutionAdapter(
                        gateway=SequenceGateway([]),
                        tool_registry=registry(),
                        max_turns=1,
                        max_tool_calls=1,
                        sampling_base_seed=17,
                        sampling_coordinate=coordinate,
                    ).execute(
                        replace(
                            request(),
                            action_history=history,
                            continuation_source_agent_id="retriever",
                        )
                    )

    async def test_declared_local_sglang_requests_unrestricted_top_k(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="triviaqa-fixed-schedule",
            ordered_sequence_hash=stable_hash(["triviaqa:tc_5"]),
            sequence_position=0,
            task_id="triviaqa:tc_5",
            optimizer_step_or_anchor_ordinal=0,
        )
        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "answer"},
                    resource_id=None,
                )
            ]
        )
        base_request = request()
        local_request = replace(
            base_request,
            provider=replace(
                base_request.provider,
                metadata={
                    "sampling_backend": "sglang",
                    "deployment_locality": "local",
                },
            ),
            model=replace(
                base_request.model,
                metadata={"supports_top_k": "true"},
            ),
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
            sampling_base_seed=17,
            sampling_coordinate=coordinate,
        ).execute(local_request)

        self.assertEqual("-1", gateway.requests[0].model.metadata["top_k"])
        self.assertEqual(
            -1,
            response.metadata["model_calls"][0]["requested_sampling"]["top_k"],
        )

    async def test_failed_generation_preserves_scientific_sampling_receipt(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="triviaqa-fixed-schedule",
            ordered_sequence_hash=stable_hash(["triviaqa:tc_5"]),
            sequence_position=0,
            task_id="triviaqa:tc_5",
            optimizer_step_or_anchor_ordinal=0,
        )

        class FailingGateway:
            async def generate(self, request):
                del request
                raise RuntimeError("provider unavailable")

        with self.assertRaises(ReactGenerationError) as raised:
            await ToolReactExecutionAdapter(
                gateway=FailingGateway(),
                tool_registry=registry(),
                max_turns=1,
                max_tool_calls=1,
                sampling_base_seed=17,
                sampling_coordinate=coordinate,
            ).execute(request())

        self.assertEqual("RuntimeError", raised.exception.cause_error_type)
        self.assertEqual(1, len(raised.exception.model_calls))
        failed_call = raised.exception.model_calls[0]
        self.assertEqual("failed", failed_call["request_status"])
        self.assertEqual(
            "skillev-scientific-sampling@1",
            failed_call["algorithm"],
        )
        self.assertEqual(
            failed_call["scientific_sampling"]["generation_seed"],
            failed_call["requested_sampling"]["seed"],
        )

    async def test_cancelled_generation_preserves_request_scoped_receipt(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="triviaqa-fixed-schedule",
            ordered_sequence_hash=stable_hash(["triviaqa:tc_5"]),
            sequence_position=0,
            task_id="triviaqa:tc_5",
            optimizer_step_or_anchor_ordinal=0,
        )

        class CancelledGateway:
            async def generate(self, request):
                del request
                raise asyncio.CancelledError

        adapter = ToolReactExecutionAdapter(
            gateway=CancelledGateway(),
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
            sampling_base_seed=17,
            sampling_coordinate=coordinate,
        )
        item = request()
        with self.assertRaises(asyncio.CancelledError):
            await adapter.execute(item)

        metadata = adapter.take_cancelled_failure_metadata(item.request_id)
        self.assertEqual(1, len(metadata["model_calls"]))
        self.assertEqual(
            "cancelled",
            metadata["model_calls"][0]["request_status"],
        )
        self.assertEqual(
            "skillev-scientific-sampling@1",
            metadata["model_calls"][0]["algorithm"],
        )
        self.assertEqual({}, dict(adapter.take_cancelled_failure_metadata(item.request_id)))

    async def test_public_action_history_continues_after_canvas_repair(self) -> None:
        completed = action(
            "complete",
            name="complete",
            arguments={"value": "Ada Lovelace"},
            resource_id=None,
        )
        gateway = SequenceGateway([completed])
        continued_request = request()
        continued_request = AgentRequest(
            request_id=continued_request.request_id,
            run_id=continued_request.run_id,
            graph_revision=2,
            problem=continued_request.problem,
            agent=continued_request.agent,
            model=continued_request.model,
            provider=continued_request.provider,
            phase=continued_request.phase,
            action_history=(
                {
                    "turn": 1,
                    "structured_action": {
                        "arguments": {"query": "Ada Lovelace"},
                        "kind": "tool",
                        "name": "search",
                        "resource_id": "wiki.search",
                        "skill_id": None,
                    },
                    "observation": {
                        "observation_status": "success",
                        "tool_id": "wiki.search",
                        "executed_action": {
                            "arguments": {"query": "Ada Lovelace"},
                            "kind": "tool",
                            "name": "search",
                            "resource_id": "wiki.search",
                            "skill_id": None,
                        },
                        "result": {"passage_ids": ["p1"]},
                    },
                },
                {
                    "turn": 2,
                    "observation_status": "schema_invalid",
                    "public_error_code": "completion_schema_invalid",
                    "repair_instruction": "repair the completion schema",
                },
            ),
            prior_tool_receipts=(
                {"tool_id": "wiki.search", "success": True},
            ),
            continuation_source_agent_id="failed_reasoner",
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=2,
        ).execute(continued_request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(2, response.metadata["continued_action_history_count"])
        self.assertEqual(1, response.metadata["continued_tool_receipt_count"])
        self.assertEqual(
            "failed_reasoner",
            response.metadata["continuation_source_agent_id"],
        )
        self.assertEqual(3, response.metadata["react_turns_used"])
        self.assertEqual(1, response.metadata["new_react_turns_used"])
        self.assertTrue(
            response.metadata["react_trace"][0][
                "continued_from_prior_revision"
            ]
        )
        self.assertEqual(
            "failed_reasoner",
            response.metadata["react_trace"][0][
                "continuation_source_agent_id"
            ],
        )
        self.assertIn(
            "completion_schema_invalid",
            gateway.requests[0].agent.contract,
        )

    async def test_fixed_action_adapter_rejects_dynamic_native_capability(self) -> None:
        gateway = SequenceGateway([])
        dynamic_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    FakeTool({}),
                    ToolCapability(
                        tool_id="wiki.search",
                        dataset_scope=("triviaqa",),
                        action_schemas={},
                        input_schema={"type": "object"},
                        output_schema={"type": "object"},
                        side_effect="environment_state_transition",
                        timeout_seconds=1.0,
                        version="dynamic-test-v1",
                    ),
                ),
            )
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=dynamic_registry,
            max_turns=1,
            max_tool_calls=1,
        )

        with self.assertRaisesRegex(ReactExecutionError, "fixed action schemas"):
            await adapter.execute(request())
        self.assertEqual([], gateway.requests)

    async def test_parse_error_tool_observation_and_explicit_completion(self) -> None:
        gateway = SequenceGateway(
            [
                "not JSON",
                action(
                    "tool",
                    name="search",
                    arguments={"query": "first published algorithm author"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=4,
            max_tool_calls=2,
            max_action_tokens=256,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(3, response.metadata["react_turns_used"])
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(
            "parse_error", response.metadata["react_trace"][0]["observation_status"]
        )
        self.assertEqual(
            "not JSON", response.metadata["react_trace"][0]["action_text"]
        )
        self.assertNotIn("not JSON", gateway.requests[1].agent.contract)
        self.assertIn('"observation_status":"parse_error"', gateway.requests[1].agent.contract)
        self.assertIn(
            '"expected_top_level_fields":["arguments","kind","name","resource_id","skill_id"]',
            gateway.requests[1].agent.contract,
        )
        self.assertIn(
            '"forbidden_wrapper_fields":["action_envelope","argument_json_schema"]',
            gateway.requests[1].agent.contract,
        )
        self.assertEqual(
            "wiki.search", response.metadata["tool_receipts"][0]["tool_id"]
        )
        self.assertIn('name is "search"', gateway.requests[0].agent.contract)
        self.assertIn('"required":["query"]', gateway.requests[0].agent.contract)
        self.assertNotIn('"name":"tool action"', gateway.requests[0].agent.contract)
        self.assertIn("passage_ids", gateway.requests[2].agent.contract)
        self.assertIn("executed_action", gateway.requests[2].agent.contract)
        self.assertEqual("256", gateway.requests[0].model.metadata["max_tokens"])

    async def test_truncated_action_gets_one_short_same_model_regeneration(
        self,
    ) -> None:
        gateway = ResponseSequenceGateway(
            [
                AgentResponse(
                    '{"arguments":{"value":"unfinished"',
                    {
                        "finish_reason": "length",
                        "provider_request_id": "first",
                    },
                ),
                AgentResponse(
                    action(
                        "complete",
                        name="complete",
                        arguments={"value": "Ada Lovelace"},
                        resource_id=None,
                    ),
                    {
                        "finish_reason": "stop",
                        "provider_request_id": "repair",
                    },
                ),
            ]
        )
        active_request = replace(
            request(),
            model=ModelSpec(
                "m",
                "fake",
                metadata={
                    "chat_template_enable_thinking": "true",
                    "require_reasoning_trace": "true",
                },
            ),
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=4,
            max_tool_calls=0,
            max_action_tokens=4096,
            thinking_budget_tokens=1024,
        ).execute(active_request)

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(2, len(gateway.requests))
        self.assertEqual(
            gateway.requests[0].model.model_id,
            gateway.requests[1].model.model_id,
        )
        self.assertEqual(
            gateway.requests[0].provider.provider_id,
            gateway.requests[1].provider.provider_id,
        )
        self.assertEqual("5120", gateway.requests[0].model.metadata["max_tokens"])
        self.assertEqual(
            "1024",
            gateway.requests[0].model.metadata["thinking_budget_tokens"],
        )
        self.assertEqual("512", gateway.requests[1].model.metadata["max_tokens"])
        self.assertEqual(
            "false",
            gateway.requests[1].model.metadata["chat_template_enable_thinking"],
        )
        self.assertEqual(
            "false",
            gateway.requests[1].model.metadata["require_reasoning_trace"],
        )
        self.assertNotIn(
            "thinking_budget_tokens",
            gateway.requests[1].model.metadata,
        )
        first_trace = response.metadata["react_trace"][0]
        self.assertEqual("parse_error", first_trace["observation_status"])
        self.assertEqual(
            "structured_action_truncated",
            first_trace["public_error_code"],
        )
        self.assertEqual(
            '{"arguments":{"value":"unfinished"',
            first_trace["action_text"],
        )
        self.assertEqual(
            ["length", "stop"],
            [
                item["metadata"]["finish_reason"]
                for item in response.metadata["model_calls"]
            ],
        )
        self.assertEqual(
            "structured_action_truncation_regeneration",
            response.metadata["model_calls"][1]["generation_mode"],
        )

    async def test_truncated_action_regeneration_fails_fast_after_two_calls(
        self,
    ) -> None:
        gateway = ResponseSequenceGateway(
            [
                AgentResponse("first invalid prefix", {"finish_reason": "length"}),
                AgentResponse("second invalid response", {"finish_reason": "stop"}),
            ]
        )
        active_request = replace(
            request(),
            model=ModelSpec(
                "m",
                "fake",
                metadata={
                    "chat_template_enable_thinking": "true",
                    "require_reasoning_trace": "true",
                },
            ),
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=8,
            max_tool_calls=0,
            max_action_tokens=4096,
            thinking_budget_tokens=1024,
        )

        with self.assertRaises(ReactExecutionError) as raised:
            await adapter.execute(active_request)

        failure = raised.exception
        self.assertEqual(2, len(gateway.requests))
        self.assertEqual(
            "structured_action_serialization_failure",
            failure.failure_category,
        )
        self.assertEqual("output_truncation", failure.failure_reason)
        self.assertEqual(1, failure.bounded_regeneration_attempt_count)
        self.assertTrue(failure.regeneration_exhausted)
        self.assertFalse(failure.tool_plan_exhausted)
        self.assertEqual(2, len(failure.model_calls))
        self.assertEqual(2, len(failure.react_trace))
        self.assertEqual(
            "structured_action_regeneration_failed",
            failure.react_trace[-1]["public_error_code"],
        )

    async def test_length_reason_with_complete_structured_action_is_consumed(
        self,
    ) -> None:
        gateway = ResponseSequenceGateway(
            [
                AgentResponse(
                    action(
                        "complete",
                        name="complete",
                        arguments={"value": "Ada Lovelace"},
                        resource_id=None,
                    ),
                    {"finish_reason": "length"},
                )
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=0,
            max_action_tokens=4096,
        ).execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(1, len(gateway.requests))
        self.assertEqual(
            "completed",
            response.metadata["react_trace"][0]["observation_status"],
        )

    async def test_argument_schema_is_enforced_before_tool_dispatch(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={},
                    resource_id="wiki.search",
                ),
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=1,
        ).execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        invalid = response.metadata["react_trace"][0]
        self.assertEqual(
            "tool_arguments_schema_invalid",
            invalid["public_error_code"],
        )
        self.assertIn("query", invalid["argument_validation"]["message"])

    async def test_unregistered_action_name_is_rejected_before_backend(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="tool action",
                    arguments={"query": "wrong action name"},
                    resource_id="wiki.search",
                ),
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=2,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            "tool_action_not_registered",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertEqual(
            ["search"],
            response.metadata["react_trace"][0]["allowed_action_names"],
        )

    async def test_skill_action_remains_unavailable_to_executor(self) -> None:
        gateway = SequenceGateway(
            [
                json.dumps(
                    {
                        "kind": "skill",
                        "name": "invoke",
                        "arguments": {},
                        "resource_id": "wiki.search",
                        "skill_id": "candidate.skill",
                    }
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=1,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(0, response.metadata["tool_calls"])
        self.assertEqual(
            "skill_action_not_admitted",
            response.metadata["react_trace"][0]["public_error_code"],
        )

    async def test_turn_budget_requires_explicit_completion(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={"query": "x"},
                    resource_id="wiki.search",
                )
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
        )
        with self.assertRaisesRegex(ReactExecutionError, "exhausted") as raised:
            await adapter.execute(request())
        self.assertEqual(1, len(raised.exception.react_trace))
        self.assertEqual(0, len(raised.exception.tool_receipts))
        self.assertEqual(1, len(raised.exception.model_calls))
        self.assertEqual(
            "tool_action_not_admitted",
            raised.exception.react_trace[0]["public_error_code"],
        )
        response_schema = json.loads(
            gateway.requests[0].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            "complete", response_schema["properties"]["kind"]["const"]
        )
        self.assertIn(
            "admits only the explicit complete StructuredAction",
            gateway.requests[0].agent.contract,
        )

    async def test_payload_ok_false_is_not_reported_as_tool_success(self) -> None:
        failed_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    FakeTool(
                        {
                            "search": lambda arguments: {
                                "ok": False,
                                "error": "query rejected",
                                "query": arguments["query"],
                            }
                        }
                    ),
                    registry().require_capability("wiki.search"),
                ),
            )
        )
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "insufficient evidence"},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=failed_registry,
            max_turns=2,
            max_tool_calls=1,
        ).execute(request())

        self.assertEqual("insufficient evidence", response.text)
        observation = response.metadata["react_trace"][0]["observation"]
        self.assertEqual("tool_error", observation["observation_status"])
        self.assertEqual("tool_result_not_ok", observation["public_error_code"])
        self.assertEqual("completed", observation["tool_invocation_status"])
        self.assertFalse(observation["result"]["ok"])
        self.assertIsNone(response.metadata["tool_receipts"][0]["error_type"])

    async def test_tool_stdout_never_becomes_completion_without_complete_action(self) -> None:
        stdout_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    FakeTool(
                        {
                            "search": lambda arguments: {
                                "ok": True,
                                "stdout": "Ada Lovelace",
                                "query": arguments["query"],
                            }
                        }
                    ),
                    registry().require_capability("wiki.search"),
                ),
            )
        )
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                "still not a StructuredAction",
            ]
        )

        with self.assertRaisesRegex(ReactExecutionError, "without a valid completion"):
            await ToolReactExecutionAdapter(
                gateway=gateway,
                tool_registry=stdout_registry,
                max_turns=2,
                max_tool_calls=1,
            ).execute(request())

        self.assertEqual(2, len(gateway.requests))
        final_schema = json.loads(
            gateway.requests[1].model.metadata["response_json_schema"]
        )
        self.assertEqual(
            "complete", final_schema["properties"]["kind"]["const"]
        )

    async def test_identical_failed_tool_request_is_not_dispatched_twice(self) -> None:
        class FailingTool:
            def __init__(self) -> None:
                self.calls = 0

            async def invoke(self, request):
                del request
                self.calls += 1
                raise TimeoutError("public timeout")

        backend = FailingTool()
        tool_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    backend,
                    registry().require_capability("wiki.search"),
                ),
            )
        )
        repeated = action(
            "tool",
            name="search",
            arguments={"query": "same failed query"},
            resource_id="wiki.search",
        )
        gateway = SequenceGateway(
            [
                repeated,
                repeated,
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "insufficient evidence"},
                    resource_id=None,
                ),
            ]
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=3,
            max_tool_calls=3,
        ).execute(request())

        self.assertEqual(1, backend.calls)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(
            "duplicate_tool_request",
            response.metadata["react_trace"][1]["public_error_code"],
        )
        self.assertEqual(
            1, response.metadata["react_trace"][1]["repeat_count"]
        )
        self.assertEqual(
            "tool_error",
            response.metadata["react_trace"][1]["cached_observation"][
                "observation_status"
            ],
        )
        self.assertIn("cached observation", gateway.requests[2].agent.contract)
        self.assertIn("executed_action", gateway.requests[2].agent.contract)


if __name__ == "__main__":
    unittest.main()
