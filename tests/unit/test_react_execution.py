from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
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


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(self.outputs.pop(0), {"provider_request_id": len(self.requests)})


class ToolReactExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_opt_in_completion_domain_is_enforced_after_parse(self) -> None:
        class SearchFirstAdapter(ToolReactExecutionAdapter):
            def _state_conditioned_action_domain(
                self,
                request: AgentRequest,
                observations: list[dict[str, object]],
            ) -> tuple[frozenset[tuple[str, str]], bool]:
                del request
                if any(
                    item.get("observation_status") == "success"
                    for item in observations
                ):
                    return frozenset(), True
                return frozenset({("wiki.search", "search")}), False

        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "premature"},
                    resource_id=None,
                ),
                action(
                    "tool",
                    name="search",
                    arguments={"query": "published algorithm"},
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
        adapter = SearchFirstAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=1,
            enforce_state_conditioned_completion_admission=True,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(
            "state_completion_not_admitted",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertEqual(1, response.metadata["tool_calls"])

    async def test_completion_domain_enforcement_is_opt_in(self) -> None:
        class SearchFirstAdapter(ToolReactExecutionAdapter):
            def _state_conditioned_action_domain(
                self,
                request: AgentRequest,
                observations: list[dict[str, object]],
            ) -> tuple[frozenset[tuple[str, str]], bool]:
                del request, observations
                return frozenset({("wiki.search", "search")}), False

        gateway = SequenceGateway(
            [
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "historical completion"},
                    resource_id=None,
                )
            ]
        )

        response = await SearchFirstAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
        ).execute(request())

        self.assertEqual("historical completion", response.text)

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
            },
            response.metadata["model_calls"][0]["requested_sampling"],
        )

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

    async def test_scientific_receipt_records_declared_thinking_mode(self) -> None:
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="healthbench-thinking-fixed-schedule",
            ordered_sequence_hash=stable_hash(["healthbench-professional:case-1"]),
            sequence_position=0,
            task_id="healthbench-professional:case-1",
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
        thinking_request = replace(
            base_request,
            model=replace(
                base_request.model,
                metadata={"chat_template_enable_thinking": "true"},
            ),
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
            sampling_base_seed=17,
            sampling_coordinate=coordinate,
        ).execute(thinking_request)

        requested = response.metadata["model_calls"][0][
            "scientific_sampling"
        ]["requested_sampling"]
        self.assertIs(True, requested["chat_template_enable_thinking"])
        self.assertEqual(
            "true",
            gateway.requests[0].model.metadata[
                "chat_template_enable_thinking"
            ],
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
        self.assertEqual(1, len(raised.exception.tool_receipts))
        self.assertEqual(1, len(raised.exception.model_calls))
        self.assertEqual(
            "wiki.search", raised.exception.tool_receipts[0]["tool_id"]
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
        self.assertIn("executed_action", gateway.requests[2].agent.contract)


if __name__ == "__main__":
    unittest.main()
