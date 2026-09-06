"""Integration coverage for AIME free-AgentGraph coding collaboration.

The test deliberately uses only the existing AgentGraph, AgentRuntime,
ToolReactExecutionAdapter, typed communication envelope, and AIME evaluator
boundaries.  ``coding`` is an execution mode on one freely contracted Agent;
there is no role enum or task-specific workflow implementation here.
"""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import AgentRequest, AgentResponse, AgentRuntime
from src.interactive.aime2026_adapter import (
    extract_aime2026_candidate,
    score_aime2026_integer,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.react_execution import ToolReactExecutionAdapter
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


def _structured_action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "arguments": arguments,
            "kind": kind,
            "name": name,
            "resource_id": resource_id,
            "skill_id": None,
        },
        sort_keys=True,
    )


def _model_registry() -> ModelRegistry:
    return ModelRegistry(
        (ProviderSpec("fake", kind="test"),),
        (
            ModelSpec("reasoning-a", "fake", metadata={"coding_capable": "false"}),
            ModelSpec("coding-a", "fake", metadata={"coding_capable": "true"}),
            ModelSpec("reasoning-b", "fake", metadata={"coding_capable": "false"}),
        ),
    )


def _python_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolRegistration(
                "python.compute",
                FakeTool(
                    {
                        "run": lambda arguments: {
                            "ok": True,
                            "stdout": "65\n",
                            "executed_code": arguments["code"],
                        }
                    }
                ),
                ToolCapability(
                    tool_id="python.compute",
                    dataset_scope=("aime_2026",),
                    action_schemas={
                        "run": {
                            "type": "object",
                            "required": ["code"],
                            "properties": {"code": {"type": "string"}},
                            "additionalProperties": False,
                        }
                    },
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    side_effect="none",
                    timeout_seconds=1.0,
                    version="python-compute-test-v1",
                ),
            ),
        )
    )


class _CollaborationGateway:
    """Deterministic provider double that records the real Runtime requests."""

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self._coding_turn = 0

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        agent_id = request.agent.id
        if agent_id == "derivation":
            return AgentResponse(
                "The sum 1+2+...+10 is 55; adding 10 gives 65.\n"
                "Final Answer: 65"
            )
        if agent_id == "independent":
            return AgentResponse(
                "Using n(n+1)/2 gives 10*11/2=55, hence 55+10=65.\n"
                "Final Answer: 65"
            )
        if agent_id == "compute_branch":
            self._coding_turn += 1
            if self._coding_turn == 1:
                return AgentResponse(
                    _structured_action(
                        "tool",
                        name="run",
                        arguments={
                            "code": "print(sum(range(1, 11)) + 10)"
                        },
                        resource_id="python.compute",
                    )
                )
            source = request.upstream[0]
            completed_artifact = (
                "The routed derivation gives 55+10=65, and the public Python "
                "observation independently prints 65.\n"
                "Final Answer: 65\n"
                "<artifact_assessments>"
                + json.dumps(
                    [
                        {
                            "assessed_artifact_id": source.artifact_id,
                            "candidate": "65",
                            "assessment": "supported",
                            "basis": (
                                "The routed arithmetic and the public computation "
                                "both derive 65."
                            ),
                            "counterexample": None,
                        }
                    ],
                    sort_keys=True,
                )
                + "</artifact_assessments>"
            )
            return AgentResponse(
                _structured_action(
                    "complete",
                    name="complete",
                    arguments={"value": completed_artifact},
                    resource_id=None,
                )
            )
        if agent_id == "merge":
            return AgentResponse(
                "The complete coding artifact and the independent derivation "
                "agree on 65, with no public conflict.\nFinal Answer: 65"
            )
        if agent_id == "terminal":
            return AgentResponse(r"\boxed{65}")
        raise AssertionError(f"unexpected Agent ID: {agent_id}")


class AIME2026CodingCollaborationTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_graph_routes_coding_receipts_and_nested_provenance(
        self,
    ) -> None:
        gateway = _CollaborationGateway()
        tools = _python_tool_registry()
        graph = AgentGraph(
            (
                AgentNode(
                    "derivation",
                    "reasoning-a",
                    "Derive a candidate from the problem and preserve the calculation.",
                    artifact_type="mathematical_derivation",
                ),
                AgentNode(
                    "independent",
                    "reasoning-b",
                    "Produce an independent calculation from the original problem.",
                    artifact_type="mathematical_derivation",
                ),
                AgentNode(
                    "compute_branch",
                    "coding-a",
                    (
                        "Consume the routed derivation, use the admitted computation "
                        "tool when useful, and return a complete public artifact."
                    ),
                    allowed_tools=("python.compute",),
                    execution_mode="coding",
                    artifact_type="computation_artifact",
                    completion_condition="return the complete checked calculation",
                ),
                AgentNode(
                    "merge",
                    "reasoning-b",
                    (
                        "Assess the complete routed artifacts and preserve the "
                        "evidence supporting the selected candidate."
                    ),
                    artifact_type="candidate_assessment",
                ),
                AgentNode(
                    "terminal",
                    "reasoning-a",
                    "Return the unique AIME integer from the routed public artifact.",
                    artifact_type="aime_integer",
                ),
            ),
            (
                AgentRelation("derivation", "compute_branch", True, False),
                AgentRelation("compute_branch", "merge", True, False),
                AgentRelation("independent", "merge", True, False),
                AgentRelation("merge", "terminal", True, False),
            ),
            output_agent_id="terminal",
        )
        coding_adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=tools,
            max_turns=2,
            max_tool_calls=1,
            execution_mode="coding",
        )
        runtime = AgentRuntime(
            _model_registry(),
            gateway,
            execution_adapters={"coding": coding_adapter},
            tool_registry=tools,
            dataset_id="aime_2026",
            artifact_assessment_protocol=(
                "provenance_bound_candidate_assessment_v2"
            ),
            max_concurrency=2,
        )

        result = await runtime.execute(
            graph,
            "Compute 1+2+...+10+10 and give the AIME integer answer.",
            run_id="aime-coding-collaboration",
        )

        # Free-text contracts remain the only semantic responsibilities;
        # coding is execution semantics rather than a fixed Agent role.
        self.assertTrue(all(node.role_family is None for node in graph.nodes))
        self.assertEqual(
            {"compute_branch"},
            {
                node.id
                for node in graph.nodes
                if node.execution_mode.value == "coding"
            },
        )

        coding_requests = [
            request
            for request in gateway.requests
            if request.agent.id == "compute_branch"
        ]
        self.assertEqual(2, len(coding_requests))
        coding_source = coding_requests[0].upstream[0]
        self.assertEqual("derivation", coding_source.source_agent_id)
        self.assertTrue(coding_source.artifact_complete)
        self.assertEqual(
            result.output_metadata["derivation"]["artifact_id"],
            coding_source.artifact_id,
        )
        self.assertIn(
            "55; adding 10 gives 65",
            coding_source.to_dict()["raw_output"],
        )

        coding_metadata = result.output_metadata["compute_branch"]
        self.assertEqual("coding", coding_metadata["tool_config"]["execution_mode"])
        coding_input = coding_metadata["input_artifact_provenance"][0]
        self.assertEqual("derivation", coding_input["source_agent_id"])
        self.assertEqual(coding_source.artifact_id, coding_input["artifact_id"])
        self.assertEqual(coding_source.content, coding_input["raw_output"])
        coding_trace = coding_metadata["react_trace"]
        self.assertEqual("tool", coding_trace[0]["structured_action"]["kind"])
        self.assertEqual(
            "python.compute",
            coding_trace[0]["structured_action"]["resource_id"],
        )
        self.assertEqual("complete", coding_trace[1]["structured_action"]["kind"])
        self.assertEqual(1, len(coding_metadata["tool_receipts"]))
        self.assertEqual(
            "admitted",
            coding_metadata["artifact_assessment_completion_receipt"][
                "admission_status"
            ],
        )
        self.assertEqual(
            coding_source.artifact_id,
            coding_metadata["artifact_assessment_completion_receipt"][
                "assessment_bindings"
            ][0]["artifact_id"],
        )

        # The direct fan-in receives the full coding artifact and its local
        # Tool receipt, alongside the independent reasoning artifact.
        merge_request = next(
            request for request in gateway.requests if request.agent.id == "merge"
        )
        self.assertEqual(
            ["compute_branch", "independent"],
            [message.source_agent_id for message in merge_request.upstream],
        )
        coding_envelope = next(
            message
            for message in merge_request.upstream
            if message.source_agent_id == "compute_branch"
        )
        coding_wire = coding_envelope.to_dict()
        self.assertEqual(
            result.output_metadata["compute_branch"]["artifact_id"],
            coding_wire["artifact_id"],
        )
        self.assertEqual(result.outputs["compute_branch"], coding_wire["raw_output"])
        self.assertEqual(1, len(coding_wire["tool_receipts"]))
        self.assertEqual("python.compute", coding_wire["tool_receipts"][0]["tool_id"])

        # One more reasoning hop keeps the fan-in sources and the coding Tool
        # receipt as nested (transitive), source-bound provenance rather than
        # flattening or dropping them.
        terminal_request = next(
            request
            for request in gateway.requests
            if request.agent.id == "terminal"
        )
        merge_envelope = terminal_request.upstream[0]
        self.assertEqual("merge", merge_envelope.source_agent_id)
        self.assertEqual(
            result.output_metadata["merge"]["artifact_id"],
            merge_envelope.artifact_id,
        )
        nested = {
            item["source_agent_id"]: item
            for item in merge_envelope.input_artifact_provenance
        }
        self.assertEqual({"compute_branch", "independent"}, set(nested))
        self.assertEqual(
            result.outputs["compute_branch"],
            nested["compute_branch"]["raw_output"],
        )
        self.assertEqual(
            result.output_metadata["compute_branch"]["artifact_id"],
            nested["compute_branch"]["artifact_id"],
        )
        self.assertEqual(1, len(nested["compute_branch"]["tool_receipts"]))
        self.assertEqual(
            "python.compute",
            nested["compute_branch"]["tool_receipts"][0]["tool_id"],
        )

        candidate, _, parsing_failure = extract_aime2026_candidate(
            result.final_answer
        )
        self.assertEqual("65", candidate)
        self.assertIsNone(parsing_failure)
        score = score_aime2026_integer(result.final_answer or "", ("65",))
        self.assertEqual(1.0, score.accuracy)
        self.assertEqual("65", score.canonical_prediction)


if __name__ == "__main__":
    unittest.main()
