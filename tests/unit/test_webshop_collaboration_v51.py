"""Mock-only regressions for public state, directed communication and recovery."""

from __future__ import annotations

import json
import unittest
from collections.abc import Mapping

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.environment_execution import build_environment_execution_resources
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


GOAL = "Find a foldable table with a steel frame, price lower than 30.00 dollars."


class _Session:
    environment_id = "fake:webshop:collaboration"
    task_family = "webshop"

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.reset_count = 0
        self.available_actions: object = {
            "has_search_bar": True,
            "clickables": ["Search"],
        }

    def reset(self) -> str:
        self.reset_count += 1
        return "WebShop [SEP] Search"

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        if action.startswith("search["):
            observation = (
                "Instruction: [SEP] " + GOAL + " [SEP] Back to Search "
                "[SEP] Page 1 (Total results: 1) [SEP] B000000001 "
                "[SEP] Foldable Table [SEP] $12.00"
            )
            clickables = ["Back to Search", "B000000001"]
        elif action == "click[B000000001]":
            observation = (
                "Instruction: [SEP] " + GOAL + " [SEP] Back to Search "
                "[SEP] < Prev [SEP] Foldable Table [SEP] Price: $12.00 "
                "[SEP] Description [SEP] Features [SEP] Buy Now"
            )
            clickables = ["Back to Search", "< Prev", "Description", "Features", "Buy Now"]
        elif action == "click[Features]":
            observation = (
                "Instruction: [SEP] " + GOAL + " [SEP] Back to Search "
                "[SEP] < Prev [SEP] Foldable Table has a steel frame."
            )
            clickables = ["Back to Search", "< Prev"]
        else:
            raise AssertionError(f"unexpected fake environment action: {action}")
        self.available_actions = {"has_search_bar": False, "clickables": clickables}
        # Deliberately private fields: the Canvas/public request must not copy them.
        return observation, 0.0, False, {"graded_score": 0.0, "hidden_goal": "private"}


class _Gateway:
    def __init__(self, *, fail_analysis_revision: int | None = None) -> None:
        self.requests: list[AgentRequest] = []
        self.actor_requests: list[AgentRequest] = []
        self.analysis_requests: list[AgentRequest] = []
        self.fail_analysis_revision = fail_analysis_revision

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if request.agent.id != "actor":
            self.analysis_requests.append(request)
            state = request.public_environment_state
            if state is None:
                return AgentResponse("No public product evidence is available.")
            revision = state.get("environment_revision")
            if revision == self.fail_analysis_revision:
                raise RuntimeError("mock analysis failed before a new environment action")
            return AgentResponse(
                f"Public revision {revision}: {state['current_observation']} "
                "Check the observed material against the original instruction."
            )
        self.actor_requests.append(request)
        actions = [
            ("search", {"query": "foldable table steel frame"}),
            ("click", {"target": "B000000001"}),
            ("click", {"target": "Features"}),
        ]
        name, arguments = actions[len(self.actor_requests) - 1]
        return AgentResponse(json.dumps({
            "resource_id": "webshop",
            "kind": "tool",
            "name": name,
            "arguments": arguments,
            "skill_id": None,
        }))


def _fixture(*, fail_analysis_revision: int | None = None):
    session = _Session()
    gateway = _Gateway(fail_analysis_revision=fail_analysis_revision)
    resources = build_environment_execution_resources(
        gateway=gateway,
        session_factory=lambda _request: session,
        task_family="webshop",
        max_turns=5,
        stepwise_director=True,
        structured_actions=True,
    )
    registry = ModelRegistry(
        [ProviderSpec("fake", kind="test")], [ModelSpec("m", "fake")]
    )
    runtime = AgentRuntime(
        registry,
        gateway,
        execution_adapters={"react": resources.execution_adapter},
        tool_registry=resources.tool_registry,
        dataset_id="webshop",
    )
    canvas = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem=GOAL,
        execute_on_edit=True,
        required_tool_id=resources.tool_id,
        allowed_actions=(
            "add_agent", "modify_agent", "delete_agent", "set_relation",
            "set_output", "continue", "finish",
        ),
    )
    return session, gateway, canvas, runtime


async def _build_directed_pair(canvas: AgentWorkflowEnv) -> None:
    added = await canvas.step(json.dumps({
        "action": "add_agent", "agent_id": "actor", "model_id": "m",
        "contract": "Choose one environment action using public evidence and routed advice.",
        "execution_mode": "react", "allowed_tools": ["webshop.environment"],
    }))
    assert added.accepted, added.feedback
    added = await canvas.step(json.dumps({
        "action": "add_agent", "agent_id": "analysis", "model_id": "m",
        "contract": "Check the public product evidence against the complete instruction.",
    }))
    assert added.accepted, added.feedback
    related = await canvas.step(json.dumps({
        "action": "set_relation", "source_id": "analysis", "target_id": "actor",
        "source_to_target": True, "target_to_source": False,
    }))
    assert related.accepted, related.feedback
    selected = await canvas.step('{"action":"set_output","agent_id":"actor"}')
    assert selected.accepted, selected.feedback


class WebShopCollaborationV51Tests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_refreshes_analysis_and_routes_advice_after_each_action(self) -> None:
        session, gateway, canvas, _runtime = _fixture()
        await _build_directed_pair(canvas)
        # ADD analysis and SET_OUTPUT do not consume another environment step.
        self.assertEqual(2, len(session.actions))
        self.assertEqual(1, len(gateway.analysis_requests))
        first_analysis = gateway.analysis_requests[0]
        self.assertEqual(GOAL, first_analysis.problem)
        self.assertIsInstance(first_analysis.public_environment_state, Mapping)
        assert first_analysis.public_environment_state is not None
        self.assertEqual(1, first_analysis.public_environment_state["environment_revision"])
        self.assertIn("click[B000000001]", first_analysis.public_environment_state["admissible_actions"])
        self.assertEqual((), first_analysis.upstream)  # shared state is not a fabricated edge
        self.assertEqual("analysis", gateway.actor_requests[1].upstream[0].source_agent_id)
        self.assertIn("Public revision 1", gateway.actor_requests[1].upstream[0].content)
        before_revision = canvas.revision
        result = await canvas.step('{"action":"continue"}')
        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual(before_revision, canvas.revision)
        self.assertEqual(3, len(session.actions))
        self.assertEqual(1, session.reset_count)
        self.assertEqual(2, len(gateway.analysis_requests))
        new_analysis = gateway.analysis_requests[-1]
        assert new_analysis.public_environment_state is not None
        self.assertEqual(2, new_analysis.public_environment_state["environment_revision"])
        self.assertIn("Price: $12.00", new_analysis.public_environment_state["current_observation"])
        self.assertIn("click[Features]", new_analysis.public_environment_state["admissible_actions"])
        routed = gateway.actor_requests[-1].upstream[0]
        self.assertIn("Public revision 2", routed.content)
        self.assertEqual(2, routed.environment_revision)
        self.assertNotEqual(gateway.actor_requests[1].upstream[0].artifact_version, routed.artifact_version)
        state = canvas.public_environment_state()
        assert state is not None
        self.assertEqual(3, state["environment_revision"])
        self.assertEqual("click[Features]", state["last_action"])
        self.assertIn("steel frame", state["current_observation"])
        self.assertEqual(GOAL, state["original_task_instruction"])
        self.assertIn("click[< Prev]", state["admissible_actions"])
        self.assertNotIn("graded_score", state)
        self.assertNotIn("hidden_goal", state)
        self.assertFalse(state["environment_terminal"])
        rejected = await canvas.step('{"action":"finish"}')
        self.assertFalse(rejected.accepted)  # state visibility does not relax FINISH

    async def test_failed_analysis_retains_public_state_but_not_stale_output(self) -> None:
        session, gateway, canvas, _runtime = _fixture(fail_analysis_revision=2)
        await _build_directed_pair(canvas)
        before = canvas.public_environment_state()
        assert before is not None
        await canvas.step('{"action":"continue"}')
        self.assertEqual(2, len(session.actions))
        self.assertEqual(2, len(gateway.actor_requests))
        self.assertNotIn("actor", canvas._progressive_outputs)
        self.assertNotIn("analysis", canvas._progressive_outputs)
        retained = canvas.public_environment_state()
        self.assertEqual(before, retained)
        self.assertFalse((await canvas.step('{"action":"finish"}')).accepted)
        canvas.reset("A different shopping instruction")
        self.assertIsNone(canvas.public_environment_state())

    async def test_no_public_state_does_not_inject_product_evidence_or_other_agent_artifacts(self) -> None:
        _session, gateway, _canvas, runtime = _fixture()
        graph = AgentGraph(
            [AgentNode("left", "m", "Inspect any provided public evidence."),
             AgentNode("right", "m", "Check uncertainty in the available evidence.")],
            [AgentRelation("left", "right", True, True)],
            output_agent_id="right",
        )
        result = await runtime.execute(graph, GOAL)
        self.assertEqual(4, len(result.calls))
        for request in gateway.requests:
            self.assertIsNone(request.public_environment_state)
            self.assertEqual(GOAL, request.problem)
            if request.phase is ExecutionPhase.DRAFT:
                self.assertEqual((), request.upstream)
                self.assertIsNone(request.peer_draft)
        self.assertTrue(all("No public product evidence" in text for text in result.outputs.values()))

    async def test_unavailable_tool_capability_cannot_produce_an_artifact(self) -> None:
        _session, gateway, _canvas, runtime = _fixture()
        graph = AgentGraph(
            [AgentNode("analysis", "m", "Read product data.",
                       execution_mode="react", allowed_tools=("missing.environment",))],
            output_agent_id="analysis",
        )
        with self.assertRaises(AgentRuntimeError):
            await runtime.execute(graph, GOAL)
        self.assertEqual([], gateway.requests)
