"""V4 feedback opt-in must not alter the established V3 Agent wire protocol.

Fixtures reuse the V3 synthetic receipt tests and the task-scoped runtime wiring
tests; no benchmark examples, provider calls, or model services are involved.
"""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3 as V3,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4 as V4,
    AgentResponse,
    AgentRuntime,
    CommunicationCondition,
    ExecutionPhase,
)
from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.openai_gateway import (
    MASKED_UPSTREAM_CONTENT,
    OpenAICompatibleGateway,
    _HEALTHBENCH_V3_EXECUTION_SUPPLEMENT,
    build_agent_messages,
)
from tests.unit.test_agent_runtime import RecordingGateway, registry
from tests.unit.test_healthbench_evidence_projection_v3 import (
    artifact,
    evidence,
    request,
    upstream,
)
from tests.unit.test_healthbench_query_budget_runtime_wiring import (
    SMOKE,
    _config,
    _task,
)


def test_v4_is_opt_in_and_preserves_historical_default():
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(str(root / "config/training_agent_graph.yaml"))
    assert config["agent_graph"].get("artifact_communication_profile", "legacy") != V4
    config["agent_graph"]["artifact_communication_profile"] = V4
    validate_agent_graph_config(config)
    assert AgentRuntime(registry(), RecordingGateway()).artifact_communication_profile == "legacy"


@pytest.mark.parametrize("phase", [ExecutionPhase.SINGLE, ExecutionPhase.DRAFT, ExecutionPhase.REVISION])
def test_v4_actual_payload_reuses_v3_including_peer_projection(phase):
    document = evidence("shared-source", "The original population qualifier remains visible.")
    cited = {**document, "supported_claim": "A producer interpretation.",
             "conditions_or_qualifiers": "Only in the documented population."}
    external = upstream(document, content=artifact(cited))
    kwargs = {"phase": phase}
    if phase is ExecutionPhase.REVISION:
        kwargs.update(own_draft="My draft.", peer_draft=upstream(
            document, evidence("peer-only"), source="peer", content=artifact(cited),
        ))
    old = request(external, profile=V3, **kwargs)
    new = replace(old, artifact_communication_profile=V4)
    gateway = OpenAICompatibleGateway()
    assert gateway.request_payload(new) == gateway.request_payload(old)
    text = json.dumps(gateway.request_payload(new)["messages"])
    assert cited["conditions_or_qualifiers"] in text
    assert "retrieved-not-endorsed" in text
    assert text.count('Synthetic source shared-source') == 1
    assert _HEALTHBENCH_V3_EXECUTION_SUPPLEMENT in build_agent_messages(new)[0]["content"]
    if phase is ExecutionPhase.REVISION:
        assert "peer-only" in text


def test_v4_masked_upstream_and_peer_reuse_existing_communication_condition():
    message = upstream(evidence("MASKED_DOCUMENT"), content=artifact(summary="MASKED_SUMMARY"),
                       input_artifact_provenance=(upstream(evidence("MASKED_NESTED")).to_dict(),))
    old = request(message, profile=V3, communication_condition=CommunicationCondition.UPSTREAM_MASKED,
                  phase=ExecutionPhase.REVISION, own_draft="Public own draft.", peer_draft=message)
    new = replace(old, artifact_communication_profile=V4)
    assert build_agent_messages(new) == build_agent_messages(old)
    text = json.dumps(build_agent_messages(new))
    assert MASKED_UPSTREAM_CONTENT in text
    for marker in ("MASKED_DOCUMENT", "MASKED_SUMMARY", "MASKED_NESTED", '"retrieval_evidence"'):
        assert marker not in text


def test_v4_non_healthbench_keeps_existing_generic_messages():
    old = request(upstream(evidence()), profile=V3, problem="A general factual question.")
    new = replace(old, artifact_communication_profile=V4)
    assert build_agent_messages(new) == build_agent_messages(old)
    assert _HEALTHBENCH_V3_EXECUTION_SUPPLEMENT not in build_agent_messages(new)[0]["content"]


def test_v4_runtime_routes_producer_context_and_immutable_reciprocal_drafts():
    class Gateway(RecordingGateway):
        async def generate(self, agent_request):
            self.requests.append(agent_request)
            return AgentResponse(f"{agent_request.phase.value}-{agent_request.agent.id}", {"finish_reason": "stop"})

    gateway = Gateway()
    graph = AgentGraph(
        [AgentNode("a", "m1", "Develop an independent draft."),
         AgentNode("b", "m2", "Compare the available claims.")],
        [AgentRelation("a", "b", True, True)], output_agent_id="b",
    )
    result = asyncio.run(AgentRuntime(registry(), gateway, artifact_communication_profile=V4).execute(
        graph, "An original synthetic question.", run_id="v4-reciprocal",
    ))
    assert result.final_answer == "revision-b"
    assert len(gateway.requests) == 4
    assert all(item.artifact_communication_profile == V4 for item in gateway.requests)
    revisions = [item for item in gateway.requests if item.phase is ExecutionPhase.REVISION]
    assert len(revisions) == 2
    for item in revisions:
        peer_id = "b" if item.agent.id == "a" else "a"
        assert item.own_draft == f"draft-{item.agent.id}"
        assert item.peer_draft.content == f"draft-{peer_id}"
        assert item.peer_draft.source_contract == next(node.contract for node in graph.nodes if node.id == peer_id)
        assert item.peer_draft.source_execution_mode == "reasoning"
        assert item.peer_draft.source_finish_reason == "stop"


@pytest.mark.parametrize("profile,structured", [(V3, True), (V4, True), ("legacy", False)])
def test_v4_healthbench_task_runtime_keeps_existing_evidence_validator(tmp_path, profile, structured):
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = _config()
    backend.registry = registry()
    backend.runtime = SimpleNamespace(gateway=RecordingGateway(), timeout_seconds=30.0,
                                      artifact_communication_profile=profile)
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    execution_adapter = SimpleNamespace(execute=Mock())
    with (
        patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
        patch.object(SMOKE, "open_healthbench_authoritative_tool_registry", return_value=opened),
        patch.object(SMOKE, "HealthBenchAuthoritativeReactExecutionAdapter", return_value=execution_adapter) as adapter,
    ):
        runtime, tools, close = backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert isinstance(runtime, AgentRuntime)
    assert runtime.artifact_communication_profile == profile
    assert runtime.dataset_id == "healthbench_professional"
    assert runtime.execution_adapters["react"] is execution_adapter
    assert adapter.call_args.kwargs["require_structured_evidence_artifact"] is structured
    assert tools is opened.registry
    assert close is opened.close
