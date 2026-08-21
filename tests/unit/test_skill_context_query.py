from __future__ import annotations

import importlib.util
from pathlib import Path

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.records import TaskRecord

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_agentgraph_smoke.py"
_SPEC = importlib.util.spec_from_file_location("skill_context_smoke_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_hotpot_skill_query_tags_exclude_non_public_native_annotations() -> None:
    task = TaskRecord(
        task_id="hotpotqa:context-test",
        question="Which entity supplies the bridge fact?",
        ground_truth="hidden-gold-answer",
        split="train",
        metadata={
            "dataset_key": "hotpotqa",
            "task_family": "hotpotqa",
            "task_type": "multi_hop_qa",
            "skillflow": {
                "extra": {"type": "bridge", "level": "hard"},
            },
            "evaluator_payload": {"answer": "hidden-gold-answer"},
        },
    )
    graph = AgentGraph(
        nodes=(
            AgentNode(
                "evidence",
                "model-a",
                "find first-hop evidence",
                role_family="evidence retrieval",
            ),
            AgentNode(
                "reasoner",
                "model-b",
                "resolve the bridge entity",
                role_family="bridge reasoning",
            ),
            AgentNode(
                "format",
                "model-c",
                "serialize the answer span",
                role_family="format",
            ),
        ),
        relations=(
            AgentRelation("evidence", "reasoner", True, False),
            AgentRelation("reasoner", "format", True, False),
        ),
        output_agent_id="format",
    )

    tags = set(
        _MODULE._skill_query_tags(
            task,
            graph,
            task_family="hotpotqa",
            graph_stage="before_final_answer",
            validation_issue_codes=(),
        )
    )

    assert 'task_context.dataset_key="hotpotqa"' in tags
    assert 'task_context.task_family="hotpotqa"' in tags
    assert not any(tag.startswith("task_context.task_type=") for tag in tags)
    assert not any(tag.startswith("task_context.task_subtype=") for tag in tags)
    assert not any(tag.startswith("task_context.difficulty=") for tag in tags)
    assert 'graph_prefix.topology_family="serial_3_plus"' in tags
    assert "graph_prefix.structural_depth=3" in tags
    assert 'graph_prefix.output_state="set"' in tags
    assert 'role_family.value="evidence retrieval"' in tags
    assert 'model.model_id="model-a"' in tags
    assert 'relation_motif.kind="unidirectional"' in tags
    assert 'graph_position.kind="root"' in tags
    assert 'graph_position.kind="intermediate"' in tags
    assert 'graph_position.kind="output"' in tags
    assert (
        "agent_context.role_model_relation_position="
        '{"graph_position":"root","model_id":"model-a",'
        '"relation_motif":"unidirectional","role_family":"evidence retrieval"}'
    ) in tags
    serialized = "\n".join(tags)
    assert task.task_id not in serialized
    assert "hidden-gold-answer" not in serialized
    assert "find first-hop evidence" not in serialized


def test_non_hotpot_skill_query_retains_existing_task_metadata_tags() -> None:
    task = TaskRecord(
        task_id="triviaqa:context-test",
        question="Question",
        ground_truth="answer",
        split="train",
        metadata={
            "dataset_key": "triviaqa",
            "task_family": "triviaqa",
            "task_type": "factual_qa",
            "skillflow": {"extra": {"type": "alias", "level": "hard"}},
        },
    )

    tags = set(
        _MODULE._skill_query_tags(
            task,
            AgentGraph(),
            task_family="triviaqa",
            graph_stage="empty_graph",
            validation_issue_codes=(),
        )
    )

    assert 'task_context.task_type="factual_qa"' in tags
    assert 'task_context.task_subtype="alias"' in tags
    assert 'task_context.difficulty="hard"' in tags


def test_skill_query_tags_use_current_prefix_not_a_future_topology() -> None:
    task = TaskRecord(
        task_id="hotpotqa:empty-prefix",
        question="Question",
        ground_truth="answer",
        split="train",
        metadata={"dataset_key": "hotpotqa", "task_family": "hotpotqa"},
    )

    tags = set(
        _MODULE._skill_query_tags(
            task,
            AgentGraph(),
            task_family="hotpotqa",
            graph_stage="empty_graph",
            validation_issue_codes=("output_agent_count",),
        )
    )

    assert 'graph_prefix.graph_stage="empty_graph"' in tags
    assert 'graph_prefix.topology_family="empty"' in tags
    assert "graph_prefix.structural_depth=0" in tags
    assert 'graph_prefix.output_state="unset"' in tags
    assert "output_agent_count" in tags  # backwards-compatible issue tag
    assert not any(tag.startswith("role_family.") for tag in tags)
    assert not any(tag.startswith("model.") for tag in tags)
    assert not any(tag.startswith("graph_position.") for tag in tags)


def test_missing_role_family_does_not_create_a_role_condition() -> None:
    task = TaskRecord(
        task_id="hotpotqa:legacy-role",
        question="Question",
        ground_truth="answer",
        split="train",
        metadata={"dataset_key": "hotpotqa", "task_family": "hotpotqa"},
    )
    graph = AgentGraph((AgentNode("legacy", "model-a", "free contract"),))

    tags = set(
        _MODULE._skill_query_tags(
            task,
            graph,
            task_family="hotpotqa",
            graph_stage="construction",
            validation_issue_codes=("output_agent_count",),
        )
    )

    assert 'model.model_id="model-a"' in tags
    assert 'graph_position.kind="root"' in tags
    assert 'graph_position.kind="sink"' in tags
    assert not any(tag.startswith("role_family.") for tag in tags)
    assert not any(tag.startswith("agent_context.") for tag in tags)
    assert "free contract" not in "\n".join(tags)
