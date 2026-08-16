"""Read-only AgentGraph search-distribution diagnostics for saved trajectories.

The shape calculations reuse :mod:`src.interactive.agent_graph`; this module
only reconstructs saved Canvas receipts and aggregates observed behavior.  It
does not execute a workflow, call a model, change reward, or infer causal
dependency from answer differences.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .agent_graph import (
    AgentGraph,
    AgentNode,
    AgentRelation,
    DependencyEdgeEvidence,
)


def graph_from_receipt(value: Mapping[str, Any]) -> AgentGraph:
    """Reconstruct an AgentGraph from a saved ``graph_snapshot`` mapping."""

    nodes = []
    for raw in value.get("nodes", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("graph node receipt must be an object")
        nodes.append(
            AgentNode(
                str(raw.get("id", "")),
                str(raw.get("model_id", "")),
                contract=str(raw.get("contract", raw.get("prompt", ""))),
            )
        )
    relations = []
    for raw in value.get("relations", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("graph relation receipt must be an object")
        relations.append(
            AgentRelation(
                str(raw.get("source_id", "")),
                str(raw.get("target_id", "")),
                bool(raw.get("source_to_target", False)),
                bool(raw.get("target_to_source", False)),
            )
        )
    output = value.get("output_agent_id")
    if output is not None and not isinstance(output, str):
        raise ValueError("graph output_agent_id receipt must be a string or null")
    revision = value.get("revision", 0)
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("graph revision receipt must be an integer")
    return AgentGraph(nodes, relations, output, revision)


def _prompt_state(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    prompt = turn.get("prompt")
    if not isinstance(prompt, str):
        return {}
    _, separator, payload = prompt.partition("\n\n")
    if not separator:
        return {}
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _action_name(turn: Mapping[str, Any]) -> str:
    action = turn.get("action")
    if not isinstance(action, Mapping):
        return "parse_failed"
    name = action.get("action")
    return str(name) if isinstance(name, str) and name else "parse_failed"


def _turn_accepted(turn: Mapping[str, Any]) -> bool:
    feedback = turn.get("canvas_feedback")
    return isinstance(feedback, str) and (
        feedback.startswith("accepted ") or feedback == "workflow finished"
    )


def _runtime_delivery_evidence(
    turns: Sequence[Mapping[str, Any]],
    graph: AgentGraph,
) -> tuple[DependencyEdgeEvidence, ...]:
    """Grade exact matching, non-empty runtime deliveries as weak evidence.

    Delivery proves transport but not that the downstream model used the
    artifact.  Consequently this function can never emit ``verified``.
    """

    graph_revision = graph.revision
    directed_edges = {
        edge for relation in graph.relations for edge in relation.directed_edges()
    }
    result: list[DependencyEdgeEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    for turn in turns:
        executions = turn.get("executions", ())
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            metadata = execution.get("metadata")
            request = metadata.get("request") if isinstance(metadata, Mapping) else None
            if not isinstance(request, Mapping):
                continue
            if request.get("graph_revision") != graph_revision:
                continue
            upstream = request.get("upstream", ())
            if not isinstance(upstream, Sequence) or isinstance(upstream, (str, bytes)):
                continue
            execution_id = execution.get("execution_id")
            evidence_id = execution_id if isinstance(execution_id, str) else None
            for message in upstream:
                if not isinstance(message, Mapping):
                    continue
                source = message.get("source_agent_id")
                target = message.get("target_agent_id")
                content = message.get("content")
                if (
                    not isinstance(source, str)
                    or not isinstance(target, str)
                    or (source, target) not in directed_edges
                    or not isinstance(content, str)
                    or not content.strip()
                    or message.get("graph_revision") != graph_revision
                ):
                    continue
                key = (source, target, evidence_id or "")
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    DependencyEdgeEvidence(
                        source,
                        target,
                        "weak",
                        evidence_id=evidence_id,
                    )
                )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TrajectoryGraphDiagnostic:
    task_id: str
    action_sequence: tuple[str, ...]
    turn_count: int
    rejected_turn_count: int
    parse_failure_count: int
    graph_revision: int
    agent_count: int
    relation_count: int
    structural_depth: int
    topology_family: str
    topology_motifs: tuple[str, ...]
    effective_dependency_depth: int
    effective_dependency_status: str
    full_structural_depth_evidence_status: str
    execution_turn_count: int
    executor_call_count: int
    minimum_final_construction_actions: int
    turn_overhead: int
    edit_overhead: int
    set_output_before_last_relation: bool
    explicit_finish: bool
    termination_reason: str
    max_rounds_seen: int | None
    prompt_character_mean: float
    prompt_character_max: int
    visible_history_max: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action_sequence": list(self.action_sequence),
            "turn_count": self.turn_count,
            "rejected_turn_count": self.rejected_turn_count,
            "parse_failure_count": self.parse_failure_count,
            "graph_revision": self.graph_revision,
            "agent_count": self.agent_count,
            "relation_count": self.relation_count,
            "structural_depth": self.structural_depth,
            "topology_family": self.topology_family,
            "topology_motifs": list(self.topology_motifs),
            "effective_dependency_depth": self.effective_dependency_depth,
            "effective_dependency_status": self.effective_dependency_status,
            "full_structural_depth_evidence_status": (
                self.full_structural_depth_evidence_status
            ),
            "execution_turn_count": self.execution_turn_count,
            "executor_call_count": self.executor_call_count,
            "minimum_final_construction_actions": (
                self.minimum_final_construction_actions
            ),
            "turn_overhead": self.turn_overhead,
            "edit_overhead": self.edit_overhead,
            "set_output_before_last_relation": self.set_output_before_last_relation,
            "explicit_finish": self.explicit_finish,
            "termination_reason": self.termination_reason,
            "max_rounds_seen": self.max_rounds_seen,
            "prompt_character_mean": self.prompt_character_mean,
            "prompt_character_max": self.prompt_character_max,
            "visible_history_max": self.visible_history_max,
        }


def diagnose_trajectory(record: Mapping[str, Any]) -> TrajectoryGraphDiagnostic:
    """Diagnose one saved trajectory without replaying it."""

    raw_turns = record.get("turns", ())
    if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, (str, bytes)):
        raise ValueError("trajectory turns must be a sequence")
    turns = tuple(turn for turn in raw_turns if isinstance(turn, Mapping))
    final_snapshot: Mapping[str, Any] = {}
    if turns and isinstance(turns[-1].get("graph_snapshot"), Mapping):
        final_snapshot = turns[-1]["graph_snapshot"]
    graph = graph_from_receipt(final_snapshot)
    topology = graph.topology_statistics()
    dependency = graph.effective_dependency_statistics(
        _runtime_delivery_evidence(turns, graph)
    )

    actions = tuple(_action_name(turn) for turn in turns)
    output_indices = [index for index, action in enumerate(actions) if action == "set_output"]
    relation_indices = [
        index for index, action in enumerate(actions) if action == "set_relation"
    ]
    prompt_states = [_prompt_state(turn) for turn in turns]
    max_round_values = [
        state["max_rounds"]
        for state in prompt_states
        if isinstance(state.get("max_rounds"), int)
        and not isinstance(state.get("max_rounds"), bool)
    ]
    prompt_lengths = [
        len(turn["prompt"])
        for turn in turns
        if isinstance(turn.get("prompt"), str)
    ]
    history_lengths = [
        len(state["recent_canvas_history"])
        for state in prompt_states
        if isinstance(state.get("recent_canvas_history"), list)
    ]
    execution_counts = [
        len(turn["executions"])
        for turn in turns
        if isinstance(turn.get("executions"), list)
    ]
    explicit_finish = bool(record.get("explicit_finish", False))
    minimum_actions = (
        len(graph.nodes)
        + len(graph.relations)
        + int(graph.output_agent_id is not None)
        + int(explicit_finish)
    )
    minimum_edits = (
        len(graph.nodes) + len(graph.relations) + int(graph.output_agent_id is not None)
    )
    task = record.get("task")
    task_id = task.get("task_id", "") if isinstance(task, Mapping) else ""
    return TrajectoryGraphDiagnostic(
        task_id=str(task_id),
        action_sequence=actions,
        turn_count=len(turns),
        rejected_turn_count=sum(not _turn_accepted(turn) for turn in turns),
        parse_failure_count=actions.count("parse_failed"),
        graph_revision=graph.revision,
        agent_count=len(graph.nodes),
        relation_count=len(graph.relations),
        structural_depth=int(topology["structural_depth"]),
        topology_family=str(topology["topology_family"]),
        topology_motifs=tuple(str(item) for item in topology["topology_motifs"]),
        effective_dependency_depth=int(dependency["effective_dependency_depth"]),
        effective_dependency_status=str(dependency["evidence_status"]),
        full_structural_depth_evidence_status=str(
            dependency["full_structural_depth_evidence_status"]
        ),
        execution_turn_count=sum(count > 0 for count in execution_counts),
        executor_call_count=sum(execution_counts),
        minimum_final_construction_actions=minimum_actions,
        turn_overhead=len(turns) - minimum_actions,
        edit_overhead=graph.revision - minimum_edits,
        set_output_before_last_relation=bool(
            output_indices
            and relation_indices
            and output_indices[0] < relation_indices[-1]
        ),
        explicit_finish=explicit_finish,
        termination_reason=str(record.get("termination_reason", "")),
        max_rounds_seen=max(max_round_values) if max_round_values else None,
        prompt_character_mean=fmean(prompt_lengths) if prompt_lengths else 0.0,
        prompt_character_max=max(prompt_lengths, default=0),
        visible_history_max=max(history_lengths, default=0),
    )


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def aggregate_trajectory_diagnostics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate atomic-cost, depth, topology, execution, and failure facts."""

    items = [diagnose_trajectory(record) for record in records]
    if not items:
        return {"task_count": 0, "trajectories": []}
    actions = Counter(action for item in items for action in item.action_sequence)
    sequences = Counter(" -> ".join(item.action_sequence) for item in items)
    total_turns = sum(item.turn_count for item in items)
    grouped_costs: dict[int, list[TrajectoryGraphDiagnostic]] = defaultdict(list)
    for item in items:
        grouped_costs[item.agent_count].append(item)
    return {
        "task_count": len(items),
        "agent_count_distribution": _distribution(item.agent_count for item in items),
        "structural_depth_distribution": _distribution(
            item.structural_depth for item in items
        ),
        "effective_dependency_depth_distribution": _distribution(
            item.effective_dependency_depth for item in items
        ),
        "effective_dependency_status_distribution": _distribution(
            item.effective_dependency_status for item in items
        ),
        "full_structural_depth_evidence_status_distribution": _distribution(
            item.full_structural_depth_evidence_status for item in items
        ),
        "topology_family_distribution": _distribution(
            item.topology_family for item in items
        ),
        "action_counts": dict(sorted(actions.items())),
        "action_sequence_counts": dict(sequences.most_common()),
        "turn_count_distribution": _distribution(item.turn_count for item in items),
        "mean_turn_count": fmean(item.turn_count for item in items),
        "maximum_turn_count": max(item.turn_count for item in items),
        "rejected_turn_count": sum(item.rejected_turn_count for item in items),
        "rejected_turn_rate": (
            sum(item.rejected_turn_count for item in items) / total_turns
        ),
        "parse_failure_count": sum(item.parse_failure_count for item in items),
        "explicit_finish_count": sum(item.explicit_finish for item in items),
        "max_rounds_termination_count": sum(
            item.termination_reason == "max_rounds" for item in items
        ),
        "max_rounds_values_seen": _distribution(
            item.max_rounds_seen for item in items if item.max_rounds_seen is not None
        ),
        "execution_turn_count": sum(item.execution_turn_count for item in items),
        "executor_call_count": sum(item.executor_call_count for item in items),
        "tasks_with_execution": sum(item.execution_turn_count > 0 for item in items),
        "set_output_before_last_relation_count": sum(
            item.set_output_before_last_relation for item in items
        ),
        "mean_prompt_characters": fmean(
            item.prompt_character_mean for item in items
        ),
        "maximum_prompt_characters": max(item.prompt_character_max for item in items),
        "maximum_visible_history": max(item.visible_history_max for item in items),
        "three_plus_agent_count": sum(item.agent_count >= 3 for item in items),
        "depth_three_plus_count": sum(item.structural_depth >= 3 for item in items),
        "atomic_cost_by_agent_count": {
            str(agent_count): {
                "tasks": len(group),
                "mean_minimum_final_construction_actions": fmean(
                    item.minimum_final_construction_actions for item in group
                ),
                "mean_observed_turns": fmean(item.turn_count for item in group),
                "mean_turn_overhead": fmean(item.turn_overhead for item in group),
                "mean_edit_overhead": fmean(item.edit_overhead for item in group),
            }
            for agent_count, group in sorted(grouped_costs.items())
        },
        "trajectories": [item.to_dict() for item in items],
    }


__all__ = [
    "TrajectoryGraphDiagnostic",
    "aggregate_trajectory_diagnostics",
    "diagnose_trajectory",
    "graph_from_receipt",
]
