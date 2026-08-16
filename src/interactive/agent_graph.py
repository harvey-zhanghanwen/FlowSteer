"""Free-agent graph data model, validation, revisioning, and snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Collection, Dict, Iterable, List, Optional, Set, Tuple, Union

from .model_registry import ModelRegistry


class GraphMutationError(ValueError):
    """Raised when an atomic graph mutation cannot be applied."""


DEPENDENCY_EVIDENCE_STATUSES = frozenset({"unverified", "weak", "verified"})


@dataclass(frozen=True, slots=True)
class DependencyEdgeEvidence:
    """Explicit evidence grade for one directed communication edge.

    The graph never promotes an edge from an answer change or a structural
    mask alone.  ``weak`` is appropriate for a matching runtime delivery
    receipt.  ``verified`` must be supplied only by a caller holding an
    independently validated paired-intervention receipt.  This keeps the
    read-only diagnostic separate from runtime, reward, and policy behavior.
    """

    source_id: str
    target_id: str
    status: str
    evidence_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("dependency evidence source_id must be non-empty")
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("dependency evidence target_id must be non-empty")
        source_id = self.source_id.strip()
        target_id = self.target_id.strip()
        if source_id == target_id:
            raise ValueError("dependency evidence cannot describe a self edge")
        if (
            not isinstance(self.status, str)
            or self.status not in DEPENDENCY_EVIDENCE_STATUSES
        ):
            raise ValueError(
                "dependency evidence status must be unverified, weak, or verified"
            )
        if self.evidence_id is not None and (
            not isinstance(self.evidence_id, str) or not self.evidence_id.strip()
        ):
            raise ValueError("dependency evidence_id must be non-empty when supplied")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "target_id", target_id)
        if self.evidence_id is not None:
            object.__setattr__(self, "evidence_id", self.evidence_id.strip())


@dataclass(frozen=True, slots=True, init=False)
class AgentNode:
    """A free Agent role bound to a stable model ID.

    ``prompt`` is accepted as an initialization alias for ``contract`` and is
    exposed as a read-only property for callers that use prompt terminology.
    """

    id: str
    model_id: str
    contract: str

    def __init__(
        self,
        id: str,
        model_id: str,
        contract: Optional[str] = None,
        *,
        prompt: Optional[str] = None,
    ) -> None:
        if not isinstance(id, str) or not isinstance(model_id, str):
            raise TypeError("AgentNode id and model_id must be strings")
        if contract is not None and prompt is not None and contract != prompt:
            raise ValueError("contract and prompt aliases disagree")
        resolved_contract = contract if contract is not None else prompt
        if resolved_contract is None:
            raise ValueError("AgentNode requires contract or prompt")
        if not isinstance(resolved_contract, str):
            raise TypeError("AgentNode contract must be a string")
        object.__setattr__(self, "id", id.strip())
        object.__setattr__(self, "model_id", model_id.strip())
        object.__setattr__(self, "contract", resolved_contract.strip())

    @property
    def prompt(self) -> str:
        return self.contract

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "model_id": self.model_id, "contract": self.contract}


@dataclass(frozen=True, slots=True)
class RelationBits:
    """Two directed channel bits relative to a requested endpoint ordering."""

    source_to_target: bool
    target_to_source: bool

    def __post_init__(self) -> None:
        if type(self.source_to_target) is not bool or type(self.target_to_source) is not bool:
            raise TypeError("relation bits must be bool values")

    @property
    def is_independent(self) -> bool:
        return not self.source_to_target and not self.target_to_source

    @property
    def is_bidirectional(self) -> bool:
        return self.source_to_target and self.target_to_source


@dataclass(frozen=True, slots=True)
class AgentRelation:
    """A two-bit relationship stored relative to ``source_id``/``target_id``."""

    source_id: str
    target_id: str
    source_to_target: bool
    target_to_source: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not isinstance(self.target_id, str):
            raise TypeError("relation endpoints must be strings")
        if type(self.source_to_target) is not bool or type(self.target_to_source) is not bool:
            raise TypeError("relation bits must be bool values")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "target_id", self.target_id.strip())

    @property
    def bits(self) -> RelationBits:
        return RelationBits(self.source_to_target, self.target_to_source)

    @property
    def unordered_key(self) -> Tuple[str, str]:
        return tuple(sorted((self.source_id, self.target_id)))  # type: ignore[return-value]

    def oriented(self, source_id: str, target_id: str) -> RelationBits:
        if (source_id, target_id) == (self.source_id, self.target_id):
            return self.bits
        if (source_id, target_id) == (self.target_id, self.source_id):
            return RelationBits(self.target_to_source, self.source_to_target)
        raise KeyError(f"relation does not connect {source_id!r} and {target_id!r}")

    def canonical(self) -> "AgentRelation":
        first, second = self.unordered_key
        bits = self.oriented(first, second)
        return AgentRelation(first, second, bits.source_to_target, bits.target_to_source)

    def directed_edges(self) -> Tuple[Tuple[str, str], ...]:
        result: List[Tuple[str, str]] = []
        if self.source_to_target:
            result.append((self.source_id, self.target_id))
        if self.target_to_source:
            result.append((self.target_id, self.source_id))
        return tuple(result)

    def to_dict(self) -> Dict[str, object]:
        relation = self.canonical()
        return {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "source_to_target": relation.source_to_target,
            "target_to_source": relation.target_to_source,
        }


@dataclass(frozen=True, slots=True)
class AgentGraphSnapshot:
    """Immutable, content-addressed AgentGraph state."""

    nodes: Tuple[AgentNode, ...]
    relations: Tuple[AgentRelation, ...]
    output_agent_id: Optional[str]
    revision: int

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("snapshot revision cannot be negative")

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
            "output_agent_id": self.output_agent_id,
            "revision": self.revision,
        }

    @property
    def snapshot_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphValidationIssue:
    code: str
    message: str
    agent_ids: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphValidationResult:
    issues: Tuple[GraphValidationIssue, ...]
    components: Tuple[Tuple[str, ...], ...]
    topological_blocks: Tuple[Tuple[str, ...], ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise AgentGraphValidationError(self)


class AgentGraphValidationError(ValueError):
    """Raised when execution is requested for an invalid AgentGraph."""

    def __init__(self, result: GraphValidationResult) -> None:
        self.result = result
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in result.issues)
        super().__init__(detail or "invalid AgentGraph")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


ModelCatalog = Union[ModelRegistry, Collection[str]]


class AgentGraphValidator:
    """Checks executable AgentGraph invariants.

    Partial validation enforces mutation safety (the first six invariants) but
    defers output/reachability rules until a graph is finished or executed.
    """

    def __init__(self, model_catalog: Optional[ModelCatalog] = None) -> None:
        self._model_catalog = model_catalog

    def validate(self, graph: "AgentGraph", *, require_complete: bool = True) -> GraphValidationResult:
        issues: List[GraphValidationIssue] = []
        node_ids = [node.id for node in graph.nodes]
        id_counts: Dict[str, int] = {}
        for agent_id in node_ids:
            id_counts[agent_id] = id_counts.get(agent_id, 0) + 1
        for agent_id, count in sorted(id_counts.items()):
            if count > 1:
                issues.append(
                    GraphValidationIssue(
                        "duplicate_agent_id",
                        f"agent ID {agent_id!r} occurs {count} times",
                        (agent_id,),
                    )
                )

        unique_ids = set(node_ids)
        for node in graph.nodes:
            if not node.id:
                issues.append(GraphValidationIssue("empty_agent_id", "agent ID is empty"))
            if not node.contract:
                issues.append(
                    GraphValidationIssue(
                        "empty_contract",
                        f"agent {node.id!r} has an empty contract",
                        (node.id,),
                    )
                )
            if not node.model_id:
                issues.append(
                    GraphValidationIssue(
                        "unknown_model_id",
                        f"agent {node.id!r} has an empty model ID",
                        (node.id,),
                    )
                )
            elif self._model_catalog is not None and node.model_id not in self._model_catalog:
                issues.append(
                    GraphValidationIssue(
                        "unknown_model_id",
                        f"agent {node.id!r} references unknown model {node.model_id!r}",
                        (node.id,),
                    )
                )

        pair_counts: Dict[Tuple[str, str], int] = {}
        for relation in graph.relations:
            pair_counts[relation.unordered_key] = pair_counts.get(relation.unordered_key, 0) + 1
            if relation.source_id == relation.target_id:
                issues.append(
                    GraphValidationIssue(
                        "self_relation",
                        f"self relation on agent {relation.source_id!r} is not allowed",
                        (relation.source_id,),
                    )
                )
            missing = tuple(
                endpoint
                for endpoint in (relation.source_id, relation.target_id)
                if endpoint not in unique_ids
            )
            if missing:
                issues.append(
                    GraphValidationIssue(
                        "unknown_relation_endpoint",
                        f"relation references missing agents: {', '.join(missing)}",
                        missing,
                    )
                )
            if relation.bits.is_independent:
                issues.append(
                    GraphValidationIssue(
                        "empty_relation",
                        "independent relations must be represented by absence",
                        relation.unordered_key,
                    )
                )
        for pair, count in sorted(pair_counts.items()):
            if count > 1:
                issues.append(
                    GraphValidationIssue(
                        "duplicate_relation",
                        f"agent pair {pair!r} has {count} relation records",
                        pair,
                    )
                )

        union_find = _UnionFind(unique_ids)
        for relation in graph.relations:
            if (
                relation.bits.is_bidirectional
                and relation.source_id in unique_ids
                and relation.target_id in unique_ids
                and relation.source_id != relation.target_id
            ):
                union_find.union(relation.source_id, relation.target_id)

        members: Dict[str, List[str]] = {}
        for agent_id in sorted(unique_ids):
            members.setdefault(union_find.find(agent_id), []).append(agent_id)
        components = tuple(sorted(tuple(values) for values in members.values()))
        for component in components:
            if len(component) > 2:
                issues.append(
                    GraphValidationIssue(
                        "bidirectional_block_too_large",
                        f"bidirectional block has {len(component)} agents; maximum is 2",
                        component,
                    )
                )

        component_for: Dict[str, Tuple[str, ...]] = {
            agent_id: component for component in components for agent_id in component
        }
        adjacency: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {
            component: set() for component in components
        }
        indegree: Dict[Tuple[str, ...], int] = {component: 0 for component in components}
        for relation in graph.relations:
            for source_id, target_id in relation.directed_edges():
                if source_id not in component_for or target_id not in component_for:
                    continue
                source_component = component_for[source_id]
                target_component = component_for[target_id]
                if source_component == target_component or target_component in adjacency[source_component]:
                    continue
                adjacency[source_component].add(target_component)
                indegree[target_component] += 1

        ready = sorted(component for component, degree in indegree.items() if degree == 0)
        topological: List[Tuple[str, ...]] = []
        while ready:
            component = ready.pop(0)
            topological.append(component)
            for target in sorted(adjacency[component]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(topological) != len(components):
            cyclic = tuple(sorted(agent_id for component in components for agent_id in component))
            issues.append(
                GraphValidationIssue(
                    "quotient_cycle",
                    "contracting bidirectional blocks leaves a directed cycle",
                    cyclic,
                )
            )

        if require_complete:
            output_id = graph.output_agent_id
            if output_id is None:
                issues.append(
                    GraphValidationIssue("output_agent_count", "exactly one output agent is required")
                )
            elif output_id not in unique_ids:
                issues.append(
                    GraphValidationIssue(
                        "unknown_output_agent",
                        f"output agent {output_id!r} does not exist",
                        (output_id,),
                    )
                )
            else:
                output_component = component_for[output_id]
                if adjacency[output_component]:
                    issues.append(
                        GraphValidationIssue(
                            "output_not_sink",
                            "the output agent block must be a quotient-graph sink",
                            output_component,
                        )
                    )

                reverse: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {
                    component: set() for component in components
                }
                for source, targets in adjacency.items():
                    for target in targets:
                        reverse[target].add(source)
                reaches_output: Set[Tuple[str, ...]] = {output_component}
                frontier = [output_component]
                while frontier:
                    current = frontier.pop()
                    for predecessor in reverse[current]:
                        if predecessor not in reaches_output:
                            reaches_output.add(predecessor)
                            frontier.append(predecessor)
                unreachable = tuple(
                    sorted(
                        agent_id
                        for component in components
                        if component not in reaches_output
                        for agent_id in component
                    )
                )
                if unreachable:
                    issues.append(
                        GraphValidationIssue(
                            "cannot_reach_output",
                            "every agent block must be able to reach the output block",
                            unreachable,
                        )
                    )

        return GraphValidationResult(tuple(issues), components, tuple(topological))


class AgentGraph:
    """Mutable canvas with immutable snapshots and monotonic revisions."""

    def __init__(
        self,
        nodes: Iterable[AgentNode] = (),
        relations: Iterable[AgentRelation] = (),
        output_agent_id: Optional[str] = None,
        revision: int = 0,
    ) -> None:
        if revision < 0:
            raise ValueError("revision cannot be negative")
        self._nodes: List[AgentNode] = list(nodes)
        self._relations: List[AgentRelation] = list(relations)
        self._output_agent_id = output_agent_id
        self._revision = revision

    @property
    def nodes(self) -> Tuple[AgentNode, ...]:
        return tuple(self._nodes)

    @property
    def relations(self) -> Tuple[AgentRelation, ...]:
        return tuple(self._relations)

    @property
    def output_agent_id(self) -> Optional[str]:
        return self._output_agent_id

    @property
    def revision(self) -> int:
        return self._revision

    def get_node(self, agent_id: str) -> AgentNode:
        matches = [node for node in self._nodes if node.id == agent_id]
        if len(matches) != 1:
            if not matches:
                raise GraphMutationError(f"unknown agent_id: {agent_id}")
            raise GraphMutationError(f"agent_id is not unique: {agent_id}")
        return matches[0]

    def add_agent(self, node: AgentNode) -> None:
        if any(existing.id == node.id for existing in self._nodes):
            raise GraphMutationError(f"duplicate agent_id: {node.id}")
        self._nodes.append(node)
        self._revision += 1

    def modify_agent(
        self,
        agent_id: str,
        *,
        model_id: Optional[str] = None,
        contract: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        current = self.get_node(agent_id)
        if contract is not None and prompt is not None and contract != prompt:
            raise GraphMutationError("contract and prompt aliases disagree")
        resolved_contract = contract if contract is not None else prompt
        replacement = AgentNode(
            id=current.id,
            model_id=current.model_id if model_id is None else model_id,
            contract=current.contract if resolved_contract is None else resolved_contract,
        )
        if replacement == current:
            return
        index = self._nodes.index(current)
        self._nodes[index] = replacement
        self._revision += 1

    def delete_agent(self, agent_id: str) -> None:
        node = self.get_node(agent_id)
        self._nodes.remove(node)
        self._relations = [
            relation
            for relation in self._relations
            if agent_id not in (relation.source_id, relation.target_id)
        ]
        if self._output_agent_id == agent_id:
            self._output_agent_id = None
        self._revision += 1

    def relation_bits(self, source_id: str, target_id: str) -> RelationBits:
        for relation in self._relations:
            if relation.unordered_key == tuple(sorted((source_id, target_id))):
                return relation.oriented(source_id, target_id)
        return RelationBits(False, False)

    def set_relation(
        self,
        source_id: str,
        target_id: str,
        source_to_target: bool,
        target_to_source: bool,
    ) -> None:
        if type(source_to_target) is not bool or type(target_to_source) is not bool:
            raise GraphMutationError("relation bits must be bool values")
        self.get_node(source_id)
        self.get_node(target_id)
        if source_id == target_id:
            raise GraphMutationError("self relations are not allowed")
        pair = tuple(sorted((source_id, target_id)))
        existing = [relation for relation in self._relations if relation.unordered_key == pair]
        if len(existing) > 1:
            raise GraphMutationError(f"duplicate relation records for pair {pair!r}")
        previous = self.relation_bits(source_id, target_id)
        requested = RelationBits(source_to_target, target_to_source)
        if previous == requested:
            return
        self._relations = [relation for relation in self._relations if relation.unordered_key != pair]
        if not requested.is_independent:
            self._relations.append(
                AgentRelation(source_id, target_id, source_to_target, target_to_source).canonical()
            )
        self._revision += 1

    def set_output(self, agent_id: str) -> None:
        self.get_node(agent_id)
        if self._output_agent_id == agent_id:
            return
        self._output_agent_id = agent_id
        self._revision += 1

    def validate(
        self,
        model_catalog: Optional[ModelCatalog] = None,
        *,
        require_complete: bool = True,
    ) -> GraphValidationResult:
        return AgentGraphValidator(model_catalog).validate(self, require_complete=require_complete)

    def _quotient_structure(
        self,
    ) -> Tuple[
        GraphValidationResult,
        Dict[str, Tuple[str, ...]],
        Dict[Tuple[str, ...], Set[Tuple[str, ...]]],
        Dict[Tuple[str, ...], Set[Tuple[str, ...]]],
    ]:
        """Build the reciprocal-contracted graph used by validation and diagnostics."""

        validation = self.validate(require_complete=False)
        component_for = {
            agent_id: component
            for component in validation.components
            for agent_id in component
        }
        predecessors: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {
            component: set() for component in validation.components
        }
        successors: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {
            component: set() for component in validation.components
        }
        for relation in self._relations:
            for source_id, target_id in relation.directed_edges():
                if source_id not in component_for or target_id not in component_for:
                    continue
                source_component = component_for[source_id]
                target_component = component_for[target_id]
                if source_component == target_component:
                    continue
                successors[source_component].add(target_component)
                predecessors[target_component].add(source_component)
        return validation, component_for, predecessors, successors

    def topology_statistics(self) -> Dict[str, object]:
        """Return read-only DAG shape facts for Canvas feedback and analysis.

        This is the AgentGraph analogue of FlowSteer's
        ``WorkflowGraph.get_statistics``.  It reports only observed structure;
        no shape is rewarded or required.
        """

        validation, component_for, component_predecessors, component_successors = (
            self._quotient_structure()
        )
        agent_in_degree = {node.id: 0 for node in self._nodes}
        agent_out_degree = {node.id: 0 for node in self._nodes}
        for relation in self._relations:
            for source_id, target_id in relation.directed_edges():
                if source_id not in component_for or target_id not in component_for:
                    continue
                agent_out_degree[source_id] += 1
                agent_in_degree[target_id] += 1
        depth_by_component: Dict[Tuple[str, ...], int] = {}
        width_by_depth: Dict[int, int] = {}
        for component in validation.topological_blocks:
            predecessors = component_predecessors[component]
            depth = (
                1
                if not predecessors
                else 1 + max(depth_by_component[item] for item in predecessors)
            )
            depth_by_component[component] = depth
            width_by_depth[depth] = width_by_depth.get(depth, 0) + 1

        roots = sorted(
            agent_id for agent_id, degree in agent_in_degree.items() if degree == 0
        )
        sinks = sorted(
            agent_id for agent_id, degree in agent_out_degree.items() if degree == 0
        )
        quotient_edge_count = sum(len(targets) for targets in component_successors.values())
        component_count = len(validation.components)
        reciprocal_pair_count = sum(
            relation.bits.is_bidirectional for relation in self._relations
        )
        structural_depth = max(depth_by_component.values(), default=0)
        fan_in = any(len(items) > 1 for items in component_predecessors.values())
        fan_out = any(len(items) > 1 for items in component_successors.values())
        simple_serial = (
            component_count > 1
            and quotient_edge_count == component_count - 1
            and all(len(items) <= 1 for items in component_predecessors.values())
            and all(len(items) <= 1 for items in component_successors.values())
        )

        motifs: List[str] = []
        if simple_serial:
            motifs.append("serial_2" if structural_depth == 2 else "serial_3_plus")
        elif max(width_by_depth.values(), default=0) > 1:
            motifs.append("parallel")
        if fan_in:
            motifs.append("fan_in")
        if fan_out:
            motifs.append("fan_out")
        if reciprocal_pair_count:
            motifs.append("reciprocal")

        if not self._nodes:
            topology_family = "empty"
        elif len(self._nodes) == 1:
            topology_family = "single"
        elif reciprocal_pair_count and component_count == 1:
            topology_family = "reciprocal"
        elif simple_serial:
            topology_family = motifs[0]
        elif fan_in and fan_out:
            topology_family = "mixed"
        elif fan_in:
            topology_family = "fan_in"
        elif fan_out:
            topology_family = "fan_out"
        elif "parallel" in motifs:
            topology_family = "parallel"
        else:
            topology_family = "mixed"

        return {
            "agent_count": len(self._nodes),
            "relation_count": len(self._relations),
            "directed_edge_count": sum(agent_out_degree.values()),
            "quotient_directed_edge_count": quotient_edge_count,
            "reciprocal_pair_count": reciprocal_pair_count,
            "component_count": component_count,
            # ``max_depth`` remains for receipt compatibility.  The explicit
            # name documents that finite reciprocal blocks count as one node.
            "max_depth": structural_depth,
            "structural_depth": structural_depth,
            "max_width": max(width_by_depth.values(), default=0),
            "topology_family": topology_family,
            "topology_motifs": motifs,
            "root_agent_ids": roots,
            "sink_agent_ids": sinks,
            "root_component_count": sum(
                not items for items in component_predecessors.values()
            ),
            "sink_component_count": sum(
                not items for items in component_successors.values()
            ),
            "fan_in_agent_ids": sorted(
                agent_id for agent_id, degree in agent_in_degree.items() if degree > 1
            ),
            "fan_out_agent_ids": sorted(
                agent_id for agent_id, degree in agent_out_degree.items() if degree > 1
            ),
            "output_agent_id": self._output_agent_id,
        }

    def construction_progress(self) -> Dict[str, object]:
        """Return a neutral lower bound for finishing through atomic edits.

        The count preserves every current node and relation.  It may choose a
        quotient sink as Output and connect other quotient sinks to it, then
        uses the existing explicit FINISH action.  It is state feedback only:
        no topology, role, Agent count, or edit is recommended.
        """

        validation, component_for, _, successors = self._quotient_structure()
        if not self._nodes:
            add_agent_actions = 1
            relation_actions = 0
            output_actions = 1
        else:
            add_agent_actions = 0
            sink_count = sum(not targets for targets in successors.values())
            relation_actions = max(sink_count - 1, 0)
            output_component = component_for.get(self._output_agent_id or "")
            output_actions = int(
                output_component is None or bool(successors.get(output_component))
            )
        finish_actions = 1
        minimum = (
            add_agent_actions + relation_actions + output_actions + finish_actions
        )
        return {
            "atomic_edits_applied": self._revision,
            "structurally_finishable_now": self.validate(
                require_complete=True
            ).valid,
            "minimum_remaining_actions": minimum,
            "minimum_remaining_breakdown": {
                "add_agent": add_agent_actions,
                "set_relation": relation_actions,
                "set_output": output_actions,
                "finish": finish_actions,
            },
            "partial_graph_valid": validation.valid,
        }

    def effective_dependency_statistics(
        self,
        evidence: Iterable[DependencyEdgeEvidence] = (),
    ) -> Dict[str, object]:
        """Conservatively aggregate explicit dependency evidence over the DAG.

        Structural edges default to ``unverified``.  This method never reads
        answer text or infers causality from a masked-output difference; only
        evidence grades explicitly supplied by the diagnostic caller can
        increase the reported effective depth.
        """

        validation, component_for, predecessors, _ = self._quotient_structure()
        rank = {"unverified": 0, "weak": 1, "verified": 2}
        directed_edges = {
            edge
            for relation in self._relations
            for edge in relation.directed_edges()
        }
        edge_status = {edge: "unverified" for edge in directed_edges}
        evidence_ids: Dict[Tuple[str, str], List[str]] = {
            edge: [] for edge in directed_edges
        }
        for item in evidence:
            if not isinstance(item, DependencyEdgeEvidence):
                raise TypeError("dependency evidence items must be DependencyEdgeEvidence")
            edge = (item.source_id, item.target_id)
            if edge not in directed_edges:
                raise ValueError(f"dependency evidence references absent edge: {edge!r}")
            if rank[item.status] > rank[edge_status[edge]]:
                edge_status[edge] = item.status
            if item.evidence_id is not None:
                evidence_ids[edge].append(item.evidence_id)

        component_edge_status: Dict[
            Tuple[Tuple[str, ...], Tuple[str, ...]], str
        ] = {}
        for edge, status in edge_status.items():
            source_component = component_for[edge[0]]
            target_component = component_for[edge[1]]
            if source_component == target_component:
                continue
            key = (source_component, target_component)
            previous = component_edge_status.get(key, "unverified")
            if rank[status] > rank[previous]:
                component_edge_status[key] = status

        def longest_depth(minimum_rank: int) -> int:
            depth: Dict[Tuple[str, ...], int] = {}
            for component in validation.topological_blocks:
                supported = [
                    predecessor
                    for predecessor in predecessors[component]
                    if rank[
                        component_edge_status.get(
                            (predecessor, component), "unverified"
                        )
                    ]
                    >= minimum_rank
                ]
                depth[component] = 1 + max(
                    (depth[item] for item in supported), default=0
                )
            return max(depth.values(), default=0)

        structural_depth = int(self.topology_statistics()["structural_depth"])
        verified_depth = longest_depth(rank["verified"])
        weak_or_verified_depth = longest_depth(rank["weak"])
        if not component_edge_status:
            status = "not_applicable" if structural_depth <= 1 else "unverified"
        elif verified_depth == weak_or_verified_depth and verified_depth > 1:
            status = "verified"
        elif weak_or_verified_depth > 1:
            status = "weak"
        else:
            status = "unverified"

        if structural_depth <= 1:
            full_depth_status = "not_applicable"
        elif verified_depth >= structural_depth:
            full_depth_status = "verified"
        elif weak_or_verified_depth >= structural_depth:
            full_depth_status = "weak"
        else:
            full_depth_status = "unverified"

        status_counts = {name: 0 for name in sorted(DEPENDENCY_EVIDENCE_STATUSES)}
        for value in edge_status.values():
            status_counts[value] += 1
        return {
            "structural_depth": structural_depth,
            "effective_dependency_depth": weak_or_verified_depth,
            "verified_dependency_depth": verified_depth,
            "evidence_status": status,
            "full_structural_depth_evidence_status": full_depth_status,
            "directed_edge_status_counts": status_counts,
            "directed_edge_evidence": [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "status": edge_status[(source_id, target_id)],
                    "evidence_ids": sorted(evidence_ids[(source_id, target_id)]),
                }
                for source_id, target_id in sorted(directed_edges)
            ],
        }

    def snapshot(self) -> AgentGraphSnapshot:
        nodes = tuple(sorted(self._nodes, key=lambda node: (node.id, node.model_id, node.contract)))
        relations = tuple(
            sorted(
                (relation.canonical() for relation in self._relations),
                key=lambda relation: (relation.source_id, relation.target_id),
            )
        )
        return AgentGraphSnapshot(nodes, relations, self._output_agent_id, self._revision)

    def fork(self) -> "AgentGraph":
        return AgentGraph.from_snapshot(self.snapshot())

    @classmethod
    def from_snapshot(cls, snapshot: AgentGraphSnapshot) -> "AgentGraph":
        return cls(
            nodes=snapshot.nodes,
            relations=snapshot.relations,
            output_agent_id=snapshot.output_agent_id,
            revision=snapshot.revision,
        )

    def to_dict(self) -> Dict[str, object]:
        return self.snapshot().to_dict()


__all__ = [
    "AgentGraph",
    "AgentGraphSnapshot",
    "AgentGraphValidationError",
    "AgentGraphValidator",
    "AgentNode",
    "AgentRelation",
    "GraphMutationError",
    "GraphValidationIssue",
    "GraphValidationResult",
    "RelationBits",
]
