"""Frozen dynamic-ledger observations for natural HotpotQA rollouts.

This module is the narrow adapter between ``AgentGraphRolloutCollector``'s
``RolloutObservationHook`` protocol and the epoch-scoped combination ledger.
It observes accepted Canvas edits and execution receipts; it never changes an
``AgentAction``, an evaluator result, a terminal reward, or the policy update.

Free Agent contracts are mapped to the fixed ledger role catalog exclusively
through the injected :class:`RoleClassifier`.  A missing or ambiguous graph
binding therefore produces an explicit unclassifiable sidecar instead of a
keyword/default-role fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..agent_action_parser import (
    AgentAction,
    AgentActionParseError,
    AgentActionParser,
    AgentActionType,
)
from ..agent_graph import AgentGraphSnapshot, AgentNode, AgentRelation
from ..agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowSnapshot,
    AgentWorkflowStepResult,
)
from ..director import DirectorResponse
from ..records import TaskRecord, TrajectoryRecord
from .combination_ledger import DecisionKey, SurfaceSignal
from .latent_loss import (
    LatentLossConfig,
    PostExecutionReadout,
    changed_field,
    single_field_candidates,
)
from .ledger_epoch import LedgerEpoch, StepRecord, combine_step_readouts
from .role_classifier import (
    ROLE_CLUSTERS,
    RoleClassification,
    RoleClassificationError,
    RoleClassifier,
)


ROLLOUT_HOOK_SCHEMA_VERSION = "hotpotqa-dynamic-ledger-hook-v1"
TASK_FAMILY = "hotpotqa"
_EDGE_TYPES = ("independent", "unidirectional", "bidirectional")


def _frozen_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _mapping_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Detach only the shallow JSON-like sidecar boundary."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class TrajectoryLedgerSidecar:
    """Ledger-only records registered after a rollout is finalized.

    Terminal evaluator fields are intentionally absent.  The epoch closer is
    responsible for pairing these observations with a binary terminal result.
    """

    trajectory_id: str
    task_id: str
    condition_id: str
    policy_version: str
    steps: tuple[StepRecord, ...]
    events: tuple[Mapping[str, Any], ...]
    probe_sites: tuple["ProbeSite", ...] = ()
    schema_version: str = ROLLOUT_HOOK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "task_id", "condition_id", "policy_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if any(step.trajectory_id != self.trajectory_id for step in self.steps):
            raise ValueError("all StepRecords must belong to the sidecar trajectory")
        if any(site.trajectory_id != self.trajectory_id for site in self.probe_sites):
            raise ValueError("all ProbeSites must belong to the sidecar trajectory")
        object.__setattr__(
            self,
            "events",
            tuple(_mapping_copy(event) for event in self.events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "condition_id": self.condition_id,
            "policy_version": self.policy_version,
            "steps": [step.to_dict() for step in self.steps],
            "events": [dict(event) for event in self.events],
            "probe_sites": [site.to_dict() for site in self.probe_sites],
        }


@dataclass(frozen=True)
class ProbeSite:
    """Read-only same-snapshot input for the paired-intervention runner."""

    trajectory_id: str
    step_id: int
    pre_snapshot: AgentWorkflowSnapshot
    actual_action: AgentAction
    current_key: DecisionKey
    candidate_key: DecisionKey
    risk_probability: float | None
    straddle_score: float
    warning: bool

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or self.step_id < 0:
            raise ValueError("ProbeSite identity is invalid")
        if not isinstance(self.pre_snapshot, AgentWorkflowSnapshot):
            raise TypeError("pre_snapshot must be AgentWorkflowSnapshot")
        if not isinstance(self.actual_action, AgentAction):
            raise TypeError("actual_action must be AgentAction")
        changed_field(self.current_key, self.candidate_key)
        if self.risk_probability is not None and not 0.0 <= self.risk_probability <= 1.0:
            raise ValueError("risk_probability must lie in [0, 1]")
        if self.straddle_score < 0.0:
            raise ValueError("straddle_score must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "step_id": self.step_id,
            "pre_snapshot_id": self.pre_snapshot.snapshot_id,
            "actual_action": self.actual_action.to_dict(),
            "current_key": self.current_key.to_dict(),
            "candidate_key": self.candidate_key.to_dict(),
            "risk_probability": self.risk_probability,
            "straddle_score": self.straddle_score,
            "warning": self.warning,
        }


@dataclass
class _TrajectoryState:
    task_id: str
    prefix: list[DecisionKey] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    events: list[Mapping[str, Any]] = field(default_factory=list)
    probe_sites: list[ProbeSite] = field(default_factory=list)
    last_key: DecisionKey | None = None
    last_focal_agent_id: str | None = None
    last_post: PostExecutionReadout | None = None
    warning_presented: bool = False


@dataclass(frozen=True)
class _KeyResolution:
    key: DecisionKey
    focal_agent_id: str
    upstream_agent_id: str | None
    classification: RoleClassification


class _Unclassifiable(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        classification_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.classification_receipt = classification_receipt


class HotpotQADynamicLedgerHook:
    """Produce advisory ledger/Skill observations and immutable step sidecars.

    ``model_catalog`` is copied at construction and is the only model domain
    used to enumerate alternatives.  ``LedgerEpoch`` similarly freezes the
    posterior and ACTIVE Skill snapshot.  Consequently, every trajectory in a
    condition sees the same search-space support even if the live registry is
    modified elsewhere.
    """

    def __init__(
        self,
        *,
        epoch: LedgerEpoch,
        role_classifier: RoleClassifier,
        model_catalog: Sequence[str],
        max_rounds: int,
        latent_config: LatentLossConfig = LatentLossConfig(),
        random_seed: int = 0,
    ) -> None:
        if not isinstance(epoch, LedgerEpoch):
            raise TypeError("epoch must be LedgerEpoch")
        if not isinstance(role_classifier, RoleClassifier):
            raise TypeError("role_classifier must be RoleClassifier")
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not isinstance(latent_config, LatentLossConfig):
            raise TypeError("latent_config must be LatentLossConfig")

        self.epoch = epoch
        self.role_classifier = role_classifier
        self.model_catalog = _frozen_strings(model_catalog, "model_catalog")
        self.max_rounds = max_rounds
        self.latent_config = latent_config
        self.random_seed = random_seed
        self._parser = AgentActionParser()
        self._states: dict[str, _TrajectoryState] = {}
        self._completed: dict[str, TrajectoryLedgerSidecar] = {}
        self._role_cache: dict[str, RoleClassification] = {}

    @property
    def completed_sidecars(self) -> Mapping[str, TrajectoryLedgerSidecar]:
        return MappingProxyType(dict(self._completed))

    @property
    def probe_sites(self) -> Mapping[str, tuple[ProbeSite, ...]]:
        result = {
            trajectory_id: tuple(state.probe_sites)
            for trajectory_id, state in self._states.items()
            if state.probe_sites
        }
        return MappingProxyType(result)

    def sidecar(self, trajectory_id: str) -> TrajectoryLedgerSidecar:
        try:
            return self._completed[trajectory_id]
        except KeyError as error:
            raise KeyError(f"no completed ledger sidecar for {trajectory_id!r}") from error

    async def before_turn(
        self,
        *,
        task: TaskRecord,
        trajectory_id: str,
        round_index: int,
        environment: AgentWorkflowEnv,
    ) -> Mapping[str, Any]:
        state = self._state(task, trajectory_id)
        try:
            resolution = await self._reference_resolution(
                environment.snapshot(),
                state,
                round_index=round_index,
            )
        except _Unclassifiable as error:
            state.warning_presented = False
            return self._unclassifiable_observation(error.reason)

        current = resolution.key
        candidates = self._ranked_candidates(current, step_id=round_index)
        skills = self.epoch.select_skill_texts(
            task_family=TASK_FAMILY,
            role_cluster=current.role_cluster,
            stage=current.stage,
            available_models=self.model_catalog,
        )
        warning_text = ""
        if state.last_post is not None and state.last_post.warning:
            warning_text = state.last_post.text
        state.warning_presented = bool(warning_text)
        return {
            "schema_version": ROLLOUT_HOOK_SCHEMA_VERSION,
            "condition_id": self.epoch.condition.condition_id,
            "status": "available",
            "reference_decision": current.to_dict(),
            "candidates": [
                {
                    "field": candidate.changed_field,
                    "value": candidate.changed_value,
                    "expected_gain": candidate.mean,
                    "lower": candidate.lower,
                    "upper": candidate.upper,
                    "unknown": candidate.unknown,
                }
                for candidate in candidates.candidates[:5]
            ],
            "warning": warning_text,
            "active_skills": list(skills.texts[:3]),
            "active_skill_ids": list(skills.skill_ids[:3]),
            "advisory_only": True,
        }

    async def after_turn(
        self,
        *,
        task: TaskRecord,
        trajectory_id: str,
        round_index: int,
        pre_snapshot: Any,
        response: DirectorResponse,
        canvas: Any,
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state = self._state(task, trajectory_id)
        if not isinstance(pre_snapshot, AgentWorkflowSnapshot):
            return self._record_event(
                state,
                trajectory_id,
                round_index,
                "missing_pre_canvas_snapshot",
            )
        snapshot_id = pre_snapshot.snapshot_id
        try:
            parsed = self._parser.parse(response.text)
        except (AgentActionParseError, TypeError, ValueError) as error:
            return self._record_event(
                state,
                trajectory_id,
                round_index,
                "action_parse_failed",
                detail=type(error).__name__,
                pre_canvas_snapshot_id=snapshot_id,
            )
        actual = getattr(canvas, "action", None)
        if not isinstance(actual, AgentAction) or actual.to_dict() != parsed.to_dict():
            return self._record_event(
                state,
                trajectory_id,
                round_index,
                "action_receipt_mismatch",
                pre_canvas_snapshot_id=snapshot_id,
            )
        if getattr(canvas, "accepted", None) is not True:
            return self._record_event(
                state,
                trajectory_id,
                round_index,
                "canvas_rejected",
                action=actual.to_dict(),
                pre_canvas_snapshot_id=snapshot_id,
            )

        try:
            resolution = await self._resolve_action(
                actual,
                canvas,
                round_index=round_index,
            )
        except _Unclassifiable as error:
            return self._record_event(
                state,
                trajectory_id,
                round_index,
                error.reason,
                action=actual.to_dict(),
                pre_canvas_snapshot_id=snapshot_id,
                classification_receipt=error.classification_receipt,
            )

        current = resolution.key
        pre = self._ranked_candidates(
            current,
            step_id=round_index,
            candidate_domain=self._translatable_candidates(actual, current),
        )
        candidate_keys = tuple(candidate.key for candidate in pre.candidates[:5])
        skills = self.epoch.select_skill_texts(
            task_family=TASK_FAMILY,
            role_cluster=current.role_cluster,
            stage=current.stage,
            available_models=self.model_catalog,
        )
        surface = self._surface_signal(canvas, resolution)
        remaining_rounds = max(self.max_rounds - round_index - 1, 0)
        post = self.epoch.post_execution(
            step_id=round_index,
            prefix_before=tuple(state.prefix),
            current=current,
            candidates=candidate_keys,
            surface=surface,
            remaining_rounds=remaining_rounds,
            rng=np.random.default_rng(self.random_seed + round_index),
            config=self.latent_config,
        )
        director_action = self._response_to_prior_readout(state, current)
        readout = combine_step_readouts(pre, post, skills)
        readout["presented_observation"] = dict(observation)
        step = StepRecord(
            step_id=round_index,
            trajectory_id=trajectory_id,
            snapshot_id=snapshot_id,
            key=current,
            candidates=candidate_keys,
            ledger_readout=readout,
            director_action=director_action,
            surface=surface,
        )
        state.steps.append(step)
        state.prefix.append(current)
        state.last_key = current
        state.last_focal_agent_id = resolution.focal_agent_id
        state.last_post = post
        state.warning_presented = False
        if post.best_alternative is not None:
            state.probe_sites.append(
                ProbeSite(
                    trajectory_id=trajectory_id,
                    step_id=round_index,
                    pre_snapshot=pre_snapshot,
                    actual_action=actual,
                    current_key=current,
                    candidate_key=post.best_alternative.key,
                    risk_probability=post.risk_probability,
                    straddle_score=post.straddle_score,
                    warning=post.warning,
                )
            )
        return {
            "schema_version": ROLLOUT_HOOK_SCHEMA_VERSION,
            "status": "recorded",
            "step_record": step.to_dict(),
            "role_classification": resolution.classification.receipt.to_dict(),
            "focal_agent_id": resolution.focal_agent_id,
            "upstream_agent_id": resolution.upstream_agent_id,
            "pre_canvas_snapshot_id": snapshot_id,
        }

    def after_trajectory(self, *, trajectory: TrajectoryRecord) -> None:
        state = self._states.get(trajectory.trajectory_id)
        if state is None:
            raise ValueError("trajectory has no dynamic-ledger hook state")
        if trajectory.trajectory_id in self._completed:
            raise ValueError("trajectory ledger sidecar is already registered")
        if trajectory.task.task_id != state.task_id:
            raise ValueError("trajectory task differs from ledger sidecar task")
        if trajectory.condition_id != self.epoch.condition.condition_id:
            raise ValueError("trajectory condition differs from frozen ledger epoch")
        if trajectory.versions.policy != self.epoch.condition.versions.policy:
            raise ValueError("trajectory policy differs from frozen ledger epoch")
        # Deliberately do not inspect trajectory.evaluation or update the
        # posterior.  The main epoch coordinator owns terminal-outcome routing.
        self._completed[trajectory.trajectory_id] = TrajectoryLedgerSidecar(
            trajectory_id=trajectory.trajectory_id,
            task_id=state.task_id,
            condition_id=trajectory.condition_id,
            policy_version=trajectory.versions.policy,
            steps=tuple(state.steps),
            events=tuple(state.events),
            probe_sites=tuple(state.probe_sites),
        )

    def _state(self, task: TaskRecord, trajectory_id: str) -> _TrajectoryState:
        if not isinstance(task, TaskRecord):
            raise TypeError("task must be TaskRecord")
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            raise ValueError("trajectory_id must be non-empty")
        state = self._states.get(trajectory_id)
        if state is None:
            state = _TrajectoryState(task_id=task.task_id)
            self._states[trajectory_id] = state
        elif state.task_id != task.task_id:
            raise ValueError("trajectory_id cannot be reused for a different task")
        return state

    async def _reference_resolution(
        self,
        snapshot: AgentWorkflowSnapshot,
        state: _TrajectoryState,
        *,
        round_index: int,
    ) -> _KeyResolution:
        graph = snapshot.graph
        focal = state.last_focal_agent_id
        node_ids = {node.id for node in graph.nodes}
        if focal not in node_ids:
            if graph.output_agent_id in node_ids:
                focal = graph.output_agent_id
            elif len(graph.nodes) == 1:
                focal = graph.nodes[0].id
            else:
                raise _Unclassifiable("no_unique_reference_decision")
        return await self._key_for_node(
            graph,
            focal,
            round_index=round_index,
        )

    async def _resolve_action(
        self,
        action: AgentAction,
        canvas: AgentWorkflowStepResult,
        *,
        round_index: int,
    ) -> _KeyResolution:
        graph = canvas.snapshot.graph
        if action.action_type in {AgentActionType.ADD_AGENT, AgentActionType.MODIFY_AGENT}:
            if action.agent_id is None:
                raise _Unclassifiable("missing_agent_id")
            return await self._key_for_node(
                graph,
                action.agent_id,
                round_index=round_index,
            )
        if action.action_type is AgentActionType.SET_RELATION:
            return await self._key_for_relation(
                graph,
                action,
                round_index=round_index,
            )
        raise _Unclassifiable(
            f"decision_key_not_defined_for_{action.action_type.value}"
        )

    async def _key_for_node(
        self,
        graph: AgentGraphSnapshot,
        agent_id: str,
        *,
        round_index: int,
    ) -> _KeyResolution:
        node = self._node(graph, agent_id)
        self._require_catalog_model(node.model_id)
        incoming = self._incoming_relations(graph, agent_id)
        if len(incoming) > 1:
            raise _Unclassifiable("ambiguous_multiple_upstream_agents")
        if not incoming:
            edge_type = "independent"
            upstream_id = None
            same_model = False
        else:
            relation, upstream_id = incoming[0]
            edge_type = "bidirectional" if relation.bits.is_bidirectional else "unidirectional"
            upstream = self._node(graph, upstream_id)
            self._require_catalog_model(upstream.model_id)
            same_model = upstream.model_id == node.model_id
        classification = await self._classify(
            node.contract,
            seed=self.random_seed + round_index,
        )
        return _KeyResolution(
            key=DecisionKey(
                task_family=TASK_FAMILY,
                role_cluster=classification.role_cluster,
                model_id=node.model_id,
                edge_type=edge_type,
                same_model_as_upstream=same_model,
                stage=self._stage(graph, agent_id),
            ),
            focal_agent_id=agent_id,
            upstream_agent_id=upstream_id,
            classification=classification,
        )

    async def _key_for_relation(
        self,
        graph: AgentGraphSnapshot,
        action: AgentAction,
        *,
        round_index: int,
    ) -> _KeyResolution:
        source_id = action.source_id
        target_id = action.target_id
        forward = action.source_to_target
        reverse = action.target_to_source
        if source_id is None or target_id is None or forward is None or reverse is None:
            raise _Unclassifiable("incomplete_relation_action")
        if forward and reverse:
            focal_id, upstream_id, edge_type = target_id, source_id, "bidirectional"
        elif forward:
            focal_id, upstream_id, edge_type = target_id, source_id, "unidirectional"
        elif reverse:
            focal_id, upstream_id, edge_type = source_id, target_id, "unidirectional"
        else:
            focal_id, upstream_id, edge_type = target_id, None, "independent"
        focal = self._node(graph, focal_id)
        self._require_catalog_model(focal.model_id)
        if upstream_id is None:
            same_model = False
        else:
            upstream = self._node(graph, upstream_id)
            self._require_catalog_model(upstream.model_id)
            same_model = upstream.model_id == focal.model_id
        classification = await self._classify(
            focal.contract,
            seed=self.random_seed + round_index,
        )
        return _KeyResolution(
            key=DecisionKey(
                task_family=TASK_FAMILY,
                role_cluster=classification.role_cluster,
                model_id=focal.model_id,
                edge_type=edge_type,
                same_model_as_upstream=same_model,
                stage=self._stage(graph, focal_id),
            ),
            focal_agent_id=focal_id,
            upstream_agent_id=upstream_id,
            classification=classification,
        )

    async def _classify(self, contract: str, *, seed: int) -> RoleClassification:
        cached = self._role_cache.get(contract)
        if cached is not None:
            return cached
        try:
            result = await self.role_classifier.classify(contract, seed=seed)
        except RoleClassificationError as error:
            raise _Unclassifiable(
                "role_classification_failed",
                classification_receipt=error.receipt.to_dict(),
            ) from error
        self._role_cache[contract] = result
        return result

    def _ranked_candidates(
        self,
        current: DecisionKey,
        *,
        step_id: int,
        candidate_domain: Sequence[DecisionKey] | None = None,
    ):
        candidates = (
            tuple(candidate_domain)
            if candidate_domain is not None
            else single_field_candidates(
                current,
                model_ids=self.model_catalog,
                edge_types=_EDGE_TYPES,
                role_clusters=ROLE_CLUSTERS,
                same_model_options=(False, True),
            )
        )
        return self.epoch.pre_execution(
            step_id=step_id,
            current=current,
            candidates=candidates,
            config=self.latent_config,
        )

    def _translatable_candidates(
        self,
        action: AgentAction,
        current: DecisionKey,
    ) -> tuple[DecisionKey, ...]:
        """Restrict post-action alternatives to the existing translator domain."""

        if action.action_type in {AgentActionType.ADD_AGENT, AgentActionType.MODIFY_AGENT}:
            model_ids = self.model_catalog if action.model_id == current.model_id else ()
            roles = ROLE_CLUSTERS if action.contract is not None else ()
            return single_field_candidates(
                current,
                model_ids=model_ids,
                role_clusters=roles,
            )
        if action.action_type is AgentActionType.SET_RELATION:
            return single_field_candidates(current, edge_types=_EDGE_TYPES)
        return ()

    def _surface_signal(
        self,
        canvas: AgentWorkflowStepResult,
        resolution: _KeyResolution,
    ) -> SurfaceSignal:
        key = resolution.key
        if "execution_error=" in canvas.feedback:
            return SurfaceSignal.for_key("exec_error", True, key)
        execution = canvas.execution
        if execution is None:
            return SurfaceSignal.for_key("none", False, key)

        if key.role_cluster == "verify":
            values = [
                call.response.metadata.get("verifier_pass")
                for call in execution.calls
                if call.request.agent.id == resolution.focal_agent_id
                and type(call.response.metadata.get("verifier_pass")) is bool
            ]
            if values and all(value == values[0] for value in values):
                return SurfaceSignal.for_key("verifier_pass", bool(values[0]), key)

        if key.edge_type == "bidirectional" and resolution.upstream_agent_id is not None:
            outputs = execution.outputs
            focal_output = outputs.get(resolution.focal_agent_id)
            upstream_output = outputs.get(resolution.upstream_agent_id)
            if isinstance(focal_output, str) and isinstance(upstream_output, str):
                return SurfaceSignal.for_key(
                    "bidir_consensus",
                    focal_output.strip() == upstream_output.strip(),
                    key,
                )
        return SurfaceSignal.for_key("exec_error", False, key)

    @staticmethod
    def _response_to_prior_readout(
        state: _TrajectoryState,
        current: DecisionKey,
    ) -> str:
        if not state.warning_presented or state.last_key is None:
            return "not_triggered"
        try:
            changed_field(state.last_key, current)
        except ValueError:
            return "continue"
        return "revise"

    def _unclassifiable_observation(self, reason: str) -> Mapping[str, Any]:
        return {
            "schema_version": ROLLOUT_HOOK_SCHEMA_VERSION,
            "condition_id": self.epoch.condition.condition_id,
            "status": "unclassifiable",
            "reason": reason,
            "candidates": [],
            "active_skills": [],
            "active_skill_ids": [],
            "advisory_only": True,
        }

    def _record_event(
        self,
        state: _TrajectoryState,
        trajectory_id: str,
        round_index: int,
        reason: str,
        **details: Any,
    ) -> Mapping[str, Any]:
        event = {
            "schema_version": ROLLOUT_HOOK_SCHEMA_VERSION,
            "status": "unclassifiable",
            "trajectory_id": trajectory_id,
            "step_id": round_index,
            "reason": reason,
            **{key: value for key, value in details.items() if value is not None},
        }
        state.events.append(event)
        state.warning_presented = False
        return event

    def _require_catalog_model(self, model_id: str) -> None:
        if model_id not in self.model_catalog:
            raise _Unclassifiable("model_not_in_frozen_catalog")

    @staticmethod
    def _node(graph: AgentGraphSnapshot, agent_id: str) -> AgentNode:
        nodes = [node for node in graph.nodes if node.id == agent_id]
        if len(nodes) != 1:
            raise _Unclassifiable("agent_binding_not_unique")
        return nodes[0]

    @staticmethod
    def _incoming_relations(
        graph: AgentGraphSnapshot,
        agent_id: str,
    ) -> tuple[tuple[AgentRelation, str], ...]:
        incoming: list[tuple[AgentRelation, str]] = []
        for relation in graph.relations:
            for source_id, target_id in relation.directed_edges():
                if target_id == agent_id:
                    incoming.append((relation, source_id))
        return tuple(incoming)

    @staticmethod
    def _stage(graph: AgentGraphSnapshot, agent_id: str) -> str:
        output_id = graph.output_agent_id
        if output_id is None:
            return "other"
        if agent_id == output_id:
            return "before_output"
        for relation in graph.relations:
            if (agent_id, output_id) in relation.directed_edges():
                return "before_output"
        return "other"


__all__ = [
    "HotpotQADynamicLedgerHook",
    "ProbeSite",
    "ROLLOUT_HOOK_SCHEMA_VERSION",
    "TASK_FAMILY",
    "TrajectoryLedgerSidecar",
]
