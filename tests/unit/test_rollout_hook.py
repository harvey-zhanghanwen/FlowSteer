"""Mock-only tests for the HotpotQA dynamic-ledger rollout hook."""

from __future__ import annotations

import asyncio

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import DirectorResponse
from src.interactive.exploration.combination_ledger import CombinationPosterior
from src.interactive.exploration.latent_loss import changed_field
from src.interactive.exploration.ledger_epoch import LedgerEpoch, SkillTextSelection
from src.interactive.exploration.role_classifier import (
    RoleClassifier,
    RoleCompletionResponse,
)
from src.interactive.exploration.rollout_hook import HotpotQADynamicLedgerHook
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.records import EvaluationReceipt, TaskRecord, TrajectoryRecord
from src.interactive.versioning import VersionBundle


class ScriptedRoleCompletion:
    def __init__(self, *roles: str) -> None:
        self.roles = list(roles)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        role = self.roles.pop(0)
        return RoleCompletionResponse(
            f'{{"role_cluster":"{role}"}}',
            {"provider_request_id": f"role-{len(self.requests)}"},
        )


class InvalidRoleCompletion:
    async def complete(self, request):
        del request
        return RoleCompletionResponse("retrieve", {"provider_request_id": "invalid"})


class ReceiptGateway:
    def __init__(self, output: str = "same supported answer") -> None:
        self.output = output

    async def generate(self, request):
        return AgentResponse(
            self.output,
            {
                "provider_request_id": request.request_id,
                "provider_model": request.model.model_name,
                "verifier_pass": True,
            },
        )


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", endpoint="http://provider.invalid/v1")],
        [
            ModelSpec("model-a", "provider"),
            ModelSpec("model-b", "provider"),
            ModelSpec("model-c", "provider"),
        ],
    )


def _versions() -> VersionBundle:
    return VersionBundle(
        policy="policy-e0",
        model_catalog="catalog-frozen-v1",
        evaluator="hotpot-em-v1",
        prompt="minimal-director-v1",
        tool="agentgraph-v1",
        feature_schema="decision-key-v1",
    )


def _epoch() -> LedgerEpoch:
    return LedgerEpoch.freeze(CombinationPosterior("policy-e0"), (), _versions())


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="hotpotqa:train:0001",
        question="Which city is linked by the two stated facts?",
        ground_truth="answer kept outside the hook",
        split="train",
        metadata={"source": "HotpotQA"},
    )


def _hook(completion, *, epoch: LedgerEpoch | None = None):
    return HotpotQADynamicLedgerHook(
        epoch=epoch or _epoch(),
        role_classifier=RoleClassifier(
            completion,
            request_id_factory=lambda: "role-request",
        ),
        model_catalog=("model-a", "model-b", "model-c"),
        max_rounds=12,
        random_seed=17,
    )


async def _observe_action(hook, env, task, trajectory_id, round_index, action_text):
    pre_snapshot = env.snapshot()
    response = DirectorResponse(action_text)
    canvas = await env.step(action_text)
    sidecar = await hook.after_turn(
        task=task,
        trajectory_id=trajectory_id,
        round_index=round_index,
        pre_snapshot=pre_snapshot,
        response=response,
        canvas=canvas,
        observation={},
    )
    return pre_snapshot, canvas, sidecar


def test_before_turn_is_bounded_and_add_uses_only_llm_role(monkeypatch) -> None:
    task = _task()
    epoch = _epoch()
    monkeypatch.setattr(
        epoch,
        "select_skill_texts",
        lambda **kwargs: SkillTextSelection(
            skill_ids=("skill-1", "skill-2", "skill-3"),
            texts=("skill text 1", "skill text 2", "skill text 3"),
        ),
    )
    completion = ScriptedRoleCompletion("solve")
    hook = _hook(completion, epoch=epoch)
    env = AgentWorkflowEnv(_registry(), ReceiptGateway())
    env.reset(task.question)

    initial = asyncio.run(
        hook.before_turn(
            task=task,
            trajectory_id="trajectory-1",
            round_index=0,
            environment=env,
        )
    )
    assert initial["status"] == "unclassifiable"
    assert initial["candidates"] == []

    contract = "Retrieve every relevant source and quote it."  # wording must not decide the role
    action = (
        '{"action":"add_agent","agent_id":"a","model_id":"model-a",'
        f'"contract":"{contract}"}}'
    )
    pre_snapshot, _, recorded = asyncio.run(
        _observe_action(hook, env, task, "trajectory-1", 0, action)
    )
    step = recorded["step_record"]
    assert recorded["status"] == "recorded"
    assert step["key"]["role_cluster"] == "solve"
    assert step["key"]["task_family"] == "hotpotqa"
    assert step["key"]["model_id"] == "model-a"
    assert step["snapshot_id"] == pre_snapshot.snapshot_id
    assert len(step["candidates"]) <= 5
    assert len(completion.requests) == 1

    next_observation = asyncio.run(
        hook.before_turn(
            task=task,
            trajectory_id="trajectory-1",
            round_index=1,
            environment=env,
        )
    )
    assert next_observation["status"] == "available"
    assert len(next_observation["candidates"]) <= 5
    assert next_observation["active_skills"] == [
        "skill text 1",
        "skill text 2",
        "skill text 3",
    ]
    assert len(completion.requests) == 1  # identical contract came from the frozen cache

    site = hook.probe_sites["trajectory-1"][0]
    assert site.pre_snapshot == pre_snapshot
    assert site.actual_action.to_dict() == {
        "action": "add_agent",
        "agent_id": "a",
        "model_id": "model-a",
        "contract": contract,
    }
    assert changed_field(site.current_key, site.candidate_key) in {
        "model_id",
        "role_cluster",
    }


def test_verifier_signal_comes_from_execution_receipt() -> None:
    task = _task()
    completion = ScriptedRoleCompletion("solve", "verify")
    hook = _hook(completion)
    env = AgentWorkflowEnv(_registry(), ReceiptGateway(), execute_on_edit=True)
    env.reset(task.question)

    add = (
        '{"action":"add_agent","agent_id":"answerer","model_id":"model-a",'
        '"contract":"Draft a supported answer."}'
    )
    asyncio.run(_observe_action(hook, env, task, "trajectory-2", 0, add))
    asyncio.run(env.step('{"action":"set_output","agent_id":"answerer"}'))
    modify = (
        '{"action":"modify_agent","agent_id":"answerer",'
        '"contract":"Check the evidence binding before emitting the answer."}'
    )
    _, canvas, recorded = asyncio.run(
        _observe_action(hook, env, task, "trajectory-2", 2, modify)
    )

    assert canvas.execution is not None
    surface = recorded["step_record"]["surface"]
    assert surface["kind"] == "verifier_pass"
    assert surface["value"] is True
    assert recorded["step_record"]["key"]["stage"] == "before_output"
    assert recorded["role_classification"]["role_cluster"] == "verify"


def test_bidirectional_relation_uses_exact_runtime_outputs_and_edge_candidates() -> None:
    task = _task()
    hook = _hook(ScriptedRoleCompletion("arbitrate"))
    env = AgentWorkflowEnv(_registry(), ReceiptGateway(), execute_on_edit=True)
    env.reset(task.question)
    asyncio.run(
        env.step(
            '{"action":"add_agent","agent_id":"left","model_id":"model-a",'
            '"contract":"Produce one supported answer."}'
        )
    )
    asyncio.run(
        env.step(
            '{"action":"add_agent","agent_id":"right","model_id":"model-b",'
            '"contract":"Reconcile the available evidence."}'
        )
    )
    asyncio.run(env.step('{"action":"set_output","agent_id":"right"}'))
    relation = (
        '{"action":"set_relation","source_id":"left","target_id":"right",'
        '"source_to_target":true,"target_to_source":true}'
    )
    pre_snapshot, canvas, recorded = asyncio.run(
        _observe_action(hook, env, task, "trajectory-3", 3, relation)
    )

    assert canvas.execution is not None
    assert recorded["step_record"]["key"]["edge_type"] == "bidirectional"
    assert recorded["step_record"]["surface"]["kind"] == "bidir_consensus"
    assert recorded["step_record"]["surface"]["value"] is True
    assert {
        changed_field(
            hook.probe_sites["trajectory-3"][0].current_key,
            hook.probe_sites["trajectory-3"][0].candidate_key,
        )
    } == {"edge_type"}
    assert hook.probe_sites["trajectory-3"][0].pre_snapshot == pre_snapshot


def test_role_failure_is_explicit_and_after_trajectory_does_not_update_epoch() -> None:
    task = _task()
    epoch = _epoch()
    hook = _hook(InvalidRoleCompletion(), epoch=epoch)
    env = AgentWorkflowEnv(_registry(), ReceiptGateway())
    env.reset(task.question)
    action = (
        '{"action":"add_agent","agent_id":"a","model_id":"model-a",'
        '"contract":"Retrieve and verify every fact."}'
    )
    _, _, event = asyncio.run(
        _observe_action(hook, env, task, "trajectory-4", 0, action)
    )
    assert event["status"] == "unclassifiable"
    assert event["reason"] == "role_classification_failed"
    assert event["classification_receipt"]["status"] == "failed"
    assert "trajectory-4" not in hook.probe_sites

    before = epoch.ledger.state_dict()
    trajectory = TrajectoryRecord(
        trajectory_id="trajectory-4",
        task=task,
        group_id="group-4",
        condition_id=epoch.condition.condition_id,
        rollout_id="rollout-4",
        versions=epoch.condition.versions,
        turns=(),
        final_answer=None,
        evaluation=EvaluationReceipt(
            evaluator_version=epoch.condition.versions.evaluator,
            valid=False,
            reward=None,
            reason="not used by hook",
        ),
        termination_reason="max_rounds",
        explicit_finish=False,
    )
    hook.after_trajectory(trajectory=trajectory)
    sidecar = hook.sidecar("trajectory-4")
    assert sidecar.steps == ()
    assert sidecar.events[0]["reason"] == "role_classification_failed"
    after = epoch.ledger.state_dict()
    assert after["probe_count"] == before["probe_count"] == 0
    assert after["task_baseline"] == before["task_baseline"]
