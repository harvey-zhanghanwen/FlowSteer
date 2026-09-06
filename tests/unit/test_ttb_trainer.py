from __future__ import annotations

from dataclasses import replace
import json

import pytest

from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.ttb_trainer import (
    Qwen35TTBTrainer,
    TTBStepCoordinate,
    TTBTrainerConfig,
    TTBTrainingSummary,
)
from src.interactive.versioning import VersionBundle


BEHAVIOR_THETA = "theta-step-000"
UPDATED_THETA = "theta-step-001"
BEHAVIOR_PHI = "phi-step-000"
UPDATED_PHI = "phi-step-001"
BEHAVIOR_Z = "z-step-000"
UPDATED_Z = "z-step-001"
SERVER_WEIGHT = "server-weight-base"


def trainer_config(**changes: object) -> TTBTrainerConfig:
    values = {
        "model_path": "/models/qwen3.5-9b",
        "tokenizer_path": "/models/qwen3.5-9b",
        "behavior_policy_version": BEHAVIOR_THETA,
        "updated_policy_version": UPDATED_THETA,
        "behavior_phi_version": BEHAVIOR_PHI,
        "updated_phi_version": UPDATED_PHI,
        "behavior_z_version": BEHAVIOR_Z,
        "updated_z_version": UPDATED_Z,
        "behavior_server_weight_version": SERVER_WEIGHT,
        "update_step": 1,
        "learner_device": "cuda:3",
        "gradient_replica_device": "cuda:5",
    }
    values.update(changes)
    return TTBTrainerConfig(**values)


def step_coordinate(**changes: object) -> TTBStepCoordinate:
    values = {
        "step": 1,
        "behavior_theta_version": BEHAVIOR_THETA,
        "updated_theta_version": UPDATED_THETA,
        "behavior_phi_version": BEHAVIOR_PHI,
        "updated_phi_version": UPDATED_PHI,
        "behavior_z_version": BEHAVIOR_Z,
        "updated_z_version": UPDATED_Z,
        "behavior_policy_adapter": None,
        "behavior_server_weight_version": SERVER_WEIGHT,
    }
    values.update(changes)
    return TTBStepCoordinate(**values)


def _sampling(task_id: str, sequence_position: int) -> dict[str, object]:
    base_seed = 101
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(
            base_seed=base_seed
        ),
        schedule_purpose="train",
        ordered_sequence_hash=stable_hash([f"mbppplus-{index}" for index in range(7)]),
        sequence_position=sequence_position,
        task_id=task_id,
        optimizer_step_or_anchor_ordinal=1,
    )
    return {
        "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
        "base_seed": base_seed,
        "coordinate": coordinate.to_value(),
        "phase": GenerationPhase.ACTION.value,
    }


def _trajectory(
    *,
    question_index: int,
    rollout_index: int,
    runtime_summary: dict[str, object] | None = None,
) -> TrajectoryRecord:
    task_id = f"mbppplus-{question_index}"
    sampling = _sampling(task_id, rollout_index)
    coordinate = ScientificSamplingCoordinate.from_value(sampling["coordinate"])
    snapshot = GraphSnapshotEvent.create(
        1,
        {"nodes": [{"id": "agent-1"}], "relations": [], "output_agent_id": "agent-1"},
    )
    turn = TurnRecord(
        turn_id=f"turn-{question_index}-{rollout_index}",
        round_index=0,
        prompt="Build an AgentGraph action.",
        policy_response='Thought: inspect task\n{"action":"finish"}',
        prompt_token_ids=(11, 12, 13),
        output_token_ids=(21, 22, 23, 24),
        behavior_log_probs=(-0.1, -0.2, -0.3, -0.4),
        executed_prefix_tokens=4,
        structured_action_token_start=2,
        structured_action_token_count=2,
        action={"action": "finish"},
        canvas_feedback="finished",
        graph_revision=1,
        graph_snapshot=snapshot.to_dict()["graph"],
        graph_snapshot_id=snapshot.snapshot_id,
        previous_graph_snapshot_id=None,
        policy_version=BEHAVIOR_THETA,
        policy_adapter=None,
        server_weight_version=SERVER_WEIGHT,
        runtime_summary=runtime_summary or {},
        director_generation_seed=derive_generation_seed(
            base_seed=sampling["base_seed"],
            coordinate=coordinate,
            step_index=1,
            phase=GenerationPhase.ACTION,
        ),
        receipt_verified=True,
    )
    versions = VersionBundle(
        policy=BEHAVIOR_THETA,
        model_catalog="catalog-v1",
        evaluator="mbppplus-evaluator-v1",
        prompt="director-prompt-v1",
        tool="python-tools-v1",
    )
    return TrajectoryRecord(
        trajectory_id=f"trajectory-{question_index}-{rollout_index}",
        task=TaskRecord(task_id, f"Problem {question_index}", "reference", "train"),
        group_id=f"{task_id}:train:{BEHAVIOR_THETA}",
        condition_id="mbppplus-ttb-train-v1",
        rollout_id=f"rollout-{question_index}-{rollout_index}",
        versions=versions,
        turns=(turn,),
        final_answer="candidate program",
        evaluation=EvaluationReceipt("mbppplus-evaluator-v1", True, 1.0),
        termination_reason="finish",
        explicit_finish=True,
        director_sampling=sampling,
    )


def _batch(counts: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4)) -> tuple[TrajectoryRecord, ...]:
    return tuple(
        _trajectory(question_index=question_index, rollout_index=rollout_index)
        for question_index, count in enumerate(counts)
        for rollout_index in range(count)
    )


def test_trainer_config_locks_the_formal_skillflow_profile() -> None:
    config = trainer_config()

    assert config.expected_batch_size == 28
    assert (config.questions_per_batch, config.trajectories_per_question) == (7, 4)
    assert config.max_trajectory_edges == 12
    assert config.beta == 1.0
    assert config.epsilon_min == 0.1
    assert config.learning_rate == 1.0e-4
    assert config.max_grad_norm == 3.0
    assert config.kl_coefficient == 0.01
    assert config.checkpoint_interval == 10
    assert (config.theta_rank, config.theta_alpha) == (64, 128)
    assert config.theta_dropout == 0.05
    assert config.theta_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert (config.phi_rank, config.phi_alpha) == (16, 32)
    assert config.phi_dropout == 0.05
    assert config.phi_target_modules == ("q_proj", "v_proj")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"theta_rank": 32}, "theta-LoRA"),
        ({"theta_dropout": 0.0}, "theta-LoRA"),
        ({"theta_target_modules": ("q_proj", "v_proj")}, "theta-LoRA"),
        ({"phi_rank": 32}, "phi-LoRA"),
        ({"phi_dropout": 0.0}, "phi-LoRA"),
        ({"phi_target_modules": ("q_proj",)}, "phi-LoRA"),
        ({"expected_batch_size": 14}, "effective batch"),
        ({"questions_per_batch": 14, "trajectories_per_question": 2}, "batch shape"),
        ({"max_trajectory_edges": 11}, "maximum trajectory horizon"),
        ({"learning_rate": 2.0e-4}, "learning_rate"),
        ({"beta": 0.5}, "beta"),
        ({"epsilon_min": 0.01}, "epsilon_min"),
        ({"kl_coefficient": 0.0}, "kl_coefficient"),
        ({"max_grad_norm": 1.0}, "max_grad_norm"),
        ({"checkpoint_interval": 20}, "checkpoint interval"),
        ({"require_unconstrained_action_sampling": False}, "unconstrained action sampling"),
    ],
)
def test_trainer_config_rejects_changes_to_fixed_parameters(
    change: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        trainer_config(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"updated_policy_version": BEHAVIOR_THETA},
        {"updated_phi_version": BEHAVIOR_PHI},
        {"updated_z_version": BEHAVIOR_Z},
        {"behavior_policy_version": ""},
        {"learner_device": "cuda:3", "gradient_replica_device": "cuda:3"},
        {"update_step": 2},
    ],
)
def test_trainer_config_rejects_non_advancing_or_unrecoverable_versions(
    change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        trainer_config(**change)


def test_step_coordinate_requires_positive_step_and_advancing_versions() -> None:
    assert step_coordinate().step == 1
    for change in (
        {"step": 0},
        {"updated_theta_version": BEHAVIOR_THETA},
        {"updated_phi_version": BEHAVIOR_PHI},
        {"updated_z_version": BEHAVIOR_Z},
        {"behavior_server_weight_version": ""},
    ):
        with pytest.raises(ValueError):
            step_coordinate(**change)


def test_batch_admission_accepts_exactly_seven_questions_by_four_trajectories() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = _batch()

    admitted = trainer._validate_batch(batch, step_coordinate())

    assert admitted == batch
    assert len(admitted) == 28
    assert {record.task.task_id for record in admitted} == {
        f"mbppplus-{index}" for index in range(7)
    }


def test_batch_admission_rejects_wrong_size_and_wrong_question_shape() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    valid = _batch()

    with pytest.raises(ValueError, match="exactly 28"):
        trainer._validate_batch(valid[:-1], step_coordinate())
    with pytest.raises(ValueError, match="seven questions with four trajectories"):
        trainer._validate_batch(_batch((4, 4, 4, 4, 4, 4, 2, 2)), step_coordinate())


def test_batch_admission_rejects_mixed_version_bundle() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = list(_batch())
    batch[0] = replace(
        batch[0],
        versions=replace(batch[0].versions, prompt="different-prompt-version"),
    )

    with pytest.raises(ValueError, match="mixed VersionBundle regimes"):
        trainer._validate_batch(batch, step_coordinate())


def test_batch_admission_rejects_duplicate_rollout_id() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = list(_batch())
    batch[1] = replace(batch[1], rollout_id=batch[0].rollout_id)

    with pytest.raises(ValueError, match="duplicate rollout IDs"):
        trainer._validate_batch(batch, step_coordinate())


def test_batch_admission_rejects_duplicate_scientific_sampling_coordinate() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = list(_batch())
    batch[1] = replace(batch[1], director_sampling=batch[0].director_sampling)

    with pytest.raises(ValueError, match="duplicate scientific sampling coordinates"):
        trainer._validate_batch(batch, step_coordinate())


def test_batch_admission_rejects_mixed_group_within_question() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = list(_batch())
    batch[1] = replace(batch[1], group_id="foreign-group")

    with pytest.raises(ValueError, match="one frozen trajectory group regime"):
        trainer._validate_batch(batch, step_coordinate())


@pytest.mark.parametrize(
    "runtime_summary",
    [
        {"director_action_decoding": "grammar"},
        {"director_action_schema_version": "agentgraph-action-schema-v3"},
        {"director_action_schema_branch": "add_subgraph"},
        {"director_action_target_domain_version": "target-domains-v1"},
    ],
)
def test_batch_admission_rejects_grammar_or_hierarchical_action_receipts(
    runtime_summary: dict[str, object]
) -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    batch = list(_batch())
    original = batch[0]
    batch[0] = replace(
        original,
        turns=(replace(original.turns[0], runtime_summary=runtime_summary),),
    )

    assert batch[0].ttb_eligible
    with pytest.raises(ValueError, match="grammar-constrained or hierarchical"):
        trainer._validate_batch(batch, step_coordinate())


def test_token_cost_partition_is_balanced_and_preserves_original_order() -> None:
    records = list(_batch()[:4])
    desired_costs = (10, 9, 8, 7)
    for index, desired_cost in enumerate(desired_costs):
        original = records[index]
        output_ids = tuple(original.turns[0].output_token_ids)
        prompt_length = desired_cost - len(output_ids)
        records[index] = replace(
            original,
            turns=(
                replace(
                    original.turns[0],
                    prompt_token_ids=tuple(range(prompt_length)),
                ),
            ),
        )

    partitions, costs = Qwen35TTBTrainer._partition_by_token_cost(records)

    assert costs == (17, 17)
    assert tuple(index for index, _record in partitions[0]) == (0, 3)
    assert tuple(index for index, _record in partitions[1]) == (1, 2)
    assert sorted(index for partition in partitions for index, _record in partition) == [0, 1, 2, 3]


def test_step_validation_requires_contiguous_steps_and_current_behavior_versions() -> None:
    trainer = Qwen35TTBTrainer(trainer_config())
    trainer._validate_step_coordinate(step_coordinate())

    trainer._committed_step = 1
    trainer._current_theta_version = UPDATED_THETA
    trainer._current_phi_version = UPDATED_PHI
    trainer._current_z_version = UPDATED_Z
    trainer._current_policy_adapter = "/adapters/theta-step-001"
    trainer._current_server_weight_version = "server-weight-step-001"
    second = TTBStepCoordinate(
        step=2,
        behavior_theta_version=UPDATED_THETA,
        updated_theta_version="theta-step-002",
        behavior_phi_version=UPDATED_PHI,
        updated_phi_version="phi-step-002",
        behavior_z_version=UPDATED_Z,
        updated_z_version="z-step-002",
        behavior_policy_adapter="/adapters/theta-step-001",
        behavior_server_weight_version="server-weight-step-001",
    )
    trainer._validate_step_coordinate(second)

    with pytest.raises(ValueError, match="contiguous"):
        trainer._validate_step_coordinate(replace(second, step=3))
    for field, stale in (
        ("behavior_theta_version", BEHAVIOR_THETA),
        ("behavior_phi_version", BEHAVIOR_PHI),
        ("behavior_z_version", BEHAVIOR_Z),
    ):
        with pytest.raises(ValueError, match="behavior version does not match trainer state"):
            trainer._validate_step_coordinate(replace(second, **{field: stale}))


def _summary(tmp_path) -> TTBTrainingSummary:
    return TTBTrainingSummary(
        optimizer_updates=1,
        update_step=1,
        input_trajectories=28,
        eligible_trajectories=28,
        filtered_trajectories=0,
        ttb_loss=1.0,
        total_loss=1.0,
        delta_squared_mean=1.0,
        delta_squared_sum=28.0,
        delta_min=-1.0,
        delta_max=1.0,
        log_z_mean=-2.3,
        kl_estimate=0.0,
        reward_mean=0.5,
        reward_min=0.0,
        reward_max=1.0,
        trajectory_edges_mean=1.0,
        trajectory_edges_min=1,
        trajectory_edges_max=1,
        action_token_count=56,
        theta_grad_norm=1.0,
        phi_grad_norm=1.0,
        z_grad_norm=1.0,
        theta_update_l2=0.1,
        phi_update_l2=0.1,
        z_update_l2=0.1,
        behavior_policy_version=BEHAVIOR_THETA,
        updated_policy_version=UPDATED_THETA,
        phi_behavior_version=BEHAVIOR_PHI,
        phi_updated_version=UPDATED_PHI,
        z_behavior_version=BEHAVIOR_Z,
        z_updated_version=UPDATED_Z,
        checkpoint_dir=str(tmp_path / "publication" / "theta"),
        phi_checkpoint_dir=str(tmp_path / "recovery" / "phi"),
        recovery_checkpoint_dir=str(tmp_path / "recovery"),
        official_checkpoint_dir="",
        official_checkpoint_saved=False,
        exclusions={},
        step_seconds=1.0,
        started_at="2026-09-06T00:00:00+00:00",
        completed_at="2026-09-06T00:00:01+00:00",
    )


def _trainer_awaiting_publication(tmp_path) -> tuple[Qwen35TTBTrainer, dict[str, object]]:
    trainer = Qwen35TTBTrainer(trainer_config())
    trainer._output_root = tmp_path
    trainer._learner_model = object()
    coordinate = step_coordinate()
    step_root = tmp_path / "step"
    recovery_root = step_root / "recovery"
    publication_theta = step_root / "publication" / "theta"
    recovery_root.mkdir(parents=True)
    publication_theta.mkdir(parents=True)
    (recovery_root / "training_state.json").write_text(
        json.dumps(
            {
                "optimizer_committed_step": 1,
                "committed_step": None,
                "theta_version": UPDATED_THETA,
                "phi_version": UPDATED_PHI,
                "z_version": UPDATED_Z,
                "publication_committed": False,
            }
        ),
        encoding="utf-8",
    )
    (publication_theta / "policy_version.json").write_text(
        json.dumps(
            {
                "updated_policy_version": UPDATED_THETA,
                "external_publication_pending": True,
            }
        ),
        encoding="utf-8",
    )
    trainer._active_transaction_root = step_root
    trainer._active_transaction = {
        "step": 1,
        "phase": "AWAITING_PUBLICATION",
        "optimizer_committed": True,
        "publication_committed": False,
    }
    trainer._pending_publication = {
        "coordinate": coordinate,
        "summary": _summary(step_root),
        "step_root": step_root,
        "recovery_root": recovery_root,
        "official_root": None,
        "publication_theta": publication_theta,
        "transaction": dict(trainer._active_transaction),
    }
    receipt = {
        "success": True,
        "status": "published",
        "canary_succeeded": True,
        "training_performed": True,
        "policy_published": True,
        "gate_used": True,
        "gate_drained": True,
        "route_switch_requested": True,
        "route_switch_succeeded": True,
        "behavior_policy_version": BEHAVIOR_THETA,
        "candidate_policy_version": UPDATED_THETA,
        "new_policy_version": UPDATED_THETA,
        "previous_adapter": None,
        "adapter_name": "theta-live-step-001",
        "server_weight_version": "server-weight-step-001",
        "checkpoint_path": str(publication_theta),
    }
    return trainer, receipt


def test_pending_publication_blocks_the_next_optimizer_step(tmp_path) -> None:
    trainer, _receipt = _trainer_awaiting_publication(tmp_path)

    with pytest.raises(RuntimeError, match="awaiting verified publication"):
        trainer._validate_step_coordinate(step_coordinate())
    with pytest.raises(RuntimeError, match="awaiting verified publication"):
        trainer.train_step(_batch(), coordinate=step_coordinate())


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("success", False),
        ("status", "failed"),
        ("canary_succeeded", False),
        ("policy_published", False),
        ("gate_used", False),
        ("gate_drained", False),
        ("route_switch_requested", False),
        ("route_switch_succeeded", False),
        ("new_policy_version", "foreign-policy"),
        ("adapter_name", ""),
        ("server_weight_version", ""),
        ("checkpoint_path", "/wrong/checkpoint"),
    ],
)
def test_publication_acknowledgement_rejects_incomplete_or_mismatched_receipt(
    tmp_path, field: str, invalid_value: object
) -> None:
    trainer, receipt = _trainer_awaiting_publication(tmp_path)
    receipt[field] = invalid_value

    with pytest.raises(ValueError):
        trainer.acknowledge_publication(receipt)

    assert trainer.committed_step == 0
    assert trainer._current_theta_version == BEHAVIOR_THETA
    assert trainer._pending_publication is not None


def test_successful_publication_acknowledgement_commits_versions_route_and_receipts(
    tmp_path,
) -> None:
    trainer, receipt = _trainer_awaiting_publication(tmp_path)
    pending = trainer._pending_publication
    assert pending is not None
    step_root = pending["step_root"]
    recovery_root = pending["recovery_root"]
    publication_theta = pending["publication_theta"]

    summary = trainer.acknowledge_publication(receipt)

    assert summary.update_step == 1
    assert trainer.committed_step == 1
    assert trainer._current_theta_version == UPDATED_THETA
    assert trainer._current_phi_version == UPDATED_PHI
    assert trainer._current_z_version == UPDATED_Z
    assert trainer._current_policy_adapter == "theta-live-step-001"
    assert trainer._current_server_weight_version == "server-weight-step-001"
    assert trainer._pending_publication is None
    assert trainer._active_transaction is None
    assert trainer._active_transaction_root is None

    committed_state = json.loads(
        (recovery_root / "training_state.json").read_text(encoding="utf-8")
    )
    assert committed_state["committed_step"] == 1
    assert committed_state["publication_committed"] is True
    assert committed_state["published_policy_adapter"] == "theta-live-step-001"
    assert (
        committed_state["published_server_weight_version"]
        == "server-weight-step-001"
    )
    assert json.loads(
        (recovery_root / "publication_receipt.json").read_text(encoding="utf-8")
    )["ttb_step"] == 1
    assert json.loads(
        (step_root / "publication_receipt.json").read_text(encoding="utf-8")
    )["canary_succeeded"] is True
    assert json.loads(
        (publication_theta / "policy_version.json").read_text(encoding="utf-8")
    )["external_publication_pending"] is False
    assert json.loads(
        (step_root / "step_transaction.json").read_text(encoding="utf-8")
    )["phase"] == "COMMITTED"


def test_faulted_trainer_refuses_retry(monkeypatch) -> None:
    trainer = Qwen35TTBTrainer(trainer_config())

    def fail_attempt(*_args, **_kwargs):
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(trainer, "_train_step_attempt", fail_attempt)
    with pytest.raises(RuntimeError, match="synthetic preflight failure"):
        trainer.train_step(_batch(), coordinate=step_coordinate())
    assert trainer._faulted is True
    with pytest.raises(RuntimeError, match="faulted"):
        trainer.train_step(_batch(), coordinate=step_coordinate())


class _FakeBoolean:
    def __init__(self, value: bool) -> None:
        self._value = value

    def all(self):
        return self

    def item(self) -> bool:
        return self._value


class _FakeGradient:
    def __init__(self, shape=(2, 2), *, finite: bool = True) -> None:
        self.shape = shape
        self._finite = finite
        self.added = 0

    def detach(self):
        return self

    def isfinite(self):
        return _FakeBoolean(self._finite)

    def to(self, _device):
        return self

    def add_(self, _other):
        self.added += 1
        return self


class _FakeParameter:
    def __init__(
        self,
        shape=(2, 2),
        *,
        grad: _FakeGradient | None = None,
        requires_grad: bool = True,
    ) -> None:
        self.shape = shape
        self.grad = grad
        self.requires_grad = requires_grad
        self.device = "cpu"


class _FakeModule:
    def __init__(self, parameters: dict[str, _FakeParameter]) -> None:
        self._parameters = parameters

    def named_parameters(self):
        return tuple(self._parameters.items())


def _fake_gradient_modules(*, missing: bool = False, finite: bool = True):
    learner_model = _FakeModule(
        {
            "layer.lora_A.theta.weight": _FakeParameter(
                grad=None if missing else _FakeGradient(finite=finite)
            ),
            "layer.lora_B.phi.weight": _FakeParameter(grad=_FakeGradient()),
        }
    )
    replica_model = _FakeModule(
        {
            "layer.lora_A.theta.weight": _FakeParameter(grad=_FakeGradient()),
            "layer.lora_B.phi.weight": _FakeParameter(grad=_FakeGradient()),
        }
    )
    learner_z = _FakeModule({"head.weight": _FakeParameter(grad=_FakeGradient())})
    replica_z = _FakeModule({"head.weight": _FakeParameter(grad=_FakeGradient())})
    return learner_model, replica_model, learner_z, replica_z


def test_replica_layout_rejects_missing_or_mismatched_parameters() -> None:
    learner, replica, learner_z, replica_z = _fake_gradient_modules()
    Qwen35TTBTrainer._validate_replica_layout(learner, replica, learner_z, replica_z)

    mismatched = _FakeModule(
        {"layer.lora_A.theta.weight": _FakeParameter(shape=(3, 2))}
    )
    with pytest.raises(RuntimeError, match="different LoRA layouts"):
        Qwen35TTBTrainer._validate_replica_layout(
            learner, mismatched, learner_z, replica_z
        )
    with pytest.raises(RuntimeError, match="different Z-head layouts"):
        Qwen35TTBTrainer._validate_replica_layout(
            learner,
            replica,
            learner_z,
            _FakeModule({}),
        )


@pytest.mark.parametrize(
    ("missing", "finite"),
    [(True, True), (False, False)],
)
def test_replica_gradient_merge_fails_closed_on_missing_or_non_finite_gradients(
    missing: bool, finite: bool
) -> None:
    modules = _fake_gradient_modules(missing=missing, finite=finite)

    with pytest.raises(RuntimeError, match="incomplete or non-finite"):
        Qwen35TTBTrainer._merge_replica_gradients(*modules)
