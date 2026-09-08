from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import yaml

from src.interactive.persistence import stable_id
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import (
    EvaluationReceipt,
    ExecutionRecord,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.step_transaction import StepPhase, StepTransaction


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_hotpotqa_grpo.py"
_SPEC = importlib.util.spec_from_file_location("train_hotpotqa_grpo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_CONFIG = _SCRIPT.parents[1] / "config" / "training_hotpotqa_grpo.yaml"
_DYNAMIC_CONFIG = (
    _SCRIPT.parents[1] / "config" / "training_hotpotqa_dynamic_ledger_grpo.yaml"
)

HotpotTrainingError = _MODULE.HotpotTrainingError
run_hotpotqa_training = _MODULE.run_hotpotqa_training
sample_hotpotqa_tasks = _MODULE.sample_hotpotqa_tasks
validate_hotpotqa_training_config = _MODULE.validate_hotpotqa_training_config
WandbTracker = _MODULE.WandbTracker
validate_phase0_rollout_batch = _MODULE.validate_phase0_rollout_batch


class FakeTracker:
    run_id = "wandb-test-run"
    run_url = "https://wandb.invalid/test"

    def __init__(self) -> None:
        self.logs: list[tuple[int, dict]] = []
        self.summary: dict = {}
        self.finished: list[int] = []
        self.artifacts: list[dict] = []

    def log(self, values, *, step: int) -> None:
        self.logs.append((step, dict(values)))

    def update_summary(self, values) -> None:
        self.summary.update(values)

    def log_checkpoint(self, checkpoint_dir, *, metadata, aliases):
        receipt = {
            "name": "hotpotqa-director-checkpoint",
            "version": f"v{len(self.artifacts)}",
            "aliases": list(aliases),
            "checkpoint_dir": str(checkpoint_dir),
            "metadata": dict(metadata),
        }
        self.artifacts.append(receipt)
        return receipt

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class FakeSummary:
    def __init__(self, values: dict) -> None:
        self.values = values

    def to_dict(self) -> dict:
        return dict(self.values)


class FakeReceipt:
    def __init__(self, values: dict) -> None:
        self.values = values

    def to_dict(self) -> dict:
        return dict(self.values)


def _trajectory(task, rollout_index, versions, *, adapter: str) -> TrajectoryRecord:
    graph: dict = {}
    snapshot_id = stable_id(
        "snapshot",
        {"revision": 0, "graph": graph, "previous_snapshot_id": None},
    )
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}:{rollout_index}",
        round_index=0,
        prompt="ordinary prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=(1,),
        output_token_ids=(2,),
        behavior_log_probs=(-0.1,),
        executed_prefix_tokens=1,
        action={"action": "finish"},
        canvas_feedback="workflow finished",
        graph_revision=0,
        graph_snapshot=graph,
        graph_snapshot_id=snapshot_id,
        previous_graph_snapshot_id=None,
        policy_version=versions.policy,
        policy_adapter=adapter,
        server_weight_version="default",
        receipt_verified=True,
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{rollout_index}:{versions.policy}",
        task=task,
        group_id=f"{task.task_id}:hotpotqa_grpo_natural_v1:{versions.policy}",
        condition_id="hotpotqa_grpo_natural_v1",
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=reward,
            metrics={"exact_match": reward, "token_f1": reward},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )


def _phase0_trajectory(task, rollout_index, versions, *, adapter: str) -> TrajectoryRecord:
    graph = AgentGraph()
    graph.add_agent(AgentNode("solver", "worker-model", "Answer the question."))
    first_snapshot = GraphSnapshotEvent.create(1, graph.to_dict())
    graph.set_output("solver")
    second_snapshot = GraphSnapshotEvent.create(
        2,
        graph.to_dict(),
        first_snapshot.snapshot_id,
    )
    third_snapshot = GraphSnapshotEvent.create(
        2,
        graph.to_dict(),
        second_snapshot.snapshot_id,
    )
    run_id = f"run:{task.task_id}:{rollout_index}"
    runtime = {
        "run_id": run_id,
        "graph_revision": 2,
        "output_agent_id": "solver",
        "final_answer": "answer",
        "outputs": {"solver": "answer"},
        "block_completion_order": [["solver"]],
    }
    execution = ExecutionRecord(
        execution_id=f"execution:{task.task_id}:{rollout_index}",
        experiment_id=run_id,
        graph_revision=2,
        agent_id="solver",
        model_id="worker-model",
        model_fingerprint="model-fingerprint",
        provider="provider",
        request_hash="request-receipt",
        output="answer",
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        metadata={
            "request": {
                "request_id": f"agent-request:{task.task_id}:{rollout_index}",
                "run_id": run_id,
                "graph_revision": 2,
                "problem": task.question,
                "agent": {
                    "id": "solver",
                    "model_id": "worker-model",
                    "contract": "Answer the question.",
                },
                "model": {"model_id": "worker-model"},
                "provider_id": "provider",
                "upstream": [],
            },
            "response": {
                "provider_request_id": (
                    f"provider-request:{task.task_id}:{rollout_index}"
                ),
                "attempt_count": 1,
            },
        },
    )

    def turn(
        index: int,
        response: str,
        action: dict,
        snapshot: GraphSnapshotEvent,
        feedback: str,
        *,
        executions=(),
        runtime_summary=None,
        reused=False,
    ) -> TurnRecord:
        return TurnRecord(
            turn_id=f"turn:{task.task_id}:{rollout_index}:{index}",
            round_index=index,
            prompt=f"prompt {index}",
            policy_response=response,
            prompt_token_ids=(1,),
            output_token_ids=(2,),
            behavior_log_probs=(-0.1,),
            executed_prefix_tokens=1,
            action=action,
            canvas_feedback=feedback,
            graph_revision=snapshot.revision,
            graph_snapshot=snapshot.to_dict()["graph"],
            graph_snapshot_id=snapshot.snapshot_id,
            previous_graph_snapshot_id=snapshot.previous_snapshot_id,
            executions=executions,
            runtime_summary=runtime_summary or {},
            execution_reused=reused,
            director_request_id=(
                f"director-request:{task.task_id}:{rollout_index}:{index}"
            ),
            director_attempt_count=1,
            policy_version=versions.policy,
            policy_adapter=adapter,
            server_weight_version="default",
            receipt_verified=True,
        )

    turns = (
        turn(
            0,
            '{"action":"add_agent","agent_id":"solver","model_id":"worker-model","contract":"Answer the question."}',
            {
                "action": "add_agent",
                "agent_id": "solver",
                "model_id": "worker-model",
                "contract": "Answer the question.",
            },
            first_snapshot,
            "accepted add_agent at revision 1",
        ),
        turn(
            1,
            '{"action":"set_output","agent_id":"solver"}',
            {"action": "set_output", "agent_id": "solver"},
            second_snapshot,
            "accepted set_output at revision 2",
            executions=(execution,),
            runtime_summary=runtime,
        ),
        turn(
            2,
            '{"action":"finish"}',
            {"action": "finish"},
            third_snapshot,
            "workflow finished",
            runtime_summary=runtime,
            reused=True,
        ),
    )
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{rollout_index}:{versions.policy}",
        task=task,
        group_id=f"{task.task_id}:hotpotqa_grpo_natural_v1:{versions.policy}",
        condition_id="hotpotqa_grpo_natural_v1",
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=turns,
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=1.0,
            metrics={"exact_match": 1.0, "token_f1": 1.0},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )
class FakeBackend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, config: dict, events: list[str]) -> None:
        self.config = config
        self.events = events
        self.training_step = int(config["experiment"]["training_step"])
        self.absolute_step = int(config["experiment"]["update_step"])
        self.active_policy = str(config["director"]["behavior_policy_version"])
        self.active_adapter = str(config["director"]["behavior_adapter_name"])

    async def ensure_behavior_ready(self):
        self.events.append(f"ready:{self.training_step}:{self.active_policy}")
        return {"success": True, "status": "ready"}

    async def collect(
        self,
        task,
        rollout_index,
        versions,
        *,
        expected_task_split="train",
    ):
        assert expected_task_split == task.split
        if expected_task_split == "validation":
            kind = "validation"
        else:
            kind = "canary" if rollout_index >= 1_000_000 else "rollout"
        self.events.append(
            f"collect:{self.training_step}:{kind}:{versions.policy}:{self.active_adapter}"
        )
        return _trajectory(
            task,
            rollout_index,
            versions,
            adapter=self.active_adapter,
        )

    def train(self, trajectories, output_dir):
        self.events.append(f"train:{self.training_step}")
        checkpoint = Path(output_dir) / "checkpoint_final" / "supervisor_lora" / "theta"
        checkpoint.mkdir(parents=True)
        training_state = checkpoint / "training_state.pt"
        training_state.write_bytes(b"state")
        values = {
            "optimizer_updates": 1,
            "input_trajectories": len(trajectories),
            "informative_groups": 7,
            "trained_trajectories": len(trajectories),
            "loss": 0.25 / self.training_step,
            "grad_norm": 1.0,
            "trainable_update_l2": 0.1,
            "behavior_policy_version": self.active_policy,
            "updated_policy_version": self.config["director"][
                "updated_policy_version"
            ],
            "micro_batch_size_used": 4,
            "oom_backoff_count": 0,
            "checkpoint_dir": str(checkpoint),
            "optimizer_state_checkpoint": str(training_state),
            "optimizer_state_saved": True,
            "training_state_checkpoint": str(training_state),
            "training_state_saved": True,
            "scheduler_state_saved": True,
            "rng_state_saved": True,
            "checkpoint_recoverable": True,
            "scheduler_resume_status": "restored_scheduler_and_rng",
            "learning_rate": 1.0e-4,
            "gpu_memory_allocated_mib": {"cuda:3": 1024.0, "cuda:5": 1024.0},
            "committed_step": self.absolute_step,
        }
        return FakeSummary(values)

    async def publish(self, summary):
        self.events.append(f"publish:{self.training_step}")
        candidate = summary.to_dict()["updated_policy_version"]
        adapter = (
            self.config["policy_sync"]["adapter_name_prefix"]
            + f"{self.absolute_step:06d}"
        )
        previous = self.active_policy
        self.active_policy = candidate
        self.active_adapter = adapter
        return FakeReceipt(
            {
                "success": True,
                "status": "published",
                "behavior_policy_version": previous,
                "candidate_policy_version": candidate,
                "new_policy_version": candidate,
                "adapter_name": adapter,
                "checkpoint_version": f"checkpoint:{candidate}",
                "route_switch_success": True,
            }
        )


def _task(
    index: int,
    *,
    source: str = "hotpotqa",
    split: str = "train",
) -> TaskRecord:
    return TaskRecord(
        task_id=f"{source}:{split}:{index}",
        question=f"Question {source} {index}?",
        ground_truth="answer",
        split=split,
        metadata={
            "dataset_key": source,
            "source": "HotpotQA" if source == "hotpotqa" else "TriviaQA",
            "sampling": {"base_task_id": f"{source}:{split}:base:{index}"},
        },
    )


def _write_tasks(path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(512):
            task = _task(index)
            stream.write(
                json.dumps(
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                )
                + "\n"
            )
        for index in range(2):
            task = _task(index, source="triviaqa")
            stream.write(
                json.dumps(
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                )
                + "\n"
            )


def _write_validation_tasks(path: Path) -> list[str]:
    task_ids = []
    with path.open("w", encoding="utf-8") as stream:
        for index in range(128):
            task = _task(index, split="validation")
            task_ids.append(task.task_id)
            stream.write(
                json.dumps(
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                )
                + "\n"
            )
    return task_ids


def _create_project(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    config["data"]["train_path"] = "data/train.jsonl"
    config["data"]["validation_path"] = "data/validation.jsonl"
    config["experiment"]["output_dir"] = "artifacts/training"
    config["experiment"]["training_enabled"] = True
    config["gpu"]["training_enabled"] = True
    config["tracking"]["validation_protocol"]["status"] = "frozen"
    config["tracking"]["validation_protocol"]["monitor_task_ids"] = [
        f"hotpotqa:validation:{index}" for index in range(7)
    ]
    config["tracking"]["checkpoint_artifact"]["status"] = "ready"
    config["md_compliance"].update(
        phase_0_status="passed",
        real_step_authorized=True,
        one_step_closure_status="passed",
        long_training_authorized=True,
    )
    config["storage"].update(
        root="artifacts/training/evidence",
        manifest_path="artifacts/training/run_manifest.json",
        state_path="artifacts/training/run_state.json",
        training_log_path="artifacts/training/training_log.jsonl",
    )
    config_path = root / "config" / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_tasks(root / "data" / "train.jsonl")
    _write_validation_tasks(root / "data" / "validation.jsonl")
    return config_path


class ConfigAndSamplingTests(unittest.TestCase):
    def test_dynamic_evidence_root_defaults_to_existing_directory(self) -> None:
        for environment in ({}, {"HOTPOTQA_EVIDENCE_ROOT": ""}):
            with (
                self.subTest(environment=environment),
                patch.dict(_MODULE.os.environ, environment, clear=True),
            ):
                config = _MODULE.load_yaml(_DYNAMIC_CONFIG)
                self.assertEqual(
                    "artifacts/hotpotqa_dynamic_ledger_grpo_300step/evidence",
                    config["storage"]["root"],
                )

    def test_dynamic_evidence_root_override_changes_only_storage_root(self) -> None:
        evidence_root = (
            "artifacts/hotpotqa_dynamic_ledger_grpo_300step/evidence_proxy_resume"
        )
        with patch.dict(_MODULE.os.environ, {}, clear=True):
            default_config = _MODULE.load_yaml(_DYNAMIC_CONFIG)
            with patch.dict(
                _MODULE.os.environ, {"HOTPOTQA_EVIDENCE_ROOT": evidence_root}
            ):
                isolated_config = _MODULE.load_yaml(_DYNAMIC_CONFIG)

        self.assertEqual(evidence_root, isolated_config["storage"]["root"])
        isolated_config["storage"]["root"] = default_config["storage"]["root"]
        self.assertEqual(
            default_config,
            isolated_config,
            "Evidence isolation must not change any sampling, model, policy, "
            "GRPO, seed, checkpoint, or other configuration field",
        )

    def test_config_fixes_grpo_and_disables_other_learning_flows(self) -> None:
        config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
        validate_hotpotqa_training_config(config)
        self.assertEqual("WANDB_BINDING_20260906_V1", config["tracking"]["binding_marker"])
        self.assertEqual("zhanghanwen6660909-dut", config["tracking"]["entity"])
        self.assertEqual("flowsteer-hotpotqa", config["tracking"]["project"])
        config["grpo"]["ttb_enabled"] = True
        with self.assertRaisesRegex(Exception, "TTB"):
            validate_hotpotqa_training_config(config)

    def test_wandb_uses_sdk_default_credentials_and_requires_real_url(self) -> None:
        config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
        captured: dict = {}
        logged_artifact = SimpleNamespace(
            name="hotpotqa-director-checkpoint:v0",
            version="v0",
            wait=MagicMock(),
        )
        artifact_calls: list[dict] = []
        run = SimpleNamespace(
            id="run-test",
            url="https://wandb.invalid/run-test",
            summary=SimpleNamespace(update=lambda values: None),
            log=lambda values, step, commit: None,
            log_artifact=lambda artifact, aliases: (
                artifact_calls.append(
                    {"artifact": artifact, "aliases": list(aliases)}
                )
                or logged_artifact
            ),
            finish=lambda exit_code: None,
        )
        module = ModuleType("wandb")

        class Artifact:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.directories = []

            def add_dir(self, path, name):
                self.directories.append((path, name))

        def init(**kwargs):
            captured.update(kwargs)
            return run

        module.init = init  # type: ignore[attr-defined]
        module.Artifact = Artifact  # type: ignore[attr-defined]
        with (
            patch.dict(sys.modules, {"wandb": module}),
            patch.dict(_MODULE.os.environ, {"WANDB_API_KEY": ""}),
        ):
            tracker = WandbTracker(config)
        self.assertEqual("run-test", tracker.run_id)
        self.assertEqual("https://wandb.invalid/run-test", tracker.run_url)
        self.assertEqual("zhanghanwen6660909-dut", captured["entity"])
        self.assertEqual("flowsteer-hotpotqa", captured["project"])
        self.assertEqual("online", captured["mode"])
        tracker.log(
            {
                name: 0.0
                for name in config["tracking"]["required_step_fields"]
            },
            step=1,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            receipt = tracker.log_checkpoint(
                Path(directory),
                metadata={"global_step": 1},
                aliases=["latest", "best"],
            )
        self.assertEqual("v0", receipt["version"])
        self.assertEqual(["latest", "best"], artifact_calls[0]["aliases"])
        logged_artifact.wait.assert_called_once_with()

    def test_skillflow_seeded_sample_is_repeatable_and_unique(self) -> None:
        pool = tuple(_task(index) for index in range(10))
        first = sample_hotpotqa_tasks(
            pool,
            optimizer_step=3,
            tasks_per_step=7,
            seed=42,
            seed_offset=0,
        )
        second = sample_hotpotqa_tasks(
            pool,
            optimizer_step=3,
            tasks_per_step=7,
            seed=42,
            seed_offset=0,
        )
        self.assertEqual([item.task_id for item in first], [item.task_id for item in second])
        self.assertEqual(7, len({_MODULE._base_task_id(item) for item in first}))

    def test_checked_in_md_config_is_fail_closed_before_runtime_construction(self) -> None:
        def forbidden_backend(config, project_root):
            raise AssertionError("backend must not be constructed")

        import asyncio

        with self.assertRaisesRegex(HotpotTrainingError, "compliance gate"):
            asyncio.run(
                run_hotpotqa_training(
                    _CONFIG,
                    project_root=_CONFIG.parents[1],
                    allow_md_grpo=True,
                    stop_after_optimizer_steps=1,
                    backend_factory=forbidden_backend,
                    tracker=FakeTracker(),
                )
            )

    def test_phase0_gate_accepts_exact_fresh_lineage_and_rejects_condition_drift(self) -> None:
        import asyncio

        tasks = [_task(index) for index in range(7)]
        trajectories = []
        rollout_index = 0
        for task in tasks:
            versions = _MODULE.version_bundle_for(
                task,
                policy_version="policy-v1",
                model_catalog_version="catalog-v1",
                prompt_version="prompt-v1",
                tool_version="tool-v1",
            )
            for _ in range(4):
                trajectories.append(
                    _phase0_trajectory(
                        task,
                        rollout_index,
                        versions,
                        adapter="theta-v1",
                    )
                )
                rollout_index += 1

        receipt = asyncio.run(
            validate_phase0_rollout_batch(
                trajectories,
                expected_task_ids=[task.task_id for task in tasks],
                expected_count=28,
                behavior_policy="policy-v1",
                behavior_adapter="theta-v1",
            )
        )
        self.assertEqual("passed", receipt["status"])
        self.assertEqual(28, receipt["evaluator_replay_count"])

        trajectories[0] = replace(trajectories[0], condition_satisfied=False)
        with self.assertRaisesRegex(HotpotTrainingError, "non-GRPO-eligible"):
            asyncio.run(
                validate_phase0_rollout_batch(
                    trajectories,
                    expected_task_ids=[task.task_id for task in tasks],
                    expected_count=28,
                    behavior_policy="policy-v1",
                    behavior_adapter="theta-v1",
                )
            )

    def test_phase0_gate_accepts_complete_agent_retry_lineage_and_rejects_gaps(self) -> None:
        import asyncio

        tasks = [_task(index) for index in range(7)]
        trajectories = []
        rollout_index = 0
        for task in tasks:
            versions = _MODULE.version_bundle_for(
                task,
                policy_version="policy-v1",
                model_catalog_version="catalog-v1",
                prompt_version="prompt-v1",
                tool_version="tool-v1",
            )
            for _ in range(4):
                trajectories.append(
                    _phase0_trajectory(
                        task,
                        rollout_index,
                        versions,
                        adapter="theta-v1",
                    )
                )
                rollout_index += 1

        original = trajectories[0]
        turns = list(original.turns)
        execution = turns[1].executions[0]
        metadata = dict(execution.metadata)
        response = dict(metadata["response"])
        request_id = metadata["request"]["request_id"]
        provider_request_id = response["provider_request_id"]
        response.update(
            {
                "attempt_count": 2,
                "retry_receipts": [
                    {
                        "attempt": 1,
                        "request_id": request_id,
                        "provider_id": execution.provider,
                        "model_id": execution.model_id,
                        "status": "retryable_failure",
                        "error_type": "HTTPError",
                        "http_status": 429,
                        "retryable": True,
                        "backoff_seconds": 1.0,
                        "latency_ms": 2.0,
                    },
                    {
                        "attempt": 2,
                        "request_id": request_id,
                        "provider_id": execution.provider,
                        "model_id": execution.model_id,
                        "provider_request_id": provider_request_id,
                        "status": "completed",
                        "http_status": 200,
                        "retryable": False,
                        "backoff_seconds": 0.0,
                        "latency_ms": 3.0,
                    },
                ],
            }
        )
        metadata["response"] = response
        turns[1] = replace(
            turns[1],
            executions=(replace(execution, metadata=metadata),),
        )
        trajectories[0] = replace(original, turns=tuple(turns))

        receipt = asyncio.run(
            validate_phase0_rollout_batch(
                trajectories,
                expected_task_ids=[task.task_id for task in tasks],
                expected_count=28,
                behavior_policy="policy-v1",
                behavior_adapter="theta-v1",
            )
        )
        self.assertEqual(1, receipt["agent_provider_retry_count"])

        broken_turns = list(trajectories[0].turns)
        broken_execution = broken_turns[1].executions[0]
        broken_metadata = dict(broken_execution.metadata)
        broken_response = dict(broken_metadata["response"])
        broken_response["retry_receipts"] = broken_response["retry_receipts"][1:]
        broken_metadata["response"] = broken_response
        broken_turns[1] = replace(
            broken_turns[1],
            executions=(replace(broken_execution, metadata=broken_metadata),),
        )
        broken = list(trajectories)
        broken[0] = replace(trajectories[0], turns=tuple(broken_turns))
        with self.assertRaisesRegex(
            HotpotTrainingError,
            "retry count differs from lineage",
        ):
            asyncio.run(
                validate_phase0_rollout_batch(
                    broken,
                    expected_task_ids=[task.task_id for task in tasks],
                    expected_count=28,
                    behavior_policy="policy-v1",
                    behavior_adapter="theta-v1",
                )
            )


class StepTransactionTests(unittest.TestCase):
    def test_journal_and_recovery_state_are_durable_and_clearable(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            transaction = StepTransaction(directory)
            transaction.transition(step=1, phase=StepPhase.PREPARED, policy="v0")
            transaction.seal(step=1, state={"phase": "rollout_complete"})
            self.assertEqual(1, transaction.load()["step"])
            journal = [
                json.loads(line)
                for line in transaction.journal.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("prepared", journal[0]["phase"])
            transaction.clear()
            self.assertIsNone(transaction.load())


class SequentialRunnerTests(unittest.TestCase):
    def test_two_step_boundary_has_no_cross_step_prefetch(self) -> None:
        import tempfile

        async def inline_to_thread(function, /, *args, **kwargs):
            return function(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _create_project(root)
            events: list[str] = []
            tracker = FakeTracker()

            def factory(config, project_root):
                self.assertEqual(root, project_root)
                return FakeBackend(config, events)

            # Keep production fsync semantics while avoiding the workspace
            # filesystem's sync latency in this dependency-light unit test.
            with (
                patch.object(_MODULE.os, "fsync"),
                patch.object(_MODULE.asyncio, "to_thread", new=inline_to_thread),
            ):
                import asyncio

                manifest = asyncio.run(
                    run_hotpotqa_training(
                        config_path,
                        project_root=root,
                        allow_md_grpo=True,
                        stop_after_optimizer_steps=2,
                        backend_factory=factory,
                        tracker=tracker,
                    )
                )

            self.assertEqual("paused_at_requested_boundary", manifest["status"])
            self.assertEqual(2, manifest["optimizer_updates_completed"])
            self.assertEqual([1, 2], [step for step, _ in tracker.logs])
            self.assertEqual([0], tracker.finished)
            for _, values in tracker.logs:
                self.assertGreater(values["train/grad_norm"], 0)
                self.assertGreater(values["train/lora_update_l2"], 0)
                self.assertTrue(values["policy/sync_success"])
                self.assertTrue(values["policy/canary_success"])
                self.assertEqual(7, values["validation/valid_count"])
                self.assertTrue(values["checkpoint/saved"])
            self.assertEqual(2, len(tracker.artifacts))
            self.assertEqual(["latest", "best"], tracker.artifacts[0]["aliases"])
            self.assertEqual(["latest"], tracker.artifacts[1]["aliases"])

            first_publish = events.index("publish:1")
            first_canary = next(
                index
                for index, event in enumerate(events)
                if event.startswith("collect:1:canary:")
            )
            second_rollout = next(
                index
                for index, event in enumerate(events)
                if event.startswith("collect:2:rollout:")
            )
            self.assertLess(first_publish, first_canary)
            self.assertLess(first_canary, second_rollout)
            first_validation = next(
                index
                for index, event in enumerate(events)
                if event.startswith("collect:1:validation:")
            )
            self.assertLess(first_canary, first_validation)
            self.assertLess(first_validation, second_rollout)
            self.assertEqual(
                28,
                sum(event.startswith("collect:1:rollout:") for event in events),
            )
            self.assertEqual(
                28,
                sum(event.startswith("collect:2:rollout:") for event in events),
            )

            state = json.loads(
                (root / "artifacts/training/run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, state["optimizer_updates_completed"])
            self.assertEqual(
                "qwen35-9b-hotpotqa-grpo-step-000002",
                state["behavior_policy_version"],
            )
            self.assertTrue(Path(state["optimizer_state_checkpoint"]).is_file())
            journal = [
                json.loads(line)
                for line in (
                    root / "artifacts/training/recovery/step_journal.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            expected = [phase.value for phase in StepPhase]
            for step in (1, 2):
                self.assertEqual(
                    expected,
                    [item["phase"] for item in journal if item["step"] == step],
                )

    def test_prepare_only_does_not_construct_backend_or_tracker(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _create_project(root)

            def forbidden_backend(config, project_root):
                raise AssertionError("backend must not be constructed")

            with patch.object(_MODULE.os, "fsync"):
                import asyncio

                manifest = asyncio.run(
                    run_hotpotqa_training(
                        config_path,
                        project_root=root,
                        prepare_only=True,
                        backend_factory=forbidden_backend,
                    )
                )
            self.assertEqual("prepared", manifest["status"])
            self.assertEqual(0, manifest["optimizer_updates_completed"])


if __name__ == "__main__":
    unittest.main()
