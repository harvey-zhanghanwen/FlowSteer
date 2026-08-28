from __future__ import annotations

import asyncio
import json
import importlib.util
import os
import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentResponse, AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.aime2026_adapter import AIME2026_EVALUATOR_VERSION
from src.interactive.computation_tools import (
    AIME_CALCULATOR_TOOL_ID,
    AIME_PYTHON_EXEC_TOOL_ID,
)
from src.interactive.coding_tools import SWEBENCH_REPOSITORY_TOOL_ID
from src.interactive.config_loader import ConfigurationError, load_model_registry
from src.interactive.director import (
    AgentGraphOrchestrator,
    HOTPOTQA_DIRECTOR_PROMPT_VERSION,
    decode_director_transcript,
)
from src.interactive.healthbench_tool_adapter import (
    FrozenMedRAGBM25Corpus,
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
    build_healthbench_medrag_tool_registry,
)
from src.interactive.persistence import stable_id
from src.interactive.hotpot_training_schedule import (
    HotpotTrainingCursorState,
    freeze_hotpot_training_schedule,
)
from src.interactive.joint_qa_training_schedule import (
    JointQATrainingCursorState,
    freeze_joint_qa_training_schedule,
)
from src.interactive.qa_retrieval import QARetrievalReceipt, build_keyword_query
from src.interactive.qa_tool_adapter import build_qa_tool_registry
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.rollout_collector import (
    _runtime_summary,
    execution_record_from_call,
)
from src.interactive.scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.skills import (
    SkillEvidence,
    SkillLifecycleManager,
    SkillRecord,
    SkillStatus,
    SkillStore,
)
from src.interactive.versioning import VersionBundle


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_agentgraph_smoke.py"
_SPEC = importlib.util.spec_from_file_location("train_agentgraph_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

EXPECTED_SOURCE_ORDER = _MODULE.EXPECTED_SOURCE_ORDER
SmokeRunError = _MODULE.SmokeRunError
evaluator_version_for = _MODULE.evaluator_version_for
run_smoke = _MODULE.run_smoke
select_smoke_tasks = _MODULE.select_smoke_tasks
validate_smoke_bounds = _MODULE.validate_smoke_bounds
validate_resumed_initial_rollouts = _MODULE._validate_resumed_initial_rollouts
audit_active_skills_after_policy_update = (
    _MODULE._audit_active_skills_after_policy_update
)
workflow_problem = _MODULE._workflow_problem
qa_tool_runtime_settings = _MODULE._qa_tool_runtime_settings
aime_tool_runtime_settings = _MODULE._aime_tool_runtime_settings
healthbench_tool_runtime_settings = _MODULE._healthbench_tool_runtime_settings
environment_runtime_settings = _MODULE._environment_runtime_settings
swe_coding_runtime_settings = _MODULE._swe_coding_runtime_settings
environment_replay_trace_from_runtime = (
    _MODULE._environment_replay_trace_from_runtime
)
requires_format_agent = _MODULE._requires_format_agent


SOURCE_NAMES = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "aime_2026": "AIME 2026",
    "healthbench_professional": "HealthBench Professional",
    "webshop": "WebShop",
    "alfworld": "ALFWorld",
    "swe_bench": "SWE-bench",
}


def make_task(source: str, index: int, *, base_id: str | None = None) -> TaskRecord:
    task_id = f"{source}:{index}"
    return TaskRecord(
        task_id=task_id,
        question=f"Question {source} {index}?",
        ground_truth="answer",
        split="train",
        metadata={
            "dataset_key": source,
            "source": SOURCE_NAMES[source],
            "sampling": {"base_task_id": base_id or task_id},
        },
    )


def aligned_row(task: TaskRecord) -> dict:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        **task.to_dict(),
    }


def test_live_backend_rejects_invalid_model_admissible_sampling_contract():
    root = Path(__file__).resolve().parents[2]
    source = yaml.safe_load(
        (root / "config/evaluation_hotpotqa_unified_architecture_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["execution_timeout"] = 37.0
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: object()
        )
    )
    invalid_cases = (
        (
            {
                "action_decoding": "unconstrained",
                "sampling_action_profile": "model_admissible_canvas_actions",
                "sampling_schema_version": (
                    "agentgraph.model-admissible-action-mask.v2"
                ),
            },
            "requires evaluation-only json_schema",
        ),
        (
            {
                "action_decoding": "json_schema",
                "sampling_action_profile": "model_admissible_canvas_actions",
                "sampling_schema_version": "wrong-schema",
            },
            "model-admissible Director sampling requires",
        ),
    )
    for director_values, expected_message in invalid_cases:
        config = copy.deepcopy(source)
        config["director"].update(director_values)
        with patch.dict(
            os.environ,
            {"VECTOR_ENGINE_API_KEY": "unit-test-placeholder"},
            clear=False,
        ), patch.dict(sys.modules, {"transformers": fake_transformers}):
            with unittest.TestCase().assertRaisesRegex(
                ConfigurationError,
                expected_message,
            ):
                _MODULE.LiveSmokeBackend.from_config(
                    config,
                    root,
                    evaluation_only=True,
                )


def test_live_backend_requires_and_wires_explicit_execution_timeout():
    root = Path(__file__).resolve().parents[2]
    source = yaml.safe_load(
        (root / "config/evaluation_hotpotqa_unified_architecture_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    for invalid_value in (None, 0, -1.0, True, "37", float("inf"), float("nan")):
        config = copy.deepcopy(source)
        config["execution_timeout"] = invalid_value
        with unittest.TestCase().assertRaisesRegex(
            ConfigurationError,
            "execution_timeout must be an explicit positive finite number",
        ):
            _MODULE.LiveSmokeBackend.from_config(
                config,
                root,
                evaluation_only=True,
            )

    config = copy.deepcopy(source)
    config["execution_timeout"] = 37.0
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: object()
        )
    )
    with patch.dict(
        os.environ,
        {"VECTOR_ENGINE_API_KEY": "unit-test-placeholder"},
        clear=False,
    ), patch.dict(sys.modules, {"transformers": fake_transformers}), patch.object(
        _MODULE,
        "SGLangReceiptDirectorClient",
        return_value=object(),
    ):
        backend = _MODULE.LiveSmokeBackend.from_config(
            config,
            root,
            evaluation_only=True,
        )

    assert backend.runtime.timeout_seconds == 37.0
    assert backend.runtime.gateway.timeout_seconds == 37.0


def test_interactive_workflow_problem_exposes_only_the_execution_contract():
    task = make_task("webshop", 0)
    config = {
        "webshop_evaluation": {
            "direct_contract": "Return exactly one admissible WebShop action."
        }
    }

    value = workflow_problem(task, config)

    assert value.startswith(task.question + "\n\nExecution interface:")
    assert "invoked once per environment step" in value
    assert "current observation" in value
    assert "Return exactly one admissible WebShop action." in value
    assert "topology" not in value.lower()
    assert "skill" not in value.lower()


def test_explicit_environment_runtime_exposes_its_episode_contract():
    task = make_task("webshop", 0)
    config = {
        "experiment": {"condition_id": "webshop_ragen_react_stable_zero"},
        "evaluation": {
            "max_environment_steps_by_source": {"webshop": 3},
        },
        "environment_runtime": {
            "enabled": True,
            "condition_id": "webshop_ragen_react_stable_zero",
            "mode": "model_driven_ragen_react",
            "dataset_scope": ["webshop"],
            "ragen_adapter_path": "vendor/SkillFlow/src/ragen_adapter.py",
            "tool_timeout_seconds": 9.0,
            "max_environment_steps_by_source": {"webshop": 3},
        },
    }
    config["webshop_evaluation"] = {
        "direct_contract": "Return exactly one executable WebShop action."
    }

    value = workflow_problem(task, config)

    assert "execution_mode `react`" in value
    assert "`webshop.environment`" in value
    assert "request-scoped episode" in value
    assert "terminal state or the fixed evaluator step budget" in value
    assert "same graph is invoked once per environment step" not in value
    assert "reward" not in value.lower()
    assert "evaluator info" not in value.lower()


def test_static_workflow_problem_remains_the_immutable_question():
    task = make_task("hotpotqa", 0)
    assert workflow_problem(task, {}) == task.question


def trajectory(task: TaskRecord, rollout_index: int, versions) -> TrajectoryRecord:
    graph = {}
    snapshot_id = stable_id(
        "snapshot",
        {"revision": 0, "graph": graph, "previous_snapshot_id": None},
    )
    base_seed = 42
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(
            base_seed=base_seed
        ),
        schedule_purpose="natural_smoke",
        ordered_sequence_hash=stable_hash([task.task_id]),
        sequence_position=rollout_index,
        task_id=task.task_id,
        optimizer_step_or_anchor_ordinal=0,
    )
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}:{rollout_index}",
        round_index=0,
        prompt="prompt",
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
        receipt_verified=True,
        server_weight_version="default",
        policy_adapter=(
            "theta_smoke_step_000001" if rollout_index >= 10_000 else None
        ),
        director_generation_seed=derive_generation_seed(
            base_seed=base_seed,
            coordinate=coordinate,
            step_index=1,
            phase=GenerationPhase.ACTION,
        ),
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{versions.policy}:{rollout_index}",
        task=task,
        group_id=f"{task.task_id}:natural_smoke:{versions.policy}",
        condition_id="natural_smoke",
        rollout_id=(
            f"{task.task_id}:natural_smoke:{versions.policy}:"
            f"rollout:{rollout_index:04d}"
        ),
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=reward,
            metrics={"score": reward},
        ),
        termination_reason="finish",
        explicit_finish=True,
        director_sampling={
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": GenerationPhase.ACTION.value,
        },
    )


class Summary:
    def __init__(self, checkpoint: Path, updates: int = 1) -> None:
        self.optimizer_updates = updates
        self.behavior_policy_version = "qwen35-9b-base-step-0000"
        self.updated_policy_version = (
            "qwen35-9b-smoke-step-0001" if updates else ""
        )
        self.checkpoint_dir = str(checkpoint) if updates else ""

    def to_dict(self):
        return {
            "optimizer_updates": self.optimizer_updates,
            "behavior_policy_version": self.behavior_policy_version,
            "updated_policy_version": self.updated_policy_version,
            "checkpoint_dir": self.checkpoint_dir,
            "optimizer_state_saved": False,
            "trainable_update_l2": 1.0 if self.optimizer_updates else 0.0,
        }


class Receipt:
    adapter_name = "theta_smoke_step_000001"
    new_policy_version = "qwen35-9b-smoke-step-0001"

    def to_dict(self):
        return {
            "success": True,
            "status": "published",
            "adapter_name": self.adapter_name,
            "behavior_policy_version": "qwen35-9b-base-step-0000",
            "candidate_policy_version": self.new_policy_version,
            "new_policy_version": self.new_policy_version,
        }


class FakeBackend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, *, updates: int = 1) -> None:
        self.updates = updates
        self.events: list[str] = []
        self.train_inputs = []
        self.publish_summary = None

    async def collect(self, task, rollout_index, versions):
        self.events.append(f"collect:{rollout_index}")
        return trajectory(task, rollout_index, versions)

    def train(self, trajectories, output_dir):
        self.events.append("train")
        self.train_inputs = list(trajectories)
        checkpoint = output_dir / "checkpoint_final" / "supervisor_lora"
        if self.updates:
            checkpoint.mkdir(parents=True, exist_ok=True)
        summary = Summary(checkpoint, self.updates)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary.to_dict()) + "\n",
            encoding="utf-8",
        )
        return summary

    async def publish(self, summary):
        self.events.append("publish")
        self.publish_summary = summary
        return Receipt()


class QAToolRuntimeWiringTests(unittest.TestCase):
    class _Gateway:
        async def generate(self, request):  # pragma: no cover - no model call
            raise AssertionError(f"unexpected model call: {request.request_id}")

    class _Index:
        manifest = SimpleNamespace(
            corpus_name="public-test-corpus",
            corpus_version="v1",
            index_id="public-test-index-v1",
            format="sqlite",
            retrieval_backend="skillflow-test",
        )

        def search(self, query, *, limit):  # pragma: no cover - prompt only
            del query, limit
            return ()

        def read(self, passage_id):  # pragma: no cover - prompt only
            raise KeyError(passage_id)

        def close(self):
            return None

    def test_exact_answer_syntax_does_not_require_a_format_agent(self) -> None:
        self.assertFalse(
            requires_format_agent(
                {"require_format_agent": False},
                terminal_protocol="exact_single_answer_tag",
            )
        )
        self.assertTrue(
            requires_format_agent(
                {},
                terminal_protocol="exact_single_answer_tag",
            )
        )
        self.assertTrue(
            requires_format_agent(
                {"require_format_agent": True},
                terminal_protocol="none",
            )
        )
        with self.assertRaises(ConfigurationError):
            requires_format_agent(
                {"require_format_agent": "false"},
                terminal_protocol="exact_single_answer_tag",
            )

    @staticmethod
    def _config(
        *,
        enabled: bool = True,
        passage_source: str = "external_corpus",
    ) -> dict:
        config = {
            "experiment": {"condition_id": "hotpotqa_tool_react_stable_zero"},
        }
        if enabled:
            config["qa_tool_runtime"] = {
                "enabled": True,
                "condition_id": "hotpotqa_tool_react_stable_zero",
                "mode": "model_driven_search_read",
                "dataset_scope": ["hotpotqa"],
                "skillflow_source": "vendor/SkillFlow/src",
                "index_path": "data/public-retrieval.sqlite3",
                "passage_source": passage_source,
                "tool_timeout_seconds": 7.0,
                "max_turns_per_agent_call": 5,
                "max_tool_calls_per_agent_call": 3,
            }
        return config

    def test_closed_context_condition_keeps_original_runtime(self) -> None:
        task = make_task("hotpotqa", 0)
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        base_runtime = AgentRuntime(registry, self._Gateway())
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = self._config(enabled=False)
        backend.registry = registry
        backend.runtime = base_runtime
        backend.project_root = Path(self._temp_dir.name)

        with patch.object(
            _MODULE,
            "open_qa_tool_registry",
            side_effect=AssertionError("closed context must not open retrieval"),
        ):
            runtime, tool_registry, close = backend._runtime_for_task(task)

        self.assertIs(base_runtime, runtime)
        self.assertIsNone(tool_registry)
        close()
        with self.assertRaisesRegex(ConfigurationError, "task-scoped runtime"):
            backend._runtime_for_task(
                task,
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )

    def test_sampling_coordinate_is_bound_to_runtime_task_and_seed(self) -> None:
        task = make_task("hotpotqa", 0)
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = self._config()
        backend.registry = registry
        backend.runtime = AgentRuntime(registry, self._Gateway())
        backend.project_root = Path(self._temp_dir.name)

        wrong_task = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
            schedule_purpose="test",
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=0,
            task_id="hotpotqa:different-task",
            optimizer_step_or_anchor_ordinal=0,
        )
        wrong_schedule = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=18),
            schedule_purpose="test",
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=0,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=0,
        )
        with patch.object(
            _MODULE,
            "open_qa_tool_registry",
            side_effect=AssertionError("invalid sampling must fail before Tool setup"),
        ):
            with self.assertRaisesRegex(ConfigurationError, "task_id"):
                backend._runtime_for_task(
                    task,
                    sampling_base_seed=17,
                    sampling_coordinate=wrong_task,
                )
            with self.assertRaisesRegex(ConfigurationError, "schedule hash"):
                backend._runtime_for_task(
                    task,
                    sampling_base_seed=17,
                    sampling_coordinate=wrong_schedule,
                )

    def test_tool_condition_shares_registry_and_exposes_only_capabilities(self) -> None:
        root = Path(self._temp_dir.name)
        task = TaskRecord(
            task_id="HotpotQA:tool-test",
            question="Which public fact answers this question?",
            ground_truth="EVALUATOR_TRUTH_MUST_NOT_APPEAR",
            split="validation",
            metadata={"dataset_key": "hotpotqa", "source": "HotpotQA"},
        )
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        tool_registry = build_qa_tool_registry(self._Index())
        owner = SimpleNamespace(
            registry=tool_registry,
            closed=False,
        )

        def close() -> None:
            owner.closed = True

        owner.close = close
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = self._config()
        backend.registry = registry
        backend.runtime = AgentRuntime(registry, self._Gateway())
        backend.runtime.timeout_seconds = 37.0
        backend.project_root = root

        with patch.object(
            _MODULE, "open_qa_tool_registry", return_value=owner
        ) as opened:
            runtime, shared_registry, close_runtime = backend._runtime_for_task(
                task,
                semantic_protocol="hotpotqa_verified_answer_slot_v1",
            )

        self.assertIs(tool_registry, shared_registry)
        self.assertIs(tool_registry, runtime.tool_registry)
        self.assertIs(
            tool_registry,
            runtime.execution_adapters["react"]._tool_registry,
        )
        self.assertEqual("hotpotqa", runtime.dataset_id)
        self.assertEqual(
            "hotpotqa_verified_answer_slot_v1",
            runtime.semantic_protocol,
        )
        self.assertEqual(37.0, runtime.timeout_seconds)
        opened.assert_called_once_with(
            index_path=root / "data/public-retrieval.sqlite3",
            skillflow_source=root / "vendor/SkillFlow/src",
            dataset_scope=("hotpotqa",),
            timeout_seconds=7.0,
        )

        environment = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem=task.question,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=shared_registry,
            prompt_version=HOTPOTQA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
            recovery_policy="preserve_diagnose_repair_augment",
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(environment, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        rendered = messages[-1]["content"]
        state = json.loads(rendered.split("\n\n", 1)[1])
        self.assertEqual(
            {"qa-retrieval"},
            {item["tool_id"] for item in state["tool_catalog"]},
        )
        self.assertNotIn(task.ground_truth, rendered)
        close_runtime()
        self.assertTrue(owner.closed)

    def test_hotpot_provided_context_opens_task_scoped_skillflow_index(self) -> None:
        root = Path(self._temp_dir.name)
        task = TaskRecord(
            task_id="HotpotQA:provided-context",
            question="What book contains Widsith?",
            ground_truth="EVALUATOR_TRUTH_MUST_NOT_APPEAR",
            split="validation",
            metadata={
                "dataset_key": "hotpotqa",
                "source": "HotpotQA",
                "skillflow": {
                    "task_type": "multi_hop_qa",
                    "context": [
                        "[Widsith] Widsith survives in the Exeter Book."
                    ],
                },
            },
        )
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        tool_registry = build_qa_tool_registry(self._Index())
        owner = SimpleNamespace(registry=tool_registry, close=lambda: None)
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = self._config(passage_source="provided_context")
        backend.registry = registry
        backend.runtime = AgentRuntime(registry, self._Gateway())
        backend.project_root = root

        with patch.object(
            _MODULE,
            "open_provided_context_qa_tool_registry",
            return_value=owner,
        ) as opened_context, patch.object(
            _MODULE,
            "open_qa_tool_registry",
            side_effect=AssertionError("external corpus must not open"),
        ):
            runtime, shared_registry, close_runtime = backend._runtime_for_task(task)

        opened_context.assert_called_once_with(
            ["[Widsith] Widsith survives in the Exeter Book."],
            skillflow_source=root / "vendor/SkillFlow/src",
            dataset_scope=("hotpotqa",),
            timeout_seconds=7.0,
        )
        self.assertIs(shared_registry, tool_registry)
        self.assertEqual(
            "multi_hop_qa",
            runtime.execution_adapters["react"]._task_type,
        )
        close_runtime()

    def test_tool_condition_rejects_condition_or_dataset_aliasing(self) -> None:
        task = make_task("hotpotqa", 0)
        mismatched = self._config()
        mismatched["experiment"]["condition_id"] = "closed_context"
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            qa_tool_runtime_settings(mismatched, task)

        with self.assertRaisesRegex(ConfigurationError, "active rollout"):
            qa_tool_runtime_settings(
                self._config(),
                task,
                condition_id="different_runtime_arm",
            )

        out_of_scope = self._config()
        out_of_scope["qa_tool_runtime"]["dataset_scope"] = ["triviaqa"]
        with self.assertRaisesRegex(ConfigurationError, "not configured"):
            qa_tool_runtime_settings(out_of_scope, task)

    def test_completion_policy_defaults_to_required_tool_call_and_accepts_variants(
        self,
    ) -> None:
        task = make_task("hotpotqa", 0)
        settings = qa_tool_runtime_settings(self._config(), task)
        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual("required_tool_call", settings["completion_policy"])

        optional = self._config()
        optional["qa_tool_runtime"]["completion_policy"] = "optional"
        optional_settings = qa_tool_runtime_settings(optional, task)
        self.assertIsNotNone(optional_settings)
        assert optional_settings is not None
        self.assertEqual("optional", optional_settings["completion_policy"])

        required = self._config()
        required["qa_tool_runtime"]["completion_policy"] = "required_evidence"
        required_settings = qa_tool_runtime_settings(required, task)
        self.assertIsNotNone(required_settings)
        assert required_settings is not None
        self.assertEqual(
            "required_evidence",
            required_settings["completion_policy"],
        )

        invalid = self._config()
        invalid["qa_tool_runtime"]["completion_policy"] = "required_dispatch_twice"
        with self.assertRaisesRegex(ConfigurationError, "completion_policy"):
            qa_tool_runtime_settings(invalid, task)

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


class AIMEComputationRuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    class _NoCallGateway:
        async def generate(self, request):  # pragma: no cover - no model call
            raise AssertionError(f"unexpected model call: {request.request_id}")

    class _StructuredActionGateway:
        def __init__(self) -> None:
            self.requests = []
            self.outputs = [
                {
                    "kind": "tool",
                    "name": "calculator",
                    "arguments": {"expression": "20 + 22"},
                    "resource_id": AIME_CALCULATOR_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "tool",
                    "name": "python_exec",
                    "arguments": {"code": "print(6 * 7)"},
                    "resource_id": AIME_PYTHON_EXEC_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "complete",
                    "name": "complete",
                    "arguments": {"value": "42"},
                    "resource_id": None,
                    "skill_id": None,
                },
            ]

        async def generate(self, request):
            self.requests.append(request)
            value = self.outputs.pop(0)
            return AgentResponse(
                json.dumps(value),
                {
                    "provider_request_id": f"fake:{len(self.requests)}",
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "latency_ms": 1.0,
                },
            )

    @staticmethod
    def _config(*, enabled: bool | None = True) -> dict:
        condition_id = "aime_2026_computation_react_stable_zero"
        config = {"experiment": {"condition_id": condition_id}}
        if enabled is not None:
            config["aime_tool_runtime"] = {
                "enabled": enabled,
                "condition_id": condition_id,
                "mode": "model_driven_computation",
                "dataset_scope": ["aime_2026"],
                "max_turns_per_agent_call": 5,
                "max_tool_calls_per_agent_call": 3,
                "calculator_timeout_seconds": 2.0,
                "python_timeout_seconds": 4.0,
            }
        return config

    @staticmethod
    def _task() -> TaskRecord:
        return TaskRecord(
            task_id="AIME2026:tool-test",
            question="Find the required integer.",
            ground_truth="EVALUATOR_TRUTH_MUST_NOT_APPEAR",
            split="validation",
            metadata={"dataset_key": "aime_2026", "source": "AIME 2026"},
        )

    def _backend(self, config: dict, gateway=None):
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        resolved_gateway = gateway or self._NoCallGateway()
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = config
        backend.registry = registry
        backend.runtime = AgentRuntime(registry, resolved_gateway)
        backend.project_root = Path(self._temp_dir.name)
        return backend

    def test_missing_or_disabled_condition_keeps_original_runtime(self) -> None:
        task = self._task()
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                backend = self._backend(self._config(enabled=enabled))
                with patch.object(
                    _MODULE,
                    "create_aime_computation_registry",
                    side_effect=AssertionError(
                        "closed computation condition must not create Tools"
                    ),
                ):
                    runtime, tool_registry, close = backend._runtime_for_task(task)

                self.assertIs(backend.runtime, runtime)
                self.assertIsNone(tool_registry)
                close()

    def test_tool_condition_shares_exact_registry_and_public_catalog(self) -> None:
        task = self._task()
        backend = self._backend(self._config())
        backend.runtime.timeout_seconds = 37.0
        with patch.object(
            _MODULE,
            "create_aime_computation_registry",
            wraps=_MODULE.create_aime_computation_registry,
        ) as created:
            runtime, shared_registry, close = backend._runtime_for_task(
                task,
                condition_id="aime_2026_computation_react_stable_zero",
            )

        created.assert_called_once_with(
            python_timeout_seconds=4.0,
            calculator_timeout_seconds=2.0,
        )
        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertIs(
            shared_registry,
            runtime.execution_adapters["react"]._tool_registry,
        )
        self.assertEqual("aime_2026", runtime.dataset_id)
        self.assertEqual(37.0, runtime.timeout_seconds)

        orchestrator = AgentGraphOrchestrator(
            backend.registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=shared_registry,
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=task.question,
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(environment, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        rendered = messages[-1]["content"]
        state = json.loads(rendered.split("\n\n", 1)[1])
        self.assertEqual(
            {AIME_CALCULATOR_TOOL_ID, AIME_PYTHON_EXEC_TOOL_ID},
            {entry["tool_id"] for entry in state["tool_catalog"]},
        )
        self.assertNotIn(task.ground_truth, rendered)
        self.assertNotIn("reward", json.dumps(state["tool_catalog"]).lower())
        close()
        close()

    def test_condition_mode_scope_and_rollout_aliasing_are_rejected(self) -> None:
        task = self._task()

        experiment_mismatch = self._config()
        experiment_mismatch["experiment"]["condition_id"] = "closed_computation"
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            aime_tool_runtime_settings(experiment_mismatch, task)

        with self.assertRaisesRegex(ConfigurationError, "active rollout"):
            aime_tool_runtime_settings(
                self._config(),
                task,
                condition_id="different_condition",
            )

        wrong_mode = self._config()
        wrong_mode["aime_tool_runtime"]["mode"] = "deterministic_prefetch"
        with self.assertRaisesRegex(ConfigurationError, "model_driven_computation"):
            aime_tool_runtime_settings(wrong_mode, task)

        wrong_scope = self._config()
        wrong_scope["aime_tool_runtime"]["dataset_scope"] = ["aime_2026", "hotpotqa"]
        with self.assertRaisesRegex(ConfigurationError, "exactly"):
            aime_tool_runtime_settings(wrong_scope, task)

        with self.assertRaisesRegex(ConfigurationError, "not configured"):
            aime_tool_runtime_settings(self._config(), make_task("hotpotqa", 0))

    async def test_calculator_and_python_receipts_use_collector_persistence(self) -> None:
        gateway = self._StructuredActionGateway()
        backend = self._backend(self._config(), gateway=gateway)
        task = self._task()
        runtime, shared_registry, close = backend._runtime_for_task(task)
        graph = AgentGraph(
            (
                AgentNode(
                    "solver",
                    "qwen3.5-9b-local",
                    "Solve the problem using admitted computation Tools when useful.",
                    allowed_tools=(
                        AIME_CALCULATOR_TOOL_ID,
                        AIME_PYTHON_EXEC_TOOL_ID,
                    ),
                    execution_mode="react",
                    artifact_type="answer",
                    completion_condition="return the required integer",
                ),
            ),
            output_agent_id="solver",
        )

        result = await runtime.execute(
            graph,
            task.question,
            run_id="aime-computation-receipt-test",
        )

        self.assertEqual("42", result.final_answer)
        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertEqual(1, len(result.calls))
        execution = execution_record_from_call(result.calls[0])
        response_receipt = execution.metadata["response"]
        self.assertEqual(
            [AIME_CALCULATOR_TOOL_ID, AIME_PYTHON_EXEC_TOOL_ID],
            [entry["tool_id"] for entry in response_receipt["tool_receipts"]],
        )
        self.assertTrue(
            all(
                entry["error_type"] is None
                and entry["result"]["value"]["ok"] is True
                for entry in response_receipt["tool_receipts"]
            )
        )
        runtime_summary = _runtime_summary(result)
        self.assertEqual(
            response_receipt["tool_receipts"],
            runtime_summary["output_metadata"]["solver"]["tool_receipts"],
        )
        json.dumps(execution.to_dict())
        json.dumps(runtime_summary)
        self.assertNotIn(task.ground_truth, gateway.requests[0].problem)
        close()

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


class HealthBenchMedRAGRuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    SOURCE_REVISION = "medrag-fixture-revision"

    class _NoCallGateway:
        async def generate(self, request):  # pragma: no cover - no model call
            raise AssertionError(f"unexpected model call: {request.request_id}")

    class _StructuredActionGateway:
        def __init__(self) -> None:
            self.requests = []
            self.outputs = [
                {
                    "kind": "tool",
                    "name": "search",
                    "arguments": {
                        "query": "aspirin gastrointestinal bleeding"
                    },
                    "resource_id": HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "complete",
                    "name": "complete",
                    "arguments": {
                        "value": (
                            "Aspirin can increase gastrointestinal bleeding risk; "
                            "seek individualized advice from a clinician."
                        )
                    },
                    "resource_id": None,
                    "skill_id": None,
                },
            ]

        async def generate(self, request):
            self.requests.append(request)
            value = self.outputs.pop(0)
            return AgentResponse(
                json.dumps(value),
                {
                    "provider_request_id": f"health-fixture:{len(self.requests)}",
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                    "latency_ms": 2.0,
                },
            )

    @staticmethod
    def _config(*, enabled: bool | None = True) -> dict:
        condition_id = "healthbench_medrag_react_stable_zero"
        config = {"experiment": {"condition_id": condition_id}}
        if enabled is not None:
            config["healthbench_tool_runtime"] = {
                "enabled": enabled,
                "condition_id": condition_id,
                "mode": "model_driven_medrag_search",
                "dataset_scope": ["healthbench_professional"],
                "resource_dir": "resources/medrag-textbooks-runtime",
                "source_identity": "skillflow-medrag-textbooks",
                "source_revision": (
                    HealthBenchMedRAGRuntimeWiringTests.SOURCE_REVISION
                ),
                "expected_rows": 2,
                "max_turns_per_agent_call": 4,
                "max_tool_calls_per_agent_call": 2,
                "tool_timeout_seconds": 6.0,
            }
        return config

    @staticmethod
    def _task() -> TaskRecord:
        return TaskRecord(
            task_id="healthbench-professional:tool-test",
            question=(
                "Conversation:\n\n[user] Is aspirin safe if I have a history "
                "of stomach bleeding?\n\n[assistant]"
            ),
            ground_truth="PHYSICIAN_RESPONSE_EVALUATOR_ONLY",
            split="validation",
            metadata={
                "dataset_key": "healthbench_professional",
                "source": "HealthBench Professional",
                "evaluator_payload": {
                    "rubric_items": [
                        {
                            "criterion_text": "RUBRIC_EVALUATOR_ONLY",
                            "points": 10,
                        }
                    ],
                    "physician_response": "REFERENCE_EVALUATOR_ONLY",
                },
            },
        )

    @classmethod
    def _registry_owner(cls):
        documents = (
            "Aspirin can cause gastrointestinal bleeding and peptic ulcer disease.",
            "Insulin lowers blood glucose in diabetes mellitus.",
        )
        corpus = FrozenMedRAGBM25Corpus(
            source_identity="skillflow-medrag-textbooks",
            source_revision=cls.SOURCE_REVISION,
            corpus_rows=len(documents),
            _corpus=documents,
            _index={
                "avg_dl": 7.0,
                "doc_lens": [8, 7],
                "idf": {
                    "aspirin": 2.0,
                    "gastrointestinal": 2.0,
                    "bleeding": 2.0,
                },
                "inverted_index": {
                    "aspirin": [(0, 1)],
                    "gastrointestinal": [(0, 1)],
                    "bleeding": [(0, 1)],
                },
            },
        )
        owner = SimpleNamespace(
            registry=build_healthbench_medrag_tool_registry(
                corpus,
                timeout_seconds=6.0,
            ),
            closed=False,
        )

        def close() -> None:
            if owner.closed:
                return
            corpus.close()
            owner.closed = True

        owner.close = close
        return owner

    def _backend(self, config: dict, gateway=None):
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = config
        backend.registry = registry
        backend.runtime = AgentRuntime(
            registry,
            gateway or self._NoCallGateway(),
        )
        backend.project_root = Path(self._temp_dir.name)
        return backend

    def test_missing_or_disabled_condition_keeps_closed_context_runtime(self) -> None:
        task = self._task()
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                backend = self._backend(self._config(enabled=enabled))
                with patch.object(
                    _MODULE,
                    "open_healthbench_medrag_tool_registry",
                    side_effect=AssertionError(
                        "closed HealthBench condition must not open MedRAG"
                    ),
                ):
                    runtime, tool_registry, close = backend._runtime_for_task(task)

                self.assertIs(backend.runtime, runtime)
                self.assertIsNone(tool_registry)
                close()

    def test_enabled_condition_shares_registry_and_public_catalog(self) -> None:
        task = self._task()
        backend = self._backend(self._config())
        backend.runtime.timeout_seconds = 37.0
        owner = self._registry_owner()
        root = backend.project_root

        with patch.object(
            _MODULE,
            "open_healthbench_medrag_tool_registry",
            return_value=owner,
        ) as opened:
            runtime, shared_registry, close = backend._runtime_for_task(
                task,
                condition_id="healthbench_medrag_react_stable_zero",
            )

        opened.assert_called_once_with(
            corpus_root=root / "resources/medrag-textbooks-runtime",
            source_identity="skillflow-medrag-textbooks",
            expected_source_revision=self.SOURCE_REVISION,
            expected_rows=2,
            timeout_seconds=6.0,
        )
        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertIs(
            shared_registry,
            runtime.execution_adapters["react"]._tool_registry,
        )
        self.assertEqual("healthbench_professional", runtime.dataset_id)
        self.assertEqual(37.0, runtime.timeout_seconds)

        orchestrator = AgentGraphOrchestrator(
            backend.registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=shared_registry,
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=task.question,
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(environment, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        rendered = messages[-1]["content"]
        state = json.loads(rendered.split("\n\n", 1)[1])
        self.assertEqual(
            [HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID],
            [entry["tool_id"] for entry in state["tool_catalog"]],
        )
        for evaluator_only in (
            task.ground_truth,
            "RUBRIC_EVALUATOR_ONLY",
            "REFERENCE_EVALUATOR_ONLY",
            "physician_response",
            "evaluator_payload",
        ):
            self.assertNotIn(evaluator_only, rendered)

        close()
        close()
        self.assertTrue(owner.closed)

    def test_mode_scope_condition_and_resource_settings_are_strict(self) -> None:
        task = self._task()

        experiment_mismatch = self._config()
        experiment_mismatch["experiment"]["condition_id"] = "closed_healthbench"
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            healthbench_tool_runtime_settings(experiment_mismatch, task)

        with self.assertRaisesRegex(ConfigurationError, "active rollout"):
            healthbench_tool_runtime_settings(
                self._config(),
                task,
                condition_id="different_condition",
            )

        wrong_mode = self._config()
        wrong_mode["healthbench_tool_runtime"]["mode"] = "generic_retrieval"
        with self.assertRaisesRegex(ConfigurationError, "model_driven_medrag_search"):
            healthbench_tool_runtime_settings(wrong_mode, task)

        wrong_scope = self._config()
        wrong_scope["healthbench_tool_runtime"]["dataset_scope"] = [
            "healthbench_professional",
            "hotpotqa",
        ]
        with self.assertRaisesRegex(ConfigurationError, "exactly"):
            healthbench_tool_runtime_settings(wrong_scope, task)

        bad_rows = self._config()
        bad_rows["healthbench_tool_runtime"]["expected_rows"] = 0
        with self.assertRaisesRegex(ConfigurationError, "expected_rows"):
            healthbench_tool_runtime_settings(bad_rows, task)

        missing_revision = self._config()
        missing_revision["healthbench_tool_runtime"]["source_revision"] = ""
        with self.assertRaisesRegex(ConfigurationError, "source_revision"):
            healthbench_tool_runtime_settings(missing_revision, task)

        with self.assertRaisesRegex(ConfigurationError, "not configured"):
            healthbench_tool_runtime_settings(
                self._config(),
                make_task("hotpotqa", 0),
            )

    async def test_medrag_tool_receipt_is_persisted_without_evaluator_payload(
        self,
    ) -> None:
        gateway = self._StructuredActionGateway()
        backend = self._backend(self._config(), gateway=gateway)
        task = self._task()
        owner = self._registry_owner()
        with patch.object(
            _MODULE,
            "open_healthbench_medrag_tool_registry",
            return_value=owner,
        ):
            runtime, shared_registry, close = backend._runtime_for_task(task)

        graph = AgentGraph(
            (
                AgentNode(
                    "clinical_reasoner",
                    "qwen3.5-9b-local",
                    "Answer the public conversation using admitted evidence when useful.",
                    allowed_tools=(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,),
                    execution_mode="react",
                    artifact_type="clinical_response",
                    completion_condition="return a clinically useful response",
                ),
            ),
            output_agent_id="clinical_reasoner",
        )
        result = await runtime.execute(
            graph,
            task.question,
            run_id="healthbench-medrag-receipt-test",
        )

        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertEqual(1, len(result.calls))
        execution = execution_record_from_call(result.calls[0])
        response_receipt = execution.metadata["response"]
        self.assertEqual(
            [HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID],
            [entry["tool_id"] for entry in response_receipt["tool_receipts"]],
        )
        self.assertIsNone(response_receipt["tool_receipts"][0]["error_type"])
        runtime_summary = _runtime_summary(result)
        self.assertEqual(
            response_receipt["tool_receipts"],
            runtime_summary["output_metadata"]["clinical_reasoner"][
                "tool_receipts"
            ],
        )
        serialized = json.dumps(
            {
                "execution": execution.to_dict(),
                "runtime": runtime_summary,
            },
            sort_keys=True,
        )
        for evaluator_only in (
            task.ground_truth,
            "RUBRIC_EVALUATOR_ONLY",
            "REFERENCE_EVALUATOR_ONLY",
            "physician_response",
            "evaluator_payload",
        ):
            self.assertNotIn(evaluator_only, serialized)
            self.assertTrue(
                all(evaluator_only not in request.problem for request in gateway.requests)
            )
        close()
        self.assertTrue(owner.closed)

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


class EnvironmentReplayRuntimeWiringTests(unittest.TestCase):
    def test_extracts_the_only_environment_actor_trace(self) -> None:
        trace = (
            {
                "step": 0,
                "observation": "room",
                "legal_actions": ["look"],
                "action": "look",
                "next_observation": "table",
                "reward": 0.0,
                "done": False,
                "info": {},
                "state_advanced": True,
            },
        )
        runtime = SimpleNamespace(
            output_metadata={
                "planner": {"execution_mode": "reasoning"},
                "actor": {"evaluator_environment_trace": trace},
            }
        )

        extracted = environment_replay_trace_from_runtime(runtime)

        self.assertEqual(trace, extracted)
        self.assertIsNot(trace[0], extracted[0])

    def test_rejects_multiple_independent_environment_episodes(self) -> None:
        runtime = SimpleNamespace(
            output_metadata={
                "actor_a": {"evaluator_environment_trace": []},
                "actor_b": {"evaluator_environment_trace": []},
            }
        )

        with self.assertRaisesRegex(ConfigurationError, "multiple environment"):
            environment_replay_trace_from_runtime(runtime)

    def test_closed_condition_without_runtime_metadata_remains_empty(self) -> None:
        self.assertEqual((), environment_replay_trace_from_runtime(None))
        self.assertEqual(
            (),
            environment_replay_trace_from_runtime(
                SimpleNamespace(output_metadata={"solver": {}})
            ),
        )


class EnvironmentRuntimeWiringTests(unittest.TestCase):
    class _Gateway:
        async def generate(self, request):  # pragma: no cover - no model call
            raise AssertionError(f"unexpected model call: {request.request_id}")

    @staticmethod
    def _config(
        *,
        source: str = "webshop",
        enabled: bool | None = True,
        runtime_budget: int = 3,
        evaluator_budget: int = 3,
    ) -> dict:
        condition_id = f"{source}_ragen_react_stable_zero"
        config = {
            "experiment": {"condition_id": condition_id},
            "evaluation": {
                "max_environment_steps": 12,
                "max_environment_steps_by_source": {
                    source: evaluator_budget,
                },
            },
        }
        if enabled is not None:
            config["environment_runtime"] = {
                "enabled": enabled,
                "condition_id": condition_id,
                "mode": "model_driven_ragen_react",
                "dataset_scope": [source],
                "ragen_adapter_path": "vendor/SkillFlow/src/ragen_adapter.py",
                "tool_timeout_seconds": 9.0,
                "max_action_tokens": 256,
                "max_environment_steps_by_source": {
                    source: runtime_budget,
                },
            }
        return config

    @staticmethod
    def _task(source: str = "webshop") -> TaskRecord:
        env_config = (
            {"goal_index": 7, "env_seed": 1234}
            if source == "webshop"
            else {"game_file": "/games/locked-task-7.tw-pddl", "max_steps": 50}
        )
        return TaskRecord(
            task_id=f"{source}:locked-task-7",
            question="Complete the aligned environment task.",
            ground_truth="",
            split="validation",
            metadata={
                "dataset_key": source,
                "source": SOURCE_NAMES[source],
                "environment": {
                    "env_type": source,
                    "env_config": env_config,
                },
            },
        )

    def _backend(self, config: dict):
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = config
        backend.registry = registry
        backend.runtime = AgentRuntime(registry, self._Gateway())
        backend.project_root = Path(self._temp_dir.name)
        return backend

    def test_missing_or_disabled_condition_keeps_original_runtime(self) -> None:
        task = self._task()
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                backend = self._backend(self._config(enabled=enabled))
                with (
                    patch.object(
                        _MODULE,
                        "evaluator_locked_ragen_session_factory",
                        side_effect=AssertionError(
                            "closed condition must not load a RAGEN task"
                        ),
                    ),
                    patch.object(
                        _MODULE,
                        "build_environment_execution_resources",
                        side_effect=AssertionError(
                            "closed condition must not build environment tools"
                        ),
                    ),
                ):
                    runtime, tool_registry, close = backend._runtime_for_task(task)

                self.assertIs(backend.runtime, runtime)
                self.assertIsNone(tool_registry)
                close()

    def test_live_condition_shares_registry_and_binds_the_exact_record(self) -> None:
        task = self._task()
        backend = self._backend(self._config())
        backend.runtime.timeout_seconds = 37.0
        root = backend.project_root

        # The concrete builder only stores this callable until execution; no
        # simulator is started by this CPU wiring test.
        def session_factory(request):  # type: ignore[no-untyped-def]
            del request
            return None

        with (
            patch.object(
                _MODULE,
                "evaluator_locked_ragen_session_factory",
                return_value=session_factory,
            ) as locked,
            patch.object(
                _MODULE,
                "build_environment_execution_resources",
                wraps=_MODULE.build_environment_execution_resources,
            ) as built,
        ):
            runtime, shared_registry, close = backend._runtime_for_task(
                task,
                condition_id="webshop_ragen_react_stable_zero",
            )

        locked.assert_called_once_with(
            record=task,
            dataset="webshop",
            ragen_adapter_path=root / "vendor/SkillFlow/src/ragen_adapter.py",
            max_environment_steps=3,
        )
        built.assert_called_once_with(
            gateway=backend.runtime.gateway,
            session_factory=session_factory,
            task_family="webshop",
            max_turns=3,
            max_action_tokens=256,
            max_observation_chars=0,
            stepwise_director=False,
            structured_actions=False,
            timeout_seconds=9.0,
        )
        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertIs(
            shared_registry,
            runtime.execution_adapters["react"]._tool_registry,
        )
        self.assertEqual(37.0, runtime.timeout_seconds)
        self.assertEqual("webshop", runtime.dataset_id)

        orchestrator = AgentGraphOrchestrator(
            backend.registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=shared_registry,
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=task.question,
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(environment, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        state = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
        self.assertEqual(
            ["webshop.environment"],
            [entry["tool_id"] for entry in state["tool_catalog"]],
        )
        rendered_catalog = json.dumps(state["tool_catalog"], sort_keys=True)
        self.assertNotIn("reward", rendered_catalog)
        self.assertNotIn("info", rendered_catalog)

        close()
        close()

    def test_alfworld_uses_its_exact_record_and_source_budget(self) -> None:
        task = self._task("alfworld")
        backend = self._backend(
            self._config(
                source="alfworld",
                runtime_budget=50,
                evaluator_budget=50,
            )
        )

        def session_factory(request):  # type: ignore[no-untyped-def]
            del request
            return None

        with (
            patch.object(
                _MODULE,
                "evaluator_locked_ragen_session_factory",
                return_value=session_factory,
            ) as locked,
            patch.object(
                _MODULE,
                "build_environment_execution_resources",
                wraps=_MODULE.build_environment_execution_resources,
            ) as built,
        ):
            runtime, shared_registry, close = backend._runtime_for_task(task)

        locked.assert_called_once_with(
            record=task,
            dataset="alfworld",
            ragen_adapter_path=(
                backend.project_root / "vendor/SkillFlow/src/ragen_adapter.py"
            ),
            max_environment_steps=50,
        )
        self.assertEqual("alfworld", built.call_args.kwargs["task_family"])
        self.assertEqual(50, built.call_args.kwargs["max_turns"])
        self.assertEqual("alfworld", runtime.dataset_id)
        self.assertIs(shared_registry, runtime.tool_registry)
        close()

    def test_condition_scope_and_budget_mismatches_are_rejected(self) -> None:
        task = self._task()

        experiment_mismatch = self._config()
        experiment_mismatch["experiment"]["condition_id"] = "closed_condition"
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            environment_runtime_settings(experiment_mismatch, task)

        with self.assertRaisesRegex(ConfigurationError, "active rollout"):
            environment_runtime_settings(
                self._config(),
                task,
                condition_id="different_condition",
            )

        out_of_scope = self._config()
        out_of_scope["environment_runtime"]["dataset_scope"] = ["alfworld"]
        with self.assertRaisesRegex(ConfigurationError, "not configured"):
            environment_runtime_settings(out_of_scope, task)

        budget_mismatch = self._config(runtime_budget=2, evaluator_budget=3)
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            environment_runtime_settings(budget_mismatch, task)

    def test_terminal_or_budget_trace_is_passed_without_legacy_resampling(self) -> None:
        task = self._task()
        backend = self._backend(self._config())
        backend.judge = None
        backend.judge_model = ""
        backend.swe_harness = None
        outcome = _MODULE.EvaluationOutcome(
            valid=True,
            reward=0.0,
            metrics={"success": 0.0},
            reason="evaluated",
            evaluator_version=_MODULE.RAGEN_EVALUATOR_VERSION,
        )
        trace = tuple(
            {
                "step": index,
                "observation": f"observation-{index}",
                "legal_actions": ["look"],
                "action": "look",
                "next_observation": f"observation-{index + 1}",
                "reward": 0.0,
                "done": False,
                "info": {"private_evaluator_field": index},
                "state_advanced": True,
            }
            for index in range(3)
        )
        captured: dict = {}

        async def fake_evaluate_task(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured["args"] = args
            captured["kwargs"] = kwargs
            return outcome

        with patch.object(_MODULE, "evaluate_task", side_effect=fake_evaluate_task):
            result = asyncio.run(
                backend.evaluate_final_graph(
                    task,
                    "observation-3",
                    {"nodes": [], "relations": [], "revision": 0},
                    rollout_index=4,
                    environment_replay_trace=trace,
                )
            )

        self.assertIs(outcome, result)
        self.assertEqual(trace, captured["kwargs"]["environment_replay_trace"])
        self.assertEqual(3, captured["kwargs"]["max_environment_steps"])
        self.assertIsNone(captured["kwargs"]["run_graph"])
        self.assertEqual(
            backend.project_root / "vendor/SkillFlow/src/ragen_adapter.py",
            captured["kwargs"]["ragen_adapter_path"],
        )
        self.assertNotIn(
            "private_evaluator_field",
            workflow_problem(task, self._config()),
        )

    def test_partial_nonterminal_trace_cannot_fall_back_to_old_runtime(self) -> None:
        task = self._task()
        backend = self._backend(self._config())
        backend.judge = None
        backend.judge_model = ""
        backend.swe_harness = None
        partial = (
            {
                "step": 0,
                "observation": "start",
                "legal_actions": ["look"],
                "action": "look",
                "next_observation": "next",
                "reward": 0.0,
                "done": False,
                "info": {},
                "state_advanced": True,
            },
        )

        with (
            patch.object(
                _MODULE,
                "evaluate_task",
                side_effect=AssertionError("partial trace must not reach evaluator"),
            ),
            self.assertRaisesRegex(ConfigurationError, "terminal state or exhaust"),
        ):
            asyncio.run(
                backend.evaluate_final_graph(
                    task,
                    "next",
                    {"nodes": [], "relations": [], "revision": 0},
                    rollout_index=0,
                    environment_replay_trace=partial,
                )
            )

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


class SWEbenchCodingRuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    _GOLD_PATCH = "GOLD_PATCH_EVALUATOR_ONLY"
    _TEST_PATCH = "TEST_PATCH_EVALUATOR_ONLY"
    _GROUND_TRUTH = "GROUND_TRUTH_EVALUATOR_ONLY"

    class _NoCallGateway:
        async def generate(self, request):  # pragma: no cover - no model call
            raise AssertionError(f"unexpected model call: {request.request_id}")

    class _CodingGateway:
        def __init__(self) -> None:
            self.requests = []
            self.outputs = [
                {
                    "kind": "tool",
                    "name": "view_file",
                    "arguments": {"path": "bug.py"},
                    "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "tool",
                    "name": "exact_edit",
                    "arguments": {
                        "path": "bug.py",
                        "old_str": "return a - b",
                        "new_str": "return a + b",
                    },
                    "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "tool",
                    "name": "run_tests",
                    "arguments": {
                        "command": [
                            sys.executable,
                            "-c",
                            "from bug import add; assert add(2, 3) == 5",
                        ],
                        "timeout_seconds": 5,
                    },
                    "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "tool",
                    "name": "diff",
                    "arguments": {},
                    "resource_id": SWEBENCH_REPOSITORY_TOOL_ID,
                    "skill_id": None,
                },
                {
                    "kind": "complete",
                    "name": "complete",
                    "arguments": {
                        "value": "model prose must not replace the workspace diff"
                    },
                    "resource_id": None,
                    "skill_id": None,
                },
            ]

        async def generate(self, request):
            self.requests.append(request)
            return AgentResponse(
                json.dumps(self.outputs.pop(0)),
                {
                    "provider_request_id": f"coding:{len(self.requests)}",
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "latency_ms": 1.0,
                },
            )

    @staticmethod
    def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _config(*, enabled: bool | None = True) -> dict:
        condition_id = "swebench_verified_coding_stable_zero"
        config = {
            "experiment": {"condition_id": condition_id},
            "evaluation": {"max_environment_steps": 12},
        }
        if enabled is not None:
            config["swe_coding_runtime"] = {
                "enabled": enabled,
                "condition_id": condition_id,
                "mode": "iterative_repository_coding",
                "dataset_scope": ["swe_bench"],
                "repository_store": "repositories",
                "worktree_root": "worktrees",
                "max_turns_per_agent_call": 6,
                "max_tool_calls_per_agent_call": 5,
                "max_test_timeout_seconds": 8.0,
                "setup_timeout_seconds": 10.0,
                "cleanup_timeout_seconds": 10.0,
            }
        return config

    def _task(self, *, instance_id: str = "owner__repo-1") -> TaskRecord:
        return TaskRecord(
            task_id=f"swe-bench:{instance_id}",
            question="Fix add so it returns the sum instead of the difference.",
            ground_truth=self._GROUND_TRUTH,
            split="validation",
            metadata={
                "dataset_key": "swe_bench",
                "source": "SWE-bench",
                "skillflow": {
                    "extra": {
                        "instance_id": instance_id,
                        "repo": "owner/repo",
                        "base_commit": self.base_commit,
                    }
                },
                "evaluator_payload": {
                    "instance_id": instance_id,
                    "patch": self._GOLD_PATCH,
                    "test_patch": self._TEST_PATCH,
                },
            },
        )

    def _backend(self, config: dict, *, gateway=None):
        registry = load_model_registry(
            Path(__file__).resolve().parents[2]
            / "config/model_catalog_triviaqa_v1.yaml"
        )
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = config
        backend.registry = registry
        backend.runtime = AgentRuntime(
            registry,
            gateway or self._NoCallGateway(),
        )
        backend.project_root = self.root
        backend.judge = None
        backend.judge_model = ""
        backend.swe_harness = None
        return backend

    @staticmethod
    def _prepared_path(registry) -> Path:
        repository_backend = registry._backend(SWEBENCH_REPOSITORY_TOOL_ID)
        return repository_backend.repo_root

    async def test_coding_condition_uses_shared_registry_and_persists_receipts(
        self,
    ) -> None:
        gateway = self._CodingGateway()
        backend = self._backend(self._config(), gateway=gateway)
        backend.runtime.timeout_seconds = 37.0
        task = self._task()

        runtime, shared_registry, close = backend._runtime_for_task(
            task,
            condition_id="swebench_verified_coding_stable_zero",
        )
        prepared_path = self._prepared_path(shared_registry)
        self.assertIs(shared_registry, runtime.tool_registry)
        self.assertIs(
            shared_registry,
            runtime.execution_adapters["coding"]._tool_registry,
        )
        self.assertEqual("swe_bench", runtime.dataset_id)
        self.assertEqual(37.0, runtime.timeout_seconds)
        self.assertEqual(
            (SWEBENCH_REPOSITORY_TOOL_ID,),
            shared_registry.resource_ids,
        )

        orchestrator = AgentGraphOrchestrator(
            backend.registry,
            client=object(),  # type: ignore[arg-type]
            tool_registry=shared_registry,
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=task.question,
        )
        messages = decode_director_transcript(
            orchestrator.build_prompt(environment, 0, ())
        )
        self.assertIsNotNone(messages)
        assert messages is not None
        rendered = messages[-1]["content"]
        state = json.loads(rendered.split("\n\n", 1)[1])
        self.assertEqual(
            [SWEBENCH_REPOSITORY_TOOL_ID],
            [entry["tool_id"] for entry in state["tool_catalog"]],
        )
        for evaluator_only in (
            self._GROUND_TRUTH,
            self._GOLD_PATCH,
            self._TEST_PATCH,
            "evaluator_payload",
        ):
            self.assertNotIn(evaluator_only, rendered)

        graph = AgentGraph(
            (
                AgentNode(
                    "coder",
                    "qwen3.5-9b-local",
                    "Inspect the repository, implement the issue, run a targeted test, and submit the workspace diff.",
                    allowed_tools=(SWEBENCH_REPOSITORY_TOOL_ID,),
                    execution_mode="coding",
                    artifact_type="patch_candidate",
                    completion_condition="submit the tested unified workspace diff",
                ),
            ),
            output_agent_id="coder",
        )
        result = await runtime.execute(
            graph,
            task.question,
            run_id="swebench-coding-runtime-test",
        )

        self.assertIn("diff --git a/bug.py b/bug.py", result.final_answer)
        self.assertIn("return a + b", result.final_answer)
        self.assertNotIn("model prose", result.final_answer)
        execution = execution_record_from_call(result.calls[0])
        response_receipt = execution.metadata["response"]
        self.assertEqual("coding", response_receipt["execution_mode"])
        self.assertEqual(4, response_receipt["tool_calls"])
        self.assertEqual(
            ["view_file", "exact_edit", "run_tests", "diff"],
            [
                receipt["request"]["action"]
                for receipt in response_receipt["tool_receipts"]
            ],
        )
        summary = _runtime_summary(result)
        self.assertEqual(
            response_receipt["tool_receipts"],
            summary["output_metadata"]["coder"]["tool_receipts"],
        )
        serialized_runtime = json.dumps(
            {
                "execution": execution.to_dict(),
                "summary": summary,
            },
            sort_keys=True,
        )
        for evaluator_only in (
            self._GROUND_TRUTH,
            self._GOLD_PATCH,
            self._TEST_PATCH,
            "evaluator_payload",
        ):
            self.assertNotIn(evaluator_only, serialized_runtime)
            self.assertTrue(
                all(
                    evaluator_only not in request.problem
                    and evaluator_only not in request.agent.contract
                    for request in gateway.requests
                )
            )

        close()
        close()
        self.assertFalse(prepared_path.exists())

    def test_each_runtime_owns_an_isolated_task_worktree(self) -> None:
        backend = self._backend(self._config())
        first_runtime, first_registry, first_close = backend._runtime_for_task(
            self._task(instance_id="owner__repo-1")
        )
        second_runtime, second_registry, second_close = backend._runtime_for_task(
            self._task(instance_id="owner__repo-2")
        )
        del first_runtime, second_runtime
        first_path = self._prepared_path(first_registry)
        second_path = self._prepared_path(second_registry)
        try:
            self.assertNotEqual(first_path, second_path)
            (first_path / "bug.py").write_text(
                "def add(a, b):\n    return 99\n",
                encoding="utf-8",
            )
            self.assertEqual(
                "def add(a, b):\n    return a - b\n",
                (second_path / "bug.py").read_text(encoding="utf-8"),
            )
        finally:
            first_close()
            second_close()
        self.assertFalse(first_path.exists())
        self.assertFalse(second_path.exists())

    def test_missing_or_disabled_condition_keeps_historical_runtime(self) -> None:
        task = self._task()
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                backend = self._backend(self._config(enabled=enabled))
                with patch.object(
                    _MODULE,
                    "prepare_swebench_worktree_for_task",
                    side_effect=AssertionError(
                        "historical no-coding condition must not prepare a repository"
                    ),
                ):
                    runtime, registry, close = backend._runtime_for_task(task)
                self.assertIs(backend.runtime, runtime)
                self.assertIsNone(registry)
                close()

    def test_condition_scope_mode_budgets_and_single_runtime_are_strict(self) -> None:
        task = self._task()

        experiment_mismatch = self._config()
        experiment_mismatch["experiment"]["condition_id"] = "old_one_shot"
        with self.assertRaisesRegex(ConfigurationError, "exactly match"):
            swe_coding_runtime_settings(experiment_mismatch, task)

        with self.assertRaisesRegex(ConfigurationError, "active rollout"):
            swe_coding_runtime_settings(
                self._config(),
                task,
                condition_id="different_condition",
            )

        wrong_scope = self._config()
        wrong_scope["swe_coding_runtime"]["dataset_scope"] = [
            "swe_bench",
            "hotpotqa",
        ]
        with self.assertRaisesRegex(ConfigurationError, "exactly"):
            swe_coding_runtime_settings(wrong_scope, task)

        wrong_mode = self._config()
        wrong_mode["swe_coding_runtime"]["mode"] = "one_shot_patch"
        with self.assertRaisesRegex(ConfigurationError, "iterative_repository_coding"):
            swe_coding_runtime_settings(wrong_mode, task)

        bad_timeout = self._config()
        bad_timeout["swe_coding_runtime"]["max_test_timeout_seconds"] = 0
        with self.assertRaisesRegex(ConfigurationError, "max_test_timeout_seconds"):
            swe_coding_runtime_settings(bad_timeout, task)

        with self.assertRaisesRegex(ConfigurationError, "not configured"):
            swe_coding_runtime_settings(self._config(), make_task("hotpotqa", 0))

        multiple = self._config()
        multiple["qa_tool_runtime"] = {
            "enabled": True,
            "condition_id": multiple["experiment"]["condition_id"],
        }
        backend = self._backend(multiple)
        with self.assertRaisesRegex(ConfigurationError, "multiple task-scoped"):
            backend._runtime_for_task(task)

    async def test_official_resolved_boundary_receives_only_workspace_diff(self) -> None:
        task = self._task()
        backend = self._backend(self._config(enabled=False))
        calls = []

        async def official_harness(record, prediction):
            calls.append((record.task_id, prediction))
            return {
                "resolved": True,
                "official_score": 1.0,
                "proxy_metric_used": False,
            }

        backend.swe_harness = official_harness
        patch_text = (
            "diff --git a/bug.py b/bug.py\n"
            "--- a/bug.py\n+++ b/bug.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n"
            "-    return a - b\n+    return a + b\n"
        )
        outcome = await backend.evaluate_final_graph(
            task,
            patch_text,
            {"nodes": [], "relations": [], "revision": 0},
            rollout_index=0,
        )

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.metrics["resolved"])
        self.assertEqual([(task.task_id, patch_text)], calls)
        self.assertFalse(outcome.details["proxy_metric_used"])

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        repository_store = self.root / "repositories"
        worktree_root = self.root / "worktrees"
        source = repository_store / "owner__repo"
        source.mkdir(parents=True)
        worktree_root.mkdir()
        self._git(source, "init", "-q")
        self._git(source, "config", "user.email", "fixture@example.invalid")
        self._git(source, "config", "user.name", "Fixture")
        (source / "bug.py").write_text(
            "def add(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        self._git(source, "add", "bug.py")
        self._git(source, "commit", "-q", "-m", "base")
        self.base_commit = self._git(
            source, "rev-parse", "HEAD"
        ).stdout.strip()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


def create_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    source_config = Path("config/training_agentgraph_smoke.yaml")
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["data"]["train_path"] = "data/train.jsonl"
    config["experiment"]["output_dir"] = "artifacts/smoke"
    config["storage"].update(
        root="artifacts/smoke/evidence",
        selected_tasks_path="artifacts/smoke/data/selected_tasks.jsonl",
        trajectories_path="artifacts/smoke/data/trajectories.jsonl",
        grpo_groups_path="artifacts/smoke/data/grpo_groups.jsonl",
        manifest_path="artifacts/smoke/data/training_manifest.json",
        sync_receipt_path="artifacts/smoke/data/sync_receipt.json",
        post_update_trajectories_path=(
            "artifacts/smoke/data/post_update_trajectories.jsonl"
        ),
    )
    config_path = root / "config" / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (root / "data" / "train.jsonl").open("w", encoding="utf-8") as handle:
        for source in EXPECTED_SOURCE_ORDER:
            for index in range(3):
                handle.write(json.dumps(aligned_row(make_task(source, index))) + "\n")
    return root, config_path


def create_hotpot_micro_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, config_path = create_project(tmp_path)
    validation_path = root / "data" / "validation.jsonl"
    test_path = root / "data" / "test.jsonl"
    for path, split, index in (
        (validation_path, "validation", 100),
        (test_path, "test", 101),
    ):
        task = TaskRecord(
            task_id=f"hotpotqa:{split}-{index}",
            question="Held-out question?",
            ground_truth="answer",
            split=split,
            metadata={"dataset_key": "hotpotqa", "source": "HotpotQA"},
        )
        path.write_text(json.dumps(aligned_row(task)) + "\n", encoding="utf-8")

    schedule = freeze_hotpot_training_schedule(
        train_path=root / "data" / "train.jsonl",
        validation_path=validation_path,
        test_path=test_path,
        task_positions=(0,),
        rollouts_per_task=2,
    )
    schedule_path = root / "artifacts" / "schedule.json"
    cursor_path = root / "artifacts" / "cursor0.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    HotpotTrainingCursorState.fresh(schedule).write_once(cursor_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"].update(
        phase="hotpotqa_micro_training",
        update_step=1,
        output_dir="artifacts/hotpot_step1/training",
    )
    config["data"].update(
        validation_path="data/validation.jsonl",
        test_path="data/test.jsonl",
        hotpot_micro={
            "split": "train",
            "dataset_key": "hotpotqa",
            "selection": "frozen_hotpot_schedule",
            "expected_total_tasks": 1,
            "schedule_path": "artifacts/schedule.json",
            "cursor_path": "artifacts/cursor0.json",
            "next_cursor_path": "artifacts/cursor1.json",
        },
    )
    config["evaluation"]["healthbench_judge_model"] = ""
    config["grpo"].update(samples_per_problem=2, expected_rollout_count=2)
    for field in (
        "root",
        "selected_tasks_path",
        "trajectories_path",
        "grpo_groups_path",
        "manifest_path",
        "sync_receipt_path",
        "post_update_trajectories_path",
    ):
        leaf = Path(str(config["storage"][field])).name
        config["storage"][field] = f"artifacts/hotpot_step1/{leaf}"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, root / "artifacts" / "cursor1.json"


def create_joint_qa_micro_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, _ = create_project(tmp_path)
    validation_path = root / "data" / "validation.jsonl"
    test_path = root / "data" / "test.jsonl"
    for path, split, index in (
        (validation_path, "validation", 100),
        (test_path, "test", 101),
    ):
        rows = []
        for source in ("hotpotqa", "triviaqa"):
            task = TaskRecord(
                task_id=f"{source}:{split}-{index}",
                question="Held-out question?",
                ground_truth="answer",
                split=split,
                metadata={"dataset_key": source, "source": SOURCE_NAMES[source]},
            )
            rows.append(json.dumps(aligned_row(task)) + "\n")
        path.write_text("".join(rows), encoding="utf-8")

    schedule = freeze_joint_qa_training_schedule(
        train_path=root / "data" / "train.jsonl",
        validation_path=validation_path,
        test_path=test_path,
        task_positions_by_dataset={"hotpotqa": (0,), "triviaqa": (0,)},
        rollouts_per_task=8,
    )
    schedule_path = root / "artifacts" / "joint_schedule.json"
    cursor_path = root / "artifacts" / "joint_cursor0.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    JointQATrainingCursorState.fresh(schedule).write_once(cursor_path)

    config = yaml.safe_load(
        Path("config/training_joint_qa_step1.yaml").read_text(encoding="utf-8")
    )
    config["data"].update(
        train_path="data/train.jsonl",
        validation_path="data/validation.jsonl",
        test_path="data/test.jsonl",
    )
    config["data"]["joint_qa_micro"].update(
        schedule_path="artifacts/joint_schedule.json",
        cursor_path="artifacts/joint_cursor0.json",
        next_cursor_path="artifacts/joint_cursor1.json",
    )
    config["experiment"]["output_dir"] = "artifacts/joint_step1/training"
    for field in (
        "root",
        "selected_tasks_path",
        "retrieval_receipts_path",
        "trajectories_path",
        "grpo_groups_path",
        "manifest_path",
        "behavior_policy_preflight_path",
        "sync_receipt_path",
        "post_update_trajectories_path",
    ):
        leaf = Path(str(config["storage"][field])).name
        config["storage"][field] = f"artifacts/joint_step1/{leaf}"
    trivia = make_task("triviaqa", 0)
    receipt = QARetrievalReceipt(
        query=build_keyword_query(trivia.question),
        search_limit=5,
        passages=(),
    )
    retrieval_path = root / config["storage"]["retrieval_receipts_path"]
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "schema_version": "flowsteer.triviaqa.public_retrieval.v1",
                "task_id": trivia.task_id,
                "question": trivia.question,
                "retrieval": receipt.to_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = root / "config" / "joint.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, root / "artifacts" / "joint_cursor1.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_fake_evidence(trajectory_path: Path, evidence_path: Path) -> None:
    trajectory_rows = read_jsonl(trajectory_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_kind": "trajectory",
                    "event_id": row["trajectory_id"],
                    "payload": row,
                }
            )
            + "\n"
            for row in trajectory_rows
        ),
        encoding="utf-8",
    )


class SelectionTests(unittest.TestCase):
    def test_bounds_require_raw_on_policy_sampling_and_fixed_oom_schedule(self) -> None:
        config = yaml.safe_load(
            Path("config/training_agentgraph_smoke.yaml").read_text(encoding="utf-8")
        )
        invalid_sampling = copy.deepcopy(config)
        invalid_sampling["director"]["top_p"] = 0.95
        with self.assertRaisesRegex(Exception, "director.top_p"):
            validate_smoke_bounds(invalid_sampling)

        invalid_backoff = copy.deepcopy(config)
        invalid_backoff["gpu"]["oom_policy"]["micro_batch_schedule"] = [2, 1]
        with self.assertRaisesRegex(Exception, "micro_batch_schedule"):
            validate_smoke_bounds(invalid_backoff)

    def test_bounds_require_explicit_healthbench_judge(self) -> None:
        config = yaml.safe_load(
            Path("config/training_agentgraph_smoke.yaml").read_text(encoding="utf-8")
        )
        config["evaluation"]["healthbench_judge_model"] = ""
        with self.assertRaisesRegex(Exception, "healthbench_judge_model"):
            validate_smoke_bounds(config)

    def test_hotpot_micro_bounds_require_zero_shaping_rewards(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, config_path, _ = create_hotpot_micro_project(Path(directory))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_smoke_bounds(config)
            config["grpo"]["structural_reward"] = 0.1
            with self.assertRaisesRegex(Exception, "structural_reward"):
                validate_smoke_bounds(config)

    def test_joint_qa_bounds_require_two_groups_and_two_canaries(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, config_path, _ = create_joint_qa_micro_project(Path(directory))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_smoke_bounds(config)
            config["grpo"]["expected_rollout_count"] = 8
            with self.assertRaisesRegex(Exception, "expected_rollout_count"):
                validate_smoke_bounds(config)

    def test_joint_qa_bounds_allow_only_frozen_versioned_skill_on(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, config_path, _ = create_joint_qa_micro_project(Path(directory))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["skills"].update(
                enabled=True,
                frozen_store=True,
                store_path="artifacts/skill_epoch/skills.json",
                retrieval_top_k=2,
                current_epoch=2,
                library_version="jointqa.skill-library.progressive.epoch2.v1",
                posterior_version="jointqa.posterior.progressive.epoch0.v1",
                required_skill_ids=["jointqa.hotpotqa.format", "jointqa.triviaqa.format"],
            )
            config["deployment"].update(
                active_skills_only=True,
                allow_forced_probes=False,
                exploration_beta=0.0,
            )
            validate_smoke_bounds(config)

            not_frozen = copy.deepcopy(config)
            not_frozen["skills"]["frozen_store"] = False
            with self.assertRaisesRegex(Exception, "skills.enabled"):
                validate_smoke_bounds(not_frozen)

            missing_version = copy.deepcopy(config)
            missing_version["skills"]["posterior_version"] = "none"
            with self.assertRaisesRegex(Exception, "skills.enabled"):
                validate_smoke_bounds(missing_version)

            duplicate_ids = copy.deepcopy(config)
            duplicate_ids["skills"]["required_skill_ids"] = ["same", "same"]
            with self.assertRaisesRegex(Exception, "required_skill_ids"):
                validate_smoke_bounds(duplicate_ids)

    def test_joint_qa_prepare_freezes_both_tasks_and_trivia_retrieval(self) -> None:
        import asyncio
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root, config_path, _ = create_joint_qa_micro_project(Path(directory))
            manifest = asyncio.run(
                run_smoke(config_path, prepare_only=True, project_root=root)
            )
            self.assertEqual("prepared", manifest["status"])
            self.assertEqual(2, manifest["bounds"]["post_update_canaries"])
            self.assertEqual(
                {"hotpotqa": 1, "triviaqa": 1},
                manifest["selected_by_source"],
            )
            selected = read_jsonl(
                root / "artifacts/joint_step1/selected_tasks.jsonl"
            )
            self.assertEqual(2, len(selected))
            self.assertIn(
                "Public retrieval observations (SkillFlow search/read)",
                selected[1]["question"],
            )

    def test_policy_update_persists_active_skill_suspension(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            store = SkillStore(Path(directory) / "skills.json")
            evidence = SkillEvidence(
                baseline="frozen-policy",
                paired_effect_mean=0.1,
                calibrated_lower=0.05,
                calibrated_upper=0.15,
                effective_pairs=2,
                independent_problem_ids=("v1", "v2"),
                discovery_problem_ids=("d1",),
                validation_problem_ids=("v1", "v2"),
                validation_splits=("validation",),
                heldout_task_families=("hotpotqa",),
                empirical_coverage=0.95,
                harm_probability=0.0,
                slice_effects={"hotpotqa": 0.1},
                evidence_ids=("e1", "e2"),
            )
            active = SkillRecord(
                skill_id="jointqa.hotpotqa.topology",
                version=1,
                status=SkillStatus.ACTIVE,
                condition={"task_family": "hotpotqa", "graph_stage": "*"},
                action={"instruction": "Use the validated dependency relation."},
                evidence=evidence,
                versions=VersionBundle(
                    policy="policy-step-0",
                    model_catalog="catalog-v1",
                    evaluator="hotpotqa.official.answer.v1",
                    prompt="prompt-v1",
                    tool="tool-v1",
                    posterior="posterior-v1",
                    skill_library="library-v1",
                ),
                created_epoch=0,
                eligible_epoch=1,
                activated_epoch=1,
                gate_config={"delta_min": 0.03},
                gate_receipt="gate-receipt",
            )
            store.upsert(active)
            pipeline = SimpleNamespace(
                skill_store=store,
                lifecycle=SkillLifecycleManager(),
            )
            receipt = audit_active_skills_after_policy_update(
                SimpleNamespace(skill_pipeline=pipeline),
                behavior_policy="policy-step-0",
                updated_policy="policy-step-1",
                adapter_name="theta-step-1",
                require_active_skills=True,
            )

            persisted = store.get(active.skill_id)
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual(SkillStatus.SUSPENDED, persisted.status)
            self.assertIn("policy", persisted.suspended_reason or "")
            self.assertEqual("completed", receipt["status"])
            self.assertEqual(1, receipt["suspended_skills"])

    def test_source_order_and_unique_base_task_are_enforced(self) -> None:
        tasks = [
            make_task("triviaqa", 0),
            make_task("hotpotqa", 0, base_id="same"),
            make_task("hotpotqa", 1, base_id="same"),
            make_task("hotpotqa", 2, base_id="different"),
            make_task("triviaqa", 1),
        ]
        selected = select_smoke_tasks(
            tasks,
            source_order=("hotpotqa", "triviaqa"),
            per_source=2,
            require_unique_base_tasks=True,
        )
        self.assertEqual(
            ["hotpotqa:0", "hotpotqa:2", "triviaqa:0", "triviaqa:1"],
            [item.task_id for item in selected],
        )

    def test_each_source_has_the_required_evaluator_version(self) -> None:
        versions = {
            source: evaluator_version_for(make_task(source, 0))
            for source in EXPECTED_SOURCE_ORDER
        }
        self.assertEqual("hotpotqa.official.answer.v1", versions["hotpotqa"])
        self.assertEqual("triviaqa.official.answer.v1", versions["triviaqa"])
        self.assertEqual(AIME2026_EVALUATOR_VERSION, versions["aime_2026"])
        self.assertEqual(versions["webshop"], versions["alfworld"])
        self.assertEqual(6, len(set(versions.values())))

    def test_exact_resume_preserves_but_excludes_a_malformed_atomic_action(self) -> None:
        task = make_task("hotpotqa", 0)
        versions = _MODULE.version_bundle_for(
            task,
            policy_version="qwen35-9b-base-step-0000",
            model_catalog_version="catalog-test-v1",
        )
        invalid = trajectory(task, 0, versions)
        invalid_turn = replace(
            invalid.turns[0],
            executed_prefix_tokens=0,
            action={},
            canvas_feedback="invalid action: malformed JSON",
        )
        invalid = replace(invalid, turns=(invalid_turn,))
        valid = trajectory(task, 1, versions)
        third = trajectory(task, 2, versions)

        validate_resumed_initial_rollouts(
            (invalid, valid, third),
            ((task, 0, versions), (task, 1, versions), (task, 2, versions)),
            condition_id="natural_smoke",
            sampling_anchor_ordinal=0,
            behavior_adapter_name=None,
            expected_server_weight_version="default",
        )

        self.assertFalse(invalid.grpo_eligible)
        self.assertTrue(valid.grpo_eligible)


class SmokeRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_only_needs_no_secret_and_writes_exactly_14(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        with patch.dict(os.environ, {}, clear=True):
            manifest = await run_smoke(
                config_path,
                prepare_only=True,
                project_root=root,
            )
        self.assertEqual("prepared", manifest["status"])
        selected = read_jsonl(root / "artifacts/smoke/data/selected_tasks.jsonl")
        self.assertEqual(14, len(selected))
        self.assertEqual(
            list(EXPECTED_SOURCE_ORDER),
            [selected[index * 2]["metadata"]["dataset_key"] for index in range(7)],
        )

    async def test_full_pipeline_orders_update_publish_and_updated_canary(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            backend=backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(28, len(backend.train_inputs))
        self.assertEqual(28, len(read_jsonl(root / "artifacts/smoke/data/trajectories.jsonl")))
        canaries = read_jsonl(
            root / "artifacts/smoke/data/post_update_trajectories.jsonl"
        )
        self.assertEqual(1, len(canaries))
        self.assertEqual(
            "qwen35-9b-smoke-step-0001",
            canaries[0]["versions"]["policy"],
        )
        self.assertEqual(
            "theta_smoke_step_000001",
            canaries[0]["turns"][0]["policy_adapter"],
        )
        self.assertLess(backend.events.index("train"), backend.events.index("publish"))
        self.assertLess(
            backend.events.index("publish"), backend.events.index("collect:10000")
        )
        groups = read_jsonl(root / "artifacts/smoke/data/grpo_groups.jsonl")
        self.assertEqual(14, len(groups))
        self.assertTrue(all(row["informative"] for row in groups))

    async def test_hotpot_micro_executes_only_current_frozen_step_and_commits_cursor(
        self,
    ) -> None:
        root, config_path, next_cursor_path = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            backend=backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(1, manifest["bounds"]["selected_tasks"])
        self.assertEqual(2, manifest["bounds"]["expected_initial_rollouts"])
        self.assertEqual(["collect:0", "collect:1"], backend.events[:2])
        self.assertEqual(2, len(backend.train_inputs))
        committed = HotpotTrainingCursorState.read(next_cursor_path)
        self.assertEqual(1, committed.cursor)
        self.assertEqual(
            committed.to_value(),
            manifest["selection_receipt"]["cursor_after"],
        )

    async def test_hotpot_micro_strict_resume_skips_initial_collection(self) -> None:
        root, config_path, _ = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # The fake trajectory helper predates formal step ordinals; pin its
        # existing sampling anchor explicitly for this persistence test.
        config["experiment"]["sampling_anchor_ordinal"] = 0
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        failed_backend = FakeBackend(updates=0)
        with self.assertRaisesRegex(SmokeRunError, "zero optimizer updates"):
            await run_smoke(
                config_path,
                backend=failed_backend,
                project_root=root,
            )
        self.assertEqual(
            ["collect:0", "collect:1", "train"], failed_backend.events
        )
        write_fake_evidence(
            root / "artifacts/hotpot_step1/trajectories.jsonl",
            root / "artifacts/hotpot_step1/evidence/trajectories.jsonl",
        )

        resumed_backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            resume_initial_rollouts=True,
            backend=resumed_backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(
            ["train", "publish", "collect:10000"], resumed_backend.events
        )
        self.assertEqual(
            {
                "mode": "strict_persisted_resume",
                "path": str(
                    root / "artifacts/hotpot_step1/trajectories.jsonl"
                ),
                "reused": 2,
                "new_collections": 0,
            },
            manifest["initial_rollout_source"],
        )

    async def test_hotpot_micro_resume_after_precheckpoint_runtime_failure(
        self,
    ) -> None:
        class PrecheckpointFailureBackend(FakeBackend):
            def train(self, trajectories, output_dir):
                del output_dir
                self.events.append("train")
                self.train_inputs = list(trajectories)
                raise RuntimeError("replica unavailable before checkpoint")

        root, config_path, _ = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment"]["sampling_anchor_ordinal"] = 0
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        failed_backend = PrecheckpointFailureBackend()
        with self.assertRaisesRegex(SmokeRunError, "one-pass smoke training failed"):
            await run_smoke(
                config_path,
                backend=failed_backend,
                project_root=root,
            )
        write_fake_evidence(
            root / "artifacts/hotpot_step1/trajectories.jsonl",
            root / "artifacts/hotpot_step1/evidence/trajectories.jsonl",
        )

        resumed_backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            resume_initial_rollouts=True,
            backend=resumed_backend,
            project_root=root,
        )

        self.assertEqual("completed", manifest["status"])
        self.assertEqual(
            ["train", "publish", "collect:10000"], resumed_backend.events
        )
        self.assertEqual(
            "failed_training_before_persistence",
            manifest["resume_preconditions"]["root_zero_update_status"],
        )

    async def test_zero_update_is_a_failed_run_and_never_publishes(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        backend = FakeBackend(updates=0)
        with self.assertRaisesRegex(SmokeRunError, "zero optimizer updates"):
            await run_smoke(config_path, backend=backend, project_root=root)
        self.assertNotIn("publish", backend.events)
        sync = json.loads(
            (root / "artifacts/smoke/data/sync_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("not_attempted_no_optimizer_update", sync["status"])
        manifest = json.loads(
            (root / "artifacts/smoke/data/training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("failed_no_optimizer_update", manifest["status"])

    async def test_live_backend_switches_director_route_inside_publisher_gate(
        self,
    ) -> None:
        class RecordingGate(_MODULE.RolloutGate):
            def __init__(self) -> None:
                super().__init__(poll_interval_seconds=0.001)
                self.events: list[str] = []

            def pause(self) -> None:
                self.events.append("pause")
                super().pause()

            def drain(self, timeout_seconds=None) -> None:
                self.events.append("drain")
                super().drain(timeout_seconds)

            def resume(self) -> None:
                self.events.append("resume")
                super().resume()

        class RouteClient:
            def __init__(self, gate) -> None:
                self.gate = gate
                self.policy_version = "behavior-v0"
                self.adapter_name = "theta_smoke_step_000000"
                self.expected_server_weight_version = "server-v0"
                self.updates: list[tuple[str, str | None, str | None]] = []

            def update_policy_route(
                self,
                *,
                policy_version,
                adapter_name,
                expected_server_weight_version,
            ) -> None:
                self.gate.require_paused_and_drained()
                self.policy_version = policy_version
                self.adapter_name = adapter_name
                self.expected_server_weight_version = expected_server_weight_version
                self.updates.append(
                    (
                        policy_version,
                        adapter_name,
                        expected_server_weight_version,
                    )
                )

        class TransactionalPublisher:
            def __init__(self, client) -> None:
                self.client = client

            def publish(self, **kwargs):
                gate = kwargs["gate"]
                gate.pause()
                try:
                    gate.drain()
                    kwargs["route_switch"](
                        "qwen35-9b-smoke-step-0001",
                        "theta_smoke_step_000001",
                    )
                    assert self.client.adapter_name == "theta_smoke_step_000001"
                finally:
                    gate.resume()
                return Receipt()

        gate = RecordingGate()
        client = RouteClient(gate)
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = {
            "director": {
                "behavior_adapter_name": "theta_smoke_step_000000",
                "expected_server_weight_version": "server-v1",
            },
            "experiment": {"update_step": 1},
        }
        backend.director_client = client
        backend.rollout_gate = gate
        backend.publisher = TransactionalPublisher(client)

        summary = Summary(Path(self._temp_dir.name))
        receipt = await backend.publish(summary)

        self.assertIsInstance(receipt, Receipt)
        self.assertEqual(gate.events, ["pause", "drain", "resume"])
        self.assertFalse(gate.paused)
        self.assertEqual(
            client.updates,
            [
                (
                    "qwen35-9b-smoke-step-0001",
                    "theta_smoke_step_000001",
                    "server-v1",
                )
            ],
        )

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
