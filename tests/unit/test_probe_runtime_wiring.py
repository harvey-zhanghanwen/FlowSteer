from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentResponse, AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.records import TaskRecord
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)
from src.interactive.versioning import VersionBundle


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_agentgraph_smoke.py"
_SPEC = importlib.util.spec_from_file_location("probe_runtime_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_SCRIPTS_ROOT = _SCRIPT.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))
from scripts import run_joint_qa_mace_skill as _RUNNER  # noqa: E402


@dataclass(frozen=True)
class _TrajectoryStub:
    turns: tuple
    condition_satisfied: bool = False


class _RecordingEvidenceStore:
    def __init__(self) -> None:
        self.events = []

    def append_snapshot(self, record) -> None:
        self.events.append(("snapshot", record))

    def append_trajectory(self, record) -> None:
        self.events.append(("trajectory", record))


class _Gateway:
    async def generate(self, request):
        return AgentResponse("unused")


class _DirectorClient:
    def executed_prefix_tokens(self, response, action):
        return 0


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("local", endpoint="http://127.0.0.1:1/v1")],
        [ModelSpec("qwen", "local")],
    )


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="hotpotqa:probe-1",
        question="Question?",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa", "task_family": "hotpotqa"},
    )


def _versions() -> VersionBundle:
    return VersionBundle(
        policy="policy-step2",
        model_catalog="catalog-test-v1",
        evaluator="hotpotqa.official.answer.v1",
        prompt="prompt-v1",
        tool="tool-v1",
    )


def _tool_registration(
    tool_id: str,
    *,
    dataset_scope: tuple[str, ...],
    availability: bool,
) -> ToolRegistration:
    return ToolRegistration(
        tool_id,
        FakeTool({"search": lambda arguments: {"query": arguments["query"]}}),
        ToolCapability(
            tool_id=tool_id,
            dataset_scope=dataset_scope,
            action_schemas={
                "search": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                }
            },
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            side_effect="none",
            timeout_seconds=1.0,
            version="tool-projection-test-v1",
            availability=availability,
        ),
    )


def _tool_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            _tool_registration(
                "hotpotqa.search",
                dataset_scope=("hotpotqa",),
                availability=True,
            ),
            _tool_registration(
                "hotpotqa.unavailable",
                dataset_scope=("hotpotqa",),
                availability=False,
            ),
            _tool_registration(
                "shared.search",
                dataset_scope=("*",),
                availability=True,
            ),
            _tool_registration(
                "triviaqa.search",
                dataset_scope=("triviaqa",),
                availability=True,
            ),
        )
    )


def _config() -> dict:
    return {
        "director": {
            "max_rounds": 8,
            "history_window": 4,
            "execute_on_edit": True,
        },
        "agent_graph": {
            "max_agents": 6,
            "terminal_protocol_by_source": {"hotpotqa": "exact_single_answer_tag"},
        },
        "experiment": {
            "seed": 41,
            "condition_id": "configured-condition",
            "sampling_schedule_purpose": "configured-shared-coordinate",
            "catalog_order_namespace": "paired-probe-catalog",
            "sampling_anchor_ordinal": 2,
        },
        "evaluation": {
            "max_environment_steps": 2,
            "max_environment_steps_by_source": {},
        },
    }


class ProbeRuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    def _backend(self):
        registry = _registry()
        return _MODULE.LiveSmokeBackend(
            config=_config(),
            registry=registry,
            runtime=AgentRuntime(registry, _Gateway()),
            director_client=_DirectorClient(),
            rollout_gate=SimpleNamespace(),
            evidence_store=None,
            trainer=None,
            publisher=SimpleNamespace(),
            judge=None,
            judge_model="",
        )

    async def test_paired_arms_share_sampling_coordinate_not_condition_receipt(
        self,
    ) -> None:
        captured = []
        sentinel = object()

        class CapturingCollector:
            def __init__(self, orchestrator, environment, versions, store, **kwargs):
                captured.append((orchestrator, kwargs))

            async def collect(
                self,
                task,
                rollout_index,
                evaluator_callback,
                *,
                workflow_problem=None,
            ):
                del task, rollout_index, evaluator_callback, workflow_problem
                return sentinel

        condition = {
            "condition_id": "candidate-a",
            "application_mode": "forced_probe_condition",
            "content": "Use the predeclared candidate instruction.",
        }
        backend = self._backend()
        with patch.object(_MODULE, "AgentGraphRolloutCollector", CapturingCollector):
            first = await backend.collect(
                _task(),
                0,
                _versions(),
                expected_task_split="validation",
                condition_id="candidate-arm",
                sampling_schedule_purpose="paired-probe-17",
                prompt_priors=(condition,),
                forced_probe=True,
                condition_satisfied=True,
                sampling_anchor_ordinal=7,
            )
            second = await backend.collect(
                _task(),
                0,
                _versions(),
                expected_task_split="validation",
                condition_id="incumbent-arm",
                sampling_schedule_purpose="paired-probe-17",
                forced_probe=True,
                condition_satisfied=True,
                sampling_anchor_ordinal=7,
            )

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        first_orchestrator, first_kwargs = captured[0]
        second_orchestrator, second_kwargs = captured[1]
        self.assertEqual(
            first_orchestrator.sampling_coordinate,
            second_orchestrator.sampling_coordinate,
        )
        self.assertEqual(
            first_orchestrator.generation_seed(0),
            second_orchestrator.generation_seed(0),
        )
        self.assertEqual("candidate-arm", first_kwargs["condition_id"])
        self.assertEqual("incumbent-arm", second_kwargs["condition_id"])
        self.assertEqual((condition,), first_kwargs["skills"])
        self.assertEqual((), second_kwargs["skills"])
        self.assertTrue(first_kwargs["forced_probe"])
        self.assertEqual("validation", first_kwargs["expected_task_split"])

    def test_skill_query_projects_only_available_dataset_scoped_tool_ids(
        self,
    ) -> None:
        backend = self._backend()
        tool_registry = _tool_registry()
        runtime = AgentRuntime(
            backend.registry,
            _Gateway(),
            tool_registry=tool_registry,
            dataset_id="hotpotqa",
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=_task().question,
        )

        query = backend._skill_query(_task(), environment)

        self.assertEqual(
            ("hotpotqa.search", "shared.search"),
            query.available_tools,
        )
        self.assertNotIn("hotpotqa.unavailable", query.available_tools)
        self.assertNotIn("triviaqa.search", query.available_tools)

    def test_skill_query_without_tool_registry_has_no_available_tools(self) -> None:
        backend = self._backend()
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=backend.runtime,
            problem=_task().question,
        )

        query = backend._skill_query(_task(), environment)

        self.assertEqual((), query.available_tools)

    def test_forced_probe_required_tools_match_natural_retrieval_subset(
        self,
    ) -> None:
        backend = self._backend()
        runtime = AgentRuntime(
            backend.registry,
            _Gateway(),
            tool_registry=_tool_registry(),
            dataset_id="hotpotqa",
        )
        environment = AgentWorkflowEnv(
            backend.registry,
            runtime=runtime,
            problem=_task().question,
        )
        query = backend._skill_query(_task(), environment)

        def condition(required_tools: tuple[str, ...]) -> dict:
            return {
                "task_family": "hotpotqa",
                "graph_stage": "empty_graph",
                "tags": [],
                "required_tools": list(required_tools),
            }

        for required_tools, expected in (
            ((), True),
            (("hotpotqa.search",), True),
            (("hotpotqa.search", "shared.search"), True),
            (("hotpotqa.unavailable",), False),
            (("triviaqa.search",), False),
        ):
            scope = condition(required_tools)
            forced_match = backend._forced_probe_condition_matches(
                {"condition": scope, "action": {}},
                query,
            )
            self.assertEqual(
                query.required_tools_satisfied(scope),
                forced_match,
            )
            self.assertEqual(expected, forced_match)

    async def test_configured_schedule_purpose_is_the_default(self) -> None:
        captured = []

        class CapturingCollector:
            def __init__(self, orchestrator, environment, versions, store, **kwargs):
                captured.append((orchestrator, kwargs))

            async def collect(
                self,
                task,
                rollout_index,
                evaluator_callback,
                *,
                workflow_problem=None,
            ):
                del task, rollout_index, evaluator_callback, workflow_problem
                return object()

        backend = self._backend()
        with patch.object(_MODULE, "AgentGraphRolloutCollector", CapturingCollector):
            await backend.collect(
                _task(),
                3,
                _versions(),
                expected_task_split="validation",
            )

        orchestrator, kwargs = captured[0]
        self.assertEqual(
            "configured-shared-coordinate",
            orchestrator.sampling_coordinate.schedule_purpose,
        )
        self.assertEqual("configured-condition", kwargs["condition_id"])
        self.assertFalse(kwargs["forced_probe"])
        self.assertIsNone(kwargs["skill_provider"])

    async def test_static_probe_condition_cannot_mix_with_active_skill_provider(
        self,
    ) -> None:
        backend = self._backend()
        backend.skill_pipeline = SimpleNamespace()
        condition = {
            "application_mode": "forced_probe_condition",
            "content": "candidate",
        }
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            await backend.collect(
                _task(),
                0,
                _versions(),
                expected_task_split="validation",
                prompt_priors=(condition,),
                forced_probe=True,
            )

    async def test_stage_conditioned_probe_uses_graph_stage_and_persists_after_receipt(
        self,
    ) -> None:
        captured = {}
        store = _RecordingEvidenceStore()
        condition = {
            "condition_id": "candidate-stage",
            "application_mode": "forced_probe_condition",
            "condition": {
                "task_family": "hotpotqa",
                "graph_stage": "before_final_answer",
                "tags": [],
            },
            "action": {"instruction": "Preserve the supported answer span."},
            "content": "candidate",
        }

        class CapturingCollector:
            def __init__(
                self, orchestrator, environment, versions, evidence_store, **kwargs
            ):
                captured["collector_store"] = evidence_store
                captured["provider"] = kwargs["skill_provider"]
                captured["initial_condition_satisfied"] = kwargs["condition_satisfied"]
                self.environment = environment
                self.versions = versions

            async def collect(
                self,
                task,
                rollout_index,
                evaluator_callback,
                *,
                workflow_problem=None,
            ):
                del rollout_index, evaluator_callback, workflow_problem
                self.environment.reset(task.question)
                provider = captured["provider"]
                captured["empty"] = provider(task, self.environment, self.versions)
                self.environment._graph = AgentGraph(
                    (AgentNode("evidence", "qwen", "find evidence"),)
                )
                captured["construction"] = provider(
                    task,
                    self.environment,
                    self.versions,
                )
                self.environment._graph = AgentGraph(
                    (AgentNode("format", "qwen", "format answer"),),
                    output_agent_id="format",
                )
                captured["before_final"] = provider(
                    task,
                    self.environment,
                    self.versions,
                )
                turn = SimpleNamespace(
                    round_index=2,
                    receipt_verified=True,
                    prompt=(
                        '{"exploration_conditions":['
                        '{"condition_id":"candidate-stage"}]}'
                    ),
                    graph_revision=1,
                    graph_snapshot=self.environment.graph.to_dict(),
                    graph_snapshot_id="snapshot-1",
                    previous_graph_snapshot_id=None,
                )
                return _TrajectoryStub((turn,))

        backend = self._backend()
        backend.evidence_store = store
        with patch.object(_MODULE, "AgentGraphRolloutCollector", CapturingCollector):
            record = await backend.collect(
                _task(),
                0,
                _versions(),
                expected_task_split="validation",
                condition_id="candidate-arm",
                sampling_schedule_purpose="paired-probe-stage",
                stage_conditioned_prompt_prior=condition,
                forced_probe=True,
                sampling_anchor_ordinal=9,
            )

        self.assertIsNone(captured["collector_store"])
        self.assertFalse(captured["initial_condition_satisfied"])
        self.assertEqual((), captured["empty"])
        self.assertEqual((), captured["construction"])
        self.assertEqual((condition,), captured["before_final"])
        self.assertTrue(record.condition_satisfied)
        self.assertEqual(["snapshot", "trajectory"], [name for name, _ in store.events])
        self.assertIs(record, store.events[-1][1])

    async def test_stage_conditioned_probe_requires_forced_probe_and_exact_stage(
        self,
    ) -> None:
        condition = {
            "condition_id": "candidate-stage",
            "application_mode": "forced_probe_condition",
            "condition": {
                "task_family": "hotpotqa",
                "graph_stage": "*",
                "tags": [],
            },
            "action": {"instruction": "candidate"},
        }
        backend = self._backend()
        with self.assertRaisesRegex(ValueError, "require forced_probe=true"):
            await backend.collect(
                _task(),
                0,
                _versions(),
                stage_conditioned_prompt_prior=condition,
            )
        with self.assertRaisesRegex(ValueError, "require one exact graph_stage"):
            await backend.collect(
                _task(),
                0,
                _versions(),
                stage_conditioned_prompt_prior=condition,
                forced_probe=True,
            )

    async def test_unreached_stage_is_persisted_as_unexposed_itt_assignment(
        self,
    ) -> None:
        store = _RecordingEvidenceStore()
        condition = {
            "condition_id": "candidate-never-visible",
            "application_mode": "forced_probe_condition",
            "condition": {
                "task_family": "hotpotqa",
                "graph_stage": "before_final_answer",
                "tags": [],
            },
            "action": {"instruction": "Preserve the supported answer span."},
            "content": "candidate",
        }

        class NeverExposedCollector:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def collect(
                self,
                task,
                rollout_index,
                evaluator_callback,
                *,
                workflow_problem=None,
            ):
                del task, rollout_index, evaluator_callback, workflow_problem
                turn = SimpleNamespace(
                    round_index=0,
                    receipt_verified=True,
                    prompt='{"current_graph":{"nodes":[]}}',
                    graph_revision=0,
                    graph_snapshot={
                        "nodes": [],
                        "relations": [],
                        "output_agent_id": None,
                        "revision": 0,
                    },
                    graph_snapshot_id="snapshot-empty",
                    previous_graph_snapshot_id=None,
                )
                return _TrajectoryStub((turn,))

        backend = self._backend()
        backend.evidence_store = store
        with patch.object(_MODULE, "AgentGraphRolloutCollector", NeverExposedCollector):
            record = await backend.collect(
                _task(),
                0,
                _versions(),
                expected_task_split="validation",
                condition_id="candidate-arm",
                sampling_schedule_purpose="paired-probe-stage",
                stage_conditioned_prompt_prior=condition,
                forced_probe=True,
                sampling_anchor_ordinal=10,
            )

        self.assertFalse(record.condition_satisfied)
        self.assertEqual(
            (),
            backend._prompt_prior_exposure_rounds(
                record,
                condition["condition_id"],
            ),
        )
        self.assertEqual(["snapshot", "trajectory"], [name for name, _ in store.events])
        self.assertIs(record, store.events[-1][1])

    async def test_runner_forwards_candidate_specific_stage_prior(self) -> None:
        spec = replace(
            _RUNNER.EPOCH6_SPEC,
            candidate_graph_stages={
                "conditional_fan_in_deferred_format": "construction",
                "exact_answer_handoff": "before_final_answer",
            },
        )
        prior = _RUNNER._prompt_condition(
            "hotpotqa",
            "exact_answer_handoff",
            spec,
        )
        self.assertEqual(
            "before_final_answer",
            prior["condition"]["graph_stage"],
        )
        captured = {}
        sentinel = object()

        class Backend:
            model_catalog_version = "catalog-test-v1"

            async def collect(self, task, rollout_index, versions, **kwargs):
                captured.update(kwargs)
                return sentinel

        result = await _RUNNER._arm(
            Backend(),
            {},
            _task(),
            condition_id="candidate-arm",
            schedule_purpose="paired-stage",
            prompt_priors=(),
            stage_conditioned_prompt_prior=prior,
            forced_probe=True,
            anchor=11,
            spec=spec,
        )

        self.assertIs(sentinel, result)
        self.assertEqual((), captured["prompt_priors"])
        self.assertEqual(prior, captured["stage_conditioned_prompt_prior"])
        self.assertTrue(captured["forced_probe"])

    def test_version_bundle_can_bind_exploration_and_skill_versions(self) -> None:
        versions = _MODULE.version_bundle_for(
            _task(),
            policy_version="policy-step2",
            model_catalog_version="catalog-test-v1",
            encoder_version="encoder-v1",
            feature_schema_version="features-v1",
            posterior_version="posterior-v1",
            skill_library_version="skills-v1",
        )
        self.assertEqual("encoder-v1", versions.encoder)
        self.assertEqual("features-v1", versions.feature_schema)
        self.assertEqual("posterior-v1", versions.posterior)
        self.assertEqual("skills-v1", versions.skill_library)


if __name__ == "__main__":
    unittest.main()
