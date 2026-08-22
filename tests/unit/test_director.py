from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
    DIRECTOR_PROMPT_VERSION,
    DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION,
    DIRECTOR_SYSTEM_PROMPT,
    HOTPOTQA_DIRECTOR_PROMPT_VERSION,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
    HOTPOTQA_SEMANTIC_PROTOCOL,
    LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
    LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
    PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
    DirectorResponse,
    OpenAIDirectorClient,
    decode_director_transcript,
    director_action_json_schema_text,
    director_system_prompt_for_version,
    director_state_conditioned_sampling_json_schema_text,
    encode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)


class ScriptedDirector:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []
        self.seeds = []
        self.schema_requests = []

    async def propose(
        self,
        prompt,
        *,
        seed=None,
        action_json_schema=None,
        action_json_schema_version=None,
        action_schema_branch=None,
    ):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        self.schema_requests.append(
            {
                "action_json_schema": action_json_schema,
                "action_json_schema_version": action_json_schema_version,
                "action_schema_branch": action_schema_branch,
            }
        )
        return DirectorResponse(self.actions.pop(0), {"policy_version": "test"})


class FakeGateway:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(f"answer from {request.agent.id}")


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", endpoint="http://local/v1")],
        [
            ModelSpec(
                "qwen",
                "provider",
                cheap_weight=10,
                fast_weight=10,
                metadata={
                    "family": "qwen",
                    "profile": "text_qa",
                    "text_qa_canary": "passed",
                    "max_tokens": "512",
                },
            ),
            ModelSpec("other", "provider", cheap_weight=1, fast_weight=1),
        ],
    )


def transcript_messages(prompt: str) -> tuple[dict[str, str], ...]:
    messages = decode_director_transcript(prompt)
    if messages is None:
        raise AssertionError("expected a canonical Director transcript")
    return tuple(dict(message) for message in messages)


def observation_payload(message: dict[str, str]) -> dict[str, object]:
    if message["role"] != "user":
        raise AssertionError("Canvas observation must be a user message")
    heading, separator, raw_payload = message["content"].partition("\n\n")
    if not separator or not heading.startswith("Canvas observation."):
        raise AssertionError("Canvas observation message has no JSON payload")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise AssertionError("Canvas observation payload must be an object")
    return payload


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_canonical_transcript_remains_decodable(self) -> None:
        legacy = encode_director_transcript(
            (
                {
                    "role": "system",
                    "content": LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
                },
                {"role": "user", "content": "legacy Canvas observation"},
            )
        )

        decoded = decode_director_transcript(legacy)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(LEGACY_DIRECTOR_SYSTEM_PROMPT_V8, decoded[0]["content"])
        self.assertEqual(
            "agentgraph.director.minimal-neutral.v10",
            DIRECTOR_PROMPT_VERSION,
        )

    async def test_versioned_prompt_resolver_preserves_legacy_and_default(self) -> None:
        self.assertIs(
            DIRECTOR_SYSTEM_PROMPT,
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION),
        )
        self.assertIs(
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
            director_system_prompt_for_version(
                "agentgraph.director.minimal-neutral.v9"
            ),
        )
        self.assertIs(
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
            director_system_prompt_for_version(
                "agentgraph.director.constrained-action.skillflow-qa.v8"
            ),
        )
        self.assertIs(
            DIRECTOR_SYSTEM_PROMPT,
            director_system_prompt_for_version("prompt-v1"),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
            director_system_prompt_for_version(HOTPOTQA_DIRECTOR_PROMPT_VERSION),
        )
        self.assertIs(
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
            director_system_prompt_for_version(
                "agentgraph.director.hotpotqa-semantic-recovery.v12"
            ),
        )
        with self.assertRaises(ValueError):
            director_system_prompt_for_version(" ")

    async def test_hotpot_v13_prompt_encodes_semantic_and_recovery_policy(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="Which player won more titles?",
            semantic_protocol=HOTPOTQA_SEMANTIC_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            required_evidence_tool_id="qa-retrieval",
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            system_prompt=HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
            prompt_version=HOTPOTQA_DIRECTOR_PROMPT_VERSION,
            semantic_protocol=HOTPOTQA_SEMANTIC_PROTOCOL,
            recovery_policy=PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        )

        messages = transcript_messages(orchestrator.build_prompt(env, 0, ()))
        state = observation_payload(messages[-1])

        self.assertEqual(HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13, messages[0]["content"])
        self.assertEqual(HOTPOTQA_SEMANTIC_PROTOCOL, state["semantic_protocol"])
        self.assertEqual(
            PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
            state["recovery_policy"],
        )
        self.assertEqual(
            "qa-retrieval",
            state["terminal_constraints"]["required_evidence_tool_id"],
        )
        self.assertEqual(
            {
                "semantic_answer_owner_role_family": "reasoner",
                "required_evidence_tool_id": "qa-retrieval",
                "required_evidence_tool_owner_role_family": "reasoner",
                "required_evidence_execution_mode": "react",
                "required_direct_role_edges": [
                    ["reasoner", "verifier"],
                    ["verifier", "format"],
                ],
                "output_role_family": "format",
                "formatter_original_question_visible": False,
                "formatter_answer_reselection_allowed": False,
                "max_agents_per_add_subgraph": 3,
                "output_agent_id_optional_until_lineage_complete": True,
            },
            state["semantic_lineage_constraints"],
        )
        self.assertIn("answer slot actually requested", messages[0]["content"])
        self.assertIn("Reasoner alone determines", messages[0]["content"])
        self.assertIn("must not select, replace", messages[0]["content"])
        self.assertIn("never the original question", messages[0]["content"])
        self.assertIn("unexpectedly equal", messages[0]["content"])
        self.assertIn("preserve -> diagnose -> repair -> augment", messages[0]["content"])
        self.assertIn("both direction flags to false removes", messages[0]["content"])
        self.assertIn("never directly into the Verifier", messages[0]["content"])

        default = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        default_state = observation_payload(
            transcript_messages(default.build_prompt(env, 0, ()))[-1]
        )
        self.assertNotIn("semantic_protocol", default_state)
        self.assertNotIn("semantic_lineage_constraints", default_state)
        self.assertNotIn("recovery_policy", default_state)

    async def test_orchestrator_rejects_prompt_and_version_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            AgentGraphOrchestrator(
                registry(),
                ScriptedDirector([]),
                system_prompt=HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
                prompt_version=DIRECTOR_PROMPT_VERSION,
            )

    async def test_openai_director_boundary_is_local_qwen_supervisor(self) -> None:
        OpenAIDirectorClient(
            base_url="http://127.0.0.1:8015/v1",
            model="supervisor_theta",
        )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="https://provider.example/v1",
                model="supervisor_theta",
            )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="http://127.0.0.1:8015/v1",
                model="gpt-4o-mini",
            )

    async def test_openai_director_submits_exact_transcript_messages(self) -> None:
        messages = (
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": "initial Canvas observation"},
            {"role": "assistant", "content": '{"action":"finish"}'},
            {"role": "user", "content": "current Canvas observation"},
        )
        client = OpenAIDirectorClient(max_retries=0)
        captured = {}

        def fake_post(api_key, payload):
            captured.update(api_key=api_key, payload=payload)
            return {
                "id": "director-request",
                "model": "supervisor_theta",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }

        client._post = fake_post  # type: ignore[method-assign]
        response = await client.propose(encode_director_transcript(messages), seed=9)

        self.assertEqual(list(messages), captured["payload"]["messages"])
        self.assertEqual(9, captured["payload"]["seed"])
        self.assertEqual('{"action":"finish"}', response.text)

    async def test_end_to_end_scripted_canvas_and_execution(self) -> None:
        model_registry = registry()
        add_action = (
            '{"action":"add_agent","agent_id":"solver","model_id":"qwen",'
            '"contract":"solve"}'
        )
        add_response = "preface\n" + add_action + "\ndiscarded suffix"
        client = ScriptedDirector(
            [
                add_response,
                '{"action":"set_output","agent_id":"solver"}',
                '{"action":"finish"}',
            ]
        )
        gateway = FakeGateway()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=gateway,
            execute_on_edit=True,
            max_agents=10,
        )
        result = await AgentGraphOrchestrator(model_registry, client, seed=1).run(
            env, "2+2?"
        )
        self.assertEqual(result.final_answer, "answer from solver")
        self.assertEqual(len(result.turns), 3)
        self.assertNotIn("weighted_preferred_model", client.prompts[0])
        self.assertNotIn("available_skills", client.prompts[0])
        self.assertEqual(result.final_graph["output_agent_id"], "solver")
        self.assertEqual(client.seeds, [1, 2, 3])

        # Each completed Agent configuration executes immediately.  Changing
        # the node to the Output role re-executes that dirty node.
        self.assertIsNotNone(result.turns[0].canvas_result.execution)
        self.assertIsNotNone(result.turns[1].canvas_result.execution)
        self.assertIn("execution_result=", result.turns[0].canvas_result.feedback)
        self.assertIn("execution_result=", result.turns[1].canvas_result.feedback)
        self.assertIn("answer from solver", result.turns[1].canvas_result.feedback)
        self.assertNotIn("output_format", result.turns[1].canvas_result.feedback)
        self.assertNotIn('"final_answer"', result.turns[1].canvas_result.feedback)
        self.assertIn("output_inbox", result.turns[1].canvas_result.feedback)
        # Finish reuses the successful result for the unchanged graph revision.
        self.assertEqual(len(gateway.requests), 2)

        initial_messages = transcript_messages(client.prompts[0])
        continued_messages = transcript_messages(client.prompts[1])
        complete_messages = transcript_messages(client.prompts[2])
        self.assertEqual(
            ["system", "user"], [item["role"] for item in initial_messages]
        )
        self.assertEqual(
            ["system", "user", "assistant", "user"],
            [item["role"] for item in continued_messages],
        )
        self.assertEqual(
            ["system", "user", "assistant", "user", "assistant", "user"],
            [item["role"] for item in complete_messages],
        )
        self.assertEqual(initial_messages, complete_messages[:2])
        first_action = result.turns[0].canvas_result.action
        self.assertIsNotNone(first_action)
        assert first_action is not None
        self.assertEqual(
            add_response[: first_action.consumed_end],
            continued_messages[2]["content"],
        )
        self.assertNotIn("discarded suffix", continued_messages[2]["content"])

        initial_state = observation_payload(initial_messages[-1])
        complete_state = observation_payload(complete_messages[-1])
        self.assertEqual(initial_state["max_agents"], 10)
        self.assertEqual(initial_state["task"], "2+2?")
        self.assertNotIn("directed_edges", initial_state)
        self.assertNotIn("structural_issues", initial_state)
        self.assertNotIn("terminal_format_issue", initial_state)
        qwen_catalog = next(
            item
            for item in initial_state["model_catalog"]
            if item["model_id"] == "qwen"
        )
        self.assertEqual(
            {
                "family": "qwen",
                "profile": "text_qa",
                "text_qa_canary": "passed",
            },
            qwen_catalog["routing_metadata"],
        )
        self.assertEqual("solver", complete_state["current_graph"]["output_agent_id"])
        self.assertNotIn("directed_edges", complete_state)
        self.assertNotIn("structural_issues", complete_state)
        self.assertNotIn("task", complete_state)
        self.assertNotIn("model_catalog", complete_state)
        self.assertIn("execution_result=", complete_state["canvas_feedback"])
        self.assertEqual("empty", initial_state["topology_statistics"]["topology_family"])
        self.assertEqual("single", complete_state["topology_statistics"]["topology_family"])
        for state in (initial_state, complete_state):
            for removed_cue in (
                "max_rounds",
                "remaining_rounds",
                "graph_validation",
                "structurally_complete",
                "recent_canvas_history",
                "construction_progress",
                "output_format",
            ):
                self.assertNotIn(removed_cue, state)
        # The latest progressive result occurs only in the current user
        # observation, rather than in a reconstructed history field.
        self.assertEqual(2, client.prompts[2].count("execution_result="))

    async def test_intermediate_component_defers_terminal_constraints_until_finish(
        self,
    ) -> None:
        model_registry = registry()
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="two-hop question",
            require_exact_answer_tag=True,
            require_format_agent=True,
        )
        component_action = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"left","model_id":"qwen","contract":"left evidence"},'
            '{"agent_id":"right","model_id":"other","contract":"right evidence"},'
            '{"agent_id":"merge","model_id":"qwen","contract":"merge evidence"}'
            '],"relations":['
            '{"source_id":"left","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false},'
            '{"source_id":"right","target_id":"merge",'
            '"source_to_target":true,"target_to_source":false}'
            ']}'
        )

        initial_prompt = orchestrator.build_prompt(env, 0, ())
        initial_state = observation_payload(transcript_messages(initial_prompt)[-1])
        self.assertNotIn("structural_issues", initial_state)
        self.assertNotIn("terminal_format_issue", initial_state)
        self.assertEqual(
            {
                "explicit_finish_required": True,
                "require_exact_answer_tag": True,
                "require_format_agent": True,
                "required_tool_id": None,
            },
            initial_state["terminal_constraints"],
        )

        component = await env.step(component_action)
        self.assertTrue(component.accepted)
        self.assertIsNone(env.graph.output_agent_id)
        component_prompt = orchestrator.continue_prompt(
            initial_prompt,
            component_action,
            env,
            (),
        )
        component_state = observation_payload(
            transcript_messages(component_prompt)[-1]
        )
        self.assertEqual(
            "fan_in",
            component_state["topology_statistics"]["topology_family"],
        )
        self.assertNotIn("structural_issues", component_state)
        self.assertNotIn("terminal_format_issue", component_state)

        finish_action = '{"action":"finish"}'
        premature_finish = await env.step(finish_action)
        self.assertFalse(premature_finish.accepted)
        self.assertIn(
            "output_agent_count",
            {issue.code for issue in premature_finish.validation_issues},
        )
        finish_prompt = orchestrator.continue_prompt(
            component_prompt,
            finish_action,
            env,
            (),
        )
        finish_state = observation_payload(transcript_messages(finish_prompt)[-1])
        self.assertIn("cannot finish", finish_state["canvas_feedback"])
        self.assertIn("output_agent_count", finish_state["canvas_feedback"])
        self.assertNotIn("structural_issues", finish_state)
        self.assertNotIn("terminal_format_issue", finish_state)

    async def test_selected_invalid_output_reports_terminal_format_issue(self) -> None:
        model_registry = registry()
        graph = AgentGraph(
            nodes=(
                AgentNode(
                    "solver",
                    "qwen",
                    "compute the semantic answer",
                    role_family="reasoning",
                ),
            ),
            output_agent_id="solver",
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            problem="question",
            graph=graph,
            require_format_agent=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )

        state = observation_payload(
            transcript_messages(orchestrator.build_prompt(env, 0, ()))[-1]
        )
        self.assertNotIn("structural_issues", state)
        self.assertIn("terminal_format_issue", state)
        self.assertIn("distinct Format Agent", state["terminal_format_issue"])

    async def test_canvas_exposes_only_positive_revision_local_finish_admission(
        self,
    ) -> None:
        model_registry = registry()

        class FormatGateway(FakeGateway):
            async def generate(self, request):
                self.requests.append(request)
                if request.is_format_agent:
                    return AgentResponse("<answer>Paris</answer>")
                return AgentResponse("Candidate answer: Paris\nEvidence: supplied fact")

        env = AgentWorkflowEnv(
            model_registry,
            gateway=FormatGateway(),
            problem="question",
            execute_on_edit=True,
            require_exact_answer_tag=True,
            require_format_agent=True,
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            prompt_version=HOTPOTQA_DIRECTOR_PROMPT_VERSION,
        )
        initial = observation_payload(
            transcript_messages(orchestrator.build_prompt(env, 0, ()))[-1]
        )
        self.assertFalse(initial["finish_admissibility"]["admissible"])
        self.assertEqual("graph_validation", initial["finish_admissibility"]["stage"])
        self.assertEqual(["add_subgraph"], initial["admissible_action_types"])

        action = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"solver","model_id":"qwen",'
            '"contract":"compute one semantic answer"},'
            '{"agent_id":"format","model_id":"other",'
            '"contract":"serialize one candidate","role_family":"format"}'
            '],"relations":['
            '{"source_id":"solver","target_id":"format",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"format"}'
        )
        accepted = await env.step(action)
        self.assertTrue(accepted.accepted)
        prompt = orchestrator.continue_prompt(
            orchestrator.build_prompt(env.fork(), 0, ()),
            action,
            env,
            (),
        )
        state = observation_payload(transcript_messages(prompt)[-1])
        self.assertEqual(
            {
                "admissible": True,
                "graph_revision": env.revision,
                "submission_semantics": "explicit_finish",
            },
            state["finish_admissibility"],
        )
        self.assertNotIn("remaining_rounds", state)
        self.assertIn("finish", state["admissible_action_types"])

    async def test_director_terminal_policy_is_issue_driven_without_role_template(
        self,
    ) -> None:
        self.assertIn("exactly one valid JSON action each turn", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn(
            "action types listed in admissible_action_types",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "add_subgraph adds one functional subgraph of one to three Agents as one transaction",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "A directed relation routes the source artifact to the target",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Each accepted edit is executed once",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Use finish only when finish_admissibility is present and admissible",
            DIRECTOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not assume a fixed workflow topology or an unlisted Skill",
            DIRECTOR_SYSTEM_PROMPT,
        )
        for prohibited in (
            "answer span",
            "answer type",
            "source-grounded evidence",
            "qualifier",
            "Format Agent",
            "fan-in",
            "Researcher",
            "Critic",
            "must use three",
            "singleton",
        ):
            self.assertNotIn(prohibited.casefold(), DIRECTOR_SYSTEM_PROMPT.casefold())

    async def test_history_window_keeps_real_recent_message_pairs(self) -> None:
        model_registry = registry()
        actions = [
            '{"action":"add_agent","agent_id":"source","model_id":"qwen","contract":"produce evidence"}',
            '{"action":"add_agent","agent_id":"sink","model_id":"other","contract":"consume source evidence"}',
            '{"action":"set_relation","source_id":"source","target_id":"sink","source_to_target":true,"target_to_source":false}',
            '{"action":"set_output","agent_id":"sink"}',
        ]
        client = ScriptedDirector(actions)
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())

        await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=4,
            history_window=2,
        ).run(env, "same task")

        messages = transcript_messages(client.prompts[3])
        self.assertEqual(
            ["system", "user", "assistant", "user", "assistant", "user"],
            [item["role"] for item in messages],
        )
        self.assertEqual(actions[1], messages[2]["content"])
        self.assertEqual(actions[2], messages[4]["content"])
        self.assertNotIn(actions[0], [item["content"] for item in messages[2:]])
        self.assertNotIn("recent_canvas_history", client.prompts[3])

    async def test_catalog_order_is_decoupled_from_rollout_sampling_seed(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )
        first = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=101,
            catalog_order_seed="condition:same-task",
        )
        second = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=202,
            catalog_order_seed="condition:same-task",
        )

        first_state = observation_payload(
            transcript_messages(first.build_prompt(env, 0, ()))[-1]
        )
        second_state = observation_payload(
            transcript_messages(second.build_prompt(env, 0, ()))[-1]
        )
        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.seed, second.seed)

    async def test_scientific_rollout_ordinal_changes_sampling_not_catalog(
        self,
    ) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )

        def orchestrator(rollout_ordinal: int) -> AgentGraphOrchestrator:
            coordinate = ScientificSamplingCoordinate(
                sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
                schedule_purpose="architecture-dev",
                ordered_sequence_hash=stable_hash(["hotpotqa:one"]),
                sequence_position=rollout_ordinal,
                task_id="hotpotqa:one",
                optimizer_step_or_anchor_ordinal=0,
            )
            return AgentGraphOrchestrator(
                model_registry,
                ScriptedDirector([]),
                seed=17,
                catalog_order_seed="architecture-dev:hotpotqa:one",
                sampling_base_seed=17,
                sampling_coordinate=coordinate,
            )

        first = orchestrator(0)
        second = orchestrator(1)
        first_state = observation_payload(
            transcript_messages(first.build_prompt(env, 0, ()))[-1]
        )
        second_state = observation_payload(
            transcript_messages(second.build_prompt(env, 0, ()))[-1]
        )

        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.generation_seed(0), second.generation_seed(0))
        self.assertEqual(
            "skillev-scientific-sampling@1",
            first.sampling_receipt["algorithm"],
        )

    async def test_forced_probe_condition_is_separate_from_active_skill_prior(
        self,
    ) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(
            model_registry, gateway=FakeGateway(), problem="same task"
        )
        orchestrator = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
        )
        active_prior = {
            "skill_id": "skill-active",
            "application_mode": "rejectable_prompt_prior",
            "content": "Optional evidence-gated instruction.",
        }
        forced_condition = {
            "condition_id": "candidate-a",
            "application_mode": "forced_probe_condition",
            "content": "Apply the predeclared intervention for this probe arm.",
        }

        messages = transcript_messages(
            orchestrator.build_prompt(
                env,
                0,
                (active_prior, forced_condition),
            )
        )
        state = observation_payload(messages[-1])

        self.assertEqual([active_prior], state["available_skills"])
        self.assertEqual([forced_condition], state["exploration_conditions"])
        self.assertNotIn("validated Skill facts", messages[-1]["content"])

    async def test_round_limit_is_explicit_failure(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                '{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}'
            ]
        )
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())
        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=1,
        ).run(env, "task")

        self.assertIsNone(result.final_answer)
        self.assertFalse(result.explicit_finish)
        self.assertEqual("max_rounds", result.termination_reason)
        self.assertEqual(1, len(result.turns))

    async def test_progressive_action_mask_switches_only_after_finish_admission(
        self,
    ) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                (
                    '{"action":"add_subgraph","agents":['
                    '{"agent_id":"solver","model_id":"qwen",'
                    '"contract":"solve"}],"relations":[],'
                    '"output_agent_id":"solver"}'
                ),
                '{"action":"finish"}',
            ]
        )
        env = AgentWorkflowEnv(
            model_registry,
            gateway=FakeGateway(),
            execute_on_edit=True,
        )

        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=2,
            sampling_action_profile=DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
        ).run(env, "task")

        self.assertTrue(result.explicit_finish)
        self.assertEqual(
            ["add_subgraph", "finish"],
            [item["action_schema_branch"] for item in client.schema_requests],
        )
        self.assertEqual(
            director_state_conditioned_sampling_json_schema_text("add_subgraph"),
            client.schema_requests[0]["action_json_schema"],
        )
        self.assertEqual(
            director_state_conditioned_sampling_json_schema_text("finish"),
            client.schema_requests[1]["action_json_schema"],
        )
        self.assertEqual(
            [DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION] * 2,
            [
                item["action_json_schema_version"]
                for item in client.schema_requests
            ],
        )

    def test_state_conditioned_relation_schema_repeats_full_object_contract(self) -> None:
        schema = json.loads(
            director_state_conditioned_sampling_json_schema_text("add_subgraph")
        )
        relation = schema["properties"]["relations"]["items"]
        expected_required = [
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        ]
        self.assertEqual(2, len(relation["anyOf"]))
        for branch in relation["anyOf"]:
            self.assertEqual(expected_required, branch["required"])
            self.assertEqual(set(expected_required), set(branch["properties"]))
            self.assertFalse(branch["additionalProperties"])
        self.assertEqual(
            {"const": True},
            relation["anyOf"][0]["properties"]["source_to_target"],
        )
        self.assertEqual(
            {"const": True},
            relation["anyOf"][1]["properties"]["target_to_source"],
        )


if __name__ == "__main__":
    unittest.main()
