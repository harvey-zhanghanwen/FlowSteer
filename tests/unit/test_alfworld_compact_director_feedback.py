from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2,
    _director_compact_environment_state_projection,
    _director_compact_execution_feedback_projection,
    decode_director_transcript,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("local", endpoint="http://local/v1")],
        [ModelSpec("qwen", "local")],
    )


class NoModelGateway:
    async def generate(self, request):
        raise AssertionError("Director projection tests must never call a model")


def execution_feedback(body: str) -> str:
    return "accepted continue at revision 2; execution_result=" + json.dumps({
        "executed_agent_ids": ["source", "owner"],
        "output_inbox": [{
            "source_agent_id": "source", "target_agent_id": "owner",
            "artifact_id": "source-v2", "raw_output": body,
        }],
        "agent_artifacts": [
            {"agent_id": "source", "artifact_id": "source-v2", "artifact_body": body},
            {
                "agent_id": "owner", "artifact_id": "owner-v2",
                "artifact_body": '{"action":"open drawer 1"}',
                "input_artifact_provenance": [{
                    "source_agent_id": "source", "target_agent_id": "owner",
                    "artifact_id": "source-v2", "artifact_body": body,
                }],
            },
        ],
    }, separators=(",", ":"))


def state(revision: int = 2) -> dict:
    observation = "The open drawer contains an apple 1. " + "visible detail " * 90
    actions = ["take apple 1 from drawer 1", "close drawer 1", "go to shelf 1"]
    progress = {"held_target_instances": [], "placed_target_instances": []}
    latest = {
        "action": "open drawer 1", "raw_action": '{"action":"open drawer 1"}',
        "turn": revision, "observation": "The drawer is closed.",
        "observation_result": observation, "next_observation": observation,
        "observation_result_clipped": False, "observation_status": "success",
        "state_advanced": True, "environment_terminal": False,
        "environment_revision_before": revision - 1,
        "environment_revision_after": revision,
        "task_instruction": "Put one apple on the shelf.",
        "admissible_actions_before": ["open drawer 1", "go to shelf 1"],
        "next_admissible_actions": actions,
        "next_policy_action_domain": actions,
        "goal_progress_before": progress, "goal_progress_after": progress,
        "goal_progress": progress, "goal_progress_changed": False,
        "remaining_action_budget": 20 - revision,
        "remaining_action_budget_after": 20 - revision, "total_action_budget": 20,
    }
    return {
        "task_family": "alfworld", "environment_revision": revision,
        "current_observation": observation, "admissible_actions": actions,
        "policy_action_domain": actions, "goal_progress": progress,
        "remaining_action_budget": 20 - revision, "total_action_budget": 20,
        "environment_terminal": False,
        "latest_action_observation": latest,
        "action_observation_history": [dict(latest)],
    }


def payload(message: dict) -> dict:
    return json.loads(message["content"].partition("\n\n")[2])


class CompactALFWorldDirectorFeedbackTests(unittest.TestCase):
    def test_latest_artifact_is_complete_once_and_raw_receipt_unchanged(self):
        body = "START " + "unabridged collaboration evidence " * 180 + " END"
        original = execution_feedback(body)
        projected = _director_compact_execution_feedback_projection(original)
        result = json.loads(projected.split("execution_result=", 1)[1])
        self.assertEqual(body, result["agent_artifacts"][0]["artifact_body"])
        self.assertEqual(1, projected.count(body))
        self.assertEqual(3, original.count(body))
        self.assertEqual(
            {"artifact_id": "source-v2"}, result["output_inbox"][0]["raw_output_ref"]
        )
        self.assertEqual(
            {"artifact_id": "source-v2"},
            result["agent_artifacts"][1]["input_artifact_provenance"][0]["artifact_body_ref"],
        )
        # An unmatched version is an independent observation, not a duplicate.
        mismatched = original.replace('"artifact_id":"source-v2","raw_output"',
                                      '"artifact_id":"source-v1","raw_output"', 1)
        result = json.loads(_director_compact_execution_feedback_projection(
            mismatched
        ).split("execution_result=", 1)[1])
        self.assertEqual(body, result["output_inbox"][0]["raw_output"])

    def test_latest_native_transition_and_complete_action_domain_remain_available(self):
        original = state()
        before = json.dumps(original, sort_keys=True)
        compact = _director_compact_environment_state_projection(original)
        latest = compact["latest_action_observation"]
        self.assertEqual(original["latest_action_observation"]["observation_result"],
                         compact["current_observation"])
        self.assertEqual("current_observation", latest["observation_result_ref"])
        self.assertEqual(compact["current_observation"],
                         compact[latest["next_observation_ref"]])
        self.assertFalse(latest["observation_result_clipped"])
        self.assertEqual(original["admissible_actions"], compact["admissible_actions"])
        self.assertEqual("admissible_actions", latest["next_admissible_actions_ref"])
        self.assertEqual("goal_progress", latest["goal_progress_after_ref"])
        self.assertEqual(original["goal_progress"], compact["goal_progress"])
        self.assertEqual("Put one apple on the shelf.", latest["task_instruction"])
        self.assertEqual("open drawer 1", latest["action"])
        self.assertEqual(1, latest["environment_revision_before"])
        self.assertEqual(2, latest["environment_revision_after"])
        self.assertEqual(18, latest["remaining_action_budget_after"])
        self.assertEqual(20, latest["total_action_budget"])
        self.assertFalse(latest["environment_terminal"])
        self.assertEqual(before, json.dumps(original, sort_keys=True))
        self.assertLess(len(json.dumps(compact)), len(before))

    def test_disabled_option_and_non_alfworld_keep_existing_observation(self):
        reg = registry()
        env = AgentWorkflowEnv(reg, gateway=NoModelGateway())
        env.reset("Put one apple on the shelf.")
        original = state()
        baseline = AgentGraphOrchestrator(reg, object())
        enabled = AgentGraphOrchestrator(reg, object(), compact_execution_feedback=True)
        disabled = AgentGraphOrchestrator(reg, object(), compact_execution_feedback=False)
        with patch.object(env, "public_environment_state", return_value=original):
            expected = baseline._canvas_observation(env, include_task_context=False, skills=())
            actual = disabled._canvas_observation(env, include_task_context=False, skills=())
            compact = enabled._canvas_observation(env, include_task_context=False, skills=())
        self.assertEqual(expected, actual)
        self.assertNotEqual(expected["environment_state"], compact["environment_state"])
        original["task_family"] = "webshop"
        with patch.object(env, "public_environment_state", return_value=original):
            self.assertEqual(
                baseline._canvas_observation(env, include_task_context=False, skills=()),
                enabled._canvas_observation(env, include_task_context=False, skills=()),
            )

    def test_history_compacts_only_old_artifacts_and_preserves_error_receipts(self):
        orchestrator = AgentGraphOrchestrator(
            registry(), object(), compact_execution_feedback=True,
            prompt_version=STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2,
        )
        body = "START " + "artifact evidence " * 300 + " END"
        feedback = _director_compact_execution_feedback_projection(execution_feedback(body))
        old = {"canvas_feedback": feedback, "environment_state": state(2)}
        current = {"canvas_feedback": feedback, "environment_state": state(3)}
        error = {"canvas_feedback": 'execution_error={"error_type":"InvalidAction"}',
                 "failure_receipt": {"agent_id": "owner", "action": "bad action"}}
        messages = [
            {"role": "system", "content": orchestrator.system_prompt},
            {"role": "user", "content": "immutable original task and catalog"},
            {"role": "assistant", "content": '{"action":"add_subgraph"}'},
            {"role": "user", "content": orchestrator._observation_message(old)},
            {"role": "assistant", "content": '{"action":"continue"}'},
            {"role": "user", "content": orchestrator._observation_message(error)},
            {"role": "assistant", "content": '{"action":"continue"}'},
            {"role": "user", "content": orchestrator._observation_message(current)},
        ]
        actual = orchestrator._compact_historical_messages(messages)
        self.assertEqual(messages[:3], actual[:3])
        self.assertEqual(messages[-1], actual[-1])
        self.assertNotIn("environment_state", payload(actual[3]))
        self.assertNotIn(body, actual[3]["content"])
        self.assertIn("START", actual[3]["content"])
        self.assertIn(" END", actual[3]["content"])
        historical_result = json.loads(
            payload(actual[3])["canvas_feedback"].split("execution_result=", 1)[1]
        )
        historical_artifacts = {
            item["artifact_id"]: item for item in historical_result["agent_artifacts"]
        }
        inbox = historical_result["output_inbox"][0]
        self.assertNotIn("raw_output_ref", inbox)
        preview_source = inbox["raw_output_preview_ref"]["artifact_id"]
        self.assertIn("artifact_preview", historical_artifacts[preview_source])
        provenance = historical_result["agent_artifacts"][1]["input_artifact_provenance"][0]
        self.assertNotIn("artifact_body_ref", provenance)
        preview_source = provenance["artifact_body_preview_ref"]["artifact_id"]
        self.assertIn("artifact_preview", historical_artifacts[preview_source])
        self.assertEqual(error, payload(actual[5]))
        self.assertEqual([m for m in messages if m["role"] == "assistant"],
                         [m for m in actual if m["role"] == "assistant"])

    def test_window_retains_original_observation_action_pair_and_latest_exact(self):
        reg = registry()
        env = AgentWorkflowEnv(reg, gateway=NoModelGateway())
        env.reset("Put one apple on the shelf.")
        orchestrator = AgentGraphOrchestrator(
            reg, object(), history_window=2, compact_execution_feedback=True,
            prompt_version=STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2,
        )
        with patch.object(env, "public_environment_state", return_value=state(0)):
            prompt = orchestrator.build_prompt(env, 0, ())
        initial = decode_director_transcript(prompt)
        for revision in range(1, 5):
            env._last_feedback = f"accepted continue; environment revision {revision}"
            with patch.object(env, "public_environment_state", return_value=state(revision)):
                prompt = orchestrator.continue_prompt(
                    prompt, json.dumps({"action": "continue", "turn": revision}), env, ()
                )
        messages = decode_director_transcript(prompt)
        self.assertEqual(["system", "user", "assistant", "user", "assistant", "user"],
                         [m["role"] for m in messages])
        self.assertEqual(initial[1], messages[1])
        self.assertEqual(1, json.loads(messages[2]["content"])["turn"])
        self.assertEqual(4, json.loads(messages[4]["content"])["turn"])
        self.assertEqual(4, payload(messages[-1])["environment_state"]["environment_revision"])
        self.assertEqual("accepted continue; environment revision 3",
                         payload(messages[3])["canvas_feedback"])


if __name__ == "__main__":
    unittest.main()
