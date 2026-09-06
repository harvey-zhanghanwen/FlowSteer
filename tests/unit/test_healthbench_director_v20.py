"""Director continuation policy checks; no model, Tool or grader calls."""

import json
import unittest

from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_PROMPT_VERSION,
    DIRECTOR_PROMPT_VERSION_V19,
    DIRECTOR_PROMPT_VERSION_V20,
    DIRECTOR_SYSTEM_PROMPT,
    DIRECTOR_SYSTEM_PROMPT_V19,
    DIRECTOR_SYSTEM_PROMPT_V20,
    director_system_prompt_for_version,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


class DirectorV20Tests(unittest.TestCase):
    def test_version_is_explicit_without_changing_existing_default(self):
        for version, prompt in (
            (DIRECTOR_PROMPT_VERSION, DIRECTOR_SYSTEM_PROMPT),
            (DIRECTOR_PROMPT_VERSION_V19, DIRECTOR_SYSTEM_PROMPT_V19),
            (DIRECTOR_PROMPT_VERSION_V20, DIRECTOR_SYSTEM_PROMPT_V20),
        ):
            self.assertEqual(director_system_prompt_for_version(version), prompt)

    def test_policy_uses_public_evidence_without_prescribing_roles_or_topology(self):
        prompt = DIRECTOR_SYSTEM_PROMPT_V20
        for instruction in (
            "ordered Agent/Tool Action-Observation receipts",
            "current_artifact_receipts, not a short preview",
            "Successful execution or admissible FINISH does not establish answer correctness",
            "State the unresolved question, not a predetermined answer",
            "repeated agreement is not independent evidence",
            "not the Agent's original responsibility",
            "Missing case-specific facts remain unknown",
            "Keep Agent roles, count, models and topology open",
            "one functional subgraph of one to three Agents",
            "execution_mode describes execution, not role",
        ):
            self.assertIn(instruction, prompt)
        for forbidden in ("doctor", "verifier", "formatter", "healthbench", "rubric", "must use three"):
            self.assertNotIn(forbidden, prompt.casefold())
        self.assertLess(len(prompt.split()), 550)
        self.assertNotIn("current_artifact_receipts, not a short preview", DIRECTOR_SYSTEM_PROMPT_V19)

    def test_richer_current_artifacts_are_not_replayed_in_history(self):
        registry = ModelRegistry([ProviderSpec("test", kind="test")], [ModelSpec("a", "test")])
        director = AgentGraphOrchestrator(registry, object(), prompt_version=DIRECTOR_PROMPT_VERSION_V20)
        def observation(body):
            return "Canvas observation.\n\n" + json.dumps(body)
        old = observation({
            "current_artifact_receipts": [{"producer_artifact": "OLD_BODY", "retrieval_evidence": "OLD_SOURCE"}],
            "canvas_feedback": "accepted edit; source node-a; revision 1; tool succeeded",
        })
        latest = observation({
            "current_artifact_receipts": [{"producer_artifact": "LIVE_BODY", "retrieval_evidence": "LIVE_SOURCE"}],
            "task_goal": "Original task and requested relation",
        })
        messages = [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT_V20},
            {"role": "user", "content": "Immutable original conversation and catalog"},
            {"role": "assistant", "content": '{"action":"add_subgraph"}'},
            {"role": "user", "content": old},
            {"role": "assistant", "content": '{"action":"set_output"}'},
            {"role": "user", "content": latest},
        ]
        result = director._compact_historical_messages(messages)
        self.assertEqual(messages[:3], result[:3])
        self.assertEqual(messages[-2:], result[-2:])
        self.assertNotIn("OLD_BODY", result[3]["content"])
        self.assertNotIn("OLD_SOURCE", result[3]["content"])
        self.assertIn("source node-a; revision 1; tool succeeded", result[3]["content"])
        self.assertIn("LIVE_SOURCE", result[-1]["content"])


if __name__ == "__main__":
    unittest.main()
