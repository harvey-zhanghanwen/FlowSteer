from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import GRADER_TEMPLATE, evaluate_task


def task(
    source: str,
    *,
    question: str = "Question?",
    ground_truth: str = "answer",
    payload: dict | None = None,
    environment: dict | None = None,
) -> TaskRecord:
    metadata = {
        "source": source,
        "dataset_key": source.lower().replace(" ", "_"),
        "evaluator_payload": payload or {},
    }
    if environment is not None:
        metadata["environment"] = environment
    return TaskRecord(
        task_id=f"{source}:1",
        question=question,
        ground_truth=ground_truth,
        split="train",
        metadata=metadata,
    )


class StaticEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_hotpot_and_trivia_use_skillflow_token_f1(self) -> None:
        hotpot = await evaluate_task(task("HotpotQA", ground_truth="the red fox"), "red fox")
        trivia = await evaluate_task(
            task(
                "TriviaQA",
                ground_truth="wrong | Alexander Fleming",
                payload={"accepted_answers": ["wrong", "Alexander Fleming"]},
            ),
            "Fleming",
        )
        self.assertTrue(hotpot.valid)
        self.assertEqual(1.0, hotpot.reward)
        self.assertGreater(trivia.reward or 0.0, 0.6)
        self.assertEqual("skillflow.training.reward.v1", trivia.evaluator_version)

    async def test_aime_uses_skillflow_exact_answer_extraction(self) -> None:
        outcome = await evaluate_task(task("AIME 2026", ground_truth="56"), "Thus \\boxed{56}.")
        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(1.0, outcome.metrics["exact_match"])


class HealthBenchEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_signed_weighting_records_raw_and_clips_grpo_reward(self) -> None:
        record = task(
            "HealthBench Professional",
            question="Conversation:\n\n[user] Is this safe?\n\n[assistant]",
            payload={
                "rubric_items": [
                    {"criterion_text": "gives unsafe advice", "points": -4},
                    {"criterion_text": "recommends care", "points": 8},
                ]
            },
        )
        calls: list[tuple[list[dict[str, str]], str]] = []

        async def judge(messages, model):
            calls.append((messages, model))
            if "[-4]" in messages[0]["content"]:
                return '```json\n{"explanation":"unsafe", "criteria_met":true}\n```'
            return {"explanation": "missing", "criteria_met": False}

        outcome = await evaluate_task(record, "Do it.", judge=judge, judge_model="grader")
        self.assertTrue(outcome.valid)
        self.assertEqual(-0.5, outcome.metrics["raw_score"])
        self.assertEqual(0.0, outcome.reward)
        self.assertEqual(2, len(calls))
        self.assertTrue(all(model == "grader" for _, model in calls))
        prompt = calls[0][0][0]["content"]
        self.assertIn("user: Is this safe?", prompt)
        self.assertIn("assistant: Do it.", prompt)
        self.assertIn("Return just the json object", prompt)
        self.assertIn("<<conversation>>", GRADER_TEMPLATE)

    async def test_invalid_judge_output_cannot_form_grpo_reward(self) -> None:
        record = task(
            "HealthBench Professional",
            payload={
                "rubric_items": [{"criterion_text": "is correct", "points": 5}]
            },
        )

        async def judge(messages, model):
            return '{"criteria_met":"yes"}'

        outcome = await evaluate_task(record, "answer", judge=judge)
        self.assertFalse(outcome.valid)
        self.assertIsNone(outcome.reward)
        self.assertEqual("healthbench_judge_response_invalid", outcome.reason)


class SWEbenchEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_gold_patch_similarity_is_never_a_reward(self) -> None:
        record = task("SWE-bench", ground_truth="diff --git a/x b/x\n+fix")
        outcome = await evaluate_task(record, record.ground_truth)
        self.assertFalse(outcome.valid)
        self.assertIsNone(outcome.reward)
        self.assertEqual("swebench_harness_unavailable", outcome.reason)
        self.assertFalse(outcome.details["proxy_similarity_used"])

    async def test_real_harness_callback_may_report_resolved(self) -> None:
        async def harness(record, prediction):
            return {"resolved": True, "instance_id": "repo__1"}

        outcome = await evaluate_task(task("SWE-bench"), "patch", swe_harness=harness)
        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual("repo__1", outcome.details["instance_id"])


class EnvironmentEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_webshop_uses_ragen_and_a_short_neutral_action_prompt(self) -> None:
        prompts: list[str] = []

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=128)
                self.available_actions = ["search[fries seasoning]"]

            def reset(self, env_type, env_config, question="", extra=None):
                self.reset_args = (env_type, env_config, question, extra)
                return "Shopping task observation"

            def step(self, action):
                self.available_actions = []
                return "Bought", 0.75, True, {"graded_score": 0.75}

        async def run_graph(problem):
            prompts.append(problem)
            return "search[fries seasoning]"

        record = task(
            "WebShop",
            question="secret dataset wrapper text",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 128},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        self.assertEqual(0.75, outcome.reward)
        self.assertEqual(1, len(prompts))
        self.assertEqual(
            "Observation:\nShopping task observation\n\n"
            "Legal actions:\nsearch[fries seasoning]\n\n"
            "Return exactly one legal action.",
            prompts[0],
        )
        self.assertNotIn("secret dataset wrapper text", prompts[0])

    async def test_alfworld_is_locked_to_record_game_file_before_actions(self) -> None:
        target = "/games/game-b.tw-pddl"
        seeds: list[int] = []

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [
                    "/games/game-a.tw-pddl",
                    target,
                    "/games/game-c.tw-pddl",
                ]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                seeds.append(env_config["seed"])
                self._env = SimpleNamespace(current_game_file=target)
                return "Room"

            def step(self, action):
                return "Won", 10.0, True, {"won": True}

        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        record = task(
            "ALFWorld",
            environment={
                "env_type": "alfworld",
                "env_config": {
                    "config_file": "/config.yaml",
                    "mode": "train",
                    "seed": 999,
                    "game_file": target,
                },
            },
        )

        async def run_graph(problem):
            return "look"

        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        self.assertEqual([1], seeds)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(1, outcome.details["locked_game_index"])
        self.assertEqual(target, outcome.details["actual_game_file"])

    async def test_alfworld_retry_to_another_game_is_invalid(self) -> None:
        target = "/games/game-b.tw-pddl"
        callback_called = False

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [target, "/games/game-c.tw-pddl"]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                self._env = SimpleNamespace(current_game_file="/games/game-c.tw-pddl")
                return "Wrong room"

        async def run_graph(problem):
            nonlocal callback_called
            callback_called = True
            return "look"

        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        record = task(
            "ALFWorld",
            environment={
                "env_type": "alfworld",
                "env_config": {"game_file": target},
            },
        )
        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(record, "", run_graph=run_graph)
        self.assertFalse(outcome.valid)
        self.assertEqual("alfworld_task_lock_mismatch", outcome.reason)
        self.assertFalse(callback_called)


if __name__ == "__main__":
    unittest.main()
