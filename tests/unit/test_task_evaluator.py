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
        self.assertEqual(1.0, hotpot.metrics["exact_match"])
        self.assertEqual(1.0, hotpot.metrics["token_f1"])
        self.assertGreater(trivia.reward or 0.0, 0.6)
        self.assertEqual(0.0, trivia.metrics["exact_match"])
        self.assertEqual("skillflow.training.reward.v1", trivia.evaluator_version)

    async def test_aime_uses_skillflow_exact_answer_extraction(self) -> None:
        outcome = await evaluate_task(task("AIME 2026", ground_truth="56"), "Thus \\boxed{56}.")
        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(1.0, outcome.metrics["exact_match"])

    async def test_explicit_answer_boundary_is_scored_and_raw_output_is_retained(self) -> None:
        raw = "A long explanation with distractor 999. <answer>red fox</answer>"
        outcome = await evaluate_task(
            task("HotpotQA", ground_truth="the red fox"), raw
        )
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(raw, outcome.details["raw_prediction"])
        self.assertEqual("red fox", outcome.details["scored_prediction"])
        self.assertTrue(outcome.details["structured_answer_extracted"])

        math = await evaluate_task(
            task("AIME 2026", ground_truth="56"),
            "First 999. <answer>56</answer> Trailing 123.",
        )
        self.assertEqual(1.0, math.reward)

    async def test_untagged_static_response_keeps_strict_historical_scoring(self) -> None:
        raw = "The answer is red fox plus unrelated words"
        outcome = await evaluate_task(
            task("TriviaQA", ground_truth="red fox"), raw
        )
        self.assertLess(outcome.reward or 0.0, 1.0)
        self.assertEqual(raw, outcome.details["scored_prediction"])
        self.assertFalse(outcome.details["structured_answer_extracted"])


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
    async def test_webshop_uses_ragen_and_a_stateful_action_prompt(self) -> None:
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
        self.assertIn("Your task is to: secret dataset wrapper text.", prompts[0])
        self.assertIn("already taken 0 step(s)", prompts[0])
        self.assertIn("step 1", prompts[0])
        self.assertIn("Shopping task observation", prompts[0])
        self.assertIn("search[fries seasoning]", prompts[0])

    async def test_webshop_expands_action_dict_and_preserves_recent_history(self) -> None:
        prompts: list[str] = []
        stepped_actions: list[str] = []

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(
                    current_goal_index=7,
                    current_goal_instruction="buy the matching mug",
                )
                self.available_actions = {
                    "has_search_bar": True,
                    "clickables": ["Back to Search"],
                }

            def reset(self, env_type, env_config, question="", extra=None):
                return "WebShop home"

            def step(self, action):
                stepped_actions.append(action)
                if len(stepped_actions) == 1:
                    self.available_actions = {
                        "has_search_bar": False,
                        "clickables": ["B000000001"],
                    }
                    return "Search results", 0.0, False, {}
                return "Bought", 1.0, True, {"graded_score": 1.0}

        async def run_graph(problem):
            prompts.append(problem)
            if len(prompts) == 1:
                return "Reasoning. <action>search[matching mug]</action>"
            return "click[B000000001]"

        record = task(
            "WebShop",
            question="wrapper",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 7},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(
            ["search[matching mug]", "click[B000000001]"], stepped_actions
        )
        self.assertIn("search[<your query>]", prompts[0])
        self.assertIn("click[Back to Search]", prompts[0])
        self.assertIn("Your task is to: buy the matching mug.", prompts[0])
        self.assertIn("Your task is to: buy the matching mug.", prompts[1])
        self.assertIn("already taken 1 step(s)", prompts[1])
        self.assertIn("Action: 'search[matching mug]'", prompts[1])
        self.assertIn("Result: 'Search results'", prompts[1])

    async def test_environment_retries_prose_without_guessing_or_advancing(self) -> None:
        prompts: list[str] = []
        stepped_actions: list[str] = []

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=8)
                self.available_actions = {
                    "has_search_bar": False,
                    "clickables": ["Buy Now"],
                }

            def reset(self, env_type, env_config, question="", extra=None):
                return "Product page"

            def step(self, action):
                stepped_actions.append(action)
                return "Bought", 1.0, True, {}

        async def run_graph(problem):
            prompts.append(problem)
            if len(prompts) == 1:
                return "I think the right choice is click[Buy Now]."
            return "<action>click[Buy Now]</action>"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 8},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(["click[Buy Now]"], stepped_actions)
        self.assertEqual(2, len(prompts))
        self.assertIn("Product page", prompts[0])
        self.assertIn("Product page", prompts[1])
        self.assertIn("Action: '<INVALID>'", prompts[1])
        self.assertIn("[INVALID] No valid <action> tag found.", prompts[1])
        self.assertEqual(2.0, outcome.metrics["steps"])
        failed_attempt = outcome.details["trace"][0]
        self.assertTrue(failed_attempt["parse_error"])
        self.assertFalse(failed_attempt["state_advanced"])
        self.assertEqual(0.0, failed_attempt["reward"])
        self.assertEqual("Product page", failed_attempt["next_observation"])
        self.assertEqual(
            "I think the right choice is click[Buy Now].",
            failed_attempt["raw_graph_output"],
        )

    async def test_invalid_actions_exhaust_budget_as_valid_zero_reward(self) -> None:
        step_called = False

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=9)
                self.available_actions = ["click[Buy Now]"]

            def reset(self, env_type, env_config, question="", extra=None):
                return "Product page"

            def step(self, action):
                nonlocal step_called
                step_called = True
                return "Bought", 1.0, True, {}

        async def run_graph(problem):
            return "click[an action that is not admissible]"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 9},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(
                record,
                "",
                run_graph=run_graph,
                max_environment_steps=2,
            )

        self.assertTrue(outcome.valid)
        self.assertEqual(0.0, outcome.reward)
        self.assertEqual("environment_step_limit", outcome.reason)
        self.assertFalse(step_called)
        self.assertEqual(2.0, outcome.metrics["steps"])
        self.assertEqual(2, len(outcome.details["trace"]))
        self.assertTrue(
            all(entry["parse_error"] for entry in outcome.details["trace"])
        )

    async def test_environment_callback_failure_remains_evaluator_invalid(self) -> None:
        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=10)
                self.available_actions = ["click[Buy Now]"]

            def reset(self, env_type, env_config, question="", extra=None):
                return "Product page"

        async def run_graph(problem):
            raise RuntimeError("executor unavailable")

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 10},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertFalse(outcome.valid)
        self.assertIsNone(outcome.reward)
        self.assertEqual("environment_graph_callback_failed", outcome.reason)

    async def test_webshop_protocol_mismatch_is_invalid_before_callback(self) -> None:
        callback_called = False

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(
                    current_goal_index=12,
                    human_goals=True,
                    use_small=True,
                    num_products=None,
                    goal_split="test",
                    file_path="/data/items.json",
                    attr_path="/data/attrs.json",
                )
                self.available_actions = ["click[Buy Now]"]

            def reset(self, env_type, env_config, question="", extra=None):
                return "Product page"

        async def run_graph(problem):
            nonlocal callback_called
            callback_called = True
            return "click[Buy Now]"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {
                    "goal_index": 12,
                    "human_goals": True,
                    "use_small": False,
                    "goal_split": "test",
                    "file_path": "/data/items.json",
                    "attr_path": "/data/attrs.json",
                },
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertFalse(outcome.valid)
        self.assertEqual("webshop_protocol_mismatch", outcome.reason)
        self.assertEqual(
            {"requested": False, "actual": True},
            outcome.details["protocol_mismatches"]["use_small"],
        )
        self.assertFalse(callback_called)

    async def test_environment_unavailable_after_step_is_invalid(self) -> None:
        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=13)
                self.available_actions = ["click[Buy Now]"]

            def reset(self, env_type, env_config, question="", extra=None):
                return "Product page"

            def step(self, action):
                return "[ENV_UNAVAILABLE] No environment initialized.", 0.0, False, {}

        async def run_graph(problem):
            return "click[Buy Now]"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 13},
            },
        )
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=SimpleNamespace(RAGENAdapter=Adapter),
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertFalse(outcome.valid)
        self.assertIsNone(outcome.reward)
        self.assertEqual("environment_unavailable", outcome.reason)
        self.assertEqual(1, len(outcome.details["trace"]))

    async def test_alfworld_is_locked_to_record_game_file_before_actions(self) -> None:
        target = "/games/game-b.tw-pddl"
        seeds: list[int] = []
        prompts: list[str] = []
        stepped_actions: list[str] = []

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
                stepped_actions.append(action)
                if len(stepped_actions) == 1:
                    self.available_actions = ["inventory"]
                    return "Hallway", 0.0, False, {}
                return "Won", 10.0, True, {"won": True}

        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        record = task(
            "ALFWorld",
            question="Look at a CD by lamp light.",
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
            prompts.append(problem)
            if len(prompts) == 1:
                return "I would probably look."
            return "<action>look</action>" if len(prompts) == 2 else "inventory"

        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        self.assertEqual([1], seeds)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(["look", "inventory"], stepped_actions)
        self.assertEqual(1, outcome.details["locked_game_index"])
        self.assertEqual(target, outcome.details["actual_game_file"])
        self.assertIn("Action format examples:", prompts[0])
        self.assertIn("> go to cabinet 1", prompts[0])
        self.assertIn("> examine shelf 1", prompts[0])
        self.assertIn("Your task is to: Look at a CD by lamp light.", prompts[0])
        self.assertIn("Your task is to: Look at a CD by lamp light.", prompts[1])
        self.assertIn("already taken 1 step(s)", prompts[1])
        self.assertIn("Action: '<INVALID>'", prompts[1])
        self.assertIn("Room", prompts[1])
        self.assertIn("already taken 2 step(s)", prompts[2])
        self.assertIn("Action: 'look'", prompts[2])
        self.assertIn("Result: 'Hallway'", prompts[2])
        self.assertIn("inventory", prompts[2])

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
