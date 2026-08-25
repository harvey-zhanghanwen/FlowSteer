from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import (
    GRADER_TEMPLATE,
    HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    TRIVIAQA_ANSWER_EVALUATOR_VERSION,
    _webshop_instruction_matches,
    _load_ragen_module,
    evaluate_task,
)


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
    async def test_hotpot_and_trivia_use_their_official_answer_metrics(self) -> None:
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
        self.assertEqual(HOTPOTQA_ANSWER_EVALUATOR_VERSION, hotpot.evaluator_version)
        self.assertEqual("answer_only", hotpot.details["metric_scope"])
        self.assertGreater(trivia.reward or 0.0, 0.6)
        self.assertEqual(0.0, trivia.metrics["exact_match"])
        self.assertEqual(TRIVIAQA_ANSWER_EVALUATOR_VERSION, trivia.evaluator_version)
        self.assertEqual("answer_only", trivia.details["metric_scope"])
        self.assertEqual(
            "accepted_answer_canonicalization_mismatch",
            trivia.details["answer_mismatch_type"],
        )
        self.assertEqual(
            "evaluator_only",
            trivia.details["answer_mismatch_diagnostic_scope"],
        )

    async def test_trivia_matches_skillflow_formal_protocol_normalization(self) -> None:
        punctuation = await evaluate_task(
            task("TriviaQA", ground_truth="The F.E.A.R."),
            "fear",
        )
        hyphen = await evaluate_task(
            task("TriviaQA", ground_truth="Jean-Luc Picard"),
            "Jean Luc Picard",
        )
        alias = await evaluate_task(
            task(
                "TriviaQA",
                ground_truth="wrong",
                payload={"accepted_answers": ["Harry Sinclair Lewis", "Sinclair Lewis"]},
            ),
            "Sinclair Lewis",
        )

        self.assertEqual(1.0, punctuation.metrics["exact_match"])
        self.assertEqual(1.0, punctuation.metrics["token_f1"])
        # SkillFlow removes punctuation; it does not turn a hyphen into a
        # whitespace boundary before removal.
        self.assertEqual(0.0, hyphen.metrics["exact_match"])
        self.assertAlmostEqual(0.4, hyphen.metrics["token_f1"])
        self.assertEqual(1.0, alias.metrics["exact_match"])
        self.assertEqual(1.0, alias.metrics["token_f1"])

    async def test_trivia_decade_alias_is_diagnostic_only(self) -> None:
        record = task(
            "TriviaQA",
            question="In which decade did Billboard publish its first chart?",
            ground_truth="30s | 30-39",
            payload={"accepted_answers": ["30s", "30-39"]},
        )
        canonicalization = await evaluate_task(record, "<answer>1930s</answer>")
        wrong_decade = await evaluate_task(record, "<answer>1950s</answer>")
        wrong_scope = await evaluate_task(
            task(
                "TriviaQA",
                question="When did Billboard publish its first chart?",
                ground_truth="30s",
                payload={"accepted_answers": ["30s"]},
            ),
            "<answer>1930s</answer>",
        )

        self.assertEqual(0.0, canonicalization.metrics["exact_match"])
        self.assertEqual(0.0, canonicalization.metrics["token_f1"])
        self.assertEqual(
            "accepted_answer_canonicalization_mismatch",
            canonicalization.details["answer_mismatch_type"],
        )
        self.assertEqual(
            "no_accepted_answer_overlap",
            wrong_decade.details["answer_mismatch_type"],
        )
        self.assertEqual(
            "no_accepted_answer_overlap",
            wrong_scope.details["answer_mismatch_type"],
        )

    async def test_hotpot_yes_no_and_hyphen_rules_match_official_scorer(self) -> None:
        yes_with_explanation = await evaluate_task(
            task("HotpotQA", ground_truth="yes"),
            "yes, because the passage says so",
        )
        no_with_explanation = await evaluate_task(
            task("HotpotQA", ground_truth="no"),
            "no, that is incorrect",
        )
        date = await evaluate_task(
            task("HotpotQA", ground_truth="February 5, 1953"),
            "1953-02-05",
        )

        self.assertEqual(0.0, yes_with_explanation.metrics["token_f1"])
        self.assertEqual(0.0, no_with_explanation.metrics["token_f1"])
        self.assertEqual(0.0, date.metrics["token_f1"])

    async def test_hotpot_official_normalization_and_counter_overlap(self) -> None:
        normalized = await evaluate_task(
            task("HotpotQA", ground_truth="The F.E.A.R."),
            "fear",
        )
        repeated = await evaluate_task(
            task("HotpotQA", ground_truth="red red fox"),
            "red fox fox",
        )

        self.assertEqual(1.0, normalized.metrics["exact_match"])
        self.assertAlmostEqual(2.0 / 3.0, repeated.metrics["token_f1"])

    async def test_aime_uses_skillev_private_integer_submission(self) -> None:
        outcome = await evaluate_task(
            task("AIME 2026", ground_truth="56"), "<answer>056</answer>"
        )
        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(1.0, outcome.metrics["accuracy"])
        self.assertNotIn("exact_match", outcome.metrics)
        self.assertTrue(outcome.details["parsing_succeeded"])
        self.assertEqual("56", outcome.details["canonical_prediction"])

        free_form = await evaluate_task(
            task("AIME 2026", ground_truth="56"), "Thus \\boxed{56}."
        )
        self.assertEqual(1.0, free_form.reward)
        self.assertTrue(free_form.details["parsing_succeeded"])
        self.assertEqual("56", free_form.details["canonical_prediction"])

    async def test_aime_direct_and_agentgraph_share_one_evaluator_contract(self) -> None:
        benchmark_task = task("AIME 2026", ground_truth="56")
        direct = await evaluate_task(benchmark_task, "<answer>056</answer>")
        agentgraph = await evaluate_task(benchmark_task, "<answer>056</answer>")

        self.assertEqual(direct.evaluator_version, agentgraph.evaluator_version)
        self.assertEqual(direct.metrics, agentgraph.metrics)
        self.assertEqual(
            direct.details["canonical_prediction"],
            agentgraph.details["canonical_prediction"],
        )
        self.assertEqual("56", direct.details["canonical_prediction"])

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

    async def test_harness_infrastructure_diagnostic_is_preserved_structurally(
        self,
    ) -> None:
        diagnostic = SimpleNamespace(
            classification="docker_daemon_unavailable",
            phase="docker",
            retryable=True,
            test_output_present=False,
            report_present=False,
            container_exit_code=137,
            container_oom_killed=True,
            log_relative_path="must-not-be-copied.log",
        )

        class HarnessInfrastructureError(RuntimeError):
            def __init__(self) -> None:
                self.diagnostic = diagnostic
                super().__init__("unstructured diagnostic prose")

        async def harness(record, prediction):
            raise HarnessInfrastructureError()

        outcome = await evaluate_task(task("SWE-bench"), "patch", swe_harness=harness)

        self.assertFalse(outcome.valid)
        self.assertIsNone(outcome.reward)
        self.assertEqual("swebench_harness_failed", outcome.reason)
        self.assertEqual(
            {
                "error_type": "HarnessInfrastructureError",
                "error": "unstructured diagnostic prose",
                "classification": "docker_daemon_unavailable",
                "phase": "docker",
                "retryable": True,
                "test_output_present": False,
                "report_present": False,
                "container_exit_code": 137,
                "oom_killed": True,
            },
            outcome.details,
        )
        self.assertNotIn("log_relative_path", outcome.details)


class EnvironmentEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    def test_deployed_ragen_module_is_reused_within_the_process(self) -> None:
        module_name = "_flowsteer_deployed_ragen_adapter"
        previous = sys.modules.pop(module_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "ragen_adapter.py"
                source.write_text("VALUE = object()\n", encoding="utf-8")
                first = _load_ragen_module(source)
                second = _load_ragen_module(source)
                self.assertIs(first, second)
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

    def test_webshop_goal_match_accepts_only_upstream_price_suffix(self) -> None:
        goal = "i need a natural looking hair extension"
        self.assertTrue(_webshop_instruction_matches(goal, goal))
        self.assertTrue(
            _webshop_instruction_matches(
                goal,
                goal + ", and price lower than 40.00 dollars",
            )
        )
        self.assertFalse(
            _webshop_instruction_matches(
                goal,
                goal + ", and color must be black",
            )
        )

    async def test_webshop_seeds_upstream_goal_generation_before_reset(self) -> None:
        lifecycle: list[object] = []

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=3)
                self.available_actions = ["click[Buy Now]"]

            def reset(self, env_type, env_config, question="", extra=None):
                lifecycle.append("reset")
                return "Product page"

            def step(self, action):
                return "Bought", 1.0, True, {"graded_score": 1.0}

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 3, "env_seed": 31415},
            },
        )

        async def run_graph(problem):
            return "click[Buy Now]"

        with (
            patch(
                "src.interactive.task_evaluator._load_ragen_module",
                return_value=SimpleNamespace(
                    RAGENAdapter=Adapter,
                    _check_webshop=lambda: lifecycle.append("check_webshop") or True,
                ),
            ),
            patch(
                "src.interactive.task_evaluator.random.seed",
                side_effect=lambda value: lifecycle.append(("seed", value)),
            ) as seed,
        ):
            outcome = await evaluate_task(record, "", run_graph=run_graph)

        self.assertTrue(outcome.valid)
        seed.assert_called_once_with(31415)
        self.assertEqual(["check_webshop", ("seed", 31415), "reset"], lifecycle)

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

    async def test_environment_replay_restores_prefix_and_calls_graph_only_for_suffix(
        self,
    ) -> None:
        prompts: list[str] = []
        stepped_actions: list[str] = []

        class Adapter:
            def __init__(self):
                self._env = SimpleNamespace(current_goal_index=21)
                self.available_actions = ["search[coffee]"]

            def reset(self, env_type, env_config, question="", extra=None):
                return "Home"

            def step(self, action):
                stepped_actions.append(action)
                if action == "search[coffee]":
                    self.available_actions = ["click[P1]"]
                    return "Results", 0.0, False, {"page": "results"}
                if action == "click[P1]":
                    self.available_actions = ["click[Buy Now]"]
                    return "Product", 0.0, False, {"page": "product"}
                return "Bought", 1.0, True, {"graded_score": 1.0}

        replay_trace = [
            {
                "step": 0,
                "observation": "Home",
                "legal_actions": ["search[coffee]"],
                "action": "search[coffee]",
                "raw_graph_output": "<action>search[coffee]</action>",
                "next_observation": "Results",
                "reward": 0.0,
                "done": False,
                "info": {"page": "results"},
            },
            {
                "step": 1,
                "observation": "Results",
                "legal_actions": ["click[P1]"],
                "action": "click[P1]",
                "raw_graph_output": "click[P1]",
                "next_observation": "Product",
                "reward": 0.0,
                "done": False,
                "info": {"page": "product"},
            },
        ]

        async def run_graph(problem):
            prompts.append(problem)
            return "click[Buy Now]"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 21},
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
                environment_replay_trace=replay_trace,
            )

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(
            ["search[coffee]", "click[P1]", "click[Buy Now]"],
            stepped_actions,
        )
        self.assertEqual(1, len(prompts))
        self.assertIn("already taken 2 step(s)", prompts[0])
        self.assertIn("Action: 'search[coffee]'", prompts[0])
        self.assertIn("Action: 'click[P1]'", prompts[0])
        self.assertEqual(2, outcome.details["replayed_environment_steps"])
        self.assertEqual(3.0, outcome.metrics["steps"])

    async def test_environment_replay_rejects_non_sequence_before_runtime(
        self,
    ) -> None:
        callback_called = False

        async def run_graph(problem):
            nonlocal callback_called
            callback_called = True
            return "click[Buy Now]"

        record = task(
            "WebShop",
            environment={
                "env_type": "webshop",
                "env_config": {"goal_index": 21},
            },
        )
        for invalid_trace in (None, "not-a-trace", b"not-a-trace", {"step": 0}):
            with self.subTest(invalid_trace=invalid_trace):
                with patch(
                    "src.interactive.task_evaluator._load_ragen_module",
                    side_effect=AssertionError("runtime must not be loaded"),
                ) as load:
                    outcome = await evaluate_task(
                        record,
                        "",
                        run_graph=run_graph,
                        environment_replay_trace=invalid_trace,  # type: ignore[arg-type]
                    )

                self.assertFalse(outcome.valid)
                self.assertIsNone(outcome.reward)
                self.assertEqual("environment_replay_trace_invalid", outcome.reason)
                load.assert_not_called()
        self.assertFalse(callback_called)

    async def test_environment_replay_mismatches_fail_closed_before_graph_call(
        self,
    ) -> None:
        cases = {
            "observation": (
                {"observation": "Different home"},
                "environment_replay_state_mismatch",
            ),
            "legal_actions": (
                {"legal_actions": ["click[P2]"]},
                "environment_replay_actions_mismatch",
            ),
            "next_observation": (
                {"next_observation": "Different results"},
                "environment_replay_transition_mismatch",
            ),
            "reward": (
                {"reward": 0.25},
                "environment_replay_transition_mismatch",
            ),
            "done": (
                {"done": True},
                "environment_replay_transition_mismatch",
            ),
            "info": (
                {"info": {"page": "different"}},
                "environment_replay_transition_mismatch",
            ),
            "raw_graph_output": (
                {"raw_graph_output": "click[P2]"},
                "environment_replay_action_invalid",
            ),
        }

        for name, (change, expected_reason) in cases.items():
            with self.subTest(name=name):
                callback_called = False

                class Adapter:
                    def __init__(self):
                        self._env = SimpleNamespace(current_goal_index=22)
                        self.available_actions = ["click[P1]"]

                    def reset(self, env_type, env_config, question="", extra=None):
                        return "Home"

                    def step(self, action):
                        self.available_actions = ["click[Buy Now]"]
                        return "Results", 0.0, False, {"page": "results"}

                async def run_graph(problem):
                    nonlocal callback_called
                    callback_called = True
                    return "click[Buy Now]"

                entry = {
                    "step": 0,
                    "observation": "Home",
                    "legal_actions": ["click[P1]"],
                    "action": "click[P1]",
                    "raw_graph_output": "<action>click[P1]</action>",
                    "next_observation": "Results",
                    "reward": 0.0,
                    "done": False,
                    "info": {"page": "results"},
                }
                entry.update(change)
                record = task(
                    "WebShop",
                    environment={
                        "env_type": "webshop",
                        "env_config": {"goal_index": 22},
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
                        environment_replay_trace=[entry],
                    )

                self.assertFalse(outcome.valid)
                self.assertIsNone(outcome.reward)
                self.assertEqual(expected_reason, outcome.reason)
                self.assertFalse(callback_called)

    async def test_environment_replay_rejects_infrastructure_failure_transition(
        self,
    ) -> None:
        cases = (
            (
                "[ENV_UNAVAILABLE] No environment initialized.",
                {},
            ),
            (
                "[ERROR] Environment step failed",
                {"error": "environment failed"},
            ),
        )
        for next_observation, info in cases:
            with self.subTest(next_observation=next_observation):
                callback_called = False

                class Adapter:
                    def __init__(self):
                        self._env = SimpleNamespace(current_goal_index=23)
                        self.available_actions = ["click[P1]"]

                    def reset(self, env_type, env_config, question="", extra=None):
                        return "Home"

                    def step(self, action):
                        return next_observation, 0.0, False, info

                async def run_graph(problem):
                    nonlocal callback_called
                    callback_called = True
                    return "click[P1]"

                replay_trace = [
                    {
                        "step": 0,
                        "observation": "Home",
                        "legal_actions": ["click[P1]"],
                        "action": "click[P1]",
                        "raw_graph_output": "click[P1]",
                        "next_observation": next_observation,
                        "reward": 0.0,
                        "done": False,
                        "info": info,
                    }
                ]
                record = task(
                    "WebShop",
                    environment={
                        "env_type": "webshop",
                        "env_config": {"goal_index": 23},
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
                        environment_replay_trace=replay_trace,
                    )

                self.assertFalse(outcome.valid)
                self.assertEqual(
                    "environment_replay_transition_invalid", outcome.reason
                )
                self.assertFalse(callback_called)

    async def test_alfworld_replayed_legal_terminal_zero_remains_valid(self) -> None:
        target = "/games/game-a.tw-pddl"
        callback_called = False

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [target]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                self._env = SimpleNamespace(current_game_file=target)
                return "Room"

            def step(self, action):
                return "Finished", 0.0, True, {"won": False}

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
            question="look",
            environment={
                "env_type": "alfworld",
                "env_config": {"game_file": target},
            },
        )
        replay_trace = [
            {
                "step": 0,
                "observation": "Room",
                "legal_actions": ["look"],
                "action": "look",
                "raw_graph_output": "look",
                "next_observation": "Finished",
                "reward": 0.0,
                "done": True,
                "info": {"won": False},
            }
        ]
        with patch(
            "src.interactive.task_evaluator._load_ragen_module",
            return_value=module,
        ):
            outcome = await evaluate_task(
                record,
                "",
                run_graph=run_graph,
                environment_replay_trace=replay_trace,
            )

        self.assertTrue(outcome.valid)
        self.assertEqual(0.0, outcome.reward)
        self.assertEqual(0.0, outcome.metrics["success"])
        self.assertEqual("evaluated", outcome.reason)
        self.assertEqual(1, outcome.details["replayed_environment_steps"])
        self.assertFalse(callback_called)

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
                    env_seed=1000,
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
                    "env_seed": 2000,
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
        self.assertEqual(
            {"requested": 2000, "actual": 1000},
            outcome.details["protocol_mismatches"]["env_seed"],
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

    async def test_alfworld_requires_exact_canonical_instruction(self) -> None:
        target = "/games/game-a.tw-pddl"
        callback_called = False

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [target]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                self._env = SimpleNamespace(current_game_file=target)
                return "Welcome. Your task is to: look at alarmclock under the desklamp."

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
            question="Look at an alarm clock by lamp light.",
            environment={
                "env_type": "alfworld",
                "env_config": {"game_file": target, "max_steps": 50},
            },
        )
        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(
                record,
                "",
                run_graph=run_graph,
                max_environment_steps=50,
            )

        self.assertFalse(outcome.valid)
        self.assertEqual("alfworld_instruction_mismatch", outcome.reason)
        self.assertFalse(callback_called)

    async def test_alfworld_requires_boolean_terminal_won(self) -> None:
        target = "/games/game-a.tw-pddl"

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [target]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                self._env = SimpleNamespace(current_game_file=target)
                return "Room"

            def step(self, action):
                return "Finished", 10.0, True, {}

        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        record = task(
            "ALFWorld",
            question="look",
            environment={
                "env_type": "alfworld",
                "env_config": {"game_file": target, "max_steps": 50},
            },
        )
        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(
                record,
                "",
                run_graph=lambda _: "look",
                max_environment_steps=50,
            )

        self.assertFalse(outcome.valid)
        self.assertEqual("alfworld_terminal_success_unavailable", outcome.reason)
        self.assertIsNone(outcome.reward)

    async def test_alfworld_pins_the_recorded_step_limit(self) -> None:
        target = "/games/game-a.tw-pddl"

        class AlfredEnvConfig:
            config_file = ""

        class Inventory:
            def __init__(self, config, mode):
                self.game_files = [target]

        class Adapter:
            def __init__(self):
                self._env = None
                self.available_actions = ["look"]

            def reset(self, env_type, env_config, question="", extra=None):
                self._env = SimpleNamespace(current_game_file=target)
                return "Room"

        module = SimpleNamespace(
            AlfredEnvConfig=AlfredEnvConfig,
            ALFWorldEnv=Inventory,
            RAGENAdapter=Adapter,
        )
        record = task(
            "ALFWorld",
            question="look",
            environment={
                "env_type": "alfworld",
                "env_config": {"game_file": target, "max_steps": 50},
            },
        )
        with patch("src.interactive.task_evaluator._load_ragen_module", return_value=module):
            outcome = await evaluate_task(
                record,
                "",
                run_graph=lambda _: "look",
                max_environment_steps=20,
            )

        self.assertFalse(outcome.valid)
        self.assertEqual("alfworld_step_limit_mismatch", outcome.reason)


if __name__ == "__main__":
    unittest.main()
