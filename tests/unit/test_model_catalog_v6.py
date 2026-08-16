from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import load_model_registry
from src.interactive.director import AgentGraphOrchestrator


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "config/model_catalog_hotpotqa_deep_v6.yaml"
ARTIFACT_DIR = ROOT / "artifacts/hotpotqa_multiagent_skill/model_catalog_v6"
REMOTE_IDS = {
    "qwen3.5-flash",
    "qwen3.5-plus",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gpt-4o-mini",
    "MiniMax-M2.5",
    "MiniMax-M3",
    "glm-4.5-flash",
    "kimi-k2",
}
ALL_IDS = REMOTE_IDS | {"qwen3.5-9b-local"}


class _UnusedGateway:
    async def generate(self, request):  # pragma: no cover - prompt rendering never executes
        raise AssertionError(f"unexpected model call: {request.request_id}")


class ModelCatalogV6Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_model_registry(CATALOG_PATH)
        self.model_list = json.loads(
            (ARTIFACT_DIR / "model_list_receipt.json").read_text(encoding="utf-8")
        )
        self.new_canary = json.loads(
            (ARTIFACT_DIR / "canary_receipt.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (ARTIFACT_DIR / "evidence_manifest.json").read_text(encoding="utf-8")
        )
        self.visible_receipt = json.loads(
            (ARTIFACT_DIR / "director_visible_catalog_receipt.json").read_text(
                encoding="utf-8"
            )
        )

    def test_catalog_is_ten_equal_prior_callable_arms(self) -> None:
        self.assertEqual(ALL_IDS, set(self.registry.model_ids))
        self.assertEqual(10, len(self.registry))
        for model_id in self.registry.model_ids:
            with self.subTest(model_id=model_id):
                model = self.registry.require_model(model_id)
                self.assertEqual(1.0, model.selection_weight)
                self.assertEqual(1.0, model.cheap_weight)
                self.assertEqual(1.0, model.fast_weight)
                self.assertIsNotNone(self.registry.provider_for(model_id))
                graph = AgentGraph(
                    [AgentNode("output", model_id, "Answer from supplied evidence.")],
                    output_agent_id="output",
                )
                self.assertTrue(graph.validate(self.registry, require_complete=True).valid)

        for model_id in REMOTE_IDS:
            self.assertEqual(model_id, self.registry.require_model(model_id).model_name)

    def test_remote_arms_are_exact_current_text_ids(self) -> None:
        self.assertEqual(524, self.model_list["model_count"])
        objects = {
            str(item["id"]): item
            for item in self.model_list["models"]
            if isinstance(item, dict) and item.get("id")
        }
        self.assertTrue(REMOTE_IDS <= set(objects))
        for model_id in REMOTE_IDS:
            with self.subTest(model_id=model_id):
                item = objects[model_id]
                self.assertEqual("文本", item["model_type"])
                self.assertIn("openai", item["supported_endpoint_types"])

    def test_new_candidates_have_one_successful_canary_each(self) -> None:
        canaries = self.new_canary["canaries"]
        self.assertEqual({"glm-4.5-flash", "kimi-k2"}, {x["model_id"] for x in canaries})
        for receipt in canaries:
            with self.subTest(model_id=receipt["model_id"]):
                self.assertEqual("passed", receipt["status"])
                self.assertTrue(receipt["compatible"])
                self.assertEqual(1, receipt["attempt_count"])
                self.assertEqual("<answer>Paris</answer>", receipt["response"])
                self.assertIsInstance(receipt["request_id"], str)
                self.assertGreater(receipt["total_tokens"], 0)
                self.assertGreater(receipt["latency_ms"], 0)

        self.assertEqual(
            {
                "model_list_requests": 1,
                "text_canary_requests": 2,
                "retries": 0,
                "total": 3,
            },
            self.manifest["new_external_api_calls"],
        )

    def test_every_arm_has_passed_non_secret_evidence(self) -> None:
        evidence = {item["catalog_model_id"]: item for item in self.manifest["models"]}
        self.assertEqual(ALL_IDS, set(evidence))
        for model_id, item in evidence.items():
            with self.subTest(model_id=model_id):
                self.assertEqual("passed", item["evidence"]["status"])
                source = ROOT / item["evidence"]["source"]
                self.assertTrue(source.is_file(), source)
                if model_id in REMOTE_IDS:
                    self.assertTrue(item["model_list_present"])
                if item["evidence"]["kind"] == "successful_hotpot_executor_execution":
                    record = json.loads(
                        source.read_text(encoding="utf-8").splitlines()[
                            item["evidence"]["jsonl_line"] - 1
                        ]
                    )
                    executions = [
                        execution
                        for turn in record["turns"]
                        for execution in turn["executions"]
                        if execution["execution_id"] == item["evidence"]["execution_id"]
                    ]
                    self.assertEqual(1, len(executions))
                    self.assertIsNone(executions[0]["error_type"])
                    self.assertIsNotNone(
                        re.fullmatch(r"<answer>[^<>]+</answer>", executions[0]["output"].strip())
                    )

    def test_existing_renderer_exposes_all_arms_and_neutral_attributes(self) -> None:
        env = AgentWorkflowEnv(
            self.registry,
            gateway=_UnusedGateway(),
            problem="A neutral prompt-rendering test.",
        )
        orchestrator = AgentGraphOrchestrator(
            self.registry,
            client=object(),  # type: ignore[arg-type]
            catalog_order_seed="hotpotqa-architecture-v6",
        )
        rendered = orchestrator.build_prompt(env, 0, ())
        state = json.loads(rendered.split("\n\n", 1)[1])
        visible = {item["model_id"]: item for item in state["model_catalog"]}
        self.assertEqual(ALL_IDS, set(visible))
        self.assertEqual(state["model_catalog"], self.visible_receipt["model_catalog"])

        required_profile_facts = (
            "provider=",
            "locality=",
            "reasoning=",
            "latency=",
            "context=",
            "instruction=",
            "concise=",
            "availability=",
        )
        forbidden_recipes = ("bridge", "comparison", "researcher", "verifier", "hotpot")
        for model_id, item in visible.items():
            with self.subTest(model_id=model_id):
                metadata = item["routing_metadata"]
                profile = metadata["profile"].lower()
                for fact in required_profile_facts:
                    self.assertIn(fact, profile)
                for recipe in forbidden_recipes:
                    self.assertNotIn(recipe, profile)
                self.assertTrue(metadata["text_qa_canary"].startswith("passed"))
                self.assertEqual(
                    "hotpotqa_multiagent_skill/model_catalog_v6/evidence_manifest.json",
                    metadata["canary_source"],
                )


if __name__ == "__main__":
    unittest.main()
