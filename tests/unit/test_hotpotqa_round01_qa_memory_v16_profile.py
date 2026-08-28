import json
from pathlib import Path
import unittest

import yaml

from src.interactive.config_loader import validate_agent_graph_config


ROOT = Path(__file__).resolve().parents[2]


class HotpotQARound01QAMemoryV16ProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.best = yaml.safe_load(
            (ROOT / "config/evaluation_hotpotqa_round_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.candidate = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_qa_memory_v16.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_round01_director_and_canvas_profile_is_frozen(self) -> None:
        for key in (
            "seed",
            "prompt_version",
        ):
            self.assertEqual(
                self.candidate["experiment"][key], self.best["experiment"][key]
            )
        self.assertEqual(self.candidate["director"], self.best["director"])
        self.assertEqual(
            self.candidate["agent_graph"],
            {
                **self.best["agent_graph"],
                "model_catalog_path": (
                    "config/model_catalog_hotpotqa_round01_frozen_v1.yaml"
                ),
                "required_evidence_tool_id": "hotpotqa.qa_memory",
                "require_evidence_relation": True,
                "director_feedback_mode": "control_plane",
            },
        )

    def test_round01_ignored_catalog_is_materialized_without_semantic_change(self) -> None:
        candidate_path = ROOT / self.candidate["agent_graph"]["model_catalog_path"]
        expected_path = ROOT.parent.parent / "FlowSteer" / "config/model_catalog.yaml"
        self.assertTrue(candidate_path.is_file())
        self.assertTrue(expected_path.is_file())
        candidate_catalog = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        expected_catalog = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(candidate_catalog, expected_catalog)

    def test_round01_selection_and_inactive_blocks_are_frozen(self) -> None:
        for key in (
            "dataset_key",
            "split",
            "selection",
            "sample_count",
            "rollouts_per_task",
            "concurrency",
            "direct_model_id",
        ):
            self.assertEqual(
                self.candidate["hotpotqa_evaluation"][key],
                self.best["hotpotqa_evaluation"][key],
            )
        for section in (
            "evaluation",
            "grpo",
            "policy_sync",
            "exploration",
            "skills",
            "deployment",
        ):
            self.assertEqual(self.candidate[section], self.best[section])
        validate_agent_graph_config(self.candidate)

    def test_prepared_validation_tasks_match_round01_exactly(self) -> None:
        def task_projection(path: Path) -> list[tuple[str, str, str]]:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                rows.append(
                    (row["task_id"], row["question"], row["ground_truth"])
                )
            return rows

        original = task_projection(
            ROOT / "artifacts/hotpotqa_round_01/selected_tasks.jsonl"
        )
        candidate = task_projection(
            ROOT
            / "artifacts/hotpotqa_round01_qa_memory_v16/selected_tasks.jsonl"
        )
        self.assertEqual(len(original), 128)
        self.assertEqual(candidate, original)

    def test_only_worker_qa_memory_retrieval_is_enabled(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        self.assertTrue(retrieval["enabled"])
        self.assertEqual(retrieval["tool_id"], "hotpotqa.qa_memory")
        self.assertEqual(retrieval["corpus_kind"], "train_qa_memory")
        self.assertEqual(retrieval["search_top_k"], 2)
        self.assertFalse(retrieval["web_search_enabled"])
        self.assertEqual(
            self.candidate["agent_graph"]["required_evidence_tool_id"],
            "hotpotqa.qa_memory",
        )
        self.assertTrue(
            self.candidate["agent_graph"]["require_evidence_relation"]
        )
        self.assertEqual(
            self.candidate["agent_graph"]["director_feedback_mode"],
            "control_plane",
        )
        self.assertNotIn("sampling_action_profile", self.candidate["director"])
        self.assertNotIn("terminal_protocol", self.candidate["agent_graph"])
        self.assertNotIn("recovery_policy", self.candidate["agent_graph"])

    def test_inference_only(self) -> None:
        self.assertFalse(self.candidate["experiment"]["training_enabled"])
        self.assertFalse(self.candidate["grpo"]["enabled"])
        self.assertEqual(self.candidate["grpo"]["max_optimizer_updates"], 0)
        self.assertFalse(self.candidate["skills"]["enabled"])


if __name__ == "__main__":
    unittest.main()
