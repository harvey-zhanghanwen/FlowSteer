from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from src.interactive.hotpotqa_embedding_tool import (
    HOTPOTQA_FACT_MEMORY_TOOL_ID,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
)
from src.interactive.tool_runtime import ToolRequest


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts" / "evaluate_hotpotqa_round.py"
    spec = importlib.util.spec_from_file_location("evaluate_hotpotqa_round_v21", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner()


@dataclass(frozen=True)
class _Manifest:
    schema_version: str = "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
    index_id: str = "hotpotqa-full-dataset-fact-test-v1"
    corpus_version: str = FULL_DATASET_FACT_MEMORY_CORPUS_VERSION
    source: str = "HotpotQA declarative facts"
    source_splits: tuple[str, ...] = ("train", "validation")
    embedding_model: str = "test-encoder"
    embedding_dimension: int = 3
    normalized: bool = True
    similarity: str = "cosine"
    frozen_top_k: int = 2
    source_record_count: int = 97_852
    source_train_count: int = 90_447
    source_validation_count: int = 7_405
    unique_source_count: int = 97_852
    cycled_record_count: int = 0
    question_rewrite_count: int = 97_852
    fact_count: int = 97_852
    semantic_rewrite_coverage: float = 1.0
    frozen_evaluation_count: int = 128
    evaluation_overlap_count: int = 128
    contains_evaluation_source_facts: bool = True
    contains_raw_questions: bool = False
    contains_raw_answers: bool = False
    document_format: str = "declarative_fact_only"
    indexed_text_field: str = "fact_text"
    evaluation_scope: str = FULL_DATASET_EVALUATION_SCOPE
    official_heldout_eligible: bool = False
    paraphrase_versions: tuple[str, ...] = ("fact-v1",)
    paraphrase_provenances: tuple[str, ...] = ("local-qwen3.5-9b",)


@dataclass(frozen=True)
class _Hit:
    memory_id: str
    fact_snippet: str
    similarity: float
    rank: int


@dataclass(frozen=True)
class _Fact:
    memory_id: str
    fact_text: str


class _Index:
    manifest = _Manifest()

    def search(self, query: str, k: int) -> tuple[_Hit, ...]:
        del query
        assert k == 2
        return tuple(
            _Hit(f"m{rank}", f"Declarative fact {rank}.", 1.0 - rank / 10.0, rank)
            for rank in range(1, 3)
        )

    def read(self, memory_id: str) -> _Fact:
        return _Fact(memory_id, "Ada authored the work.")


class HotpotQARound01FullDatasetFactMemoryV21ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v19 = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_transductive_qa_memory_v19.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.candidate = yaml.safe_load(
            (
                ROOT
                / "config/evaluation_hotpotqa_round01_full_dataset_fact_memory_v21.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.topk5_build = yaml.safe_load(
            (
                ROOT
                / "config/build_hotpotqa_round01_full_dataset_fact_memory_topk5_v21.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_round01_architecture_is_preserved_except_fact_tool_identity(self) -> None:
        self.assertEqual(self.v19["director"], self.candidate["director"])
        baseline_graph = dict(self.v19["agent_graph"])
        candidate_graph = dict(self.candidate["agent_graph"])
        baseline_graph.pop("required_evidence_tool_id", None)
        candidate_graph.pop("required_evidence_tool_id", None)
        self.assertEqual(baseline_graph, candidate_graph)
        for section in (
            "evaluation",
            "grpo",
            "policy_sync",
            "exploration",
            "skills",
            "gpu",
            "deployment",
        ):
            self.assertEqual(self.v19[section], self.candidate[section])
        _RUNNER.validate_hotpot_config(self.candidate)

    def test_fact_scope_and_worker_boundary_are_explicit(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        graph = self.candidate["agent_graph"]
        self.assertEqual("full_dataset_fact_memory", retrieval["corpus_kind"])
        self.assertEqual(97_852, retrieval["source_record_count"])
        self.assertEqual(90_447, retrieval["source_train_count"])
        self.assertEqual(7_405, retrieval["source_validation_count"])
        self.assertEqual(128, retrieval["evaluation_overlap_count"])
        self.assertEqual(FULL_DATASET_EVALUATION_SCOPE, retrieval["evaluation_scope"])
        self.assertTrue(retrieval["contains_evaluation_source_facts"])
        self.assertFalse(retrieval["contains_raw_questions"])
        self.assertFalse(retrieval["contains_raw_answers"])
        self.assertEqual("declarative_fact_only", retrieval["document_format"])
        self.assertEqual("fact_text", retrieval["indexed_text_field"])
        self.assertFalse(retrieval["official_heldout_eligible"])
        self.assertFalse(retrieval["web_search_enabled"])
        self.assertTrue(retrieval["semantic_query_rewrite_required"])
        self.assertTrue(retrieval["raw_question_embedding_query_forbidden"])
        self.assertNotIn("allowed_tools", self.candidate["director"])
        self.assertEqual("control_plane", graph["director_feedback_mode"])
        self.assertEqual(HOTPOTQA_FACT_MEMORY_TOOL_ID, graph["required_evidence_tool_id"])
        self.assertTrue(graph["require_evidence_relation"])
        self.assertEqual(
            "public_task_dynamic_full_dataset_fact_memory_search_read",
            _RUNNER._input_context(self.candidate),
        )

    def test_temporary_topk5_build_is_isolated_and_rejected_by_evaluator(self) -> None:
        build_experiment = self.topk5_build["experiment"]
        final_experiment = self.candidate["experiment"]
        self.assertTrue(build_experiment["build_only"])
        self.assertEqual("hotpotqa_retrieval_index_build", build_experiment["phase"])
        self.assertNotEqual(
            final_experiment["condition_id"], build_experiment["condition_id"]
        )
        self.assertNotEqual(
            final_experiment["output_dir"], build_experiment["output_dir"]
        )
        self.assertEqual(
            build_experiment["condition_id"],
            self.topk5_build["qa_embedding_retrieval"]["condition_id"],
        )
        self.assertEqual(
            "data/hotpotqa_full_dataset_fact_memory_v1/index_topk5",
            self.topk5_build["qa_embedding_retrieval"]["index_dir"],
        )
        self.assertEqual(5, self.topk5_build["qa_embedding_retrieval"]["search_top_k"])
        for field in (
            "root",
            "selected_tasks_path",
            "direct_predictions_path",
            "trajectories_path",
            "failures_path",
            "paired_results_path",
            "wrong_demos_path",
            "error_demos_path",
            "manifest_path",
            "preflight_receipt_path",
            "report_json_path",
            "report_markdown_path",
        ):
            self.assertNotEqual(
                self.candidate["storage"][field],
                self.topk5_build["storage"][field],
            )
        with self.assertRaisesRegex(
            _RUNNER.ConfigurationError,
            "build-only",
        ):
            _RUNNER.validate_hotpot_config(self.topk5_build)

    def test_final_profile_records_and_requires_topk_selection_evidence(self) -> None:
        selection_path = (
            ROOT
            / "data/hotpotqa_full_dataset_fact_memory_v1/top_k_selection.json"
        )
        paths = _RUNNER._paths(self.candidate, ROOT)

        self.assertEqual(selection_path, paths["retrieval_profile_selection"])
        self.assertEqual(
            (
                ROOT
                / "artifacts/hotpotqa_round01_full_dataset_fact_memory_v21/"
                "fact_memory_index_smoke_receipt.json"
            ),
            paths["retrieval_index_smoke"],
        )
        self.assertEqual(
            (
                "retrieval_profile_selection",
                "retrieval_index_manifest",
                "retrieval_index_smoke",
                "paraphrase_manifest",
            ),
            _RUNNER._retrieval_evidence_names(
                self.candidate["qa_embedding_retrieval"]
            ),
        )

    def test_final_topk_must_match_selection_and_index_manifest(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        _RUNNER._validate_full_dataset_top_k_freeze(
            retrieval,
            {"frozen_top_k": 2},
            {"selected_top_k": 2},
        )
        with self.assertRaisesRegex(
            _RUNNER.HotpotRoundError,
            "selected, configured, and indexed Top-K differ",
        ):
            _RUNNER._validate_full_dataset_top_k_freeze(
                retrieval,
                {"frozen_top_k": 2},
                {"selected_top_k": 5},
            )

    def test_pre_run_evidence_gate_requires_matching_passed_smoke(self) -> None:
        retrieval = self.candidate["qa_embedding_retrieval"]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "retrieval_profile_selection": root / "selection.json",
                "retrieval_index_manifest": root / "index.json",
                "retrieval_index_smoke": root / "smoke.json",
                "paraphrase_manifest": root / "materialization.json",
            }
            index_manifest = {"index_id": "fact-index-final", "frozen_top_k": 2}
            paths["retrieval_profile_selection"].write_text(
                json.dumps({"selected_top_k": 2}), encoding="utf-8"
            )
            paths["retrieval_index_manifest"].write_text(
                json.dumps(index_manifest), encoding="utf-8"
            )
            paths["paraphrase_manifest"].write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                _RUNNER.HotpotRoundError,
                "artifact is missing",
            ):
                _RUNNER._read_and_validate_retrieval_evidence(retrieval, paths)

            smoke = {
                "passed": True,
                "same_query_top_k_deterministic": True,
                "worker_receipt_ownership_valid": True,
                "fact_only_search_read_projection_valid": True,
                "qa_fields_absent_from_retrieval_payload": True,
                "qa_wire_absent_from_retrieval_payload": True,
                "tool_id": "hotpotqa.fact_memory",
                "web_search_used": False,
                "model_api_calls": 0,
                "index_manifest": index_manifest,
            }
            paths["retrieval_index_smoke"].write_text(
                json.dumps({**smoke, "passed": False}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                _RUNNER.HotpotRoundError,
                "smoke failed",
            ):
                _RUNNER._read_and_validate_retrieval_evidence(retrieval, paths)

            paths["retrieval_index_smoke"].write_text(
                json.dumps(smoke), encoding="utf-8"
            )
            evidence = _RUNNER._read_and_validate_retrieval_evidence(
                retrieval,
                paths,
            )
            self.assertEqual(set(paths), set(evidence))

    def test_retrieval_evidence_gate_runs_before_backend_construction(self) -> None:
        with TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            config = copy.deepcopy(self.candidate)
            for field in (
                "root",
                "selected_tasks_path",
                "direct_predictions_path",
                "trajectories_path",
                "failures_path",
                "paired_results_path",
                "wrong_demos_path",
                "error_demos_path",
                "manifest_path",
                "preflight_receipt_path",
                "report_json_path",
                "report_markdown_path",
            ):
                config["storage"][field] = str(temporary_root / field)
            config_path = temporary_root / "evaluation.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (
                patch.object(
                    _RUNNER,
                    "_read_and_validate_retrieval_evidence",
                    side_effect=_RUNNER.HotpotRoundError("missing frozen evidence"),
                ) as evidence_gate,
                patch.object(_RUNNER.LiveSmokeBackend, "from_config") as backend,
            ):
                with self.assertRaisesRegex(
                    _RUNNER.HotpotRoundError,
                    "evidence preflight failed",
                ):
                    asyncio.run(
                        _RUNNER.run_hotpot_round(
                            config_path,
                            project_root=ROOT,
                        )
                    )
            evidence_gate.assert_called_once()
            backend.assert_not_called()

    def test_canary_and_full_trajectory_boundary_requires_three_assertions(self) -> None:
        boundary = {
            "director_tool_calls": 0,
            "retrieval_tool_calls_by_worker": 4,
            "retrieval_artifact_routed_via_relation": True,
        }
        _RUNNER._validate_memory_retrieval_execution_boundary(boundary)
        for field, invalid in (
            ("director_tool_calls", 1),
            ("retrieval_tool_calls_by_worker", 0),
            ("retrieval_artifact_routed_via_relation", False),
        ):
            with self.assertRaisesRegex(
                _RUNNER.HotpotRoundError,
                "Director/worker/relation",
            ):
                _RUNNER._validate_memory_retrieval_execution_boundary(
                    {**boundary, field: invalid}
                )

    def test_registry_exposes_fact_only_search_and_read(self) -> None:
        registry = build_hotpotqa_embedding_tool_registry(
            _Index(),
            task_id="hotpotqa:fixed-evaluation",
            tool_id=HOTPOTQA_FACT_MEMORY_TOOL_ID,
        )
        search_result, search_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_FACT_MEMORY_TOOL_ID,
                ToolRequest("search", {"query": "Who authored the work", "k": 2}),
            )
        )
        self.assertIsNone(search_receipt.error_type)
        assert search_result is not None
        identity = search_result.value["retrieval_index"]
        self.assertEqual("full_dataset_fact_memory", identity["corpus_kind"])
        self.assertFalse(identity["contains_raw_questions"])
        self.assertFalse(identity["contains_raw_answers"])
        self.assertEqual("fact_text", identity["indexed_text_field"])
        hit = search_result.value["hits"][0]
        self.assertEqual(
            {"memory_id", "fact_snippet", "similarity", "rank"}, set(hit)
        )

        read_result, read_receipt = asyncio.run(
            registry.ainvoke_with_receipt(
                HOTPOTQA_FACT_MEMORY_TOOL_ID,
                ToolRequest("read", {"memory_id": "m1"}),
            )
        )
        self.assertIsNone(read_receipt.error_type)
        assert read_result is not None
        self.assertEqual(
            {"memory_id", "fact_text"}, set(read_result.value["fact"])
        )
        forbidden = {
            "question",
            "original_question",
            "canonical_answer",
            "paraphrase_question",
            "paraphrase_answer_statement",
            "ground_truth",
        }
        self.assertTrue(forbidden.isdisjoint(read_result.value["fact"]))


if __name__ == "__main__":
    unittest.main()
