"""Question-aware evidence corpus, reusing SkillFlow retrieval and Tool receipts.

Only public conversations select evidence. No rubric, reference response,
generated answer, grade or per-task answer mapping enters the searchable corpus.
This is a transductive retrieval condition, not an untouched-test claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore
from src.interactive.healthbench_professional_adapter import parse_model_visible_conversation
from src.interactive.healthbench_tool_adapter import FrozenMedRAGBM25Corpus
from src.interactive.openai_gateway import _healthbench_medrag_evidence, _healthbench_search_candidates
from src.interactive.config_loader import load_yaml
from report_multidataset_stable_zero import _tool_receipts
from train_agentgraph_smoke import _write_json


def public_questions(path):
    tasks = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            # Select fields before any retrieval; evaluator payload is unused.
            tasks.append((str(row["task_id"]), str(row["question"])))
    if len(tasks) != 525 or len({key for key, _ in tasks}) != 525:
        raise ValueError("expected the original 525 unique public conversations")
    return tasks


def public_queries(question):
    messages = parse_model_visible_conversation(question)
    users = [m["content"] for m in messages if m["role"] == "user"]
    # Final user query plus complete dialogue context; no model query expansion.
    # Retrieval windows do not replace the full question in model inputs.
    values = [users[-1] if users else question, "\n".join(m["content"] for m in messages)]
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def build(config_path, output, receipt_paths):
    config = load_yaml(config_path)
    tasks = public_questions(ROOT / config["data"]["test_path"])
    ids = {key for key, _ in tasks}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("corpus already frozen; use its existing manifest or a new directory")
    settings = config["healthbench_tool_runtime"]
    store = HealthBenchKnowledgeStore(output, (), Path(settings["skillflow_source"]))
    source_counts = Counter()
    task_receipts = []
    cache_task_ids = set()
    cache_sources = []
    def ingest(evidence):
        accepted = store.ingest_evidence(evidence)
        if accepted:
            source_counts[str(evidence["source"])] += 1
        return accepted

    try:
        # Reuse only actual external Tool observations, irrespective of scores.
        # No failed/successful-demo selection and no Agent summary ingestion.
        for path in receipt_paths:
            count = 0
            with Path(path).open(encoding="utf-8") as stream:
                for line in stream:
                    trajectory = json.loads(line)
                    task_id = trajectory.get("task", {}).get("task_id")
                    if task_id not in ids:
                        continue
                    receipts = _tool_receipts(trajectory)
                    for _, _, evidence in _healthbench_search_candidates(receipts):
                        count += int(ingest(evidence))
                    if receipts:
                        cache_task_ids.add(task_id)
            cache_sources.append({"path": str(Path(path).resolve()), "new_evidence_records": count})
            print(json.dumps({"stage": "external_receipts_imported", **cache_sources[-1]}), flush=True)

        corpus = FrozenMedRAGBM25Corpus.open(
            settings["resource_dir"], source_identity=settings["source_identity"],
            expected_source_revision=settings["source_revision"], expected_rows=settings["expected_rows"],
        )
        try:
            for index, (task_id, question) in enumerate(tasks, 1):
                queries = public_queries(question)
                returned = 0
                for query in queries:
                    chunks = corpus.search(query)
                    returned += len(chunks)
                    for evidence in _healthbench_medrag_evidence({
                        "frozen_corpus": {"source": settings["source_identity"]}, "ranked_chunks": chunks,
                    }):
                        ingest(evidence)
                task_receipts.append({"task_id": task_id, "public_queries": list(queries),
                    "retrieval_result_count": returned, "correctness_verified": False})
                if index % 50 == 0 or index == len(tasks):
                    print(json.dumps({"stage": "public_question_retrieval", "processed": index,
                                      "total": len(tasks), "records": store.counts()}), flush=True)
        finally:
            corpus.close()

        databases = {}
        for name in ("medical_references", "drug_labels"):
            result = store.search(name, "clinical", limit=1)
            if result["index_receipt"]:
                databases[name] = result["index_receipt"]
        manifest = {
            "schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "corpus_version": "healthbench-public525-external-evidence-20260907-v1",
            "status": "frozen", "question_count": len(tasks),
            "question_retrieval_nonempty": sum(r["retrieval_result_count"] > 0 for r in task_receipts),
            "coverage_interpretation": "lexical retrieval returned passages; not verified answer coverage",
            "source_counts": dict(source_counts), "databases": databases,
            "cached_tool_task_count": len(cache_task_ids), "cached_tool_sources": cache_sources,
            "construction_inputs": "525 public conversations plus external database documents/Tool observations",
            "excluded_from_corpus": ["rubric", "reference_response", "final_answer", "grade",
                                    "agent_summary", "all_other_task_conversations", "task_to_answer_mapping"],
            "medical_model_generation_calls": 0, "grader_calls": 0, "new_network_calls": 0,
            "evaluation_condition": "question-aware retrieval adaptation; all test questions available at corpus construction",
        }
        with (output / "question_retrieval_receipts.jsonl").open("x", encoding="utf-8") as stream:
            for row in task_receipts:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_json(manifest_path, manifest)
        print(json.dumps({"manifest": str(manifest_path), "question_count": len(tasks),
                          "sources": dict(source_counts), "databases": store.counts()}), flush=True)
        return manifest
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, action="append", default=[])
    args = parser.parse_args()
    build(args.config, args.output, args.receipts)
