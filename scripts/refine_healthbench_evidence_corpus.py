"""Offline source enrichment over the existing SkillFlow-backed evidence store.

Reuse build_healthbench_question_corpus's source admission, Tool receipt
extraction and immutable index publication. No rubric/task-answer input,
clinical generation, network requests, or score-dependent record selection.
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

from src.interactive.healthbench_knowledge_store import (
    HealthBenchKnowledgeStore, load_frozen_public_evidence, _PUBLIC_EVIDENCE_FIELDS,
)
from src.interactive.openai_gateway import _healthbench_search_candidates
from src.interactive.qa_retrieval import DEFAULT_SKILLFLOW_SOURCE
from report_healthbench_agentgraph_development import _evidence_trajectory
from report_multidataset_stable_zero import _tool_receipts


def refine(source_manifest, medrag_path, output, receipt_paths=()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = list(load_frozen_public_evidence(source_manifest))
    imported = 0
    # Every supplied trajectory contributes only real external Tool evidence,
    # regardless of score or whether the producer subsequently failed.
    for receipt_path in receipt_paths:
        with Path(receipt_path).open(encoding="utf-8") as stream:
            for line in stream:
                trajectory = json.loads(line)
                for _, _, evidence in _healthbench_search_candidates(_tool_receipts(_evidence_trajectory(trajectory))):
                    rows.append(dict(evidence))
                    imported += 1
    wanted = {str(row["document_id"]) for row in rows if row.get("source") == "MedRAG/textbooks"}
    full = {}
    with Path(medrag_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row.get("id")) in wanted:
                text = row.get("contents")
                if isinstance(text, str) and text.strip():
                    full[str(row["id"])] = text
    store = HealthBenchKnowledgeStore(output, (), DEFAULT_SKILLFLOW_SOURCE,
                                     metadata_aware_retrieval=True)
    counts = Counter()
    expanded = 0
    identity_fields_added = 0
    missing_source_ids = set()
    try:
        for original in rows:
            row = {key: value for key, value in original.items() if key in _PUBLIC_EVIDENCE_FIELDS}
            if row.get("source") == "MedRAG/textbooks":
                text = full.get(str(row["document_id"]))
                if text is not None:
                    was_expanded = len(text) > len(str(row.get("excerpt", "")))
                    row.update(excerpt=text, text_scope="full_local_source_passage",
                               total_characters=len(text), truncated=False, offset=0,
                               next_offset=None)
                else:
                    missing_source_ids.add(str(row["document_id"]))
                    row.setdefault("text_scope", "retrieved_excerpt_full_text_unconfirmed")
                    was_expanded = False
            else:
                was_expanded = False
                row.setdefault("text_scope", "retrieved_excerpt_full_text_unconfirmed")
            added_identifier = False
            # NCBI's existing document ID is a PMID, not a model-inferred alias.
            if row.get("source") == "NCBI PubMed" and str(row.get("document_id", "")).isdigit():
                if not row.get("pmid"):
                    row["pmid"] = str(row["document_id"])
                    added_identifier = True
                row.setdefault("source_id", "pubmed:" + str(row["document_id"]))
            if store.ingest_evidence(row):
                counts[str(row["source"])] += 1
                expanded += int(was_expanded)
                identity_fields_added += int(added_identifier)
        databases = {}
        for name in ("medical_references", "drug_labels"):
            result = store.search(name, "clinical", limit=1)
            if result["index_receipt"]:
                databases[name] = result["index_receipt"]
        manifest = {
            "schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "corpus_version": "healthbench-public525-external-evidence-20260907-v2",
            "status": "frozen", "source_manifest": str(Path(source_manifest).resolve()),
            "source_counts": dict(counts), "databases": databases,
            "construction_inputs": "previous external evidence plus local original MedRAG passages and all supplied external Tool observations",
            "medrag_source_path": str(Path(medrag_path).resolve()),
            "tool_receipt_paths": [str(Path(path).resolve()) for path in receipt_paths],
            "tool_evidence_occurrences_considered": imported,
            "expanded_source_records": expanded,
            "source_identifier_fields_added": identity_fields_added,
            "missing_local_document_ids": sorted(missing_source_ids),
            "coverage_interpretation": "more complete indexed source material; not verified answer coverage or a QA score",
            "excluded_from_corpus": ["rubric", "reference_response", "grade", "final_answer", "agent_summary", "task_to_answer_mapping"],
            "medical_model_generation_calls": 0, "grader_calls": 0, "new_network_calls": 0,
            "evaluation_condition": "public-test development retrieval adaptation, not untouched held-out evaluation",
        }
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"manifest": str(output / "manifest.json"), **{key: manifest[key] for key in (
            "source_counts", "expanded_source_records", "source_identifier_fields_added", "missing_local_document_ids",
        )}}, ensure_ascii=False), flush=True)
        return manifest
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--medrag-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, action="append", default=[])
    args = parser.parse_args()
    refine(args.source_manifest, args.medrag_path, args.output, args.receipts)
