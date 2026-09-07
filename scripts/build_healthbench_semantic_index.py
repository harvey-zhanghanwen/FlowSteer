"""Build local BGE evidence vectors and answer-free retrieval receipts for 525 questions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from src.interactive.healthbench_knowledge_store import load_frozen_public_evidence, _PUBLIC_EVIDENCE_FIELDS
from src.interactive.healthbench_semantic_retrieval import (
    SCHEMA, encoder, windows, encode_texts, source_text, SemanticEvidenceIndex,
)
from src.interactive.config_loader import load_yaml
from build_healthbench_question_corpus import public_questions, public_queries


def build(source_manifest, output, model_path, config_path, *, skip_query_probes=False):
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = encoder(str(Path(model_path).resolve()))
    records = [{k: v for k, v in row.items() if k in _PUBLIC_EVIDENCE_FIELDS}
               for row in load_frozen_public_evidence(source_manifest)]
    chunks, owners = [], []
    with (output / "records.jsonl").open("x", encoding="utf-8") as rows, \
         (output / "chunk_offsets.jsonl").open("x", encoding="utf-8") as offsets:
        for index, record in enumerate(records):
            rows.write(json.dumps(record, ensure_ascii=False) + "\n")
            for text, start, end in windows(model, source_text(record)):
                chunks.append(text)
                owners.append(index)
                offsets.write(json.dumps({"record_index": index, "text_start": start, "text_end": end}) + "\n")
    vectors = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype="float32",
        shape=(len(chunks), model.get_sentence_embedding_dimension()))
    for start in range(0, len(chunks), 256):
        batch = encode_texts(model, chunks[start:start+256])
        vectors[start:start+len(batch)] = batch
        vectors.flush()
        print(json.dumps({"stage": "encode_sources", "completed": start + len(batch),
            "total": len(chunks), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    np.save(output / "owners.npy", np.asarray(owners, dtype=np.int64), allow_pickle=False)
    manifest = {"schema_version": SCHEMA, "status": "frozen",
        "encoder": "BAAI/bge-base-en-v1.5", "encoder_path": str(Path(model_path).resolve()),
        "source_manifest": str(Path(source_manifest).resolve()), "record_count": len(records),
        "window_count": len(chunks), "dimension": int(vectors.shape[1]),
        "window_tokens": 480, "overlap_tokens": 64, "source_text_preserved": True,
        "source_counts": dict(Counter(row["source"] for row in records)),
        "retrieval": "SkillFlow normalized BM25 0.4 + BGE cosine 0.6; threshold 0.15",
        "scope": "semantic access to existing public525 external-evidence library; not a new full-corpus harvest",
        "training_performed": False, "paid_model_calls": 0, "grader_calls": 0}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    if skip_query_probes:
        print(json.dumps({"stage": "complete", **manifest, "public_query_probes_executed": False,
                          "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
        return
    index = SemanticEvidenceIndex(output / "manifest.json")
    config = load_yaml(config_path)
    tasks = public_questions(ROOT / config["data"]["test_path"])
    with (output / "public525_semantic_query_receipts.jsonl").open("x", encoding="utf-8") as stream:
        for number, (task_id, question) in enumerate(tasks, 1):
            for query in public_queries(question):
                hits, receipt = index.rank(query, records)
                stream.write(json.dumps({"task_id": task_id, "public_query": query,
                    "hits": [{k: row.get(k) for k in ("source", "source_id", "document_id", "title", "retrieval_scores")} for row in hits],
                    "retrieval_receipt": receipt, "correctness_verified": False}, ensure_ascii=False) + "\n")
                stream.flush()
            if number % 50 == 0 or number == len(tasks):
                print(json.dumps({"stage": "public_semantic_queries", "completed": number, "total": len(tasks)}), flush=True)
    print(json.dumps({"stage": "complete", **manifest, "public_question_count": len(tasks),
        "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-query-probes", action="store_true",
                        help="Build only the local source index; do not replay 525 public queries")
    args = parser.parse_args()
    build(args.source_manifest, args.output, args.model_path, args.config,
          skip_query_probes=args.skip_query_probes)
