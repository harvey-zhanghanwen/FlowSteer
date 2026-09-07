"""Thin, CPU-only SkillFlow hybrid retrieval over external source records.

Source: GenericTaskEnvironment._embed_score/_bm25_score/_search_passages
and training/tools.py::_search_context. Preserve upstream normalized cosine,
0.4 normalized BM25 + 0.6 dense weights, and 0.15 admission threshold.
Necessary adapters: persistent passage vectors, 512-token encoder windows,
structured source receipts, and reuse within the existing request-scoped Tool.
No Agent summaries, benchmark answers, model training, or evaluator access.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from threading import RLock

import numpy as np

from .healthbench_knowledge_store import _PUBLIC_EVIDENCE_FIELDS

SCHEMA = "flowsteer.healthbench.semantic-evidence-index.v1"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_ENCODER_LOCK = RLock()


@lru_cache(maxsize=2)
def encoder(model_path):
    from sentence_transformers import SentenceTransformer
    # Same model/device as SkillFlow; local-only prevents hidden downloads or
    # silent substitution if the configured resource is unavailable.
    if not Path(model_path).is_dir():
        raise RuntimeError("configured local semantic encoder is unavailable")
    return SentenceTransformer(str(model_path), device="cpu", local_files_only=True)


def source_identity(record):
    return json.dumps({k: v for k, v in record.items() if k in _PUBLIC_EVIDENCE_FIELDS},
                      sort_keys=True, ensure_ascii=False, allow_nan=False)


def source_text(record):
    return str(record.get("title") or record["document_id"]) + "\n" + str(record["excerpt"])


def windows(model, text):
    """Keep every source token using overlapping windows, not silent truncation."""
    offsets = model.tokenizer(text, add_special_tokens=False,
                              return_offsets_mapping=True, verbose=False)["offset_mapping"]
    width = min(480, model.max_seq_length - 32)
    if width <= 64:
        raise ValueError("semantic encoder context is too small")
    rows = []
    for start in range(0, len(offsets), width - 64):
        end = min(start + width, len(offsets))
        left, right = offsets[start][0], offsets[end - 1][1]
        rows.append((text[left:right], left, right))
        if end == len(offsets):
            break
    return rows


def encode_texts(model, texts):
    with _ENCODER_LOCK:
        # DIRECT_REUSE: SkillFlow normalized sentence embeddings on CPU.
        return model.encode(texts, normalize_embeddings=True, batch_size=16,
                            convert_to_numpy=True, show_progress_bar=False)


class SemanticEvidenceIndex:
    @staticmethod
    def _extract_query_terms(query: str) -> list:

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "to", "for", "and", "or", "but", "on", "at",
            "by", "with", "from", "that", "this", "it", "as", "not",
            "what", "who", "where", "when", "which", "how", "does", "did",
            "do", "has", "have", "had", "will", "would", "can", "could",
        }
        tokens = re.findall(r'\b\w+\b', query)
        return [t for t in tokens if t.lower() not in stop_words and len(t) > 1]

    def _bm25_score(self, query: str, passages: list) -> list:

        import math as _math

        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return [0.0] * len(passages)

        doc_tokens_list = []
        doc_lens = []
        for p in passages:
            tokens = re.findall(r'\b\w+\b', p.lower())
            doc_tokens_list.append(tokens)
            doc_lens.append(len(tokens))

        avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
        n_docs = len(passages)

        from collections import Counter
        df = Counter()
        for tokens in doc_tokens_list:
            df.update(set(tokens))

        k1, b = 1.5, 0.75
        scores = []
        for i, (tokens, dl) in enumerate(zip(doc_tokens_list, doc_lens)):
            tf_map = Counter(tokens)
            score = 0.0
            for qt in query_terms:
                qt_lower = qt.lower()
                tf = tf_map.get(qt_lower, 0)
                if tf == 0:
                    continue
                idf = _math.log(1 + (n_docs - df.get(qt_lower, 0) + 0.5) /
                                (df.get(qt_lower, 0) + 0.5))
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                score += idf * tf_norm
            scores.append(score)

        return scores

    def __init__(self, manifest_path):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA or self.manifest.get("status") != "frozen":
            raise ValueError("semantic evidence index is not frozen or compatible")
        self.model = encoder(self.manifest["encoder_path"])
        root = Path(manifest_path).parent
        with (root / "records.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.record_lookup = {source_identity(row): i for i, row in enumerate(records)}
        self.vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.owners = np.load(root / "owners.npy", allow_pickle=False)
        self.record_count = len(records)
        if self.vectors.shape[0] != len(self.owners) or len(records) != self.manifest["record_count"]:
            raise ValueError("semantic passage/record dimensions differ")

    def rank(self, query, records, limit=3):
        query_parts = windows(self.model, query)
        if not query_parts or not records:
            return [], {"backend": "skillflow-bm25-bge-hybrid", "matched_count": 0}
        queries = encode_texts(self.model, [QUERY_INSTRUCTION + t for t, _, _ in query_parts])
        # Same normalized dot-product as SkillFlow. Max over windows preserves
        # evidence near the end of a long source without copying it into prompts.
        with _ENCODER_LOCK:
            chunk_scores = (queries @ self.vectors.T).max(axis=0)
        dense_by_record = np.full(self.record_count, -1.0, dtype=np.float32)
        np.maximum.at(dense_by_record, self.owners, chunk_scores)
        dense, new_chunks, new_owners = [], [], []
        for i, record in enumerate(records):
            index = self.record_lookup.get(source_identity(record))
            dense.append(float(dense_by_record[index]) if index is not None else -1.0)
            if index is None:
                parts = windows(self.model, source_text(record))
                new_chunks.extend(t for t, _, _ in parts)
                new_owners.extend([i] * len(parts))
        if new_chunks:
            extra = encode_texts(self.model, new_chunks)
            extra_scores = (queries @ extra.T).max(axis=0)
            for owner, score in zip(new_owners, extra_scores):
                dense[owner] = max(dense[owner], float(score))
        bm25 = self._bm25_score(query, [source_text(row) for row in records])
        bm25_max = max(bm25) if bm25 else 1.0
        normalized = [score / max(bm25_max, 1e-6) for score in bm25]
        hybrid = [0.4 * b + 0.6 * d for b, d in zip(normalized, dense)]
        order = sorted(range(len(records)), key=lambda i: (-hybrid[i], i))
        selected = [i for i in order[:limit] if hybrid[i] >= 0.15]
        evidence = [{**records[i], "rank": rank, "retrieval_scores": {
            "bm25": bm25[i], "dense_cosine": dense[i], "hybrid": hybrid[i]}}
            for rank, i in enumerate(selected, 1)]
        return evidence, {"backend": "skillflow-bm25-bge-hybrid", "semantic_manifest": self.manifest_path,
            "encoder": "BAAI/bge-base-en-v1.5", "lexical_weight": 0.4, "dense_weight": 0.6,
            "query_windows": len(query_parts), "matched_count": len(evidence),
            "current_source_count": len(records), "new_source_windows_encoded": len(new_chunks),
            "similarity_is_not_medical_verification": True}


@lru_cache(maxsize=2)
def load_semantic_index(manifest_path):
    return SemanticEvidenceIndex(str(Path(manifest_path).resolve()))
