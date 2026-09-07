"""Request-scoped, source-separated HealthBench retrieval over real receipts.

DIRECT_REUSE_SKILLFLOW: ``DocumentPassage``, ``build_retrieval_index`` and
``RetrievalIndex.search/read`` own the SQLite FTS5 schema and ranking.
NECESSARY_PROJECT_ADAPTATION: follow ``open_provided_context_qa_tool_registry``
and reuse its ``_ThreadAffineRetrievalWorker`` for SQLite connection affinity;
separate the supplied conversation from externally retrieved medical evidence,
and retain public source metadata outside the upstream passage schema.

This is not a clinical answer database. Callers provide only the current
conversation and actual Tool evidence available to this AgentRequest, never a
TaskRecord, benchmark rubric, model-generated answer, or another node's state.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from threading import RLock
from typing import Any, Mapping, Sequence

from .qa_retrieval import DEFAULT_SKILLFLOW_SOURCE, _load_retrieval_module, build_keyword_query, _QUERY_TOKEN
from .qa_tool_adapter import _ThreadAffineRetrievalWorker


DATABASES = ("conversation", "medical_references", "drug_labels")


def load_frozen_public_evidence(manifest_path: str | Path) -> tuple[dict[str, object], ...]:
    """Load published evidence from an explicitly selected, frozen library.

    No task-to-answer lookup. Existing ingest_evidence owns source admission
    and metadata projection; query construction receipts are not loaded.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "flowsteer.healthbench.public-question-evidence-corpus.v1" or manifest.get("status") != "frozen":
        raise ValueError("incompatible frozen public evidence corpus")
    records = []
    for name, receipt in manifest["databases"].items():
        if name not in ("medical_references", "drug_labels"):
            raise ValueError("shared corpus may not contain conversations or benchmark answers")
        with Path(receipt["records_path"]).open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    return tuple(records)
_SOURCE_DATABASE = {
    "MedRAG/textbooks": "medical_references",
    "NCBI PubMed": "medical_references",
    "Europe PMC": "medical_references",
    "ClinicalTrials.gov": "medical_references",
    "NCBI Bookshelf": "medical_references",
    "NLM MeSH": "medical_references",
    "NLM DailyMed": "drug_labels",
}
_PUBLIC_EVIDENCE_FIELDS = frozenset({
    "source_type", "source", "source_id", "document_id", "title", "excerpt",
    "date", "url", "content_type", "version", "offset", "truncated",
    "next_offset", "published_date", "total_characters",
    "pmid", "pmcid", "doi", "full_text_source_id", "publication_types",
    "is_open_access", "is_preprint",
    "matched_title", "book_title", "full_text_availability", "text_scope",
    "collection", "repository_datestamp",
})


class HealthBenchKnowledgeStore:
    """Append real observations and query immutable, per-request FTS5 indices.

    A unique child directory is allocated even when callers reuse a parent
    directory. Existing stores are never implicitly loaded or shared. External
    retrieval remains the source of evidence; this store only indexes content
    already returned to this request or explicitly supplied by its upstream
    communication receipts.
    """

    def __init__(
        self,
        directory: Path,
        conversation: Sequence[Mapping[str, str]],
        skillflow_source: Path = DEFAULT_SKILLFLOW_SOURCE,
        semantic_index=None,
        metadata_aware_retrieval: bool = False,
    ) -> None:
        if isinstance(conversation, (str, bytes, Mapping)):
            raise TypeError("conversation must be a sequence of role/content messages")
        turns: list[dict[str, object]] = []
        for index, turn in enumerate(conversation):
            if not isinstance(turn, Mapping):
                raise TypeError("conversation messages must be mappings")
            role, content = turn.get("role"), turn.get("content")
            if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
                raise ValueError("conversation messages require a role and text content")
            # Complete original content, without claims of external authority.
            turns.append({"role": role, "content": content, "turn_index": index})
        self._module = _load_retrieval_module(Path(skillflow_source))
        self._semantic_index = semantic_index
        if type(metadata_aware_retrieval) is not bool:
            raise TypeError("metadata_aware_retrieval must be boolean")
        self._metadata_aware_retrieval = metadata_aware_retrieval
        parent = Path(directory)
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(mkdtemp(prefix="request-", dir=parent))
        self._lock = RLock()
        self._closed = False
        self._records: dict[str, list[dict[str, object]]] = {
            name: [] for name in DATABASES
        }
        self._seen: dict[str, set[str]] = {name: set() for name in DATABASES}
        self._workers: dict[str, _ThreadAffineRetrievalWorker] = {}
        self._indexed_counts: dict[str, int] = {name: 0 for name in DATABASES}
        self._revisions: dict[str, int] = {name: 0 for name in DATABASES}
        self._receipts: dict[str, dict[str, object]] = {}
        for name in DATABASES:
            (self.directory / name).mkdir()
        for turn in turns:
            self._append("conversation", turn)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("HealthBench knowledge store is closed")

    def _append(self, database: str, record: dict[str, object]) -> None:
        passage_id = f"{database}-{len(self._records[database]) + 1:06d}"
        value = {**record, "passage_id": passage_id}
        # Persist original public records even when no later search is needed.
        with (self.directory / database / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self._records[database].append(value)

    def ingest_evidence(self, evidence: Mapping[str, object]) -> bool:
        """Admit one real source record; exact repeated observations are no-ops.

        Version, excerpt and page metadata participate in identity so new label
        versions and subsequent source pages never overwrite earlier receipts.
        Unknown and nested fields are not copied into the retrieval corpus.
        """
        if not isinstance(evidence, Mapping):
            return False
        source = evidence.get("source")
        database = _SOURCE_DATABASE.get(source) if isinstance(source, str) else None
        if database is None:
            return False
        if any(
            not isinstance(evidence.get(field), str) or not str(evidence[field]).strip()
            for field in ("document_id", "excerpt")
        ):
            return False
        record = {
            key: value for key, value in evidence.items()
            if key in _PUBLIC_EVIDENCE_FIELDS and (
                value is None or type(value) in (str, int, bool)
                or (key == "publication_types" and isinstance(value, list)
                    and all(isinstance(item, str) for item in value))
            )
        }
        identity = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._require_open()
            if identity in self._seen[database]:
                return False
            self._append(database, record)
            self._seen[database].add(identity)
        return True

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {name: len(records) for name, records in self._records.items()}

    def _build(self, database: str) -> None:
        records = self._records[database]
        if not records or self._indexed_counts[database] == len(records):
            return
        revision = self._revisions[database] + 1
        version = f"request-scoped-v1-revision-{revision:06d}"
        index_path = self.directory / database / f"revision-{revision:06d}.sqlite3"
        metadata_path = index_path.with_suffix(".metadata.json")
        passages = []
        for record in records:
            if database == "conversation":
                document_id = f"conversation-turn-{record['turn_index']}"
                title = f"Conversation turn {record['turn_index']} ({record['role']})"
                text = f"role: {record['role']}\nturn_index: {record['turn_index']}\n{record['content']}"
            else:
                document_id = str(record["document_id"])
                title = str(record.get("title") or document_id)
                text = str(record["excerpt"])
                if self._metadata_aware_retrieval:
                    # SkillFlow FTS5 already weights title 5:1 over body. Index
                    # observed identifiers there; return the original title.
                    title += " " + " ".join(str(record.get(key) or "") for key in (
                        "pmid", "pmcid", "doi", "source_id", "document_id",
                    ))
            passages.append(self._module.DocumentPassage(
                passage_id=str(record["passage_id"]),
                document_id=self._module.normalize_json(document_id),
                title=self._module.normalize_json(title),
                text=self._module.normalize_json(text),
            ))
        # Upstream builder atomically publishes a NEW path; never replace an
        # index already opened with SkillFlow's read-only immutable connection.
        self._module.build_retrieval_index(
            index_path, tuple(passages),
            corpus_name=f"healthbench-request-{database}", corpus_version=version,
        )
        self._revisions[database] = revision
        receipt: dict[str, object] = {
            "database": database,
            "source": "supplied_conversation" if database == "conversation" else "observed_tool_evidence",
            "index_path": str(index_path), "metadata_path": str(metadata_path),
            "records_path": str(self.directory / database / "records.jsonl"),
            "corpus_version": version, "passage_count": len(passages),
            "retrieval_backend": "sqlite-fts5-lexical",
        }
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump({"index_receipt": receipt, "records": records}, stream,
                      ensure_ascii=False, allow_nan=False)
        worker = _ThreadAffineRetrievalWorker(self._module.RetrievalIndex, index_path)
        previous = self._workers.get(database)
        self._workers[database] = worker
        self._receipts[database] = receipt
        self._indexed_counts[database] = len(records)
        if previous is not None:
            previous.close()

    def search(self, database: str, query: str, limit: int = 3) -> dict[str, object]:
        """Return full indexed passages with their original public provenance."""
        if database not in DATABASES:
            raise ValueError(f"database must be one of {DATABASES}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            self._require_open()
            result: dict[str, Any] = {
                "database": database, "query": query, "status": "empty",
                "evidence": [], "conversation_matches": [],
                "index_receipt": None, "record_count": len(self._records[database]),
            }
            if not self._records[database]:
                return result
            metadata_hits = []
            if self._metadata_aware_retrieval and database != "conversation":
                metadata_hits = self._metadata_matches(database, query, limit)
            if self._semantic_index is not None and database != "conversation":
                evidence, receipt = self._semantic_index.rank(query, self._records[database], limit)
                if self._metadata_aware_retrieval:
                    evidence = self._merge_metadata_matches(metadata_hits, evidence, limit)
                    receipt = {**receipt, "metadata_lookup": {
                        "backend": "skillflow-fts5-title-weight-5-body-weight-1",
                        "metadata_match_count": len(metadata_hits),
                        "clinical_correctness_verified": False,
                        "corpus_is_exhaustive": False,
                    }}
                result.update(evidence=evidence, index_receipt=receipt,
                              status="ok" if evidence else "no_matches")
                return result
            self._build(database)
            worker = self._workers[database]
            # The existing worker's public interface is asynchronous. This
            # synchronous Tool backend uses its SAME owning executor/index,
            # rather than asyncio.run inside an already-running event loop.
            hits = worker._executor.submit(
                lambda: worker._index.search(query, limit=limit)
            ).result()
            public_records = {
                record["passage_id"]: record for record in self._records[database]
            }
            for hit in hits:
                passage = worker._executor.submit(worker._index.read, hit.passage_id).result()
                record = {**public_records[hit.passage_id], "rank": hit.rank}
                if database == "conversation":
                    result["conversation_matches"].append(record)
                else:
                    # Upstream indices require NFC; the source record retains
                    # the original excerpt and provenance without rewriting it.
                    result["evidence"].append(record)
            result["status"] = "ok" if hits else "no_matches"
            result["index_receipt"] = dict(self._receipts[database])
            if self._metadata_aware_retrieval and database != "conversation":
                result["evidence"] = self._merge_metadata_matches(metadata_hits, result["evidence"], limit)
                result["index_receipt"]["metadata_match_count"] = len(metadata_hits)
                result["index_receipt"]["clinical_correctness_verified"] = False
            return result

    def _metadata_matches(self, database: str, query: str, limit: int):
        """Exact public metadata candidate match; not clinical verification.

        Reuse SkillFlow title-weighted FTS5 and QA keyword normalization. The
        default hybrid remains the fallback, with unchanged weights/threshold.
        No study aliases, task IDs, evaluator vocabulary or learned ranker.
        """
        terms = {token.casefold() for token in _QUERY_TOKEN.findall(build_keyword_query(query, max_terms=24))}
        if not terms:
            return []
        self._build(database)
        worker = self._workers[database]
        hits = worker._executor.submit(
            lambda: worker._index.search(query, limit=len(self._records[database]))
        ).result()
        records = {row["passage_id"]: row for row in self._records[database]}
        matches = []
        for hit in hits:
            row = records[hit.passage_id]
            identifiers = [str(row.get(key) or "").casefold() for key in ("pmid", "pmcid", "doi", "source_id", "document_id")]
            metadata = " ".join([str(row.get("title") or ""), *identifiers])
            metadata_terms = {token.casefold() for token in _QUERY_TOKEN.findall(metadata)}
            identifier_match = query.strip().casefold() in identifiers
            if not (identifier_match or (len(terms) >= 2 and terms <= metadata_terms)):
                continue
            matches.append({**row, "retrieval_match": {
                "type": "exact_identifier" if identifier_match else "all_query_terms_in_metadata",
                "matched_terms": sorted(terms), "clinical_correctness_verified": False,
            }})
            if len(matches) == limit:
                break
        return matches

    @staticmethod
    def _merge_metadata_matches(metadata, fallback, limit):
        result, seen = [], set()
        for row in [*metadata, *fallback]:
            identity = row["passage_id"]
            if identity in seen:
                continue
            seen.add(identity)
            result.append({**row, "rank": len(result) + 1})
            if len(result) == limit:
                break
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()
            self._closed = True


__all__ = ["DATABASES", "HealthBenchKnowledgeStore"]
