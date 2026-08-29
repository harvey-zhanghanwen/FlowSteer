#!/usr/bin/env python3
"""Materialize verified HotpotQA declarative facts with local Qwen3.5-9B.

Every native question is semantically reworded and paired with one
self-contained declarative fact.  The JSONL is an index-external provenance
sidecar; the index builder projects only ``fact_statement`` into the runtime
corpus.  There is deliberately no verbatim or dataset-pair fallback.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_hotpotqa_qa_memory import (  # noqa: E402
    _generate_json,
    _json_schema,
    _read_jsonl,
    _write_jsonl,
)
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (  # noqa: E402
    FULL_DATASET_EVALUATION_SCOPE,
    canonical_answer_is_declarative_clause,
    load_hotpotqa_full_dataset_qa_sources,
    materialize_hotpotqa_declarative_facts,
    validate_hotpotqa_fact_statement,
    validate_hotpotqa_question_rewrite,
)
from src.interactive.hotpotqa_qa_memory_index import (  # noqa: E402
    HotpotQATrainQASource,
)


PROMPT_VERSION = "hotpotqa.full_dataset_fact.qwen35.field_repair.v2"
PARAPHRASE_VERSION = "hotpotqa-full-dataset-declarative-fact-v2"
PARAPHRASE_PROVENANCE = (
    "local-qwen3.5-9b-semantic-rewrite-and-field-verification-v2"
)
GENERATION_ROUND_SEED_STRIDE = 100_000_000

FACT_GENERATION_SCHEMA = _json_schema(
    {
        "paraphrase_question": {"type": "string", "minLength": 1},
        "fact_statement": {"type": "string", "minLength": 1},
    }
)
QUESTION_REPAIR_SCHEMA = _json_schema(
    {"paraphrase_question": {"type": "string", "minLength": 1}}
)
FACT_REPAIR_SCHEMA = _json_schema(
    {"fact_statement": {"type": "string", "minLength": 1}}
)
QUESTION_VERIFICATION_SCHEMA = _json_schema(
    {
        "semantic_preserved": {"type": "boolean"},
        "question_changed": {"type": "boolean"},
        "constraints_preserved": {"type": "boolean"},
        "answer_slot_preserved": {"type": "boolean"},
    }
)
FACT_VERIFICATION_SCHEMA = _json_schema(
    {
        "fact_declarative": {"type": "boolean"},
        "fact_self_contained": {"type": "boolean"},
        "fact_supported_by_qa": {"type": "boolean"},
        "canonical_span_preserved_when_required": {"type": "boolean"},
        "no_qa_wire_format": {"type": "boolean"},
    }
)
_REQUIRED_QUESTION_VERIFICATION_FIELDS = tuple(
    QUESTION_VERIFICATION_SCHEMA["required"]
)
_REQUIRED_FACT_VERIFICATION_FIELDS = tuple(
    FACT_VERIFICATION_SCHEMA["required"]
)


class FactMaterializationRejected(RuntimeError):
    """Bounded strict-generation rejection with complete attempt receipts."""

    def __init__(
        self,
        source_id: str,
        attempt_receipts: Sequence[Mapping[str, object]],
    ) -> None:
        self.source_id = source_id
        self.attempt_receipts = tuple(dict(item) for item in attempt_receipts)
        last = self.attempt_receipts[-1] if self.attempt_receipts else {}
        detail = str(last.get("error", "strict field verification exhausted"))
        super().__init__(
            f"full-dataset fact materialization failed for {source_id}: {detail}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _generated_text(generated: Mapping[str, object], field: str) -> str:
    value = generated.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return " ".join(value.split())


def _candidate(
    source: HotpotQATrainQASource,
    generated: Mapping[str, object],
) -> dict[str, object]:
    question = validate_hotpotqa_question_rewrite(
        source, generated.get("paraphrase_question")
    )
    normalized_fact = validate_hotpotqa_fact_statement(
        source, generated.get("fact_statement")
    )
    return {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": question.strip(),
        "fact_statement": normalized_fact,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_version": PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


async def _materialize_one(
    source: HotpotQATrainQASource,
    *,
    index: int,
    model: object,
    provider: object,
    seed: int,
    max_attempts: int,
    generation_rounds: int = 2,
) -> tuple[dict[str, object], dict[str, object]]:
    if generation_rounds < 1:
        raise ValueError("generation_rounds must be positive")
    binding_mode = (
        "declarative_clause_paraphrase"
        if canonical_answer_is_declarative_clause(
            source.canonical_answer,
            question=source.question,
        )
        else "answer_slot_binding"
    )
    question: str | None = None
    fact: str | None = None
    question_admitted = False
    fact_admitted = False
    question_rejection = "not yet generated"
    fact_rejection = "not yet generated"
    attempt_receipts: list[dict[str, object]] = []

    for generation_round in range(generation_rounds):
        for attempt in range(max_attempts):
            request_seed = (
                seed
                + index * max_attempts * 8
                + generation_round * GENERATION_ROUND_SEED_STRIDE
                + attempt * 8
            )
            trace: dict[str, object] = {
                "generation_round": generation_round + 1,
                "attempt": attempt + 1,
                "fact_binding_mode": binding_mode,
                "question": {"preserved_from_prior_attempt": question_admitted},
                "fact": {"preserved_from_prior_attempt": fact_admitted},
            }
            question_trace = trace["question"]
            fact_trace = trace["fact"]
            assert isinstance(question_trace, dict)
            assert isinstance(fact_trace, dict)

            if question is None and fact is None:
                try:
                    generated, generation_receipt = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=FACT_GENERATION_SCHEMA,
                        contract=(
                            "Semantically reword the HotpotQA question while "
                            "preserving every entity, relation, scope, constraint, "
                            "multi-hop path, answer slot, name, number, date, and "
                            "quoted span. Replace real wording, not only word order. "
                            + (
                                "The dataset answer is already a declarative clause; "
                                "write a semantically equivalent self-contained "
                                "declarative fact from that clause alone. Do not "
                                "invent a relation between a possibly mismatched "
                                "question and answer. "
                                if binding_mode == "declarative_clause_paraphrase"
                                else
                                "Bind the dataset answer semantics to the original "
                                "question's answer slot in a self-contained "
                                "declarative fact. Preserve proper names, numbers, "
                                "dates, and quoted titles; ordinary phrases may use "
                                "equivalent wording. Express yes/no as the matching "
                                "affirmative/negative proposition. Preserve relation "
                                "direction and scope. "
                            )
                            + "Do not add entities, aliases, facts, or Q-A labels. "
                            "Return only the requested JSON fields."
                        ),
                        problem=(
                            f"Source question:\n{source.question}\n\n"
                            f"Dataset answer:\n{source.canonical_answer}"
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:generate:{attempt:02d}"
                        ),
                        seed=request_seed,
                        temperature=0.1,
                    )
                    question = _generated_text(
                        generated, "paraphrase_question"
                    )
                    fact = _generated_text(generated, "fact_statement")
                    trace["joint_generation_response"] = dict(
                        generation_receipt
                    )
                except Exception as exc:
                    trace["error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    attempt_receipts.append(trace)
                    continue
            else:
                if not question_admitted:
                    try:
                        repaired, response = await _generate_json(
                            model=model,
                            provider=provider,
                            schema=QUESTION_REPAIR_SCHEMA,
                            contract=(
                                "Repair only the HotpotQA question paraphrase. "
                                "Preserve every entity, relation, scope, constraint, "
                                "multi-hop path, answer slot, name, number, date, "
                                "and quoted span. Replace at least one non-entity "
                                "word or phrase; changing only word order is invalid. "
                                "Do not reveal the answer. Return only JSON."
                            ),
                            problem=(
                                f"Source question:\n{source.question}\n\n"
                                f"Rejected paraphrase:\n{question or ''}\n\n"
                                f"Prior admission result:\n{question_rejection}"
                            ),
                            request_id=(
                                f"hotpotqa-full-dataset-fact:{index:06d}:"
                                f"round:{generation_round:02d}:"
                                f"repair-question:{attempt:02d}"
                            ),
                            seed=request_seed,
                            temperature=0.0,
                        )
                        question = _generated_text(
                            repaired, "paraphrase_question"
                        )
                        question_trace["generation_response"] = dict(response)
                    except Exception as exc:
                        question_trace["generation_error"] = (
                            f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                        )
                        question_rejection = str(
                            question_trace["generation_error"]
                        )
                if not fact_admitted:
                    try:
                        repaired, response = await _generate_json(
                            model=model,
                            provider=provider,
                            schema=FACT_REPAIR_SCHEMA,
                            contract=(
                                "Repair only one self-contained declarative fact. "
                                + (
                                    "The dataset answer is a complete declarative "
                                    "clause. Semantically paraphrase only that clause; "
                                    "preserve all names, numbers, dates, quoted spans, "
                                    "relations, and scope. Do not bind it to or infer "
                                    "anything from the question. "
                                    if binding_mode
                                    == "declarative_clause_paraphrase"
                                    else
                                    "Bind the dataset answer semantics to the "
                                    "question's original answer slot. Preserve proper "
                                    "names, numbers, dates, and quoted titles; ordinary "
                                    "phrases may use equivalent wording. Express yes/no "
                                    "as the matching affirmative/negative proposition. "
                                    "Copy the authoritative relation and scope; do not "
                                    "reverse arguments or add a fact. "
                                )
                                + "Do not add entities, aliases, or Q-A labels. "
                                "Return only JSON."
                            ),
                            problem=(
                                (
                                    f"Dataset answer clause:\n"
                                    f"{source.canonical_answer}\n\n"
                                    if binding_mode
                                    == "declarative_clause_paraphrase"
                                    else
                                    f"Source question:\n{source.question}\n\n"
                                    f"Dataset answer:\n{source.canonical_answer}\n\n"
                                )
                                + f"Rejected fact:\n{fact or ''}\n\n"
                                + f"Prior admission result:\n{fact_rejection}"
                            ),
                            request_id=(
                                f"hotpotqa-full-dataset-fact:{index:06d}:"
                                f"round:{generation_round:02d}:"
                                f"repair-fact:{attempt:02d}"
                            ),
                            seed=request_seed + 1,
                            temperature=0.0,
                        )
                        fact = _generated_text(repaired, "fact_statement")
                        fact_trace["generation_response"] = dict(response)
                    except Exception as exc:
                        fact_trace["generation_error"] = (
                            f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                        )
                        fact_rejection = str(fact_trace["generation_error"])

            if not question_admitted and question is not None:
                try:
                    question = validate_hotpotqa_question_rewrite(
                        source, question
                    )
                    question_trace["deterministic_admission"] = True
                    verified, response = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=QUESTION_VERIFICATION_SCHEMA,
                        contract=(
                            "Verify only semantic equivalence of the question "
                            "paraphrase. Preserve entity identity, relation, scope, "
                            "constraints, multi-hop path, answer slot, and answer "
                            "cardinality. Surface wording must genuinely change. "
                            "Do not solve the question. Evaluate each boolean "
                            "independently and return only JSON."
                        ),
                        problem=(
                            f"Source question:\n{source.question}\n\n"
                            f"Paraphrased question:\n{question}"
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:"
                            f"verify-question:{attempt:02d}"
                        ),
                        seed=request_seed + 2,
                        temperature=0.0,
                    )
                    failed = [
                        name
                        for name in _REQUIRED_QUESTION_VERIFICATION_FIELDS
                        if verified.get(name) is not True
                    ]
                    question_trace["verification"] = dict(verified)
                    question_trace["verification_response"] = dict(response)
                    question_trace["failed_fields"] = failed
                    question_admitted = not failed
                    question_rejection = (
                        "accepted"
                        if question_admitted
                        else "semantic verifier rejected: " + ",".join(failed)
                    )
                except Exception as exc:
                    question_admitted = False
                    question_trace["deterministic_or_verification_error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    question_rejection = str(
                        question_trace["deterministic_or_verification_error"]
                    )

            if not fact_admitted and fact is not None:
                try:
                    fact = validate_hotpotqa_fact_statement(source, fact)
                    fact_trace["deterministic_admission"] = True
                    verified, response = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=FACT_VERIFICATION_SCHEMA,
                        contract=(
                            "Verify only the proposed declarative fact. It must "
                            "be self-contained, declarative, free of Q-A labels, "
                            + (
                                "and semantically equivalent to the supplied "
                                "dataset answer clause alone. Do not require or "
                                "invent a relation to the question. "
                                if binding_mode
                                == "declarative_clause_paraphrase"
                                else
                                "and supported by binding the dataset answer semantics "
                                "to the source question's answer slot without changing "
                                "relation direction, polarity, or scope. Proper names, "
                                "numbers, dates, and quoted titles must be preserved; "
                                "ordinary phrases may use equivalent wording. "
                            )
                            + "Reject added entities, aliases, facts, numbers, or "
                            "dates. Evaluate every boolean independently and "
                            "return only JSON."
                        ),
                        problem=(
                            (
                                f"Dataset answer clause:\n"
                                f"{source.canonical_answer}\n\n"
                                if binding_mode
                                == "declarative_clause_paraphrase"
                                else
                                f"Source question:\n{source.question}\n\n"
                                f"Dataset answer:\n{source.canonical_answer}\n\n"
                            )
                            + f"Declarative fact:\n{fact}"
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:"
                            f"verify-fact:{attempt:02d}"
                        ),
                        seed=request_seed + 3,
                        temperature=0.0,
                    )
                    failed = [
                        name
                        for name in _REQUIRED_FACT_VERIFICATION_FIELDS
                        if verified.get(name) is not True
                    ]
                    fact_trace["verification"] = dict(verified)
                    fact_trace["verification_response"] = dict(response)
                    fact_trace["failed_fields"] = failed
                    fact_admitted = not failed
                    fact_rejection = (
                        "accepted"
                        if fact_admitted
                        else "fact verifier rejected: " + ",".join(failed)
                    )
                except Exception as exc:
                    fact_admitted = False
                    fact_trace["deterministic_or_verification_error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    fact_rejection = str(
                        fact_trace["deterministic_or_verification_error"]
                    )

            trace["candidate"] = {
                "paraphrase_question": question,
                "fact_statement": fact,
            }
            trace["error"] = (
                "question="
                + ("accepted" if question_admitted else "rejected")
                + ",fact="
                + ("accepted" if fact_admitted else "rejected")
            )
            attempt_receipts.append(trace)
            if question_admitted and fact_admitted:
                candidate = _candidate(
                    source,
                    {
                        "paraphrase_question": question,
                        "fact_statement": fact,
                    },
                )
                materialize_hotpotqa_declarative_facts(
                    (source,), (candidate,)
                )
                return candidate, {
                    "source_train_task_id": source.source_train_task_id,
                    "status": "accepted",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "generation_round": generation_round + 1,
                    "attempt": attempt + 1,
                    "generation_seed": request_seed,
                    "fact_binding_mode": binding_mode,
                    "attempt_receipts": attempt_receipts,
                    "completed_at": _utc_now(),
                }

    raise FactMaterializationRejected(
        source.source_train_task_id, attempt_receipts
    )


async def materialize(args: argparse.Namespace) -> dict[str, object]:
    source_bundle = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=Path(args.dataset_catalog),
        expected_train_count=args.train_count,
        expected_validation_count=args.validation_count,
    )
    sources = source_bundle.combined
    if args.limit is not None:
        sources = sources[: args.limit]
    source_by_id = {source.source_train_task_id: source for source in sources}
    source_index = {
        source.source_train_task_id: index for index, source in enumerate(sources)
    }
    ordered_ids = [source.source_train_task_id for source in sources]

    output_path = Path(args.output).expanduser().resolve()
    receipt_path = Path(args.receipts).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_rows = _read_jsonl(output_path)
    accepted = {
        str(value["source_train_task_id"]): value
        for value in output_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(accepted) != len(output_rows):
        raise ValueError("resume sidecar has duplicate or foreign source IDs")
    receipt_rows = _read_jsonl(receipt_path)
    receipts = {
        str(value["source_train_task_id"]): value
        for value in receipt_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(receipts) != len(receipt_rows):
        raise ValueError("resume receipts have duplicate or foreign source IDs")
    resume_rejected: list[str] = []
    for source_id, value in list(accepted.items()):
        try:
            materialize_hotpotqa_declarative_facts(
                (source_by_id[source_id],), (value,)
            )
        except (TypeError, ValueError) as exc:
            accepted.pop(source_id)
            previous_status = receipts.get(source_id, {}).get("status")
            receipts[source_id] = {
                "source_train_task_id": source_id,
                "status": "resume_rejected_by_current_admission",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "previous_status": previous_status,
                "admission_error_type": type(exc).__name__,
                "admission_error": " ".join(str(exc).split())[:512],
                "completed_at": _utc_now(),
            }
            resume_rejected.append(source_id)
            continue
        previous = dict(receipts.get(source_id, {}))
        previous.update(
            {
                "source_train_task_id": source_id,
                "status": "resume_revalidated",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "current_deterministic_admission": True,
                "completed_at": _utc_now(),
            }
        )
        receipts[source_id] = previous

    if resume_rejected:
        # Remove stale attestations before any model request.  The successful
        # checkpoint remains source-order stable and the rejected IDs become
        # ordinary pending rows on this same resume invocation.
        _write_jsonl(
            output_path,
            [accepted[item] for item in ordered_ids if item in accepted],
        )
        _write_jsonl(
            receipt_path,
            [receipts[item] for item in ordered_ids if item in receipts],
        )

    pending = [
        source
        for source in sources
        if source.source_train_task_id not in accepted
    ]
    model = None
    provider = None
    if pending:
        registry = load_model_registry(Path(args.model_catalog))
        model = registry.require_model(args.model_id)
        provider = registry.provider_for(args.model_id)
        if model.model_id != "qwen3.5-9b-local":
            raise ValueError("full-dataset generator must be local Qwen3.5-9B")
        # DIRECT_REUSE: TriviaQA's SkillFlow-derived materializer sizes its
        # HTTP worker pool from the requested generation concurrency.  The
        # shared Gateway uses ``asyncio.to_thread``; without this assignment,
        # Python silently caps real request concurrency at its small default
        # executor size even when the materializer semaphore is larger.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(
                max_workers=args.concurrency,
                thread_name_prefix="hotpotqa-fact-materializer",
            )
        )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate(source: HotpotQATrainQASource) -> object:
        assert model is not None and provider is not None
        async with semaphore:
            return await _materialize_one(
                source,
                index=source_index[source.source_train_task_id],
                model=model,
                provider=provider,
                seed=args.seed,
                max_attempts=args.max_attempts,
                generation_rounds=getattr(args, "generation_rounds", 2),
            )

    failed_source_ids: list[str] = []
    checkpoint_size = max(args.checkpoint_every, args.concurrency)
    for offset in range(0, len(pending), checkpoint_size):
        chunk = pending[offset : offset + checkpoint_size]
        results = await asyncio.gather(
            *(generate(source) for source in chunk),
            return_exceptions=True,
        )
        for source, result in zip(chunk, results):
            source_id = source.source_train_task_id
            if isinstance(result, BaseException):
                failed_source_ids.append(source_id)
                rejection_receipt: dict[str, object] = {
                    "source_train_task_id": source_id,
                    "status": "rejected_after_bounded_attempts",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "generation_error_type": type(result).__name__,
                    "generation_error": " ".join(str(result).split())[:512],
                    "completed_at": _utc_now(),
                }
                if isinstance(result, FactMaterializationRejected):
                    rejection_receipt["attempt_receipts"] = list(
                        result.attempt_receipts
                    )
                receipts[source_id] = rejection_receipt
                continue
            candidate, receipt = result
            accepted[source_id] = candidate
            receipts[source_id] = receipt
        _write_jsonl(
            output_path,
            [accepted[item] for item in ordered_ids if item in accepted],
        )
        _write_jsonl(
            receipt_path,
            [receipts[item] for item in ordered_ids if item in receipts],
        )

    if failed_source_ids:
        raise RuntimeError(
            f"{len(failed_source_ids)} fact materializations failed; accepted "
            f"progress was preserved ({len(accepted)}/{len(sources)}); first="
            f"{failed_source_ids[0]}; rerun with a different --seed to retry "
            "only rejected sources"
        )

    ordered = [accepted[source.source_train_task_id] for source in sources]
    materialize_hotpotqa_declarative_facts(sources, ordered)
    train_ids = {
        source.source_train_task_id for source in source_bundle.train
    }
    source_train_count = sum(
        source.source_train_task_id in train_ids for source in sources
    )
    manifest: dict[str, object] = {
        "schema_version": (
            "flowsteer.hotpotqa.full_dataset_fact_materialization.v1"
        ),
        "prompt_version": PROMPT_VERSION,
        "paraphrase_version": PARAPHRASE_VERSION,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_versions": sorted(
            {str(value["paraphrase_version"]) for value in ordered}
        ),
        "paraphrase_provenances": sorted(
            {str(value["paraphrase_provenance"]) for value in ordered}
        ),
        "source_dataset": "HotpotQA",
        "source_configuration": "distractor",
        "source_splits": ["train", "validation"],
        "source_record_count": len(sources),
        "source_train_count": source_train_count,
        "source_validation_count": len(sources) - source_train_count,
        "unique_source_count": len({source.base_task_id for source in sources}),
        "cycled_record_count": sum(source.cycled for source in sources),
        "question_rewrite_count": len(ordered),
        "fact_count": len(ordered),
        "semantic_rewrite_coverage": 1.0,
        "fallback_count": 0,
        "index_external_metadata_fields": [
            "source_train_task_id",
            "paraphrase_question",
            "paraphrase_provenance",
            "paraphrase_version",
        ],
        "indexed_text_field": "fact_statement",
        "document_format": "declarative_fact_only",
        "contains_raw_questions_in_fact_records": False,
        "contains_raw_answers_in_fact_records": False,
        "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
        "contains_evaluation_source_facts": True,
        "official_heldout_eligible": False,
        "generator_model_id": args.model_id,
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "generation_rounds": getattr(args, "generation_rounds", 2),
        "generation_concurrency": args.concurrency,
        "http_executor_workers": args.concurrency,
        "checkpoint_every": checkpoint_size,
        "accepted_count": len(ordered),
        "rejected_count": 0,
        "materialization_path": str(output_path),
        "generation_receipts_path": str(receipt_path),
        "completed_at": _utc_now(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-catalog", default="config/datasets_agentgraph.yaml")
    parser.add_argument(
        "--model-catalog",
        default="config/model_catalog_hotpotqa_qa_memory_v10.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument(
        "--output",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "fact_provenance_sidecar.jsonl"
        ),
    )
    parser.add_argument(
        "--receipts",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "generation_receipts.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "materialization_manifest.json"
        ),
    )
    parser.add_argument("--train-count", type=_positive_integer, default=90_447)
    parser.add_argument("--validation-count", type=_positive_integer, default=7_405)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--concurrency", type=_positive_integer, default=64)
    parser.add_argument("--checkpoint-every", type=_positive_integer, default=1_024)
    parser.add_argument("--max-attempts", type=_positive_integer, default=4)
    parser.add_argument(
        "--generation-rounds", type=_positive_integer, default=2
    )
    parser.add_argument("--limit", type=_positive_integer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(materialize(args))
    except Exception as exc:
        print(
            f"HotpotQA full-dataset fact materialization failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
