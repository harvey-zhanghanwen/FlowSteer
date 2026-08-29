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
    load_hotpotqa_full_dataset_qa_sources,
    materialize_hotpotqa_declarative_facts,
)
from src.interactive.hotpotqa_qa_memory_index import (  # noqa: E402
    HotpotQATrainQASource,
)


PROMPT_VERSION = "hotpotqa.full_dataset_fact.qwen35.generate_verify.v1"
PARAPHRASE_VERSION = "hotpotqa-full-dataset-declarative-fact-v1"
PARAPHRASE_PROVENANCE = (
    "local-qwen3.5-9b-semantic-rewrite-and-fact-verification-v1"
)

FACT_GENERATION_SCHEMA = _json_schema(
    {
        "paraphrase_question": {"type": "string", "minLength": 1},
        "fact_statement": {"type": "string", "minLength": 1},
    }
)
FACT_VERIFICATION_SCHEMA = _json_schema(
    {
        "semantic_preserved": {"type": "boolean"},
        "question_changed": {"type": "boolean"},
        "constraints_preserved": {"type": "boolean"},
        "answer_slot_preserved": {"type": "boolean"},
        "fact_declarative": {"type": "boolean"},
        "fact_self_contained": {"type": "boolean"},
        "fact_supported_by_qa": {"type": "boolean"},
        "canonical_span_preserved_when_required": {"type": "boolean"},
        "no_qa_wire_format": {"type": "boolean"},
    }
)
_REQUIRED_VERIFICATION_FIELDS = tuple(FACT_VERIFICATION_SCHEMA["required"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _candidate(
    source: HotpotQATrainQASource,
    generated: Mapping[str, object],
) -> dict[str, object]:
    question = generated.get("paraphrase_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("paraphrase_question must be non-empty text")
    if _normalized_text(question) == _normalized_text(source.question):
        raise ValueError("paraphrase_question must change the source wording")
    fact = generated.get("fact_statement")
    if not isinstance(fact, str) or not fact.strip():
        raise ValueError("fact_statement must be non-empty text")
    normalized_fact = fact.strip()
    lowered = normalized_fact.casefold()
    if lowered.startswith("question:") or lowered.startswith("answer:"):
        raise ValueError("fact_statement cannot use Question/Answer labels")
    if "\nquestion:" in lowered or "\nanswer:" in lowered:
        raise ValueError("fact_statement cannot contain a Q-A wire format")
    if _normalized_text(normalized_fact) == _normalized_text(
        source.canonical_answer
    ):
        raise ValueError("fact_statement must be self-contained, not a bare answer")
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
) -> tuple[dict[str, object], dict[str, object]]:
    errors: list[str] = []
    for attempt in range(max_attempts):
        generation_seed = seed + index * max_attempts * 2 + attempt * 2
        try:
            generated, generation_receipt = await _generate_json(
                model=model,
                provider=provider,
                schema=FACT_GENERATION_SCHEMA,
                contract=(
                    "Semantically reword one HotpotQA question while preserving "
                    "its scope, entities, relations, constraints, multi-hop path, "
                    "and answer slot. Also express the supplied dataset answer as "
                    "one self-contained declarative fact that states the complete "
                    "question-answer relation without Question/Answer labels. "
                    "Preserve names, proper nouns, numbers, and dates exactly when "
                    "they cannot be safely paraphrased. Do not add aliases or facts. "
                    "Return only the requested JSON fields."
                ),
                problem=(
                    f"Source question:\n{source.question}\n\n"
                    f"Dataset answer:\n{source.canonical_answer}"
                ),
                request_id=(
                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                    f"generate:{attempt:02d}"
                ),
                seed=generation_seed,
                temperature=0.2,
            )
            candidate = _candidate(source, generated)
            verified, verification_receipt = await _generate_json(
                model=model,
                provider=provider,
                schema=FACT_VERIFICATION_SCHEMA,
                contract=(
                    "Verify the proposed semantic rewrite and declarative fact "
                    "against only the supplied source question and dataset answer. "
                    "The rewrite must genuinely change surface wording without "
                    "changing semantics, scope, constraints, relations, multi-hop "
                    "path, or answer slot. The fact must be declarative, "
                    "self-contained, supported by the pair, and free of Q-A labels. "
                    "Canonical spans that cannot be safely paraphrased must remain. "
                    "Evaluate every boolean independently and return only JSON."
                ),
                problem=(
                    f"Source question:\n{source.question}\n\n"
                    f"Dataset answer:\n{source.canonical_answer}\n\n"
                    f"Paraphrased question:\n{candidate['paraphrase_question']}\n\n"
                    f"Declarative fact:\n{candidate['fact_statement']}"
                ),
                request_id=(
                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                    f"verify:{attempt:02d}"
                ),
                seed=generation_seed + 1,
                temperature=0.0,
            )
            if any(
                verified.get(name) is not True
                for name in _REQUIRED_VERIFICATION_FIELDS
            ):
                failed = [
                    name
                    for name in _REQUIRED_VERIFICATION_FIELDS
                    if verified.get(name) is not True
                ]
                raise ValueError(
                    "semantic/fact verifier rejected: " + ",".join(failed)
                )
            materialize_hotpotqa_declarative_facts((source,), (candidate,))
            return candidate, {
                "source_train_task_id": source.source_train_task_id,
                "status": "accepted",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "attempt": attempt + 1,
                "generation_seed": generation_seed,
                "verification_seed": generation_seed + 1,
                "generation_response": dict(generation_receipt),
                "verification_response": dict(verification_receipt),
                "verification": verified,
                "completed_at": _utc_now(),
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {' '.join(str(exc).split())}")
    raise RuntimeError(
        f"full-dataset fact materialization failed for "
        f"{source.source_train_task_id}: {errors[-1]}"
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
    for source_id, value in accepted.items():
        materialize_hotpotqa_declarative_facts(
            (source_by_id[source_id],), (value,)
        )
        receipts.setdefault(
            source_id,
            {
                "source_train_task_id": source_id,
                "status": "resume_admitted",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "paraphrase_provenance": value.get("paraphrase_provenance"),
                "completed_at": _utc_now(),
            },
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
                receipts[source_id] = {
                    "source_train_task_id": source_id,
                    "status": "rejected_after_bounded_attempts",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "generation_error_type": type(result).__name__,
                    "generation_error": " ".join(str(result).split())[:512],
                    "completed_at": _utc_now(),
                }
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
        "generation_concurrency": args.concurrency,
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
