#!/usr/bin/env python3
"""Materialize all native HotpotQA Q-A memories with local Qwen3.5-9B.

The source projection and QA-memory admission boundary are reused from the
existing HotpotQA implementation.  Work is bounded by ``concurrency`` and
``checkpoint_every``; every checkpoint is an atomic, source-ordered JSONL, so
an interrupted run resumes without replaying admitted rows.  A separately
versioned dataset-pair fallback can be enabled only explicitly, matching the
bounded fallback used by the TriviaQA full-source materializer.
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
    GENERATION_SCHEMA,
    VERIFICATION_SCHEMA,
    _generate_json,
    _read_jsonl,
    _write_jsonl,
)
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.hotpotqa_full_dataset_qa_memory_index import (  # noqa: E402
    FULL_DATASET_EVALUATION_SCOPE,
    load_hotpotqa_full_dataset_qa_sources,
)
from src.interactive.hotpotqa_qa_memory_index import (  # noqa: E402
    HotpotQATrainQASource,
    materialize_hotpotqa_qa_memories,
)


PROMPT_VERSION = "hotpotqa.full_dataset_qa_paraphrase.qwen35.v1"
PARAPHRASE_VERSION = "hotpotqa-full-dataset-qa-paraphrase-v1"
PARAPHRASE_PROVENANCE = "local-qwen3.5-9b-full-dataset-generate-and-verify-v1"
DATASET_PAIR_FALLBACK_VERSION = "hotpotqa-full-dataset-pair-fallback-v1"
DATASET_PAIR_FALLBACK_PROVENANCE = (
    "deterministic-hotpotqa-dataset-pair-fallback-v1"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _candidate(
    source: HotpotQATrainQASource,
    generated: Mapping[str, object],
) -> dict[str, object]:
    question = generated.get("paraphrase_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("paraphrase_question must be non-empty text")
    answer_statement = generated.get("paraphrase_answer_statement")
    if not isinstance(answer_statement, str) or not answer_statement.strip():
        raise ValueError("paraphrase_answer_statement must be non-empty text")
    canonical = " ".join(source.canonical_answer.casefold().split())
    if canonical not in " ".join(answer_statement.casefold().split()):
        answer_statement = (
            f"The canonical answer is {source.canonical_answer}. "
            f"{answer_statement.strip()}"
        )
    return {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": question.strip(),
        "paraphrase_answer_statement": answer_statement.strip(),
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_version": PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


def _dataset_pair_fallback(
    source: HotpotQATrainQASource,
) -> dict[str, object]:
    """Preserve a paired dataset association with independent provenance."""

    question = " ".join(source.question.split())
    answer = " ".join(source.canonical_answer.split())
    value: dict[str, object] = {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": (
            f"HotpotQA dataset source prompt (verbatim): {question}"
        ),
        "paraphrase_answer_statement": (
            "For this HotpotQA dataset source prompt, the paired response is "
            f"{answer}."
        ),
        "paraphrase_provenance": DATASET_PAIR_FALLBACK_PROVENANCE,
        "paraphrase_version": DATASET_PAIR_FALLBACK_VERSION,
        "semantic_preservation_attested": True,
    }
    materialize_hotpotqa_qa_memories((source,), (value,))
    return value


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
                schema=GENERATION_SCHEMA,
                contract=(
                    "Paraphrase one HotpotQA dataset question without changing its "
                    "scope, entities, relations, constraints, answer slot, or answer. "
                    "Replace wording and sentence structure while preserving every "
                    "multi-hop relation. Produce one declarative answer statement "
                    "that contains the canonical answer span exactly. Return only "
                    "the requested JSON fields; do not add aliases or facts."
                ),
                problem=(
                    f"Original dataset question:\n{source.question}\n\n"
                    f"Canonical dataset answer:\n{source.canonical_answer}"
                ),
                request_id=(
                    f"hotpotqa-full-dataset-qa-memory:{index:06d}:"
                    f"generate:{attempt:02d}"
                ),
                seed=generation_seed,
                temperature=0.2,
            )
            candidate = _candidate(source, generated)
            materialize_hotpotqa_qa_memories((source,), (candidate,))
            verified, verification_receipt = await _generate_json(
                model=model,
                provider=provider,
                schema=VERIFICATION_SCHEMA,
                contract=(
                    "Verify that the HotpotQA paraphrase preserves the exact task "
                    "semantics, entities, requested relations, answer slot, scope, "
                    "and every constraint, while genuinely changing wording. Return "
                    "only the requested JSON fields."
                ),
                problem=(
                    f"Original question:\n{source.question}\n\n"
                    f"Paraphrased question:\n{candidate['paraphrase_question']}"
                ),
                request_id=(
                    f"hotpotqa-full-dataset-qa-memory:{index:06d}:"
                    f"verify:{attempt:02d}"
                ),
                seed=generation_seed + 1,
                temperature=0.0,
            )
            if any(
                verified.get(name) is not True
                for name in (
                    "semantic_preserved",
                    "question_changed",
                    "constraints_preserved",
                )
            ):
                raise ValueError("semantic verifier rejected paraphrase")
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
        f"full-dataset paraphrase failed for {source.source_train_task_id}: "
        f"{errors[-1]}"
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
    bootstrap_by_id: dict[str, dict[str, object]] = {}
    for bootstrap_value in args.bootstrap_materialization:
        bootstrap_path = Path(bootstrap_value).expanduser().resolve()
        if not bootstrap_path.is_file():
            raise FileNotFoundError(
                f"bootstrap materialization is absent: {bootstrap_path}"
            )
        for value in _read_jsonl(bootstrap_path):
            source_id = value.get("source_train_task_id")
            if not isinstance(source_id, str) or source_id not in source_by_id:
                raise ValueError(
                    "bootstrap materialization references a foreign source ID"
                )
            previous = bootstrap_by_id.get(source_id)
            if previous is not None and previous != value:
                raise ValueError(
                    f"bootstrap materializations conflict for {source_id}"
                )
            materialize_hotpotqa_qa_memories(
                (source_by_id[source_id],),
                (value,),
            )
            bootstrap_by_id[source_id] = value
    output_rows = _read_jsonl(output_path)
    resumed = {
        str(value["source_train_task_id"]): value
        for value in output_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(resumed) != len(output_rows):
        raise ValueError("resume materialization has duplicate or foreign source IDs")
    # Preserve the upstream resume precedence: a current output row overrides
    # the same admitted bootstrap source, while every other bootstrap row
    # avoids a repeated model request.
    accepted = {**bootstrap_by_id, **resumed}
    bootstrap_reused_ids = set(bootstrap_by_id).difference(resumed)
    receipt_rows = _read_jsonl(receipt_path)
    receipts = {
        str(value["source_train_task_id"]): value
        for value in receipt_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(receipts) != len(receipt_rows):
        raise ValueError("resume receipts have duplicate or foreign source IDs")
    for source_id, value in accepted.items():
        materialize_hotpotqa_qa_memories((source_by_id[source_id],), (value,))

    pending = [source for source in sources if source.source_train_task_id not in accepted]
    model = None
    provider = None
    if pending and not args.materialize_pending_as_dataset_pair_fallback:
        registry = load_model_registry(Path(args.model_catalog))
        model = registry.require_model(args.model_id)
        provider = registry.provider_for(args.model_id)
        if model.model_id != "qwen3.5-9b-local":
            raise ValueError("full-dataset generator must be local Qwen3.5-9B")

    if args.materialize_pending_as_dataset_pair_fallback:
        for source in pending:
            source_id = source.source_train_task_id
            accepted[source_id] = _dataset_pair_fallback(source)
            receipts[source_id] = {
                "source_train_task_id": source_id,
                "status": "dataset_pair_fallback",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "paraphrase_provenance": DATASET_PAIR_FALLBACK_PROVENANCE,
                "completed_at": _utc_now(),
            }
        pending = []
        _write_jsonl(output_path, [accepted[item] for item in ordered_ids])
        _write_jsonl(receipt_path, [receipts[item] for item in ordered_ids])

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
                if not args.allow_dataset_pair_fallback:
                    failed_source_ids.append(source_id)
                    continue
                accepted[source_id] = _dataset_pair_fallback(source)
                receipts[source_id] = {
                    "source_train_task_id": source_id,
                    "status": "bounded_failure_dataset_pair_fallback",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "paraphrase_provenance": DATASET_PAIR_FALLBACK_PROVENANCE,
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
                f"{len(failed_source_ids)} paraphrases failed; accepted progress "
                f"was preserved ({len(accepted)}/{len(sources)}); first="
                f"{failed_source_ids[0]}"
            )

    ordered = [accepted[source.source_train_task_id] for source in sources]
    materialize_hotpotqa_qa_memories(sources, ordered)
    fallback_count = sum(
        value.get("paraphrase_provenance") == DATASET_PAIR_FALLBACK_PROVENANCE
        for value in ordered
    )
    strict_count = len(ordered) - fallback_count
    train_ids = {
        source.source_train_task_id for source in source_bundle.train
    }
    source_train_count = sum(source.source_train_task_id in train_ids for source in sources)
    manifest: dict[str, object] = {
        "schema_version": (
            "flowsteer.hotpotqa.full_dataset_qa_memory_materialization.v1"
        ),
        "prompt_version": PROMPT_VERSION,
        "paraphrase_version": PARAPHRASE_VERSION,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "dataset_pair_fallback_version": DATASET_PAIR_FALLBACK_VERSION,
        "dataset_pair_fallback_provenance": DATASET_PAIR_FALLBACK_PROVENANCE,
        "source_dataset": "HotpotQA",
        "source_configuration": "distractor",
        "source_splits": ["train", "validation"],
        "source_record_count": len(sources),
        "source_train_count": source_train_count,
        "source_validation_count": len(sources) - source_train_count,
        "unique_source_count": len({source.base_task_id for source in sources}),
        "cycled_record_count": sum(source.cycled for source in sources),
        "paraphrase_count": len(ordered),
        "strict_local_qwen_paraphrase_count": strict_count,
        "dataset_pair_fallback_count": fallback_count,
        "bootstrap_materialization_paths": [
            str(Path(value).expanduser().resolve())
            for value in args.bootstrap_materialization
        ],
        "bootstrap_admitted_count": len(bootstrap_by_id),
        "bootstrap_reused_count": len(bootstrap_reused_ids),
        "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
        "contains_evaluation_answers": True,
        "official_heldout_eligible": False,
        "generator_model_id": args.model_id if strict_count else None,
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
        default="config/model_catalog_hotpotqa_embedding_retrieval_v1.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument(
        "--bootstrap-materialization",
        action="append",
        default=[],
        help=(
            "Existing admitted HotpotQA paraphrase JSONL to reuse by source ID; "
            "repeat the option to supply more than one file."
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_full_dataset_qa_memory_v1/paraphrases.jsonl"
        ),
    )
    parser.add_argument(
        "--receipts",
        default=(
            "artifacts/hotpotqa_full_dataset_qa_memory_v1/"
            "paraphrase_receipts.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "artifacts/hotpotqa_full_dataset_qa_memory_v1/"
            "materialization_manifest.json"
        ),
    )
    parser.add_argument("--train-count", type=_positive_integer, default=90_447)
    parser.add_argument("--validation-count", type=_positive_integer, default=7_405)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--concurrency", type=_positive_integer, default=8)
    parser.add_argument("--checkpoint-every", type=_positive_integer, default=1_024)
    parser.add_argument("--max-attempts", type=_positive_integer, default=3)
    parser.add_argument("--limit", type=_positive_integer)
    parser.add_argument("--allow-dataset-pair-fallback", action="store_true")
    parser.add_argument(
        "--materialize-pending-as-dataset-pair-fallback",
        action="store_true",
        help=(
            "Bypass model generation only for not-yet-materialized rows and use "
            "the separately tagged paired-dataset fallback."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(materialize(args))
    except Exception as exc:
        print(
            f"HotpotQA full-dataset QA-memory materialization failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
