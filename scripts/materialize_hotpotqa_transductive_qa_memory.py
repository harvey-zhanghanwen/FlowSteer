#!/usr/bin/env python3
"""Materialize the explicit HotpotQA 512+128 transductive QA paraphrases.

The generation gateway, JSON schemas, resume-safe JSONL persistence and
semantic verification are reused from ``materialize_hotpotqa_qa_memory``.
This entry point changes only the source projection and provenance: it sees
the frozen 512 training QA plus the frozen 128 evaluation QA, so every output
is permanently labelled as a transductive retrieval diagnostic.
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

from scripts.materialize_hotpotqa_qa_memory import (
    GENERATION_SCHEMA,
    VERIFICATION_SCHEMA,
    _generate_json,
    _read_jsonl,
    _write_jsonl,
)
from src.interactive.config_loader import load_model_registry
from src.interactive.hotpotqa_qa_memory_index import (
    HotpotQATrainQASource,
    materialize_hotpotqa_qa_memories,
)
from src.interactive.hotpotqa_transductive_qa_memory_index import (
    TRANSDUCTIVE_EVALUATION_REGIME,
    load_hotpotqa_transductive_qa_sources,
)


PROMPT_VERSION = "hotpotqa.transductive_qa_paraphrase.qwen35.generate_verify.v2"
PARAPHRASE_VERSION = "hotpotqa-transductive-qa-paraphrase-v2"
PARAPHRASE_PROVENANCE = "local-qwen3.5-9b-transductive-generate-and-verify-v2"


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
        # This is the legacy v1 memory wire key used by the reused
        # HotpotQAQAMemory record for both source splits.
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": question.strip(),
        "paraphrase_answer_statement": answer_statement.strip(),
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_version": PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


def _bootstrap_candidates(path: Path) -> list[dict[str, object]]:
    """Project already-validated train memories into paraphrase records."""

    candidates: list[dict[str, object]] = []
    for value in _read_jsonl(path):
        candidates.append(
            {
                "source_train_task_id": value["source_train_task_id"],
                "paraphrase_question": value["paraphrase_question"],
                "paraphrase_answer_statement": value[
                    "paraphrase_answer_statement"
                ],
                "paraphrase_provenance": value["paraphrase_provenance"],
                "paraphrase_version": value["paraphrase_version"],
                "semantic_preservation_attested": True,
            }
        )
    return candidates


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
                    "Paraphrase one HotpotQA question without changing its scope, "
                    "entities, relations, constraints, or answer. Change wording and "
                    "sentence structure. Preserve the interrogative answer-slot "
                    "category exactly (for example, Who must remain Who; do not "
                    "replace it with a guessed entity class). When the question is "
                    "short or lexically ambiguous, prefer a minimal synonym or "
                    "voice change over adding an interpretation. Produce a "
                    "declarative answer statement that "
                    "contains the canonical answer span exactly. Return only the "
                    "requested JSON fields and invent no aliases. This source belongs "
                    "to an explicitly transductive retrieval diagnostic; do not "
                    "reinterpret or correct the supplied dataset answer."
                ),
                problem=(
                    f"Original dataset question:\n{source.question}\n\n"
                    f"Canonical dataset answer:\n{source.canonical_answer}"
                ),
                request_id=(
                    f"hotpotqa-transductive-qa-memory:{index:04d}:"
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
                    "Verify whether a HotpotQA question paraphrase preserves the exact "
                    "task semantics, all entities, relations, and constraints; whether "
                    "it is genuinely reworded; and whether every original constraint "
                    "remains. Return only the requested JSON fields."
                ),
                problem=(
                    f"Original question:\n{source.question}\n\n"
                    f"Paraphrased question:\n{candidate['paraphrase_question']}"
                ),
                request_id=(
                    f"hotpotqa-transductive-qa-memory:{index:04d}:"
                    f"verify:{attempt:02d}"
                ),
                seed=generation_seed + 1,
                temperature=0.0,
            )
            required_true = (
                "semantic_preserved",
                "question_changed",
                "constraints_preserved",
            )
            if any(verified.get(name) is not True for name in required_true):
                raise ValueError(
                    "semantic verifier rejected paraphrase: "
                    f"verification={verified!r}, "
                    f"question={candidate['paraphrase_question']!r}"
                )
            return candidate, {
                "source_train_task_id": source.source_train_task_id,
                "status": "accepted",
                "evaluation_regime": TRANSDUCTIVE_EVALUATION_REGIME,
                "contains_evaluation_answers": True,
                "official_heldout_eligible": False,
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
        f"transductive paraphrase failed for {source.source_train_task_id}: "
        f"{errors[-1]}"
    )


async def materialize(args: argparse.Namespace) -> dict[str, object]:
    train_path = Path(args.train_path).expanduser().resolve()
    evaluation_path = Path(args.validation_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    receipt_path = Path(args.receipts).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    source_bundle = load_hotpotqa_transductive_qa_sources(
        train_jsonl=train_path,
        evaluation_jsonl=evaluation_path,
        expected_train_count=args.train_count,
        expected_evaluation_count=args.validation_count,
    )
    sources = source_bundle.combined
    if args.limit is not None:
        sources = sources[: args.limit]

    registry = load_model_registry(Path(args.model_catalog))
    model = registry.require_model(args.model_id)
    provider = registry.provider_for(args.model_id)
    if model.model_id != "qwen3.5-9b-local":
        raise ValueError("transductive paraphrase generator must be local Qwen3.5-9B")

    source_by_id = {source.source_train_task_id: source for source in sources}
    bootstrap_path = Path(args.bootstrap_materialization).expanduser().resolve()
    bootstrap = {
        str(value["source_train_task_id"]): value
        for value in _bootstrap_candidates(bootstrap_path)
        if value.get("source_train_task_id") in source_by_id
    }
    accepted = {
        str(value["source_train_task_id"]): value
        for value in _read_jsonl(output_path)
        if value.get("source_train_task_id") in source_by_id
    }
    accepted = {**bootstrap, **accepted}
    receipts = {
        str(value["source_train_task_id"]): value
        for value in _read_jsonl(receipt_path)
        if value.get("source_train_task_id") in source_by_id
    }
    for source_id, value in accepted.items():
        materialize_hotpotqa_qa_memories((source_by_id[source_id],), (value,))

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def run(index: int, source: HotpotQATrainQASource) -> None:
        if source.source_train_task_id in accepted:
            return
        async with semaphore:
            candidate, receipt = await _materialize_one(
                source,
                index=index,
                model=model,
                provider=provider,
                seed=args.seed,
                max_attempts=args.max_attempts,
            )
        async with lock:
            accepted[source.source_train_task_id] = candidate
            receipts[source.source_train_task_id] = receipt
            ordered_ids = [item.source_train_task_id for item in sources]
            _write_jsonl(
                output_path,
                [accepted[item] for item in ordered_ids if item in accepted],
            )
            _write_jsonl(
                receipt_path,
                [receipts[item] for item in ordered_ids if item in receipts],
            )

    results = await asyncio.gather(
        *(run(index, source) for index, source in enumerate(sources)),
        return_exceptions=True,
    )
    failures = [value for value in results if isinstance(value, BaseException)]
    if failures:
        details = "; ".join(
            f"{type(value).__name__}: {' '.join(str(value).split())}"
            for value in failures[:3]
        )
        raise RuntimeError(
            f"{len(failures)} transductive QA paraphrases failed; accepted progress "
            f"was preserved ({len(accepted)}/{len(sources)}); {details}"
        )
    ordered = [accepted[source.source_train_task_id] for source in sources]
    materialize_hotpotqa_qa_memories(sources, ordered)
    paraphrase_versions = sorted(
        {str(value["paraphrase_version"]) for value in ordered}
    )
    paraphrase_provenances = sorted(
        {str(value["paraphrase_provenance"]) for value in ordered}
    )
    manifest = {
        "schema_version": (
            "flowsteer.hotpotqa.transductive_qa_memory_paraphrase_manifest.v1"
        ),
        "prompt_version": PROMPT_VERSION,
        "paraphrase_version": PARAPHRASE_VERSION,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_versions": paraphrase_versions,
        "paraphrase_provenances": paraphrase_provenances,
        "bootstrap_materialization_path": str(bootstrap_path),
        "bootstrap_count": len(bootstrap),
        "generator_model_id": model.model_id,
        "provider_id": provider.provider_id,
        "provider_model": model.model_name,
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "source_record_count": len(sources),
        "source_train_count": min(len(source_bundle.train), len(sources)),
        "source_evaluation_count": max(0, len(sources) - len(source_bundle.train)),
        "frozen_validation_count": args.validation_count,
        "evaluation_overlap_count": max(
            0, len(sources) - len(source_bundle.train)
        ),
        "contains_evaluation_answers": True,
        "evaluation_regime": TRANSDUCTIVE_EVALUATION_REGIME,
        "official_heldout_eligible": False,
        "accepted_count": len(ordered),
        "rejected_count": 0,
        "materialization_path": str(output_path),
        "generation_receipts_path": str(receipt_path),
        "completed_at": _utc_now(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default=(
            "artifacts/hotpotqa_qa_memory_source_freeze_v2/aligned/train.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-path",
        default=(
            "artifacts/hotpotqa_qa_memory_source_freeze_v2/aligned/validation.jsonl"
        ),
    )
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument(
        "--model-catalog",
        default="config/model_catalog_hotpotqa_embedding_retrieval_v1.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument(
        "--bootstrap-materialization",
        default=(
            "artifacts/hotpotqa_qa_memory_source_freeze_v2/index/memories.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotpotqa_transductive_qa_memory_v1/paraphrases.jsonl",
    )
    parser.add_argument(
        "--receipts",
        default=(
            "artifacts/hotpotqa_transductive_qa_memory_v1/"
            "paraphrase_receipts.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "artifacts/hotpotqa_transductive_qa_memory_v1/"
            "paraphrase_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if (
        args.train_count < 1
        or args.validation_count < 1
        or args.concurrency < 1
        or args.max_attempts < 1
    ):
        parser.error("counts, concurrency, and max-attempts must be positive")
    try:
        value = asyncio.run(materialize(args))
    except Exception as exc:
        print(
            f"HotpotQA transductive QA-memory materialization failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
