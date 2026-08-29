from __future__ import annotations

import asyncio
from argparse import Namespace
import json
from pathlib import Path
from typing import Mapping

import pytest

from scripts import materialize_hotpotqa_full_dataset_fact_memory as materializer
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    HotpotQAFullDatasetQASources,
)
from src.interactive.hotpotqa_qa_memory_index import HotpotQATrainQASource


def _source(
    index: int,
    *,
    question: str = "Who authored Atlas?",
    answer: str = "Ada Lovelace",
) -> HotpotQATrainQASource:
    task_id = f"hotpotqa:field-repair-{index}"
    return HotpotQATrainQASource(
        source_train_task_id=task_id,
        base_task_id=task_id,
        cycled=False,
        question=question,
        canonical_answer=answer,
    )


def _question_verification(**overrides: bool) -> dict[str, bool]:
    value = {
        name: True for name in materializer._REQUIRED_QUESTION_VERIFICATION_FIELDS
    }
    value.update(overrides)
    return value


def _fact_verification(**overrides: bool) -> dict[str, bool]:
    value = {
        name: True for name in materializer._REQUIRED_FACT_VERIFICATION_FIELDS
    }
    value.update(overrides)
    return value


def _sidecar(
    source: HotpotQATrainQASource,
    *,
    question: str,
    fact: str,
) -> dict[str, object]:
    return {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": question,
        "fact_statement": fact,
        "paraphrase_provenance": materializer.PARAPHRASE_PROVENANCE,
        "paraphrase_version": materializer.PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


def _args(tmp_path: Path, *, count: int) -> Namespace:
    return Namespace(
        dataset_catalog=str(tmp_path / "unused.yaml"),
        train_count=count,
        validation_count=1,
        limit=None,
        output=str(tmp_path / "sidecar.jsonl"),
        receipts=str(tmp_path / "receipts.jsonl"),
        manifest=str(tmp_path / "manifest.json"),
        model_catalog=str(tmp_path / "unused-models.yaml"),
        model_id="qwen3.5-9b-local",
        concurrency=1,
        checkpoint_every=1,
        seed=17,
        max_attempts=1,
        generation_rounds=1,
    )


class _Registry:
    def require_model(self, _model_id: str) -> object:
        return type("Model", (), {"model_id": "qwen3.5-9b-local"})()

    def provider_for(self, _model_id: str) -> object:
        return object()


def test_field_level_repair_preserves_admitted_question_and_repairs_only_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(0)
    request_ids: list[str] = []
    fact_verifications = 0

    async def fake_generate_json(**kwargs: object):
        nonlocal fact_verifications
        request_id = str(kwargs["request_id"])
        request_ids.append(request_id)
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "Which individual wrote Atlas?",
                "fact_statement": "Ada Lovelace authored Atlas.",
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "repair-question" in request_id:
            raise AssertionError("an admitted question must not be regenerated")
        if "repair-fact" in request_id:
            return {
                "fact_statement": "Atlas was authored by Ada Lovelace."
            }, {"request_id": request_id}
        if "verify-fact" in request_id:
            fact_verifications += 1
            verified = (
                _fact_verification(fact_supported_by_qa=False)
                if fact_verifications == 1
                else _fact_verification()
            )
            return verified, {"request_id": request_id}
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    candidate, receipt = asyncio.run(
        materializer._materialize_one(
            source,
            index=0,
            model=object(),
            provider=object(),
            seed=19,
            max_attempts=2,
            generation_rounds=1,
        )
    )

    assert candidate["paraphrase_question"] == "Which individual wrote Atlas?"
    assert candidate["fact_statement"] == "Atlas was authored by Ada Lovelace."
    assert sum(":generate:" in request_id for request_id in request_ids) == 1
    assert sum("verify-question" in request_id for request_id in request_ids) == 1
    assert not any("repair-question" in request_id for request_id in request_ids)
    assert sum("repair-fact" in request_id for request_id in request_ids) == 1
    assert sum("verify-fact" in request_id for request_id in request_ids) == 2
    attempts = receipt["attempt_receipts"]
    assert isinstance(attempts, list)
    assert attempts[0]["fact"]["failed_fields"] == ["fact_supported_by_qa"]
    assert attempts[1]["question"]["preserved_from_prior_attempt"] is True
    assert attempts[1]["fact"]["preserved_from_prior_attempt"] is False


@pytest.mark.parametrize(
    ("source", "paraphrase", "fact", "expected_mode", "fact_problem_prefix"),
    (
        (
            _source(
                1,
                question="What disability did singer Al Hibbler have?",
                answer="He was blind",
            ),
            "Which impairment did singer Al Hibbler live with?",
            "Singer Al Hibbler could not see.",
            "declarative_clause_paraphrase",
            "Dataset answer clause:",
        ),
        (
            _source(2),
            "Which individual wrote Atlas?",
            "Ada Lovelace authored Atlas.",
            "answer_slot_binding",
            "Dataset answer:\n",
        ),
    ),
)
def test_clausal_answer_and_short_span_use_distinct_binding_modes(
    monkeypatch: pytest.MonkeyPatch,
    source: HotpotQATrainQASource,
    paraphrase: str,
    fact: str,
    expected_mode: str,
    fact_problem_prefix: str,
) -> None:
    calls: list[Mapping[str, object]] = []

    async def fake_generate_json(**kwargs: object):
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": paraphrase,
                "fact_statement": fact,
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "verify-fact" in request_id:
            return _fact_verification(), {"request_id": request_id}
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    candidate, receipt = asyncio.run(
        materializer._materialize_one(
            source,
            index=0,
            model=object(),
            provider=object(),
            seed=23,
            max_attempts=1,
            generation_rounds=1,
        )
    )

    assert candidate["fact_statement"] == fact
    assert receipt["fact_binding_mode"] == expected_mode
    fact_verification = next(
        call for call in calls if "verify-fact" in str(call["request_id"])
    )
    assert fact_problem_prefix in str(fact_verification["problem"])
    if expected_mode == "declarative_clause_paraphrase":
        assert "Source question:" in str(fact_verification["problem"])
        assert source.question in str(fact_verification["problem"])
    else:
        assert source.canonical_answer in fact


def test_rejected_materialization_receipt_persists_candidate_and_failed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(0)
    monkeypatch.setattr(
        materializer,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: HotpotQAFullDatasetQASources((source,), ()),
    )
    monkeypatch.setattr(materializer, "load_model_registry", lambda _path: _Registry())

    async def fake_generate_json(**kwargs: object):
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "Which individual wrote Atlas?",
                "fact_statement": "Ada Lovelace authored Atlas.",
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "verify-fact" in request_id:
            return _fact_verification(fact_supported_by_qa=False), {
                "request_id": request_id
            }
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    args = _args(tmp_path, count=1)
    with pytest.raises(RuntimeError, match="1 fact materializations failed"):
        asyncio.run(materializer.materialize(args))

    receipts = [
        json.loads(line)
        for line in Path(args.receipts).read_text(encoding="utf-8").splitlines()
    ]
    assert len(receipts) == 1
    rejected = receipts[0]
    assert rejected["status"] == "rejected_after_bounded_attempts"
    assert rejected["generation_error_type"] == "FactMaterializationRejected"
    attempts = rejected["attempt_receipts"]
    assert attempts[0]["candidate"] == {
        "paraphrase_question": "Which individual wrote Atlas?",
        "fact_statement": "Ada Lovelace authored Atlas.",
    }
    assert attempts[0]["question"]["failed_fields"] == []
    assert attempts[0]["fact"]["failed_fields"] == ["fact_supported_by_qa"]
    assert not Path(args.manifest).exists()


def test_resume_revalidates_good_row_removes_stale_row_and_generates_only_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = _source(0, question="Who authored Alpha?", answer="Ada")
    beta = _source(1, question="Who authored Beta?", answer="Grace")
    gamma = _source(2, question="Who authored Gamma?", answer="Katherine")
    sources = (alpha, beta, gamma)
    monkeypatch.setattr(
        materializer,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: HotpotQAFullDatasetQASources(sources, ()),
    )
    monkeypatch.setattr(materializer, "load_model_registry", lambda _path: _Registry())
    args = _args(tmp_path, count=3)
    good = _sidecar(
        alpha,
        question="Which individual wrote Alpha?",
        fact="Ada authored Alpha.",
    )
    stale = _sidecar(
        beta,
        question=beta.question,
        fact="Grace authored Beta.",
    )
    Path(args.output).write_text(
        "".join(json.dumps(row) + "\n" for row in (good, stale)),
        encoding="utf-8",
    )
    Path(args.receipts).write_text(
        "".join(
            json.dumps(
                {
                    "source_train_task_id": row["source_train_task_id"],
                    "status": "accepted-by-old-admission",
                }
            )
            + "\n"
            for row in (good, stale)
        ),
        encoding="utf-8",
    )

    called: list[str] = []
    output_seen_by_first_generation: list[str] = []

    async def fake_materialize_one(
        source: HotpotQATrainQASource, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        if not called:
            output_seen_by_first_generation.extend(
                json.loads(line)["source_train_task_id"]
                for line in Path(args.output).read_text(encoding="utf-8").splitlines()
            )
        called.append(source.source_train_task_id)
        label = source.canonical_answer
        entity = source.question.removeprefix("Who authored ").removesuffix("?")
        candidate = _sidecar(
            source,
            question=f"Which individual wrote {entity}?",
            fact=f"{label} authored {entity}.",
        )
        return candidate, {
            "source_train_task_id": source.source_train_task_id,
            "status": "accepted",
        }

    monkeypatch.setattr(materializer, "_materialize_one", fake_materialize_one)
    manifest = asyncio.run(materializer.materialize(args))

    assert output_seen_by_first_generation == [alpha.source_train_task_id]
    assert called == [beta.source_train_task_id, gamma.source_train_task_id]
    final_rows = [
        json.loads(line)
        for line in Path(args.output).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["source_train_task_id"] for row in final_rows] == [
        source.source_train_task_id for source in sources
    ]
    assert final_rows[0] == good
    assert final_rows[1] != stale
    assert manifest["accepted_count"] == 3
    receipt_rows = {
        row["source_train_task_id"]: row
        for row in (
            json.loads(line)
            for line in Path(args.receipts).read_text(encoding="utf-8").splitlines()
        )
    }
    assert receipt_rows[alpha.source_train_task_id]["status"] == "resume_revalidated"
    assert receipt_rows[beta.source_train_task_id]["status"] == "accepted"
    assert receipt_rows[gamma.source_train_task_id]["status"] == "accepted"


def test_materializer_rolling_pool_refills_before_slow_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = tuple(_source(index) for index in range(4))
    monkeypatch.setattr(
        materializer,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: HotpotQAFullDatasetQASources(sources, ()),
    )
    monkeypatch.setattr(materializer, "load_model_registry", lambda _path: _Registry())
    release_first = asyncio.Event()
    third_started = asyncio.Event()

    async def fake_materialize_one(
        source: HotpotQATrainQASource,
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if source is sources[0]:
            await release_first.wait()
        if source is sources[2]:
            third_started.set()
        return _sidecar(
            source,
            question="Which individual wrote Atlas?",
            fact="Atlas was authored by Ada Lovelace.",
        ), {
            "source_train_task_id": source.source_train_task_id,
            "status": "accepted",
        }

    monkeypatch.setattr(materializer, "_materialize_one", fake_materialize_one)
    args = _args(tmp_path, count=4)
    args.concurrency = 2
    args.checkpoint_every = 2

    async def exercise() -> dict[str, object]:
        running = asyncio.create_task(materializer.materialize(args))
        await asyncio.wait_for(third_started.wait(), timeout=1.0)
        assert not release_first.is_set()
        release_first.set()
        return await running

    manifest = asyncio.run(exercise())
    assert manifest["accepted_count"] == 4
