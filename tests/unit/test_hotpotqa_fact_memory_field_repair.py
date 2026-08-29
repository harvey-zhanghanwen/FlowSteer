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


def _clause_fact_verification(**overrides: bool) -> dict[str, bool]:
    value = {
        name: True
        for name in materializer._REQUIRED_CLAUSE_FACT_VERIFICATION_FIELDS
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
    ("source", "paraphrase", "fact", "expected_mode"),
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
        ),
        (
            _source(2),
            "Which individual wrote Atlas?",
            "Ada Lovelace authored Atlas.",
            "answer_slot_binding",
        ),
    ),
)
def test_clausal_answer_and_short_span_use_distinct_binding_modes(
    monkeypatch: pytest.MonkeyPatch,
    source: HotpotQATrainQASource,
    paraphrase: str,
    fact: str,
    expected_mode: str,
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
            verification = (
                _clause_fact_verification()
                if expected_mode == "declarative_clause_paraphrase"
                else _fact_verification()
            )
            return verification, {"request_id": request_id}
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
    fact_problem = json.loads(str(fact_verification["problem"]))
    assert fact_problem["fact_binding_mode"] == expected_mode
    assert fact_problem["canonical_training_answer"] == source.canonical_answer
    assert fact_problem["declarative_fact"] == fact
    if expected_mode == "declarative_clause_paraphrase":
        assert fact_problem["original_question"] == source.question
        assert set(fact_verification["schema"]["required"]) == set(
            materializer._REQUIRED_CLAUSE_FACT_VERIFICATION_FIELDS
        )
        assert "answer_slot_bound" not in fact_verification["schema"][
            "required"
        ]
    else:
        assert source.canonical_answer in fact


def test_targeted_repair_uses_triviaqa_immutable_and_answer_slot_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        3,
        question="In 2012, who authored Atlas?",
        answer="Ada Lovelace",
    )
    calls: list[Mapping[str, object]] = []
    fact_verifications = 0

    async def fake_generate_json(**kwargs: object):
        nonlocal fact_verifications
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "Who in 2012 authored Atlas?",
                "fact_statement": "Ada Lovelace authored Atlas.",
            }, {"request_id": request_id}
        if "repair-question" in request_id:
            return {
                "paraphrase_question": "In 2012, which person wrote Atlas?",
                "replaced_source_token": "authored",
                "replacement_phrase": "wrote",
            }, {"request_id": request_id}
        if "repair-fact" in request_id:
            return {
                "fact_statement": (
                    "In 2012, Atlas was authored by Ada Lovelace."
                ),
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "verify-fact" in request_id:
            fact_verifications += 1
            return (
                _fact_verification(relation_direction_preserved=False)
                if fact_verifications == 1
                else _fact_verification()
            ), {"request_id": request_id}
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    candidate, receipt = asyncio.run(
        materializer._materialize_one(
            source,
            index=0,
            model=object(),
            provider=object(),
            seed=29,
            max_attempts=2,
            generation_rounds=1,
        )
    )

    assert candidate["paraphrase_question"] == (
        "In 2012, which person wrote Atlas?"
    )
    assert candidate["fact_statement"] == (
        "In 2012, Atlas was authored by Ada Lovelace."
    )
    question_repair = next(
        call for call in calls if "repair-question" in str(call["request_id"])
    )
    question_payload = json.loads(str(question_repair["problem"]))
    assert question_payload["immutable_number_or_date_tokens"] == ["2012"]
    assert question_payload["immutable_original_entity_tokens"] == ["Atlas"]
    assert question_payload["forbidden_question_canonical_tokens"] == [
        "ada",
        "lovelace",
    ]
    assert question_payload["lexical_replacement_source_tokens"] == [
        "authored"
    ]
    assert question_payload["required_source_token_to_replace"] == "authored"
    assert "canonical_training_answer" not in question_payload

    fact_repair = next(
        call for call in calls if "repair-fact" in str(call["request_id"])
    )
    fact_payload = json.loads(str(fact_repair["problem"]))
    assert fact_payload["canonical_training_answer"] == "Ada Lovelace"
    assert fact_payload["immutable_answer_entity_tokens"] == [
        "Ada",
        "Lovelace",
    ]
    assert fact_payload["allowed_fact_number_or_date_tokens"] == ["2012"]
    assert fact_payload["fact_repair_strategy"] == (
        "authoritative_answer_slot_reconstruction"
    )
    assert "rejected_fact" not in fact_payload
    assert "relation_direction_preserved" in fact_payload[
        "prior_admission_result"
    ]

    question_verification = next(
        call for call in calls if "verify-question" in str(call["request_id"])
    )
    verifier_payload = json.loads(str(question_verification["problem"]))
    assert verifier_payload["canonical_answer_leakage_checked_deterministically"] is True
    assert "canonical_training_answer" not in verifier_payload
    assert "answer_not_revealed" in materializer._REQUIRED_QUESTION_VERIFICATION_FIELDS
    assert {
        "answer_slot_bound",
        "relation_direction_preserved",
        "no_new_fact_or_relation",
    }.issubset(materializer._REQUIRED_FACT_VERIFICATION_FIELDS)
    assert receipt["attempt_receipts"][0]["fact"]["failed_fields"] == [
        "relation_direction_preserved"
    ]
    assert receipt["attempt_receipts"][1]["question"][
        "required_source_token_to_replace"
    ] == "authored"
    assert receipt["attempt_receipts"][1]["fact"]["repair_strategy"] == (
        "authoritative_answer_slot_reconstruction"
    )
    assert receipt["attempt_receipts"][1]["fact"]["repair_temperature"] == 0.1
    assert fact_repair["temperature"] == 0.1


def test_fact_repair_strategy_reconstructs_semantics_but_preserves_immutable_fix() -> None:
    assert materializer._fact_repair_strategy(
        "fact verifier rejected: fact_supported_by_qa,answer_slot_bound"
    ) == "authoritative_answer_slot_reconstruction"
    assert materializer._fact_repair_strategy(
        "ValueError: fact_statement is identical to the canonical answer"
    ) == "authoritative_answer_slot_reconstruction"
    assert materializer._fact_repair_strategy(
        "ValueError: fact_statement introduced a number or date"
    ) == "preserve_and_repair_immutable_fields"


def test_question_synonym_only_repair_rewrites_authoritative_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        4,
        question="In 2012, who authored Atlas?",
        answer="Ada Lovelace",
    )
    calls: list[Mapping[str, object]] = []

    async def fake_generate_json(**kwargs: object):
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "In 2013, which person wrote Atlas?",
                "fact_statement": "Ada Lovelace authored Atlas.",
            }, {"request_id": request_id}
        if "repair-question-synonym" in request_id:
            return {
                "source_token": "authored",
                "replacement_phrase": "wrote",
            }, {"request_id": request_id}
        if "repair-question" in request_id:
            return {
                "paraphrase_question": "In 2012, which person wrote Atlas?",
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
            seed=31,
            max_attempts=2,
            generation_rounds=1,
        )
    )

    assert candidate["paraphrase_question"] == "In 2012, who wrote Atlas?"
    assert candidate["paraphrase_question"] != source.question
    synonym_call = next(
        call
        for call in calls
        if "repair-question-synonym" in str(call["request_id"])
    )
    synonym_payload = json.loads(str(synonym_call["problem"]))
    assert synonym_payload["required_source_token"] == "authored"
    assert "rejected_question" not in synonym_payload
    second_attempt = receipt["attempt_receipts"][1]["question"]
    assert second_attempt["repair_mode"] == "synonym_only"
    assert second_attempt["structured_repair_skipped"] is True
    assert "structured_repair_error" in second_attempt
    assert second_attempt["failed_fields"] == []


def test_repeated_fact_forces_rotating_source_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        5,
        question="Did Ada author Atlas?",
        answer="yes",
    )
    calls: list[Mapping[str, object]] = []
    fact_repairs = 0
    fact_verifications = 0

    async def fake_generate_json(**kwargs: object):
        nonlocal fact_repairs, fact_verifications
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "Was Atlas authored by Ada?",
                "fact_statement": "Ada authored Atlas.",
            }, {"request_id": request_id}
        if "repair-fact" in request_id:
            fact_repairs += 1
            return {
                "fact_statement": (
                    "Ada authored Atlas."
                    if fact_repairs == 1
                    else "Atlas was authored by Ada."
                )
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "verify-fact" in request_id:
            fact_verifications += 1
            return (
                _fact_verification(fact_supported_by_qa=False)
                if fact_verifications < 3
                else _fact_verification()
            ), {"request_id": request_id}
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    candidate, receipt = asyncio.run(
        materializer._materialize_one(
            source,
            index=0,
            model=object(),
            provider=object(),
            seed=37,
            max_attempts=3,
            generation_rounds=1,
        )
    )

    assert candidate["fact_statement"] == "Atlas was authored by Ada."
    repair_calls = [
        call for call in calls if "repair-fact" in str(call["request_id"])
    ]
    repair_payloads = [
        json.loads(str(call["problem"])) for call in repair_calls
    ]
    assert all(call["temperature"] == 0.1 for call in repair_calls)
    assert [
        payload["answer_reconstruction_pattern"]
        for payload in repair_payloads
    ] == [
        "literal_answer_slot_substitution",
        "relative_clause_binding",
    ]
    assert all("rejected_fact" not in payload for payload in repair_payloads)
    attempts = receipt["attempt_receipts"]
    assert [attempt["fact"]["candidate_number"] for attempt in attempts] == [
        1,
        2,
        3,
    ]
    assert [attempt["fact"]["candidate_repeated"] for attempt in attempts] == [
        False,
        True,
        False,
    ]


def test_complete_fact_gets_local_terminal_punctuation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(8)
    calls: list[Mapping[str, object]] = []

    async def fake_generate_json(**kwargs: object):
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "Which individual wrote Atlas?",
                "fact_statement": "Atlas was authored by Ada Lovelace",
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
            seed=47,
            max_attempts=2,
            generation_rounds=1,
        )
    )

    assert candidate["fact_statement"] == "Atlas was authored by Ada Lovelace."
    assert not any(
        "repair-fact" in str(call["request_id"])
        for call in calls
    )
    second_fact = receipt["attempt_receipts"][1]["fact"]
    assert second_fact["repair_strategy"] == "terminal_punctuation_only"
    assert second_fact["generation_response"] == {
        "local_surface_repair": "terminal_punctuation_only"
    }
    assert materializer._repair_missing_terminal_punctuation(
        "Ada Lovelace"
    ) is None
    assert materializer._clause_requires_synonym_repair(
        "fact_statement removed an immutable entity from the answer clause"
    )


def test_clause_identical_tail_uses_synonym_repair_and_clause_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(
        6,
        question="What occurred to Atlas in 2012?",
        answer="Atlas was released in 2012.",
    )
    calls: list[Mapping[str, object]] = []

    async def fake_generate_json(**kwargs: object):
        calls.append(dict(kwargs))
        request_id = str(kwargs["request_id"])
        if ":generate:" in request_id:
            return {
                "paraphrase_question": "What happened to Atlas during 2012?",
                "fact_statement": "Atlas was released in 2012.",
            }, {"request_id": request_id}
        if "repair-fact-clause-synonym" in request_id:
            return {
                "source_token": "released",
                "replacement_phrase": "issued",
            }, {"request_id": request_id}
        if "verify-question" in request_id:
            return _question_verification(), {"request_id": request_id}
        if "verify-fact" in request_id:
            return _clause_fact_verification(), {"request_id": request_id}
        raise AssertionError(f"unexpected request: {request_id}")

    monkeypatch.setattr(materializer, "_generate_json", fake_generate_json)
    candidate, receipt = asyncio.run(
        materializer._materialize_one(
            source,
            index=0,
            model=object(),
            provider=object(),
            seed=41,
            max_attempts=2,
            generation_rounds=1,
        )
    )

    assert candidate["fact_statement"] == "Atlas was issued in 2012."
    clause_repair = next(
        call
        for call in calls
        if "repair-fact-clause-synonym" in str(call["request_id"])
    )
    clause_payload = json.loads(str(clause_repair["problem"]))
    assert clause_payload["required_source_token"] == "released"
    assert clause_repair["temperature"] == 0.0
    verifier_call = next(
        call for call in calls if "verify-fact" in str(call["request_id"])
    )
    assert set(verifier_call["schema"]["required"]) == set(
        materializer._REQUIRED_CLAUSE_FACT_VERIFICATION_FIELDS
    )
    second_fact = receipt["attempt_receipts"][1]["fact"]
    assert second_fact["repair_strategy"] == "clause_synonym_only"
    assert second_fact["verification_mode"] == "answer_clause"
    assert second_fact["repair_temperature"] == 0.0


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
