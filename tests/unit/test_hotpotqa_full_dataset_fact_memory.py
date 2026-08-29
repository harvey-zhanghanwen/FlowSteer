from __future__ import annotations

import asyncio
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import materialize_hotpotqa_full_dataset_fact_memory as materializer
from src.interactive import hotpotqa_full_dataset_fact_memory_index as fact_index
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_DOCUMENT_FORMAT,
    FULL_DATASET_FACT_DOCUMENT_TEMPLATE,
    FULL_DATASET_FACT_INDEXED_TEXT_FIELD,
    FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
    FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION,
    HotpotQADeclarativeFact,
    HotpotQAFullDatasetFactMemoryIndex,
    HotpotQAFullDatasetFactMemoryIndexManifest,
    HotpotQAFullDatasetQASources,
    build_hotpotqa_full_dataset_fact_memory_index,
    load_hotpotqa_full_dataset_qa_sources,
    materialize_hotpotqa_declarative_facts,
)
from src.interactive.hotpotqa_qa_memory_index import HotpotQATrainQASource


def _source(index: int, *, answer: str | None = None) -> HotpotQATrainQASource:
    return HotpotQATrainQASource(
        source_train_task_id=f"hotpotqa:source-{index}",
        base_task_id=f"hotpotqa:source-{index}",
        cycled=False,
        question=f"Who was person {index}?",
        canonical_answer=answer or f"Person {index}",
    )


def _materialization(
    source: HotpotQATrainQASource,
    *,
    fact: str | None = None,
) -> dict[str, object]:
    return {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": f"Which individual was designated {source.base_task_id}?",
        "fact_statement": fact or f"The designated individual was {source.canonical_answer}.",
        "paraphrase_provenance": materializer.PARAPHRASE_PROVENANCE,
        "paraphrase_version": materializer.PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


def _record(
    task_id: str,
    *,
    split: str,
    question: str,
    answer: str,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "split": split,
        "question": (
            "Based on the following passages, answer the question.\n\n"
            "[[private context that must not survive]]\n\n"
            f"Question: {question}"
        ),
        "ground_truth": answer,
        "metadata": {"evaluator_payload": {"supporting_facts": "private"}},
    }


def _catalog(path: Path) -> None:
    path.write_text(
        """
sources:
  hotpotqa:
    path: /unused
    files:
      train: train-*.parquet
      validation: validation-*.parquet
    candidate_sequence: [train, validation]
""".lstrip(),
        encoding="utf-8",
    )


def test_loader_keeps_raw_qa_only_in_index_external_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "datasets.yaml"
    _catalog(catalog)
    records = (
        _record(
            "hotpotqa:train-1",
            split="train",
            question="Who wrote it?",
            answer="Ada",
        ),
        _record(
            "hotpotqa:validation-1",
            split="validation",
            question="Where was it built?",
            answer="Rome",
        ),
    )
    monkeypatch.setattr(fact_index, "_hotpot_records", lambda _config: iter(records))

    sources = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=catalog,
        expected_train_count=1,
        expected_validation_count=1,
    )

    assert sources.train[0].question == "Who wrote it?"
    assert sources.validation[0].canonical_answer == "Rome"
    assert "private context" not in repr(sources.combined).casefold()
    assert "supporting" not in repr(sources.combined).casefold()


def test_loader_rejects_native_split_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "datasets.yaml"
    _catalog(catalog)
    records = (
        _record("hotpotqa:same", split="train", question="Who?", answer="Ada"),
        _record(
            "hotpotqa:same", split="validation", question="Where?", answer="Rome"
        ),
    )
    monkeypatch.setattr(fact_index, "_hotpot_records", lambda _config: iter(records))
    with pytest.raises(ValueError, match="overlap"):
        load_hotpotqa_full_dataset_qa_sources(
            dataset_catalog_path=catalog,
            expected_train_count=1,
            expected_validation_count=1,
        )


def test_materialization_projects_only_declarative_fact() -> None:
    source = _source(0, answer="Ada Lovelace")
    fact = materialize_hotpotqa_declarative_facts(
        (source,),
        (_materialization(source, fact="Ada Lovelace authored the work."),),
    )[0]

    assert fact.to_value() == {
        "memory_id": "hotpotqa-fact-000000",
        "fact_text": "Ada Lovelace authored the work.",
    }
    serialized = json.dumps(fact.to_value())
    for forbidden in (
        "question",
        "answer",
        "canonical",
        "ground_truth",
        "paraphrase",
        "source_train_task_id",
    ):
        assert forbidden not in serialized.casefold()


def test_materialization_rejects_verbatim_question_and_qa_wire() -> None:
    source = _source(0)
    verbatim = _materialization(source)
    verbatim["paraphrase_question"] = source.question
    with pytest.raises(ValueError, match="identical"):
        materialize_hotpotqa_declarative_facts((source,), (verbatim,))

    qa_wire = _materialization(source, fact="Question: Who?\nAnswer: Person 0")
    with pytest.raises(ValueError, match="Question/Answer"):
        materialize_hotpotqa_declarative_facts((source,), (qa_wire,))

    sentence_answer = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:raw-answer",
        base_task_id="hotpotqa:raw-answer",
        cycled=False,
        question="How many genera and species are in Stachyuraceae?",
        canonical_answer=(
            "Stachyuraceae contains a single genus with eight species."
        ),
    )
    raw_answer = _materialization(
        sentence_answer,
        fact=sentence_answer.canonical_answer,
    )
    raw_answer["paraphrase_question"] = (
        "What are the genus and species counts within Stachyuraceae?"
    )
    with pytest.raises(ValueError, match="identical to the canonical answer"):
        materialize_hotpotqa_declarative_facts(
            (sentence_answer,), (raw_answer,)
        )


@pytest.mark.parametrize(
    "pronoun",
    ("It", "He", "She", "They", "This", "That", "These", "Those"),
)
def test_materialization_rejects_unbound_anaphoric_subject(pronoun: str) -> None:
    source = _source(0, answer="Ada Lovelace")
    candidate = _materialization(
        source,
        fact=f"{pronoun} authored the referenced work as Ada Lovelace.",
    )

    with pytest.raises(ValueError, match="unbound anaphoric subject"):
        materialize_hotpotqa_declarative_facts((source,), (candidate,))


def test_materialization_rejects_lexically_bare_canonical_answer() -> None:
    source = _source(0, answer="The Right Stuff!")
    candidate = _materialization(source, fact='"The Right Stuff."')

    with pytest.raises(ValueError, match="identical to the canonical answer"):
        materialize_hotpotqa_declarative_facts((source,), (candidate,))


def test_materialization_rejects_lexically_verbatim_yes_no_question() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:verbatim-binary",
        base_task_id="hotpotqa:verbatim-binary",
        cycled=False,
        question="Erin Wiedner and Monte Hellman are both film directors?",
        canonical_answer="yes",
    )
    candidate = _materialization(
        source,
        fact="Erin Wiedner and Monte Hellman are both film directors.",
    )

    with pytest.raises(ValueError, match="complete source question lexical surface"):
        materialize_hotpotqa_declarative_facts((source,), (candidate,))


def test_materialization_rejects_verbatim_question_prefix_plus_answer() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:verbatim-prefix",
        base_task_id="hotpotqa:verbatim-prefix",
        cycled=False,
        question="The Seattle University team plays in a complex known as?",
        canonical_answer="the Seattle Center",
    )
    candidate = _materialization(
        source,
        fact=(
            "The Seattle University team plays in a complex known as: "
            "the Seattle Center."
        ),
    )

    with pytest.raises(ValueError, match="complete source question lexical surface"):
        materialize_hotpotqa_declarative_facts((source,), (candidate,))


@pytest.mark.parametrize(
    "fact",
    (
        "The answer to the question of whether both entities match is no.",
        "Ada Lovelace is the answer to the query about who authored the work.",
        "Ada Lovelace is the subject of the inquiry about the work's author.",
        "The public university in the question is located in Boston.",
        "The individual described in the original query was Ada Lovelace.",
        "Ada Lovelace is the target of this specific query.",
        "Ada Lovelace is the subject of this specific ranking query.",
        "Ada Lovelace is the subject of a question asking about authorship.",
        "The question asks whether both entities match.",
        "The description provided in the query identifies Ada Lovelace.",
        "Ada Lovelace appears in the context of the query.",
        "The proposition is the one identified by the dataset answer.",
        "The banjo appears directly behind another song on the HotpotQA dataset.",
        "Both descriptions have the answer Ada Lovelace.",
    ),
)
def test_materialization_rejects_qa_meta_framing(fact: str) -> None:
    source = _source(0, answer="Ada Lovelace")
    candidate = _materialization(source, fact=fact)

    with pytest.raises(ValueError, match="Question/Answer wire"):
        materialize_hotpotqa_declarative_facts((source,), (candidate,))


@pytest.mark.parametrize(
    ("question", "answer", "paraphrase", "fact"),
    (
        (
            "Are Jane and First for Women both women's magazines?",
            "yes",
            "Do Jane and First for Women both qualify as women's magazines?",
            "Jane and First for Women are both women's magazines.",
        ),
        (
            "How old is the female main protagonist of Catching Fire?",
            "16-year-old",
            "What is the age of the female lead in Catching Fire?",
            "The female main protagonist of Catching Fire is 16 years old.",
        ),
    ),
)
def test_materialization_allows_semantically_equivalent_answer_realization(
    question: str,
    answer: str,
    paraphrase: str,
    fact: str,
) -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:equivalent-answer",
        base_task_id="hotpotqa:equivalent-answer",
        cycled=False,
        question=question,
        canonical_answer=answer,
    )
    value = _materialization(source, fact=fact)
    value["paraphrase_question"] = paraphrase
    admitted = materialize_hotpotqa_declarative_facts((source,), (value,))
    assert admitted[0].fact_text == fact


def test_materialization_allows_quote_style_and_entity_description_rewrite() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:semantic-surface-rewrite",
        base_task_id="hotpotqa:semantic-surface-rewrite",
        cycled=False,
        question='What American band recorded "The Track"?',
        canonical_answer="The Example Band",
    )
    value = _materialization(
        source,
        fact="The Example Band recorded 'The Track'.",
    )
    value["paraphrase_question"] = (
        "Which U.S. musical group made the recording called 'The Track'?"
    )
    admitted = materialize_hotpotqa_declarative_facts((source,), (value,))
    assert admitted[0].fact_text == "The Example Band recorded 'The Track'."


def test_materialization_preserves_canonical_identity_without_capitalization_novelty() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:immutable-answer",
        base_task_id="hotpotqa:immutable-answer",
        cycled=False,
        question="Who directed the Paris Opera Ballet?",
        canonical_answer="Rudolf Khametovich Nureyev",
    )
    capitalization_is_not_an_entity_recognizer = _materialization(
        source,
        fact=(
            "Rudolf Khametovich Nureyev and Michelle Pfeiffer directed the "
            "Paris Opera Ballet."
        ),
    )
    # TriviaQA delegates unsupported-fact detection to the semantic verifier;
    # deterministic projection must not infer entity novelty from title case.
    materialize_hotpotqa_declarative_facts(
        (source,), (capitalization_is_not_an_entity_recognizer,)
    )

    truncated_name = _materialization(
        source,
        fact="Rudolf Nureyev directed the Paris Opera Ballet.",
    )
    with pytest.raises(ValueError, match="immutable answer tokens"):
        materialize_hotpotqa_declarative_facts((source,), (truncated_name,))


def test_hotpot_numeric_surface_normalization_and_title_binding() -> None:
    numeric = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:a2002",
        base_task_id="hotpotqa:a2002",
        cycled=False,
        question=(
            "Ted Sutton plays Sergeant Cunningham in a2002 film that stars "
            "Mel Gibson as what character?"
        ),
        canonical_answer="Graham Hess",
    )
    value = _materialization(
        numeric,
        fact=(
            "In the 2002 film featuring Ted Sutton as Sergeant Cunningham, "
            "Mel Gibson portrays Graham Hess."
        ),
    )
    value["paraphrase_question"] = (
        "In the 2002 movie featuring Ted Sutton as Sergeant Cunningham, "
        "which character is portrayed by Mel Gibson?"
    )
    assert materialize_hotpotqa_declarative_facts((numeric,), (value,))

    title = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:title",
        base_task_id="hotpotqa:title",
        cycled=False,
        question="Which song did the artist release?",
        canonical_answer="I Knew You Were Trouble",
    )
    assert not fact_index.canonical_answer_is_declarative_clause(
        title.canonical_answer,
        question=title.question,
    )


def test_ordinary_answer_head_may_be_semantically_realized() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:dancer",
        base_task_id="hotpotqa:dancer",
        cycled=False,
        question="Maurice Hines and his brother were famous for what?",
        canonical_answer="dancer Gregory Hines",
    )
    value = _materialization(
        source,
        fact="Maurice Hines and his brother Gregory Hines were famous for dancing.",
    )
    value["paraphrase_question"] = (
        "For what activity were Maurice Hines and his brother renowned?"
    )
    assert materialize_hotpotqa_declarative_facts((source,), (value,))

    quoted = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:quoted-title",
        base_task_id="hotpotqa:quoted-title",
        cycled=False,
        question="On which popular show did Alamgir serve as a judge?",
        canonical_answer='"Music Icons"',
    )
    quoted_value = _materialization(
        quoted,
        fact="Alamgir served as a judge on the popular show Music Icons.",
    )
    quoted_value["paraphrase_question"] = (
        "Which well-known program had Alamgir on its judging panel?"
    )
    assert materialize_hotpotqa_declarative_facts((quoted,), (quoted_value,))

    demonym = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:demonym",
        base_task_id="hotpotqa:demonym",
        cycled=False,
        question="12 Stones and Halestorm are bands from which country?",
        canonical_answer="American",
    )
    demonym_value = _materialization(
        demonym,
        fact="12 Stones and Halestorm are bands from America.",
    )
    demonym_value["paraphrase_question"] = (
        "From which nation do the groups 12 Stones and Halestorm originate?"
    )
    assert materialize_hotpotqa_declarative_facts(
        (demonym,), (demonym_value,)
    )


def test_complete_sentence_answer_and_repeated_numeric_reference() -> None:
    sentence = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:sentence-answer",
        base_task_id="hotpotqa:sentence-answer",
        cycled=False,
        question=(
            "Who was responsible for a commemorative half dollar struck by "
            "the Philadelphia Mint in 1936?"
        ),
        canonical_answer=(
            "Melish was responsible for two United States commemorative coin "
            "issues, the Cincinnati Musical Center half dollar and the "
            "Cleveland Centennial half dollar."
        ),
    )
    assert fact_index.canonical_answer_is_declarative_clause(
        sentence.canonical_answer,
        question=sentence.question,
    )

    world_war = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:wwii",
        base_task_id="hotpotqa:wwii",
        cycled=False,
        question=(
            "Which group won WWII in the alternate universe of the August "
            "2009 Raven Software game?"
        ),
        canonical_answer="Nazis",
    )
    value = _materialization(
        world_war,
        fact=(
            "The Raven Software game released in August 2009 depicts the Nazis "
            "winning World War II; its alternate history repeats the World War "
            "II premise."
        ),
    )
    value["paraphrase_question"] = (
        "In Raven Software's August 2009 alternate-history game, which group "
        "is victorious in World War II?"
    )
    assert materialize_hotpotqa_declarative_facts((world_war,), (value,))


def test_binary_both_nonbinary_official_answer_uses_clause_binding() -> None:
    question = "Were Alpha and Beta both founded in California?"
    assert fact_index.canonical_answer_is_declarative_clause(
        "Alpha was founded in California, while Beta was founded in Nevada.",
        question=question,
    )
    assert not fact_index.canonical_answer_is_declarative_clause(
        "yes",
        question=question,
    )
    assert not fact_index.canonical_answer_is_declarative_clause(
        "California",
        question="Where were Alpha and Beta founded?",
    )


def test_date_comma_spacing_preserves_numeric_atoms() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:date-comma",
        base_task_id="hotpotqa:date-comma",
        cycled=False,
        question="What occurred on April 13,1979?",
        canonical_answer="The Atlas release",
    )
    assert fact_index.validate_hotpotqa_question_rewrite(
        source,
        "Which event happened on April 13, 1979?",
    )
    with pytest.raises(ValueError, match="immutable number or date"):
        fact_index.validate_hotpotqa_question_rewrite(
            source,
            "Which event happened on April 14, 1979?",
        )


def test_internal_title_question_mark_is_declarative_but_terminal_is_not() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:question-title",
        base_task_id="hotpotqa:question-title",
        cycled=False,
        question="Which game was produced before Catan?",
        canonical_answer="Guess Who?",
    )
    assert fact_index.validate_hotpotqa_fact_statement(
        source,
        "Guess Who? was produced before Catan.",
    )
    with pytest.raises(ValueError, match="must be declarative"):
        fact_index.validate_hotpotqa_fact_statement(source, "Did X?")


def test_materializer_has_no_fallback_option() -> None:
    parser = materializer._parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--allow-dataset-pair-fallback" not in option_strings
    assert "--materialize-pending-as-dataset-pair-fallback" not in option_strings
    assert "--bootstrap-materialization" not in option_strings


def test_materializer_continues_then_resume_retries_only_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tuple(_source(index) for index in range(4))
    monkeypatch.setattr(
        materializer,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: HotpotQAFullDatasetQASources(sources, ()),
    )

    class Registry:
        def require_model(self, _model_id: str) -> object:
            return type("Model", (), {"model_id": "qwen3.5-9b-local"})()

        def provider_for(self, _model_id: str) -> object:
            return object()

    monkeypatch.setattr(materializer, "load_model_registry", lambda _path: Registry())
    called: list[str] = []

    async def fail_first(
        source: HotpotQATrainQASource, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        called.append(source.source_train_task_id)
        if source.source_train_task_id == sources[0].source_train_task_id:
            raise RuntimeError("bounded verifier rejection")
        return _materialization(source), {
            "source_train_task_id": source.source_train_task_id,
            "status": "accepted",
        }

    monkeypatch.setattr(materializer, "_materialize_one", fail_first)
    output = tmp_path / "sidecar.jsonl"
    receipts = tmp_path / "receipts.jsonl"
    args = Namespace(
        dataset_catalog=str(tmp_path / "unused.yaml"),
        train_count=4,
        validation_count=1,
        limit=None,
        output=str(output),
        receipts=str(receipts),
        manifest=str(tmp_path / "manifest.json"),
        model_catalog=str(tmp_path / "unused-models.yaml"),
        model_id="qwen3.5-9b-local",
        concurrency=1,
        checkpoint_every=1,
        seed=7,
        max_attempts=1,
    )
    with pytest.raises(RuntimeError, match=r"1 fact materializations failed.*\(3/4\)"):
        asyncio.run(materializer.materialize(args))
    assert called == [source.source_train_task_id for source in sources]
    assert not Path(args.manifest).exists()

    called.clear()

    async def succeed(
        source: HotpotQATrainQASource, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        called.append(source.source_train_task_id)
        return _materialization(source), {
            "source_train_task_id": source.source_train_task_id,
            "status": "accepted",
        }

    monkeypatch.setattr(materializer, "_materialize_one", succeed)
    manifest = asyncio.run(materializer.materialize(args))
    assert called == [sources[0].source_train_task_id]
    assert manifest["source_record_count"] == 4
    assert manifest["question_rewrite_count"] == 4
    assert manifest["fact_count"] == 4
    assert manifest["accepted_count"] == 4
    assert manifest["semantic_rewrite_coverage"] == 1.0
    assert manifest["fallback_count"] == 0
    assert manifest["rejected_count"] == 0
    admitted = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(admitted) == 4
    assert all(
        "fallback" not in str(row["paraphrase_provenance"]).casefold()
        for row in admitted
    )


def _manifest() -> HotpotQAFullDatasetFactMemoryIndexManifest:
    return HotpotQAFullDatasetFactMemoryIndexManifest(
        schema_version=FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION,
        index_id="hotpotqa-fact-test",
        corpus_version=FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
        source="HotpotQA facts",
        source_splits=("train", "validation"),
        embedding_model="local-bge",
        embedding_model_path="/models/bge",
        embedding_dimension=2,
        normalized=True,
        similarity="cosine",
        frozen_top_k=1,
        source_record_count=2,
        source_train_count=1,
        source_validation_count=1,
        unique_source_count=2,
        cycled_record_count=0,
        question_rewrite_count=2,
        fact_count=2,
        semantic_rewrite_coverage=1.0,
        frozen_evaluation_count=1,
        evaluation_overlap_count=1,
        contains_evaluation_source_facts=True,
        contains_raw_questions=False,
        contains_raw_answers=False,
        evaluation_scope=FULL_DATASET_EVALUATION_SCOPE,
        official_heldout_eligible=False,
        paraphrase_versions=("test-v1",),
        paraphrase_provenances=("unit-test",),
        document_template=FULL_DATASET_FACT_DOCUMENT_TEMPLATE,
        document_format=FULL_DATASET_FACT_DOCUMENT_FORMAT,
        indexed_text_field=FULL_DATASET_FACT_INDEXED_TEXT_FIELD,
        source_dataset_catalog_path="/datasets/catalog.yaml",
        source_train_path="/datasets/train.parquet",
        source_validation_path="/datasets/validation.parquet",
        facts_path="facts.jsonl",
        embeddings_path="embeddings.npy",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("question_rewrite_count", 1, "question must be rewritten"),
        ("fact_count", 1, "one declarative fact"),
        ("semantic_rewrite_coverage", 0.5, "exactly 100%"),
        ("contains_raw_questions", True, "raw Q-A"),
        ("contains_raw_answers", True, "raw Q-A"),
    ),
)
def test_index_manifest_fails_closed_on_incomplete_or_raw_qa_corpus(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_manifest(), **{field: value})


def test_index_search_and_read_expose_only_fact_wire() -> None:
    class Model:
        def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
            assert texts
            return np.asarray([[1.0, 0.0] for _item in texts], dtype=np.float32)

    facts = (
        HotpotQADeclarativeFact("hotpotqa-fact-000000", "Ada authored the work."),
        HotpotQADeclarativeFact("hotpotqa-fact-000001", "Rome is in Italy."),
    )
    index = HotpotQAFullDatasetFactMemoryIndex(
        manifest=_manifest(),
        facts=facts,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model=Model(),
    )
    hit = asyncio.run(index.search("Who authored the work?", 1))[0]
    assert set(hit.__dataclass_fields__) == {
        "memory_id",
        "fact_snippet",
        "similarity",
        "rank",
    }
    assert index.read(hit.memory_id).to_value() == {
        "memory_id": "hotpotqa-fact-000000",
        "fact_text": "Ada authored the work.",
    }


def test_builder_vectorizes_only_fact_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _source(0, answer="Ada")
    validation = _source(1, answer="Rome")
    sources = HotpotQAFullDatasetQASources((train,), (validation,))
    monkeypatch.setattr(
        fact_index,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: sources,
    )
    monkeypatch.setattr(
        fact_index,
        "_native_source_paths",
        lambda _path: ("/raw/train.parquet", "/raw/validation.parquet"),
    )
    monkeypatch.setattr(fact_index, "_load_sentence_transformer", lambda *_args: object())
    captured: list[str] = []

    def capture_encode(_model: object, texts: list[str], **_kwargs: object) -> np.ndarray:
        captured.extend(texts)
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(fact_index, "_encode", capture_encode)
    paraphrases = (
        _materialization(train, fact="Ada authored the work."),
        _materialization(validation, fact="Rome is the relevant location."),
    )
    manifest = build_hotpotqa_full_dataset_fact_memory_index(
        index_dir=tmp_path / "index",
        dataset_catalog_path=tmp_path / "catalog.yaml",
        frozen_evaluation_task_ids=(validation.source_train_task_id,),
        paraphrases=paraphrases,
        embedding_model_path="/models/bge",
        embedding_model_id="local-bge",
        embedding_device="cpu",
        frozen_top_k=1,
        expected_train_count=1,
        expected_validation_count=1,
    )
    assert captured == ["Ada authored the work.", "Rome is the relevant location."]
    fact_rows = [
        json.loads(line)
        for line in (tmp_path / "index" / "facts.jsonl").read_text().splitlines()
    ]
    assert all(set(row) == {"memory_id", "fact_text"} for row in fact_rows)
    assert manifest.contains_raw_questions is False
    assert manifest.contains_raw_answers is False
    assert manifest.indexed_text_field == "fact_text"
    assert manifest.semantic_rewrite_coverage == 1.0
