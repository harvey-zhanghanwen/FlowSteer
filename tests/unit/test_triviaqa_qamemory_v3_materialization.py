from __future__ import annotations

import json

import pytest

from src.interactive.triviaqa_qa_memory import (
    TriviaQAQAMemoryRecord,
    TriviaQATrainSource,
    exact_canonical_span_preserved,
    relation_bearing_answer_statement,
    validate_qa_memory_against_sources,
)
from scripts.generate_triviaqa_qa_memory_paraphrases import (
    ANSWER_VERIFICATION_SYSTEM_PROMPT,
    ANSWER_REPAIR_SYSTEM_PROMPT,
    QUESTION_REPAIR_SYSTEM_PROMPT,
    SYNONYM_REPAIR_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    build_answer_verification_messages,
    build_answer_repair_messages,
    build_paraphrase_messages,
    build_question_repair_messages,
    build_synonym_repair_messages,
    build_verification_messages,
    _birth_event_is_target_relation,
    _exact_question_identity_contaminated_fields,
    _has_lexical_or_phrase_replacement,
    _identity_token_preserved,
    _leading_answer_slot_anchor,
    _leading_answer_slot_anchor_preserved,
    _literal_subject_wh_answer_statement,
    _literal_slot_substitution_preserved,
    _listed_choice_answer_binding_preserved,
    _augment_listed_choice_answer_statement,
    _answer_statement_has_lexical_relation_lineage,
    _canonicalize_answer_statement_from_accepted_alias,
    _quoted_scope_preserved,
    _quoted_attribution_qa,
    _restore_immutable_quoted_slots,
    _original_interrogative_head_omitted,
    _participation_marker_preserved,
    _restore_authoritative_source_transpositions,
    _semantic_relation_and_scope_preserved,
    _possessive_name_answer_statement,
    _canonical_answer_is_explicit_compound,
    _called_relation_substitution_preserved,
    _clausal_canonical_relation_statement,
    _capitalized_identity_tokens,
    _deterministic_answer_slot_statement,
    _deterministic_question_paraphrase,
    SemanticPreservationError,
    LocalQwen35Paraphraser,
    load_resume_records,
    order_pending_sources_for_resume,
    partition_resume_records_for_semantic_repair,
    validate_resume_record_admission,
    parse_paraphrase_response,
    parse_answer_verification_response,
    parse_answer_repair_response,
    parse_question_repair_response,
    parse_synonym_repair_response,
    parse_verification_response,
)


def _source() -> TriviaQATrainSource:
    return TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_999",
        base_task_id="triviaqa:tc_999",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which river contains the Kariba Dam?",
        canonical_answer="Zambezi",
        native_split="train",
    )


def _record(
    *,
    question: str = "Name the river on which the Kariba Dam was built.",
    statement: str = "The Kariba Dam was built on the Zambezi river.",
) -> TriviaQAQAMemoryRecord:
    return TriviaQAQAMemoryRecord.create(
        source=_source(),
        paraphrase_question=question,
        paraphrase_answer_statement=statement,
        paraphrase_version="triviaqa.qa_memory.paraphrase.v5",
        paraphrase_method="semantic-preserving-question-and-answer-paraphrase",
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v5",
        generation_seed=20260827,
    )


def test_v5_materialization_keeps_relation_and_exact_canonical_span() -> None:
    record = _record()

    assert record.canonical_answer in record.paraphrase_answer_statement
    validate_qa_memory_against_sources((record,), (_source(),), require_complete=True)


@pytest.mark.parametrize(
    "statement",
    (
        "Zambezi",
        "Zambezi.",
        "The answer is Zambezi",
        "The canonical answer is Zambezi.",
        "Zambezi is the answer.",
    ),
)
def test_v5_materialization_rejects_answer_only_statement(statement: str) -> None:
    with pytest.raises(ValueError, match="express the question relation beyond"):
        _record(statement=statement)


def test_relation_bearing_statement_requires_context_beyond_canonical() -> None:
    assert relation_bearing_answer_statement(
        "The Kariba Dam was built on the Zambezi river.",
        "Zambezi",
    )
    assert not relation_bearing_answer_statement("Zambezi", "Zambezi")
    assert not relation_bearing_answer_statement(
        "The creator was Arthur.",
        "Art",
    )
    assert not exact_canonical_span_preserved("Arthur", "Art")
    assert exact_canonical_span_preserved("The medium was Art.", "Art")


def test_v7_resume_replays_current_deterministic_admission() -> None:
    drifted = _record(
        question="Name the river on which the Hoover Dam was built.",
    )
    with pytest.raises(ValueError, match="removed or replaced an original entity"):
        validate_resume_record_admission((drifted,), (_source(),))


def _semantic_source(original: str, canonical: str) -> TriviaQATrainSource:
    return TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_9999",
        base_task_id="triviaqa:tc_9999",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=original,
        canonical_answer=canonical,
        native_split="train",
    )


def _alias_source(
    *,
    original: str,
    canonical: str,
    accepted: tuple[str, ...],
) -> TriviaQATrainSource:
    return TriviaQATrainSource(
        source_train_task_id="triviaqa:alias_canonicalization",
        base_task_id="triviaqa:alias_canonicalization",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=original,
        canonical_answer=canonical,
        native_split="train",
        accepted_answers_for_admission=accepted,
    )


@pytest.mark.parametrize(
    ("source", "statement", "expected"),
    (
        (
            _alias_source(
                original="How many vice presidents did Franklin D Roosevelt have?",
                canonical="Three",
                accepted=("Three", "3", "three"),
            ),
            "The number of vice presidents Franklin D Roosevelt had was 3.",
            "The number of vice presidents Franklin D Roosevelt had was Three.",
        ),
        (
            _alias_source(
                original=(
                    "What is the name of the alley in which cartoon character "
                    "Top Cat lives?"
                ),
                canonical="Hoagy’s Alley",
                accepted=("Hoagy’s Alley",),
            ),
            "Top Cat lives in Hoagy's Alley.",
            "Top Cat lives in Hoagy’s Alley.",
        ),
        (
            _alias_source(
                original=(
                    "Dick Dudgeon and Reverend Anthony Anderson are characters "
                    "in which play by George Bernard Shaw?"
                ),
                canonical="THE DEVIL’S DISCIPLE",
                accepted=(
                    "THE DEVIL’S DISCIPLE",
                    "The Devil’s Disciple",
                    "Devil's Disciple",
                    "The Devil's Disciple",
                    "The Devils Disciple",
                ),
            ),
            "The play by George Bernard Shaw is The Devil’s Disciple.",
            "The play by George Bernard Shaw is THE DEVIL’S DISCIPLE.",
        ),
    ),
)
def test_accepted_alias_canonicalization_replaces_only_unique_longest_span(
    source: TriviaQATrainSource,
    statement: str,
    expected: str,
) -> None:
    assert _canonicalize_answer_statement_from_accepted_alias(
        source,
        statement,
    ) == expected


@pytest.mark.parametrize(
    ("source", "statement"),
    (
        (
            _alias_source(
                original="How many groups qualified?",
                canonical="Three",
                accepted=("Three", "3"),
            ),
            "There were 3 groups and 3 qualifying teams.",
        ),
        (
            _alias_source(
                original="Does 3 identify the number of qualifying teams?",
                canonical="Three",
                accepted=("Three", "3"),
            ),
            "The number of qualifying teams was 3.",
        ),
        (
            _alias_source(
                original="Is Hoagy’s Alley the place where Top Cat lives?",
                canonical="Top Cat's alley",
                accepted=("Top Cat's alley", "Hoagy's Alley"),
            ),
            "Top Cat lives in Hoagy's Alley.",
        ),
        (
            _alias_source(
                original="How many groups qualified?",
                canonical="Three",
                accepted=("Three", "3"),
            ),
            "The number of qualifying teams was a trio.",
        ),
        (
            _alias_source(
                original="Which item is requested?",
                canonical="Target",
                accepted=("Target", "the"),
            ),
            "The requested item is the.",
        ),
        (
            _alias_source(
                original="Which label identifies the target?",
                canonical="Target",
                accepted=("Target", "Alpha Beta", "Beta Gamma"),
            ),
            "The label is Alpha Beta Gamma.",
        ),
    ),
)
def test_accepted_alias_canonicalization_fails_closed(
    source: TriviaQATrainSource,
    statement: str,
) -> None:
    assert _canonicalize_answer_statement_from_accepted_alias(
        source,
        statement,
    ) is None


def test_answer_repair_prefers_verified_rejected_statement_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _alias_source(
        original="How many vice presidents did Franklin D Roosevelt have?",
        canonical="Three",
        accepted=("Three", "3", "three"),
    )
    client = object.__new__(LocalQwen35Paraphraser)
    verified: list[str] = []
    monkeypatch.setattr(
        client,
        "_complete",
        lambda **_: pytest.fail("accepted alias repair must precede model repair"),
    )
    monkeypatch.setattr(
        client,
        "_answer_statement_verified",
        lambda source, *, statement, seed: verified.append(statement) or True,
    )

    repaired = client._repair_answer_statement(
        source,
        question="What is the count of vice presidents Franklin D Roosevelt had?",
        rejected_statement=(
            "The number of vice presidents Franklin D Roosevelt had was 3."
        ),
        seed=41,
    )

    assert repaired == (
        "The number of vice presidents Franklin D Roosevelt had was Three."
    )
    assert verified == [repaired]


def test_answer_repair_canonicalizes_and_verifies_model_statement_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _alias_source(
        original="How many vice presidents did Franklin D Roosevelt have?",
        canonical="Three",
        accepted=("Three", "3", "three"),
    )
    client = object.__new__(LocalQwen35Paraphraser)
    verified: list[str] = []
    monkeypatch.setattr(
        client,
        "_complete",
        lambda **_: json.dumps(
            {
                "paraphrase_answer_statement": (
                    "The number of vice presidents Franklin D Roosevelt had was 3."
                )
            }
        ),
    )
    monkeypatch.setattr(
        client,
        "_answer_statement_verified",
        lambda source, *, statement, seed: verified.append(statement) or True,
    )

    repaired = client._repair_answer_statement(
        source,
        question="What is the count of vice presidents Franklin D Roosevelt had?",
        rejected_statement="No supported answer statement was produced.",
        seed=43,
    )

    assert repaired == (
        "The number of vice presidents Franklin D Roosevelt had was Three."
    )
    assert verified == [repaired]


@pytest.mark.parametrize(
    ("original", "canonical", "question", "statement", "failure"),
    (
        (
            "According to legend, who fired the arrow that hit Achilles in the heel?",
            "Paris",
            "According to legend, which projectile struck Achilles in the heel?",
            "Paris struck Achilles in the heel.",
            "answer-slot family",
        ),
        (
            "Who took the assumed name Sebastian Melmoth when living in Paris?",
            "Oscar Wilde",
            "What assumed name did the person in Paris take when it was Sebastian Melmoth?",
            "Oscar Wilde took the assumed name Sebastian Melmoth in Paris.",
            "answer-slot family",
        ),
        (
            "General Boris Gromov was the last Soviet soldier to leave where in 1989?",
            "Afghanistan",
            "In which location did General Boris Gromov, the last Soviet soldier to depart, go in 1989?",
            "General Boris Gromov was the last Soviet soldier to leave Afghanistan in 1989.",
            "source/destination",
        ),
        (
            "In 1985 Terry Waite returned to Beirut after securing the release of four British hostages where?",
            "Libya",
            "In 1985, after securing the release of four British hostages, where did Terry Waite return to Beirut?",
            "In 1985 Terry Waite returned to Beirut after securing the release of four British hostages in Libya.",
            "source/destination",
        ),
        (
            "In which country was Arnold Schwarzenegger born?",
            "Austria",
            "In what nation did Arnold Schwarzenegger originate?",
            "Arnold Schwarzenegger originated in Austria.",
            "birth-event",
        ),
        (
            "In which decade did the Jackson 5 sign to Motown?",
            "1960s",
            "In which decade did the Jackson 5 record for Motown?",
            "The Jackson 5 signed to Motown in the 1960s.",
            "recording-contract",
        ),
        (
            "Who was the defending champion when Virginia Wade won the Wimbledon singles?",
            "Chris Evert",
            "Who was the defending champion when Virginia Wade won the Wimbledon match?",
            "Chris Evert was the champion when Virginia Wade won the Wimbledon singles.",
            "singles-event",
        ),
        (
            "When did field hockey become an Olympic event for men?",
            "1908",
            "In what year did men's hockey become an Olympic event?",
            "Men's hockey became an Olympic event in 1908.",
            "field-hockey",
        ),
        (
            "What date is Father's Day?",
            "3rd Sunday in June",
            "On which day of the month does Father's Day fall?",
            "Father's Day occurs on the 3rd Sunday in June.",
            "calendar-date",
        ),
        (
            "In which country did General Jaruzelski impose marital law in 1981?",
            "Poland",
            "In which nation did General Jaruzelski enforce martial law in 1981?",
            "General Jaruzelski enforced martial law in Poland in 1981.",
            "source token",
        ),
    ),
)
def test_v12_semantic_gate_rejects_relation_and_scope_drift(
    original: str,
    canonical: str,
    question: str,
    statement: str,
    failure: str,
) -> None:
    with pytest.raises(SemanticPreservationError, match=failure):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": question,
                    "paraphrase_answer_statement": statement,
                }
            ),
            _semantic_source(original, canonical),
        )


@pytest.mark.parametrize(
    "question",
    (
        "Where was Ada Lovelace born?",
        "Who was born first, Susan Sarandon or Glenn Close?",
        "Which Oscar-winning actress was born on this date?",
        "In which country was Ursula Andress born?",
        "What year was Alan Turing born?",
    ),
)
def test_birth_target_detector_keeps_direct_birth_questions_strict(
    question: str,
) -> None:
    assert _birth_event_is_target_relation(question)


@pytest.mark.parametrize(
    "question",
    (
        (
            "The Pogues singer Shane MacGowan was born in 1957. Which female "
            "singer featured on their Christmas single?"
        ),
        (
            "James Todd Smith, born 1968, is an American rapper. What is he "
            "better known as?"
        ),
        (
            "When Achilles was born, his mother dipped him in the Styx. "
            "Where was he left vulnerable?"
        ),
    ),
)
def test_birth_target_detector_ignores_context_for_another_relation(
    question: str,
) -> None:
    assert not _birth_event_is_target_relation(question)


def test_contextual_birth_does_not_force_birth_into_answer_statement() -> None:
    _semantic_relation_and_scope_preserved(
        original_question=(
            "The Pogues singer Shane MacGowan was born in 1957. Which female "
            "singer featured on their Christmas single?"
        ),
        paraphrase_question=(
            "Born in 1957, Shane MacGowan led The Pogues; name the female "
            "vocalist who appeared on their Christmas single."
        ),
        paraphrase_answer_statement=(
            "Kirsty MacColl featured on The Pogues' Christmas single."
        ),
    )


def test_direct_birth_target_still_requires_birth_in_answer_statement() -> None:
    with pytest.raises(SemanticPreservationError, match="birth-event"):
        _semantic_relation_and_scope_preserved(
            original_question="Where was Ada Lovelace born?",
            paraphrase_question="In which place was Ada Lovelace born?",
            paraphrase_answer_statement="Ada Lovelace was in London.",
        )


@pytest.mark.parametrize(
    ("original", "canonical", "question", "statement"),
    (
        (
            "According to legend, who fired the arrow that hit Achilles in the heel?",
            "Paris",
            "According to legend, which person shot the arrow that struck Achilles in the heel?",
            "Paris fired the arrow that hit Achilles in the heel.",
        ),
        (
            "General Boris Gromov was the last Soviet soldier to leave where in 1989?",
            "Afghanistan",
            "Which country did General Boris Gromov, the last Soviet soldier, depart from in 1989?",
            "General Boris Gromov was the last Soviet soldier to leave Afghanistan in 1989.",
        ),
        (
            "In 1985 Terry Waite returned to Beirut after securing the release of four British hostages where?",
            "Libya",
            "In 1985, in which country did Terry Waite secure the release of four British hostages before returning to Beirut?",
            "In 1985 Terry Waite returned to Beirut after securing the release of four British hostages in Libya.",
        ),
        (
            "In which country was Arnold Schwarzenegger born?",
            "Austria",
            "In what nation was Arnold Schwarzenegger brought into the world?",
            "Arnold Schwarzenegger was born in Austria.",
        ),
        (
            "In which decade did the Jackson 5 sign to Motown?",
            "1960s",
            "During which decade did the Jackson 5 enter a contract with Motown?",
            "The Jackson 5 signed to Motown in the 1960s.",
        ),
        (
            "Who was the defending champion when Virginia Wade won the Wimbledon singles?",
            "Chris Evert",
            "Who held the title when Virginia Wade won the Wimbledon singles championship?",
            "Chris Evert was the champion when Virginia Wade won the Wimbledon singles.",
        ),
        (
            "When did field hockey become an Olympic event for men?",
            "1908",
            "In what year did men's field hockey enter the Olympic program?",
            "Men's field hockey became an Olympic event in 1908.",
        ),
        (
            "What date is Father's Day?",
            "3rd Sunday in June",
            "On which date is Father's Day observed?",
            "Father's Day occurs on the 3rd Sunday in June.",
        ),
        (
            "In which country did General Jaruzelski impose marital law in 1981?",
            "Poland",
            "In which nation did General Jaruzelski enforce marital law during 1981?",
            "General Jaruzelski imposed marital law in Poland in 1981.",
        ),
    ),
)
def test_v12_semantic_gate_accepts_relation_preserving_paraphrases(
    original: str,
    canonical: str,
    question: str,
    statement: str,
) -> None:
    assert parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": question,
                "paraphrase_answer_statement": statement,
            }
        ),
        _semantic_source(original, canonical),
    ) == (question, statement)


def test_v13_literal_authority_repair_restores_only_source_transpositions() -> None:
    original = (
        "In which country did General Jaruzelski impose marital law in 1981?"
    )

    assert _restore_authoritative_source_transpositions(
        original,
        "In what nation did General Jaruzelski impose martial law in 1981?",
    ) == "In what nation did General Jaruzelski impose marital law in 1981?"
    assert _restore_authoritative_source_transpositions(
        original,
        "General Jaruzelski imposed martial law in Poland in 1981.",
    ) == "General Jaruzelski imposed marital law in Poland in 1981."
    assert _restore_authoritative_source_transpositions(
        original,
        "In what nation did General Jaruzelski enact emergency law in 1981?",
    ) == "In what nation did General Jaruzelski enact emergency law in 1981?"


def test_v12_resume_partitions_only_new_semantic_gate_failures() -> None:
    valid = _record()
    drift_source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_9998",
        base_task_id="triviaqa:tc_9998",
        selection_index=1,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "According to legend, who fired the arrow that hit Achilles?"
        ),
        canonical_answer="Paris",
        native_split="train",
    )
    drifted = TriviaQAQAMemoryRecord.create(
        source=drift_source,
        paraphrase_question=(
            "According to legend, which projectile struck Achilles?"
        ),
        paraphrase_answer_statement="Paris struck Achilles.",
        paraphrase_version="triviaqa.qa_memory.paraphrase.v12",
        paraphrase_method="semantic-preserving-question-and-answer-paraphrase",
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v12",
        generation_seed=20260828,
    )

    accepted, repair_source_ids = partition_resume_records_for_semantic_repair(
        (valid, drifted),
        (_source(), drift_source),
    )

    assert accepted == (valid,)
    assert repair_source_ids == ("triviaqa:tc_9998",)


def test_resume_partitions_newly_detected_curly_quote_drift() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:curly_quote_resume",
        base_task_id="triviaqa:curly_quote_resume",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Who recorded ‘Blue Moon’?",
        canonical_answer="Example Singer",
        native_split="train",
    )
    drifted = TriviaQAQAMemoryRecord.create(
        source=source,
        paraphrase_question="Name the performer who released 'Blue moon'.",
        paraphrase_answer_statement=(
            "Example Singer recorded Blue Moon."
        ),
        paraphrase_version="triviaqa.qa_memory.paraphrase.v12",
        paraphrase_method=(
            "semantic-preserving-question-and-answer-paraphrase"
        ),
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v12",
        generation_seed=20260828,
    )

    accepted, repair_source_ids = partition_resume_records_for_semantic_repair(
        (drifted,),
        (source,),
    )

    assert accepted == ()
    assert repair_source_ids == ("triviaqa:curly_quote_resume",)


def test_resume_orders_untouched_tail_before_admission_gaps() -> None:
    sources = tuple(
        TriviaQATrainSource(
            source_train_task_id=f"triviaqa:resume_{index}",
            base_task_id=f"triviaqa:resume_{index}",
            selection_index=index,
            cycled_training_sample=False,
            cycle_index=None,
            original_question=f"Which river is source {index}?",
            canonical_answer=f"River {index}",
            native_split="train",
        )
        for index in range(5)
    )

    def admitted(source: TriviaQATrainSource) -> TriviaQAQAMemoryRecord:
        return TriviaQAQAMemoryRecord.create(
            source=source,
            paraphrase_question=(
                f"Name the river associated with source {source.selection_index}."
            ),
            paraphrase_answer_statement=(
                f"The river associated with source {source.selection_index} is "
                f"{source.canonical_answer}."
            ),
            paraphrase_version="triviaqa.qa_memory.paraphrase.v12",
            paraphrase_method=(
                "semantic-preserving-question-and-answer-paraphrase"
            ),
            generator_provider="local-openai-compatible",
            model_id="supervisor_theta",
            model_revision="Qwen3.5-9B-local",
            prompt_template_version=(
                "triviaqa.qa_memory.qa_paraphrase.v12"
            ),
            generation_seed=20260828 + source.selection_index,
        )

    pending = order_pending_sources_for_resume(
        sources,
        (admitted(sources[0]), admitted(sources[2])),
    )

    assert tuple(source.selection_index for source in pending) == (3, 4, 1)


def test_v12_resume_partition_keeps_older_admission_errors_fail_closed() -> None:
    drifted_entity = _record(
        question="Name the river on which the Hoover Dam was built.",
    )

    with pytest.raises(ValueError, match="removed or replaced an original entity"):
        partition_resume_records_for_semantic_repair(
            (drifted_entity,),
            (_source(),),
        )


def test_response_parser_rejects_bare_canonical_answer() -> None:
    response = json.dumps(
        {
            "paraphrase_question": "Name the river holding the Kariba Dam.",
            "paraphrase_answer_statement": "Zambezi",
        }
    )

    with pytest.raises(ValueError, match="express the question relation beyond"):
        parse_paraphrase_response(response, _source())


def test_response_parser_normalizes_observed_qwen_leading_dot_key() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1045",
        base_task_id="triviaqa:tc_1045",
        selection_index=471,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which Gloria co-founded Ms magazine?",
        canonical_answer="Steinem",
        native_split="train",
        accepted_answers_for_admission=("Steinem", "Gloria Steinem"),
    )
    response = json.dumps(
        {
            ".paraphrase_question": "Which Gloria helped establish Ms magazine?",
            "paraphrase_answer_statement": "Steinem co-founded Ms magazine.",
        }
    )

    question, statement = parse_paraphrase_response(response, source)

    assert question == "Which Gloria helped establish Ms magazine?"
    assert statement == "Steinem co-founded Ms magazine."


@pytest.mark.parametrize(
    ("question_key", "statement_key"),
    (
        ("parphrase_question", "paraphrase_answer_statement"),
        ("paraphrase_question", ".paraphrase_answer_statement"),
        (".paraphrase_question", ".paraphrase_answer_statement"),
    ),
)
def test_response_parser_normalizes_known_typos_one_to_one(
    question_key: str,
    statement_key: str,
) -> None:
    response = json.dumps(
        {
            question_key: "Name the river holding the Kariba Dam.",
            statement_key: "The Kariba Dam was built on the Zambezi river.",
        }
    )

    question, statement = parse_paraphrase_response(response, _source())

    assert question == "Name the river holding the Kariba Dam."
    assert statement.endswith("Zambezi river.")


def test_structured_output_error_reports_missing_and_unknown_fields() -> None:
    with pytest.raises(ValueError) as captured:
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrased_question": (
                        "Name the river holding the Kariba Dam."
                    )
                }
            ),
            _source(),
        )

    message = str(captured.value)
    assert "observed_fields=['paraphrased_question']" in message
    assert (
        "missing_fields=['paraphrase_answer_statement', "
        "'paraphrase_question']" in message
    )
    assert "unexpected_fields=['paraphrased_question']" in message
    assert "collision_fields=[]" in message


def test_structured_output_error_reports_alias_collision() -> None:
    with pytest.raises(ValueError) as captured:
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": (
                        "Name the river holding the Kariba Dam."
                    ),
                    ".paraphrase_question": (
                        "Identify the waterway holding the Kariba Dam."
                    ),
                    "paraphrase_answer_statement": (
                        "The Kariba Dam was built on the Zambezi river."
                    ),
                }
            ),
            _source(),
        )

    message = str(captured.value)
    assert "collision_fields=['paraphrase_question']" in message
    assert "unexpected_fields=[]" in message


def test_generate_repairs_schema_failure_with_existing_repair_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    client = object.__new__(LocalQwen35Paraphraser)
    client.max_retries = 0
    calls: list[str] = []
    responses = iter(
        (
            json.dumps(
                {
                    source.original_question: "invalid question-key payload",
                    "paraphrase_answer_statement": (
                        "The Kariba Dam was built on the Zambezi river."
                    ),
                }
            ),
            json.dumps(
                {
                    "semantic_preserved": True,
                    "entity_identity_preserved": True,
                    "relation_and_scope_preserved": True,
                    "answer_cardinality_preserved": True,
                    "answer_not_revealed": True,
                    "question_changed": True,
                }
            ),
        )
    )
    monkeypatch.setattr(client, "_complete", lambda **_: next(responses))

    def repair_question(*args: object, **kwargs: object) -> str:
        calls.append("question")
        return "Name the river holding the Kariba Dam."

    def repair_statement(*args: object, **kwargs: object) -> str:
        calls.append("statement")
        return "The Kariba Dam was built on the Zambezi river."

    monkeypatch.setattr(client, "_repair_paraphrase_question", repair_question)
    monkeypatch.setattr(client, "_repair_answer_statement", repair_statement)
    monkeypatch.setattr(
        client,
        "_answer_statement_verified",
        lambda *_, **__: True,
    )

    assert client.generate(source, seed=31) == (
        "Name the river holding the Kariba Dam.",
        "The Kariba Dam was built on the Zambezi river.",
        31,
    )
    assert calls == ["question", "statement"]


@pytest.mark.parametrize(
    "fields",
    (
        {
            "..paraphrase_question": "Which Gloria helped establish Ms magazine?",
            "paraphrase_answer_statement": "Steinem co-founded Ms magazine.",
        },
        {
            ".paraphrase_question": "Which Gloria helped establish Ms magazine?",
            "paraphrase_question": "Which Gloria helped establish Ms magazine?",
            "paraphrase_answer_statement": "Steinem co-founded Ms magazine.",
        },
        {
            "parphrase_question": "Name the river holding the Kariba Dam.",
            "paraphrase_question": "Name the river holding the Kariba Dam.",
            "paraphrase_answer_statement": (
                "The Kariba Dam was built on the Zambezi river."
            ),
        },
        {
            "paraphrase_question": "Name the river holding the Kariba Dam.",
            ".paraphrase_answer_statement": (
                "The Kariba Dam was built on the Zambezi river."
            ),
            "paraphrase_answer_statement": (
                "The Kariba Dam was built on the Zambezi river."
            ),
        },
        {
            "paraphrase_question": "Name the river holding the Kariba Dam.",
            "paraphrase_answer_statement": (
                "The Kariba Dam was built on the Zambezi river."
            ),
            "extra": True,
        },
    ),
)
def test_response_parser_rejects_unobserved_key_variants(fields: object) -> None:
    with pytest.raises(ValueError, match="fields are incompatible"):
        parse_paraphrase_response(json.dumps(fields), _source())


def test_v10_leading_answer_slot_anchor_stays_bound_to_request() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1045",
        base_task_id="triviaqa:tc_1045",
        selection_index=471,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which Gloria co-founded Ms magazine?",
        canonical_answer="Steinem",
        native_split="train",
        accepted_answers_for_admission=("Steinem", "Gloria Steinem"),
    )

    assert _leading_answer_slot_anchor(source) == "Gloria"
    assert _literal_subject_wh_answer_statement(source) == (
        "Steinem co-founded Ms magazine."
    )
    assert _leading_answer_slot_anchor_preserved(
        source,
        "Which Gloria helped establish Ms magazine?",
    )
    assert _leading_answer_slot_anchor_preserved(
        source,
        "Identify the Gloria who helped establish Ms magazine.",
    )
    assert not _leading_answer_slot_anchor_preserved(
        source,
        "Who co-founded Ms magazine alongside Gloria?",
    )
    assert not _leading_answer_slot_anchor_preserved(
        source,
        "Which person did Gloria co-found Ms magazine with?",
    )
    assert _participation_marker_preserved(
        source.original_question,
        "Which Gloria helped establish Ms magazine?",
    )
    assert not _participation_marker_preserved(
        source.original_question,
        "Which Gloria established Ms magazine?",
    )


def test_v10_leading_answer_slot_binding_rejects_relation_reversal() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1045",
        base_task_id="triviaqa:tc_1045",
        selection_index=471,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which Gloria co-founded Ms magazine?",
        canonical_answer="Steinem",
        native_split="train",
        accepted_answers_for_admission=("Steinem", "Gloria Steinem"),
    )
    response = json.dumps(
        {
            "paraphrase_question": "Who co-founded Ms magazine alongside Gloria?",
            "paraphrase_answer_statement": (
                "Gloria co-founded Ms magazine with Steinem."
            ),
        }
    )

    with pytest.raises(ValueError, match="answer-slot anchor"):
        parse_paraphrase_response(response, source)


def test_v10_lowercase_answer_type_has_no_capitalized_anchor() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1",
        base_task_id="triviaqa:tc_1",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which country does Gulf Air come from?",
        canonical_answer="Bahrain",
        native_split="train",
        accepted_answers_for_admission=("Bahrain",),
    )

    assert _leading_answer_slot_anchor(source) is None


def test_v10_answer_repair_normalizes_only_observed_leading_dot_key() -> None:
    assert parse_answer_repair_response(
        json.dumps(
            {".paraphrase_answer_statement": "Steinem co-founded Ms magazine."}
        )
    ) == "Steinem co-founded Ms magazine."
    with pytest.raises(ValueError, match="fields are incompatible"):
        parse_answer_repair_response(
            json.dumps(
                {
                    "..paraphrase_answer_statement": (
                        "Steinem co-founded Ms magazine."
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="fields are incompatible"):
        parse_answer_repair_response(
            json.dumps(
                {
                    ".paraphrase_answer_statement": (
                        "Steinem co-founded Ms magazine."
                    ),
                    "paraphrase_answer_statement": (
                        "Steinem co-founded Ms magazine."
                    ),
                }
            )
        )


def test_v10_possessive_name_binding_rejects_dangling_possessive() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_624",
        base_task_id="triviaqa:tc_624",
        selection_index=198,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What is Iggy Pop's real name?",
        canonical_answer="James Osterberg",
        native_split="train",
    )
    assert _possessive_name_answer_statement(source) == (
        "Iggy Pop's",
        "Iggy Pop's real name is James Osterberg.",
    )
    with pytest.raises(ValueError, match="dangling possessive"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": "What is Iggy Pop's actual name?",
                    "paraphrase_answer_statement": (
                        "The actual name of Iggy Pop's is James Osterberg."
                    ),
                }
            ),
            source,
        )
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": "What is Iggy Pop's actual name?",
                "paraphrase_answer_statement": (
                    "Iggy Pop's real name is James Osterberg."
                ),
            }
        ),
        source,
    )
    assert question == "What is Iggy Pop's actual name?"
    assert statement == "Iggy Pop's real name is James Osterberg."


def test_v10_clausal_canonical_binds_to_original_answer_slot() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_774",
        base_task_id="triviaqa:tc_774",
        selection_index=291,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What disability did singer Al Hibbler have?",
        canonical_answer="He was blind",
        native_split="train",
    )
    statement = _clausal_canonical_relation_statement(source)
    assert statement == (
        'The statement "He was blind" identifies the disability that singer '
        "Al Hibbler had."
    )
    question, admitted_statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "What impairment did singer Al Hibbler suffer from?"
                ),
                "paraphrase_answer_statement": statement,
            }
        ),
        source,
    )
    assert question.startswith("What impairment")
    assert admitted_statement == statement


def test_v10_quoted_attribution_preserves_dataset_quote_exactly() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_949",
        base_task_id="triviaqa:tc_949",
        selection_index=409,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            '"Who said, ""It is better to die on your feet than live on your '
            'knees?"""'
        ),
        canonical_answer="Emiliano Zapata",
        native_split="train",
    )
    repaired = _quoted_attribution_qa(source)
    assert repaired is not None
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": repaired[0],
                "paraphrase_answer_statement": repaired[1],
            }
        ),
        source,
    )
    assert question == (
        'Identify the person who said, "It is better to die on your feet than '
        'live on your knees?"'
    )
    assert statement.startswith("Emiliano Zapata said")


def test_v10_quoted_attribution_preserves_explicit_answer_type() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_775",
        base_task_id="triviaqa:tc_775",
        selection_index=292,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            '"Which writer said, "" An atheist is a man who has no invisible '
            'means of support?"""'
        ),
        canonical_answer="John Buchan",
        native_split="train",
    )
    repaired = _quoted_attribution_qa(source)
    assert repaired is not None
    assert repaired[0].startswith("Identify the writer who said")
    parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": repaired[0],
                "paraphrase_answer_statement": repaired[1],
            }
        ),
        source,
    )


def test_quoted_scope_supports_curly_quotes_and_preserves_exact_content() -> None:
    assert _quoted_scope_preserved(
        "Who recorded ‘Blue Moon’?",
        "Name the performer who released 'Blue Moon'.",
    )
    assert _quoted_scope_preserved(
        "Who wrote “The Left Hand”?",
        'Name the author of "The Left Hand".',
    )
    assert not _quoted_scope_preserved(
        "Who recorded ‘Blue Moon’?",
        "Name the performer who released 'Blue moon'.",
    )
    assert not _quoted_scope_preserved(
        "Who wrote “The Left Hand”?",
        'Name the author of "The Right Hand".',
    )


def test_parser_enforces_exact_curly_quoted_content() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:curly_quote",
        base_task_id="triviaqa:curly_quote",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Who recorded ‘Blue Moon’?",
        canonical_answer="Example Singer",
        native_split="train",
    )
    accepted = {
        "paraphrase_question": (
            "Name the performer who released 'Blue Moon'."
        ),
        "paraphrase_answer_statement": (
            "Example Singer recorded Blue Moon."
        ),
    }

    parse_paraphrase_response(json.dumps(accepted), source)
    with pytest.raises(ValueError, match="invented quoted content"):
        parse_paraphrase_response(
            json.dumps(
                {
                    **accepted,
                    "paraphrase_question": (
                        "Name the performer who released 'Blue moon'."
                    ),
                }
            ),
            source,
        )


def test_quote_slot_recovery_restores_one_slot_without_rewriting_its_frame() -> None:
    source = 'Which singer recorded “Blue Moon”?'
    candidate = 'Name the performer who released "Blue Moon!".'

    assert _restore_immutable_quoted_slots(source, candidate) == (
        'Name the performer who released "Blue Moon".'
    )


def test_quote_slot_recovery_restores_multiple_ordered_unicode_slots() -> None:
    source = 'Who recorded "Blue Moon" and ‘Red Sun’?'
    candidate = 'Name who released “Blue Moon!” and ‘Red Sun?’.'

    assert _restore_immutable_quoted_slots(source, candidate) == (
        'Name who released “Blue Moon” and ‘Red Sun’.'
    )


def test_generate_routes_quote_only_failure_through_slot_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:quote_repair_generate",
        base_task_id="triviaqa:quote_repair_generate",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question='Which singer recorded “Blue Moon”?',
        canonical_answer="Example Singer",
        native_split="train",
    )
    client = object.__new__(LocalQwen35Paraphraser)
    client.max_retries = 0
    responses = iter(
        (
            json.dumps(
                {
                    "paraphrase_question": (
                        'Name the performer who released "Blue Moon!".'
                    ),
                    "paraphrase_answer_statement": (
                        "Example Singer recorded Blue Moon."
                    ),
                }
            ),
            json.dumps(
                {
                    "semantic_preserved": True,
                    "entity_identity_preserved": True,
                    "relation_and_scope_preserved": True,
                    "answer_cardinality_preserved": True,
                    "answer_not_revealed": True,
                    "question_changed": True,
                }
            ),
        )
    )
    monkeypatch.setattr(client, "_complete", lambda **_: next(responses))
    monkeypatch.setattr(
        client,
        "_answer_statement_verified",
        lambda *_, **__: True,
    )

    assert client.generate(source, seed=17) == (
        'Name the performer who released "Blue Moon".',
        "Example Singer recorded Blue Moon.",
        17,
    )


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            'Who recorded "Blue Moon" and "Red Sun"?',
            'Name who released "Blue Moon!".',
        ),
        (
            'Who recorded "Blue Moon"?',
            'Name who released "Blue Moon!" and "Red Sun".',
        ),
        (
            'Who recorded "Blue Moon" and "Red Sun"?',
            'Name who released "Red Sun" and "Blue Moon".',
        ),
        (
            'Who recorded "Blue Moon" and "Red Sun"?',
            'Name who released "Green Hill" and "Black Lake".',
        ),
    ],
    ids=("deleted-slot", "added-slot", "reordered-slots", "unalignable-slots"),
)
def test_quote_slot_recovery_fails_closed_for_ambiguous_slot_layouts(
    source: str,
    candidate: str,
) -> None:
    assert _restore_immutable_quoted_slots(source, candidate) is None


def test_quote_slot_recovery_still_runs_all_later_admission_gates() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:quote_repair_other_gate",
        base_task_id="triviaqa:quote_repair_other_gate",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question='Who recorded "Blue Moon"?',
        canonical_answer="Example Singer",
        native_split="train",
    )
    candidate_question = 'Who performed "Blue Moon!" and who produced it?'
    repaired_question = _restore_immutable_quoted_slots(
        source.original_question,
        candidate_question,
    )

    assert repaired_question == 'Who performed "Blue Moon" and who produced it?'
    with pytest.raises(ValueError, match="requested answer cardinality"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": repaired_question,
                    "paraphrase_answer_statement": (
                        "Example Singer recorded Blue Moon."
                    ),
                }
            ),
            source,
        )


def test_v5_parser_rejects_word_order_only_rewrite() -> None:
    response = json.dumps(
        {
            "paraphrase_question": "The Kariba Dam contains which river?",
            "paraphrase_answer_statement": (
                "The Kariba Dam was built on the Zambezi river."
            ),
        }
    )

    with pytest.raises(ValueError, match="lexical or phrase replacement"):
        parse_paraphrase_response(response, _source())


def test_v5_lexical_replacement_requires_removed_and_added_content() -> None:
    assert _has_lexical_or_phrase_replacement(
        "Which river contains the Kariba Dam?",
        "Name the waterway on which the Kariba Dam was built.",
    )
    assert not _has_lexical_or_phrase_replacement(
        "Which river contains the Kariba Dam?",
        "The Kariba Dam contains which river?",
    )
    assert _has_lexical_or_phrase_replacement(
        (
            'According to Baron de Coubertin, "The essential thing is not '
            'conquering but..." what?'
        ),
        (
            'According to Baron de Coubertin, "The essential thing is not '
            'conquering but..." what is the required action?'
        ),
    )


def test_v5_relation_phrase_can_replace_copula() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_230",
        base_task_id="triviaqa:tc_230",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What is the capital of Kenya?",
        canonical_answer="Nairobi",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Which city serves as the capital of Kenya?"
            ),
            "paraphrase_answer_statement": (
                "Nairobi is the capital of Kenya."
            ),
        }
    )

    question, statement = parse_paraphrase_response(response, source)

    assert question == "Which city serves as the capital of Kenya?"
    assert statement == "Nairobi is the capital of Kenya."


def test_v5_csv_quoted_lyric_preserves_semantic_inner_quote() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_247",
        base_task_id="triviaqa:tc_247",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            '"Which song say, ""The words of the prophet are written on the '
            'subway walls?"""'
        ),
        canonical_answer="Sound of Silence",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                'Which song says, "The words of the prophet are written on '
                'the subway walls?"'
            ),
            "paraphrase_answer_statement": (
                'Sound of Silence is the song that says, "The words of the '
                'prophet are written on the subway walls?"'
            ),
        }
    )

    question, _ = parse_paraphrase_response(response, source)

    assert question.startswith("Which song says")


def test_v8_parser_rejects_added_numeric_or_quoted_scope() -> None:
    numeric_source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_997",
        base_task_id="triviaqa:tc_997",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which artist won in 1999?",
        canonical_answer="Example Artist",
        native_split="train",
    )
    with pytest.raises(ValueError, match="added, removed, or changed a numeric"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": (
                        "Name the artist who won between 1999 and 2000."
                    ),
                    "paraphrase_answer_statement": (
                        "Example Artist was the artist who won in 1999."
                    ),
                }
            ),
            numeric_source,
        )

    quoted_source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_998",
        base_task_id="triviaqa:tc_998",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question='Who recorded "Example Song"?',
        canonical_answer="Example Singer",
        native_split="train",
    )
    with pytest.raises(ValueError, match="invented quoted content"):
        parse_paraphrase_response(
            json.dumps(
                {
                        "paraphrase_question": (
                            'Name the performer who performed "example song".'
                        ),
                    "paraphrase_answer_statement": (
                        'Example Singer recorded "Example Song".'
                    ),
                }
            ),
            quoted_source,
        )
    assert _quoted_scope_preserved(
        "Who had 70s hits with Have You Seen Her and Oh Girl?",
        "Which group recorded 'Have You Seen Her' and 'Oh Girl'?",
    )
    assert _quoted_scope_preserved(
        'Which word was omitted from Mario Puzo\'s novel, "The Godfather"?',
        'Name the term absent from Mario Puzo\'s book, "The Godfather".',
    )


def test_v5_parser_rejects_answer_bearing_entity_substitution() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_527",
        base_task_id="triviaqa:tc_527",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Who sang with Crosby, Stills and Young?",
        canonical_answer="Graham Nash",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Which musician performed alongside Crosby, Stills, and Nash?"
            ),
            "paraphrase_answer_statement": (
                "Graham Nash sang with Crosby, Stills and Young."
            ),
        }
    )

    with pytest.raises(ValueError, match="entity token|canonical tokens"):
        parse_paraphrase_response(response, source)


def test_v5_parser_accepts_possessive_entity_inflection_without_answer_leakage() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_498",
        base_task_id="triviaqa:tc_498",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Who wrote the story upon which Alfred Hitchcock based his 1963 "
            "suspense film The Birds?"
        ),
        canonical_answer="Daphne du Maurier, best known for Rebecca",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Identify the author of the narrative that served as the basis "
                "for Alfred Hitchcock's 1963 suspense film The Birds."
            ),
            "paraphrase_answer_statement": (
                "Daphne du Maurier, best known for Rebecca wrote the story on "
                "which Alfred Hitchcock based his 1963 suspense film The Birds."
            ),
        }
    )

    question, statement = parse_paraphrase_response(response, source)

    assert "Hitchcock's" in question
    assert source.canonical_answer in statement
    assert "Rebecca" not in question


@pytest.mark.parametrize(
    "curly_possessive",
    (
        "liverpool’s",
        "women’s",
        "children’s",
        "sony’s",
        "mozart’s",
    ),
)
def test_identity_token_allows_only_possessive_apostrophe_orthography(
    curly_possessive: str,
) -> None:
    base = curly_possessive[:-2]
    straight_possessive = f"{base}'s"

    assert _identity_token_preserved(
        curly_possessive,
        frozenset({straight_possessive}),
    )
    assert _identity_token_preserved(
        straight_possessive,
        frozenset({curly_possessive}),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset(),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset({base}),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset({f"{base}s"}),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset({f"{base}'d"}),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset({f"{base}’d"}),
    )
    assert not _identity_token_preserved(
        curly_possessive,
        frozenset({f"{base[:-1]}x's"}),
    )


def test_parser_accepts_sony_possessive_apostrophe_orthography_only() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:sfq_24084",
        base_task_id="triviaqa:sfq_24084",
        selection_index=55634,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What was the name of Sony’s first game console?",
        canonical_answer="Play Station",
        native_split="train",
    )

    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "What designation was given to Sony's first game console?"
                ),
                "paraphrase_answer_statement": (
                    "The name of Sony’s first game console was Play Station."
                ),
            }
        ),
        source,
    )

    assert question == "What designation was given to Sony's first game console?"
    assert statement.endswith("Play Station.")


def test_v5_parser_does_not_remove_possessive_from_lexicalized_entity() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1498",
        base_task_id="triviaqa:tc_1498",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Who established the McDonald's restaurant chain?",
        canonical_answer="Ray Kroc",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Which person founded the McDonald restaurant chain?"
            ),
            "paraphrase_answer_statement": (
                "Ray Kroc established the McDonald's restaurant chain."
            ),
        }
    )

    with pytest.raises(ValueError, match="original entity token"):
        parse_paraphrase_response(response, source)


def test_v8_parser_allows_multiword_person_possessive_to_of_construction() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_624",
        base_task_id="triviaqa:tc_624",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What is Iggy Pop's real name?",
        canonical_answer="James Osterberg",
        native_split="train",
    )
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": "What is the actual name of Iggy Pop?",
                "paraphrase_answer_statement": (
                    "The real name of Iggy Pop is James Osterberg."
                ),
            }
        ),
        source,
    )
    assert question.endswith("of Iggy Pop?")
    assert statement.endswith("James Osterberg.")


def test_v9_parser_preserves_lowercase_particle_inside_named_entity() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_358",
        base_task_id="triviaqa:tc_358",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            'What is the actual title of Leonardo da Vinci\'s "Mona Lisa"?'
        ),
        canonical_answer="La Gioconda",
        native_split="train",
    )
    with pytest.raises(ValueError, match="removed or replaced an original entity"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": (
                        'What is the real name of Leonardo Vinci\'s "Mona Lisa"?'
                    ),
                    "paraphrase_answer_statement": (
                        'The actual title of Leonardo da Vinci\'s "Mona Lisa" is '
                        "La Gioconda."
                    ),
                }
            ),
            source,
        )
    question, _ = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    'What is the real name of Leonardo da Vinci\'s "Mona Lisa"?'
                ),
                "paraphrase_answer_statement": (
                    'The actual title of Leonardo da Vinci\'s "Mona Lisa" is '
                    "La Gioconda."
                ),
            }
        ),
        source,
    )
    assert "Leonardo da Vinci's" in question


def test_v5_parser_rejects_second_answer_slot() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_966",
        base_task_id="triviaqa:tc_966",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What was the first film made by Studio Example?",
        canonical_answer="Example Film",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Which movie did Studio Example make first, and who directed it?"
            ),
            "paraphrase_answer_statement": (
                "Example Film was the first movie made by Studio Example."
            ),
        }
    )

    with pytest.raises(ValueError, match="answer cardinality"):
        parse_paraphrase_response(response, source)


def test_v5_known_entity_and_single_which_remains_one_answer_slot() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_244",
        base_task_id="triviaqa:tc_244",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "The Zambesi and which other river define the borders of "
            "Matabeleland?"
        ),
        canonical_answer="Limpopo",
        native_split="train",
    )
    response = json.dumps(
        {
            "paraphrase_question": (
                "Which river, alongside the Zambesi, delineates the "
                "boundaries of Matabeleland?"
            ),
            "paraphrase_answer_statement": (
                "Limpopo is the other river that defines the borders of "
                "Matabeleland alongside the Zambesi."
            ),
        }
    )

    question, _ = parse_paraphrase_response(response, source)

    assert question.startswith("Which river")


def test_v5_verifier_requires_exact_boolean_contract() -> None:
    accepted = {
        "semantic_preserved": True,
        "entity_identity_preserved": True,
        "relation_and_scope_preserved": True,
        "answer_cardinality_preserved": True,
        "answer_not_revealed": True,
        "question_changed": True,
    }

    assert parse_verification_response(json.dumps(accepted)) == accepted
    with pytest.raises(ValueError, match="fields are incompatible"):
        parse_verification_response(
            json.dumps({**accepted, "extra": True})
        )


def test_v5_verifier_treats_original_dataset_string_as_authoritative() -> None:
    assert "authoritative dataset string" in VERIFICATION_SYSTEM_PROMPT
    assert "Do not correct the dataset" in VERIFICATION_SYSTEM_PROMPT
    payload = json.loads(build_verification_messages(
        _source(),
        paraphrase_question="Name the waterway holding the Kariba Dam.",
    )[1]["content"])
    assert "canonical_training_answer" not in payload
    assert payload["canonical_answer_leakage_checked_deterministically"] is True
    assert payload["original_interrogative_head_omitted"] is False


def test_v8_malformed_interrogative_head_flag_is_narrow() -> None:
    malformed = "In which John Logie Baird invent television?"
    explicit_type = "Richard Nixon was Vice President to which US state?"
    assert _original_interrogative_head_omitted(malformed)
    assert not _original_interrogative_head_omitted(explicit_type)
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_528",
        base_task_id="triviaqa:tc_528",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=malformed,
        canonical_answer="1920s",
        native_split="train",
    )
    generation_payload = json.loads(
        build_paraphrase_messages(source)[1]["content"]
    )
    verification_payload = json.loads(
        build_verification_messages(
            source,
            paraphrase_question=(
                "During what decade did John Logie Baird invent television?"
            ),
        )[1]["content"]
    )
    assert generation_payload["original_interrogative_head_omitted"] is True
    assert verification_payload["original_interrogative_head_omitted"] is True


def test_v8_compound_answer_flag_preserves_one_grouped_slot() -> None:
    assert _canonical_answer_is_explicit_compound("Australia & New Zealand")
    assert _canonical_answer_is_explicit_compound("Trinidad and Tobago")
    assert not _canonical_answer_is_explicit_compound("Dwight Eisenhower")
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_677",
        base_task_id="triviaqa:tc_677",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Who signed the Pacific Security Treaty with the USA in 1951?"
        ),
        canonical_answer="Australia & New Zealand",
        native_split="train",
    )
    payload = json.loads(
        build_verification_messages(
            source,
            paraphrase_question=(
                "Which nations signed the Pacific Security Treaty with the USA "
                "in 1951?"
            ),
        )[1]["content"]
    )
    assert payload["canonical_answer_is_explicit_compound"] is True


def test_v8_generation_payload_marks_answer_tokens_forbidden_in_question() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_265",
        base_task_id="triviaqa:tc_265",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What does MG sand for in Booker T & The MG's?",
        canonical_answer="Memphis Group",
        native_split="train",
    )
    payload = json.loads(build_paraphrase_messages(source)[1]["content"])
    assert payload["forbidden_question_canonical_tokens"] == ["group", "memphis"]


def test_v9_generation_payload_names_eligible_lexical_replacements() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_400",
        base_task_id="triviaqa:tc_400",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What was art-world guru Andy Warhol's name at birth?",
        canonical_answer="Andrew Warhola",
        native_split="train",
    )
    payload = json.loads(build_paraphrase_messages(source)[1]["content"])
    assert "guru" in payload["lexical_replacement_source_tokens"]
    assert "andy" not in payload["lexical_replacement_source_tokens"]
    assert "warhol's" not in payload["lexical_replacement_source_tokens"]
    repair_payload = json.loads(
        build_question_repair_messages(
            source,
            rejected_question=(
                "What was the birth name of the art-world guru Andy Warhol's?"
            ),
        )[1]["content"]
    )
    assert "canonical_training_answer" not in repair_payload
    assert repair_payload["rejected_question"].startswith("What was")
    assert repair_payload["required_source_token_to_replace"] == "guru"
    assert "genuine synonym" in QUESTION_REPAIR_SYSTEM_PROMPT
    assert parse_question_repair_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "What was the birth name of art-world authority Andy Warhol?"
                ),
                "replaced_source_token": "guru",
                "replacement_phrase": "authority",
            }
        ),
        eligible_source_tokens=payload["lexical_replacement_source_tokens"],
        required_source_token=repair_payload["required_source_token_to_replace"],
    ).endswith("Andy Warhol?")
    synonym_payload = json.loads(
        build_synonym_repair_messages(
            source,
            required_source_token="guru",
        )[1]["content"]
    )
    assert synonym_payload["required_source_token"] == "guru"
    assert "canonical_training_answer" not in synonym_payload
    assert "context-appropriate synonym" in SYNONYM_REPAIR_SYSTEM_PROMPT
    assert parse_synonym_repair_response(
        json.dumps(
            {"source_token": "guru", "replacement_phrase": "authority"}
        ),
        required_source_token="guru",
    ) == "authority"
    assert parse_question_repair_response(
        json.dumps(
            {
                "parphrase_question": (
                    "What was the birth name of art-world mentor Andy Warhol?"
                ),
                "replaced_source_token": "guru",
                "replacement_phrase": "mentor",
            }
        ),
        eligible_source_tokens=payload["lexical_replacement_source_tokens"],
        required_source_token=repair_payload["required_source_token_to_replace"],
    ).endswith("Andy Warhol?")


def test_lexical_eligibility_excludes_all_literal_constraint_families() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:literal_eligibility",
        base_task_id="triviaqa:literal_eligibility",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Which 10-year program did Project Orion use at .example for "
            "“hidden phrase” in 1999-2000?"
        ),
        canonical_answer="Program Result",
        native_split="train",
    )
    generation_payload = json.loads(
        build_paraphrase_messages(source)[1]["content"]
    )
    repair_payload = json.loads(
        build_question_repair_messages(
            source,
            rejected_question=source.original_question,
        )[1]["content"]
    )

    assert generation_payload["immutable_number_or_date_tokens"] == [
        "10",
        "1999-2000",
    ]
    assert generation_payload["immutable_quoted_spans"] == ["hidden phrase"]
    assert generation_payload["lexical_replacement_source_tokens"] == ["use"]
    assert repair_payload["lexical_replacement_source_tokens"] == (
        generation_payload["lexical_replacement_source_tokens"]
    )
    assert repair_payload["required_source_token_to_replace"] == "use"


def test_leading_interrogative_contraction_is_not_an_entity_or_repair_target() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:interrogative_contraction",
        base_task_id="triviaqa:interrogative_contraction",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What's the capital of Kenya?",
        canonical_answer="Nairobi",
        native_split="train",
    )
    payload = json.loads(build_paraphrase_messages(source)[1]["content"])

    assert payload["immutable_original_entity_tokens"] == ["Kenya"]
    assert payload["lexical_replacement_source_tokens"] == ["capital"]
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": "Name Kenya's capital.",
                "paraphrase_answer_statement": (
                    "Nairobi is the capital of Kenya."
                ),
            }
        ),
        source,
    )
    assert question == "Name Kenya's capital."
    assert statement.startswith("Nairobi")


def test_v7_answer_verifier_binds_canonical_to_original_wh_slot() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_305",
        base_task_id="triviaqa:tc_305",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Richard Nixon was Vice President to which US state?"
        ),
        canonical_answer="Dwight Eisenhower",
        native_split="train",
    )
    messages = build_answer_verification_messages(
        source,
        paraphrase_answer_statement=(
            "Richard Nixon was Vice President to Dwight Eisenhower."
        ),
    )
    payload = json.loads(messages[1]["content"])
    assert payload["canonical_training_answer"] == "Dwight Eisenhower"
    assert "authoritative dataset strings" in ANSWER_VERIFICATION_SYSTEM_PROMPT
    accepted = {
        "canonical_span_preserved": True,
        "answer_slot_bound": True,
        "relation_direction_preserved": True,
        "scope_and_constraints_preserved": True,
        "no_new_fact_or_relation": True,
    }
    assert parse_answer_verification_response(json.dumps(accepted)) == accepted
    with pytest.raises(ValueError, match="fields are incompatible"):
        parse_answer_verification_response(
            json.dumps({**accepted, "unexpected": True})
        )


def test_v7_answer_repair_contract_preserves_wh_slot_direction() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_305",
        base_task_id="triviaqa:tc_305",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Richard Nixon was Vice President to which US state?"
        ),
        canonical_answer="Dwight Eisenhower",
        native_split="train",
    )
    messages = build_answer_repair_messages(
        source,
        rejected_answer_statement=(
            "Richard Nixon served as Vice President to the US state of "
            "Dwight Eisenhower."
        ),
    )
    payload = json.loads(messages[1]["content"])
    assert payload["original_question"] == source.original_question
    assert "literal slot substitution" in ANSWER_REPAIR_SYSTEM_PROMPT
    assert parse_answer_repair_response(
        json.dumps(
            {
                "paraphrase_answer_statement": (
                    "Richard Nixon was Vice President to Dwight Eisenhower."
                )
            }
        )
    ).endswith("Dwight Eisenhower.")
    assert _literal_slot_substitution_preserved(
        original_question=source.original_question,
        canonical_answer=source.canonical_answer,
        answer_statement=(
            "Richard Nixon was Vice President to Dwight Eisenhower."
        ),
    )
    assert not _literal_slot_substitution_preserved(
        original_question=source.original_question,
        canonical_answer=source.canonical_answer,
        answer_statement=(
            "Dwight Eisenhower served as Vice President to Richard Nixon."
        ),
    )


def test_v8_answer_repair_forbids_long_canonical_answer_only_output() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_451",
        base_task_id="triviaqa:tc_451",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="What bird is named for the apostle Peter?",
        canonical_answer=(
            "The petrel, from a diminutive form of Petrus, or Peter, in Latin"
        ),
        native_split="train",
    )
    assert "Never return only the canonical answer" in ANSWER_REPAIR_SYSTEM_PROMPT
    assert "What/Which <answer type>" in ANSWER_REPAIR_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="express the question relation beyond"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": (
                        "Which avian species derives its name from the apostle Peter?"
                    ),
                    "paraphrase_answer_statement": source.canonical_answer,
                }
            ),
            source,
        )
    assert not _answer_statement_has_lexical_relation_lineage(
        original_question=source.original_question,
        canonical_answer=source.canonical_answer,
        answer_statement=(
            "The petrel, from a diminutive form of Petrus, or Peter, in Latin "
            "is the species."
        ),
    )
    with pytest.raises(ValueError, match="lost lexical relation lineage"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": (
                        "Which avian species derives its name from the apostle Peter?"
                    ),
                    "paraphrase_answer_statement": (
                        "The petrel, from a diminutive form of Petrus, or Peter, "
                        "in Latin is the species."
                    ),
                }
            ),
            source,
        )


def test_v8_answer_repair_fronts_prepositional_canonical_span() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_462",
        base_task_id="triviaqa:tc_462",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "In what language was the New Testament originally written?"
        ),
        canonical_answer="In Greek",
        native_split="train",
    )
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "Which language was used for the original writing of the "
                    "New Testament?"
                ),
                "paraphrase_answer_statement": (
                    "In Greek, the New Testament was originally written."
                ),
            }
        ),
        source,
    )
    assert question.startswith("Which language")
    assert statement.startswith("In Greek,")
    assert "fronted phrase" in ANSWER_REPAIR_SYSTEM_PROMPT


def test_v8_answer_repair_preserves_possessive_canonical_span() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_597",
        base_task_id="triviaqa:tc_597",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "The diet of what mythical monster periodically included seven "
            "youths and seven maidens?"
        ),
        canonical_answer="The Minotaur's",
        native_split="train",
    )
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "Which mythical monster's diet periodically contained seven "
                    "youths and seven maidens?"
                ),
                "paraphrase_answer_statement": (
                    "The Minotaur's diet periodically included seven youths and "
                    "seven maidens."
                ),
            }
        ),
        source,
    )
    assert question.startswith("Which mythical")
    assert statement.startswith("The Minotaur's diet")
    assert "exact possessive canonical" in ANSWER_REPAIR_SYSTEM_PROMPT


def test_v8_called_relation_has_deterministic_answer_binding_fallback() -> None:
    assert _called_relation_substitution_preserved(
        original_question="What was the world's first atomic-powered ship called?",
        canonical_answer="Lenin",
        answer_statement=(
            "The world's first atomic-powered ship was called Lenin."
        ),
    )
    assert not _called_relation_substitution_preserved(
        original_question="What was the world's first atomic-powered ship called?",
        canonical_answer="Lenin",
        answer_statement="Lenin powered the world's first atomic ship.",
    )


@pytest.mark.parametrize(
    ("original", "canonical", "paraphrase_question", "expected"),
    (
        (
            "What is the name of Mickey Mouse's pet dog?",
            "Pluto",
            "Identify Mickey Mouse's pet dog by name.",
            "The name of Mickey Mouse's pet dog is Pluto.",
        ),
        (
            "What is the young of a koala called?",
            "Joey",
            "Identify the term used for a young koala.",
            "The young of a koala is called Joey.",
        ),
        (
            "The singer Mary O’Brien was better known by what name?",
            "Dusty Springfield",
            "Under which name was the singer Mary O’Brien better recognized?",
            (
                "The singer Mary O’Brien was better known by the name "
                "Dusty Springfield."
            ),
        ),
        (
            "Which English king was known as Longshanks?",
            "Edward I",
            "Identify the English king who was known as Longshanks.",
            "Edward I was known as Longshanks.",
        ),
        (
            "How many red stripes are there on the national flag of Puerto Rico?",
            "Three",
            "State the number of red stripes there are on the national flag of Puerto Rico.",
            "There are Three red stripes on the national flag of Puerto Rico.",
        ),
        (
            "How many Madison Square Gardens have there been before the existing one?",
            "Three",
            (
                "State the number of Madison Square Gardens there have been "
                "before the existing one."
            ),
            "There have been Three Madison Square Gardens before the existing one.",
        ),
        (
            "How many players are on a baseball team?",
            "Nine",
            "State the number of players that are on a baseball team.",
            "Nine players are on a baseball team.",
        ),
        (
            "The Internet TLD for Albania is what?",
            ".AL",
            "State the value of Albania's Internet TLD.",
            "The Internet TLD for Albania is .AL.",
        ),
        (
            (
                "Glenmorangie whisky is produced by 16 men known as the "
                "Sixteen Men of where?"
            ),
            "TAIN",
            (
                "At what place is Glenmorangie whisky produced by 16 men known "
                "as the Sixteen Men?"
            ),
            (
                "Glenmorangie whisky is produced by 16 men known as the "
                "Sixteen Men of TAIN."
            ),
        ),
        (
            (
                "Complete the title of the debut novel by Marina Lewycka "
                "'A Short History of … in Ukrainian'."
            ),
            "TRACTORS",
            (
                "Supply the missing completion for the title of the debut novel "
                "by Marina Lewycka 'A Short History of … in Ukrainian'."
            ),
            (
                "The completion of the title of the debut novel by Marina "
                "Lewycka 'A Short History of … in Ukrainian' is TRACTORS."
            ),
        ),
        (
            "Which country has the internet domain .ch?",
            "Switzerland",
            "Identify the country that has the internet domain .ch.",
            "Switzerland is the country that has the internet domain .ch.",
        ),
    ),
)
def test_deterministic_answer_slot_statement_passes_existing_admission(
    original: str,
    canonical: str,
    paraphrase_question: str,
    expected: str,
) -> None:
    source = _semantic_source(original, canonical)

    candidate = _deterministic_answer_slot_statement(source)

    assert candidate == expected
    assert parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": paraphrase_question,
                "paraphrase_answer_statement": candidate,
            }
        ),
        source,
    ) == (paraphrase_question, expected)


@pytest.mark.parametrize(
    ("original", "canonical"),
    (
        ("What is the capital of Hong Kong?", "Victoria"),
        (
            "How many are gold; how many are silver?",
            "One gold and three silver",
        ),
        (
            "How many coins in one turn does each player use?",
            "Five",
        ),
        (
            "How many trophies are there: two, three, or four?",
            "Three",
        ),
        ("Complete this sentence.", "the missing words"),
        (
            "What colour Cat’s-Eyes mark the nearside of a motorway?",
            "Red",
        ),
        (
            "On which street was a shop named Walter Roberts located?",
            "Hope Street",
        ),
        (
            "What US politician's 1996 autobiography was called 'Dreams From my Father'?",
            "Barack Obama",
        ),
        (
            "Which King created the George Cross medal? George V or George VI?",
            "George VI",
        ),
        (
            (
                "Which Iain Banks novel has the name of a bird in the title? "
                "The book was also made into a television series."
            ),
            "The Crow Road",
        ),
        (
            (
                "Complete the title of the novel 'A Short History of\u0085.in "
                "Ukrainian'."
            ),
            "TRACTORS",
        ),
    ),
)
def test_deterministic_answer_slot_statement_fails_closed_for_unsafe_shapes(
    original: str,
    canonical: str,
) -> None:
    assert _deterministic_answer_slot_statement(
        _semantic_source(original, canonical)
    ) is None


def test_answer_repair_uses_verified_deterministic_answer_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _semantic_source(
        "What is the name of Mickey Mouse's pet dog?",
        "Pluto",
    )
    client = object.__new__(LocalQwen35Paraphraser)
    verified: list[str] = []
    monkeypatch.setattr(
        client,
        "_answer_statement_verified",
        lambda source, *, statement, seed: verified.append(statement) or True,
    )
    monkeypatch.setattr(
        client,
        "_complete",
        lambda **_: pytest.fail("deterministic repair must precede model repair"),
    )

    repaired = client._repair_answer_statement(
        source,
        question="Identify Mickey Mouse's pet dog by name.",
        rejected_statement="No accepted relation statement was produced.",
        seed=47,
    )

    assert repaired == "The name of Mickey Mouse's pet dog is Pluto."
    assert verified == [repaired]


def test_v7_listed_choice_repair_keeps_exact_canonical_option_label() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_297",
        base_task_id="triviaqa:tc_297",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question=(
            "Between 1952 and 1954 did the number of TV stations in the USA "
            "double, triple or quadruple?"
        ),
        canonical_answer="Triple",
        native_split="train",
    )
    messages = build_answer_repair_messages(
        source,
        rejected_answer_statement=(
            "The number of TV stations in the USA tripled between 1952 and 1954."
        ),
    )
    payload = json.loads(messages[1]["content"])
    assert payload["canonical_training_answer"] == "Triple"
    assert "selected option label" in ANSWER_REPAIR_SYSTEM_PROMPT
    augmented = _augment_listed_choice_answer_statement(
        original_question=source.original_question,
        canonical_answer=source.canonical_answer,
        rejected_answer_statement=(
            "The number of TV stations in the USA tripled between 1952 and 1954."
        ),
    )
    assert augmented == (
        "The number of TV stations in the USA tripled between 1952 and 1954; "
        "the selected listed option is Triple."
    )
    question, statement = parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": (
                    "Between 1952 and 1954, did the quantity of TV stations in "
                    "the USA double, triple or quadruple?"
                ),
                "paraphrase_answer_statement": (
                    "The number of TV stations in the USA tripled between 1952 "
                    "and 1954; the selected listed option is Triple."
                ),
            }
        ),
        source,
    )
    assert question.startswith("Between 1952")
    assert statement.count("Triple") == 1
    assert _listed_choice_answer_binding_preserved(
        original_question=source.original_question,
        canonical_answer=source.canonical_answer,
        answer_statement=statement,
    )
    with pytest.raises(ValueError, match="selected listed option"):
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": question,
                    "paraphrase_answer_statement": (
                        "Between 1952 and 1954 did the number of TV stations in "
                        "the USA double, triple or quadruple. Triple."
                    ),
                }
            ),
            source,
        )


@pytest.mark.parametrize(
    ("question", "canonical", "statement"),
    (
        (
            "Who was born first, Susan Sarandon or Glenn Close?",
            "Susan Sarandon",
            "Susan Sarandon was born before Glenn Close.",
        ),
        (
            "Who was born first, James Caan or Michael Douglas?",
            "James Caan",
            "James Caan was born before Michael Douglas.",
        ),
    ),
)
def test_v8_person_choice_augmentation_preserves_relation_and_label(
    question: str,
    canonical: str,
    statement: str,
) -> None:
    augmented = _augment_listed_choice_answer_statement(
        original_question=question,
        canonical_answer=canonical,
        rejected_answer_statement=statement,
    )
    assert augmented is not None
    assert augmented.endswith(f"the selected listed option is {canonical}.")
    assert _listed_choice_answer_binding_preserved(
        original_question=question,
        canonical_answer=canonical,
        answer_statement=augmented,
    )


def test_resume_keeps_valid_rows_and_drops_only_answer_only_rows(tmp_path) -> None:
    valid = _record()
    raw_invalid = valid.to_value()
    raw_invalid["source_train_task_id"] = "triviaqa:tc_1000"
    raw_invalid["base_task_id"] = "triviaqa:tc_1000"
    raw_invalid["selection_index"] = 1
    raw_invalid["paraphrase_answer_statement"] = "Zambezi"
    # Resume filtering happens before strict record construction, so the stale
    # identity from the rejected row is intentionally not recomputed.
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (valid.to_value(), raw_invalid)
        )
        + "\n",
        encoding="utf-8",
    )

    records, rejected = load_resume_records(checkpoint)

    assert records == (valid,)
    assert rejected == ("triviaqa:tc_1000",)


def test_v5_materialization_rejects_new_answer_surface_in_question() -> None:
    record = _record(question="Is the answer Zambezi?")

    with pytest.raises(ValueError, match="introduced the canonical answer"):
        validate_qa_memory_against_sources(
            (record,),
            (_source(),),
            require_complete=True,
        )


def test_exact_query_contamination_is_attributed_to_each_field() -> None:
    source = _source()
    original = source.original_question
    assert _exact_question_identity_contaminated_fields(
        source,
        paraphrase_question=f"Determine this: {original}",
        paraphrase_answer_statement=(
            f"For {original}, the Zambezi is the requested river."
        ),
    ) == frozenset(
        {"paraphrase_question", "paraphrase_answer_statement"}
    )
    assert not _exact_question_identity_contaminated_fields(
        source,
        paraphrase_question="Name the waterway holding the Kariba Dam.",
        paraphrase_answer_statement=(
            "The Kariba Dam was built on the Zambezi river."
        ),
    )


def test_exact_query_recovery_routes_only_contaminated_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    client = object.__new__(LocalQwen35Paraphraser)
    calls: list[str] = []

    def repair_question(*args: object, **kwargs: object) -> str:
        calls.append("question")
        return "Name the waterway holding the Kariba Dam."

    def repair_statement(*args: object, **kwargs: object) -> str:
        calls.append("statement")
        return "The Kariba Dam was built on the Zambezi river."

    monkeypatch.setattr(client, "_repair_paraphrase_question", repair_question)
    monkeypatch.setattr(client, "_repair_answer_statement", repair_statement)
    question, statement = client._repair_exact_question_identity_shortcut(
        source,
        question=f"Determine this: {source.original_question}",
        statement=(
            f"For {source.original_question}, the Zambezi is the requested river."
        ),
        seed=7,
    )

    assert calls == ["question", "statement"]
    assert question == "Name the waterway holding the Kariba Dam."
    assert statement == "The Kariba Dam was built on the Zambezi river."


def test_exact_query_recovery_remains_fail_closed_after_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    client = object.__new__(LocalQwen35Paraphraser)
    contaminated = f"Determine this: {source.original_question}"
    monkeypatch.setattr(
        client,
        "_repair_paraphrase_question",
        lambda *args, **kwargs: contaminated,
    )

    with pytest.raises(ValueError, match="complete original question"):
        client._repair_exact_question_identity_shortcut(
            source,
            question=contaminated,
            statement="The Kariba Dam was built on the Zambezi river.",
            seed=11,
        )


def test_v5_materialization_allows_original_spelling_normalization() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_735",
        base_task_id="triviaqa:tc_735",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which state is misspelled as Rode island?",
        canonical_answer="Rhode Island",
        native_split="train",
    )
    record = TriviaQAQAMemoryRecord.create(
        source=source,
        paraphrase_question="Which state is intended by the spelling Rhode Island?",
        paraphrase_answer_statement=(
            "The state intended by the misspelling is Rhode Island."
        ),
        paraphrase_version="triviaqa.qa_memory.paraphrase.v5",
        paraphrase_method="semantic-preserving-question-and-answer-paraphrase",
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v5",
        generation_seed=20260827,
    )

    validate_qa_memory_against_sources((record,), (source,), require_complete=True)


@pytest.mark.parametrize(
    ("original", "canonical", "statement", "expected"),
    (
        (
            "Who became Queen of the Netherlands in 1980?",
            "Beatrix",
            "Beatrix became Queen of the Netherlands in 1980.",
            "Identify the person who became Queen of the Netherlands in 1980.",
        ),
        (
            "Where was Albert Einstein born?",
            "Ulm",
            "Albert Einstein was born in Ulm.",
            "At what location was Albert Einstein born?",
        ),
        (
            "When was Apollo 11 launched?",
            "1969",
            "Apollo 11 was launched in 1969.",
            "At what time was Apollo 11 launched?",
        ),
        (
            "How many players are on a baseball team?",
            "Nine",
            "Nine players are on a baseball team.",
            "State the number of players that are on a baseball team.",
        ),
        (
            "Name the river containing the Kariba Dam.",
            "Zambezi",
            "The river containing the Kariba Dam is the Zambezi.",
            "Identify the river containing the Kariba Dam.",
        ),
        (
            'Complete the title "A Tale of Two ___."',
            "Cities",
            'The title "A Tale of Two ___." is completed with Cities.',
            'Supply the missing completion for the title "A Tale of Two ___."',
        ),
        (
            'Finish the proverb "A stitch in time saves ___."',
            "nine",
            'The ending of the proverb "A stitch in time saves ___." is nine.',
            'Supply the ending of the proverb "A stitch in time saves ___."',
        ),
        (
            "Which river contains the Kariba Dam?",
            "Zambezi",
            "The Zambezi river contains the Kariba Dam.",
            "Identify the river that contains the Kariba Dam.",
        ),
        (
            "Which Gloria co-founded Ms magazine?",
            "Gloria Steinem",
            "Gloria Steinem co-founded Ms magazine.",
            "Identify the Gloria that co-founded Ms magazine.",
        ),
    ),
)
def test_deterministic_question_fallback_passes_full_materialization_gate(
    original: str,
    canonical: str,
    statement: str,
    expected: str,
) -> None:
    source = _semantic_source(original, canonical)

    candidate = _deterministic_question_paraphrase(source)

    assert candidate == expected
    assert " ".join(original.split()).casefold() not in candidate.casefold()
    assert _has_lexical_or_phrase_replacement(original, candidate)
    assert parse_paraphrase_response(
        json.dumps(
            {
                "paraphrase_question": candidate,
                "paraphrase_answer_statement": statement,
            }
        ),
        source,
    ) == (candidate, statement)


def test_question_repair_uses_deterministic_fallback_only_after_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _semantic_source(
        "Who became Queen of the Netherlands in 1980?",
        "Beatrix",
    )
    client = object.__new__(LocalQwen35Paraphraser)
    calls = 0

    def reject_model_repair(**_: object) -> str:
        nonlocal calls
        calls += 1
        raise ValueError("synthetic bounded repair failure")

    monkeypatch.setattr(client, "_complete", reject_model_repair)

    assert client._repair_paraphrase_question(
        source,
        rejected_question=source.original_question,
        seed=19,
    ) == "Identify the person who became Queen of the Netherlands in 1980."
    assert calls > 0


@pytest.mark.parametrize(
    ("original", "canonical"),
    (
        ("Who did Elizabeth Taylor marry?", "Richard Burton"),
        ("What is the capital of Hong Kong?", "Victoria"),
        (
            "How many are gold; how many are silver?",
            "One gold and three silver",
        ),
        ("Complete.", "the missing text"),
        (
            "When You’re Strange is a tribute band to which band?",
            "The Doors",
        ),
        (
            "When eating out, what French phrase is the opposite of a la carte?",
            "table d'hôte",
        ),
        ("Where Eagles Dare starred which actor?", "Richard Burton"),
    ),
)
def test_deterministic_question_fallback_fails_closed_for_unsafe_shapes(
    original: str,
    canonical: str,
) -> None:
    assert _deterministic_question_paraphrase(
        _semantic_source(original, canonical)
    ) is None


def test_sentence_boundary_operators_are_not_named_entities() -> None:
    identities = _capitalized_identity_tokens(
        "I'm ready. Don't guess. Name Liverpool's river. "
        "You're certain; Identify Sony's label."
    )

    assert "i'm" not in identities
    assert "don't" not in identities
    assert "name" not in identities
    assert "you're" not in identities
    assert "identify" not in identities
    assert "liverpool's" in identities
    assert "sony's" in identities


def test_contraction_exception_is_position_sensitive_and_quotes_stay_immutable() -> None:
    assert "don't" in _capitalized_identity_tokens(
        "Don't Panic released a record."
    )
    assert "don't" in _capitalized_identity_tokens(
        "The band Don't Panic released a record."
    )
    assert _quoted_scope_preserved(
        'Which band recorded "Don\'t Stop"?',
        'Identify the band that recorded "Don\'t Stop".',
    )
    assert not _quoted_scope_preserved(
        'Which band recorded "Don\'t Stop"?',
        'Identify the band that recorded "Do Not Stop".',
    )
