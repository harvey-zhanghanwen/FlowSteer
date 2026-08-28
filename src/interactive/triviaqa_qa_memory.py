"""Frozen TriviaQA train-QA memory with SkillFlow-compatible dense retrieval.

This is a dataset adaptation over the existing TriviaQA embedding index, not a
second Tool runtime.  It reuses the same local BGE encoder, normalized
dot-product retrieval, deterministic ranking, and atomic persistence helpers.

Only the frozen project train split can contribute memory records.  The held-
out validation file is read only for ``base_task_id`` split-isolation checks.
For each train row, ``accepted_answers[0]`` is the canonical answer.  The
pipe-joined ``ground_truth``/``answer`` values and all remaining accepted
aliases are deliberately outside the memory projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType
from typing import Any

import numpy as np

from .triviaqa_embedding_index import (
    BGE_QUERY_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingEncoder,
    _canonical_json,
    _file_sha256,
    _get_embedding_model,
    _nonnegative_integer,
    _normalized_embeddings,
    _positive_integer,
    _required_sha256,
    _required_text,
    _stable_sha256,
    _write_atomic_bytes,
)


QA_MEMORY_TOOL_ID = "triviaqa.qa_memory"
MEMORY_SCHEMA_VERSION = "flowsteer.triviaqa.qa_memory.record.v1"
MANIFEST_SCHEMA_VERSION = "flowsteer.triviaqa.qa_memory.manifest.v1"
INDEX_FORMAT = "flowsteer.triviaqa.qa-memory-embedding-index.v1"
RETRIEVAL_BACKEND = "sentence-transformers-bge-normalized-dot-product"
CORPUS_NAME = "triviaqa-frozen-train-qa-memory"
SOURCE_DATASET = "TriviaQA"
SOURCE_PROJECT_SPLIT = "train"
NORMALIZATION = "l2"
SIMILARITY = "dot_product"
EMBEDDING_TEXT_TEMPLATE = (
    "Question: {paraphrase_question}\nAnswer: {paraphrase_answer_statement}"
)
_STRICT_PARAPHRASE_PROMPT_VERSIONS = frozenset(
    {
        "triviaqa.qa_memory.qa_paraphrase.v4",
        "triviaqa.qa_memory.qa_paraphrase.v5",
        "triviaqa.qa_memory.qa_paraphrase.v6",
        "triviaqa.qa_memory.qa_paraphrase.v7",
        "triviaqa.qa_memory.qa_paraphrase.v8",
        "triviaqa.qa_memory.qa_paraphrase.v9",
        "triviaqa.qa_memory.qa_paraphrase.v10",
        "triviaqa.qa_memory.qa_paraphrase.v11",
        "triviaqa.qa_memory.qa_paraphrase.v12",
    }
)

MANIFEST_FILENAME = "manifest.json"
MEMORIES_FILENAME = "memories.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"

# PROJECT_NECESSARY_ADAPTATION: the earlier 512-row slice happened to contain
# only ``tc_*`` IDs.  The canonical TriviaQA train split also contains qz_,
# qw_, sfq_, odql_, and other public question-ID namespaces.  Keep the native
# opaque ID after the dataset prefix instead of narrowing the full dataset back
# to the first namespace.
_BASE_TASK_ID = re.compile(r"triviaqa:[A-Za-z0-9_.-]+\Z")
_CYCLED_TASK_ID = re.compile(
    r"triviaqa:[A-Za-z0-9_.-]+:cycle-[0-9]{4,}\Z"
)
_MEMORY_ID = re.compile(r"triviaqa-qa-memory-[0-9a-f]{64}\Z")
_MEMORY_FIELDS = frozenset(
    {
        "schema_version",
        "memory_id",
        "tool_id",
        "source_dataset",
        "source_split",
        "source_train_task_id",
        "base_task_id",
        "selection_index",
        "cycled_training_sample",
        "cycle_index",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "canonical_answer",
        "paraphrase_version",
        "paraphrase_method",
        "generator_provider",
        "model_id",
        "model_revision",
        "prompt_template_version",
        "generation_seed",
        "canonical_span_preserved",
    }
)


def canonical_is_original_spelling_variant(
    original_question: str,
    canonical_answer: str,
) -> bool:
    """Recognize an answer surface already present with a minor misspelling."""

    canonical_tokens = canonical_answer.casefold().split()
    original_tokens = original_question.casefold().split()
    width = len(canonical_tokens)
    if width < 1 or len(original_tokens) < width:
        return False
    canonical = " ".join(canonical_tokens)
    return any(
        SequenceMatcher(
            None,
            canonical,
            " ".join(original_tokens[start : start + width]),
        ).ratio()
        >= 0.9
        for start in range(len(original_tokens) - width + 1)
    )
_TOOL_BUDGET_FIELDS = frozenset(
    {"max_tool_calls_per_agent_call", "max_turns_per_agent_call"}
)
_FILE_FIELDS = frozenset({"memories", "embeddings"})
_FILE_ENTRY_FIELDS = frozenset({"name", "sha256"})


def _base_task_id(value: object) -> str:
    text = _required_text(value, field_name="base_task_id")
    if _BASE_TASK_ID.fullmatch(text) is None:
        raise ValueError("TriviaQA base_task_id must preserve a native question_id")
    return text


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def deterministic_answer_statement(canonical_answer: str) -> str:
    """Render one answer statement while preserving the exact canonical span."""

    canonical = _required_text(canonical_answer, field_name="canonical_answer")
    return f"The answer is {canonical}"


_GENERIC_ANSWER_WRAPPER_TOKENS = frozenset(
    {
        "a",
        "an",
        "answer",
        "are",
        "be",
        "been",
        "being",
        "canonical",
        "is",
        "it",
        "question",
        "result",
        "that",
        "the",
        "this",
        "to",
        "value",
        "was",
        "were",
    }
)
_STATEMENT_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def exact_canonical_span_preserved(
    text: object,
    canonical_answer: object,
) -> bool:
    """Require the exact case-sensitive answer span at lexical boundaries."""

    if not isinstance(text, str) or not text.strip():
        return False
    if not isinstance(canonical_answer, str) or not canonical_answer.strip():
        return False
    canonical = " ".join(canonical_answer.split())
    return re.search(
        rf"(?<!\w){re.escape(canonical)}(?!\w)",
        " ".join(text.split()),
    ) is not None


def relation_bearing_answer_statement(
    statement: object,
    canonical_answer: object,
) -> bool:
    """Check that an answer is a relation-bearing statement, not a bare span.

    Semantic equivalence remains the paraphraser's generation contract.  This
    deterministic materialization boundary enforces the properties that can be
    checked without another model call: the exact canonical span is retained,
    and the remaining text contains non-generic relation context rather than an
    answer-only wrapper.
    """

    if not isinstance(statement, str) or not statement.strip():
        return False
    if not isinstance(canonical_answer, str) or not canonical_answer.strip():
        return False
    normalized_statement = " ".join(statement.split())
    canonical = " ".join(canonical_answer.split())
    if not exact_canonical_span_preserved(normalized_statement, canonical):
        return False
    terminal_punctuation = " .,!?:;\"'`()[]{}"
    if (
        normalized_statement.strip(terminal_punctuation).casefold()
        == canonical.strip(terminal_punctuation).casefold()
    ):
        return False
    relation_context = re.sub(
        rf"(?<!\w){re.escape(canonical)}(?!\w)",
        " ",
        normalized_statement,
    )
    context_tokens = tuple(
        token.casefold() for token in _STATEMENT_TOKEN.findall(relation_context)
    )
    return len(context_tokens) >= 2 and any(
        token not in _GENERIC_ANSWER_WRAPPER_TOKENS for token in context_tokens
    )


@dataclass(frozen=True, slots=True)
class TriviaQATrainSource:
    """Minimal projection of one frozen aligned train row.

    ``accepted_answers_for_admission`` is train-only loader state used for
    deterministic semantic admission.  It is never projected into a memory
    record, embedding payload, Tool observation, or Agent request.
    """

    source_train_task_id: str
    base_task_id: str
    selection_index: int
    cycled_training_sample: bool
    cycle_index: int | None
    original_question: str
    canonical_answer: str
    native_split: str
    accepted_answers_for_admission: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task_id = _required_text(
            self.source_train_task_id,
            field_name="source_train_task_id",
        )
        base_id = _base_task_id(self.base_task_id)
        object.__setattr__(self, "source_train_task_id", task_id)
        object.__setattr__(self, "base_task_id", base_id)
        _nonnegative_int(self.selection_index, field_name="selection_index")
        if type(self.cycled_training_sample) is not bool:
            raise ValueError("cycled_training_sample must be boolean")
        if self.cycled_training_sample:
            if type(self.cycle_index) is not int or self.cycle_index < 1:
                raise ValueError("cycled training source requires positive cycle_index")
            if _CYCLED_TASK_ID.fullmatch(task_id) is None or not task_id.startswith(
                base_id + ":cycle-"
            ):
                raise ValueError("cycled source_train_task_id is incompatible")
        else:
            if self.cycle_index is not None:
                raise ValueError("non-cycled training source cannot have cycle_index")
            if task_id != base_id:
                raise ValueError("non-cycled source_train_task_id must equal base_task_id")
        object.__setattr__(
            self,
            "original_question",
            _required_text(self.original_question, field_name="original_question"),
        )
        object.__setattr__(
            self,
            "canonical_answer",
            _required_text(self.canonical_answer, field_name="canonical_answer"),
        )
        object.__setattr__(
            self,
            "native_split",
            _required_text(self.native_split, field_name="native_split"),
        )
        accepted = tuple(
            _required_text(answer, field_name="accepted_answers_for_admission")
            for answer in self.accepted_answers_for_admission
        )
        if accepted and accepted[0] != self.canonical_answer:
            raise ValueError(
                "accepted_answers_for_admission must begin with canonical_answer"
            )
        object.__setattr__(self, "accepted_answers_for_admission", accepted)


def _sampling(metadata: Mapping[str, object], *, line_number: int) -> Mapping[str, object]:
    value = metadata.get("sampling")
    if not isinstance(value, Mapping):
        raise ValueError(f"aligned TriviaQA row {line_number} has no sampling metadata")
    return value


def _validation_base_task_ids(
    path: Path,
    *,
    expected_count: int,
) -> frozenset[str]:
    """Read only validation identity/split fields, never validation labels."""

    base_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"validation JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"validation row {line_number} is not an object")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get(
                "dataset_key"
            ) != "triviaqa":
                continue
            if row.get("split") != "validation":
                raise ValueError("TriviaQA validation row has an incompatible split")
            sampling = _sampling(metadata, line_number=line_number)
            base_ids.append(_base_task_id(sampling.get("base_task_id")))
    if len(base_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} TriviaQA validation rows, found {len(base_ids)}"
        )
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("TriviaQA validation base_task_id values are not unique")
    return frozenset(base_ids)


def _train_sources(path: Path, *, expected_count: int) -> tuple[TriviaQATrainSource, ...]:
    sources: list[TriviaQATrainSource] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"train JSON is invalid at line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"train row {line_number} is not an object")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get(
                "dataset_key"
            ) != "triviaqa":
                continue
            if row.get("split") != SOURCE_PROJECT_SPLIT:
                raise ValueError("TriviaQA train row has an incompatible split")
            sampling = _sampling(metadata, line_number=line_number)
            cycled = sampling.get("cycled_training_sample")
            if type(cycled) is not bool:
                raise ValueError("cycled_training_sample must be boolean")
            payload = metadata.get("evaluator_payload")
            accepted = (
                payload.get("accepted_answers")
                if isinstance(payload, Mapping)
                else None
            )
            # FIELD BOUNDARY: train accepted answers are loader-only admission
            # state.  Only accepted_answers[0] becomes canonical_answer; no
            # alias is projected into the materialized record or index.
            if (
                not isinstance(accepted, list)
                or not accepted
                or any(
                    not isinstance(answer, str) or not answer.strip()
                    for answer in accepted
                )
            ):
                raise ValueError(
                    f"train row {line_number} has no canonical accepted_answers[0]"
                )
            sources.append(
                TriviaQATrainSource(
                    source_train_task_id=_required_text(
                        row.get("task_id"), field_name="task_id"
                    ),
                    base_task_id=_base_task_id(sampling.get("base_task_id")),
                    selection_index=_nonnegative_int(
                        sampling.get("selection_index"),
                        field_name="selection_index",
                    ),
                    cycled_training_sample=cycled,
                    cycle_index=(
                        sampling.get("cycle_index") if cycled else None
                    ),
                    original_question=_required_text(
                        row.get("question"), field_name="question"
                    ),
                    canonical_answer=accepted[0],
                    native_split=_required_text(
                        metadata.get("native_split", SOURCE_PROJECT_SPLIT),
                        field_name="native_split",
                    ),
                    accepted_answers_for_admission=tuple(accepted),
                )
            )
    if len(sources) != expected_count:
        raise ValueError(
            f"expected {expected_count} TriviaQA train rows, found {len(sources)}"
        )
    task_ids = [source.source_train_task_id for source in sources]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("TriviaQA train source_train_task_id values are not unique")
    selection_indices = [source.selection_index for source in sources]
    if sorted(selection_indices) != list(range(expected_count)):
        raise ValueError("TriviaQA train selection_index must cover the frozen order")
    noncycled_base_ids = [
        source.base_task_id for source in sources if not source.cycled_training_sample
    ]
    if len(set(noncycled_base_ids)) != len(noncycled_base_ids):
        raise ValueError("non-cycled TriviaQA train base_task_id values are not unique")
    admitted_base_ids = set(noncycled_base_ids)
    for source in sources:
        if source.cycled_training_sample and source.base_task_id not in admitted_base_ids:
            raise ValueError("cycled TriviaQA source has no non-cycled training origin")
    return tuple(sorted(sources, key=lambda source: source.selection_index))


def load_triviaqa_qa_memory_sources(
    train_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    *,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
    allow_validation_overlap: bool = False,
) -> tuple[tuple[TriviaQATrainSource, ...], frozenset[str]]:
    """Load paired Q-A sources and enforce the declared evaluation scope.

    The default remains the held-out training protocol.  An explicit overlap
    flag is reserved for an in-database/transductive retrieval condition where
    the evaluated Q-A memories are intentionally present in the index.  This
    keeps that condition visible instead of bypassing split checks with a
    different validation file.
    """

    expected_train_count = _positive_integer(
        expected_train_count,
        field_name="expected_train_count",
    )
    expected_validation_count = _positive_integer(
        expected_validation_count,
        field_name="expected_validation_count",
    )
    if type(allow_validation_overlap) is not bool:
        raise TypeError("allow_validation_overlap must be boolean")
    train_path = Path(train_tasks_path)
    validation_path = Path(validation_tasks_path)
    if not train_path.is_file():
        raise FileNotFoundError("frozen TriviaQA train JSONL is unavailable")
    if not validation_path.is_file():
        raise FileNotFoundError("frozen TriviaQA validation JSONL is unavailable")
    validation_ids = _validation_base_task_ids(
        validation_path,
        expected_count=expected_validation_count,
    )
    sources = _train_sources(train_path, expected_count=expected_train_count)
    overlap = sorted(
        {source.base_task_id for source in sources}.intersection(validation_ids)
    )
    if overlap and not allow_validation_overlap:
        preview = ", ".join(overlap[:8])
        raise ValueError(
            "TriviaQA train and validation base_task_id values overlap: " + preview
        )
    return sources, validation_ids


@dataclass(frozen=True, slots=True)
class TriviaQAQAMemoryRecord:
    """One materialized semantic-preserving train QA paraphrase."""

    schema_version: str
    memory_id: str
    tool_id: str
    source_dataset: str
    source_split: str
    source_train_task_id: str
    base_task_id: str
    selection_index: int
    cycled_training_sample: bool
    cycle_index: int | None
    paraphrase_question: str
    paraphrase_answer_statement: str
    canonical_answer: str
    paraphrase_version: str
    paraphrase_method: str
    generator_provider: str
    model_id: str
    model_revision: str
    prompt_template_version: str
    generation_seed: int
    canonical_span_preserved: bool

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SCHEMA_VERSION:
            raise ValueError("QA-memory record schema version is unsupported")
        if self.tool_id != QA_MEMORY_TOOL_ID:
            raise ValueError("QA-memory record tool_id is unsupported")
        if self.source_dataset != SOURCE_DATASET:
            raise ValueError("QA-memory source_dataset is unsupported")
        if self.source_split != SOURCE_PROJECT_SPLIT:
            raise ValueError("QA-memory source_split must be train")
        memory_id = _required_text(self.memory_id, field_name="memory_id")
        if _MEMORY_ID.fullmatch(memory_id) is None:
            raise ValueError("QA-memory memory_id is incompatible")
        source = TriviaQATrainSource(
            source_train_task_id=self.source_train_task_id,
            base_task_id=self.base_task_id,
            selection_index=self.selection_index,
            cycled_training_sample=self.cycled_training_sample,
            cycle_index=self.cycle_index,
            original_question=self.paraphrase_question,
            canonical_answer=self.canonical_answer,
            native_split=SOURCE_PROJECT_SPLIT,
        )
        object.__setattr__(
            self,
            "source_train_task_id",
            source.source_train_task_id,
        )
        object.__setattr__(self, "base_task_id", source.base_task_id)
        question = _required_text(
            self.paraphrase_question,
            field_name="paraphrase_question",
        )
        statement = _required_text(
            self.paraphrase_answer_statement,
            field_name="paraphrase_answer_statement",
        )
        canonical = source.canonical_answer
        object.__setattr__(self, "paraphrase_question", question)
        object.__setattr__(self, "paraphrase_answer_statement", statement)
        object.__setattr__(self, "canonical_answer", canonical)
        if type(self.canonical_span_preserved) is not bool or not self.canonical_span_preserved:
            raise ValueError("canonical_span_preserved must be true")
        if not exact_canonical_span_preserved(statement, canonical):
            raise ValueError("paraphrase answer statement does not preserve canonical span")
        if (
            self.prompt_template_version in _STRICT_PARAPHRASE_PROMPT_VERSIONS
            and not relation_bearing_answer_statement(statement, canonical)
        ):
            raise ValueError(
                "paraphrase answer statement must be declarative and express "
                "the question relation beyond the canonical answer span"
            )
        for field_name in (
            "paraphrase_version",
            "paraphrase_method",
            "generator_provider",
            "model_id",
            "model_revision",
            "prompt_template_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        _nonnegative_int(self.generation_seed, field_name="generation_seed")
        expected_id = self._derived_memory_id()
        if memory_id != expected_id:
            raise ValueError("memory_id differs from QA-memory record identity")

    @classmethod
    def create(
        cls,
        *,
        source: TriviaQATrainSource,
        paraphrase_question: str,
        paraphrase_answer_statement: str,
        paraphrase_version: str,
        paraphrase_method: str,
        generator_provider: str,
        model_id: str,
        model_revision: str,
        prompt_template_version: str,
        generation_seed: int,
    ) -> "TriviaQAQAMemoryRecord":
        base: dict[str, object] = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memory_id": "triviaqa-qa-memory-" + ("0" * 64),
            "tool_id": QA_MEMORY_TOOL_ID,
            "source_dataset": SOURCE_DATASET,
            "source_split": SOURCE_PROJECT_SPLIT,
            "source_train_task_id": source.source_train_task_id,
            "base_task_id": source.base_task_id,
            "selection_index": source.selection_index,
            "cycled_training_sample": source.cycled_training_sample,
            "cycle_index": source.cycle_index,
            "paraphrase_question": _required_text(
                paraphrase_question,
                field_name="paraphrase_question",
            ),
            "paraphrase_answer_statement": _required_text(
                paraphrase_answer_statement,
                field_name="paraphrase_answer_statement",
            ),
            "canonical_answer": source.canonical_answer,
            "paraphrase_version": paraphrase_version,
            "paraphrase_method": paraphrase_method,
            "generator_provider": generator_provider,
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_template_version": prompt_template_version,
            "generation_seed": generation_seed,
            "canonical_span_preserved": True,
        }
        identity = dict(base)
        identity.pop("memory_id")
        base["memory_id"] = "triviaqa-qa-memory-" + _stable_sha256(identity)
        return cls.from_value(base)

    def _derived_memory_id(self) -> str:
        identity = self.to_value()
        identity.pop("memory_id")
        return "triviaqa-qa-memory-" + _stable_sha256(identity)

    def embedding_text(self) -> str:
        """Return the complete and exclusive embedding-model input."""

        return EMBEDDING_TEXT_TEMPLATE.format(
            paraphrase_question=self.paraphrase_question,
            paraphrase_answer_statement=self.paraphrase_answer_statement,
        )

    # SkillFlow RetrievalIndex protocol compatibility.  The QA-memory-aware
    # Tool adapter can additionally consume the explicit record attributes.
    @property
    def passage_id(self) -> str:
        return self.memory_id

    @property
    def document_id(self) -> str:
        return self.source_train_task_id

    @property
    def title(self) -> str:
        return self.paraphrase_question

    @property
    def text(self) -> str:
        return self.embedding_text()

    @property
    def paraphrase_provenance(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "paraphrase_method": self.paraphrase_method,
                "generator_provider": self.generator_provider,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "prompt_template_version": self.prompt_template_version,
                "generation_seed": self.generation_seed,
                "canonical_span_preserved": self.canonical_span_preserved,
            }
        )

    def to_value(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "tool_id": self.tool_id,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_train_task_id": self.source_train_task_id,
            "base_task_id": self.base_task_id,
            "selection_index": self.selection_index,
            "cycled_training_sample": self.cycled_training_sample,
            "cycle_index": self.cycle_index,
            "paraphrase_question": self.paraphrase_question,
            "paraphrase_answer_statement": self.paraphrase_answer_statement,
            "canonical_answer": self.canonical_answer,
            "paraphrase_version": self.paraphrase_version,
            "paraphrase_method": self.paraphrase_method,
            "generator_provider": self.generator_provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_template_version": self.prompt_template_version,
            "generation_seed": self.generation_seed,
            "canonical_span_preserved": self.canonical_span_preserved,
        }

    @classmethod
    def from_value(cls, value: object) -> "TriviaQAQAMemoryRecord":
        if not isinstance(value, Mapping) or set(value) != _MEMORY_FIELDS:
            raise ValueError("QA-memory row fields are incompatible")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class TriviaQAQAMemorySearchHit:
    """Ranked QA-memory hit with SkillFlow compatibility properties."""

    record: TriviaQAQAMemoryRecord
    rank: int
    similarity: float
    snippet: str

    def __post_init__(self) -> None:
        _positive_integer(self.rank, field_name="rank")
        if not isinstance(self.similarity, (int, float)) or not math.isfinite(
            float(self.similarity)
        ):
            raise ValueError("similarity must be finite")
        object.__setattr__(self, "similarity", float(self.similarity))
        object.__setattr__(
            self,
            "snippet",
            _required_text(self.snippet, field_name="snippet"),
        )

    @property
    def passage_id(self) -> str:
        return self.record.memory_id

    @property
    def document_id(self) -> str:
        return self.record.source_train_task_id

    @property
    def title(self) -> str:
        return self.record.paraphrase_question

    @property
    def memory_id(self) -> str:
        return self.record.memory_id

    @property
    def source_train_task_id(self) -> str:
        return self.record.source_train_task_id

    @property
    def base_task_id(self) -> str:
        return self.record.base_task_id

    @property
    def cycled_training_sample(self) -> bool:
        return self.record.cycled_training_sample

    @property
    def cycle_index(self) -> int | None:
        return self.record.cycle_index

    @property
    def paraphrase_question(self) -> str:
        return self.record.paraphrase_question

    @property
    def paraphrase_answer_statement(self) -> str:
        return self.record.paraphrase_answer_statement

    @property
    def canonical_answer(self) -> str:
        return self.record.canonical_answer

    @property
    def paraphrase_version(self) -> str:
        return self.record.paraphrase_version

    @property
    def paraphrase_provenance(self) -> Mapping[str, object]:
        return self.record.paraphrase_provenance

    def to_value(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "title": self.title,
            "snippet": self.snippet,
            "rank": self.rank,
            "similarity": self.similarity,
            "memory_id": self.memory_id,
            "source_train_task_id": self.source_train_task_id,
            "base_task_id": self.base_task_id,
            "cycled_training_sample": self.cycled_training_sample,
            "cycle_index": self.cycle_index,
            "paraphrase_question": self.paraphrase_question,
            "paraphrase_answer_statement": self.paraphrase_answer_statement,
            "canonical_answer": self.canonical_answer,
            "paraphrase_version": self.paraphrase_version,
            "paraphrase_provenance": dict(self.paraphrase_provenance),
        }


def load_materialized_qa_memory(
    path: str | Path,
    *,
    expected_count: int | None = None,
) -> tuple[TriviaQAQAMemoryRecord, ...]:
    """Load strict materialized paraphrases without admitting extra fields."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("materialized TriviaQA QA-memory JSONL is unavailable")
    records: list[TriviaQAQAMemoryRecord] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(TriviaQAQAMemoryRecord.from_value(value))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"QA-memory JSON is invalid at line {line_number}: {exc}"
                ) from exc
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} QA-memory rows, found {len(records)}"
        )
    memory_ids = [record.memory_id for record in records]
    source_ids = [record.source_train_task_id for record in records]
    selection_indices = [record.selection_index for record in records]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("QA-memory memory_id values are not unique")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("QA-memory source_train_task_id values are not unique")
    if len(set(selection_indices)) != len(selection_indices):
        raise ValueError("QA-memory selection_index values are not unique")
    if expected_count is not None and sorted(selection_indices) != list(
        range(expected_count)
    ):
        raise ValueError("QA-memory selection_index differs from frozen train order")
    return tuple(records)


def validate_qa_memory_against_sources(
    records: Sequence[TriviaQAQAMemoryRecord],
    sources: Sequence[TriviaQATrainSource],
    *,
    require_complete: bool,
) -> None:
    """Bind materialized paraphrases to the frozen train projection."""

    source_by_id = {source.source_train_task_id: source for source in sources}
    record_ids: set[str] = set()
    for record in records:
        if record.source_train_task_id in record_ids:
            raise ValueError("duplicate QA-memory source_train_task_id")
        record_ids.add(record.source_train_task_id)
        source = source_by_id.get(record.source_train_task_id)
        if source is None:
            raise ValueError("QA-memory row references a non-train source")
        comparisons = {
            "base_task_id": (record.base_task_id, source.base_task_id),
            "selection_index": (record.selection_index, source.selection_index),
            "cycled_training_sample": (
                record.cycled_training_sample,
                source.cycled_training_sample,
            ),
            "cycle_index": (record.cycle_index, source.cycle_index),
            "canonical_answer": (record.canonical_answer, source.canonical_answer),
        }
        for field_name, (actual, expected) in comparisons.items():
            if actual != expected:
                raise ValueError(
                    f"QA-memory {field_name} differs from frozen train source"
                )
        canonical = source.canonical_answer
        if (
            record.prompt_template_version
            in _STRICT_PARAPHRASE_PROMPT_VERSIONS
            and canonical.casefold() not in source.original_question.casefold()
            and canonical.casefold() in record.paraphrase_question.casefold()
            and not canonical_is_original_spelling_variant(
                source.original_question,
                source.canonical_answer,
            )
        ):
            raise ValueError(
                "QA-memory paraphrase_question introduced the canonical answer"
            )
    if require_complete and record_ids != set(source_by_id):
        raise ValueError("QA-memory rows do not exactly cover the frozen train split")


def write_materialized_qa_memory(
    path: str | Path,
    records: Sequence[TriviaQAQAMemoryRecord],
) -> None:
    """Persist materialized paraphrases atomically in frozen train order."""

    ordered = sorted(records, key=lambda record: record.selection_index)

    def writer(handle: Any) -> None:
        for record in ordered:
            handle.write(_canonical_json(record.to_value()) + b"\n")

    _write_atomic_bytes(Path(path), writer)


def _validated_tool_budget(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != _TOOL_BUDGET_FIELDS:
        raise ValueError("QA-memory tool_budget fields are incompatible")
    budget = {
        key: _positive_integer(value[key], field_name=f"tool_budget.{key}")
        for key in sorted(_TOOL_BUDGET_FIELDS)
    }
    if budget["max_turns_per_agent_call"] <= budget[
        "max_tool_calls_per_agent_call"
    ]:
        raise ValueError("ReAct turn budget must leave one completion turn")
    return MappingProxyType(budget)


def _validated_files(value: object) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _FILE_FIELDS:
        raise ValueError("QA-memory files manifest is incompatible")
    expected_names = {
        "memories": MEMORIES_FILENAME,
        "embeddings": EMBEDDINGS_FILENAME,
    }
    result: dict[str, Mapping[str, str]] = {}
    for key in sorted(_FILE_FIELDS):
        entry = value[key]
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_FIELDS:
            raise ValueError(f"QA-memory files.{key} is incompatible")
        name = _required_text(entry["name"], field_name=f"files.{key}.name")
        if name != expected_names[key]:
            raise ValueError(f"QA-memory files.{key}.name is incompatible")
        result[key] = MappingProxyType(
            {
                "name": name,
                "sha256": _required_sha256(
                    entry["sha256"], field_name=f"files.{key}.sha256"
                ),
            }
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class TriviaQAQAMemoryManifest:
    """Versioned identity of one frozen train-only QA-memory index."""

    schema_version: str
    format: str
    record_kind: str
    tool_id: str
    retrieval_backend: str
    index_id: str
    corpus_name: str
    corpus_version: str
    source_dataset: str
    source_split: str
    train_count: int
    validation_isolation_count: int
    validation_content_indexed: bool
    unique_source_count: int
    cycled_count: int
    paraphrase_count: int
    memory_count: int
    paraphrase_version: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    normalization: str
    similarity: str
    query_prefix: str
    embedding_text_template: str
    frozen_top_k: int
    snippet_characters: int
    tool_budget: Mapping[str, int]
    files: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        constants = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "format": INDEX_FORMAT,
            "record_kind": "qa_memory",
            "tool_id": QA_MEMORY_TOOL_ID,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "corpus_name": CORPUS_NAME,
            "source_dataset": SOURCE_DATASET,
            "source_split": SOURCE_PROJECT_SPLIT,
            "normalization": NORMALIZATION,
            "similarity": SIMILARITY,
            "query_prefix": BGE_QUERY_PREFIX,
            "embedding_text_template": EMBEDDING_TEXT_TEMPLATE,
        }
        for field_name, expected in constants.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"QA-memory manifest {field_name} is unsupported")
        if type(self.validation_content_indexed) is not bool:
            raise ValueError("validation_content_indexed must be boolean")
        train_count = _positive_integer(self.train_count, field_name="train_count")
        validation_count = _nonnegative_integer(
            self.validation_isolation_count,
            field_name="validation_isolation_count",
        )
        if not self.validation_content_indexed and validation_count < 1:
            raise ValueError(
                "held-out QA-memory requires positive validation isolation"
            )
        unique_count = _positive_integer(
            self.unique_source_count,
            field_name="unique_source_count",
        )
        cycled_count = _nonnegative_integer(
            self.cycled_count,
            field_name="cycled_count",
        )
        paraphrase_count = _positive_integer(
            self.paraphrase_count,
            field_name="paraphrase_count",
        )
        memory_count = _positive_integer(
            self.memory_count,
            field_name="memory_count",
        )
        if unique_count + cycled_count != train_count:
            raise ValueError("QA-memory train unique/cycled accounting does not close")
        if train_count != paraphrase_count or train_count != memory_count:
            raise ValueError("QA-memory train/paraphrase/memory counts differ")
        _required_text(self.paraphrase_version, field_name="paraphrase_version")
        _required_text(self.embedding_model, field_name="embedding_model")
        _required_text(
            self.embedding_model_revision,
            field_name="embedding_model_revision",
        )
        _positive_integer(
            self.embedding_dimension,
            field_name="embedding_dimension",
        )
        frozen_top_k = _positive_integer(
            self.frozen_top_k,
            field_name="frozen_top_k",
        )
        if frozen_top_k > memory_count:
            raise ValueError("frozen_top_k exceeds QA-memory count")
        _positive_integer(
            self.snippet_characters,
            field_name="snippet_characters",
        )
        object.__setattr__(self, "tool_budget", _validated_tool_budget(self.tool_budget))
        object.__setattr__(self, "files", _validated_files(self.files))
        memories_digest = self.files["memories"]["sha256"]
        if self.corpus_version != f"sha256:{memories_digest}":
            raise ValueError("QA-memory corpus_version differs from memories file")
        _required_sha256(self.index_id, field_name="index_id")
        if self.index_id != _stable_sha256(self._identity_value()):
            raise ValueError("QA-memory index_id differs from manifest identity")

    def _identity_value(self) -> dict[str, object]:
        value = self.to_value()
        value.pop("index_id")
        return value

    def to_value(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "record_kind": self.record_kind,
            "tool_id": self.tool_id,
            "retrieval_backend": self.retrieval_backend,
            "index_id": self.index_id,
            "corpus_name": self.corpus_name,
            "corpus_version": self.corpus_version,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "train_count": self.train_count,
            "validation_isolation_count": self.validation_isolation_count,
            "validation_content_indexed": self.validation_content_indexed,
            "unique_source_count": self.unique_source_count,
            "cycled_count": self.cycled_count,
            "paraphrase_count": self.paraphrase_count,
            "memory_count": self.memory_count,
            "paraphrase_version": self.paraphrase_version,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization,
            "similarity": self.similarity,
            "query_prefix": self.query_prefix,
            "embedding_text_template": self.embedding_text_template,
            "frozen_top_k": self.frozen_top_k,
            "snippet_characters": self.snippet_characters,
            "tool_budget": dict(self.tool_budget),
            "files": {key: dict(self.files[key]) for key in sorted(self.files)},
        }

    @classmethod
    def create(
        cls,
        *,
        train_count: int,
        validation_isolation_count: int,
        validation_content_indexed: bool = False,
        unique_source_count: int,
        cycled_count: int,
        paraphrase_version: str,
        embedding_model: str,
        embedding_model_revision: str,
        embedding_dimension: int,
        frozen_top_k: int,
        snippet_characters: int,
        max_tool_calls_per_agent_call: int,
        max_turns_per_agent_call: int,
        memories_sha256: str,
        embeddings_sha256: str,
    ) -> "TriviaQAQAMemoryManifest":
        value: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "format": INDEX_FORMAT,
            "record_kind": "qa_memory",
            "tool_id": QA_MEMORY_TOOL_ID,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "index_id": "0" * 64,
            "corpus_name": CORPUS_NAME,
            "corpus_version": f"sha256:{memories_sha256}",
            "source_dataset": SOURCE_DATASET,
            "source_split": SOURCE_PROJECT_SPLIT,
            "train_count": train_count,
            "validation_isolation_count": validation_isolation_count,
            "validation_content_indexed": validation_content_indexed,
            "unique_source_count": unique_source_count,
            "cycled_count": cycled_count,
            "paraphrase_count": train_count,
            "memory_count": train_count,
            "paraphrase_version": paraphrase_version,
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_model_revision,
            "embedding_dimension": embedding_dimension,
            "normalization": NORMALIZATION,
            "similarity": SIMILARITY,
            "query_prefix": BGE_QUERY_PREFIX,
            "embedding_text_template": EMBEDDING_TEXT_TEMPLATE,
            "frozen_top_k": frozen_top_k,
            "snippet_characters": snippet_characters,
            "tool_budget": {
                "max_tool_calls_per_agent_call": max_tool_calls_per_agent_call,
                "max_turns_per_agent_call": max_turns_per_agent_call,
            },
            "files": {
                "memories": {
                    "name": MEMORIES_FILENAME,
                    "sha256": memories_sha256,
                },
                "embeddings": {
                    "name": EMBEDDINGS_FILENAME,
                    "sha256": embeddings_sha256,
                },
            },
        }
        identity = dict(value)
        identity.pop("index_id")
        value["index_id"] = _stable_sha256(identity)
        return cls.from_value(value)

    @classmethod
    def from_value(cls, value: object) -> "TriviaQAQAMemoryManifest":
        fields = frozenset(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("QA-memory manifest fields are incompatible")
        return cls(**{name: value[name] for name in fields})


def build_triviaqa_qa_memory_index(
    *,
    paraphrases_path: str | Path,
    train_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    output_dir: str | Path,
    embedding_model: str,
    embedding_model_revision: str,
    frozen_top_k: int,
    max_tool_calls_per_agent_call: int,
    max_turns_per_agent_call: int,
    encoder: EmbeddingEncoder | None = None,
    batch_size: int = 64,
    snippet_characters: int = 512,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
    validation_content_indexed: bool = False,
) -> TriviaQAQAMemoryManifest:
    """Build an immutable dense index from already-materialized paraphrases."""

    model_name = _required_text(embedding_model, field_name="embedding_model")
    model_revision = _required_text(
        embedding_model_revision,
        field_name="embedding_model_revision",
    )
    frozen_top_k = _positive_integer(frozen_top_k, field_name="frozen_top_k")
    batch_size = _positive_integer(batch_size, field_name="batch_size")
    snippet_characters = _positive_integer(
        snippet_characters,
        field_name="snippet_characters",
    )
    max_tool_calls_per_agent_call = _positive_integer(
        max_tool_calls_per_agent_call,
        field_name="max_tool_calls_per_agent_call",
    )
    max_turns_per_agent_call = _positive_integer(
        max_turns_per_agent_call,
        field_name="max_turns_per_agent_call",
    )
    if max_turns_per_agent_call <= max_tool_calls_per_agent_call:
        raise ValueError("ReAct turn budget must leave one completion turn")
    if type(validation_content_indexed) is not bool:
        raise TypeError("validation_content_indexed must be boolean")
    sources, validation_ids = load_triviaqa_qa_memory_sources(
        train_tasks_path,
        validation_tasks_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
        allow_validation_overlap=validation_content_indexed,
    )
    records = load_materialized_qa_memory(
        paraphrases_path,
        expected_count=expected_train_count,
    )
    validate_qa_memory_against_sources(records, sources, require_complete=True)
    indexed_validation_ids = {
        record.base_task_id for record in records if record.base_task_id in validation_ids
    }
    if indexed_validation_ids and not validation_content_indexed:
        raise ValueError("QA-memory contains a held-out validation base_task_id")
    if validation_content_indexed and indexed_validation_ids != validation_ids:
        raise ValueError(
            "transductive QA-memory must index every declared evaluation base_task_id"
        )
    paraphrase_versions = {record.paraphrase_version for record in records}
    if len(paraphrase_versions) != 1:
        raise ValueError("QA-memory build requires one frozen paraphrase_version")
    if frozen_top_k > len(records):
        raise ValueError("frozen_top_k exceeds QA-memory count")

    ordered = tuple(sorted(records, key=lambda record: record.memory_id))
    resolved_encoder = encoder or _get_embedding_model(model_name, model_revision)
    embeddings = _normalized_embeddings(
        resolved_encoder,
        [record.embedding_text() for record in ordered],
        batch_size=batch_size,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    memories_path = root / MEMORIES_FILENAME
    embeddings_path = root / EMBEDDINGS_FILENAME
    manifest_path = root / MANIFEST_FILENAME

    def write_memories(handle: Any) -> None:
        for record in ordered:
            handle.write(_canonical_json(record.to_value()) + b"\n")

    _write_atomic_bytes(memories_path, write_memories)
    _write_atomic_bytes(
        embeddings_path,
        lambda handle: np.save(handle, embeddings, allow_pickle=False),
    )
    memories_sha256 = _file_sha256(memories_path)
    embeddings_sha256 = _file_sha256(embeddings_path)
    cycled_count = sum(source.cycled_training_sample for source in sources)
    manifest = TriviaQAQAMemoryManifest.create(
        train_count=len(sources),
        validation_isolation_count=len(validation_ids - indexed_validation_ids),
        validation_content_indexed=validation_content_indexed,
        unique_source_count=len({source.base_task_id for source in sources}),
        cycled_count=cycled_count,
        paraphrase_version=next(iter(paraphrase_versions)),
        embedding_model=model_name,
        embedding_model_revision=model_revision,
        embedding_dimension=int(embeddings.shape[1]),
        frozen_top_k=frozen_top_k,
        snippet_characters=snippet_characters,
        max_tool_calls_per_agent_call=max_tool_calls_per_agent_call,
        max_turns_per_agent_call=max_turns_per_agent_call,
        memories_sha256=memories_sha256,
        embeddings_sha256=embeddings_sha256,
    )
    _write_atomic_bytes(
        manifest_path,
        lambda handle: handle.write(_canonical_json(manifest.to_value()) + b"\n"),
    )
    return manifest


class TriviaQAQAMemoryIndex:
    """Immutable QA-memory index implementing the existing search/read protocol."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: TriviaQAQAMemoryManifest,
        records: tuple[TriviaQAQAMemoryRecord, ...],
        embeddings: np.ndarray,
        encoder: EmbeddingEncoder,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._records = records
        self._record_by_id = {record.memory_id: record for record in records}
        self._embeddings: np.ndarray | None = embeddings
        self._encoder = encoder
        self._lock = RLock()
        self._closed = False

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        encoder: EmbeddingEncoder | None = None,
    ) -> "TriviaQAQAMemoryIndex":
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError("TriviaQA QA-memory index directory is unavailable")
        manifest_path = root_path / MANIFEST_FILENAME
        memories_path = root_path / MEMORIES_FILENAME
        embeddings_path = root_path / EMBEDDINGS_FILENAME
        for path in (manifest_path, memories_path, embeddings_path):
            if not path.is_file():
                raise FileNotFoundError(f"QA-memory index file is unavailable: {path.name}")
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("QA-memory manifest JSON is invalid") from exc
        manifest = TriviaQAQAMemoryManifest.from_value(manifest_value)
        if _file_sha256(memories_path) != manifest.files["memories"]["sha256"]:
            raise ValueError("memories file differs from QA-memory manifest")
        if _file_sha256(embeddings_path) != manifest.files["embeddings"]["sha256"]:
            raise ValueError("embeddings file differs from QA-memory manifest")
        records = load_materialized_qa_memory(
            memories_path,
            expected_count=manifest.memory_count,
        )
        memory_ids = [record.memory_id for record in records]
        if memory_ids != sorted(memory_ids):
            raise ValueError("QA-memory rows must use canonical memory_id ordering")
        if len({record.base_task_id for record in records}) != manifest.unique_source_count:
            raise ValueError("QA-memory unique source count differs from manifest")
        if sum(record.cycled_training_sample for record in records) != manifest.cycled_count:
            raise ValueError("QA-memory cycled count differs from manifest")
        if {record.paraphrase_version for record in records} != {
            manifest.paraphrase_version
        }:
            raise ValueError("QA-memory paraphrase version differs from manifest")
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        if embeddings.dtype != np.dtype("float32"):
            raise ValueError("QA-memory embedding dtype must be float32")
        if embeddings.shape != (
            manifest.memory_count,
            manifest.embedding_dimension,
        ):
            raise ValueError("QA-memory embedding shape differs from manifest")
        if not np.isfinite(embeddings).all():
            raise ValueError("QA-memory embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("QA-memory embedding matrix is not l2-normalized")
        resolved_encoder = encoder or _get_embedding_model(
            manifest.embedding_model,
            manifest.embedding_model_revision,
        )
        return cls(
            root=root_path,
            manifest=manifest,
            records=records,
            embeddings=embeddings,
            encoder=resolved_encoder,
        )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open_embeddings(self) -> np.ndarray:
        if self._closed or self._embeddings is None:
            raise RuntimeError("TriviaQA QA-memory index is closed")
        return self._embeddings

    def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[TriviaQAQAMemorySearchHit, ...]:
        query_text = _required_text(query, field_name="query")
        if type(limit) is not int or limit != self.manifest.frozen_top_k:
            raise ValueError("search limit differs from the frozen QA-memory top-k")
        with self._lock:
            embeddings = self._require_open_embeddings()
            embedding_query = f"{self.manifest.query_prefix}Question: {query_text}"
            query_embedding = _normalized_embeddings(
                self._encoder,
                [embedding_query],
                batch_size=1,
            )[0]
            scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
            order = sorted(
                range(len(self._records)),
                key=lambda index: (
                    -float(scores[index]),
                    self._records[index].memory_id,
                ),
            )[:limit]
            return tuple(
                TriviaQAQAMemorySearchHit(
                    record=self._records[index],
                    snippet=self._records[index].paraphrase_answer_statement[
                        : self.manifest.snippet_characters
                    ],
                    rank=rank,
                    similarity=float(scores[index]),
                )
                for rank, index in enumerate(order, start=1)
            )

    def read(self, memory_id: str) -> TriviaQAQAMemoryRecord:
        resolved_id = _required_text(memory_id, field_name="memory_id")
        with self._lock:
            self._require_open_embeddings()
            try:
                return self._record_by_id[resolved_id]
            except KeyError as exc:
                raise KeyError("unknown TriviaQA QA-memory memory_id") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            embeddings = self._embeddings
            self._embeddings = None
            self._closed = True
            mmap = getattr(embeddings, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "TriviaQAQAMemoryIndex":
        if self.closed:
            raise RuntimeError("TriviaQA QA-memory index is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "CORPUS_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_TEXT_TEMPLATE",
    "EMBEDDINGS_FILENAME",
    "INDEX_FORMAT",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "MEMORIES_FILENAME",
    "MEMORY_SCHEMA_VERSION",
    "QA_MEMORY_TOOL_ID",
    "TriviaQAQAMemoryIndex",
    "TriviaQAQAMemoryManifest",
    "TriviaQAQAMemoryRecord",
    "TriviaQAQAMemorySearchHit",
    "TriviaQATrainSource",
    "build_triviaqa_qa_memory_index",
    "deterministic_answer_statement",
    "relation_bearing_answer_statement",
    "load_materialized_qa_memory",
    "load_triviaqa_qa_memory_sources",
    "validate_qa_memory_against_sources",
    "write_materialized_qa_memory",
]
