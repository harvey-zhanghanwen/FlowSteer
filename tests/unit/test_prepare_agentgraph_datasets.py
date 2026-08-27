import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "prepare_agentgraph_datasets.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_agentgraph_datasets", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TASK_SCHEMA_VERSION = _MODULE.TASK_SCHEMA_VERSION
_compat_record = _MODULE._compat_record
_conversation_prompt = _MODULE._conversation_prompt
_hotpot_records = _MODULE._hotpot_records
_annotate_hotpotqa_training_source_anomaly = (
    _MODULE._annotate_hotpotqa_training_source_anomaly
)
_uniform_sample = _MODULE._uniform_sample


def test_compat_record_exposes_both_upstream_field_sets():
    record = _compat_record(
        dataset_key="triviaqa",
        source="TriviaQA",
        task_id="triviaqa:1",
        question="Question?",
        ground_truth="answer | alias",
        split="validation",
        task_type="factual_qa",
        metric="token_f1",
        evaluator_payload={"accepted_answers": ["answer", "alias"]},
    )
    assert record["schema_version"] == TASK_SCHEMA_VERSION
    assert record["ground_truth"] == record["answer"]
    assert record["metadata"]["source"] == record["extra"]["source"]
    assert "accepted_answers" not in record["question"]


def test_healthbench_prompt_contains_only_conversation_messages():
    prompt = _conversation_prompt(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ]
        }
    )
    assert prompt == (
        "Conversation:\n\n[user] first\n\n[assistant] second\n\n"
        "[user] third\n\n[assistant]"
    )


def test_hotpot_alignment_keeps_evidence_after_300_characters(monkeypatch, tmp_path):
    tail = "decisive evidence after the former boundary"
    row = {
        "id": "item-1",
        "question": "Who wrote it?",
        "answer": "Author",
        "context": {
            "title": ["Book"],
            "sentences": [["x" * 350, tail]],
        },
    }
    monkeypatch.setattr(_MODULE, "_iter_parquet_rows", lambda _: iter([row]))
    config = {
        "path": str(tmp_path),
        "display_name": "HotpotQA",
        "task_type": "multi_hop_qa",
        "metric": "token_f1",
        "files": {"train": "unused.parquet"},
    }

    record = next(_hotpot_records(config))

    assert tail in record["question"]
    assert tail in record["context"][0]


def _malformed_binary_hotpot_record(task_id: str):
    return _compat_record(
        dataset_key="hotpotqa",
        source="HotpotQA",
        task_id=task_id,
        question=(
            "Based on the following passages, answer the question.\n\n"
            "[[Alpha] Alpha is a record producer.]\n\n"
            "[[Beta] Beta is a professional wrestler.]\n\n"
            "Question: Are Alpha and Beta both record producers?"
        ),
        ground_truth="Beta is a professional wrestler",
        split="train",
        task_type="multi_hop_qa",
        metric="token_f1",
        context=(
            "[Alpha] Alpha is a record producer.",
            "[Beta] Beta is a professional wrestler.",
        ),
        extra={"type": "comparison", "level": "hard"},
        evaluator_payload={
            "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 0]}
        },
    )


def test_hotpot_training_binary_source_anomaly_is_generic_and_does_not_relabel():
    annotated = _annotate_hotpotqa_training_source_anomaly(
        _malformed_binary_hotpot_record("hotpotqa:source-defect")
    )

    assert annotated["ground_truth"] == "Beta is a professional wrestler"
    assert annotated["answer"] == "Beta is a professional wrestler"
    annotation = annotated["metadata"]["source_answer_annotation"]
    assert annotation["status"] == "official_source_annotation_anomaly"
    assert (
        annotation["rule"]
        == "hotpotqa.training.binary_both_nonbinary_source_answer.v1"
    )


def test_hotpot_source_alignment_transform_never_changes_heldout_record():
    heldout_source = _malformed_binary_hotpot_record("hotpotqa:heldout")
    training_source = _malformed_binary_hotpot_record("hotpotqa:train")

    heldout, train, unique_count = _uniform_sample(
        [heldout_source, training_source],
        heldout_split="validation",
        heldout_count=1,
        train_count=1,
        train_transform=_annotate_hotpotqa_training_source_anomaly,
    )

    assert unique_count == 1
    assert heldout[0]["ground_truth"] == "Beta is a professional wrestler"
    assert "source_answer_annotation" not in heldout[0]["metadata"]
    assert train[0]["ground_truth"] == "Beta is a professional wrestler"
    assert (
        train[0]["metadata"]["source_answer_annotation"]["status"]
        == "official_source_annotation_anomaly"
    )
