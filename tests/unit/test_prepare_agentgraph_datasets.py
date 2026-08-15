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
