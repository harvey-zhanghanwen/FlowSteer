# TriviaQA v6.2 architecture backup

Backup branch: `backup/triviaqa-v6.2-20260821`.

This backup contains the effective TriviaQA v6.2 evaluation-only adaptation of
the existing FlowSteer/SkillFlow runtime, its fixed development report, source
mapping, and the summary of one rejected v6.3 cache-invalidation candidate. It
does not contain credentials, model weights, training, optimizer updates, or
the large per-task runtime artifacts.

## Population and evaluator

- Fixed population: the first 128 TriviaQA records in the `joint_qa_v2`
  development partition; these are exposed architecture-development examples,
  not held-out test.
- Evaluator: `triviaqa.official.answer.v1`, accepted-answer maximum Exact Match
  and token F1 after official TriviaQA normalization.
- Direct condition: local Qwen3.5-9B question-only receipts reused from the
  same fixed task IDs.

## Effective result

- Qwen3.5-9B Direct: 45/128, EM 35.15625%, token F1 40.81597%.
- AgentGraph v6.2: 66/128, EM 51.5625%, token F1 60.10441%.
- Explicit FINISH: 116/128; evaluator-valid: 128/128; collection failures: 0.
- No GRPO, LoRA, backward, optimizer update, policy sync, Skill retrieval, or
  Skill injection occurred in this run.

The rejected v6.3 development panel held EM at 69.2308% and raised F1 by
2.1978 points on 13 affected tasks, but reduced FINISH from 13/13 to 12/13.
Its source change was reverted; its config/report is retained only as rejection
evidence and is not the effective architecture.

## Restore and validate

Prepare the frozen selection without starting a model:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_triviaqa_orchestration_tool_v6_2.yaml \
  --prepare-only
```

After restoring `.env`, the local Qwen3.5-9B Supervisor and the external
Step-0 adapter, run Stable Zero and the fixed development round:

```bash
set -a
source .env
set +a

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_triviaqa_orchestration_tool_v6_2.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_triviaqa_orchestration_tool_v6_2.yaml
```

## External artifacts not stored in Git

- `.env`
- local Qwen3.5-9B model and tokenizer
- `artifacts/hotpotqa_multiagent_skill/policy_step_000000/theta`
- `artifacts/qa_orchestration_tool_v6_2_development/triviaqa/`
- `artifacts/triviaqa_format_predecessor_v6_3_panel/`

The corresponding compact JSON/Markdown reports remain stored under
`reports/`; no API key is present in the backup.
