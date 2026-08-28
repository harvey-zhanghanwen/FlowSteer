# 全量 TriviaQA Q–A memory 条件下的直接准确率

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **122/128**; terminal failures: **6**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B 闭卷 Direct（question-only 对照） | 128 | 128 | 35.16% | 40.82% |
| 全量 TriviaQA Q–A memory + AgentGraph Tool 检索直接准确率 | 128 | 128 | 95.31% | 95.31% |

QA-memory AgentGraph − 闭卷 Direct: **+60.16 EM**, **+54.50 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 81
- `equal_exact_match`: 41
- `relation_or_answer_slot_binding_failure`: 4
- `structured_output_or_format_failure`: 2
