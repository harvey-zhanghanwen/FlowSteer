# TriviaQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **112/128**; terminal failures: **16**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 35.16% | 40.82% |
| AgentGraph | 128 | 128 | 87.50% | 87.50% |

AgentGraph - Direct: **+52.34 EM**, **+46.68 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 71
- `equal_exact_match`: 41
- `relation_or_answer_slot_binding_failure`: 15
- `structured_output_or_format_failure`: 1
