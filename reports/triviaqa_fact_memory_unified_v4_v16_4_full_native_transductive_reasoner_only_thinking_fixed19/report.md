# TriviaQA Architecture Validation

Fixed validation samples: **19**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **19/19**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 19 | 19 | 26.32% | 32.46% |
| AgentGraph | 19 | 19 | 94.74% | 98.25% |

AgentGraph - Direct: **+68.42 EM**, **+65.79 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `accepted_answer_canonicalization_mismatch`: 1
- `agentgraph_higher_exact_match`: 13
- `equal_exact_match`: 5
