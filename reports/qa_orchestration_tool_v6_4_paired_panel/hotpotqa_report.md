# HotpotQA Architecture Validation

Fixed validation samples: **16**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`HotpotQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **15/16**; terminal failures: **1**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 16 | 16 | 56.25% | 74.38% |
| AgentGraph | 16 | 16 | 37.50% | 60.62% |

AgentGraph - Direct: **-18.75 EM**, **-13.75 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_terminal_failure`: 1
- `direct_higher_exact_match`: 2
- `equal_exact_match`: 13
