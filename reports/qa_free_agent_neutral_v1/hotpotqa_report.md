# HotpotQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`HotpotQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **127/128**; terminal failures: **1**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 70.31% | 78.68% |
| AgentGraph | 128 | 128 | 68.75% | 79.32% |

AgentGraph - Direct: **-1.56 EM**, **+0.64 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 6
- `agentgraph_terminal_failure`: 1
- `direct_higher_exact_match`: 7
- `equal_exact_match`: 114
