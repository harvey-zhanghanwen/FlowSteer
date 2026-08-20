# HotpotQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`HotpotQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **125/128**; terminal failures: **3**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 70.31% | 78.68% |
| AgentGraph | 128 | 128 | 72.66% | 82.05% |

AgentGraph - Direct: **+2.34 EM**, **+3.37 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 14
- `agentgraph_terminal_failure`: 3
- `direct_higher_exact_match`: 8
- `equal_exact_match`: 103
