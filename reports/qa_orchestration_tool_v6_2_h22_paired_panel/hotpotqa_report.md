# HotpotQA Architecture Validation

Fixed validation samples: **6**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`HotpotQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **6/6**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 6 | 6 | 66.67% | 75.00% |
| AgentGraph | 6 | 6 | 50.00% | 58.33% |

AgentGraph - Direct: **-16.67 EM**, **-16.67 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 1
- `direct_higher_exact_match`: 2
- `equal_exact_match`: 3
