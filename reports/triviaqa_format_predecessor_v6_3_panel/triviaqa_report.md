# TriviaQA Architecture Validation

Fixed validation samples: **13**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **12/13**; terminal failures: **1**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 13 | 13 | 46.15% | 50.00% |
| AgentGraph | 13 | 13 | 69.23% | 73.63% |

AgentGraph - Direct: **+23.08 EM**, **+23.63 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 4
- `agentgraph_terminal_failure`: 1
- `equal_exact_match`: 8
