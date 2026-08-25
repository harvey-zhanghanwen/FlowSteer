# TriviaQA Architecture Validation

Fixed validation samples: **1**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **0/1**; terminal failures: **0**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 1 | 1 | 0.00% | 0.00% |
| AgentGraph | 0 | 0 | 0.00% | 0.00% |

AgentGraph - Direct: **+0.00 EM**, **+0.00 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_operational_or_evaluator_failure`: 1
