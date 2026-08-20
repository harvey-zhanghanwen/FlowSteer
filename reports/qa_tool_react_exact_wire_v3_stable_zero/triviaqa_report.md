# TriviaQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **125/128**; terminal failures: **3**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 35.16% | 40.82% |
| AgentGraph | 128 | 128 | 50.78% | 63.36% |

AgentGraph - Direct: **+15.62 EM**, **+22.55 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 26
- `agentgraph_terminal_failure`: 3
- `direct_higher_exact_match`: 5
- `equal_exact_match`: 94
