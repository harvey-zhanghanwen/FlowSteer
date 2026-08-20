# TriviaQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **116/128**; terminal failures: **12**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 35.16% | 40.82% |
| AgentGraph | 128 | 128 | 51.56% | 60.10% |

AgentGraph - Direct: **+16.41 EM**, **+19.29 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_exact_match`: 28
- `agentgraph_terminal_failure`: 12
- `direct_higher_exact_match`: 4
- `equal_exact_match`: 84
