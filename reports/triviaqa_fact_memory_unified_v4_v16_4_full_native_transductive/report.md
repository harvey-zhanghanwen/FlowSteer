# TriviaQA Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native metrics: **exact_match** and **token_f1** (`TriviaQA_official_normalization_exact_match_and_token_F1`). AgentGraph explicit FINISH: **110/128**; terminal failures: **18**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 35.16% | 40.82% |
| AgentGraph | 128 | 128 | 82.81% | 84.94% |

AgentGraph - Direct: **+47.66 EM**, **+44.12 F1**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `accepted_answer_canonicalization_mismatch`: 5
- `agentgraph_higher_exact_match`: 66
- `agentgraph_terminal_failure`: 1
- `equal_exact_match`: 39
- `knowledge_base_coverage_failure`: 2
- `relation_or_answer_slot_binding_failure`: 12
- `structured_output_or_format_failure`: 3
