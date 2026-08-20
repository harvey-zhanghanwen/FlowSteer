# AIME 2026 Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **exact_match** (`SkillFlow_exact_answer_extraction_and_exact_match`). AgentGraph explicit FINISH: **108/128**; terminal failures: **20**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict exact_match |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 5.47% |
| AgentGraph | 128 | 128 | 48.44% |

AgentGraph - Direct: **+42.97 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_exact_match_gain`: 57
- `agentgraph_exact_match_regression`: 2
- `agentgraph_terminal_failure`: 20
- `both_exact`: 5
- `both_incorrect`: 44
