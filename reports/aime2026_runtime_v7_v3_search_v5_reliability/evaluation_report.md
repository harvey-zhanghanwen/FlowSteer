# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **29/30**; terminal failures: **1**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 30 | 29 | 43.33% |

AgentGraph - Direct: **+23.33 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 11
- `agentgraph_accuracy_regression`: 3
- `agentgraph_terminal_failure`: 1
- `both_correct`: 2
- `both_incorrect`: 13
