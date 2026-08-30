# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **28/30**; terminal failures: **1**; operational/evaluator failures: **1**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 29 | 28 | 33.33% |

AgentGraph - Direct: **+13.33 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 7
- `agentgraph_accuracy_regression`: 2
- `agentgraph_operational_or_evaluator_failure`: 1
- `agentgraph_terminal_failure`: 1
- `both_correct`: 3
- `both_incorrect`: 16
