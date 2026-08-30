# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **30/30**; terminal failures: **0**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 30 | 30 | 26.67% |

AgentGraph - Direct: **+6.67 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 5
- `agentgraph_accuracy_regression`: 3
- `both_correct`: 3
- `both_incorrect`: 19
