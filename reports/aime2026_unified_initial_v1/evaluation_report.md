# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **25/30**; terminal failures: **4**; operational/evaluator failures: **1**.

Terminal-output parsing failures: **Direct 20**, **AgentGraph 5**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 30.00% |
| AgentGraph | 29 | 25 | 13.33% |

AgentGraph - Direct: **-16.67 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 1
- `agentgraph_accuracy_regression`: 3
- `agentgraph_operational_or_evaluator_failure`: 1
- `agentgraph_terminal_failure`: 4
- `both_correct`: 3
- `both_incorrect`: 13
- `output_parsing_failure`: 5
