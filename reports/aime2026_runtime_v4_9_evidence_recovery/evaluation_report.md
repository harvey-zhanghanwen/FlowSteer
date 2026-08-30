# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **17/30**; terminal failures: **3**; operational/evaluator failures: **10**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 20 | 17 | 13.33% |

AgentGraph - Direct: **-6.67 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 2
- `agentgraph_operational_or_evaluator_failure`: 10
- `agentgraph_terminal_failure`: 3
- `both_correct`: 2
- `both_incorrect`: 13
