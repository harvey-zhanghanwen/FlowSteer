# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **22/30**; terminal failures: **5**; operational/evaluator failures: **3**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 4**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 27 | 22 | 26.67% |

AgentGraph - Direct: **+6.67 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 7
- `agentgraph_accuracy_regression`: 3
- `agentgraph_operational_or_evaluator_failure`: 3
- `agentgraph_terminal_failure`: 5
- `both_correct`: 1
- `both_incorrect`: 7
- `output_parsing_failure`: 4
