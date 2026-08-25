# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **22/30**; terminal failures: **5**; operational/evaluator failures: **3**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 5**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 27 | 22 | 23.33% |

AgentGraph - Direct: **+3.33 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 6
- `agentgraph_accuracy_regression`: 3
- `agentgraph_operational_or_evaluator_failure`: 3
- `agentgraph_terminal_failure`: 5
- `both_correct`: 1
- `both_incorrect`: 7
- `output_parsing_failure`: 5
