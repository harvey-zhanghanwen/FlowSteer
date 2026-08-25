# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **26/30**; terminal failures: **2**; operational/evaluator failures: **2**.

Terminal-output parsing failures: **Direct 22**, **AgentGraph 1**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 20.00% |
| AgentGraph | 28 | 26 | 33.33% |

AgentGraph - Direct: **+13.33 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_accuracy_gain`: 7
- `agentgraph_accuracy_regression`: 2
- `agentgraph_operational_or_evaluator_failure`: 2
- `agentgraph_terminal_failure`: 2
- `both_correct`: 3
- `both_incorrect`: 13
- `output_parsing_failure`: 1
