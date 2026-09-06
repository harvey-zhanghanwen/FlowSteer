# AIME 2026 Architecture Validation

Fixed test samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **accuracy** (`SkillEval_canonicalized_integer_exact_accuracy`). AgentGraph explicit FINISH: **27/30**; terminal failures: **3**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 11**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 63.33% |
| AgentGraph | 30 | 27 | 80.00% |

AgentGraph - Direct: **+16.67 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_accuracy_gain`: 6
- `agentgraph_terminal_failure`: 3
- `both_correct`: 18
- `both_incorrect`: 3
