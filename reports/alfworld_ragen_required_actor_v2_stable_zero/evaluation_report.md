# ALFWorld Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **107/128**; terminal failures: **0**; operational/evaluator failures: **21**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 21.09% |
| AgentGraph | 107 | 107 | 20.31% |

AgentGraph - Direct: **-0.78 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 13
- `agentgraph_operational_or_evaluator_failure`: 21
- `direct_higher_success`: 12
- `equal_success`: 82
