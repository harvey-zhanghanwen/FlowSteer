# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 127 | 19.53% |
| AgentGraph | 128 | 128 | 16.41% |

AgentGraph - Direct: **-3.12 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 5
- `direct_higher_success`: 9
- `direct_operational_or_evaluator_failure`: 1
- `equal_success`: 113
