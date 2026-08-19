# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **125/128**; terminal failures: **1**; operational/evaluator failures: **2**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 24.22% |
| AgentGraph | 126 | 126 | 22.66% |

AgentGraph - Direct: **-1.56 percentage points**.

## Failure types

- `agentgraph_higher_success`: 12
- `agentgraph_operational_or_evaluator_failure`: 2
- `agentgraph_terminal_failure`: 1
- `direct_higher_success`: 14
- `equal_success`: 99
