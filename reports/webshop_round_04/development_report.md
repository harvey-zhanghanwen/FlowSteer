# WebShop Architecture Validation

Fixed train samples: **16**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **16/16**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 16 | 16 | 25.00% |
| AgentGraph | 16 | 16 | 31.25% |

AgentGraph - Direct: **+6.25 percentage points**.

## Failure types

- `agentgraph_higher_success`: 2
- `direct_higher_success`: 1
- `equal_success`: 13
