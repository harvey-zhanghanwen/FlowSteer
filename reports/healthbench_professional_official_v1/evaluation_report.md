# HealthBench Professional Architecture Validation

Public test samples: **525**. No training, GRPO, backward pass, optimizer update, LoRA publication, MACE, Bayesian update, or Skill evolution ran. No Skill was injected.

Primary metric: **overall_score_length_adjusted** using the OpenAI simple-evals HealthBench Professional reference protocol. AgentGraph explicit FINISH: **503/525**; terminal failures: **22**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict raw score | Strict length-adjusted score | Valid-only length-adjusted score |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 525 | 525 | 18.97% | 19.17% | 19.17% |
| AgentGraph | 525 | 503 | 22.65% | 20.24% | 21.12% |

AgentGraph - Direct (strict length-adjusted): **+1.07 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_overall_score_length_adjusted`: 197
- `agentgraph_terminal_failure`: 22
- `direct_higher_overall_score_length_adjusted`: 302
- `equal_overall_score_length_adjusted`: 4
