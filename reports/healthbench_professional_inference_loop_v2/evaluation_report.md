# HealthBench Professional Architecture Validation

Public test samples: **525**. No training, GRPO, backward pass, optimizer update, LoRA publication, MACE, Bayesian update, or Skill evolution ran. No Skill was injected.

Primary metric: **overall_score_length_adjusted** using the OpenAI simple-evals HealthBench Professional reference protocol. AgentGraph explicit FINISH: **488/525**; terminal failures: **0**; operational/evaluator failures: **52**.

| Condition | Completed | Evaluator valid | Strict raw score | Strict length-adjusted score | Valid-only length-adjusted score |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 525 | 510 | 16.78% | 17.30% | 17.81% |
| AgentGraph | 488 | 488 | 20.53% | 18.33% | 19.72% |

AgentGraph - Direct (strict length-adjusted): **+1.03 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_overall_score_length_adjusted`: 170
- `agentgraph_operational_or_evaluator_failure`: 37
- `direct_higher_overall_score_length_adjusted`: 302
- `direct_operational_or_evaluator_failure`: 15
- `equal_overall_score_length_adjusted`: 1
