# HealthBench Professional Architecture Validation

Public test samples: **525**. No training, GRPO, backward pass, optimizer update, LoRA publication, MACE, Bayesian update, or Skill evolution ran. No Skill was injected.

Primary metric: **overall_score_length_adjusted** using the OpenAI simple-evals HealthBench Professional reference protocol. AgentGraph explicit FINISH: **525/525**; terminal failures: **0**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Strict raw score | Strict length-adjusted score | Valid-only length-adjusted score |
|---|---:|---:|---:|---:|---:|
| Single-Agent ReAct + MedRAG | 524 | 524 | 22.14% | 23.81% | 23.85% |
| Free AgentGraph + MedRAG | 525 | 525 | 17.59% | 17.72% | 17.72% |

AgentGraph - Direct (strict length-adjusted): **-6.09 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_overall_score_length_adjusted`: 219
- `direct_higher_overall_score_length_adjusted`: 305
- `direct_operational_or_evaluator_failure`: 1
