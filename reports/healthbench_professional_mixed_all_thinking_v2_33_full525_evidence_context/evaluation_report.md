# HealthBench Professional Architecture Validation

Public test samples: **525**. No training, GRPO, backward pass, optimizer update, LoRA publication, MACE, Bayesian update, or Skill evolution ran. No Skill was injected.

Primary metric: **overall_score_length_adjusted** using the OpenAI simple-evals HealthBench Professional reference protocol. AgentGraph explicit FINISH: **460/525**; terminal failures: **0**; operational/evaluator failures: **117**.

| Condition | Completed | Evaluator valid | Strict raw score | Strict length-adjusted score | Valid-only length-adjusted score |
|---|---:|---:|---:|---:|---:|
| Single-Agent ReAct + MedRAG | 459 | 459 | 13.97% | 16.76% | 19.17% |
| Free AgentGraph + MedRAG | 460 | 460 | 29.52% | 27.46% | 31.34% |

AgentGraph - Direct (strict length-adjusted): **+10.70 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_overall_score_length_adjusted`: 199
- `agentgraph_operational_or_evaluator_failure`: 51
- `direct_higher_overall_score_length_adjusted`: 209
- `direct_operational_or_evaluator_failure`: 66
