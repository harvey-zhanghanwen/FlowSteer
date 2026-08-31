# HealthBench Professional Architecture Validation

Public test samples: **20**. No training, GRPO, backward pass, optimizer update, LoRA publication, MACE, Bayesian update, or Skill evolution ran. No Skill was injected.

Primary metric: **overall_score_length_adjusted** using the OpenAI simple-evals HealthBench Professional reference protocol. AgentGraph explicit FINISH: **20/20**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict raw score | Strict length-adjusted score | Valid-only length-adjusted score |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 20 | 20 | 27.72% | 21.56% | 21.56% |
| AgentGraph | 20 | 20 | 44.18% | 39.98% | 39.98% |

AgentGraph - Direct (strict length-adjusted): **+18.42 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

Detailed fixed20 architecture, topology, per-task, recovery, and Wrong Demo
analysis: [`optimization_report_zh.md`](optimization_report_zh.md).

## Failure types

- `agentgraph_higher_overall_score_length_adjusted`: 11
- `direct_higher_overall_score_length_adjusted`: 9
