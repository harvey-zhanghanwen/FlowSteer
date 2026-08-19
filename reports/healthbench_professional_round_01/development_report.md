# HealthBench Professional Architecture Validation

Fixed train samples: **32**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **raw_score** (`OpenAI_simple_evals_HealthBench_rubric_raw_score`). AgentGraph explicit FINISH: **32/32**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict raw_score |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 32 | 32 | 10.75% |
| AgentGraph | 32 | 32 | 15.33% |

AgentGraph - Direct: **+4.58 percentage points**.

## Failure types

- `agentgraph_higher_raw_score`: 4
- `direct_higher_raw_score`: 4
- `equal_raw_score`: 24
