# HealthBench Professional Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **raw_score** (`OpenAI_simple_evals_HealthBench_rubric_raw_score`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict raw_score |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 13.18% |
| AgentGraph | 128 | 128 | 20.75% |

AgentGraph - Direct: **+7.57 percentage points**.

## Failure types

- `agentgraph_higher_raw_score`: 35
- `direct_higher_raw_score`: 19
- `equal_raw_score`: 74
