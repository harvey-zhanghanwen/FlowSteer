# HealthBench Professional Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **raw_score** (`OpenAI_simple_evals_HealthBench_rubric_raw_score`). AgentGraph explicit FINISH: **123/128**; terminal failures: **1**; operational/evaluator failures: **5**.

| Condition | Completed | Evaluator valid | Strict raw_score |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 123 | 123 | 16.53% |
| AgentGraph | 124 | 124 | 17.09% |

AgentGraph - Direct: **+0.56 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_raw_score`: 21
- `agentgraph_terminal_failure`: 1
- `direct_higher_raw_score`: 22
- `direct_operational_or_evaluator_failure`: 5
- `equal_raw_score`: 79
