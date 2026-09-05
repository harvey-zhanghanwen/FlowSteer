# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 14.84% |
| AgentGraph | 128 | 128 | 64.77 | 33.59% |

AgentGraph - Direct: **+30.90 Average Score**, **+18.75 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 0 | 65 | 63 | 0 | 0 |
| AgentGraph | 594 | 594 | 0 | 0 | 115 | 13 | 0 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 885 | 871 | 14 | 65 |
| AgentGraph | 128 | 594 | 594 | 0 | 115 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 128}`
- Relation count distribution: `{'0': 128}`
- Topology distribution: `{'single': 128}`
- Director `max_rounds`: **0**
- Runtime failed turns: **0**
- Runtime failure types: `{}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **0**

## Failure types

- `agentgraph_higher_average_score`: 69
- `direct_higher_average_score`: 6
- `equal_average_score`: 53
