# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 32.69 | 14.06% |
| AgentGraph | 128 | 128 | 62.82 | 35.94% |

AgentGraph - Direct: **+30.13 Average Score**, **+21.88 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 894 | 876 | 18 | 0 | 63 | 65 | 0 | 0 |
| AgentGraph | 744 | 744 | 0 | 0 | 110 | 18 | 0 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 894 | 876 | 18 | 63 |
| AgentGraph | 128 | 744 | 744 | 0 | 110 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 52, '2': 68, '3': 5, '4': 1, '5': 1, '6': 1}`
- Relation count distribution: `{'0': 52, '1': 68, '2': 3, '3': 3, '5': 1, '6': 1}`
- Topology distribution: `{'fan_in': 2, 'mixed': 3, 'serial_2': 69, 'serial_3_plus': 2, 'single': 52}`
- Director `max_rounds`: **0**
- Runtime failed turns: **0**
- Runtime failure types: `{}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **0**

## Failure types

- `agentgraph_higher_average_score`: 72
- `direct_higher_average_score`: 10
- `equal_average_score`: 46
