# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 14.84% |
| AgentGraph | 128 | 128 | 50.34 | 14.06% |

AgentGraph - Direct: **+16.47 Average Score**, **-0.78 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 0 | 65 | 63 | 0 | 0 |
| AgentGraph | 610 | 457 | 153 | 0 | 109 | 19 | 0 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 885 | 871 | 14 | 65 |
| AgentGraph | 128 | 610 | 457 | 153 | 109 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 128}`
- Relation count distribution: `{'0': 128}`
- Topology distribution: `{'single': 128}`
- Director `max_rounds`: **0**
- Runtime failed turns: **134**
- Runtime failure types: `{'EnvironmentExecutionError': 128, 'OpenAICompatibleGatewayError': 6}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **0**

## Failure types

- `agentgraph_higher_average_score`: 57
- `direct_higher_average_score`: 21
- `equal_average_score`: 50
