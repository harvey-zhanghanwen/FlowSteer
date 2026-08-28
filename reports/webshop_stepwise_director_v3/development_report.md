# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **127/128**; terminal failures: **0**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 14.84% |
| AgentGraph | 127 | 127 | 39.89 | 14.84% |

AgentGraph - Direct: **+6.02 Average Score**, **+0.00 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 0 | 65 | 63 | 0 | 0 |
| AgentGraph | 916 | 916 | 0 | 0 | 75 | 52 | 0 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 885 | 871 | 14 | 65 |
| AgentGraph | 127 | 916 | 916 | 0 | 75 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 127}`
- Relation count distribution: `{'0': 127}`
- Topology distribution: `{'single': 127}`
- Director `max_rounds`: **0**
- Runtime failed turns: **247**
- Runtime failure types: `{'EnvironmentExecutionError': 130, 'OpenAICompatibleGatewayError': 117}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **5**

## Failure types

- `agentgraph_higher_average_score`: 37
- `agentgraph_operational_or_evaluator_failure`: 1
- `direct_higher_average_score`: 25
- `equal_average_score`: 65
