# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **126/128**; terminal failures: **1**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 14.84% |
| AgentGraph | 127 | 126 | 36.85 | 14.06% |

AgentGraph - Direct: **+2.99 Average Score**, **-0.78 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 0 | 65 | 63 | 0 | 0 |
| AgentGraph | 912 | 912 | 0 | 9 | 72 | 54 | 1 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 885 | 871 | 14 | 65 |
| AgentGraph | 127 | 921 | 921 | 0 | 72 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 127}`
- Relation count distribution: `{'0': 127}`
- Topology distribution: `{'single': 127}`
- Director `max_rounds`: **1**
- Runtime failed turns: **249**
- Runtime failure types: `{'EnvironmentExecutionError': 129, 'OpenAICompatibleGatewayError': 120}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **2**

## Failure types

- `agentgraph_higher_average_score`: 36
- `agentgraph_operational_or_evaluator_failure`: 1
- `agentgraph_terminal_failure`: 1
- `direct_higher_average_score`: 29
- `equal_average_score`: 61
