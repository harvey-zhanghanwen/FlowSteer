# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **125/128**; terminal failures: **2**; operational/evaluator failures: **1**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 14.84% |
| AgentGraph | 127 | 125 | 34.67 | 13.28% |

AgentGraph - Direct: **+0.80 Average Score**, **-1.56 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 0 | 65 | 63 | 0 | 0 |
| AgentGraph | 947 | 804 | 143 | 15 | 67 | 58 | 2 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 885 | 871 | 14 | 65 |
| AgentGraph | 127 | 962 | 819 | 143 | 67 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 126, '2': 1}`
- Relation count distribution: `{'0': 126, '1': 1}`
- Topology distribution: `{'serial_2': 1, 'single': 126}`
- Director `max_rounds`: **2**
- Runtime failed turns: **238**
- Runtime failure types: `{'EnvironmentExecutionError': 176, 'OpenAICompatibleGatewayError': 62}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **1**

## Failure types

- `agentgraph_higher_average_score`: 34
- `agentgraph_operational_or_evaluator_failure`: 1
- `agentgraph_terminal_failure`: 2
- `direct_higher_average_score`: 33
- `equal_average_score`: 58
