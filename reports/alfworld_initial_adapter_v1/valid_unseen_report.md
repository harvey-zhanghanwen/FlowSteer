# ALFWorld Architecture Validation

Fixed test samples: **134**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **38/134**; terminal failures: **96**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 134 | 134 | 20.90% |
| AgentGraph | 134 | 134 | 28.36% |

AgentGraph - Direct: **+7.46 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 17
- `agentgraph_terminal_failure`: 96
- `equal_success`: 21


## ALFWorld native outcome

Official split: **valid_unseen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 28 | 134 | 134 | 20.90% | 20.90% |
| AgentGraph | 38 | 134 | 134 | 28.36% | 28.36% |

AgentGraph termination: explicit FINISH **38/134**; max_rounds **96**; terminal failures **96**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2227 | 0 | 204 | 0 | 127 | 28 | 0.208955223880597 |
| AgentGraph | 333 | 0 | 9 | 0 | 6 | 38 | 0.2835820895522388 |

## AgentGraph structure

- Agent count distribution: `{"0": 1, "1": 45, "2": 50, "3": 34, "4": 2, "5": 1, "8": 1}`
- Topology distribution: `{"empty": 1, "fan_in": 13, "fan_out": 2, "mixed": 4, "parallel": 25, "reciprocal": 1, "serial_2": 39, "serial_3_plus": 4, "single": 45}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **569**; structured runtime failure distribution: `{"CancelledError": 70, "EnvironmentExecutionError": 592}`
- Historical collection failure attempts: **3**; recovered attempts: **3**; unresolved task-condition pairs: **0**

## Wrong Demo: first observable failure

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_unseen:00000 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00001 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00002 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00003 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00004 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00005 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00009 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00012 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00013 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_unseen:00017 | tool_interface | 0 | EnvironmentExecutionError |

