# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **46/140**; terminal failures: **94**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 30.71% |
| AgentGraph | 140 | 140 | 32.86% |

AgentGraph - Direct: **+2.14 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 12
- `agentgraph_terminal_failure`: 94
- `equal_success`: 34


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 43 | 140 | 140 | 30.71% | 30.71% |
| AgentGraph | 46 | 140 | 140 | 32.86% | 32.86% |

AgentGraph termination: explicit FINISH **46/140**; max_rounds **94**; terminal failures **94**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2250 | 0 | 134 | 0 | 81 | 43 | 0.30714285714285716 |
| AgentGraph | 2176 | 0 | 114 | 0 | 83 | 46 | 0.32857142857142857 |

## AgentGraph structure

- Agent count distribution: `{"0": 6, "1": 58, "2": 44, "3": 24, "4": 5, "5": 3}`
- Topology distribution: `{"empty": 6, "fan_in": 13, "mixed": 6, "parallel": 18, "reciprocal": 3, "serial_2": 31, "serial_3_plus": 5, "single": 58}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **530**; structured runtime failure distribution: `{"CancelledError": 32, "EnvironmentExecutionError": 565}`
- Historical collection failure attempts: **0**; recovered attempts: **0**; unresolved task-condition pairs: **0**

## Wrong Demo: first observable failure

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00003 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00007 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00008 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00009 | director_canvas | 0 | canvas_edit_rejected |
| alfworld:valid_seen:00010 | director_canvas | 2 | canvas_edit_rejected |
| alfworld:valid_seen:00012 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00013 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00014 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00016 | director_canvas | 3 | canvas_edit_rejected |
| alfworld:valid_seen:00017 | tool_interface | 0 | EnvironmentExecutionError |

