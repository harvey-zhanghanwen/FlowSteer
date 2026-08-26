# ALFWorld Architecture Validation

Fixed test samples: **134**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **38/134**; terminal failures: **96**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 134 | 134 | 20.90% |
| AgentGraph | 134 | 134 | 29.85% |

AgentGraph - Direct: **+8.96 percentage points**.

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
| AgentGraph | 40 | 134 | 134 | 29.85% | 29.85% |

AgentGraph termination: explicit FINISH **38/134**; max_rounds **96**; terminal failures **96**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2227 | 0 | 204 | 0 | 127 | 28 | 0.208955223880597 |
| AgentGraph | 2059 | 0 | 178 | 0 | 153 | 40 | 0.29850746268656714 |

## AgentGraph structure

- Agent count distribution: `{"0": 1, "1": 45, "2": 50, "3": 34, "4": 2, "5": 1, "8": 1}`
- Topology distribution: `{"empty": 1, "fan_in": 13, "fan_out": 2, "mixed": 4, "parallel": 25, "reciprocal": 1, "serial_2": 39, "serial_3_plus": 4, "single": 45}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **569**; structured runtime failure distribution: `{"CancelledError": 70, "EnvironmentExecutionError": 592}`
- Historical collection failure attempts: **3**; recovered attempts: **3**; unresolved task-condition pairs: **0**

## Receipt-causal primary failure taxonomy

Denominator: **94 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 35 | 37.23% | `alfworld:valid_unseen:00024`, `alfworld:valid_unseen:00028`, `alfworld:valid_unseen:00033` |
| Object grounding/affordance | 40 | 42.55% | `alfworld:valid_unseen:00000`, `alfworld:valid_unseen:00001`, `alfworld:valid_unseen:00003` |
| Subgoal sequencing/action policy | 18 | 19.15% | `alfworld:valid_unseen:00002`, `alfworld:valid_unseen:00004`, `alfworld:valid_unseen:00005` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 1 | 1.06% | `alfworld:valid_unseen:00075` |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 0 | 0.00% | None |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 0 | 0.00% | None |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **94** unsuccessful episodes; missing explicit FINISH: **94**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

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

