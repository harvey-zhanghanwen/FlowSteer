# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **130/140**; terminal failures: **3**; operational/evaluator failures: **7**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 30.71% |
| AgentGraph | 133 | 133 | 42.86% |

AgentGraph - Direct: **+12.14 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 28
- `agentgraph_operational_or_evaluator_failure`: 7
- `agentgraph_terminal_failure`: 3
- `direct_higher_success`: 11
- `equal_success`: 91


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 43 | 140 | 140 | 30.71% | 30.71% |
| AgentGraph | 60 | 140 | 133 | 42.86% | 45.11% |

AgentGraph termination: explicit FINISH **130/140**; max_rounds **3**; terminal failures **3**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2240 | 0 | 123 | 0 | 78 | 43 | 0.30714285714285716 |
| AgentGraph | 1936 | 0 | 40 | 0 | 114 | 60 | 0.45112781954887216 |

## AgentGraph structure

- Agent count distribution: `{"1": 125, "2": 6, "3": 2}`
- Topology distribution: `{"fan_in": 1, "fan_out": 1, "parallel": 1, "serial_2": 5, "single": 125}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **139**; structured runtime failure distribution: `{"EnvironmentExecutionError": 139}`
- Historical collection failure attempts: **7**; recovered attempts: **0**; unresolved task-condition pairs: **7**

## Receipt-causal primary failure taxonomy

Denominator: **80 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 37 | 46.25% | `alfworld:valid_seen:00004`, `alfworld:valid_seen:00014`, `alfworld:valid_seen:00016` |
| Object grounding/affordance | 13 | 16.25% | `alfworld:valid_seen:00007`, `alfworld:valid_seen:00013`, `alfworld:valid_seen:00020` |
| Subgoal sequencing/action policy | 23 | 28.75% | `alfworld:valid_seen:00001`, `alfworld:valid_seen:00003`, `alfworld:valid_seen:00008` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 0 | 0.00% | None |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 0 | 0.00% | None |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 7 | 8.75% | `alfworld:valid_seen:00124`, `alfworld:valid_seen:00126`, `alfworld:valid_seen:00127` |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **3** unsuccessful episodes; missing explicit FINISH: **10**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00001 | director_canvas | 3 | canvas_edit_rejected |
| alfworld:valid_seen:00003 | director_canvas | 1 | canvas_edit_rejected |
| alfworld:valid_seen:00004 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00007 | director_canvas | 1 | canvas_edit_rejected |
| alfworld:valid_seen:00008 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00010 | director_canvas | 1 | canvas_edit_rejected |
| alfworld:valid_seen:00013 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00014 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00015 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00016 | tool_interface | 1 | EnvironmentExecutionError |

