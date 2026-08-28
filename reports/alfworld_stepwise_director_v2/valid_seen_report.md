# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **137/140**; terminal failures: **3**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 30.00% |
| AgentGraph | 140 | 140 | 31.43% |

AgentGraph - Direct: **+1.43 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 14
- `agentgraph_terminal_failure`: 3
- `direct_higher_success`: 11
- `equal_success`: 112


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 42 | 140 | 140 | 30.00% | 30.00% |
| AgentGraph | 44 | 140 | 140 | 31.43% | 31.43% |

AgentGraph termination: explicit FINISH **137/140**; max_rounds **3**; terminal failures **3**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2281 | 0 | 118 | 0 | 71 | 42 | 0.3 |
| AgentGraph | 2198 | 0 | 100 | 0 | 111 | 44 | 0.3142857142857143 |

## AgentGraph structure

- Agent count distribution: `{"1": 130, "2": 10}`
- Topology distribution: `{"parallel": 2, "serial_2": 8, "single": 130}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **156**; structured runtime failure distribution: `{"CancelledError": 8, "EnvironmentExecutionError": 156}`
- Historical collection failure attempts: **0**; recovered attempts: **0**; unresolved task-condition pairs: **0**

## Receipt-causal primary failure taxonomy

Denominator: **96 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 46 | 47.92% | `alfworld:valid_seen:00003`, `alfworld:valid_seen:00004`, `alfworld:valid_seen:00008` |
| Object grounding/affordance | 23 | 23.96% | `alfworld:valid_seen:00007`, `alfworld:valid_seen:00013`, `alfworld:valid_seen:00016` |
| Subgoal sequencing/action policy | 27 | 28.12% | `alfworld:valid_seen:00000`, `alfworld:valid_seen:00001`, `alfworld:valid_seen:00010` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 0 | 0.00% | None |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 0 | 0.00% | None |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 0 | 0.00% | None |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **3** unsuccessful episodes; missing explicit FINISH: **3**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00000 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00001 | tool_interface | 0 | EnvironmentExecutionError |
| alfworld:valid_seen:00003 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00004 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00007 | director_canvas | 1 | canvas_edit_rejected |
| alfworld:valid_seen:00008 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00009 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00010 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00013 | tool_interface | 1 | EnvironmentExecutionError |
| alfworld:valid_seen:00016 | director_canvas | 1 | canvas_edit_rejected |
