# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **139/140**; terminal failures: **1**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 40.00% |
| AgentGraph | 140 | 140 | 69.29% |

AgentGraph - Direct: **+29.29 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_success`: 50
- `agentgraph_terminal_failure`: 1
- `direct_higher_success`: 9
- `equal_success`: 80


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 56 | 140 | 140 | 40.00% | 40.00% |
| AgentGraph | 97 | 140 | 140 | 69.29% | 69.29% |

AgentGraph termination: explicit FINISH **139/140**; max_rounds **1**; terminal failures **1**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2171 | 0 | 94 | 0 | 31 | 56 | 0.4 |
| AgentGraph | 1896 | 0 | 20 | 0 | 0 | 97 | 0.6928571428571428 |

## AgentGraph structure

- Agent count distribution: `{"1": 12, "2": 92, "3": 11, "4": 18, "5": 1, "6": 2, "7": 1, "8": 3}`
- Topology distribution: `{"fan_in": 13, "fan_out": 2, "mixed": 9, "parallel": 5, "serial_2": 93, "serial_3_plus": 6, "single": 12}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **0**; structured runtime failure distribution: `{}`
- Historical collection failure attempts: **1**; recovered attempts: **1**; unresolved task-condition pairs: **0**

## Receipt-causal primary failure taxonomy

Denominator: **43 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 31 | 72.09% | `alfworld:valid_seen:00004`, `alfworld:valid_seen:00007`, `alfworld:valid_seen:00013` |
| Object grounding/affordance | 0 | 0.00% | None |
| Subgoal sequencing/action policy | 11 | 25.58% | `alfworld:valid_seen:00030`, `alfworld:valid_seen:00042`, `alfworld:valid_seen:00076` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 0 | 0.00% | None |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 1 | 2.33% | `alfworld:valid_seen:00113` |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 0 | 0.00% | None |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **1** unsuccessful episodes; missing explicit FINISH: **1**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00004 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00007 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00013 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00014 | director_canvas | 21 | canvas_edit_rejected |
| alfworld:valid_seen:00016 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00018 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00025 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00027 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00030 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00040 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |

