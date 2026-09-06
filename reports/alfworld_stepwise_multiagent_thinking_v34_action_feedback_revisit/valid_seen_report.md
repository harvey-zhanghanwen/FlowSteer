# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **136/140**; terminal failures: **1**; operational/evaluator failures: **3**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 40.00% |
| AgentGraph | 137 | 137 | 62.14% |

AgentGraph - Direct: **+22.14 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_success`: 39
- `agentgraph_operational_or_evaluator_failure`: 3
- `agentgraph_terminal_failure`: 1
- `direct_higher_success`: 8
- `equal_success`: 89


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 56 | 140 | 140 | 40.00% | 40.00% |
| AgentGraph | 87 | 140 | 137 | 62.14% | 63.50% |

AgentGraph termination: explicit FINISH **136/140**; max_rounds **1**; terminal failures **1**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2171 | 0 | 94 | 0 | 31 | 56 | 0.4 |
| AgentGraph | 1910 | 0 | 31 | 0 | 0 | 87 | 0.635036496350365 |

## AgentGraph structure

- Agent count distribution: `{"1": 12, "2": 105, "3": 9, "4": 6, "5": 4, "8": 1}`
- Topology distribution: `{"fan_in": 3, "mixed": 4, "parallel": 7, "serial_2": 105, "serial_3_plus": 6, "single": 12}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **0**; structured runtime failure distribution: `{}`
- Historical collection failure attempts: **5**; recovered attempts: **2**; unresolved task-condition pairs: **3**

## Receipt-causal primary failure taxonomy

Denominator: **53 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 36 | 67.92% | `alfworld:valid_seen:00004`, `alfworld:valid_seen:00007`, `alfworld:valid_seen:00008` |
| Object grounding/affordance | 0 | 0.00% | None |
| Subgoal sequencing/action policy | 13 | 24.53% | `alfworld:valid_seen:00018`, `alfworld:valid_seen:00030`, `alfworld:valid_seen:00042` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 0 | 0.00% | None |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 1 | 1.89% | `alfworld:valid_seen:00103` |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 3 | 5.66% | `alfworld:valid_seen:00102`, `alfworld:valid_seen:00110`, `alfworld:valid_seen:00136` |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **1** unsuccessful episodes; missing explicit FINISH: **4**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00004 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00007 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00008 | director_canvas | 2 | canvas_edit_rejected |
| alfworld:valid_seen:00016 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00018 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00025 | director_canvas | 1 | canvas_edit_rejected |
| alfworld:valid_seen:00027 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00030 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00032 | director_canvas | 19 | canvas_edit_rejected |
| alfworld:valid_seen:00034 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |

