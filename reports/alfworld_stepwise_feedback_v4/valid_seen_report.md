# ALFWorld Architecture Validation

Fixed test samples: **140**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **success** (`SkillFlow_RAGEN_official_environment_terminal_success`). AgentGraph explicit FINISH: **139/140**; terminal failures: **1**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict success |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 140 | 140 | 28.57% |
| AgentGraph | 140 | 140 | 42.86% |

AgentGraph - Direct: **+14.29 percentage points**.

Direct and AgentGraph use protocol-equivalent task/evaluator conditions.

## Failure types

- `agentgraph_higher_success`: 24
- `agentgraph_terminal_failure`: 1
- `direct_higher_success`: 4
- `equal_success`: 111


## ALFWorld native outcome

Official split: **valid_seen**; policy action budget: **20**; TextWorld hard limit: **50**.

Success Rate over total tasks treats missing/invalid episodes as unsuccessful. Success Rate over evaluator-valid tasks excludes them.

| Condition | Success | Total | Evaluator valid | SR (total) | SR (evaluator-valid) |
|---|---:|---:|---:|---:|---:|
| Direct | 40 | 140 | 140 | 28.57% | 28.57% |
| AgentGraph | 60 | 140 | 140 | 42.86% | 42.86% |

AgentGraph termination: explicit FINISH **139/140**; max_rounds **1**; terminal failures **1**.

## ALFWorld environment receipts

| Condition | Environment actions | Invalid | Repeated | No effect | Parse errors | Terminal episodes | Mean episode score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 2286 | 0 | 141 | 0 | 66 | 40 | 0.2857142857142857 |
| AgentGraph | 2124 | 0 | 61 | 0 | 25 | 60 | 0.42857142857142855 |

## AgentGraph structure

- Agent count distribution: `{"1": 138, "2": 2}`
- Topology distribution: `{"parallel": 1, "serial_2": 1, "single": 138}`

## Runtime and provider receipts

- Direct execution error distribution: `{}`
- AgentGraph execution error distribution: `{}`
- AgentGraph runtime failed turns: **11**; structured runtime failure distribution: `{"EnvironmentExecutionError": 11}`
- Historical collection failure attempts: **0**; recovered attempts: **0**; unresolved task-condition pairs: **0**

## Receipt-causal primary failure taxonomy

Denominator: **80 unsuccessful AgentGraph episodes**. Each episode receives one mutually exclusive primary cause. Early typed errors that were repaired before a complete native episode remain in the trajectory but are not relabeled as the terminal root cause.

| Primary failure class | Count | Share | Representative task IDs |
|---|---:|---:|---|
| Environment exploration/search | 61 | 76.25% | `alfworld:valid_seen:00004`, `alfworld:valid_seen:00007`, `alfworld:valid_seen:00008` |
| Object grounding/affordance | 9 | 11.25% | `alfworld:valid_seen:00013`, `alfworld:valid_seen:00022`, `alfworld:valid_seen:00045` |
| Subgoal sequencing/action policy | 10 | 12.50% | `alfworld:valid_seen:00042`, `alfworld:valid_seen:00044`, `alfworld:valid_seen:00074` |
| Native action parser | 0 | 0.00% | None |
| Tool/execution-profile | 0 | 0.00% | None |
| Director/Canvas construction | 0 | 0.00% | None |
| Agent communication | 0 | 0.00% | None |
| Agent runtime | 0 | 0.00% | None |
| Environment runtime | 0 | 0.00% | None |
| Terminal control | 0 | 0.00% | None |
| Evaluator | 0 | 0.00% | None |
| Provider/collection | 0 | 0.00% | None |

`max_rounds` is reported as a cross-cutting terminal manifestation, not automatically as the root cause: **1** unsuccessful episodes; missing explicit FINISH: **1**.

Retrieval/database and final-answer formatting/canonicalization are not applicable to the native ALFWorld reward protocol. Environment observations and admissible actions come from the stateful Tool, and success comes only from the native terminal evaluator.

## Wrong Demo: first observable typed failure (diagnostic, not necessarily root cause)

| Task | Failure layer | First error turn | Error |
|---|---|---:|---|
| alfworld:valid_seen:00004 | agent_action_parsing | 15 | native_action_parse_error |
| alfworld:valid_seen:00007 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00008 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00009 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00013 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00014 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00016 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00018 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00020 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |
| alfworld:valid_seen:00022 | agent_policy | 20 | action_budget_exhausted_before_environment_terminal |

