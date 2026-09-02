# WebShop Architecture Validation

Fixed validation samples: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Native evaluator: **WebShop Average Score** and **Success Rate** (`WebShop_official_environment_Average_Score_and_Success_Rate`). AgentGraph explicit FINISH: **128/128**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 32.69 | 14.06% |
| AgentGraph | 128 | 128 | 60.71 | 35.16% |

AgentGraph - Direct: **+28.02 Average Score**, **+21.09 percentage points Success Rate**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 894 | 876 | 18 | 0 | 63 | 65 | 0 | 0 |
| AgentGraph | 754 | 754 | 0 | 0 | 107 | 21 | 0 | 0 |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | 128 | 894 | 876 | 18 | 63 |
| AgentGraph | 128 | 754 | 754 | 0 | 107 |

## Natural AgentGraph structure

- Agent count distribution: `{'1': 45, '2': 76, '3': 2, '4': 4, '5': 1}`
- Relation count distribution: `{'0': 45, '1': 76, '2': 1, '3': 3, '4': 2, '5': 1}`
- Topology distribution: `{'fan_in': 2, 'mixed': 4, 'serial_2': 76, 'serial_3_plus': 1, 'single': 45}`
- Director `max_rounds`: **0**
- Runtime failed turns: **0**
- Runtime failure types: `{}`
- Executor/provider error types: `{}`
- Direct provider error types: `{}`
- Collection failures: **0**

## Single-Agent and multi-Agent outcomes

These are observational subgroups produced by the Director, not randomized
conditions. They describe whether the free AgentGraph search space is being
used; they do not establish a causal benefit from adding Agents.

| Final graph subgroup | Tasks | Successes | Success Rate | Average Score (/100) |
|---|---:|---:|---:|---:|
| Single Agent | 45 | 16 | 35.56% | 60.04 |
| Multi-Agent | 83 | 29 | 34.94% | 61.07 |
| Two-Agent serial topology | 76 | 27 | 35.53% | 62.53 |
| Non-chain topology (`fan_in`, `mixed`, or `serial_3_plus`) | 7 | 2 | 28.57% | 45.24 |

In **73/83** multi-Agent episodes, the environment-executing Agent consumed at
least one artifact from a directed predecessor. Those 73 episodes obtained
**23/73 = 31.51% Success Rate** and **58.71 Average Score**. The remaining 10
multi-Agent episodes obtained 6/10 successes, but this small, Director-selected
subgroup is not a valid counterfactual comparison.

The architecture therefore supports Agent creation, directed communication,
artifact consumption, stateful environment ownership, and non-chain topology.
The complete panel does **not** show that the current untrained Director has
learned when multi-Agent coordination is beneficial: multi-Agent Success Rate
is statistically descriptive and slightly below the single-Agent subgroup.

## Validation assessment

- The fixed-20 gate passed at **55.00% Success Rate** (11/20), so the frozen
  128-task condition was run as specified.
- The full 128-task result is **35.16% Success Rate** (45/128) and **60.71
  Average Score**. The fixed-20 result was therefore optimistic and must not be
  reported as the full-panel performance.
- All 128 task records have a valid native WebShop evaluator receipt and an
  explicit AgentGraph `FINISH`; there were no collection, parsing, provider,
  evaluator, or AgentGraph terminal failures.
- No optimizer update, Skill injection, MACE update, Bayesian update, or
  topology/role template was used. The result is an evaluation-only Stable
  Zero architecture result, not evidence that a coordination policy was
  learned.
- Direct is a frozen, separately executed comparator reused under matching
  task IDs, model condition, protocol, and seed. The reported delta is
  descriptive rather than a paired causal effect.

## Failure types

- `agentgraph_higher_average_score`: 71
- `direct_higher_average_score`: 9
- `equal_average_score`: 48
