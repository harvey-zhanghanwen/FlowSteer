# ALFWorld stepwise multi-Agent v10 Stable Zero report

## Condition

- official split: `valid_seen`
- frozen canary tasks: 2, identical Task IDs and order to v9
- Director prompt: `agentgraph.director.minimal-neutral-add-subgraph-stepwise.v2`
- environment Tool: `skillflow.alfworld.native-stepwise-recovery.v5`
- native action budget: 20 per episode
- Canvas turn budget: 32
- training, GRPO, LoRA, MACE, Bayesian update and Skills: disabled

## Result

| Condition | Success / total | Canary SR | Episode scores | Environment actions | Explicit FINISH | Terminal failure | Collection failure |
|---|---:|---:|---|---:|---:|---:|---:|
| Direct | 2 / 2 | 100% | 1, 1 | 9, 18 | N/A | 0 | 0 |
| AgentGraph | 2 / 2 | 100% | 1, 1 | 12, 8 | 2 / 2 | 0 | 0 |

This is a two-task Stable Zero canary, not the formal 140-task `valid_seen`
Success Rate.

## Multi-Agent execution evidence

- both tasks used two Agents and one directed collaborator-to-Tool-owner
  relation;
- no Agent declared ReAct as a role;
- each task had exactly one `execution_mode="react"` Agent owning
  `alfworld.environment` and one `execution_mode="reasoning"` stateless
  collaborator;
- 20/20 Tool-owner execution records received an upstream collaborator
  artifact before the native environment action;
- every native action produced an immediate public Observation, environment
  revision, admissible-action list and remaining action budget for the next
  Director turn;
- environment reward and success came only from the terminal ALFWorld
  evaluator receipt.

## Per-task AgentGraph result

| Task ID | Instruction | Agents | Native actions | Score | Success | Termination |
|---|---|---:|---:|---:|---:|---|
| `alfworld:valid_seen:00000` | put two toiletpaper in toilet. | 2 | 12 | 1 | 1 | explicit `FINISH` |
| `alfworld:valid_seen:00001` | put a hot mug in cabinet. | 2 | 8 | 1 | 1 | explicit `FINISH` |

There is no Wrong Demo in this two-task canary.  Failure-type counts for this
run are therefore all zero; no case is fabricated.  A formal failure taxonomy
requires the complete frozen evaluation rather than extrapolation from these
two tasks.
