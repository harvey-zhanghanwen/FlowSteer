# ALFWorld stepwise recovery v5 source map

This evaluation condition keeps the unified AgentGraph orchestration core and
the official ALFWorld environment evaluator unchanged. It adds no training,
GRPO, MACE, Bayesian update, LoRA update, or Skill injection.

## Direct reuse

- SkillFlow/RAGEN task-scoped ALFWorld environment session, reset, native
  action schema, public observation, state transition, termination, and reward:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`.
- FlowSteer Canvas editing, Agent execution runtime, execution feedback, and
  trajectory receipts: `src/interactive/agent_workflow_env.py`,
  `src/interactive/environment_execution.py`, and
  `src/interactive/rollout_collector.py`.

## Necessary compatibility adaptations

- SkillFlow public task parsing and visible Action--Observation memory were
  consulted as the source for the thin compatibility projection:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/environment.py`
  (`_parse_alfworld_task`, `_build_alfworld_visible_memory`, and
  `_extract_visited_receptacles`). The prescriptive
  `_rank_alfworld_unvisited_go_options` policy is deliberately not copied into
  the neutral Director condition.
- The compact Director conversation preserves the initial user/action pair,
  complete recent user/action pairs, and the exact current Observation before
  requesting the next Canvas edit. This keeps the model-visible transcript
  role-alternating while retaining the current public environment state.
- The ALFWorld Tool holder receives a durable, task-scoped public scene memory
  built only from completed native Action--Observation transitions. It includes
  the explicit environment-action budget and a structured stall receipt derived
  from public observations, admissible actions, and action history.
- Tool ownership recovery preserves accepted evidence and the shared episode,
  diagnoses the current execution profile, and repairs or augments the graph
  without creating a second environment owner. These changes implement
  `preserve -> diagnose -> repair -> augment`; they do not prescribe an Agent
  role sequence or a fixed topology.

These are compatibility adaptations between the SkillFlow/RAGEN ALFWorld
session and FlowSteer's stepwise Canvas runtime. They do not change ALFWorld's
native action strings, transition function, terminal state, or reward.

## Information and evaluator boundary

- Director and Agents receive only the task text and public runtime artifacts:
  native actions, public observations, admissible actions, environment revision,
  remaining action budget, and completed Action--Observation history.
- Simulator hidden state, terminal reward, and evaluator information are never
  included in Director or Agent prompts.
- Success is authoritative only when returned by the official ALFWorld terminal
  environment receipt. Canvas `finish` and Agent text cannot create reward.
- Direct and AgentGraph use the same sequential 140 tasks from official
  `valid_seen`, Qwen3.5-9B model, 20-action budget, environment Tool, and
  evaluator. AgentGraph v5 additionally constrains its one action response to
  the live admissible-action enum; Direct retains SkillFlow's raw/tag parser.
  The report therefore marks their delta as descriptive rather than a paired
  causal estimate. Outputs are isolated under
  `artifacts/alfworld_stepwise_recovery_v5/valid_seen` and
  `reports/alfworld_stepwise_recovery_v5`.

## AgentGraph and training boundary

- `Agent = agent_id + model_id + free-text contract` remains the only Agent
  definition. No Navigator, Manipulator, Planner, Verifier, chain, or parallel
  topology is required by the v5 condition.
- Director retains the open Canvas action space and independently selects Agent
  count, contract, model, relations, Output Agent, `continue`, and `finish`.
- ReAct remains an Agent execution mode, not an Agent role.
- Training, GRPO, optimizer updates, MACE, Bayesian inference, LoRA, exploration,
  policy synchronization, and Skill retrieval/evolution are not implemented or
  enabled in this condition.

## Versioned condition

- Evaluation config:
  `config/evaluation_alfworld_stepwise_recovery_v5.yaml`.
- Tool adapter version: `skillflow.alfworld.native-stepwise-recovery.v5`.
- Storage schema: `flowsteer.alfworld.stepwise-recovery.v5`.
- Behavior policy receipt:
  `qwen35-9b-base-alfworld-stepwise-recovery-v5`.
- The evaluation-only local Qwen3.5-9B Supervisor is assigned to physical
  GPU3 on port 8023 for this condition; the training-only GPU fields remain
  inactive.
