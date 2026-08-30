# ALFWorld stepwise recovery v6 source map

This condition is the v5 AgentGraph/runtime architecture with one evaluator
compatibility repair. It adds no training, GRPO, MACE, Bayesian update, LoRA
update, or Skill injection.

## Direct reuse

- ALFWorld task-scoped reset, native action strings, public observations,
  state transitions, termination, reward, and `valid_seen` inventory remain the
  deployed SkillFlow/RAGEN implementation in
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`.
- Runtime action validation follows SkillFlow's public admissible-action
  contract in
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/environment.py`.
- FlowSteer Canvas editing, execute-on-edit feedback, Agent communication, and
  trajectory persistence remain in `src/interactive/agent_workflow_env.py`,
  `src/interactive/environment_execution.py`, and
  `src/interactive/rollout_collector.py`.
- All v5 public-state, task-scoped episode, single Tool-owner, action-budget,
  stall diagnosis, and `preserve -> diagnose -> repair -> augment` behavior is
  retained without modification. See
  `docs/ALFWORLD_STEPWISE_RECOVERY_V5_SOURCE_MAP.md`.

## Necessary evaluator compatibility repair

`src/interactive/environment_execution.py::_parse_action` already unwraps the
strict one-field constrained-decoding response `{"action": "<native action>"}`
and passes only the native action string to `alfworld.environment`. The
evaluator ledger deliberately preserves both the native `action` and the
original `raw_graph_output` so terminal replay can verify their correspondence.

The v5 terminal replay parser accepted only a bare native action or one
`<action>` tag. It therefore rejected the constrained JSON response even when
the runtime had executed the corresponding native action and received a valid
terminal transition. v6 applies the same strict projection in
`src/interactive/task_evaluator.py::_parse_environment_action` only for
ALFWorld: the decoded value must be an object whose key set is exactly
`{"action"}`, whose value is a string, and whose stripped value is in the
current public `legal_actions`. Extra fields, non-string values, malformed JSON,
and non-admissible actions fail closed. WebShop parsing is unchanged.

This is a compatibility repair between the constrained model-response boundary
and the official native-action replay boundary. It does not alter the ALFWorld
Tool protocol, transition function, hidden state, terminal reward, task order,
action budget, Director prompt, AgentGraph search space, or model condition.

## Frozen evaluation condition

- Config: `config/evaluation_alfworld_stepwise_recovery_v6.yaml`.
- Official split and denominator: sequential 140 tasks from `valid_seen`.
- Stable Zero gate: the first two frozen tasks, followed by the same full 140
  only if both Direct and AgentGraph have evaluator-valid terminal receipts.
- Direct and AgentGraph use Qwen3.5-9B, one task-scoped environment episode,
  a 20-action policy budget, and the official environment evaluator. AgentGraph
  additionally uses the live admissible-action enum for constrained decoding,
  so the Direct/AgentGraph difference remains descriptive rather than a paired
  causal estimate.
- Tool adapter remains `skillflow.alfworld.native-stepwise-recovery.v5` because
  the Tool/runtime protocol did not change. Storage and policy receipts are
  versioned as `flowsteer.alfworld.stepwise-recovery.v6` and
  `qwen35-9b-base-alfworld-stepwise-recovery-v6`.
- Outputs are isolated under
  `artifacts/alfworld_stepwise_recovery_v6/valid_seen` and
  `reports/alfworld_stepwise_recovery_v6`.

## Unified AgentGraph boundary

`Agent = agent_id + model_id + free-text contract` remains unchanged. Director
still selects Agent count, contracts, model, relations, Output Agent,
`continue`, and `finish`; no fixed Agent role sequence or topology is imposed.
ReAct is an execution mode. Only one ReAct Agent may own
`alfworld.environment`, which prevents concurrent mutation of the shared world
state. Success still comes only from the real terminal environment receipt.
