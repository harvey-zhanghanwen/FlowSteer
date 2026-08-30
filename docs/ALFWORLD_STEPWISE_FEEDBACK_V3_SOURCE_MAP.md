# ALFWorld stepwise public-feedback v3 source map

This condition changes only the public Action--Observation feedback wiring.
The unified AgentGraph, Canvas action space, single stateful Tool owner and
native ALFWorld evaluator remain unchanged. No training or Skill injection is
part of this condition.

## Direct reuse

- SkillFlow/RAGEN task-scoped ALFWorld session and native action protocol:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`.
- SkillFlow bounded ReAct prompt and recent Action--Observation history:
  `src/interactive/task_evaluator.py::_environment_prompt` and
  `::_recent_environment_history`.
- FlowSteer progressive Canvas execution feedback and `continue` boundary:
  `src/interactive/agent_workflow_env.py::_accepted_feedback` and
  `::public_environment_state`.
- FlowSteer Director current-state observation:
  `src/interactive/director.py::_canvas_observation`.

## Necessary thin adaptation

Source: SkillFlow
`training/environment.py::_extract_visited_receptacles` and
`::_build_alfworld_progress_block`.

Adapted in `src/interactive/environment_execution.py`:

- `_public_state_feedback` retains neutral visited-receptacle facts and their
  last public observations. The upstream prescriptive `Next-action rule` is
  deliberately not copied.
- `_public_action_observation_history` projects each completed public
  transition into the lossless runtime metadata without reward, `won`, score,
  hidden simulator state or evaluator fields.
- `_action_prompt` supplies the same cumulative public state to the unique
  ReAct environment Agent before its next native action.
- `_current_public_state` supplies the original task, latest transition,
  current observation, current admissible actions, environment revision,
  remaining action budget and cumulative public state to the Director.
- Observable placement progress is reversible: a later public `take` removes
  the corresponding earlier `move` from current progress.

The Director receives only the latest Action--Observation receipt plus the
compact cumulative public-state summary. The complete per-action history is
retained in trajectory metadata rather than duplicated in every Director chat
turn, keeping the 8K context boundary bounded.

## Failure semantics

- Parse failure: `state_advanced=false`, revision unchanged, public state
  preserved and returned on the next turn.
- Runtime/provider failure: existing
  `preserve_diagnose_repair_augment` recovery remains authoritative.
- Environment success: only the native terminal evaluator receipt is
  authoritative; Canvas `finish` never creates reward.

## Versioned condition and tests

- Config: `config/evaluation_alfworld_stepwise_feedback_v3.yaml`.
- Tests: `tests/unit/test_environment_execution.py` verifies per-action
  feedback to both the ReAct Agent and Director, visited receptacle memory,
  reversible placement progress, single task-scoped episode and hidden-state
  isolation.
