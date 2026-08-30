# ALFWorld stepwise public-feedback v4 source map

This condition fixes the ALFWorld adapter and live action-domain wiring without
changing the unified AgentGraph action space or the native environment reward.
It contains no training, GRPO, MACE, Bayesian update, or Skill injection.

## Direct reuse

- SkillFlow/RAGEN task-scoped ALFWorld session, reset, native action list,
  observation, transition, terminal state, and reward:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`.
- SkillFlow public ALFWorld task parsing and visible-memory fields:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/environment.py`
  (`_parse_alfworld_task`, `_build_alfworld_visible_memory`).
- FlowSteer progressive Canvas actions, execution feedback, and trajectory:
  `src/interactive/agent_workflow_env.py` and
  `src/interactive/rollout_collector.py`.

## Necessary compatibility adaptations

- `src/interactive/director.py` binds scalar `add_agent` constrained decoding
  to the current live model domain and to the complete registered
  `execution_mode`/`allowed_tools` profile. This is required because the v2
  flat action schema did not contain the fields required by the ALFWorld Tool
  owner runtime.
- `src/interactive/environment_execution.py` exposes only public state from
  the task, admissible actions, observation, and completed Action--Observation
  history. It retains all current navigation/open targets, action-type counts,
  target/destination mentions, required-transform progress, placement
  progress, and visible entity-binding conflicts. It filters the non-task
  `help` command in the same way as SkillFlow.
- The local task parser accepts the grammatical article `an` and rejects
  unresolved `it`/`them` as an entity class. This is a minimal compatibility
  fix for task wording not handled by the upstream regular expressions; it
  does not inspect simulator hidden state or reference reward.
- `src/interactive/agent_workflow_env.py` applies
  preserve--diagnose--repair after two identical parse failures at an unchanged
  environment revision by allowing a contract repair for the existing Tool
  owner. It never resets the episode or discards accepted public evidence.
- Once the environment is terminal or the action budget is exhausted, the
  action mask admits only the remaining structural closure edits needed for
  `FINISH`; it does not allow more environment actions or unrelated graph
  growth.
- `src/interactive/rollout_collector.py` validates the complete scalar
  `add_agent` receipt and fails closed before generation when the prompt plus
  requested output would exceed the configured context window.
- `scripts/train_agentgraph_smoke.py` passes the versioned context limit from
  the evaluation config to the receipt client. The evaluation runner reuses
  this backend; no training path is enabled by the v4 condition.

## Evaluator and protocol boundary

- Success remains authoritative only when returned by the native ALFWorld
  terminal receipt. Canvas `finish`, Agent text, and public-state summaries do
  not create reward.
- Direct and AgentGraph use the same official `valid_seen` tasks, 20-action
  budget, Qwen3.5-9B condition, native environment, and evaluator.
- Versioned config:
  `config/evaluation_alfworld_stepwise_feedback_v4.yaml`.
- Prepare-only manifest:
  `artifacts/alfworld_stepwise_feedback_v4/valid_seen/run_manifest.json`.
- The frozen 140-task selection is byte-identical to the v3 formal run.

## Verification

Targeted tests cover live scalar `add_agent`, receipt validation, context-limit
failure, public-state coreference and entity binding, transform/placement
conjunction, revisits, repeated parse-error repair, one shared episode, unique
Tool ownership, Canvas revision, and terminal closure.
