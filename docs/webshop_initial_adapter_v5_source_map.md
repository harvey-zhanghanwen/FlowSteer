# WebShop initial adapter v5 source map

This condition is evaluation-only. Training, GRPO, MACE, Bayesian posterior
updates, Skill retrieval/evolution, backward, optimizer updates and LoRA
publication are disabled.

## Direct reuse

- SkillFlow WebShop session and action semantics:
  - deployed `src/ragen_adapter.py::RAGENAdapter.reset/step`;
  - one mutable WebShop environment instance per bounded episode;
  - live `available_actions` from the current DOM;
  - native `search[query]` and `click[value]` actions;
  - simulator terminal reward in `[0, 1]`;
  - fixed ten-turn episode budget and zero-reward truncation.
- Original WebShop evaluator:
  - graded environment reward is the per-episode score;
  - `Average Score = mean(reward) * 100`;
  - `Success Rate = mean(reward == 1.0) * 100`.
- FlowSteer progressive Canvas:
  - one atomic Director edit per turn;
  - execute accepted edits and expose public execution feedback;
  - persist graph revisions, model calls, Agent communication and trajectory;
  - explicit `FINISH` is required for formal AgentGraph evaluation.

## Necessary adaptation

- `src/interactive/environment_execution.py`
  - binds the evaluator-locked WebShop record to one request-scoped
    `RAGENAdapter` session;
  - projects dynamic native actions without inventing a static shopping action
    schema;
  - keeps raw terminal HTML/reward/info only in evaluator replay data while
    public Agent/Canvas messages receive the visible terminal acknowledgement;
  - records `environment_truncated` and the fixed episode limit when the
    SkillFlow action budget is exhausted without simulator termination.
- `src/interactive/agent_workflow_env.py`
  - reuses the existing `required_tool_id` and stateful single-owner
    execution contract;
  - projects the unmet capability into the revision-local action mask:
    empty graph admits free `ADD_AGENT`; an existing graph admits only the
    atomic `MODIFY_AGENT` field/value needed to establish exactly one ReAct
    owner of `webshop.environment`;
  - immediately restores the full configured AgentGraph search space after the
    capability exists;
  - accepts either a measured simulator terminal transition or the exact
    fixed-budget truncation before `FINISH`.
- `src/interactive/director.py`
  - exposes scalar `action_target_domains` and the current neutral
    `finish_admissibility` diagnosis in the Canvas observation.
- `src/interactive/rollout_collector.py`
  - persists the bounded-episode `environment_truncated` and
    `environment_max_turns` receipts already emitted by the environment
    adapter; it does not expose evaluator reward or hidden target fields.
- `scripts/evaluate_completion_benchmark_round.py`
  - validates either a measured simulator terminal transition or an exact
    fixed-budget truncation receipt during Stable Zero checking, matching
    SkillFlow's completed/truncated episode semantics;
  - keeps the official evaluator episode separate from execute-on-edit
    diagnostics, deduplicates reused environment artifacts by `artifact_id`,
    and reports saved non-formal prefixes separately from formal actions.

These adaptations constrain execution capability only. They do not select an
Agent ID, model, free-text contract, role, Output Agent, edge direction, Agent
count or topology. No Searcher, Reviewer, Buyer or fixed shopping workflow is
introduced.

## Project-algorithm additions

None in this evaluation condition.

## Not implemented or intentionally disabled

- GRPO / backward / optimizer update / LoRA;
- MACE exploration;
- Bayesian posterior fitting;
- Skill retrieval, publication or evolution;
- reward shaping, LLM judge, goal-aware action filtering;
- concurrent writes to one stateful WebShop session.

## Condition lineage

- v1 is rejected: WebShop's terminal page exposed evaluator-private score and
  hidden target fields to later model-visible artifacts.
- v2 fixes the public/private observation boundary but fails Stable Zero on one
  of two tasks because the required environment capability was visible only at
  the terminal gate, allowing unrelated topology edits to consume all Canvas
  rounds.
- v3 keeps v2's leakage fix and adds the generic capability/action-mask and
  upstream truncation boundaries. Its two-task smoke run completed both
  trajectories and evaluators, but the Stable Zero checker rejected one valid
  fixed-budget truncation because the truncation receipt was not persisted.
- v4 adds the missing receipt persistence and passes its two-task Stable Zero
  run. Its checker can still accept a stale earlier Canvas revision or a
  contradictory trace, so v4 remains a frozen smoke artifact rather than the
  formal 128-task condition.
- v5 binds certification to the final `FINISH` turn, requires exactly one
  dataset-matched environment trace, consecutive step receipts, internally
  consistent counts, and mutually exclusive terminal/truncation states.
  v1/v2/v3/v4 artifacts are retained under their original condition
  directories and are never mixed into v5.
