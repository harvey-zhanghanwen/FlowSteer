# WebShop v41 stateful fan-in freshness source map

## Scope

This version is evaluation-only. It does not enable training, GRPO, MACE,
Bayesian updating, LoRA, or Skill evolution. The unified orchestration core is
unchanged outside the WebShop stateful execution boundary.

## Source classification

| Module / behavior | Classification | Source and reason |
| --- | --- | --- |
| One mutable WebShop session, native `search` / `click`, Action--Observation history, terminal reward | Direct reuse | SkillFlow RAGEN WebShop adapter and the original WebShop environment/evaluator |
| Canvas actions and execute-after-edit feedback | Direct reuse | FlowSteer progressive Canvas/runtime boundary |
| Unique stateful environment owner | Necessary adaptation | SkillFlow serializes one environment session; multiple Agents may reason, but only one Agent may mutate that session |
| Environment owner as execution sink | Necessary adaptation | Every WebShop Action--Observation must return to the Director before another environment action; a post-owner Agent would otherwise consume a pre-action Runtime snapshot |
| Refresh all directed ancestors when a Canvas edit dirties the environment owner | Necessary adaptation | Combines FlowSteer's execute-after-edit semantics with SkillFlow's requirement that the next policy step use the latest public Action--Observation state; prevents an older revision's cached proposal from entering a new fan-in |
| Independent auxiliary cache reuse | Direct reuse | Existing FlowSteer incremental Runtime cache remains active when an edit does not reach the environment owner |
| Official Average Score and Success Rate | Direct reuse | Original WebShop evaluator through SkillFlow's RAGEN adapter |

## Confirmed failure and repair

The v40 trajectory for `webshop:00525` formed a six-Agent `mixed` topology.
At the repair `ADD_SUBGRAPH`, the new branch produced the reformulated action
`search[xx-small hoodie under $50]`, while an older directed ancestor was
reused from a previous environment revision and proposed the already-rejected
query. The environment owner received both artifacts and selected the stale
one.

`AgentWorkflowEnv._stateful_owner_feedback_dirty_closure` now extends the
existing `AgentGraph.dirty_closure` only when the accepted edit already
reschedules the unique stateful owner. In that case, all routed ancestors and
their dependents execute against one current public state. Unrelated auxiliary
edits do not reschedule the owner and do not consume another environment turn.

## Verification

- Targeted suite: 151 passed; 12 subtests passed.
- Regression test: `test_fan_in_edit_refreshes_existing_owner_ancestors`.
- The test verifies two independent advisor artifacts fan into one stateful
  owner, the historical advisor is not reused, both advisors contain current
  observation evidence, and the owner advances the environment exactly once.
