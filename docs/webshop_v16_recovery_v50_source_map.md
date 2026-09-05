# WebShop v16-recovery v50: baseline and controlled delta

## Exact highest-Average-Score baseline

This condition starts from the complete **WebShop stepwise Director v16**
source tree, not the later v49 implementation with an older configuration name.

- Evaluated source commit: `3ca1b3443a97dc10f0bf2da16ad1e37d8b683da9`.
- Source plus final reports: `921486fbbad48adb792219a302a44374399c5018`.
  This direct child only adds three report files; its `src`, `config`, and
  `scripts` match the evaluated source commit.
- Existing baseline branch: `feature/webshop-stepwise-director-v16-20260830`.
- Baseline configuration: `config/evaluation_webshop_stepwise_director_v16.yaml`.
- Baseline report: `reports/webshop_stepwise_director_v16/development_report.json`.
- Baseline interpretation: `reports/webshop_stepwise_director_v16/FINAL_REPORT_ZH.md`.

On the complete fixed native validation panel `webshop:00500` through
`webshop:00627`, v16 records **Average Score 63.72265625 / 100** and **Success
Rate 34.375% (44/128)**, with 128 evaluator-valid results and 128 explicit
`FINISH` receipts. The same-panel v37 result has a slightly lower Average
Score, 62.82421875, but a higher Success Rate, 35.9375% (46/128). The primary
selection criterion is official Average Score, so v16 is the modification
baseline. The later best-profile pointer to v37 omitted this earlier v16
result; a pointer alone is not evidence that v37 had the highest Average Score.

The v16 and v37 selected-task arrays are identical. Both result populations
use `skillflow.ragen_adapter.v2` and the same native WebShop protocol:
`env_seed=1000`, `human_goals=true`, `goal_split=all`, `use_small=false`,
`num_products=null`, and the same `items_ins_v2.json` / `items_shuffle.json`.
The v31 result is not selected despite its higher-than-v37 strict Average
Score because only 127 results are evaluator-valid and explicitly finished.
Small canaries, Direct arms, incomplete runs, prepared runs, and other splits
are not candidates for the complete-panel AgentGraph baseline.

## Architecture and scientific controls retained from v16

The profile retains FlowSteer's progressive Canvas `edit -> execute ->
feedback`, free-text Agent declarations and contracts, directed relations,
unique Output Agent, and stateful one-action/one-observation WebShop Tool
boundary. It does not prescribe a shopping role, Agent count, or fixed
topology. v16's 127 single-Agent graphs and one two-Agent serial graph were
Director choices within AgentGraph, not Direct inference. ReAct is execution
control, not an assigned shopping role.

The executable v50 configuration preserves these v16 controls:

- Dataset/stage/split/selection/count:
  `webshop/development/validation/sequential/128`.
- Task IDs: the same native validation prefix `00500..00627`.
- Experiment and Direct generation seed: `20260825`.
- Sampling schedule purpose: `webshop_native_validation_stepwise_director_v1`.
- Catalog-order namespace: `webshop_stepwise_v16_local_catalog`.
- Rollouts/concurrency/task timeout: `1/1/900 seconds`.
- Environment action limit / Director round limit: `10/20`.
- Director: local Qwen3.5-9B, SGLang `supervisor_theta`, context 32768,
  JSON-schema decoding, action limit 1024, temperature 1.0, top-p 1.0,
  top-k -1, history window 2, and execute-on-edit.
- Director prompt identity: `agentgraph.director.minimal-neutral-scalar-stepwise.v2`.
- Executor: the evaluated v13 catalog's single Qwen3.5-9B arm, context
  **8192**, `max_tokens=4096`, and `chat_template_enable_thinking=false`.
  The Executor is not enlarged to a later 32768-context catalog.
- Canvas actions: `add_agent`, `modify_agent`, `delete_agent`, `set_relation`,
  `set_output`, `continue`, and `finish`; no `add_subgraph` migration.
- Maximum Agents 8, two-bit relation encoding, unique output and all Agents
  reaching output, and `preserve_diagnose_repair_augment` recovery policy.
- Original native WebShop reward/evaluator and public observation boundary.
- Training, GRPO, LoRA, policy synchronization, forced exploration and Skills
  remain disabled; optimizer updates and learning rate remain zero.

## Bounded repair scope

The v50 implementation scope is deliberately limited to public state and
action feedback, using the original task instruction, public observations,
currently legal actions, and previous public Action--Observation receipts:

1. Remove v16's global repeated-query hard rejection so a legal repeated
   search is executed as a real native WebShop transition. Preserve useful
   repeated-search feedback for the model without fabricating a non-state
   transition or automatically replacing the model's query.
2. Label the existing option/price purchase checks as **partial-scope**
   checks. Keep the complete original task visible and explicitly identify
   semantic requirements not verified by those checks. Passing a partial
   mechanical precondition must not be reported as satisfying every goal
   requirement. Retain the native option and price checks.
3. Thinly port the later public radio-group option and last-page `Next`
   compatibility repairs. These repairs must not install a shopping topology
   or perform an automatic purchase.
4. Preserve and regression-check action availability under a low remaining
   action budget. **The v16 baseline contains no budget-only `Buy Now`
   filter**, so v50 must not claim to remove such nonexistent baseline code.
   Budget feedback must not force `Buy Now` as the sole action simply because
   a short purchase path exists.

These statements describe the authorized implementation and regression
contract. They are not a claim that a new full evaluation has already passed.
No hidden goal attributes, terminal reward, or evaluator-private fields may
enter Director or Executor inputs. Native legal actions and native scoring
remain the authority; public feedback is not an alternate evaluator.

## Isolated endpoint and fresh evidence namespace

The new executable configuration is
`config/evaluation_webshop_v16_recovery_v50_128.yaml`. Its condition is
`webshop_v16_recovery_v50_128`, with fresh v50 Tool/policy/storage identities.
New AgentGraph outputs and reports belong only under:

- `artifacts/webshop_v16_recovery_v50_128/validation/`
- `reports/webshop_v16_recovery_v50_128/`

Both Director and Executor endpoints explicitly use
`http://127.0.0.1:8016/v1`. The new
`config/model_catalog_webshop_v16_recovery_v50.yaml` changes only the evaluated
v13 catalog's endpoint, not its model or metadata. Disabled policy-sync and
GPU supervisor URL fields also point to 8016. No v50 field silently routes
an Executor call to the original catalog's literal port 8015.

## Lossless reuse of the actually evaluated Direct arm

No new Direct inference is requested. The v50 config references the existing
v16 materialized Direct file by absolute path:

`/ssd1/iclr/1/.tmp/FlowSteer-webshop-stateful-action-v15/artifacts/webshop_stepwise_director_v16/development/direct_predictions.jsonl`

This file is present and contains 128 unique task IDs `00500..00627`, all
with valid evaluation records. It is the Direct arm materialized by the
completed v16 run, not the later v34/v37 Direct arm. The v16 manifest records
its own upstream reuse from:

`/ssd1/iclr/1/.tmp/FlowSteer-webshop-stateful-action-v14/artifacts/webshop_stepwise_director_v14/development/direct_predictions.jsonl`

No baseline AgentGraph trajectory, paired result, wrong demo, manifest, or
report is reused as a new v50 outcome. A full v50 result requires all 128
fresh AgentGraph tasks and strict full-denominator aggregation; partial or
prepared output must remain explicitly identified as incomplete.

## Upstream function mapping and verification

- SkillFlow `training/environment.py::_react_step` invokes the native
  `RAGENAdapter.step -> WebShopEnv.step -> WebAgentTextEnv.step` path.
  `_append_webshop_neutral_env_feedback` reports repeated queries without
  prohibiting them. This optional feedback is disabled by default upstream;
  v50 thinly adapts its feedback semantics, not an upstream default gate.
  `_build_react_prompt` and `_build_webshop_visible_state_block` retain the
  original instruction, action history, current observation and actions.
- Native `/home/test/datasets/WebShop/web_agent_site/envs/web_agent_text_env.py`
  `WebAgentTextEnv.step` accepts legal nonempty search and lowercases the
  action argument; `get_available_actions` exposes public radio elements.
  `SimServer.receive` handles navigation/reset and option assignment.
  `src/interactive/environment_execution.py::RAGENEnvironmentSession.step`
  transfers only the clicked public radio group/value to its receipt.
- Native `web_agent_site/engine/engine.py::search` passes the query to Lucene
  and uses `PRODUCT_WINDOW=10`. v50 preserves query punctuation and excludes
  Next only when the public result count proves the next page empty.
- Existing FlowSteer-derived Canvas/AgentRuntime/trajectory/FINISH machinery
  remains unchanged from v16. The only runtime delta is in the environment
  adapter and its public feedback. Full-goal semantic verification is **not**
  implemented as a new oracle: `task_satisfaction=unverified` is deliberate,
  and model reasoning must use public same-product evidence.

Targeted verification on 2026-09-05: **47 tests passed, 19 subtests passed**
across `test_webshop_constraint_coverage_v50.py`,
`test_environment_execution.py`, and
`test_webshop_stateful_action_policy_v15.py`. This includes genuine native
search recovery in the session interface, public radio binding, last-page
navigation, one-action return to Director, goal preservation and budget
availability. Fixtures are interface regressions, not benchmark scores.

Evaluation serving is isolated on physical **GPU4 / port8016**, keeping
v16's full 32768-token server context, 8192-token Executor registry limit,
concurrency 1 and non-thinking model condition. Other services are untouched.
No adapter is loaded or updated; LoRA server capability is not training.

## Canary-discovered receipt serialization correction

The two-task canary exposed one native evaluator replay mismatch: the initial
implementation placed `public_option_assignment` inside native `info`, whose
fields are compared exactly on replay. The correction stores it separately on
the session/transition and public receipt. Native `info` is now unchanged. This
does not change Agent/Director inputs or any sampled action; it corrects the
evaluator transport boundary, not the evaluator or its success criterion.

`scripts/replay_webshop_public_option_receipt.py` is an explicitly bounded
artifact migration using the existing `_retry_terminal_evaluator` and
EvidenceStore append protocol. It takes the complete frozen runtime trace,
requires each moved field to match the independently saved public receipt,
and replays **every** action through the unmodified native evaluator. Only the
known project-added field is removed from native `info`; observations, actions,
rewards and all original native fields remain unchanged and strictly checked.
The original invalid event is retained. No generation or trajectory resampling
occurs. The invalid evaluator prefix is not treated as a complete episode.
Newly collected trajectories do not require this migration.
