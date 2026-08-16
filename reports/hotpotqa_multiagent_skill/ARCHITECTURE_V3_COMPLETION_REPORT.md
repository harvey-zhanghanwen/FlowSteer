# HotpotQA architecture-v3 completion report

## Completed

Architecture-v3 retains the v2 free AgentGraph search space and makes four
evidence-driven compatibility repairs:

1. catalog presentation order is stable within a same-task/same-condition
   rollout group and independent of the rollout sampling seed;
2. the Canvas rejects multiple, nested, and empty answer wrappers using one
   shared feedback/FINISH parser;
3. copied Direct predictions carry an explicit reuse receipt, and their source
   and counts survive final manifest generation;
4. the 128-task v3 config omits two fields that the evaluation runtime did not
   consume, leaving actual serving context and adapter state to preflight.

## Preserved architecture boundaries

- local Qwen3.5-9B remains the only Director;
- Executor model IDs remain the frozen canary-backed catalog;
- six atomic Canvas actions, free-text contracts, arbitrary legal DAGs,
  parallel/fan-in/fan-out and finite reciprocal pairs remain available;
- there is no fixed role enum, topology template, minimum Agent count,
  complexity reward, communication reward, exploration reward, or Skill
  reward;
- evaluator output never enters Canvas feedback or graph construction.

## Verification before live evaluation

- targeted Director, Canvas, Direct-reuse, Hotpot runner, and trajectory tests:
  39 passed;
- complete unit/regression suite excluding the unrelated optional pandas-only
  dataset-preparation module: 175 passed;
- Ruff on the changed runtime, runner, and tests: passed;
- v2 already demonstrated the complete live chain on 14/14 tasks.  The v3
  changes are protocol/receipt repairs and will next be checked by canary then
  resumed across the fixed 128-task architecture-development view.

## Status boundary

```text
ARCHITECTURE_INTERFACES_COMPLETE = YES
V3_STATIC_AND_UNIT_VERIFICATION = YES
V3_LIVE_DEV128_COMPLETE = NO
FORMAL_POLICY_STEP_000000 = NO
TRAINING_PERFORMED = NO
SKILL_PIPELINE_ACTIVE = NO
```

The existing adapter is explicitly a cross-dataset smoke warm start.  It is
used only to compare architecture behavior and is not the formal untrained
HotpotQA Step 0 requested for later controlled learning.
