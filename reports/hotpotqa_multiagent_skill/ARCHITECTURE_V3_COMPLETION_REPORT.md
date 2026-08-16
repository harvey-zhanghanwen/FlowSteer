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
  dataset-preparation module: 176 passed after the final reporting fix;
- Ruff on the changed runtime, runner, and tests: passed;
- v2 already demonstrated the complete live chain on 14/14 tasks.  The v3
  changes are protocol/receipt repairs.  Canary completed 2/2, then the same
  output directory resumed across all 128 fixed development tasks without
  repeating the successful canary calls.

## Live development result

- Direct: 93/128, 72.66 EM, 82.08 F1;
- AgentGraph: 87/128, 67.97 EM, 80.23 F1;
- paired delta: -4.69 EM and -1.85 F1 percentage points;
- 127/128 explicit FINISH, one natural `max_rounds` failure retained as zero;
- 128/128 evaluator receipts valid; 0 collection failures;
- 128/128 Direct records reused from Round-01; 0 new Direct calls.

The live interfaces are complete, but v3 does not meet the architecture
performance baseline and does not validate broad multi-Agent collaboration.
The detailed evidence is in `ARCHITECTURE_V3_DEV128_ANALYSIS.md`.

## Status boundary

```text
ARCHITECTURE_INTERFACES_COMPLETE = YES
V3_STATIC_AND_UNIT_VERIFICATION = YES
V3_LIVE_DEV128_RECORDED = YES
V3_ALL_TASKS_EXPLICIT_FINISH = NO
V3_OUTPERFORMS_LOCAL_DIRECT = NO
MULTI_AGENT_COLLABORATION_VALIDATED = NO
FORMAL_POLICY_STEP_000000 = NO
TRAINING_PERFORMED = NO
SKILL_PIPELINE_ACTIVE = NO
```

The existing adapter is explicitly a cross-dataset smoke warm start.  It is
used only to compare architecture behavior and is not the formal untrained
HotpotQA Step 0 requested for later controlled learning.
