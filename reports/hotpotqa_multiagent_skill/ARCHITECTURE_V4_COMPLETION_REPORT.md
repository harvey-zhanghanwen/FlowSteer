# HotpotQA architecture-v4 completion report

V4 is limited to one FlowSteer-style Canvas feedback repair: a non-FINISH
action that leaves graph revision unchanged is rejected rather than called an
accepted edit with a cached execution.  Terminal format feedback also states
that the Output contract/model or graph must change before retrying.

The free AgentGraph search space, prompt roles, model pool, relation semantics,
runtime, evaluator, terminal reward, and training-disabled boundary are
otherwise unchanged.  The 12-task regression config freezes the v3 catalog
presentation namespace so the targeted comparison does not introduce an
unrelated model-order change.

Before live regression:

- targeted Canvas/Director/Hotpot tests: 32 passed;
- full unit/regression suite excluding the unrelated optional pandas-only
  dataset-preparation module: 177 passed;
- lint and diff checks: passed;
- training, MACE, Bayesian, and Skills: disabled.

```text
V4_STATIC_COMPLETE = YES
V4_LIVE_REGRESSION12_COMPLETE = NO
FORMAL_POLICY_STEP_000000 = NO
```
