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

Static validation before live regression:

- targeted Canvas/Director/Hotpot tests: 32 passed;
- full unit/regression suite excluding the unrelated optional pandas-only
  dataset-preparation module: 177 passed;
- lint and diff checks: passed;
- training, MACE, Bayesian, and Skills: disabled.

The live 12-task regression completed with 12/12 explicit FINISH, no natural
max-round terminal failure, and no operational/evaluator failure.  Strict
AgentGraph EM/F1 was 25.00/46.67 versus 58.33/70.56 for the reused paired
Direct records.  On the same 12 task IDs, v3 was 25.00/34.23 with 11/12
explicit FINISH.

Only the terminal semantic result is attributable: the prior max-round case
now reached an explicit FINISH in three turns.  The accuracy/F1 difference is
not a controlled v3-to-v4 comparison.  Although the per-task catalog order was
held fixed, all 12 first-turn Director generation seeds changed because the
evaluation runner passed a global selected-list position as `rollout_index`.
All 12 first actions changed and four final model assignments changed.  The
new no-op rejection branch was not exercised in this sampled run.

Consequently, V4 closes the observed no-op Canvas defect, but it is not a
frozen Training Step 0.  The next required compatibility repair is direct
reuse of SkillFlow's scientific sampling coordinate so task-level sampling is
independent of dataset subsetting and order.

```text
V4_STATIC_COMPLETE = YES
V4_LIVE_REGRESSION12_COMPLETE = YES
V4_TERMINAL_RECOVERY_VALIDATED = YES
V4_SCORE_GAIN_CAUSALLY_ATTRIBUTABLE = NO
FORMAL_POLICY_STEP_000000 = NO
```
