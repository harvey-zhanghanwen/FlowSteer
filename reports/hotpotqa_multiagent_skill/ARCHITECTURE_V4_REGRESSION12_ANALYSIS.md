# HotpotQA architecture-v4 regression12 analysis

## Executive result

V4 was a one-defect FlowSteer-style Canvas repair: a graph edit that leaves
the revision unchanged is rejected, and malformed terminal feedback tells the
Director to change the Output contract/model or graph before retrying.  It did
not change the search space, model catalog, evaluator, runtime, terminal
reward, or policy weights.

| Condition | N | Explicit FINISH | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Reused local Qwen3.5-9B Direct | 12 | n/a | 58.33 | 70.56 |
| V3 AgentGraph on the same task IDs | 12 | 11 | 25.00 | 34.23 |
| V4 AgentGraph | 12 | 12 | 25.00 | 46.67 |

V4 versus the paired Direct records was -33.33 EM and -23.89 F1.  V4 versus
V3 on the same task IDs was unchanged in EM and +12.44 F1, but that score
difference is not causally attributable to the V4 edit.

## What the live run does validate

- All 12 trajectories, evaluator receipts, Director receipts, Output inboxes,
  and paired records were saved.
- All 12 trajectories ended through explicit FINISH; there were no
  operational or evaluator failures.
- The previous natural max-round task
  `hotpotqa:5a84918e5542990548d0b2cf` now reached explicit FINISH in three
  turns instead of exhausting 20 rounds.  Its answer remained wrong, so this
  is terminal recovery, not a task-performance gain.
- One malformed terminal repair path was exercised on
  `hotpotqa:5ab30bbb55429976abd1bc39`: the Director made a real graph/model
  change, re-executed, and finished.  The final answer was wrong, so the event
  validates recovery mechanics only.

The new revision-unchanged no-op rejection branch was not triggered by these
12 newly sampled trajectories.  Its behavior is therefore covered by the
unit/regression tests and by the V3 failure evidence that motivated it, not by
a live V4 trigger.

## Calls and tokens (diagnostic, not causal)

| Metric | V3 on same 12 IDs | V4 |
|---|---:|---:|
| Director/Canvas calls | 76 | 44 |
| Rejected FINISH actions | 20 | 1 |
| Executor calls | 21 | 16 |
| Director prompt tokens | 259,035 | 140,542 |
| Director output tokens | 2,561 | 1,890 |
| Executor input tokens | 35,178 | 25,897 |
| Executor output tokens | 3,874 | 9,507 |
| Cumulative Executor latency | 142.59 s | 143.30 s |

These changes are also sampling-confounded.  In V4, two `qwen3.5-flash`
calls on one task produced 9,330 tokens (98.14% of all V4 Executor output
tokens) and 95.41 seconds of cumulative Executor latency.  Fewer calls did not
therefore reduce cumulative latency in this small batch.  No new Direct API
call was made, and the artifacts do not contain a price schedule, so no
monetary-cost estimate is reported.

## Why the score comparison is confounded

The 12 task IDs and their catalog presentation order were fixed, but their
Director sampling was not:

- 12/12 first-turn generation seeds differ between the V3 full-128 run and
  the V4 12-task subset;
- 12/12 first actions differ;
- 4/12 final model assignments differ;
- the three V3-correct tasks and three V4-correct tasks are disjoint.

The cause is concrete: `LiveSmokeBackend.collect` interpreted
`rollout_index` as a sampling offset, while the evaluation runner supplied the
task's global position in the selected list.  Subsetting or reordering tasks
therefore changed the policy sample for the same task.  Holding catalog order
fixed removed only the prompt-order confound, not this generation-seed
confound.

The valid conclusion is:

```text
NO_OP_CANVAS_DEFECT_FIXED = YES
EXPLICIT_TERMINAL_RECOVERY_OBSERVED = YES
V4_SCORE_GAIN_CAUSALLY_ATTRIBUTABLE = NO
SCIENTIFIC_SAMPLING_RECEIPT_REQUIRED = YES
```

## Failure evidence

The V4 output contained two correct cases, one graph-only architecture gain,
five Direct-correct/Graph-wrong regression candidates, two partial or
overlong answers, and two shared reasoning/model failures.  Since sampling
changed, these are diagnostic examples for V4 behavior; they are not paired
counterfactual evidence about the no-op edit.

The evidence continues to separate the remaining problems:

- **Architecture defect closed:** accepted same-revision Canvas edits could
  silently reuse a bad cached execution and encourage an ineffective loop.
- **Sampling/replay defect still open:** task samples are tied to selected-list
  position rather than a scientific task/rollout coordinate.
- **Director policy limitation:** even after terminal feedback, a successful
  structural recovery can select a wrong answer.
- **Executor/model limitation:** several final spans are wrong or only
  partially match the reference.

No evidence in this run supports adding a fixed role, mandatory topology,
agent-count reward, communication reward, or Hotpot-specific answer rule.

## Source and protocol boundary

The no-op behavior follows FlowSteer's existing explicit pending-modification
feedback boundary in `workflow_env.py::_format_pending_modify_prompt_request`.
The required sampling repair must reuse SkillFlow's
`ScientificSamplingCoordinate` and `derive_generation_seed`; it should not be
replaced by another project-local seed formula.

No training, backward, optimizer step, LoRA update/publish, MACE update,
Bayesian update, or Skill activation occurred in V4.
