# HotpotQA architecture-v3 fixed-dev128 analysis

## Executive conclusion

Architecture-v3 completed and persisted all 128 fixed HotpotQA development
tasks.  It confirms that the Director, progressive Canvas, free AgentGraph,
Executor routing, real message transport, Output inbox, evaluator, and exact
trajectory receipts are connected.  It does not establish a competitive Step
0 policy:

| Condition | Correct | EM | F1 |
| --- | ---: | ---: | ---: |
| Reused local Qwen3.5-9B Direct | 93/128 | 72.66 | 82.08 |
| architecture-v3 AgentGraph | 87/128 | 67.97 | 80.23 |
| AgentGraph - Direct | -6 | -4.69 | -1.85 |

All 128 evaluator receipts are valid.  There were zero collection failures,
but one natural Director trajectory reached 20 rounds without an accepted
FINISH.  It remains in the strict denominator with an empty prediction and
zero reward.  The final status is therefore
`completed_with_terminal_failures`, not Stable Zero success.

No training, backward pass, optimizer step, LoRA update/publish, MACE update,
Bayesian update, Skill discovery, or Skill injection ran.  The evaluated
adapter is the previously documented cross-dataset smoke warm start, not
formal HotpotQA `policy_step_000000`.

## Paired outcome

| Paired class | Tasks |
| --- | ---: |
| Both exact | 77 |
| AgentGraph only | 10 |
| Direct only | 16 |
| Neither exact | 25 |

F1 increased on 17 tasks, decreased on 17, and was unchanged on 94.  Among
the 25 both-nonexact tasks, AgentGraph F1 was on average 14.37 percentage
points higher than Direct, but the 16 Direct-only regressions outweighed the
10 Graph-only gains in EM.

The runner's evidence-only failure classes are:

- `correct`: 77;
- `architecture_gain`: 10;
- `architecture_regression_candidate`: 16;
- `partial_or_overlong_answer`: 17;
- `shared_reasoning_or_model_failure_candidate`: 7;
- `director_max_rounds`: 1.

## Version comparison

Round-01 uses the same 128 tasks and evaluator:

| Version | Graph EM | Graph F1 | Graph - Direct EM/F1 |
| --- | ---: | ---: | ---: |
| Round-01 | 75.00 | 84.44 | +2.34 / +2.36 |
| v3 | 67.97 | 80.23 | -4.69 / -1.85 |

V3 lost 9 exact answers and 4.21 F1 points relative to Round-01.  The loss was
largest on the 24 tasks labelled hard: -16.67 EM and -8.16 F1 points.

On the common fixed 14-task diagnostic:

| Version | Graph EM/F1 | Paired Direct EM/F1 |
| --- | ---: | ---: |
| Round-01 | 57.14 / 64.05 | 42.86 / 48.57 |
| v1 | 35.71 / 52.86 | 42.86 / 46.43 |
| v2 | 50.00 / 61.84 | 42.86 / 48.57 |
| v3 | 28.57 / 47.65 | 42.86 / 48.57 |

These are descriptive version comparisons, not single-change causal tests:
the prompt, task-conditioned catalog presentation, sampled actions, and
Executor routing differ.  V1 also used separately generated Direct outputs.

## Workflow behavior

Final graph distribution:

- 122 singleton graphs;
- 6 two-node, one-direction chains;
- 0 graphs with 3+ nodes;
- 0 parallel, fan-in, fan-out, or reciprocal final graphs.

The six two-node tasks scored 50.00 EM / 63.33 F1.  Their paired Direct score
was 33.33 EM / 57.78 F1, but the sample was policy-selected and too small for a
causal topology conclusion.  All six chains used the same model for both
nodes, so heterogeneous model collaboration inside a workflow was 0/6.

The 465 Director turns contained:

- 135 `add_agent`, 14 `modify_agent`, 1 `delete_agent`;
- 7 `set_relation`, 136 `set_output`, 147 `finish`;
- 25 invalid actions across 21 tasks.

Invalid actions were 7 malformed JSON, 15 unknown-field objects, 2 duplicate
JSON keys, and 1 non-object.  All 21 affected tasks recovered to an evaluator
record; these invalid samples should remain policy-learning evidence rather
than be hidden by a different evaluator or structural reward.

## Communication audit

For the six final multi-Agent graphs:

- runtime order followed the directed edge in all six;
- the final Output inbox source/target matched the final graph in all six;
- every envelope contained real source, target, message type, graph revision,
  request/dependency, and non-empty artifact content;
- every upstream artifact was present in the actual rendered Output prompt;
- all 128 tasks saved an Output inbox; singleton inboxes were correctly empty.

This validates transport and receipts on the observed cases.  It does not
validate causal communication use because Output Agents also saw the full
question/context and no new Normal-versus-Masked replay was performed.  No
communication reward is introduced.

## Model routing

Final Output routing used all eight frozen catalog IDs:

| Model | Tasks | Observed EM |
| --- | ---: | ---: |
| DeepSeek V4 Flash | 25 | 68.00 |
| DeepSeek V4 Pro | 17 | 70.59 |
| GPT-4o-mini | 25 | 64.00 |
| MiniMax M2.5 | 2 | 50.00 |
| MiniMax M3 | 8 | 87.50 |
| local Qwen3.5-9B | 10 | 50.00 |
| Qwen3.5 Flash | 29 | 65.52 |
| Qwen3.5 Plus | 12 | 83.33 |

Assignments were not randomized, so this is routing telemetry rather than a
model leaderboard.  There were 139 Executor calls: 129 through the configured
provider and 10 to local Qwen3.5-9B; no provider retry or fallback was recorded.

## Terminal and continuation behavior

Twenty FINISH actions were rejected across four tasks.  Three tasks repaired
the wrapper or workflow and eventually answered exactly.  One task,
`hotpotqa:5a84918e5542990548d0b2cf`, failed naturally:

1. ground truth was `Exeter Book`;
2. the Executor returned `[Widsith]`, both the wrong entity and wrong format;
3. the Canvas correctly rejected the missing answer wrapper;
4. the Director then issued 12 rejected FINISH actions and repeatedly selected
   the already-selected Output node;
5. six repeated `set_output` actions changed no graph revision and reused the
   same invalid cached output;
6. no `modify_agent`, model change, or useful graph edit occurred before the
   20-round limit.

This exposes one concrete Canvas defect: a revision-preserving no-op is called
accepted even though it cannot produce a new execution.  Rejecting such edits
with explicit feedback is a minimal FlowSteer-style state/feedback repair; it
does not change the action space or force a workflow.

## Representative Wrong Demos

| Task suffix | Ground truth / Graph answer | First saved error |
| --- | --- | --- |
| `5a7a0693` | Arthur's Magazine / `1844 1989` | Director asked for two years but dropped the requested comparison and magazine name. |
| `5ae3b4d0` | Todd Phillips / Old School | Contract targeted the film and year, while the question asked for its director. |
| `5a7d9019` | Dessau / Junkers | Contract changed a requested city into the manufacturer entity. |
| `5ae7a9c8` | 3000 metres steeplechase / none | A modified contract inserted an unsupported concurrence interpretation and steered the answer to none. |
| `5a736bfa` | Glenn Hughes / full comparison sentence | Contract requested evidence text instead of the one-person Output span. |
| `5ac3e8c6` | yes / unknown | Two-Agent transport was intact, but the Output contract precommitted to unsupported/false and restricted the answer to no/unknown. |
| `5a742488` | Dennis Howard Marks / Howard Marks | The full canonical name arrived in the Output inbox; the Output Executor truncated it. |
| `5a84918e` | Exeter Book / no final answer | Wrong/unwrapped Executor result plus repeated no-op continuation exhausted 20 rounds. |
| `5ac2a912` | The Wolfhounds / two bands plus years | Initial contract requested intermediate year lines instead of the comparison result. |
| `5abd9054` | Jaime Meline / El-P | Communication was correct; the alias was not resolved to the evaluator's canonical name. |

The dominant regression pattern is target-field drift in Director contracts,
followed by Output span fidelity and Executor reasoning.  Message loss and
evaluator failure are not supported as primary causes.

## Telemetry

- 604 total attempts: 465 Director and 139 Executor;
- 1,702,849 input tokens and 128,234 output tokens;
- per-task summed-call latency median 7.39 seconds, P95 59.37 seconds, maximum
  109.00 seconds;
- Qwen3.5 Flash and Plus produced about 97.6% of Executor output tokens;
- all 128 Direct records were copied from Round-01 with explicit reuse
  receipts; new Direct calls were zero.

The final reporting invocation reused every trajectory and made no new
Executor call.  Exact per-turn times and provider receipts remain in the
trajectory files; summed latency is not wall-clock time because concurrency
was four.

## Architecture versus policy judgment

### Confirmed code/interface behavior

- arbitrary legal DAG and finite reciprocal search space exists;
- graph execution, relation direction, envelope injection, Output inbox,
  evaluator, and receipt persistence work on observed graphs;
- the hard answer-wrapper gate works and never changes reward;
- task/catalog/policy/evaluator versions and Direct reuse are recorded.

### Confirmed engineering defect

- revision-preserving `set_output` and other no-op edits are accepted and can
  recycle a known-bad cached result.

### Director policy-learning problems

- target-field drift in free contracts;
- 95.3% singleton final graphs and no heterogeneous multi-model graph;
- 25 invalid actions;
- poor recovery from repeated terminal feedback;
- little effective model switching after a failed execution.

### Executor/model limitations

- evidence/entity binding and canonical alias errors;
- Output truncation despite correct upstream artifact;
- occasional protocol non-compliance and very long reasoning outputs.

The data do not justify fixed roles, mandatory decomposition, forced model
diversity, topology templates, or structural/communication reward.  After the
minimal no-op feedback repair, further manual prompt expansion risks
overfitting this development set.  The main bottleneck should be treated as
untrained Director policy behavior.

## Readiness judgment

```text
ARCHITECTURE_RUNTIME_CHAIN_COMPLETE = YES
FULL128_EVALUATION_RECORDED = YES
FULL128_ALL_TASKS_EXPLICIT_FINISH = NO
AGENTGRAPH_OUTPERFORMS_DIRECT = NO
MULTI_MODEL_ROUTING_ACROSS_TASKS = YES
HETEROGENEOUS_ROUTING_WITHIN_GRAPH = NO
MULTI_AGENT_COMMUNICATION_TRANSPORT_VALIDATED = YES
MULTI_AGENT_COMMUNICATION_CAUSAL_GAIN_VALIDATED = NO
FORMAL_POLICY_STEP_000000 = NO
STEP0_TO_STEPN_LEARNING_VALIDATED = NO
SKILL_EVOLUTION_VALIDATED = NO
READY_FOR_NEXT_DATASET = NO
```

The next architecture action is limited to no-op edit rejection and precise
reporting semantics.  Formal learning cannot start until a deterministic
initial HotpotQA LoRA, Hotpot-only immutable step runner, optimizer-state
continuation, and adapter preload/publish receipts are connected from the
existing SkillFlow/FlowSteer paths.
