# HotpotQA architecture-v5 scientific sampling completion report

## Completion

V5 fixes the result-coordinate defect exposed by the V4 regression run.  It
directly ports SkillFlow's `ScientificSamplingCoordinate`, schedule identity,
canonical sampling identity, generation phase, and `derive_generation_seed`.
No Director prompt, AgentGraph action, topology, role, model catalog, runtime,
evaluator, terminal reward, MACE/Bayesian signal, or Skill visibility changed.

The runtime now follows one unambiguous rule:

```text
task + condition + task-local rollout ordinal + policy-step/anchor + turn
    -> exact Director generation seed
```

The selected dataset list and artifact/run names are not sampling inputs.
Every single-rollout evaluation therefore passes ordinal `0` for every task;
training keeps its existing task-local rollout loop.

## Exact receipt boundary

- One trajectory-level `director_sampling` receipt stores the SkillFlow
  algorithm, base seed, coordinate, and fixed Director phase (`action`).
- Every turn retains its exact `director_generation_seed` beside the sampled
  token IDs and behavior log probabilities.
- The collector checks that the server-reported seed equals the requested
  derived seed before accepting a turn.
- New trajectory identity includes the sampling receipt.
- `TrajectoryRecord.grpo_eligible` now requires the coordinate to belong to
  the task and every saved turn seed to be reconstructible from it.
- New records use schema `flowsteer.agentgraph.v2`; historical V1 artifacts
  remain unchanged and are not relabelled.

## Validation

- Scientific-sampling invariants, Director/catalog separation, exact
  collector receipts, task-local evaluation ordinal, trajectory admission,
  and affected smoke interfaces: 56 tests passed.
- Full unit/regression suite excluding the unrelated optional pandas-only
  dataset-preparation test module: 183 tests passed.
- Ruff on all changed Python files: passed.
- No model/API call, training, backward, optimizer update, LoRA publish,
  MACE/Bayesian update, or Skill activation was performed for this static
  compatibility repair.

## Remaining formal Step-0 blockers

Scientific replay is now complete, but formal `policy_step_000000` is still
blocked by four distinct SkillFlow compatibility boundaries:

1. deterministically materialize and preload/canary a never-updated initial
   Qwen3.5-9B LoRA;
2. freeze a Hotpot-only, write-once training task schedule and cursor;
3. save optimizer state from Step 1 onward and require exact adapter +
   optimizer continuation for later steps;
4. make SGLang publish and Director route switch one pause/drain transaction,
   then commit a step only after the new-policy canary succeeds.

The existing `theta_smoke_step_000001` remains a cross-dataset warm-start
diagnostic policy and is not renamed as formal Step 0.

```text
SCIENTIFIC_SAMPLING_READY = YES
CONTROLLED_SUBSET_REPLAY_READY = YES
FORMAL_POLICY_STEP_000000_READY = NO
READY_FOR_FORMAL_GRPO = NO
```
