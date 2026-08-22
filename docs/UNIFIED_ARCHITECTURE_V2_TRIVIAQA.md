# Unified architecture v2 — TriviaQA completion report

## Scope

This version extends the existing shared HotpotQA AgentGraph architecture to
TriviaQA.  It does not create a second dataset-specific Runtime.  The executed
path is:

`Question -> Qwen3.5-9B Flow-Director -> progressive Canvas -> AgentGraph Runtime -> search/read Tool receipts -> Reasoner -> Verifier -> Formatter -> Evaluator -> Trajectory`.

The pre-v2 source is recoverable from branch
`backup/pre-unified-architecture-v2-triviaqa-20260822` at commit
`59c88ea5232190bec0a579fc883adeea9978a133` on the existing authenticated
backup remote.  The original `origin` rejected the configured GitHub identity
with HTTP 403; no existing branch was overwritten.

## Architecture Completion Report

### Completed

- One shared semantic protocol, `qa_verified_answer_lineage_v2`, for HotpotQA
  and TriviaQA over the existing FlowSteer progressive Canvas.
- A compact neutral Director prompt and Canvas-authoritative live action
  domains; no answer, fixed workflow template, minimum topology, or structural
  reward is injected.
- SkillFlow-compatible bounded `search/read` Action–Observation execution with
  spelling normalization, alias expansion, entity disambiguation, query
  rewriting and increasing top-k recovery.
- Entity/evidence provenance admission before reasoning, and explicit
  `knowledge_base_coverage_failure` only after bounded strategy exhaustion.
- Separate Reasoner, Verifier and Formatter responsibilities.  ReAct is an
  execution strategy, never an Agent role.
- `preserve -> diagnose -> repair/augment` recovery and an immutable last valid
  answer/Runtime/graph lineage for `max_rounds` fallback.
- Exact trajectory fields for fallback use and report-only TriviaQA failure
  taxonomy.  Official-style EM/F1 computation is unchanged.
- A fixed sequential 128-task TriviaQA development selection identical in task
  ID and order to the previous v6.2 condition.

### Verified without a live model

- 335 targeted unit/regression tests passed across AgentGraph, Runtime, Tool
  adapter, gateway prompts, Director action domains, configuration, records,
  collector, evaluator and reporting.
- `git diff --check` passed.
- Prepare-only selected exactly 128 TriviaQA tasks, from `triviaqa:tc_1` through
  `triviaqa:tc_223`, in the same order as the previous condition.

### Reserved but inactive

- Skill retrieval/injection and Skill lifecycle.
- MACE exploration and Bayesian posterior updates.
- GRPO, backward, optimizer updates, LoRA publication and policy hot-sync.
- Training/gradient use of the declared learner and gradient-replica GPUs.

These are configuration and module boundaries only in this evaluation round;
they are not reported as trained or ACTIVE.

### Known issues before live validation

- The external Wikipedia retrieval index has not yet been measured on the v2
  fixed 128 tasks; operational database coverage can only be diagnosed after a
  real bounded retrieval episode.
- Live SGLang constrained decoding and all provider/model receipts still need
  a GPU0 Stable Zero canary under this exact configuration.
- The last-valid-lineage fallback intentionally remains a terminal failure and
  is excluded from training, even when its evaluator answer is correct.

### Stable Zero status

Static architecture and data-selection preconditions are complete.  Live
Stable Zero is **pending** until a GPU0 canary produces a valid end-to-end
trajectory and evaluator receipt.  Formal v2 EM/F1 are also pending; no score
is inferred from unit tests or from the previous architecture.

## Historical comparison condition

The frozen v6.2 TriviaQA development run on the same ordered 128 tasks reported
Direct EM `35.15625%`, Direct F1 `40.81597%`, AgentGraph EM `51.5625%`,
AgentGraph F1 `60.104405%`, 116 explicit FINISH trajectories and 12 terminal
`max_rounds` failures.  These values are the pre-v2 comparison condition, not
v2 results.
