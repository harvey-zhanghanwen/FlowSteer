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

- The complete unit suite passes: 765 tests passed across AgentGraph, Runtime,
  Tool adapter, gateway prompts, Director action domains, configuration,
  records, collector, evaluator and reporting.  The only warning is the
  existing Pydantic class-config deprecation in `scripts/formatter.py`.
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

### First live canary and recovery revision

The first GPU0 canary used the frozen first two TriviaQA development tasks and
preserved its artifacts under `artifacts/unified_architecture_v2/triviaqa`.
Both tasks were collected with zero collection failures, but Stable Zero
failed at 1/2 valid terminal chains.  `triviaqa:tc_1` completed in seven Canvas
turns with Tool receipts, a Reasoner -> Verifier -> Formatter lineage, explicit
FINISH, answer `Harry Sinclair Lewis`, and EM/F1 `1/1`.  `triviaqa:tc_3`
reached all 28 Canvas turns with `max_rounds`, `final_answer=null`, and EM/F1
`0/0`.  It executed five search and five read actions; passage
`000000874654` contained “Dench was born in Heworth, North Riding of
Yorkshire.”  The failure therefore was not a pure retrieval-recall or database
coverage failure: entity/relation/answer-slot binding and terminal lineage did
not close.  These two tasks are a chain canary, not an accuracy estimate.

The failure exposed four concrete integration defects rather than a new
workflow requirement:

- entity-surface binding was misclassified as database coverage failure;
- the factual retrieval adapter did not propagate `tool_plan_exhausted=true`,
  so the existing repair-to-augmentation transition could not open;
- a rejected candidate-bearing Director action was replayed into the next
  model input;
- the model-visible continuation accumulated repeated identical invalid
  observations until it exceeded the useful context budget.

Recovery revision r2 fixes those boundaries while retaining the same
FlowSteer Canvas and SkillFlow bounded execution.  It preserves valid read
receipts for entity-binding repair, publishes the typed exhaustion receipt,
admits augmentation after one failed repair without new Tool-receipt progress,
rejects candidate-bearing pre-execution contracts and completion conditions,
does not replay rejected raw actions, and compacts only consecutive duplicate
model-visible invalid observations.  The full trajectory remains lossless.
The r2 condition and output paths are isolated so the failed first canary is
not overwritten.

The same state-conditioned Canvas domain also closes the declared
Reasoner -> Verifier -> Formatter relations before Output selection.  After a
Tool-plan-exhausted repair makes no new Tool-receipt progress, it adds one
Evidence Retriever or Repair execution unit when necessary and then exposes
only the exact admissible ingress relation.  Successful revision-live Agent
artifacts keep their predecessor identity during this recovery sequence.

### Known issues before r2 live validation

- The external Wikipedia index has only been exercised by the two-task failed
  canary; its operational coverage on the fixed 128 tasks is not yet measured.
- Live SGLang constrained decoding and exact receipts must pass the isolated
  r2 GPU0 Stable Zero canary before the 128-task run is allowed.
- The last-valid-lineage fallback intentionally remains a terminal failure and
  is excluded from training, even when its evaluator answer is correct.

### Stable Zero status

Static architecture, recovery tests and data-selection preconditions are
complete.  The initial live canary **failed** at 1/2; recovery revision r2 is
**pending live Stable Zero**.  Formal v2 EM/F1 remain pending.  No score is
inferred from unit tests, the two-task canary, or the previous architecture.

## Historical comparison condition

The frozen v6.2 TriviaQA development run on the same ordered 128 tasks reported
Direct EM `35.15625%`, Direct F1 `40.81597%`, AgentGraph EM `51.5625%`,
AgentGraph F1 `60.104405%`, 116 explicit FINISH trajectories and 12 terminal
`max_rounds` failures.  These values are the pre-v2 comparison condition, not
v2 results.
