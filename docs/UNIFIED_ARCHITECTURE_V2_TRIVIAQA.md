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

- The complete unit suite passes: 771 tests passed across AgentGraph, Runtime,
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
bounded ReAct repair makes no new Tool-receipt progress, it adds one
Evidence Retriever or Repair execution unit when necessary and then exposes
only the exact admissible ingress relation.  Successful revision-live Agent
artifacts keep their predecessor identity during this recovery sequence.

The isolated r2 canary also failed Stable Zero, but exposed a separate
transactional defect.  On `triviaqa:tc_1`, the initial Reasoner provider
returned HTTP 429.  The live action mask correctly admitted only a cross-
provider `model_id` repair, while authoritative Canvas admission incorrectly
required the still-missing Verifier -> Formatter relation first.  The same
typed repair was therefore rejected for the remaining 27 turns.  On
`triviaqa:tc_3`, r2 completed the full lineage and explicit FINISH in four
turns with `Heworth, North Riding of Yorkshire`; the official evaluator
reported EM `0` and F1 `0.5` against accepted-answer canonicalization centered
on `York`.  That metric mismatch is retained as an evaluator result and is not
used to rewrite inference.  The report-only taxonomy uses the evaluator's
existing `partial_answer_overlap` diagnosis for this completed lineage, rather
than relabeling an earlier rejected completion as final retrieval failure.

Recovery revision r3 makes a typed mandatory Agent repair transactionally
authoritative before topology closure, matching the already-exposed action
mask.  Once the repair executes, normal semantic-spine relation closure
resumes.  r2 artifacts remain immutable under their own condition path.

The isolated r3 canary again failed Stable Zero, with the earlier provider
admission bug fixed.  `triviaqa:tc_1` recovered from HTTP 429, repaired one
ReAct exhaustion, completed in six Canvas turns and achieved EM/F1 `1/1`.
`triviaqa:tc_3` retained four successful Tool receipts and two read passages,
but later bounded calls added no receipt and repeatedly emitted a duplicate
normalized query.  Because unused theoretical Tool budget left
`tool_plan_exhausted=false`, the recovery state incorrectly admitted repeated
Reasoner repairs through `max_rounds` instead of augmentation.

Recovery revision r4 uses the existing receipt-progress invariant directly:
after one accepted repair, a complete ReAct call that fails without increasing
the successful Tool-receipt count opens non-destructive augmentation.  A call
that adds a receipt remains in repair so new evidence can be bound.  This does
not convert the failure to database coverage, consume accepted answers, or
change the evaluator.

The isolated r4 canary collected both tasks without collection failure but
failed Stable Zero at `0/2`.  It proved that the no-progress transition opened,
then exposed an action-domain inconsistency.  On `triviaqa:tc_1`, the Canvas
already contained Reasoner, Verifier and Formatter but lacked the final
semantic relation.  Constrained decoding exposed only `add_subgraph`, while
authoritative admission required the exact `set_relation`; 26 successive
sampled ADD actions were therefore rejected.  On `triviaqa:tc_3`, recovery
decoding exposed `min_new_agents=1,max_new_agents=3` while admission required
exactly one recovery Agent.  After a Formatter was added, the same ADD-versus-
relation conflict persisted through `max_rounds`.  The artifacts remain
isolated under `artifacts/unified_architecture_v2/triviaqa_recovery_r4`.

Recovery revision r5 makes the model action mask, live target domain and
authoritative admission use one state projection.  Existing recovery ingress
is routed first; an exact missing Reasoner -> Verifier -> Formatter edge is
closed next; a missing semantic responsibility is then added progressively;
the prospectively valid Formatter becomes Output; only then is another
recovery branch admitted.  Recovery ADD is constrained to exactly one
Evidence Retriever or Repair Agent, and partial semantic construction exposes
only its missing role family.  This is an inference-wire repair of the existing
FlowSteer Canvas boundary, not a topology template or training change.

The isolated r5 Stable Zero canary then passed `2/2` legal terminal chains with
zero collection, parsing or rejected-action failures.  `triviaqa:tc_1`
finished with `Harry Sinclair Lewis` and EM/F1 `1/1`; `triviaqa:tc_3`
finished with the evidence-supported surface `Heworth, North Riding of
Yorkshire` and official-style EM/F1 `0/0.5` against the accepted surface
`York`.  Both trajectories explicitly FINISHed after five Canvas turns.  The
two-task AgentGraph aggregate was EM `0.5`, F1 `0.75`; it is a chain canary,
not the fixed-128 accuracy estimate.

The r5 fixed-128 run was stopped after five persisted trajectories exposed two
systematic recovery loops.  It was not scored or reported as a completed
benchmark.  A provider HTTP 403 was correctly typed as
`provider_request_failure/permanent_configuration` in public Runtime feedback,
but the Canvas domain reserved model-only cross-provider recovery for HTTP 429
style transient failures.  The Director consequently repeated unrelated
Verifier modifications instead of switching the failed provider.  A second
case repeatedly bound `candidate_answer` to one proposition field while
`answer_slot.answer_field` selected the other; the public completion error did
not name the exact field correction.

The same partial run also exposed an answer-type narrowing defect.  The
question-only classifier mapped every `Who` question to `person`, although
the grammatical answer slot may be occupied by a team or organization (for
example, “Who won Super Bowl XX?”).  The evidence-supported answer
`Chicago Bears` reached the Formatter, but the Verifier correctly kept the
lineage inadmissible under the contradictory `person` constraint.  Because no
valid lineage existed, `max_rounds` returning `final_answer=null` was correct;
the fallback gate was not relaxed.

Recovery revision r6 keeps the same shared Runtime and changes only these
measured boundaries.  Both transient and permanent-configuration provider
failures now admit only a catalog-backed `model_id` repair, preferring another
provider and preserving every other Agent field.  Answer-field mismatch
feedback identifies which proposition field contains the candidate.  A `Who`
question now uses the non-narrowing `entity` answer-type constraint while
retaining the possessive-entity surface rule.  When a Verifier boolean rejects
evidence, entity/relation/alias binding, answer-slot type/cardinality, scope or
candidate surface, failure attribution targets the Reasoner that owns those
fields rather than repeatedly editing the Verifier.  An isolated
`tc_9/tc_10` regression configuration preserves the fixed-128 selection and
reuses existing Direct records.

### Known issues before r6 live validation

- Live SGLang constrained decoding and exact receipts must pass both the
  original two-task Stable Zero canary and the isolated `tc_9/tc_10`
  regression before the r6 fixed-128 run is allowed.
- The external Wikipedia index's operational coverage over all fixed 128 tasks
  remains to be measured by the completed run.
- The last-valid-lineage fallback intentionally remains a terminal failure and
  is excluded from training, even when its evaluator answer is correct.

### r6 live evidence and r7 recovery repair

The r6 two-task canary collected both tasks without collection failure but
failed Stable Zero at `1/2` explicit terminal lineages.  `triviaqa:tc_1`
completed in seven Canvas turns with answer `Sinclair Lewis` and EM/F1 `1/1`.
`triviaqa:tc_3` reached `max_rounds`, had no valid evidence lineage and
therefore correctly retained `final_answer=null`; it was not replaced by an
unsupported historical guess.  Provider HTTP 403 failures were all repaired
by model-only cross-provider edits, so provider recovery was not the remaining
cause.

The lossless r6 trajectory identified a state-conditioned action-domain defect.
After one auxiliary Retriever supplied any successful read receipt, direct
upstream provenance reopened `complete` after every evidence rejection.  The
Reasoner consequently emitted 284 invalid semantic actions but made only two
own searches and reads.  Existing read receipts did not contain the requested
birthplace relation.  Read-only queries against the same frozen Atlas DPR index
located public passages supporting the answer-bearing chain
`Dame Judi Dench -> born in -> Heworth -> part of the city of -> York`.
Accordingly this run is classified as retrieval recall/recovery failure, not
`knowledge_base_coverage_failure`; neither accepted answers nor evaluator data
were used by retrieval or generation.

Recovery revision r7 keeps the same Canvas and Runtime.  A direct upstream read
may admit the Reasoner's first completion, but an evidence/provenance rejection
against that state revokes completion until the Reasoner obtains a new
successful read through the public search/read continuation.  Duplicate
normalized-query feedback now explicitly requires a semantically distinct
entity-and-relation query while preserving Tool receipts.  When Runtime has a
typed failed Agent, execution-stage failure attribution now targets that Agent
instead of defaulting to the Formatter.  The provenance validator, evaluator,
accepted-answer boundary and terminal fallback gate are unchanged.

FlowSteer treats an accepted Canvas repair as a new execute-on-edit boundary.
Each ReAct call remains bounded by `max_turns_per_agent_call`; public
Action--Observation history and Tool receipts are retained across the repaired
revision as a project adaptation.  The aggregate r6 trace therefore contains
multiple bounded calls, not one unbounded SkillFlow invocation.

### r7 live evidence and r8 semantic repair

The r7 two-task canary again collected both tasks without collection failure
but passed only `1/2` legal terminal lineages.  The r6 retrieval-continuation
defect is fixed: `triviaqa:tc_3` performed query rewriting, increased top-k,
obtained a new successful Reasoner read and explicitly finished the legal
`Retriever -> Reasoner -> Verifier -> Formatter` lineage.  Its evidence-grounded
answer `Heworth, North Riding of Yorkshire` scored EM `0` and F1 `0.5` against
the accepted York surfaces, so this case is retained as an accepted-answer
canonicalization/granularity mismatch rather than rewritten during inference.

`triviaqa:tc_1` retrieved direct evidence for `Harry Sinclair Lewis`; the
Reasoner candidate, Verifier candidate and Formatter output all preserved that
same accepted surface.  The Verifier nevertheless returned
`minimal_answer_surface=false`, and the Runtime wrapped that field diagnosis
inside the full Verifier-artifact error.  The previous attribution predicate
recognized only an unwrapped `Verifier field ...` prefix, so the live action
domain repeatedly exposed Verifier repair instead of the Reasoner-owned
semantic repair and ended at `max_rounds` with no admitted terminal answer.
This is a semantic-verification/repair-attribution defect, not a retrieval,
Agent-communication or evaluator-input defect.

Recovery revision r8 keeps the same shared Canvas, Runtime, Tool backend and
evaluator.  A candidate that exactly copies the selected proposition argument
and one complete entity mention is no longer rejected merely because the
passage also contains a shorter coreferential alias.  The Reasoner must copy
entity surfaces exactly from each proposition's evidence span and represent
coreference through a separate evidence-supported identity proposition.
Wrapped Verifier false-field diagnostics are attributed to the Reasoner that
owns the candidate and answer slot; malformed Verifier structure remains a
Verifier repair.  These rules are answer-free and apply identically to the
shared HotpotQA/TriviaQA semantic protocol.

### r8 live evidence and r9 structured-action recovery

The r8 canary completed `triviaqa:tc_3` with a legal explicit FINISH in seven
Canvas turns.  Its bounded ReAct call used timeout recovery, query rewriting,
top-k escalation, one successful read and one answer-field schema correction
in five Action--Observation turns.  The final evidence-grounded answer remained
`Heworth, North Riding of Yorkshire`, with EM `0` and F1 `0.5`; the trajectory
is preserved as an accepted-answer canonicalization/granularity mismatch.

`triviaqa:tc_1` did not produce an evaluable r8 trajectory.  A hierarchical
`modify_agent` parameter generation ended with an unterminated JSON string.
The Canvas correctly returned an invalid-action observation, but the collector
treated `canvas.action=None` as a fatal receipt error before constructing the
TurnRecord.  The whole task was therefore marked `collection_failed`, even
though the sampled token/log-probability receipt itself was exact.  No zero
EM/F1 value from that unavailable paired record is interpreted as a task
score.

Recovery revision r9 restores FlowSteer's invalid-action continuation
semantics for this v3 hierarchical boundary.  The selected branch, live target
domain, parameter phase, token IDs and behavior log probabilities are still
validated exactly.  The malformed final sample is never repaired or converted
to an executed action: it is recorded with `action={}` and
`executed_prefix_tokens=0`, the public parse feedback is returned to the next
Director turn, and the trajectory is ineligible for GRPO.  A regression test
covers the complete collector path rather than only the JSON parser.

### Stable Zero status

Static architecture, 772 unit tests and both frozen data-selection
preconditions are complete.  The initial, r2, r3 and r4 canaries failed for
the documented causes; r5 **passed Stable Zero** but its incomplete fixed-128
run was stopped after measured recovery defects appeared.  Recovery revision
r6 failed Stable Zero for the documented retrieval-recovery defect.  Recovery
revision r7 fixed that defect but failed Stable Zero for the documented
semantic-verification/repair-attribution defect.  Recovery revision r8 is
answer-lineage complete for `tc_3` but failed Stable Zero because one malformed
Director parameter sample incorrectly aborted `tc_1` collection.  Recovery
revision r9 is **pending live Stable Zero and targeted regression**.  Formal
fixed-128 v2 EM/F1 remain pending.  No score is inferred from unit tests, a
two-task canary, an incomplete run, or the previous architecture.

## Historical comparison condition

The frozen v6.2 TriviaQA development run on the same ordered 128 tasks reported
Direct EM `35.15625%`, Direct F1 `40.81597%`, AgentGraph EM `51.5625%`,
AgentGraph F1 `60.104405%`, 116 explicit FINISH trajectories and 12 terminal
`max_rounds` failures.  These values are the pre-v2 comparison condition, not
v2 results.
