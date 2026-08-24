# Unified architecture v2 — TriviaQA completion report

## Scope

This version extends the existing shared HotpotQA AgentGraph architecture to
TriviaQA.  It does not create a second dataset-specific Runtime.  The executed
path is:

`Question -> Qwen3.5-9B Flow-Director -> progressive Canvas -> AgentGraph Runtime -> role-conditional Agent execution/communication -> selected Output Agent -> Evaluator -> Trajectory`.

For the current fixed-128 condition, Evidence Retriever, Reasoner, Verifier and
Formatter are required terminal responsibilities, not a prompt-fixed Agent
inventory or prescribed serial workflow.  The Director may still construct
parallel retrieval, fan-in, repair and reciprocal non-Formatter subgraphs.
ReAct is selected per Agent as an execution mode; the live Canvas domain
determines which Agents, relations and Output role are admissible for the
current state.

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
- These five labels are a model-visible execution schedule.  The lossless
  trace records the actual query, top-k and Tool receipt.  The public
  continuation now records each stage's question-invariant validity and
  whether the attempt was conditioned on prior Tool receipts, including the
  associated public passage IDs.  This is observability rather than an oracle:
  only the later entity/relation/evidence gate may call a read receipt grounded.
- Entity/evidence provenance admission before reasoning, and explicit
  `knowledge_base_coverage_failure` only after bounded strategy exhaustion.
- Successful public `read` Tool receipts follow the routed artifact lineage
  from Retriever to Reasoner and onward to Verifier/Output.  The Runtime keeps
  upstream and current receipts in artifact metadata with structural
  de-duplication, so a downstream semantic check does not lose the evidence
  provenance used by its predecessor.
- Geographic-scope grounding is proposition-clause scoped: a named scope may
  satisfy a location claim only when it occurs in the same evidence clause
  that binds the proposition subject, relation and object.  An unrelated
  clause in the same retrieved passage cannot close that semantic lineage.
- Required Evidence Retriever, Reasoner, Verifier and Formatter terminal
  responsibilities with no prompt-fixed serial chain.  ReAct is an execution
  mode selected per Agent, never an Agent role.
- `preserve -> diagnose -> repair/augment` recovery and an immutable last valid
  answer/Runtime/graph lineage for `max_rounds` fallback.
- Exact trajectory fields for fallback use and report-only TriviaQA failure
  taxonomy.  Official-style EM/F1 computation is unchanged.
- A fixed sequential 128-task TriviaQA development selection identical in task
  ID and order to the previous v6.2 condition.

### Verified without a live model

- The current complete unit suite passes: 958 tests and 177 subtests across
  AgentGraph, Runtime, Tool adapter, gateway prompts, Director action domains,
  configuration, records, collector, evaluator and reporting.  The only
  warning is the existing Pydantic class-config deprecation in
  `scripts/formatter.py`.
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

### r9 live evidence and r10 retrieval recovery

The r9 main Stable Zero canary produced `2/2` legal explicit-FINISH lineages.
`triviaqa:tc_1` returned `Sinclair Lewis` with EM `1.0` and F1 `1.0`;
`triviaqa:tc_3` retained the evidence-grounded surface
`Heworth, North Riding of Yorkshire` with EM `0.0` and F1 `0.5`.  The latter
remains an evaluator-only accepted-answer canonicalization/granularity
mismatch and is not rewritten by inference.

The required isolated `tc_9/tc_10` regression passed only `1/2` in r9.
`tc_10` explicitly finished with `Chicago Bears` (EM/F1 `1.0/1.0`).  `tc_9`
ended at `max_rounds` with no legal evidence lineage.  Its failed Reasoner had
two successful searches and two reads, but 296 later attempts repeated the
same normalized query.  Consecutive duplicate feedback was compacted without
a repeat count, so the deterministic Director input did not change.  Every
new Evidence Retriever started without the failed Reasoner's public
Action--Observation history or Tool receipts, reread the same irrelevant
David Crockett passage, and the generic completion schema admitted unsupported
city text.  This is retrieval recall/entity disambiguation and recovery-state
handoff failure, not `knowledge_base_coverage_failure`: the bounded strategy
sequence was never genuinely exhausted.

Recovery revision r10 is a minimal shared-runtime repair.  Duplicate public
errors retain one compact observation plus `repeat_count`; a semantic Evidence
Retriever may complete only with an answer-free, receipt-bound artifact
containing question scope, explicit entity identity, target relation, exact
evidence span and passage ID.  When one ReAct-repair-exhausted Reasoner is
augmented by one new ReAct Evidence Retriever, the FlowSteer edit--execute
boundary temporarily projects only that Reasoner's public Action--Observation
history and Tool receipts to the new node, phase-bound to `single` and recorded
with `continuation_source_agent_id`.  Existing receipts continue to count
against the same bounded Tool budget; no semantic answer, hidden reasoning,
Ground Truth or evaluator state is transferred.  Agent contracts may express
query rewriting, entity disambiguation and expanded top-k responsibilities,
but concrete query, limit and passage-ID values remain owned by the Runtime's
state-conditioned Tool schema.

### r10 live evidence and r11 role-bound recovery

The r10 main canary collected both tasks without operational or evaluator
failure but passed only `1/2` legal terminal lineages.  `triviaqa:tc_1`
explicitly finished with `Harry Sinclair Lewis` and official EM/F1 `1.0/1.0`.
`triviaqa:tc_3` ended at `max_rounds` with a null terminal answer and EM/F1
`0.0/0.0`, while its Retriever had already read the public passage
`atlas-dpr-wikipedia:000000874654` containing the exact sentence
`Dench was born in Heworth, North Riding of Yorkshire.`  The failure therefore
is not database coverage failure.

Two implementation defects caused the terminal failure.  First, the Retriever
submitted the abstract label `birthplace` for a field that requires an exact
predicate surface from the cited span (`was born in`).  The strict provenance
gate correctly rejected that field, but r10 incorrectly routed the rejection
to a new search/read instead of repairing the structured artifact on the
preserved read; repeated retries were then mislabeled
`knowledge_base_coverage_failure`.  Second, the cross-Agent projection copied
the failed Reasoner's role-specific completion observations into a new
Retriever.  One helper received 64 such prior turns, including rejected
`Chester` completions, and then repeated normalized queries.  A helper whose
first provider call failed also lost the projected Tool state before its
model-only repair.

Recovery revision r11 keeps the same strict receipt, exact-span and exact-
predicate requirements.  It routes Retriever field-shape and lexical-field
alignment errors through the existing structured-artifact repair on the same
successful read; missing receipt/span or unproven alias lineage still requires
new evidence.  Entity fields explicitly require concise mentions rather than
the whole question or sentence.  Cross-Agent recovery now transfers only
dispatched Tool Action--Observation entries and same-Tool receipts, excluding
source-role completion errors while retaining failed Tool dispatches that
consumed budget.  If the target's first provider call fails, the filtered Tool
projection survives the admitted model repair.  The Director admission gate
also rejects natural-language hard-coded search phrases and concrete alias
mappings while allowing neutral Tool references and retrieval responsibilities.

### r11 live evidence and r12 recovery preparation

The r11 main Stable Zero canary collected both frozen tasks but passed only
`1/2` legal terminal lineages.  `triviaqa:tc_1` explicitly finished with
`Harry Sinclair Lewis` and official EM/F1 `1.0/1.0`.  `triviaqa:tc_3`
reached `max_rounds`, retained `final_answer=null`, and scored EM/F1
`0.0/0.0`.  Because the main gate failed, the isolated `tc_9/tc_10`
regression and fixed-128 run were not started.

The r11 lossless trajectory identifies two measured causes.  First, `node_4`
was a replacement Reasoner with several active predecessors.  The Runtime's
AND-join required every routed predecessor to be ready, so this replacement
could not execute even though a usable evidence branch existed.  Second,
preserved auxiliary artifacts from `node_5` and `node_6` bound birth-date
evidence to the requested birthplace relation.  Those artifacts were therefore
false-positive evidence, not a valid downstream takeover, and cannot authorize
deletion of the failed responsibility or terminal admission.

Recovery revision r12 is a minimal shared-Runtime correction, not a new
workflow.  It retains FlowSteer's execute-on-edit transaction and
permits replacement takeover only after a successful same-role downstream
artifact has actually executed; deletion remains unavailable until that
takeover exists.  SkillFlow's public Tool Action--Observation entries and
receipts remain the only cross-revision continuation state.  The auxiliary
Retriever uses an answer-free `evidence_proposition` with the same proposition
shape already required from the Reasoner, and constrains the proposition object
with the question-only `qa_answer_type_constraint`; it does not own or receive
an answer.  A sibling fail-fast `CancelledError` is treated as orchestration
cancellation rather than attributed as an Agent execution failure.

### r12/r13 live evidence and r14 recovery preparation

The r12 main Stable Zero canary passed `2/2` legal explicit-FINISH lineages.
`triviaqa:tc_1` scored official EM/F1 `1.0/1.0` and `triviaqa:tc_3`
scored `0.0/0.5`; the latter still finished explicitly with a receipt-grounded
surface and remains an evaluator-only accepted-answer canonicalization or
granularity mismatch.  These two development tasks are a gate result, not a
fixed-128 accuracy estimate.

The required isolated r12 `tc_9/tc_10` regression passed `0/2`.  Both tasks
ended at `max_rounds` with `final_answer=null` and no Env-owned valid evidence
lineage.  A Reasoner, Verifier or Formatter emitting a plausible answer string
did not replace the required Tool-receipt-grounded
`Evidence Retriever -> Reasoner -> Verifier -> Formatter` lineage.  The
fixed-128 run was therefore not started.

The lossless r12 trajectories expose three recovery-boundary defects.  First,
successful reads could not be materialized as a valid Retriever artifact when
the question entity was a contextual event mention rather than one argument
of the answer-bearing proposition.  Second, relation edits oscillated around a
failed auxiliary ingress instead of establishing one executable replacement
path.  Third, a measured `repair_exhausted` Retriever remained in the live
`modify_agent` target domain and consumed the remaining Canvas rounds.  These
are structured-artifact and orchestration defects, not evidence that a correct
string may bypass provenance or terminal validation.

Recovery revision r13 is a minimal continuation of the same shared architecture.
It keeps FlowSteer's state-conditioned action mask, authoritative admission and
execute-on-edit transaction; preserves SkillFlow's StructuredAction and public
Tool continuation; and adds only the necessary contextual-entity adaptation
required to represent an evidence proposition whose question entity provides
scope rather than a proposition argument.  Its Tool contract remains
`skillflow.qa-retrieval.factual-semantic-retry.v6`.

The r13 main Stable Zero canary produced `2/2` legal explicit-FINISH lineages.
The mandatory isolated r13 `tc_9/tc_10` regression produced one valid result out
of two: `triviaqa:tc_10` explicitly finished at official EM/F1 `1.0/1.0`, while
`triviaqa:tc_9` was recorded as `collection_failed` because the declaration
phase sampled text was not complete JSON.  The old collection-failure path did
not persist the raw sampled text or its phase receipt, so that artifact cannot
support a more specific decoding-failure diagnosis.  The fixed-128 run was not
started.

Recovery revision r14 was prepared under new main and isolated condition IDs,
artifact roots and report paths.  It reuses the existing strict malformed
final-parameter boundary: an exact receipt is retained, the rejected sample is
represented as `action={}`, zero action prefix is executed, and public parse
feedback is returned to the next Director turn.  The JSON parser is not
loosened and a partial `ADD` is never executed.  Tool version v6, the ordered
fixed-128 selection, isolated `triviaqa:tc_9`/`triviaqa:tc_10` IDs and the
question-only Direct reuse source remain unchanged.  Training, GRPO, backward,
optimizer updates, policy synchronization, Skills and exploration remain
disabled.

Static integration for r14 passed the complete unit suite (`803 passed`,
`142 subtests passed`) and both prepare-only freezes.  The main freeze retains
the same ordered 128 tasks, while the isolated freeze retains exactly
`triviaqa:tc_9` then `triviaqa:tc_10`; both continue to use the frozen
question-only Direct prediction source.

The r14 main Stable Zero run collected both frozen tasks but passed only `1/2`
legal terminal lineages.  `triviaqa:tc_1` recovered from a receipt-preserved
malformed declaration, explicitly finished with `Harry Sinclair Lewis`, and
scored official EM/F1 `1.0/1.0`.  `triviaqa:tc_3` ended at `max_rounds` with
`final_answer=null`.  Its public Tool receipts included the exact read passage
stating that Dench was born in Heworth, North Riding of Yorkshire, so this is
not a retrieval-recall or database-coverage failure.  The retrieved fact was
lost at the structured identity/alias boundary; repeated failed Retriever
ingress then formed an AND-join that prevented later replacements from
executing.  Because the main gate failed, the r14 isolated regression and
fixed-128 run were not started.

Recovery revision r15 is a minimal correction on the same shared architecture.
The Tool v7 adapter admits a completion-only `entity_identity` repair when one
successful read receipt itself establishes the question surface (optionally
after removing one leading linguistic honorific), passage title and exact-span
evidence surface.  It does not choose relation direction or the answer slot,
and it preserves r13 contextual event anchors.  The Canvas recovery path now
executes a same-role Evidence Retriever replacement as an isolated accepted
`ADD_SUBGRAPH` before exposing one directed replacement-to-Reasoner relation.
Only a successful replacement artifact may enter that relation domain; failed
ingress edits must strictly remove edges, and takeover/deletion still uses the
existing receipt-gated path.

Static r15 integration passed the complete unit suite (`808 passed`, `144
subtests passed`).  The main prepare-only freeze retains the ordered 128 tasks
from `triviaqa:tc_1` through `triviaqa:tc_223`, and the isolated freeze retains
exactly `triviaqa:tc_9` then `triviaqa:tc_10`.  Both retain the frozen 128-row
question-only Direct source.  Training, optimizer updates, Skills and policy
synchronization remain disabled.  r15 has not yet produced a live score.

Recovery revision r16 keeps the shared Runtime and introduces Tool v8.  It
allows the same normalized retrieval query only at a strictly larger top-k,
rejects an exact repeated `(query, top_k)` pair, and reuses one public repair
instruction in both the next model input and the exhausted trace.  Answer-type
mismatch is evaluated before an identity-only repair.  On the Canvas side, a
same-role auxiliary replacement is first executed as an isolated functional
unit; a successful Evidence Retriever replacement must carry a successful
public `read` receipt before it may be routed into a Reasoner.  The failed
Agent, its typed diagnosis, its relations and all previously successful
artifacts remain preserved while this replacement is tested.

Static r16 integration passed the complete unit suite (`808 passed`, `144
subtests passed`) and both prepare-only freezes.  Its main live canary failed
the `2/2` Stable Zero gate.  `triviaqa:tc_3` explicitly finished with
`Heworth, North Riding of Yorkshire` and official EM/F1 `0.0/0.5`.
`triviaqa:tc_1` had already produced the correct
`<answer>Harry Sinclair Lewis</answer>` semantic chain, but a successful,
receipt-grounded Evidence Retriever remained terminal-unreachable.  The
preserved-input gate incorrectly removed the exact one-way
`Evidence Retriever -> Reasoner` relation from the live domain, so complete
graph reachability prevented `FINISH` and the task reached `max_rounds`.
Because the main gate failed, r16 did not run the isolated regression or
fixed-128 evaluation.

Recovery revision r17 changes only that measured relation-admission boundary.
It permits exactly one successful, terminal-unreachable Evidence Retriever
with a valid `qa-retrieval` read receipt to add one one-way predecessor edge to
the active Reasoner.  Reciprocal ingress, a non-Retriever source, a non-Reasoner
target and arbitrary successful-predecessor mutation remain rejected.  The
accepted relation uses the existing FlowSteer execute-on-edit transaction to
recompute Reasoner, Verifier and Formatter, while their old revision artifacts
are retained in the previous-revision preservation store.

Static r17 integration passed the complete unit suite (`809 passed`, `144
subtests passed`) and both prepare-only freezes.  The main live gate passed
`2/2`: `tc_1` explicitly finished at official EM/F1 `1.0/1.0`, and `tc_3`
explicitly finished at `0.0/0.5`.  The independent `tc_9`/`tc_10` gate also
passed `2/2`, both at official EM/F1 `1.0/1.0`.  All four checks retained full
turn, Output-inbox, terminal-artifact, environment-terminal and official
evaluator receipts, with zero collection failures.  Direct predictions were
reused from the frozen question-only source and no new Direct inference was
collected.  The r17 fixed-128 evaluation was admitted only after both gates and
is recorded under the same r17 condition; its final result must be read from
the completed manifest/report rather than inferred from these four gate tasks.

### r56 role-conditional inference architecture update

r56 changes only the shared HotpotQA/TriviaQA inference architecture and its
versioned TriviaQA configuration; it is not a training, RL or Skill result.
The shared QA Canvas no longer requires a named
`Reasoner -> Verifier -> Formatter` spine.  Those role families are optional
semantic capabilities, a generic `output` role is terminal-compatible, and
the Canvas validates the correlated `(execution_mode, allowed_tools)` profile
selected for each Agent.  All ordinary selected Agents must reach the unique
Output.  If a Verifier is selected, it must consume and preserve a routed
semantic candidate; if a Format Agent is selected, it is a copy-only terminal
sink.  Without those named roles, a generic Output must still preserve a
single consistent semantic candidate and have a successful `read` receipt in
its routed ancestry.

Evidence lineage is transitive across routed Agent messages.  A Reasoner
artifact may cite successful public `read` receipts produced by an upstream
Retriever as well as receipts from its own bounded ReAct execution, and those
receipts remain attached when the artifact is delivered to a Verifier or
Output.  TriviaQA location containment also requires the named geographic
scope in the same proposition-supporting clause that grounds subject,
relation and object; an unrelated clause in the same passage does not satisfy
the scope gate.

The live legal action domain and authoritative Canvas admission use the same
state projection.  When exactly one current Reasoner has no evidence ingress
and one or more executed Evidence Retrievers have a valid receipt-grounded
artifact, the next admissible action is `set_relation` over the exact
Canvas-validated Retriever-to-Reasoner candidates before an unrelated
semantic ADD.  This is a state-conditioned data dependency: it neither
chooses a fixed Retriever nor inserts a fixed role order, edge set, chain or
other topology template.  Reciprocal communication and multiple evidence
branches remain representable when legal in the current Canvas.

Recovery remains semantic-preserving and non-destructive:
`preserve -> diagnose -> repair/augment`.  Successful evidence, Tool receipts,
semantic artifacts, working relations and Output identity remain
revision-bound while the measured responsible Agent, execution profile, Tool
contract, entity/relation binding or relation is repaired.  Augmentation is
opened only by the typed live recovery state.  The otherwise strict
all-Agents-reach-Output rule has one narrow execute-on-edit exception: the
exact newly admitted, isolated Evidence Retriever or Repair unit may execute
before its validated artifact is routed in a later Canvas edit.  This does not
relax FINISH admission, authorize arbitrary disconnected Agents, or delete the
failed responsibility.  A last-valid evidence-lineage fallback, when one
exists at `max_rounds`, remains explicitly non-FINISH and GRPO-ineligible.

Director prompt receipt v5 keeps the concise, topology-neutral v4 system
instruction byte-for-byte and changes only the versioned historical-
observation policy.  It compacts stale historical Canvas observations while
retaining the initial task/catalog context, exact sampled actions, public
failure and Tool receipts, and the exact current Canvas/action domain.  The
prompt contains no fixed Agent count, role order, communication topology,
workflow template, retrieval-strategy recipe, candidate answer or unlisted
Skill.

The r56 configuration used the local Qwen3.5-9B Supervisor on GPU 0 at port
8015, explicitly set `require_format_agent: false`, and kept training, GRPO,
backward, optimizer updates, LoRA publication/policy synchronization, Skills,
MACE/Bayesian exploration and gradient-replica use disabled.  Its frozen
`triviaqa:tc_3` canary failed Stable Zero at official EM/F1 `0.0/0.0`, with
`final_answer=null`, `canvas_action_domain_exhausted`, 27 Director turns, 25
accepted edits, two rejected edits, eight Agents, one directed relation and no
Output Agent.  This is a terminal architecture failure, not a fixed-128 score.

The receipt trace establishes two separate facts.  First, the Retriever read
grounded `Dame Judi Dench -> born in -> Heworth`, the Reasoner later read that
Heworth is part of York, and upstream Tool receipts reached the Reasoner
without loss; retrieval and database coverage therefore succeeded.  Second,
the Reasoner never bound `York` to the answer slot, and later successful
recovery Retrievers remained isolated.  The live recovery relation projection
had required a strict reduction in `cannot_reach_output`, but a partial Canvas
with no Output has no such measurable set.  It consequently exposed repeated
Retriever augmentation until Agent capacity was exhausted.

The minimal post-r56 correction changes only that progress predicate.  Before
Output selection, an already materialized, receipt-valid recovery Retriever
may be routed one-way to the measured exhausted Reasoner through an ordinary
live `set_relation` candidate; after Output selection, the existing strict
terminal-reachability reduction remains mandatory.  No role, Agent identifier,
edge, order, topology or retrieval recipe is inserted into the Director
prompt.  A regression test reproduces the no-Output partial Canvas and proves
that the exact recovery edge, rather than another isolated ADD, is the sole
model-admissible action.

### v6 topology-neutral Director instruction

A model-visible prompt inspection showed that v5 did not prescribe a fixed
edge set or topology, but its system instruction still described the
`evidence_retriever`, `reasoner`, `verifier` and `format` responsibilities one
by one, while the Canvas observation repeated the same optional role inventory.
Those duplicate descriptions formed a canonical-role prior even though each
role was technically optional.

Prompt v6 removes that duplicate prior.  Following SkillFlow's compact action
guidance and FlowSteer's live Canvas boundary, the system instruction now states
only the legal action/target domains, model and Tool selection boundaries,
public execute-and-feedback cycle, question-scope preservation, non-destructive
recovery and terminal admission.  It is 1,387 characters rather than v5's
3,246 characters and contains no static per-role workflow instruction or
retrieval-strategy recipe.  Exact role families and correlated execution
profiles remain available only when they are legal in the current
`action_target_domains`; the search space itself is not reduced.  The general
Canvas observation no longer repeats `optional_role_capabilities`.

The v5 prompt and its compact-history policy remain versioned and replayable.
The fixed-128 condition is separately frozen to v6/Tool v43 under a new output
root, selects the same 128 development tasks in the same order, and keeps
training, GRPO, Skills and policy synchronization disabled.  The earlier
v6/Tool-v41 canary was stopped before any trajectory completed after the
terminal-admission audit below found that its optional-capability condition did
not implement the requested complete semantic lineage.  Its interruption
receipt remains isolated in the old artifact root and is not an evaluation
result.

### v43 required semantic lineage and evidence-before-reasoning gate

The shared QA Runtime already contained the strict evidence-to-answer lineage
used by the HotpotQA architecture.  Tool v43 reuses that implementation when
the existing `require_format_agent` configuration is true; it does not add a
separate TriviaQA workflow.  A terminal revision now requires an executed,
artifact-version-consistent `Evidence Retriever -> Reasoner -> Verifier ->
Formatter` responsibility lineage.  The Retriever completion must match the
original entity, requested relation, evidence span and passage ID to a
successful public read receipt.  The Reasoner owns the answer slot and semantic
candidate, the Verifier preserves that candidate while checking evidence and
scope, and the Formatter is a reasoning-free, Tool-free, copy-only Output sink.
A generic Output with only a successful read receipt can no longer FINISH.

This is a semantic data-dependency constraint, not a Director prompt template.
The v6 system instruction remains byte-for-byte topology-neutral and does not
list the four responsibilities.  Their exact current legality appears only in
the live `action_target_domains`; parallel Retrievers, evidence fan-in, repair
branches, reciprocal non-Formatter communication and multi-Agent executable
subgraphs remain in the search space.  The action-target-domain receipt is
versioned as `agentgraph.live-action-target-domains.v10` so the strict and
historical optional conditions cannot be mixed during replay.

Progressive execution now defers a QA Reasoner that has no direct Evidence
Retriever dependency.  Once such an edge exists, ordinary AgentGraph scheduling
runs the Retriever first, and the existing SkillFlow-derived completion gate
prevents the Reasoner from receiving an entity/relation artifact without its
matching read receipt.  Evidence Retriever execution is correspondingly limited
to bounded ReAct with `qa-retrieval`; reasoning-only, receipt-free Retriever
artifacts are not admitted.  The Canvas contract gate also rejects the two
measured concrete Tool-argument variants that escaped r56 (`limit of N` and an
explicit query symbol/operator), while continuing to admit neutral query-
rewriting and adaptive top-k responsibilities.

The first v42 diagnostic trajectory completed `triviaqa:tc_1` with official
EM/F1 `1.0/1.0` and an executed four-responsibility lineage.  It also exposed
an inconsistent contract gate: two question-faithful `American-born`
obligations were rejected after the qualifier moved beside a different head
noun, while a question-external `Nicolas Sinclair` entity precommit was
accepted.  The latter did not expose Ground Truth and the public Retriever
eventually corrected it through real search/read receipts, but the v42
condition was rejected and the remaining canary tasks were cancelled.

v43 changes only that pre-execution admission boundary.  A hyphenated
qualifier already present in the immutable question remains legal when a
contract paraphrases the responsibility.  A multi-token proper-name phrase
that introduces a token absent from the question is rejected as an entity
precommit, and an explicit `candidate answer '...'` literal is rejected even
when its text occurs in the question.  Neutral entity disambiguation, query
rewriting and adaptive top-k responsibilities remain legal; exact Tool
arguments remain owned by the state-conditioned StructuredAction schema.

The complete static suite passed `946` tests and `177` subtests.  Prepare-only
selected the same 128 task IDs, questions and Ground Truth values in the same
order (`triviaqa:tc_1` through `triviaqa:tc_223`) as the frozen comparison
condition.  No fixed-128 score is inferred from these static checks.

### v44 role-contract admission

The v43 canary was retained as a two-trajectory diagnostic and stopped before
the third task completed.  Both completed trajectories had valid Tool receipts,
complete evidence lineage, explicit FINISH and no evaluator leakage:
`triviaqa:tc_1` scored EM/F1 `1.0/1.0`; `triviaqa:tc_5` returned the
semantically valid decade `1930s`, which the official accepted-answer aliases
(`30s` variants) scored `0.0/0.0` and classified as
`accepted_answer_canonicalization_mismatch`.  The diagnostic also exposed a
Canvas/Runtime inconsistency in both trajectories: a Reasoner could retain its
role-specific Runtime schema while accepting the bare contract of another
responsibility (`format` or `retrieval`).  The resulting artifacts happened to
be valid, but the condition was not admitted to the fixed-128 run.

v44 adds only a role-contract admission guard and state-conditioned contract
responsibility metadata.  A bare contract that names another QA responsibility
is rejected transactionally, while the contract remains model-authored and
question-specific.  The live domain describes each selected role's semantic
responsibility; it does not provide a contract template or prescribe Agent
count, order, edges, topology, communication pattern, or retrieval recipe.  The
Director v6 system prompt is unchanged.  This live-domain shape change is why
the action-target-domain receipt is now
`agentgraph.live-action-target-domains.v9`.

### v45 exact relation-domain consistency

The v44 canary was retained as an operational diagnostic and stopped after
`triviaqa:tc_3` failed collection with
`set_relation exact live candidates are missing`; no v44 benchmark score is
reported.  The failure occurred when the strict semantic lineage still lacked
the Formatter: the action-type projection correctly prioritized the executed
Retriever-to-Reasoner evidence ingress, but the relation-target projection
returned an empty list merely because another semantic responsibility was
still missing.  The v3 two-phase StructuredAction boundary therefore exposed
`set_relation` and then had no exact candidate to serialize.

v45 makes both projections reuse the same receipt-grounded evidence-ingress
candidates.  The invariant is now `set_relation in admissible actions` implies
at least one exact live relation candidate, and the existing Director validator
continues to fail closed before generation if that invariant is violated.  The
regression uses a strict TriviaQA partial Canvas with a valid Retriever read
receipt, an unconnected Reasoner, and a still-missing Formatter; it validates
the action mask, exact target domain and v3 schema boundary from one snapshot.
This does not add an edge template: any valid Retriever may provide ingress,
reciprocal non-Formatter communication remains legal, and the remaining Agent
count, roles, edges and topology remain Director choices.  The target-domain
receipt is `agentgraph.live-action-target-domains.v10`; Director prompt v6 is
unchanged.

### v46 receipt-preserving structured recovery and retrieval observability

The v45 three-task canary was stopped after `triviaqa:tc_1` produced a real
collection failure: the schema-bound `add_subgraph` role-selection phase
returned non-JSON text.  That phase had not reused the one-shot bounded
serialization recovery already present at the hierarchical parameter boundary,
so the whole task terminated before a rejected Canvas turn could be recorded.
No v45 benchmark score is reported.

v46 keeps the exact sampled text, live JSON Schema, route and scientific seed,
then permits exactly one schema-bound continuation request.  A valid repair
continues the same `add_subgraph` transaction.  If the repair is still invalid,
both exact phase receipts become a typed rejected Canvas turn at graph revision
zero; no Agent is declared or executed and no third request is made.  Semantic
violations such as trailing text or an out-of-domain role remain fail-closed and
are not rewritten with regex or defaults.

The Tool adapter also publishes `strategy_proofs` for the existing bounded
retrieval schedule.  Each entry distinguishes a question-invariant strategy
attempt from an attempt conditioned on prior public Tool receipts and records
the associated passage IDs.  This metadata does not shrink the legal query
action domain and does not call a search hit evidence-grounded.  The unchanged
pre-Reasoner gate still requires exact entity identity, target relation,
evidence span, passage ID and a matching successful read receipt.  Consequently
spelling, alias, disambiguation and relation-rewrite recovery remain executable,
while `knowledge_base_coverage_failure` remains unavailable unless the complete
bounded semantic schedule is valid and no aligned evidence is obtained.

The Director v6 prompt is unchanged and remains topology-neutral.  Static
verification passes 955 tests and 177 subtests; the fixed 128 task selection is
unchanged.  These checks are not a v46 accuracy result.

### v47 receipt-preserving structured-artifact recovery

The v46 canary was stopped after its first task. `triviaqa:tc_1` obtained the
same successful search and read receipt as the earlier v43 successful
trajectory, but its Evidence Retriever exhausted the bounded ReAct loop while
constructing the structured evidence artifact. The repair instruction
simultaneously required the old evidence span to remain unchanged and required
an entity surface absent from that span. After one same-Agent repair, the live
Canvas domain then exposed the still-missing downstream roles instead of an
executable evidence repair, causing the same failed Retriever to run six times
and grow from 20 to 120 ReAct turns. No downstream artifact or valid lineage
was produced, so the final `canvas_action_domain_exhausted` and null fallback
were correct consequences of an earlier recovery-domain defect, not a
retrieval or database-coverage failure.

v47 keeps the entity, relation, passage ID, successful-read receipt and exact
contiguous-span gates unchanged. Its public repair instruction permits only an
evidence-span expansion inside the same read receipt so the exact entity
mention and requested-relation sentence can coexist. A failed TriviaQA
Retriever may modify only the state-conditioned evidence recovery contract or
completion predicate; a generic answer-producing completion predicate is not
admissible. If that bounded same-Agent repair still ends in
`schema_invalid / qa_semantic_artifact_invalid` while retaining a successful
read and an unexhausted public Tool continuation, Canvas admits exactly one
isolated same-role/same-artifact Retriever replacement. Existing SkillFlow
Action--Observation receipts are handed to that unit; downstream semantic
roles remain unavailable until a valid evidence artifact is produced. A
second replacement generation on unchanged receipts is not admitted.

This is a state-conditioned recovery boundary in the shared AgentGraph, not a
Director prompt recipe or fixed graph topology. Director prompt v6 is
unchanged. Static verification passes 956 tests and 177 subtests. These checks
are not a v47 accuracy result; Stable Zero and the fixed-128 evaluation remain
live gates.

### v48 progressive Canvas admission and answer-bearing entity surface

The v47 three-task canary was stopped after two persisted trajectories exposed
two deterministic integration defects; the interrupted third task is not
scored. `triviaqa:tc_1` formed a complete receipt-valid lineage and explicitly
FINISHed with `Lewis`, producing official EM `0` and F1 `0.6666667` against
the accepted full-name surface. The Tool had already read a passage containing
`Sinclair Lewis`, but the accepted Retriever artifact selected the shorter
surname from a second read. Reasoner then copied that proposition argument,
Verifier accepted it, and Formatter correctly copied it unchanged. This is an
answer-bearing Entity Linking/canonical-surface admission defect, not a
Formatter or evaluator defect.

`triviaqa:tc_5` never reached retrieval. Its first accepted Canvas revision
contained `Reasoner -> Verifier` and an isolated Formatter while the Evidence
Retriever responsibility was still absent. The live action mask and exact JSON
Schema exposed only `add_subgraph(evidence_retriever)`, but authoritative
admission rejected that exact action and demanded the unavailable
`set_relation(Verifier -> Formatter)`. The Director consequently produced the
same legal-domain ADD for 27 independently seeded turns and reached
`max_rounds`. All structured-generation receipts were valid; this was an
action-domain/admission inconsistency, not Director noncompliance.

v48 keeps FlowSteer's accepted Canvas edit -> execute -> feedback boundary and
changes only that inconsistent admission predicate. When the live mask exposes
an exact missing-responsibility functional subgraph and no already-valid
Retriever ingress must be routed first, the identical ADD is transactionally
admitted. Its execution feedback becomes the next observation; the following
revision then exposes the exact remaining semantic relation and Output action.
Existing mandatory repair, replacement, evidence-ingress, illegal relation and
contract-validation priorities remain unchanged. This is state-conditioned
progressive construction, not a fixed serial topology.

The factual Retriever gate now also uses the question-only wh-dependency and
the same successful read receipt's passage-title identity. If a named entity
surface occupies the answer-bearing proposition field and is a strict token
subset of a multi-token resolved title identity, it is rejected as ambiguous.
A complete contiguous body mention is required before Reasoner answer-slot
binding. Pronoun coreference and short-name aliases used only as non-answer
entity anchors remain admissible. Public recovery preserves all successful
read receipts and first reuses a receipt containing the complete mention;
bounded retrieval remains available only when no preserved receipt supplies
one. No accepted answer or evaluator state enters this gate.

Director prompt v6, action vocabulary, topology search space and official
evaluator are unchanged. The complete static suite passes 958 tests and 177
subtests. These checks are not a v48 score; the new three-task Stable Zero gate
and fixed-128 evaluation remain pending.

### v49 rejected expanded-alias preflight candidate

The v48 canary was stopped after its first persisted task; the remaining two
tasks were not scored. `triviaqa:tc_1` searched once and read the correct
receipt-backed passage on the second ReAct turn. Its first structured artifact
contained the complete body surface `Harry Sinclair Lewis` and the requested
1930 Nobel relation, but the validator rejected that full name because the
passage title was the shorter canonical surface `Sinclair Lewis`. This was an
Entity Linking false negative, not retrieval recall or database coverage.

After the first Retriever and its isolated replacement exhausted their
bounded artifact repairs, the replacement was already declared with the only
admissible recovery contract and completion condition. Nevertheless, rounds
3--27 exposed MODIFY with one discrete value equal to the current value. The
Director emitted 25 legal-domain choices, all rejected as no-op, and no Agent
executed after graph revision 3. The resulting `max_rounds` terminal therefore
came from an action-mask/state-transition mismatch.

The first v49 preflight candidate attempted to accept a receipt-body full-name
expansion from an article-lead title/name/parenthetical pattern. Read-only
adversarial review showed that the same positional pattern could admit a
Title Case descriptor such as `American Writer Ada Lovelace`. That candidate
was therefore rejected before commit and before any canary or fixed-128 run.
Its prepared directory is not an accepted architecture or score.

### v50 canonical title identity and non-no-op recovery admission

v50 keeps `entity_identity.evidence_surface` on the complete public passage-
title identity. A longer receipt-grounded body name remains unchanged in the
answer-bearing `evidence_proposition` argument; the Reasoner, rather than the
Retriever identity field, owns answer-slot binding. Thus a passage title such
as `Sinclair Lewis` can bind the exact body proposition subject
`Harry Sinclair Lewis` without admitting a surname subset or a descriptor as
an entity identity. The structured repair instruction preserves the same read,
span and proposition, and changes only `entity_identity` to that complete title
surface. No accepted answer or evaluator state enters this rule.

The recovery state now marks a TriviaQA Evidence Retriever as
`repair_exhausted` when a bounded ReAct failure occurs after both unique
recovery declaration values are already active. The live MODIFY projection
publishes only values different from the current Canvas state. If no valid
evidence artifact and no strict-progress same-responsibility replacement are
available, missing Reasoner/Verifier/Formatter roles cannot bypass Evidence
Grounding. This fail-closed boundary is TriviaQA-conditioned; HotpotQA's
previously verified missing-role completion behavior is unchanged.

Director prompt v6 remains short, topology-neutral and unchanged. No fixed
role order, chain, Agent count, candidate answer or workflow template is added
to it. The complete static suite passes 960 tests and 177 subtests. Stable Zero
and fixed-128 are still live gates, so these static results are not an accuracy
claim.

### Stable Zero status

Static architecture, 946 unit tests, 177 subtests and the current frozen
data-selection preconditions are complete.  The initial, r2, r3 and r4 canaries failed for
the documented causes; r5 **passed Stable Zero** but its incomplete fixed-128
run was stopped after measured recovery defects appeared.  Recovery revision
r6 failed Stable Zero for the documented retrieval-recovery defect.  Recovery
revision r7 fixed that defect but failed Stable Zero for the documented
semantic-verification/repair-attribution defect.  Recovery revision r8 is
answer-lineage complete for `tc_3` but failed Stable Zero because one malformed
Director parameter sample incorrectly aborted `tc_1` collection.  Recovery
revision r9 passed the main two-task canary but failed the mandatory isolated
`tc_9/tc_10` regression for the documented retrieval-recovery defects.
Recovery revision r10 failed its main canary `1/2` for the documented
role-bound structured-repair defects, so its isolated regression and fixed-128
run were not started.  Recovery revision r11 passed 782 unit tests, 142
subtests and both frozen prepare-only checks, then failed its live main canary
`1/2`: `tc_1` returned `Harry Sinclair Lewis` at EM/F1 `1.0/1.0`, while
`tc_3` ended at `max_rounds` with a null answer.  Its isolated `tc_9/tc_10`
regression and fixed-128 run were therefore not started.  Recovery revision
r12 passed its live main Stable Zero canary `2/2` (`tc_1` EM/F1 `1.0/1.0`;
`tc_3` EM/F1 `0.0/0.5`, explicit FINISH), then failed the isolated
`tc_9/tc_10` regression `0/2` with both tasks at `max_rounds` and null answers.
The r12 fixed-128 run was not started.  Recovery revision r13 passed its live
main canary `2/2`; its isolated regression produced `1/2`, with `tc_10`
explicitly finishing at EM/F1 `1.0/1.0` and `tc_9` recorded as
`collection_failed` after incomplete declaration JSON.  The r13 fixed-128 run
was not started.  Recovery revision r14 failed its live main canary `1/2`:
`tc_1` explicitly finished at EM/F1 `1.0/1.0`, while `tc_3` ended at
`max_rounds` with a null answer despite a correct public read receipt.  Its
isolated and fixed-128 runs were not started.  Recovery revision r15 passed
only its static and prepare-only gates.  Recovery revision r16 then failed its
main canary `1/2` because the exact grounded Retriever-to-Reasoner ingress was
missing from the live relation domain.  Recovery revision r17 passed both the
main `2/2` and isolated `tc_9`/`tc_10` `2/2` live gates, so its fixed-128 run
was admitted.  No fixed-128 score is inferred from unit tests, four canary
tasks, an incomplete run, or the previous architecture.  r56 subsequently
introduced the role-conditional, topology-neutral inference boundaries above,
but its `triviaqa:tc_3` live canary failed with the measured partial-Canvas
recovery-domain defect documented above.  v43 retained two valid diagnostic
trajectories, v44 exposed an empty exact relation domain, and v45 exposed the
unhandled role-selection serialization failure. v46 failed its first Stable
Zero task after a successful Tool read because its structured-artifact repair
path exposed blocked downstream roles and ended in
`canvas_action_domain_exhausted`. v47 then failed its live gate for the measured
action-domain/admission and answer-bearing entity-surface defects documented
above. v48 then failed its first persisted Stable Zero task after a correct
Tool read because full-name title expansion was rejected and the recovery
domain exposed 25 no-op MODIFY actions; the remaining two canary tasks and the
fixed-128 run were not started. The expanded-alias v49 preflight candidate was
rejected before live execution because it could admit Title Case descriptors.
v50 is statically verified, while its Stable Zero canary and fixed-128 run
remain pending. No v2 score is inferred from these static results or failed
diagnostics.

## Historical comparison condition

The frozen v6.2 TriviaQA development run on the same ordered 128 tasks reported
Direct EM `35.15625%`, Direct F1 `40.81597%`, AgentGraph EM `51.5625%`,
AgentGraph F1 `60.104405%`, 116 explicit FINISH trajectories and 12 terminal
`max_rounds` failures.  These values are the pre-v2 comparison condition, not
v2 results.
