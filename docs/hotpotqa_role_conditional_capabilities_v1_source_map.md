# HotpotQA role-conditional capabilities v1 source map

`hotpotqa_role_conditional_capabilities_v1` is an inference-only HotpotQA
condition derived from the existing semantic-lineage r4 evaluation boundary.
It preserves the same frozen 128 development tasks, evaluator, seed,
concurrency, Qwen3.5-9B Director, Direct comparator, local Executor catalog,
provided-context retrieval runtime, and concise
`agentgraph.director.minimal-neutral.v10` prompt. It does not resume, overwrite,
or merge the r4 trajectories; r4 metrics are comparison references only.

The current condition is configured in
`config/evaluation_hotpotqa_role_conditional_v1_r16.yaml`. Its artifacts and
reports use the independent `hotpotqa_role_conditional_v1_r16` namespace.

## Directly reused from FlowSteer

- Progressive Canvas editing and the edit--execute--feedback transaction:
  `src/interactive/workflow_env.py::step`, `_step_internal`, and
  `_execute_workflow`. Each accepted Canvas edit is executed before the next
  Director observation.
- AgentGraph topology and inter-Agent communication:
  `src/interactive/workflow_graph.py::WorkflowGraph`, including directed and
  reciprocal relations and the bounded reciprocal block.
- The existing Canvas action vocabulary and Runtime boundary: add or modify a
  functional subgraph, set relations or Output, execute the current graph,
  observe feedback, and continue or finish.
- Output serialization guidance from
  `scripts/prompts/prompt.py::FORMAT_PROMPT`, used only when the Director
  actually selects a Formatter capability for Output.

## Directly reused from SkillFlow

- Bounded Agent execution through
  `src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` and
  `_validate_completion`.
- ReAct tool interaction through the established
  `Thought -> Action(tool) -> Observation -> Thought -> Final` execution
  schedule. ReAct is an Agent execution mode, not an Agent role.
- Provided-context retrieval and Tool receipts through
  `src/skillev/benchmarks/retrieval.py::QARetrievalEnvironment.execute` and
  `_outcome_for_action`. Successful `qa-retrieval` read receipts remain the
  evidence-provenance authority.

## Necessary project adaptation

- `hotpotqa_role_conditional_capabilities_v1` exposes Reasoner, Verifier, and
  Formatter as optional semantic capabilities in the Director search space.
  They are neither mandatory roles nor a fixed
  `Reasoner -> Verifier -> Formatter` execution path. The Director may select
  none, one, or multiple instances when justified by the current Canvas state
  and feedback.
- The action mask and live action-target domains do not complete missing
  Reasoner, Verifier, or Formatter roles, do not defer Output until those roles
  exist, and do not require a serial ancestor relation among them. Directed,
  reciprocal, fan-in, fan-out, auxiliary retrieval, repair, and direct Output
  topologies remain in the bounded AgentGraph search space.
- Role and execution mode are orthogonal. A Reasoner may use Tool-free
  reasoning over routed evidence or use ReAct with `qa-retrieval`; defining a
  role named ReAct is not permitted by this protocol.
- FINISH validates the graph and artifacts that the Director actually chose.
  A selected Reasoner is checked for answer-slot and evidence-provenance
  consistency; a selected Verifier is checked for evidence grounding,
  entity--attribute binding, multi-hop completeness, and question-scope
  preservation; a selected Formatter is restricted to exact serialization of
  an already selected semantic candidate. Roles that are absent are not
  synthesized and are not terminal prerequisites.
- The terminal boundary still requires routed `qa-retrieval` evidence and the
  exact single-answer output syntax. This is an evidence and serialization
  requirement, not a mandatory role-template requirement.
- Recovery reuses `preserve_diagnose_repair_augment`: valid evidence,
  artifacts, dependencies, and Output identity are preserved; repair or
  augmentation precedes deletion, and deletion is admissible only after a
  replacement has taken over the required artifact.

## Configuration delta from semantic-lineage r4

The new YAML is a mechanical copy of r4 with only the independent experiment,
condition, catalog-order, output/report, and policy-sync namespaces changed,
plus the semantic protocol value
`hotpotqa_role_conditional_capabilities_v1`. All frozen sampling and evaluation
fields remain unchanged. The old
`artifacts/hotpotqa_semantic_lineage_v2_r4/hotpotqa` namespace is preserved.

The first independent canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r1/hotpotqa`. It measured two Canvas
continuation deadlocks rather than evaluator failures: an unfinished Output
artifact could not be handed off atomically to a newly routed sink, and a
selected semantic consumer with no routed input was deferred by the Runtime
without an executable ingress repair in the next live action domain. The r2
condition in `config/evaluation_hotpotqa_role_conditional_v1_r2.yaml` applies
only those measured boundary repairs: incomplete Output handoff now requires
an explicit Output assignment in the same `ADD_SUBGRAPH` transaction, and a
deferred consumer projects either a legal `SET_RELATION` from an already
materialized producer or an `ADD_SUBGRAPH` containing a producer-to-consumer
relation. Neither repair requires a Reasoner, Verifier, Formatter, or serial
role order.

The failed r2 canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r2/hotpotqa`. It exposed two additional
execution-boundary faults. First, the selected Retriever inherited a legacy
seven-field semantic completion contract even though SkillFlow retrieval owns
only search/read observations and the Reasoner owns predicate--argument and
answer-slot alignment. In r3, a role-conditional Retriever therefore exports
only receipt-grounded `passage_id`/`evidence_span` citations; legacy protocol
contracts are unchanged. Second, an exhausted auxiliary replacement received
contradictory live domains for isolated execution and Output assignment. In r3,
that measured recovery state admits exactly one isolated same-role,
same-artifact `ADD_SUBGRAPH`; relation and Output edits become available only
after the replacement artifact materializes. This bounded recovery rule does
not constrain normal-state role choice, role multiplicity, directed or
reciprocal topology, or FINISH. The new independent evaluation namespace is
configured in
`config/evaluation_hotpotqa_role_conditional_v1_r3.yaml`.

The failed r3 canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r3/hotpotqa`. Both canary tasks
materialized the correct terminal answer, but `FINISH` remained inadmissible
because the selected semantic consumer had no successful `qa-retrieval` read
receipt on its own directed ancestor path. The generic preservation gate then
rejected a Retriever ingress edit because it would change a successful
consumer's input dependency, so both trajectories ended at `max_rounds`.

The independent r4 condition in
`config/evaluation_hotpotqa_role_conditional_v1_r4.yaml` closes only that
measured execution boundary. When the actual routed semantic consumer lacks
Tool evidence, the live FlowSteer action domain first reuses an existing
receipt-bearing artifact through one `SET_RELATION` edit. If none exists and
capacity remains, it admits one canonical
`ADD_SUBGRAPH(Evidence Retriever -> selected semantic consumer)` transaction.
Both paths preserve the existing Output identity and unrelated successful
dependencies, then reuse the Runtime dirty-closure execution to rerun that
consumer and its downstream nodes. The repair target is inferred from actual
routed semantic artifacts and receipts: it may be a selected Reasoner, a
generic Output capability, or another semantic producer. An already declared
reverse edge may be completed into FlowSteer's bounded reciprocal block; no
serial topology is required. A selected Verifier or Formatter must consume an
explicit routed semantic-candidate artifact, but that producer may be a
Reasoner, another Verifier, or a repair capability, so neither role is a global
prerequisite and no three-role spine is imposed. r4 also limits newly selected
Output ownership to terminal-compatible generic Output or optional
formatting-only capabilities, and requires a generic Output to preserve any
routed semantic-candidate consensus character-for-character. These are local
action-admission and terminal-artifact contracts, not a workflow template.

The failed r4 canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r4/hotpotqa`. One trajectory already
materialized the correct evidence-grounded `<answer>Delhi</answer>` artifact,
but the canonical question classifier's broad `location` type was compared by
raw string equality with the Reasoner's explicit `city` subtype. The Director
therefore exhausted its edit budget while attempting unrelated metadata
changes. The other canary failed receipt validation because a later
`ADD_SUBGRAPH` was made to repeat the already assigned Output even though the
Canvas mutation semantics preserve Output ownership when that optional field
is omitted.

The independent r5 condition in
`config/evaluation_hotpotqa_role_conditional_v1_r5.yaml` changes only these
measured boundaries. It reuses the question-only classifier and admits an
explicit interrogative head as a lexical subtype of its canonical answer type
(for example, `city` under `location` for an unchanged `what city` question,
or `magazine` under `entity` for an unchanged `which magazine` question).
This compatibility check reads neither passages nor candidate answers,
Ground Truth, or evaluator state. A `where` question has no narrower lexical
head and continues to require `location`. r5 also preserves the current Output
across later `ADD_SUBGRAPH` edits; Output handoff remains an explicit
`SET_OUTPUT` operation. Finally, a malformed final parameter remains an exact
behavior receipt and reaches FlowSteer's invalid-action continuation instead
of being reclassified as a missing Output assignment. None of these changes
requires a role, role order, or serial topology.

The failed r5 canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r5/hotpotqa`. Both trajectories kept
the correct terminal answer artifact (`Arthur's Magazine` and `Delhi`) but
exhausted 28 rounds because a reasoning-mode Reasoner used the unconstrained
text completion path. Its artifacts were semantically correct but repeatedly
violated the already defined wire contract through values such as numeric
cardinality, alternate proposition keys, and object-valued reasoning steps.

The independent r6 condition in
`config/evaluation_hotpotqa_role_conditional_v1_r6.yaml` directly reuses
SkillFlow's request-scoped strict response-schema boundary from
`skillev/runtime/openai_provider.py` and the existing
`QARetrievalReactExecutionAdapter` Reasoner completion schema. A QA-specific
reasoning execution adapter projects that existing `arguments.value` schema
as the reasoning Agent's top-level response schema. It activates only when the
Canvas actually selected a reasoning-mode Reasoner under a verified QA
semantic protocol; every other role, execution mode, dataset runtime, and
topology keeps its prior path. This is a necessary execution-boundary adapter,
not a mandatory Reasoner, Verifier, Formatter, or serial workflow template.

The independent r7 condition in
`config/evaluation_hotpotqa_role_conditional_v1_r7.yaml` keeps every frozen
evaluation field from r6 and projects only execution profiles registered by
the current Runtime into FlowSteer's live Canvas action domain. It reuses
SkillFlow's bounded reasoning/ReAct execution and task-scoped Tool registry:
reasoning is Tool-free, ReAct is Tool-free or uses the registered
`qa-retrieval` resource, and unregistered coding/Tool combinations are absent.
Formatter remains a serialization responsibility under either Tool-free
reasoning or Tool-free ReAct. FlowSteer's atomic `ADD_SUBGRAPH` mutation is
also reused to route a recovery augmentation into an existing generic Output
in the same execute-on-edit unit. Verifier diagnostic status remains separate
from the preserved semantic candidate; FINISH validation still rejects an
unsupported candidate and attributes the routed Reasoner/Retriever for repair.
None of these boundaries requires a role to exist, fixes role order, or fixes
a serial topology.

The r7 canary exposed one deployment incompatibility before its first missing
task could produce a Canvas turn: SkillFlow's scientific sampling protocol
derives a full uint64 seed, while SGLang 0.5.15 deterministic inference first
materializes the native `sampling_seed` as `torch.int64` and subsequently
casts it to `torch.uint64` in its sampler. The independent r8 condition keeps
the original SkillFlow uint64 seed in the behavior receipt and sends the
bit-equivalent signed two's-complement representation only at the native
SGLang request boundary. This is a wire-type compatibility adapter; sampling
coordinates, 64-bit random state, policy, search space, topology, prompts,
tasks, and evaluator are unchanged.

The independent r8 canary then completed one task but exposed a Canvas/Runtime
admission mismatch on the other. The Director selected a Formatter without
assigning it as Output in the same `ADD_SUBGRAPH`; SkillFlow correctly deferred
that terminal serializer, while another generic Output had already produced
the correct semantic answer. Subsequent action targets focused on the deferred
Formatter and exhausted the edit budget without `SET_OUTPUT`. In r9, Formatter
remains absent from every required-role or FINISH prerequisite. If the Director
does select it, however, the role-selection and parameter schemas allow at
most one and require it to be assigned as Output in the same atomic FlowSteer
Canvas edit. A generic Output path remains valid with no Formatter, Reasoner,
or Verifier. This aligns the optional capability's action mask with the
existing Runtime terminal-serializer contract without prescribing a serial
topology.

The r9 source review found three residual role-ownership assumptions even
though Canvas admission and FINISH no longer required a fixed role sequence.
The role-conditional ReAct Verifier and Evidence Retriever prompts still
referred specifically to a Reasoner, and an unsupported Verifier verdict
preferred Reasoner-role attribution instead of the actual routed semantic
producer. The independent r10 condition removes only those assumptions. The
Verifier consumes a routed semantic candidate, the Retriever hands evidence
to a downstream semantic producer, and recovery attribution follows the
actual routed artifact provenance. A malformed Verifier artifact remains
attributed to the Verifier itself. Legacy protocols are unchanged. This
preserves optional Reasoner, Verifier, and Formatter capabilities, generic
Output, arbitrary admissible directed or reciprocal topology, and the short
neutral Director prompt.

The r10 canary materialized a correct semantic answer and a correct Output,
then a later dirty-closure execution received an SGLang response with
`finish_reason=abort`. The incomplete content was incorrectly published as a
successful Agent artifact and displaced the current-revision semantic
artifact, preventing FINISH. The independent r11 condition fixes only the
provider normalization boundary. It reuses SkillFlow's legal generation
boundary vocabulary (`json-root`, `length`, `stop`), retries `abort` within the
existing bounded retry loop, and never publishes partial content from an
aborted response. Retry exhaustion becomes the existing Runtime failure path;
the previous-revision preservation record remains intact but is not silently
restored into a graph revision whose dependencies changed. This transport
repair is role- and topology-independent.

The first r11 diagnostic was launched with the new config path but imported
the main worktree's older `src.interactive` package. Its empty-action failure
is therefore not evidence about the r11 source tree. The independent r12
condition makes the shared dataset root explicit while requiring the committed
condition worktree to be the Python source root. The run records remain in a
separate namespace and are not reused as r13 results.

The r12 source review found that Agent execution prompts already required an
unexpected-equal comparison to recheck question scope, entity--attribute
binding, explicit evidence, and upstream contract narrowing, while the
role-conditional Director observation omitted that existing capability. The
independent r13 condition exposes the same four checks to the Director's open
search space. It does not require a Reasoner, Verifier, Formatter, fixed edge,
or serial ordering; the check applies only when the Director selects a
semantic reasoning capability for a comparison task.

The r13 canary remains preserved in
`artifacts/hotpotqa_role_conditional_v1_r13/hotpotqa`. The Delhi task completed
through a selected Reasoner and generic Output without either Verifier or
Formatter. The comparison task materialized the correct semantic candidate,
generic Output artifact, and an independently successful Verifier branch in
its first Canvas revision, but no Output had been assigned. The preservation
gate then removed every relation that could converge the parallel branch into
the existing Output, while the complete-Canvas gate removed every Output
target until that convergence existed. The Director therefore received only
unrelated add/modify actions and reached `max_rounds` with its correct
artifacts still preserved.

The independent r14 condition maps FlowSteer's downstream `Aggregate` boundary
onto the existing AgentGraph relation vocabulary. When successful parallel
branches are the only reason an existing generic Output cannot be selected,
the live action mask admits only a directed, monotonic
`SET_RELATION(branch, Output)` that strictly reduces the current
`cannot_reach_output` set. Each accepted edge is an ordinary FlowSteer
edit--execute--feedback turn and reruns the affected Output through the
existing Runtime dirty closure. Multiple branches converge one edge per turn;
only after all current Agents reach the Output does the next live domain expose
`SET_OUTPUT`. Existing edges, source artifacts, Tool receipts, and failure
observations are preserved. A pure Formatter is never used as a multi-branch
aggregator. This is a necessary AgentGraph adaptation of FlowSteer's existing
fan-in boundary, not an automatic topology, role template, answer selection,
or FINISH operation.

The r14 formal run remains preserved as an incomplete six-task diagnostic; it
is not a 128-task metric. Two tasks reached `max_rounds`. In one, a
receipt-grounded semantic artifact had an independent route to the selected
Output while a failed upstream dependency still blocked that Output. In the
other, one member of a reciprocal block produced a useful public artifact
before its peer failed, but the block-level exception discarded that
artifact. These are measured Runtime/Canvas recovery faults, not evidence for
adding a mandatory semantic-role sequence.

The independent r15 condition repairs only those measured boundaries. It
continues to reuse FlowSteer's edit--execute--feedback transaction
(`src/interactive/workflow_env.py::_step_internal` and `_execute_workflow`),
its progressive execution cache
(`src/vllm_workflow_generator.py::_execute_workflow`), and its bounded
parallel/Aggregate execution. It also continues to reuse SkillFlow's public
bounded Action--Observation and failure receipts from
`src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` and
`src/skillev/rollout/engine.py`. AgentRuntime now retains a successful peer's
public artifact and Tool receipts as a non-terminal partial result when the
other reciprocal peer fails; that partial result remains dirty, cannot enter
the execution cache, and cannot satisfy FINISH. When a revision-live,
receipt-grounded artifact already reaches the selected Output independently,
the next Canvas domain may admit one exact directed
`SET_RELATION(source, failed ancestor)` repair before rerunning the affected
dirty closure. This one-edge projection and reciprocal partial retention are
necessary AgentGraph adapters because neither upstream implementation exposes
this project's selected-Output failure state. They do not select an answer,
execute FINISH, delete an edge, reverse an edge, or prescribe a topology.

Deletion remains the final recovery operation. A node must be explicitly
diagnosed unusable, and a same-role/same-artifact replacement must already own
all downstream responsibilities while preserving any previous-revision
semantic candidate and successful Tool evidence (or its explicit public
continuation handoff). If no configured action can perform a mandatory repair,
the Director action mask exposes an empty domain instead of advertising an
edit that the preservation gate must reject. Reasoner, Verifier, Formatter,
generic Output, retrieval, and repair remain optional capabilities; r15 adds
no fixed role set, role ordering, ancestor chain, or FINISH prerequisite.

The r15 Stable Zero canary completed, but the formal run was stopped after two
completed trajectories and one collection failure; it is not a 128-task
metric. The third frozen task reached the configured eight-Agent limit with a
failed, repair-exhausted auxiliary. Preservation correctly prohibited deleting
that node without a replacement takeover, while augmentation could not add a
replacement at capacity. The resulting empty live action domain could not be
encoded by the Director's constrained JSON schema and raised before a policy
request. The r15 artifacts remain in their independent namespace as diagnostic
evidence only.

The independent r16 condition keeps all r15 sampling and evaluation fields and
repairs only that measured controller boundary. At the Agent limit, after any
existing replacement, failed-ingress relation repair, and routed auxiliary
repair have been exhausted, a failed and repair-exhausted auxiliary that has
not been diagnosed unusable becomes the exact `MODIFY_AGENT` target. This
reopens its existing contract/completion condition while preserving the node,
relations, previous semantic artifact, and Tool continuation. If any other
state still has no admissible action, `AgentGraphOrchestrator` performs no
Director request and records `no_admissible_action` as a non-explicit terminal
failure instead of raising a collection exception. FlowSteer's upstream action
mask is a training-token mask rather than a dynamic legal-Canvas mask, and
SkillFlow distinguishes horizon exhaustion from an empty model submission, so
this is a necessary AgentGraph constrained-decoding adapter. It does not emit
an action, auto-FINISH, select an answer, or impose any role or topology.

## Not implemented or enabled in this condition

- Skill retrieval, Skill injection, SkillFlow training, or an ACTIVE Skill
  library.
- MACE credit assignment or Bayesian posterior updates.
- GRPO/LoRA training, backward, optimizer steps, adapter synchronization, or
  weight publication.
- Benchmark-answer, Ground Truth, evidence-span, or sample-specific
  hard-coding.

This source map records implementation provenance only. No canary or 128-task
evaluation result is claimed until that independent condition is actually run.
