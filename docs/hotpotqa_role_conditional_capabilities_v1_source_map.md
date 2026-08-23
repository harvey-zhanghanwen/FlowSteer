# HotpotQA role-conditional capabilities v1 source map

`hotpotqa_role_conditional_capabilities_v1` is an inference-only HotpotQA
condition derived from the existing semantic-lineage r4 evaluation boundary.
It preserves the same frozen 128 development tasks, evaluator, seed,
concurrency, Qwen3.5-9B Director, Direct comparator, local Executor catalog,
provided-context retrieval runtime, and concise
`agentgraph.director.minimal-neutral.v10` prompt. It does not resume, overwrite,
or merge the r4 trajectories; r4 metrics are comparison references only.

The condition is configured in
`config/evaluation_hotpotqa_role_conditional_v1_r1.yaml`. Its artifacts and
reports use the independent `hotpotqa_role_conditional_v1_r1` namespace.

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
