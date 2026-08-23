# HotpotQA semantic-lineage v2 source map

This condition is an inference-only adaptation of the existing
`unified_architecture_v1`. It does not add training, GRPO, optimizer updates,
LoRA publication, or an active Skill library.

## Directly reused upstream boundaries

- FlowSteer progressive Canvas transaction:
  `src/interactive/workflow_env.py::step`, `_step_internal`, and
  `_execute_workflow` from the supplied FlowSteer source. One accepted Canvas
  edit is executed before the next Director observation.
- FlowSteer graph and communication encoding:
  `src/interactive/workflow_graph.py::WorkflowGraph`. The project keeps the
  two-bit directed/reciprocal relation representation and the bounded
  two-Agent reciprocal block.
- FlowSteer terminal formatting instruction:
  `scripts/prompts/prompt.py::FORMAT_PROMPT`. Formatter input is reduced to a
  supported candidate transfer and the original question is not included.
- SkillFlow bounded Agent execution:
  `src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` and
  `_validate_completion`, adapted through the existing project Runtime.
- SkillFlow Action--Observation retrieval:
  `src/skillev/benchmarks/retrieval.py::QARetrievalEnvironment.execute` and
  `_outcome_for_action`. Successful `qa-retrieval` read receipts remain the
  evidence-provenance authority.

## Necessary project adaptation

- `hotpotqa_semantic_lineage_v2` keeps the existing Reasoner and Verifier
  structured artifact parsers, evidence-span validation, exact question-scope
  comparison, answer-slot binding, and terminal answer-tag validation.
- Role semantics and execution mode are independent. `reasoner`, `verifier`,
  and `format` are semantic responsibilities; ReAct is only
  `execution_mode='react'` with the bounded
  `Thought -> Action(tool) -> Observation -> Thought -> Final` schedule.
- The Director still chooses Agent count, model, contracts, fan-in/fan-out,
  intermediate retrieval/repair branches, and directed or reciprocal
  relations. No direct Reasoner-to-Verifier or Verifier-to-Formatter edge is
  synthesized by the semantic protocol.
- The live constrained action domain binds Runtime declarations by semantic
  capability: a newly sampled Reasoner uses ReAct with `qa-retrieval`, while
  the Canvas validator remains compatible with a pre-existing Tool-free
  Reasoner over routed evidence; Verifier and Formatter are Tool-free
  reasoning Agents. One `add_subgraph` edit may contain a bounded two-edge
  functional block, so FlowSteer's edit--execute--feedback boundary is not
  reduced to one newly sampled edge or one fixed role template.
- FlowSteer's complete-graph validation is applied prospectively when Output
  is assigned or retained. A Canvas revision with an Agent that cannot reach
  Output is rejected before execution, preventing a later conflict between
  terminal reachability repair and preservation of successful dependencies;
  this check does not prescribe a path or exclude fan-in and reciprocal
  topology.
- The r2 live action mask completes any missing Reasoner, Verifier, or
  Formatter responsibility through `add_subgraph` before Output assignment or
  unrelated edits. Once all responsibilities exist, Output admission requires
  at least one routed Reasoner--Verifier--Formatter semantic lineage and a
  routed ReAct `qa-retrieval` capability. Direct role adjacency is not
  required: arbitrary intermediate Agents, fan-in, parallel verification, and
  reciprocal non-Formatter blocks remain legal.
- FINISH validates the actually routed artifacts and Tool receipts. A valid
  terminal path must contain an evidence-grounded Reasoner candidate and a
  supported Verifier artifact with the identical candidate. Multiple valid
  paths may exist, but their candidates must agree.
- Formatter is a terminal serialization boundary. It has no Tool, receives no
  original question, and cannot infer, canonicalize, replace, or reselect the
  semantic answer.
- Recovery reuses the existing
  `preserve_diagnose_repair_augment` policy. Successful artifacts, receipts,
  dependencies, working relations, and Output identity are preserved. Delete
  admission still requires a typed unusable diagnosis and a successful
  same-responsibility replacement takeover.
- The `hotpotqa_semantic_lineage_v2_r1` evaluation condition changes only the
  Executor catalog namespace after the preceding canary recorded remote
  provider request failures. It reuses the already running local
  Qwen3.5-9B-compatible SkillFlow/SGLang entry and preserves the frozen task
  order, seed, evaluator, Director, retrieval protocol, and training-disabled
  boundary.
- The separate r2 condition keeps those frozen fields and enables the existing
  `agentgraph.model-admissible-action-mask.v3` hierarchical receipt boundary.
  The role-first declaration, relation, Output, and MODIFY schemas are derived
  from the current Canvas target domains; the failed r1 trajectories remain in
  their original namespace and are not resumed as r2 results.
- The r4 condition closes three project adaptation gaps exposed by the r3
  canary and local counterexamples. During missing-capability construction,
  the role-first domain samples distinct missing responsibilities, executes
  that accepted `add_subgraph` transaction, and defers Output ownership to a
  later `set_output` transaction. Free-text contract admission rejects an
  unrequested scope restriction using only the original question. FINISH
  enumerates the actual directed paths and requires intermediate public
  artifacts, and current Runtime call receipts when present, to preserve the
  Reasoner's candidate through the corresponding CommunicationEnvelopes.
  These gates do not impose direct semantic-role adjacency or a fixed number
  of Reasoners, Verifiers, auxiliary Agents, branches, or reciprocal blocks.

## Intentionally not implemented in this condition

- Skill retrieval/injection and MACE/Bayesian posterior updates.
- GRPO/LoRA training, backward, optimizer steps, and adapter synchronization.
- Benchmark-specific answer, evidence, Ground Truth, or evaluator hard-coding.
