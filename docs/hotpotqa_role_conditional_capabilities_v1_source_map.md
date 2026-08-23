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
