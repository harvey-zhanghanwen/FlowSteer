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

## Intentionally not implemented in this condition

- Skill retrieval/injection and MACE/Bayesian posterior updates.
- GRPO/LoRA training, backward, optimizer steps, and adapter synchronization.
- Benchmark-specific answer, evidence, Ground Truth, or evaluator hard-coding.

