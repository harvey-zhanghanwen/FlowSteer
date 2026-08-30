# TriviaQA fact-memory v16 source map

This note records implementation provenance and the frozen inference protocol
for the v16 fact-memory profiles, including the preserved v16.4 diagnostic
checkpoint.  It does not promote a protocol-invalid diagnostic score to a
formal result.

## Source mapping

| Project boundary | Upstream source | Reused contract | Necessary adaptation |
|---|---|---|---|
| `src/interactive/triviaqa_qa_memory.py::TriviaQAFactMemoryIndex` | SkillFlow retrieval index and `search`/`read` Tool semantics | Local BGE encoder, normalized embeddings, dot-product ranking, deterministic Top-K, immutable search/read lifecycle | Versioned `record_kind=fact_memory`; Agent-facing rows contain only `schema_version`, opaque `memory_id`, `tool_id`, and `fact_text` |
| `scripts/generate_triviaqa_qa_memory_paraphrases.py::build_fact_memory_projection` | Existing TriviaQA materialization boundary | Frozen source ordering and semantic-preserving paraphrase admission | Split the declarative `fact_text` corpus from the database-external Q-A provenance sidecar; provenance is not an index or Tool input |
| `src/interactive/qa_tool_adapter.py` and `src/interactive/react_execution.py` | SkillFlow bounded `StructuredAction -> Tool Observation` execution | One admitted Action/Observation per turn, bounded Tool budget, explicit receipts and completion | A worker Agent owns `triviaqa.qa_memory` search/read; the Director does not call the Tool or receive fact payloads |
| `src/interactive/agent_workflow_env.py` and `src/interactive/agent_runtime.py` | FlowSteer progressive Canvas editing, execute-after-edit feedback, explicit `FINISH`, trajectory records | One atomic Canvas edit, execution feedback, explicit AgentGraph relations, artifact-versioned communication and terminal admission | Retrieved `fact_text` is routed only over an explicit relation to reasoning, verification, and formatting consumers; Director feedback remains control-plane only |
| `src/interactive/task_evaluator.py` and `scripts/evaluate_completion_benchmark_round.py` | TriviaQA official accepted-answer normalization and FlowSteer terminal evaluation timing | Fixed 128-task selection, `triviaqa.official.answer.v1`, evaluator-only labels, persisted trajectory/evaluator receipt | The metric scope is explicitly `in_database_transductive_fact_only`; it is not a held-out retrieval-generalization result |

## Frozen v16 protocol

1. The local Qwen3.5-9B Director performs progressive Canvas editing and sees
   only public task text, Canvas state, and structured control-plane feedback.
2. A Tool-capable worker Agent executes local fact-memory `search` followed by
   ordered `read(memory_id)` actions under the frozen index Top-K and Tool
   budget. Web Search is not part of this profile.
3. Search/read receipts belong to the worker `agent_id`. Fact evidence reaches
   downstream Agents only through explicit AgentGraph relations.
4. The Output/Format Agent consumes provenance-bearing upstream artifacts and
   emits the exact single `<answer>...</answer>` terminal format.
5. The evaluator remains `triviaqa.official.answer.v1`; Direct predictions are
   reused from the existing question-only local Qwen3.5-9B baseline receipt.
6. Training, GRPO, LoRA publication, exploration, policy synchronization, and
   Skills are disabled. The only active inference assignment is rollout and
   Supervisor on GPU0; learner and gradient GPU fields are inactive validator
   placeholders.

## Data-plane boundary

- The index path is `data/triviaqa_fact_memory_full_native_v1/index`.
- Its manifest must declare `record_kind=fact_memory`, `fact_only=true`,
  `embedding_input_field=fact_text`, and
  `provenance_loaded_by_index=false`.
- Original questions, canonical answers, accepted aliases, source IDs, and
  evaluator metadata remain in the database-external provenance projection.
  No provenance path is configured in the runtime profile.
- The v16.4 fixed-128 batch completed with diagnostic EM 82.81% and F1 84.94%,
  but only 123/128 outputs had admitted terminal lineage.  A subsequent corpus
  audit also found at least one fact with unresolved reference (for example,
  ``The company called it Frosted food.``).  The checkpoint is therefore kept
  as a recoverable diagnostic baseline, not a formal protocol-valid result.
