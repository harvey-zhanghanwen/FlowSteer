# TriviaQA retrieval-first parametric fallback v19

## Scope

This inference-only condition keeps the frozen TriviaQA validation split,
Qwen3.5-9B Director, free progressive Canvas editing, model catalog, top-k,
Tool budget, sampling schedule and official-compatible evaluator from v18.
It changes only the Tool-capable worker execution policy.

## Source alignment

| Boundary | Status | Source |
|---|---|---|
| One StructuredAction followed by one public Observation | direct reuse | SkillFlow `skillev/runtime/bounded_agent.py::BoundedAgent` |
| Retrieval `search` and `read` actions | direct reuse | SkillFlow `skillev/benchmarks/retrieval.py::RetrievalIndex` |
| Bounded `search -> read -> complete` continuation | direct reuse | `src/interactive/react_execution.py::ToolReactExecutionAdapter` |
| Worker-owned Tool receipts and AgentGraph relation routing | direct reuse | `src/interactive/agent_runtime.py::AgentRuntime` and `UpstreamMessage` |
| Train-only paired Question-Answer dense index | direct reuse | `src/interactive/triviaqa_qa_memory.py::TriviaQAQAMemoryIndex` |
| Complete top-k read gate and receipt-backed sufficiency artifact | necessary TriviaQA adaptation | `src/interactive/qa_tool_adapter.py::QARetrievalReactExecutionAdapter` |
| Downstream parametric fallback after an `unsupported` artifact | necessary TriviaQA adaptation | `src/interactive/openai_gateway.py::build_agent_messages` |

No fixed Retriever/Reasoner/Verifier/Formatter chain is introduced. The
Director remains Tool-free and authors the Canvas topology. ReAct remains a
worker execution policy.

## Frozen worker transition

For `triviaqa.qa_memory`, the manifest freezes `top_k=3`,
`max_tool_calls_per_agent_call=4`, and `max_turns_per_agent_call=7`.
The admitted transition is therefore:

1. one embedding `search(query, limit=3)`;
2. one `read(memory_id)` for each of the three returned IDs;
3. one completion containing `evidence_sufficiency`, `answer_source`,
   `supporting_memory_ids`, and `value`.

The adapter rejects completion until all three read receipts exist. A
`supported` decision must cite at least one read `memory_id`, and its value
must equal the canonical answer in that selected paired Question-Answer
record. An `unsupported` decision cites no memory and produces a routed
`parametric_fallback_required` artifact. The downstream Output Agent then
answers the original public task from parametric knowledge.

The sufficiency decision can use only the public task and worker Tool
observations. Held-out labels, evaluator state, accepted aliases and Web
Search are not admitted inputs.

## Runtime entry

Configuration:
`config/evaluation_triviaqa_qa_memory_retrieval_first_fallback_v19.yaml`

The linked worktree uses a local, ignored runtime symlink
`data/joint_qa_v2 -> /ssd1/iclr/1/FlowSteer/data/joint_qa_v2`. The symlink is
not committed; a restored worktree must recreate the same data mount or place
the frozen aligned files at `data/joint_qa_v2`.

Prepare-only verification selected the fixed 128 validation tasks without
starting a model or API. Formal EM/F1 is intentionally not claimed until the
parent task runs the complete fixed-128 evaluation.

## Preflight result (2026-08-28)

- The ignored runtime link resolves to the existing frozen aligned dataset:
  `data/joint_qa_v2 -> /ssd1/iclr/1/FlowSteer/data/joint_qa_v2`.
- Prepare-only completed with `status=prepared`, selected 128 unique tasks,
  and preserved exactly the v18 task-ID order.
- The committed QA-memory manifest contains 512 unique train-split paired
  Question-Answer records, 512 paraphrases, zero cycled records, no validation
  content, BGE-base-en-v1.5 embeddings with dimension 768 and L2-normalized
  dot-product retrieval.
- Targeted verification passed: 8 retrieval-first adapter tests, 6
  control-plane tests, 4 QA-memory wiring tests, 28 OpenAI gateway tests, 46
  runner/config tests, and 100 shared QA Tool adapter tests. No model, API,
  training, optimizer, or formal evaluation process was started.
