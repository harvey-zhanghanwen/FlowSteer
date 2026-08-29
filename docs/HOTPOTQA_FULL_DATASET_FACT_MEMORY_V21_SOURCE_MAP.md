# HotpotQA full-dataset declarative-fact memory v21 source map

This inference-only condition keeps `hotpotqa_round_01_stable_zero` as the
AgentGraph baseline and changes the retrieval data plane. It uses the same
fixed 128-task HotpotQA evaluator panel.

| Project module | Upstream source | v21 status |
|---|---|---|
| Qwen3.5-9B Director, progressive Canvas editing, execute-on-edit | FlowSteer Director/Canvas Round-01 loop | Direct reuse |
| Agent execution and explicit relation artifact routing | FlowSteer `AgentRuntime` / `AgentWorkflowEnv` | Direct reuse |
| Declarative-fact record and worker `search` / `read` action shape | SkillFlow `DocumentPassage`, `SearchHit`, `QARetrievalEnvironment` | Thin adaptation |
| Normalized BGE cosine embedding/ranking | FlowSteer HotpotQA embedding index | Direct reuse |
| Worker ReAct action sequence and Tool receipts | FlowSteer `HotpotQAEmbeddingReactExecutionAdapter`, Tool runtime | Direct reuse |
| Official-compatible EM/token-F1 evaluator | Existing HotpotQA evaluator | Direct reuse |
| Native train+validation source projection | Existing `_hotpot_records` converter | Necessary scale adapter |
| Q-A → semantic rewrite + self-contained declarative fact | User-required fact-corpus compatibility boundary, using the existing local-Qwen structured generation/verification gateway | Necessary adaptation |
| Global 97,852-record fact manifest/index wrapper | SkillFlow passage record plus FlowSteer normalized embedding index | Necessary scale adapter |

## Runtime boundary

- Director Tool calls are forbidden. The Director sees the public task, Canvas
  state, and control-plane feedback only.
- Only a Canvas worker Agent with `execution_mode=react` and
  `allowed_tools=["hotpotqa.fact_memory"]` can call the Tool.
- Every worker must execute dynamic `search(query,k)` and `read(memory_id)`;
  no Web Search or static prefetch path exists.
- Retrieved fact artifacts must reach the Output Agent through an explicit
  AgentGraph relation with Tool-receipt lineage.
- Parametric answering is admitted only after the worker reports the complete
  frozen top-k group as unsupported.

## Data boundary

- Every native question must have one verified semantic rewrite. Coverage must
  be exactly 100%; no verbatim or dataset-pair fallback is implemented.
- The index corpus record is exactly `memory_id + fact_text`.
- The embedding encoder receives only `fact_text`.
- Raw questions, canonical answers, rewritten questions, source IDs, and
  generation/verification receipts remain outside the runtime index as source
  or provenance metadata.
- Search/Read Observation and Tool receipts expose only fact text/snippet,
  opaque memory ID, similarity/rank, and public index identity.

Because facts generated from all native train and validation Q-A pairs include
the fixed 128 evaluation sources, the evaluation scope is explicitly
`in_database_transductive`. The resulting EM/F1 is the requested end-to-end
retrieval-enhanced accuracy, not held-out generalization.
