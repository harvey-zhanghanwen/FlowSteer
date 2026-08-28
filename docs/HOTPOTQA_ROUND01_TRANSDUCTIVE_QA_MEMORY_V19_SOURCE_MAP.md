# HotpotQA Round-01 + transductive QA-memory v19 source map

v19 starts from the verified highest `hotpotqa_round_01_stable_zero` profile
and retains the v17 constrained scalar Canvas-action decoding, local Qwen3.5-9B
worker catalog, worker-only Tool ownership, control-plane Director feedback and
explicit evidence relation gate. It does not introduce a role sequence,
subgraph template, fixed topology, Skill, training update, or Web Search.
v19 additionally enables the existing
`preserve_diagnose_repair_augment` recovery policy; this is configuration reuse,
not a new recovery implementation.

## Retrieval condition

The isolated index contains the frozen 512 training QA and frozen 128
validation QA as paired, semantically preserving QA-memory records. It is
therefore a **transductive retrieval diagnostic**:

- `contains_evaluation_answers=true`;
- `evaluation_regime=transductive_retrieval`;
- `evaluation_overlap_count=128`;
- `official_heldout_eligible=false`.

Any v19 EM/F1 must remain separate from the held-out Round-01 best profile and
must never replace it.

## Execution boundary

The Director receives the public task, Canvas state and control-plane receipts;
its request has no Tool, retrieval payload or retrieved QA text. A Director
action may create any admissible free-form worker topology. Only a worker with
`execution_mode=react` and `allowed_tools=[hotpotqa.qa_memory]` can access the
index. Within that worker call the inherited action mask admits:

1. `search(query, k=2)` first;
2. `read(memory_id)` for every returned memory;
3. `complete` with `retrieval_sufficiency=supported` and a selected memory, or
   `retrieval_sufficiency=unsupported` with no selected memory.

Only `unsupported` admits downstream parametric fallback. A supported selected
memory is routed through an explicit AgentGraph relation before the Output
Agent can consume it. The FINISH gate still requires worker Tool provenance and
that relation, without requiring a fixed role or serial topology.
An existing successful evidence artifact remains protected until a replacement
artifact has taken over its lineage.

## Prepare-only acceptance

Before any model-backed canary or full diagnostic:

- the configuration must preserve the Round-01 tasks, seed, Director, Canvas
  action set, evaluator and disabled training blocks;
- the index manifest must prove 512+128=640 records and the transductive labels;
- the Tool registry must expose local `search/read` only and identify the corpus
  as `transductive_qa_memory`;
- the worker action domain must be search-first/read-before-complete;
- `web_search_enabled` must remain false.
