# HotpotQA Round-01 + QA-memory v17 source map

v17 starts from the verified highest HotpotQA Round-01 Director and six
scalar Canvas actions. It does not adopt a fixed role sequence or topology.
The paired-QA corpus, frozen top-2 index, worker-only Tool ownership, relation
provenance gate, fixed 128 validation tasks, evaluator, seed, round budget and
Director LoRA are unchanged from v16.

## Evidence-driven compatibility changes

The completed v16 diagnostic recorded 1,613 invalid Director actions in 2,543
turns and 466 VectorEngine HTTP 403 Executor failures. The evidence relation
gate itself directly rejected only one FINISH without provenance and behaved
as designed.

v17 therefore makes only these two runtime-availability adaptations:

1. It directly reuses FlowSteer's existing
   `model_admissible_canvas_actions` state-conditioned JSON-schema decoding.
   The schema constrains the selected Round-01 scalar action's wire fields; it
   does not add `add_subgraph`, a role template, or a topology prior.
2. It directly reuses the tracked
   `config/model_catalog_hotpotqa_qa_memory_v10.yaml` local Qwen3.5-9B worker
   catalog. The historical four-model catalog remains recorded in v16, but a
   live minimal generation probe returned HTTP 403 `user quota is not enough`.
   Treating those models as available would repeat an operational failure.

The Director action-token limit remains the Round-01 value of 256 for the
first compatibility canary. It may be raised only if a constrained scalar
action is still observably truncated; it is not raised pre-emptively.

## Acceptance boundary

Before a full 128-task run, the canary must show valid scalar Director actions,
successful local worker execution, worker-owned `search` and `read` receipts,
an explicit worker-to-downstream relation, and a reachable FINISH. Canary
scores are diagnostic and cannot replace the formal fixed-128 result.
