# HotpotQA QA-memory v15 architecture preflight

Status: `prepared-only`; no model, API, training, or 128-sample evaluation was run for this commit.

## Frozen condition

- Configuration: `config/evaluation_hotpotqa_qa_memory_v15.yaml`
- Held-out validation: the existing frozen 128 tasks
- Train QA-memory: the existing frozen 512 records, with validation isolation preserved by the source-freeze manifest
- Embedding top-k: 2
- Director: local Qwen3.5-9B, unchanged minimal-neutral v14 prompt profile
- Web Search: disabled
- Training, GRPO, LoRA, MACE, and Skill updates: disabled

## Runtime boundary

The implementation reuses SkillFlow's bounded `StructuredAction` search/read execution and FlowSteer's AgentGraph relation and Tool receipt propagation.

1. The Director receives the public task and control-plane feedback but has no retrieval Tool.
2. A dynamically selected worker with `execution_mode=react` and `allowed_tools=[hotpotqa.qa_memory]` performs one embedding search.
3. The worker must read every record returned by the frozen top-k group before completion is admitted.
4. Search and read observations expose the paired paraphrased train question/answer record; read also exposes the canonical answer and provenance.
5. The worker returns exactly one structured assessment:
   - `supported` with one selected `memory_id` from the fully read group; or
   - `unsupported` with `selected_memory_id=null`.
6. The adapter validates the selected ID against the exact persisted search/read lineage. It does not use a reference answer, evaluator receipt, or similarity threshold as entailment.
7. The assessment and Tool receipts travel only through explicit AgentGraph relations. For `unsupported`, the existing Output Agent answers from the public task and other valid routed evidence.

No role sequence or topology is fixed. The Director's Canvas action space, Agent roles, relations, Output selection, and FINISH semantics are unchanged.

## Directed checks

- `tests/unit/test_hotpotqa_embedding_tool.py`: proves full top-k search/read, strict completion schema, worker-owned receipts, explicit relation routing, public-task fallback, and absence of Director retrieval payload.
- `tests/unit/test_openai_gateway.py`: proves the Output protocol treats `unsupported` as worker abstention and uses the public task instead of copying a memory answer.
- Focused result: 22 tests passed, 3 subtests passed.
- Configuration validation: passed.

## Accuracy status

The prior evaluator-valid HotpotQA reference remains EM 75.00 and F1 84.44. v15 EM/F1 is `N/A` until a separately authorized full frozen-128 run completes; this preflight does not predict or replace formal metrics.
