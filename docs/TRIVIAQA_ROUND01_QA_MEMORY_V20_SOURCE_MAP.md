# TriviaQA Round-01 + QA-memory v20 source map

This condition is an inference-only composition. It does not introduce a new
Director policy or AgentGraph topology.

## Directly frozen from the verified Round-01 profile

Source: `config/evaluation_triviaqa_round_01.yaml`.

- Director seed `20260817`, catalog-order namespace
  `triviaqa_round_01_local_qwen35`, prompt label
  `agentgraph.director.minimal.v2`, Qwen3.5-9B endpoint and sampling values.
- Director limits: context `8192`, action tokens `512`, temperature `0.7`,
  `max_rounds=14`, edit-time execution, history window `4`.
- Canvas: `max_agents=8`, `free_text` contracts, two-bit relations, and the
  six actions `add_agent`, `modify_agent`, `delete_agent`, `set_relation`,
  `set_output`, `finish`.
- Executor selection, bidirectional-block limit, output/reachability rules,
  exact-answer terminal protocol, and `config/model_catalog_triviaqa_v1.yaml`.
- The stored Round-01 Direct prediction file and its historical protocol are
  referenced as a descriptive baseline. The v20 AgentGraph protocol is not
  claimed protocol-equivalent to this static-retrieval historical Direct arm.
  Those historical rows stored `generation_seed` inside the provider response
  receipt. On read-only reuse, the runner projects that seed to the current
  top-level field only when it exactly equals the frozen v20 seed; the existing
  task/model/protocol/execution/evaluator resume checks remain unchanged.

## Necessary QA-memory adaptation reused from v19

Sources:

- `config/evaluation_triviaqa_qa_memory_retrieval_first_fallback_v19.yaml`
- `src/interactive/qa_tool_adapter.py`
- `src/interactive/triviaqa_qa_memory.py`
- `scripts/train_agentgraph_smoke.py`

The fixed joint-QA split and official TriviaQA evaluator are retained. A
worker Agent may declare `execution_mode=react` and
`allowed_tools=["triviaqa.qa_memory"]`. Its bounded execution performs one
embedding search with frozen `top_k=3`, reads all three returned QA memories,
and emits either a supported memory answer or an unsupported receipt that
permits parametric fallback. Search/read receipts belong to that worker and
the resulting artifact must reach the Output Agent over an explicit relation.

The Director remains Tool-free and receives only control-plane feedback. The
condition disables legacy deterministic DPR prefetch and does not expose Web
Search. The additions to the Round-01 Canvas configuration are therefore only
`director_feedback_mode=control_plane`,
`required_evidence_tool_id=triviaqa.qa_memory`, and
`require_evidence_relation=true`.

## Explicitly not inherited from v19

- No `add_subgraph` action or `max_agents_per_subgraph`.
- No `minimal-neutral.v12` prompt label.
- No JSON-schema/model-admissible action sampling profile.
- No 32k Director context, 1024 action-token limit, 20-round budget, or
  `model_catalog_multidataset_tool_v7.yaml`.
- No static DPR prefetch, Web Search, training, GRPO, LoRA, Skill update, or
  optimizer step.

## Validation

- `tests/unit/test_triviaqa_round01_qa_memory_v20_profile.py` checks the frozen
  Director/Canvas fields and the minimal QA-memory boundary.
- `--prepare-only` validates the runner configuration and freezes the same
  sequential 128-item held-out selection without calling a model or API.
