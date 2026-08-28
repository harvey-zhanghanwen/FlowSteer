# TriviaQA Round-01 + QA-memory v23 source map

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
  referenced as a descriptive baseline. The v23 AgentGraph protocol is not
  claimed protocol-equivalent to this static-retrieval historical Direct arm.
  Those historical rows stored `generation_seed` inside the provider response
  receipt. On read-only reuse, the runner projects that seed to the current
  top-level field only when it exactly equals the frozen v23 seed; the existing
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
`require_evidence_relation=true`. It also explicitly sets
`require_format_agent=false`: the official exact-answer tag remains terminal
syntax, while a dedicated Format role and fixed terminal topology remain
optional.

## Evidence-driven Canvas compatibility

The v21 canary showed that an unconstrained Director could select the
QA-memory worker itself as Output even though the authoritative FINISH gate
correctly rejects that data-plane violation. v23 therefore directly reuses
FlowSteer's existing `model_admissible_canvas_actions` JSON-schema boundary.
It constrains each turn to the live scalar action and target domains; it does
not add a role inventory, action recipe, or topology prior.

## Explicitly not inherited from v19

- No `add_subgraph` action or `max_agents_per_subgraph`.
- No `minimal-neutral.v12` prompt label.
- No `add_subgraph` JSON schema; constrained sampling remains on the restored
  scalar Round-01 Canvas actions.
- No 32k Director context, 1024 action-token limit, 20-round budget, or
  `model_catalog_multidataset_tool_v7.yaml`.
- No static DPR prefetch, Web Search, training, GRPO, LoRA, Skill update, or
  optimizer step.

## Validation

- `tests/unit/test_triviaqa_round01_qa_memory_v23_profile.py` checks the frozen
  Director/Canvas fields and the minimal QA-memory boundary.
- `--prepare-only` validates the runner configuration and freezes the same
  sequential 128-item held-out selection without calling a model or API.

## Paired-QA memory examples

Every embedding document uses the manifest template
`Question: {paraphrase_question}\nAnswer: {paraphrase_answer_statement}`.  The
question and answer therefore remain paired inside one training-memory record.

| Source train task | Original QA | Indexed memory QA |
| --- | --- | --- |
| `triviaqa:tc_224` | Q: Which British general was killed at Khartoum in 1885? A: `Gordon` (plus the dataset's accepted aliases) | Q: Identify the British general who died in Khartoum during 1885. A: The answer is Gordon. |
| `triviaqa:tc_225` | Q: On the border of which two countries is Victoria Falls? A: `Zambia and Zimbabwe` | Q: Between which two nations is Victoria Falls situated? A: The answer is Zambia and Zimbabwe. |
| `triviaqa:tc_227` | Q: What is the name of the volcanic valley that runs from the Sinai peninsula to central Mozambique? A: `Great Rift Valley` (plus the dataset's accepted aliases) | Q: What is the name of the volcanic valley extending from the Sinai peninsula to central Mozambique? A: The answer is Great Rift Valley. |

The index contains 512 memories from 512 unique training tasks, with no cycled
record.  Validation content is not indexed; the fixed held-out partition has
128 tasks.
