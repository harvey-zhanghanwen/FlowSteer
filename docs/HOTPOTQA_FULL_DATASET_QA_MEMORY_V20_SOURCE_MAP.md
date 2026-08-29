# HotpotQA full-dataset QA-memory v20 source map

This condition keeps `hotpotqa_round_01_stable_zero` as the architecture
baseline and changes only the retrieval data plane. It is inference-only and
uses the fixed 128-task HotpotQA evaluator panel.

| Project module | Upstream source | Status in v20 |
|---|---|---|
| Qwen3.5-9B Director, progressive Canvas editing, execute-on-edit | FlowSteer Director/Canvas loop; Round-01 profile | Direct reuse |
| Agent execution and explicit relation artifact routing | FlowSteer `AgentRuntime` / `AgentWorkflowEnv` | Direct reuse |
| Worker `search` / `read` Tool and receipt schema | SkillFlow retrieval/Tool boundary through the existing FlowSteer HotpotQA Tool adapter | Direct reuse |
| Worker ReAct action sequence | Existing `HotpotQAEmbeddingReactExecutionAdapter` | Direct reuse |
| HotpotQA official-compatible EM/token-F1 evaluator | Existing `evaluate_hotpotqa_round.py` / task evaluator | Direct reuse |
| Native train+validation QA source projection | Existing `_hotpot_records` converter | Necessary scale adapter |
| 97,852-record QA-memory manifest/index wrapper | Existing HotpotQA transductive QA-memory manifest/index | Necessary count/provenance adapter |
| Full-dataset `corpus_kind` loader/report branch | Existing HotpotQA retrieval runtime factory | Necessary manifest compatibility adapter |

The Director has no Tool and receives no QA-memory payload. Only a Canvas
worker Agent with `execution_mode=react` can call `hotpotqa.qa_memory`.
Retrieved artifacts must reach the Output Agent through an explicit
AgentGraph relation. Web Search, static prefetch, training, GRPO, LoRA update,
MACE update, Bayesian update, and Skill training are disabled.

The database contains native train and validation question-answer pairs,
including the fixed 128 evaluation pairs, after semantic-preserving question
paraphrase and declarative answer rendering. Therefore its evaluation scope
is `in_database_transductive`; it is reported as the requested QA-memory
enhanced direct accuracy and is not labelled held-out generalization.
