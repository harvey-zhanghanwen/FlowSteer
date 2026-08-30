# TriviaQA Qwen3.5-9B thinking 兼容性验证

## 结论

当前 AgentGraph 已按 SkillFlow 的 request-scoped decoding 边界启用 thinking：

- Flow-Director：non-thinking，保持 structured Canvas action 的 exact receipt。
- Retriever/ReAct：non-thinking，保持 StructuredAction 与 Tool receipt 可解析。
- Reasoner：thinking，`thinking_budget=512`。
- Verifier：non-thinking。
- Formatter：non-thinking。

该配置没有修改 Canvas action schema、AgentGraph topology、Director search space 或 evaluator。

## 对照结果

| 条件 | 样本 | EM | F1 | FINISH | terminal failure | 状态 |
|---|---:|---:|---:|---:|---:|---|
| non-thinking，同 19 题 | 19 | 94.74 | 98.25 | 19/19 | 0 | 已完成对照 |
| 全语义 Agent thinking，同 19 题 | 19 | 78.95 | 78.95 | 15/19 | 4 | 拒绝 |
| bounded thinking，同 8 题 | 8 | 50.00 | 58.33 | 5/8 | 3 | 提前终止的诊断，不是正式 19 题结果 |
| non-thinking，同 8 题 | 8 | 87.50 | 95.83 | 8/8 | 0 | 已完成对照 |
| Reasoner-only thinking | 3 | 100.00 | 100.00 | 3/3 | 0 | Stable Zero canary |
| Direct，同 3 题 | 3 | 33.33 | 33.33 | N/A | N/A | 同题对照 |

全语义 Agent thinking 的主要退化不是 Tool failure，而是 semantic execution 的 length truncation 破坏 Canvas 闭环。bounded-thinking 8 题中，29 次 semantic call 有 10 次 `finish_reason=length`；相同 8 题 non-thinking 没有 length truncation。

## Live receipt

独立 1 题 receipt canary：

- Task：`triviaqa:tc_1`
- EM/F1：100/100
- termination：`finish`
- Reasoner：`effective_chat_template_enable_thinking=true`，`effective_chat_template_thinking_budget=512`
- Verifier：`effective_chat_template_enable_thinking=false`
- Formatter：`effective_chat_template_enable_thinking=false`
- Retriever/ReAct：5 次 model call 均为 `effective_chat_template_enable_thinking=false`

## 配置状态

Reasoner-only thinking 已成为 v17 的 next-run profile，但尚未替换既有 fixed128 best result。现有 fixed128 non-thinking 诊断仍为 EM 82.81、F1 84.94；需要在新的 strict fact-memory index 完整发布后，才能对 Reasoner-only thinking 运行同一 fixed128 正式比较。

Next-run 配置：`config/evaluation_triviaqa_fact_memory_unified_v4_v17_full_native_transductive.yaml`，其 Model Catalog 指向 `config/model_catalog_multidataset_tool_v9_reasoner_thinking.yaml`。

本轮没有 GRPO、LoRA、backward、optimizer step 或权重更新。
