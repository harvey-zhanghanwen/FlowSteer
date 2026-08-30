# TriviaQA v17 评测口径

## 条件定义

本 profile 是 **in-database transductive evaluation**，不是 held-out evaluation：

- 原始 `rc.nocontext` native train 去重后共有 76,523 条 QA；
- 固定 128 条 evaluation task 均位于这 76,523 条 source projection 中；
- 数据库仅索引语义保持改写后的自包含 `fact_text`；
- 原问题、标准答案、accepted aliases 与 evaluator receipt 只保存在数据库外 provenance，不进入 embedding 输入或 Agent-facing Tool payload；
- 检索只能由 worker Agent 在 ReAct execution 内调用本地 Tool，Director 不调用 Tool，也不接收检索正文；
- 禁止 Web Search。

因此，后续 128 条 EM/F1 必须标记为 `transductive accuracy`，不得描述为官方 held-out TriviaQA 指标，也不得与论文 held-out 数值直接等价比较。正式报告必须同时记录 overlap count=128、数据库记录数、Tool receipt 与 evaluator version。

## Thinking profile

- Director：non-thinking；
- Retriever/ReAct：non-thinking；
- Reasoner：thinking，budget=512；
- Verifier：non-thinking；
- Formatter：non-thinking。

该边界保持 FlowSteer structured Canvas action 与 SkillFlow structured Tool action 的可解析性，同时允许 Reasoner 使用 Qwen3.5-9B thinking。
