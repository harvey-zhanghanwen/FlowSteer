# HotpotQA v7：训练 QA-memory 动态检索正式报告

状态：已收束；这是一次**架构回归证据**，不替换
`hotpotqa_round_01_stable_zero`。

## 评测口径

- 固定 held-out validation：128 条；官方兼容的 HotpotQA answer EM/F1
  evaluator；严格分母始终为 128。
- 数据库：仅使用冻结训练集的 512 条 QA 样本。每个文档由语义保持的
  question paraphrase 和等义 answer statement 构成；512 个 source task
  均唯一，训练循环补齐数为 0。
- 隔离：128 条 validation 的 question、answer、accepted aliases、
  supporting facts 与 evaluator receipt 不进入 QA-memory 或 profile
  selection。
- 检索：`BAAI/bge-base-en-v1.5`（768 维，L2-normalized cosine），训练/
  architecture-development 选择后冻结 `top_k=2`。未使用 Web Search。
- 运行：Director 固定为本地 Qwen3.5-9B；Director 只处理 Canvas 控制面。
  子 Agent 在 `execution_mode=react` 下动态调用
  `hotpotqa.qa_memory.search/read`，并沿 AgentGraph relation 发送 evidence
  artifact。未运行训练、backward、optimizer、LoRA 更新、MACE、Bayesian
  或 Skill loop。

## 真实结果

| 条件 | 完成/有效 | 严格 EM | 严格 F1 |
| --- | ---: | ---: | ---: |
| 本地 Qwen3.5-9B Direct | 128 / 128 | 72.66 | 81.75 |
| QA-memory AgentGraph v7 | 126 / 126 | 4.69 | 6.40 |
| 已保存 Round-01 AgentGraph（同 evaluator 重算） | 128 / 128 | 75.00 | 83.95 |

v7 相对 Direct 为 -67.97 EM、-75.35 F1；相对 Round-01 为 -70.31 EM、
-77.55 F1。terminal failure 为 88；另有 2 条 collection failure。

## Tool 边界与使用

- `director_tool_calls=0`
- `retrieval_tool_calls_by_worker=1764`（882 search + 882 read，全部成功）
- `retrieval_artifact_routed_via_relation=true`（40/40 个已 finish 且使用
  检索的任务满足 relation routing）
- 120 个任务实际使用 Tool，120 个任务出现 query rewriting；共检索到
  311 个不同 QA-memory record / train source task。

因此，v7 的失败不是 Director 直接查询数据库、静态 prefetch 或 Web Search
导致的口径问题。主要运行失败是 `director_max_rounds`（86）与
executor/provider failure（37）；完整 failure taxonomy 和逐条 receipt 位于
同目录 JSON 报告及已忽略的大型运行 artifacts。后续默认仍应指向
Round-01 best profile，而非 v7。
