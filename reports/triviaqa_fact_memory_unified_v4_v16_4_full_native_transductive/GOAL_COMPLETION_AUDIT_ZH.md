# TriviaQA inference-only 目标完成审计

## 审计结论

当前目标 **尚未完成**。数据、索引和评测均已生成并落盘，但现有证据不能证明以下两个目标条件：

1. 76,523 条 `fact_text` 全部 self-contained 且语义保持；
2. 固定 128 条评测满足当前 AgentGraph 正式协议。

因此 EM 82.81%、token F1 84.94% 只能报告为 `in-database transductive fact-only diagnostic`，不能报告为 protocol-valid 或 held-out TriviaQA benchmark result。

## 逐项审计

| 目标要求 | 状态 | 权威证据 |
|---|---|---|
| 76,523 个 unique TriviaQA source | 已满足 | `materialization_manifest.json`；facts/provenance/index 三个 JSONL 均为 76,523 条 |
| 76,523/76,523 发生改写、无原问题 substring fallback | 结构性检查满足 | manifest：rewrite、lexical replacement、fact_text 均为 76,523；fallback=0；exact original substring=0 |
| 每条 `fact_text` 在语义上自包含 | **未满足** | `fact_memory.jsonl:116` 为 `The company called it Frosted food.`，存在未消解指代且缺少实体锚点 |
| Agent-facing embedding input 只使用 `fact_text` | 已满足 | facts/index payload 只有 `schema_version,memory_id,tool_id,fact_text`；`provenance_loaded_by_index=false` |
| CPU BGE、768 维、L2、dot product | 已满足 | index manifest；实际 embeddings 为 `(76523,768)` float32 且已归一化 |
| development-only Top-K 冻结 | 已满足 | 512 development tasks，validation 未用于选择，Top-K=5，并与 final index manifest 一致 |
| 19 条 canary 完整执行 | 采样与计分已完成；当前严格 lineage 未完全通过 | selected/paired/trajectory=19；FINISH=19；EM=94.74%、F1=98.25%；当前 lineage 15/19 |
| 固定 128 条 AgentGraph 完整采样与计分 | 已完成 | selected/paired/trajectory=128；task ID 集合一致；evaluator-valid=128；collection failure=0 |
| 固定 128 条正式协议 | **未满足** | FINISH=110/128；terminal failure=18；Output lineage=123/128；`protocol_valid=false` |
| worker-Agent 动态 Tool 检索 | 已满足旧运行协议 | Director Tool=0；worker search=189/read=939；首个 data-plane action search=128/128；Web Search=0 |
| 当前 exact receipt validator | **未满足** | 当前 validator 将旧 native artifact projection 判为 schema 不匹配，需对齐 artifact/receipt schema 后重跑 |
| 指标、receipt、错误 demo 保存 | 已落盘 | report、formal analysis、run/preflight manifest、128 trajectories、22 wrong demos 均存在 |
| 无训练、无 Web Search、未进入其他数据集 | 已满足 | `training_enabled=false`、optimizer updates=0、Web Search=0 |
| GitHub 远端备份 | **未完成** | 当前分支无 upstream；环境拒绝向未确认的 `origin` 外发 |
| 本地可恢复备份 | 已完成 | branch `backup/triviaqa-fact-memory-v16-evidence-20260830` 与 Git bundle 已建立 |

## fixed128 已完成的诊断结果

| 条件 | 分母 | EM | token F1 | evaluator-valid |
|---|---:|---:|---:|---:|
| Direct | 128 | 35.16% | 40.82% | 128/128 |
| AgentGraph non-thinking | 128 | 82.81% | 84.94% | 128/128 |

terminal failure 为 18/128：`canvas_action_domain_exhausted=16`、`max_rounds=2`。错误样本 22 条：worker execution 10、reasoning/answer selection 5、terminal 5、Agent communication/relation 2。

## thinking 决策

Reasoner-only thinking 已关闭。fixed128 前 16 个同题样本中：

- non-thinking：EM 93.75%、F1 97.92%、terminal failure 0；
- thinking：EM 87.50%、F1 91.67%、terminal failure 1。

thinking 的 EM/F1 均下降 6.25 个百分点，因此不再续跑。

## 完成目标所需的最小后续工作

1. 获得用户授权后，修复所有非 self-contained fact，重新 materialize 76,523 条 fact-memory 并重建 CPU BGE index；当前用户已明确要求停止继续构建，因此本轮不能执行。
2. 对齐 native fact artifact 与当前 exact search/read receipt schema，并修复 Output lineage/terminal recovery。
3. 冻结同一条件后重新运行完整 fixed128，使当前 analyzer 得到 `protocol_valid=true`；不能拼接不同条件的局部结果。
4. 用户在获知外发风险后明确确认现有 `origin` 目的地，才能重试 GitHub push；不得绕过环境拒绝。

在以上缺口解决前，不应将本目标标记为完成。
