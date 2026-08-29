# HealthBench Professional Artifact Routing v3 运行报告

## 结论

本轮完成了 inference-only Artifact communication 改造和静态验证，但没有
产生新的 HealthBench Professional 官方分数。固定 525 条 public-test 样本已
准备完成；canary 在 synthetic evaluator preflight 阶段被 grader 账号的
HTTP 403 `insufficient_quota` 阻断，正式 Direct/AgentGraph 生成均未开始。

HealthBench Professional 的主指标是
`overall_score_length_adjusted`，不是 Accuracy、EM 或 F1。v3 的主指标当前为
**N/A**，不能从历史结果推算。

## v3 架构改动

- 上游 Artifact 现在携带 producer Agent 的 `model_id`、free-text
  `contract`、`execution_mode`、可选 role metadata、
  `completion_condition`、provider `finish_reason`、Artifact version 和 Tool
  receipt provenance。
- 单向 relation 与 reciprocal relation 使用同一消息 envelope；只去除同一
  model input 内 byte-identical 且 version 相同的重复 envelope，不合并近似
  医学文本，也不跨 producer 合并。
- downstream component cache 在 v3 条件下依赖 Artifact version 与 producer
  context；上游 Artifact 变化时不会错误复用旧输出。
- Canvas feedback 保留 compact preview、Artifact version、reciprocal
  `peer_draft` 与 cache reuse receipt；完整内容仍只保存在 trajectory，避免
  feedback 重复膨胀。
- 保留已有 SGLang `repetition_penalty=1.05`，没有叠加训练奖励或改变模型
  权重。
- 统一 AgentGraph search space、自由 Agent 数量、free-text contract、关系
  类型和唯一 Output Agent 保持不变；未增加固定医疗 role 或 workflow。

## 验证状态

| 项目 | 结果 |
| --- | --- |
| 固定 public-test 样本 | 525，顺序与 v1/v2 完全一致 |
| Gateway/Runtime/Canvas 定向测试 | 255 passed + 56 subtests |
| Config/runner 定向测试 | 51 passed + 23 subtests；新增 preflight 诊断后 runner 单文件 38 passed |
| GPU0 Qwen3.5-9B runtime | preflight 到达 evaluator；无模型服务错误 |
| HealthBench evaluator preflight | failed：HTTP 403 `insufficient_quota`，3/3 bounded retries |
| v3 正式生成 | 未启动 |
| v3 官方主指标 | N/A |

## 当前可验证的同协议指标

以下是历史条件，不是 v3 改进后的分数：

| 条件 | Direct | AgentGraph | 差值 | 完整性 |
| --- | ---: | ---: | ---: | --- |
| official-v1，525 固定分母 | 19.1728% | 20.2395% | +1.0667 pp | 完整收束；AgentGraph 503 valid、22 terminal failures |
| inference-loop-v2，473 complete-case paired | 18.3629% | 20.5584% | +2.1955 pp | evaluator 部分完成，不能替代正式 525 指标 |
| inference-loop-v2，525 固定分母下界 | 17.2993% | 18.3264% | +1.0271 pp | 510 Direct / 488 AgentGraph evaluator-valid |

## 继续运行条件

只有 pinned grader 配额恢复后，才能先重跑 canary，再运行 525 条正式 paired
evaluation。Direct 文本与 v3 的 model/protocol/seed 一致，可复用 v2 的 525
条生成；无效的 15 条 evaluator receipt 只重新评分，不重新生成答案。
