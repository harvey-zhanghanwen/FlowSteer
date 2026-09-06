# ALFWorld v35 架构改动与评测启动记录

## 当前状态

2026-09-05（Asia/Shanghai）：架构修改、103 项定向测试和 prepare-only 已完成。固定前两题 smoke 均 evaluator-valid、成功且正常 FINISH，随后原命令继续完整 140 题。首次确认全量启动时已落盘 3 题，成功 3 题，collection failure 和 terminal failure 均为 0。此处不是最终成绩，后续进度及分数以同目录 `valid_seen_report.json` 和 artifact manifest 为准。

两题 smoke 的原生环境动作数分别为 9、8；第三题为 4。全量阶段直接复用两题原始 trajectory，没有重新采样。

## 改动与来源

- 采用 v29 已记录的 `skillflow_public_invariants_v2` 动作策略及同一 140-task panel，保持 Qwen3.5-9B、thinking、20 环境动作、32 Director 轮次和并发度 1。
- 复用现有 gateway 的 Agent 消息路由，取消 Tool Agent task 文本中的重复上游正文；来源、revision 和完整正文仍由标准路由提供。
- 按 FlowSteer 的紧凑执行反馈投影及 SkillFlow 的历史反馈处理方式适配：最新 artifact 保留一份全文；历史正文使用预览；相等冗余状态使用引用。完整原始 receipt 和 trajectory 不变。
- 保留每步的目标、当前 observation、动作执行结果、原生 admissible actions、环境 revision、剩余预算，以及统一 AgentGraph 的自由 contract/relation。没有新增固定角色或固定拓扑。
- 未训练，未启用 GRPO、LoRA、MACE、Bayesian 或 Skill evolution。

v29 未有精确对应的源码快照，因此本轮是基于其可确认配置/协议的新 v35 适配，不能声称精确恢复历史源码。

## 验证证据

103 项定向 unittest 通过；v29/v35 的 140 条 selected task 记录完全相同。五条已有长轨迹经本地 tokenizer 离线比较，输入 tokens 合计从 134767 降至 94885，减少 29.59%；每题 25 项当前公开状态字段保持原值。没有额外模型调用，此结果不是成功率指标。

历史参考：v29 为 97/140（69.29%）；v34 为 87/140（62.14%）；相同协议已有 Direct 为 56/140（40.00%）。v35 最终 SR 尚未产出，不作预测。

## 文件与恢复

- 配置：`config/evaluation_alfworld_stepwise_multiagent_thinking_v35_v29_compact.yaml`
- 完整来源与验证说明：`docs/alfworld_v35_v29_baseline_source_map.md`
- 过程与轨迹：`artifacts/alfworld_stepwise_multiagent_thinking_v35_v29_compact/valid_seen/`
- 修改前本地源码：`/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/pre_v35_source.tar.gz`
- 通过测试后本地源码：`/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/v35_validated_source.tar.gz`
- GPU2；SGLang 端口 8015；评测 tmux socket `flowsteer_alf_v35_20260905`，session `evaluation`。评测在后台持续，不依赖本轮对话保持打开。

本轮尚未执行 GitHub commit/push；以上是本地可恢复备份，不是远端备份完成声明。全部三个有界子任务已完成，无常驻子智能体。

这个固定 valid_seen panel 已用于多轮架构调整，最终分数属于同 panel 的迭代比较，不是未接触测试集的泛化成绩。
