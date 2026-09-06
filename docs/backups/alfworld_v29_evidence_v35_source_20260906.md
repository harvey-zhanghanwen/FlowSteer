# ALFWorld 当前源码与历史评测证据备份（2026-09-06）

## 备份边界

本次保存**当前 v35 工作树源码**及 v11–v35 配置、模型目录配置，保留 v29 / v34 正式评测报告和 v35 已停止评测的证据。模型目录配置不等于模型权重。

**v29 产生最高成绩时的完整源码未独立冻结。当前 v35 源码不能被标记为产生 v29 成绩的精确源码，也不能保证只切换 v29 YAML 就能复现该成绩。** 此备份补救当前源码的可恢复性，不填补已经缺失的历史源码快照。

| 条件 | 评测范围 | 成功 / 总数 | Success Rate | 状态 |
| --- | --- | --- | --- | --- |
| v29 | 官方 `valid_seen` | 97 / 140 | 69.29% | 完整正式结果；140 个 evaluator-valid；1 个 terminal failure |
| v34 | 相同 140 个任务 | 87 / 140 | 62.14% | 全部分母结果；137 个 evaluator-valid，3 个 collection failure |
| v35 | 前 14 个已完成任务 | 9 / 14 | 64.29% | 用户要求停止；不是 140 题完整结果 |

对应配置在 `config/evaluation_alfworld_stepwise_multiagent_thinking_v29.yaml`、`config/evaluation_alfworld_stepwise_multiagent_thinking_v34_action_feedback_revisit.yaml` 和 `config/evaluation_alfworld_stepwise_multiagent_thinking_v35_v29_compact.yaml`。正式报告分别位于 `reports/alfworld_stepwise_multiagent_thinking_v29/`、`reports/alfworld_stepwise_multiagent_thinking_v34_action_feedback_revisit/`；v35 实现说明位于 `reports/alfworld_stepwise_multiagent_thinking_v35_v29_compact/implementation_report.md`。

## 来源和备份位置

- 原工作树：`/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1`
- 原分支：`feature/alfworld-stepwise-multiagent-v10-20260830`
- 记录的旧代码起点：`31b8c01`；不将其解释为完整 v29 源码版本。
- 本次备份分支：`backup/alfworld-current-v35-v29-evidence-20260906`
- 既有 GitHub 仓库：<https://github.com/harvey-zhanghanwen/FlowSteer>，远端名称 `backup`。
- 本次 commit：包含本说明的独立备份提交；具体 commit ID 与 push 状态以主线实际推送回执为准，不提前宣称远端成功。
- 修改前本地快照：`/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/pre_v35_source.tar.gz`
- v35 定向验证后本地快照：`/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/v35_validated_source.tar.gz`

以上两份本地快照都不是历史 v29 精确源码快照。原始大型 trajectory / 运行日志仍以本机对应 `artifacts/` 目录为准，Git 备份应保留必要报告与可恢复证据，不包含模型权重、数据库或凭据。

本次同时保留 v29 的原始 run manifest 与 140 条 selected task，以及 v35 原始 run manifest。v35 的原始 manifest 最后停留在采集阶段，不改写历史文件；真实停止状态和 14 题逐题指标另存于 `reports/alfworld_stepwise_multiagent_thinking_v35_v29_compact/stopped_backup_snapshot.json`。历史 v11–v34 配置用于保存迭代记录，不表示每个候选均已接受、完成正式评测或成为默认版本。

## 恢复入口与外部依赖

确认主线已推送后，可在一个新的空目录中 clone 上述既有仓库，并切换 `backup/alfworld-current-v35-v29-evidence-20260906` 分支，恢复本次保存的源码、配置和报告。不要对当前工作树执行破坏性回退。

运行依赖没有打包成自包含环境，仍需按配置提供：

- 外部 SkillFlow adapter / 实现及其 Python 环境；
- ALFWorld 官方 repository、task data 和 environment；
- 本地 Qwen3.5-9B 模型与 tokenizer；
- 原有 SGLang 与评测 Python 环境及其运行配置。

因此，源码可恢复不等于所有外部资源已完成备份，也不等于评测已复现。v35 的必要适配来源和 v29 恢复限制见 `docs/alfworld_v35_v29_baseline_source_map.md`。

## 当前运行状态

评测保持停止；恢复代码不自动启动模型、评测或训练。本轮仅备份，未启用 GRPO、LoRA / optimizer update、MACE、Bayesian 或 Skill evolution；不应从这份备份推断任何新的评测成绩或训练完成状态。
