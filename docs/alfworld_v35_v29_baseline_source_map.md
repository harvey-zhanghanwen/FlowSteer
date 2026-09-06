# ALFWorld v35：基于 v29 协议的反馈去重

## 基线、范围与证据

v29 在官方 `valid_seen` 的同一批 140 个任务上成功 97 个，SR 为 69.29%；v34 成功 87 个，严格 SR 为 62.14%。同任务配对为 21 个退步、11 个改善，不能把全部差异归因于 3 个上下文溢出。v34 输入 token 总量比 v29 增加约 23.4%，提供了检查重复输入的直接理由，但没有单因素实验就不能声称这解释了全部性能变化。

v29 的源码修改当时没有保存为独立提交或源码快照，两个报告的 Git 起点都早于这些修改。本轮恢复的是 **v29 已记录的配置与动作协议**，并基于现有统一架构形成独立 v35 条件；不是声称精确复现历史源码。原始配置、报告、轨迹均保留。

修改前源码备份：`/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/pre_v35_source.tar.gz`。备份包含当前 src、scripts、config、tests、docs，不包含运行中间文件、模型或 .env。

## 来源与最小改动

| 模块 | 实际来源 | 本轮处理 |
| --- | --- | --- |
| 环境 reset/session/action/observation/reward | SkillFlow `src/ragen_adapter.py` 与项目已接入的 ALFWorld 环境 | 直接复用，仍一题一个 episode、一个环境 Tool owner、每次一个原生动作。 |
| 当前动作策略 | `config/evaluation_alfworld_stepwise_multiagent_thinking_v29.yaml` 的 `skillflow_public_invariants_v2` | 恢复此 profile；不使用 v34 新加的 v3/v4 过滤，不新增目标位置或搜索顺序规则。 |
| Tool Agent 上游通信 | `src/interactive/openai_gateway.py::build_agent_messages` / `_format_upstream` | 直接复用。`environment_execution.py::_action_prompt` 只在新条件中取消同一正文在 task 文本中的额外副本，标准 gateway 仍传完整正文及来源版本。 |
| Director 执行反馈 | `/ssd1/iclr/1/FlowSteer/src/interactive/director.py::_director_compact_execution_feedback_projection` | 必要适配：当前 ALFWorld envelope 另含 `agent_artifacts` 和输入 provenance；通过已有 artifact ID 引用重复正文，每个最新 artifact 保留一份完整正文。 |
| 历史 Action–Observation | SkillFlow `training/environment.py::_build_react_prompt`、`_alfworld_history_max`、`_alfworld_history_obs_chars`；项目已有 Director 历史窗口 | 必要适配：压缩旧反馈中的重复正文，保留真实动作、反馈和版本；保留现有 Observation–Action 对齐、初始任务上下文和最新状态。 |
| 最新公开状态 | v29 已有的 `latest_action_observation` 与现有完整环境 receipt | 必要适配：仅在 Director 输入投影中消除相等的长字段副本。当前 observation、原生可选动作、目标进展、最新动作结果、revision、预算继续可见。原环境 receipt 与 trajectory 不裁剪。 |
| AgentGraph / Canvas / FINISH | 当前 FlowSteer 衍生 `AgentWorkflowEnv`、`AgentGraphOrchestrator` | 直接复用。保持自由 contract、relation、Output 选择、每部分执行后反馈，不增加固定角色或固定拓扑。 |
| 配置与范围校验 | 项目 `config_loader.py` / `train_agentgraph_smoke.py` 的既有 runtime factory | 必要适配：`compact_execution_feedback` 为显式布尔开关，默认 False；仅允许配对开启的 ALFWorld stepwise 配置。 |
| GRPO / LoRA / MACE / Bayesian / Skill evolution | 不适用于本轮 | 未启用；没有训练、权重更新或新增 Skill。 |

## 新条件

配置：`config/evaluation_alfworld_stepwise_multiagent_thinking_v35_v29_compact.yaml`。

保持 v29 的同一 140-task selection、seed 20260825、Qwen3.5-9B 模型目录与 thinking、Agent model catalog、20 个环境动作预算、32 个 Director 轮次、history_window=3、并发度 1，以及 ALFWorld 官方环境 evaluator。Direct 继续复用原条件的 56/140 结果。

两个开关同时为 True：`director.compact_execution_feedback` 与 `environment_runtime.compact_execution_feedback`。新输出仅进入 `artifacts/alfworld_stepwise_multiagent_thinking_v35_v29_compact/valid_seen`。历史 v29/v34 输出不覆盖。

先运行固定前 2 条作为执行 smoke，确认链路和 receipt 正常后用同配置续跑全部 140 条。这两条直接复用，不基于其成功率挑选任务或更改配置。正式启动后冻结本轮代码与配置。

## 验证与报告边界

定向验证覆盖：完整上游正文只出现一次，来源与 revision 保留；当前公开状态/动作集合完整；重复字段引用可解析；原 receipt 不修改；旧配置默认行为不变；历史角色与动作顺序不变；参数与 runtime 接线正确。

对历史长 prompt 的离线投影与 token 计数用于验证输入容量，不产生新的模型答案或环境分数。真实 SR 只能由本轮 ALFWorld 环境执行结果产生。

### 已完成的验证（2026-09-05）

103 项定向 unittest 全部通过，覆盖两个新增反馈测试模块、原有环境执行与多 Agent 测试、smoke runner 的环境参数接线。prepare-only 通过；v29 与 v35 的 140 条 selected task 记录逐条相同。没有重跑 Direct，沿用相同协议的已落盘结果。

从 v34 已落盘轨迹中选取五题的最大 Director prompt，使用本地 Qwen tokenizer 离线比较输入投影，没有调用模型。原 token 数与记录的 prompt token 数相符：

| Task ID 后缀 | 原输入 tokens | 投影后 tokens | 减少 |
| --- | ---: | ---: | ---: |
| 00007 | 29027 | 14620 | 49.63% |
| 00042 | 31205 | 22848 | 26.78% |
| 00046 | 29250 | 20005 | 31.61% |
| 00089 | 19359 | 15649 | 19.16% |
| 00103 | 25926 | 21763 | 16.06% |
| 合计 | 134767 | 94885 | 29.59% |

每题 25 项当前目标、最新动作结果、原生动作集合、动作策略集合、预算及 revision 字段保持原值；当前 observation 保留直接文本。此检查仅证明这五条长输入的容量改善，不证明 SR 提升，也不能保证所有未来轨迹均无上下文超限。

通过测试后的源码另存于 `/ssd1/iclr/1/.tmp/alfworld-v35-source-backup.ad2ULN/v35_validated_source.tar.gz`。这是本地可恢复备份，不代表 GitHub push 已完成。

本轮运行于物理 GPU2、既有 Qwen3.5-9B SGLang 服务的 8015 端口。运行入口：

```bash
python scripts/evaluate_completion_benchmark_round.py --config config/evaluation_alfworld_stepwise_multiagent_thinking_v35_v29_compact.yaml --canary-only
python scripts/evaluate_completion_benchmark_round.py --config config/evaluation_alfworld_stepwise_multiagent_thinking_v35_v29_compact.yaml
```

完整命令只在 smoke 链路检查通过后继续，正式阶段复用已完成 smoke 的有效轨迹，包括 reward=0 的正常失败，不能只复用成功样本。启动时间为 2026-09-05 20:07（Asia/Shanghai），代码与配置在此时冻结。最终状态与真实分数以同 condition 的 `run_manifest.json` 和 `valid_seen_report.json` 为准；不把 smoke 成绩当作全量成绩。

最终严格分母固定为 140，并同时报告 evaluator-valid 数、成功数、运行异常、terminal/max_rounds、动作数量和典型错误。多次使用同一个 `valid_seen` panel 调整架构，因此结果属于该固定 panel 上的架构迭代对照，不是新的、未接触测试集上的泛化证明。
