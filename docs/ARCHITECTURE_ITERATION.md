# AgentGraph 单任务架构迭代

`scripts/evaluate_agentgraph_architecture.py` 是一个有界的架构诊断入口。当前 iteration 02 从七个已对齐训练源中各跳过前 3 条记录，再固定选取 1 条新任务；每个任务只采 1 条 rollout。它使用独立的 `condition_id` 和输出目录，不与 iteration 01 混合。其目的只是观察 AgentGraph Canvas、Director 动作、Executor 执行和 evaluator 接线的实际失败点。

## 来源与边界

| 分类 | 本轮使用内容 |
| --- | --- |
| 直接复用 | FlowSteer 的 progressive Canvas、逐轮 trajectory/receipt、固定 policy 与 catalog version；复用 smoke 路径的七源顺序选择、`LiveSmokeBackend.collect` 和终局 evaluator。 |
| 必要适配 | 新增 evaluation-only 外壳，将范围锁定为七源各 1 条、`skip_per_dataset: 3`，并用单任务失败隔离、严格续跑和逐数据集可测性报告承载架构诊断。WebShop 最多执行 10 个环境 step，ALFWorld 最多执行 50 个环境 step。 |
| 项目既有设计 | 使用项目文档定义的自由文本 Agent contract、两比特关系、六个 Canvas 原子动作和简洁中性的 Director 提示词；本轮没有新增训练算法。 |
| 关闭/未实现 | GRPO、optimizer、LoRA 更新、adapter 发布、MACE、贝叶斯探索和 Skill 全部关闭；SWE-bench 正式 harness 未接通时只能标为不可测。 |

共享架构已进一步收紧为源码一致的运行边界：Canvas 保存 FlowSteer 式逐轮
action/feedback 历史，Director 只读取 SkillFlow 式有限最近窗口；YAML 中的
`max_agents` 和 progressive `execute_on_edit` 已进入实际运行路径。同一 graph
revision 的 progressive 结果可供 no-op/`finish` 复用，但旧 Agent 调用不会在
新的 turn receipt 中重复登记。达到 `max_rounds` 只产生显式终局失败。

本轮读取的是 `train` split，不是 held-out validation，也不是七数据集 benchmark。单条样本不能估计准确率；behavior reward 和 `diagnostic_macro_mean` 仅用于定位架构问题。报告始终写入 `heldout_validation: false` 和 `stop_threshold_assessed: false`。采集失败与 evaluator invalid 均保留原因，invalid 不按 0 分补记。

若 trajectory 已落盘，入口只在其 `task_id`、`condition_id` 和完整 `VersionBundle` 与当前冻结运行全部一致时复用；其余任务重新采集。每条新完成的 trajectory 都会立即原子落盘，因此进程中断后可只补缺口，不会再次调用已经成功且完全匹配的付费 Executor 请求。resume 数量和本轮新采数量分别写入 manifest。

`explicit_finish: false` 且 `termination_reason: max_rounds` 一律是 Director 的自然终局失败。即使旧 evaluator receipt 意外带有环境成功 reward，诊断报告也强制记 0，并在 `evaluator_reward_ignored` 留下被忽略的原值。对于 WebShop/ALFWorld，这类 `final_answer: null` 不应启动环境；该调用边界位于共享 `LiveSmokeBackend.collect`，不是 evaluation-only 外壳。本入口已保证报告不把它算成成功，但共享 runner 仍须在 evaluator callback 接线处加入空终局短路后，才能同时避免不必要的环境调用。

## 运行

仅在用户明确开始下一轮、GPU 4 的本地 Qwen3.5-9B Director 服务已由操作者启动，并且 Executor 凭据已通过环境变量提供后运行：

```bash
python3 scripts/evaluate_agentgraph_architecture.py \
  --config config/evaluation_agentgraph_architecture.yaml
```

入口不会调用 trainer、`optimizer.step()` 或 policy publisher，也不会热同步 adapter。产物写入 `artifacts/agentgraph_architecture_iteration_02/data/`：冻结任务、完整 trajectory、逐数据集诊断报告和 manifest。部分任务失败不会取消其余任务，但七条全部采集失败时入口返回失败。

若需要评估“七项均严格超过 60%”的未来验收条件，必须另用固定 held-out validation、每项足量样本及正式 evaluator/harness；不能复用这里的训练样本、单次 rollout 或不可测记录。
