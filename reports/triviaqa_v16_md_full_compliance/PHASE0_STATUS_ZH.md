# TriviaQA v16：MD Phase 0 状态报告

## 结论

当前严格阶段仍是 **Phase 0：数据可信性**，总体状态为 **未通过**。
已经通过的是 trajectory 持久化、exact Director receipt、graph snapshot replay、
terminal evaluator lineage、execution 去重，以及新完成的 train/held-out 数据隔离
与 train-only fact index 子门禁；这些局部结果不等价于 Phase 0 整体通过，也不授权
进入 Phase 1 或启动训练。

## 已通过的子门禁

只读输入：历史 v16.4 的 128 条 TriviaQA validation trajectory。

机器 receipt：`phase0_acceptance.json`

- trajectory：128/128 可由当前 `TrajectoryRecord` 解析并 round-trip；
- Director turn：910/910 保存真实 prompt、prompt/output token IDs、behavior
  log-prob、实际消费 action prefix、request ID、generation seed、policy/server
  weight version；
- Agent execution：739 条，execution ID 全局唯一；
- execution reuse：110 个 reuse turn 均没有重复保存历史 execution；
- graph snapshot：128/128 完整 predecessor chain 可重放；
- terminal evaluator：128/128 valid，terminal reward 与
  `triviaqa.official.answer.v1` receipt 可追溯；
- 模型、Tool、evaluator 与 tokenizer 均未重新调用或重新计算。

Graph replay 的必要薄适配来自实际 FlowSteer progressive Canvas 调用链：一个
Director 原子 edit 可以触发多次内部 graph mutation，因此相邻 per-turn full
snapshot 的 revision 可以跳变；FINISH 或 execution reuse 可以保持同一 revision。
replay 现在要求 revision 非递减，并继续要求完整 predecessor linkage。

新增的数据与检索子门禁：

- TriviaQA 已按冻结规则物化 512 条 train 与 128 条 held-out validation，
  `base_task_id` overlap=0；512 条 train 均为唯一 base task，没有 cycle 补齐；
- 已从既有 fact-memory 数据流投影 512 条 train-only `fact_text`；held-out
  validation 的 memory overlap=0；
- Agent-facing index 只加载 schema/tool/memory ID 与 `fact_text`，原始 question、
  canonical/accepted answer 仅保留在 index 外的 provenance sidecar，且 index
  manifest 明确 `provenance_loaded_by_index=false`；
- index 已在 CPU 构建：BGE embedding 768 维、L2 normalization、dot-product，
  冻结 `top-k=3`。本次构建没有启动 rollout、模型推理或训练。

## Phase 0 尚未满足的条件

1. **历史最高分证据仍是 transductive，不能转化为 held-out 证据。** 固定起点配置
   明确使用 `in_database_transductive_fact_only`；新建的 512/128 隔离数据和
   train-only index 解决了本次训练输入的 split 子门禁，但历史结果不能作为 MD
   held-out validation、posterior calibration 或 Skill gate 证据。
2. **历史原始证据未随当前分支独立恢复。** 128 条原始 trajectory 约 181 MB，只存在
   另一历史 worktree；当前分支只保留 manifest 外部绝对路径和本次小型 receipt。
3. **最严格的“每次模型调用”receipt 仍不完整。** 顶层 739 条
   `ExecutionRecord` 完整，但 ReAct 内部 provider sub-call 只保存 request ID、
   sampling/version/count 和 Action–Observation trace，没有逐 sub-call 的完整
   prompt/output token IDs。
4. **尚无本次训练用 same-problem/same-condition group。** 当前 128 条全是
   validation 且每题一条，因此 `grpo_eligible=0`；不能代替训练 batch 的
   on-policy group 验收。
5. **真实 on-policy group 尚未采集。** TriviaQA frozen schedule/cursor 与 runner
   scope 已接通，`--prepare-only` 已精确选中一题及 rollout ordinals 0–3，且没有
   静态 retrieval prefetch、没有推进 cursor；但 `execution_gate.execution_enabled=false`，
   所以尚无真实 same-problem/same-condition rollout、reward 或 optimizer evidence。

TriviaQA 使用确定性的 accepted-answer EM/F1 evaluator，不调用 judge，故本数据集
记录 `judge_applicable=false`、`judge_version=N/A`，不虚构 judge ID。

## 运行时门禁

- 当前只读核对显示 8 张 GPU 均有其他项目进程占用；现有 learner、rollout
  Supervisor、gradient replica 三卡分工没有安全的不冲突组合；
- 用户允许共用未用满的卡，但只能在资源门禁确认显存余量、并发负载、端口和
  服务所有权后共享；当前没有冻结 GPU allocation receipt；
- W&B client：已安装；
- W&B binding：固定为 `zhanghanwen6660909-dut/flowsteer-triviaqa`、online；
- W&B 标准凭据：用户确认已配置，将仅由 SDK 在正式 run 初始化时读取；
- W&B online connection/run URL：尚未初始化、尚未验证；
- W&B runner adapter：固定绑定、run URL/run ID 门禁、逐 step telemetry、
  checkpoint artifact 与异常 finish 已存在；新的 materialized held-out/GPU
  evidence provider 也已实现，但 updated-policy held-out evaluation 的外层调度和
  CLI provider 注入尚未接通；
- complete checkpoint schema：已实现并通过 CPU 定向测试，但没有真实 CUDA
  optimizer step 的落盘/恢复证据；
- rollout / backward / optimizer.step / checkpoint / publish / route switch /
  post-update rollout / W&B run：全部为 0。

## 下一合法步骤

保留已物化的 512/128 split、512-fact train-only index 及已验证的 TriviaQA
micro schedule/cursor；补齐 ReAct provider sub-call receipt、updated-policy held-out
evaluation 调度及 CLI evidence-provider 注入。在资源门禁冻结可共享且不冲突的 CUDA
allocation，并且 W&B online run 初始化成功、取得 run URL 后，才可打开 1-step
execution gate，收集同一 task、condition、policy version 下的 4 条真实 train
rollout。上述条件未满足前不得进入 Phase 1 验收或长训练。
