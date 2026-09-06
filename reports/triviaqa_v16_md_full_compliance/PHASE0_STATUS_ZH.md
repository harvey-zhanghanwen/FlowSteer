# TriviaQA v16：MD Phase 0 状态报告

## 结论

当前严格阶段仍是 **Phase 0：数据可信性**，总体状态为 **未通过**。
已经通过的是 trajectory 持久化、exact Director receipt、graph snapshot replay、
terminal evaluator lineage 和 execution 去重的机器子门禁；这不等价于完整的
train/validation/test 隔离，也不授权进入 Phase 1 或启动训练。

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

## Phase 0 尚未满足的条件

1. **split isolation 未满足。** 固定起点配置
   `evaluation_triviaqa_fact_memory_unified_v4_v16_4_full_native_transductive.yaml`
   明确设置 `enforce_split_isolation: false`，evaluation scope 也明确标为
   `in_database_transductive_fact_only`。该结果不能作为 MD held-out validation、
   posterior calibration 或 Skill gate 证据。
2. **原始证据未随当前分支独立恢复。** 128 条原始 trajectory 约 181 MB，只存在
   另一历史 worktree；当前分支只保留 manifest 外部绝对路径和本次小型 receipt。
3. **最严格的“每次模型调用”receipt 仍不完整。** 顶层 739 条
   `ExecutionRecord` 完整，但 ReAct 内部 provider sub-call 只保存 request ID、
   sampling/version/count 和 Action–Observation trace，没有逐 sub-call 的完整
   prompt/output token IDs。
4. **尚无本次训练用 same-problem/same-condition group。** 当前 128 条全是
   validation 且每题一条，因此 `grpo_eligible=0`；不能代替训练 batch 的
   on-policy group 验收。

TriviaQA 使用确定性的 accepted-answer EM/F1 evaluator，不调用 judge，故本数据集
记录 `judge_applicable=false`、`judge_version=N/A`，不虚构 judge ID。

## 运行时门禁

- 当前任务 namespace 可见 CUDA device：0；
- W&B client：已安装；
- W&B online authentication：当前不可用；
- rollout / backward / optimizer.step / checkpoint / publish / route switch /
  post-update rollout / W&B run：全部为 0。

## 下一合法步骤

先建立 train-only Tool/index 与独立 validation/test 资源边界，生成 split-isolation
receipt；补齐 ReAct provider sub-call receipt；在可用且不冲突的 CUDA allocation
和 W&B online authentication 到位后，收集同一 task、condition、policy version
下的真实 train rollout group。上述条件未满足前不得进入 Phase 1 验收或长训练。
