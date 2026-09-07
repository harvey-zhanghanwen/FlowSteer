# TriviaQA v16 训练前置准备报告（2026-09-07）

## 结论

- 固定起点：`backup/triviaqa-fact-memory-v16-best-82p81-20260906` @
  `114bd8b6378af55fbd1f40cdfdda91c79314a771`
- 工作分支：`train/triviaqa-v16-md-full-compliance-20260906`
- 当前阶段：**Phase 0：数据可信性（进行中，未整体通过）**
- 主目标：terminal-only、same-problem/same-condition、Action-Masked
  One-Pass GRPO；TTB、Skill、MACE exploration 与辅助 reward 均关闭
- 真实 rollout / backward / `optimizer.step` / checkpoint / publish / canary /
  W&B run：**全部为 0**

## 已完成的非执行性准备

1. 已物化 512 条 TriviaQA train 和 128 条 held-out validation；
   `base_task_id` overlap=0，当前数据无需 cycle 补齐。
2. 已从既有 TriviaQA fact-memory 流投影 512 条 train-only declarative facts。
   Agent-facing 记录只有 `schema_version`、`memory_id`、`tool_id`、`fact_text`；
   原始 question、canonical/accepted answers 仅存在 index 外 provenance。
3. 已在 CPU 构建本地 embedding index：BGE 768 维、L2 normalization、
   dot-product、冻结 `top-k=3`、`memory_count=512`，index 不加载 provenance。
4. 已增加 TriviaQA frozen schedule/cursor 薄适配，复用 SkillFlow 的冻结任务序列、
   exact cursor 与 write-once 边界；每个 step 固定 1 个 train task × 4 trajectories。
5. runner 已支持 TriviaQA-only scope。`--prepare-only` 实际选择
   `triviaqa:tc_224` 与 rollout ordinals `[0,1,2,3]`，没有静态 retrieval
   prefetch，next cursor 未生成。
6. 已增加 materialized held-out/GPU evidence provider；它只接受现有 evaluator
   生成的新 policy report/trajectory receipts，并 fail-closed 拒绝旧 receipt、
   intervention、fallback 或错误 evaluator。
7. `execution_gate.execution_enabled=false` 已由 runner 强制执行；普通启动会在
   模型后端和 W&B 初始化前拒绝，只有 `--prepare-only` 可运行。

## 当前资源门禁

只读资源核对显示 8 张 H800 均有其他项目进程。GPU 4 有约 47 GiB 空闲并且
核对时利用率为 0%，可作为后续共享候选；但其他候选卡存在持续计算或只有约
10–25 GiB 空闲，当前无法同时为 learner、SGLang Supervisor rollout 和 gradient
replica 冻结一个不影响其他项目的三卡 allocation。

共享部分占用 GPU 只在重新核对显存余量、持续利用率、端口和进程所有权后允许；
当前没有 allocation receipt，也没有打开 execution gate。

## 尚未完成

1. ReAct worker 内部 provider model-call 的完整调用证据仍需在新 trajectory 中验收；
   Director 的 exact prompt/output token receipt 要求保持不变。
2. updated-policy 的正式 128 held-out evaluation 尚未接到训练事务；W&B evidence
   provider 尚未由 CLI/外层 driver 实例化。
3. W&B online run 未初始化，因此当前没有 run URL。
4. 尚未取得 4 条真实 on-policy trajectories、terminal reward、non-zero loss/
   gradient/LoRA delta、完整 checkpoint、adapter publish/route switch/canary 证据。
5. Phase 1–5 尚未验收；250–300 step 长训练继续禁止。

## 定向验证

- 新增准备、schedule、runner、W&B evidence 测试：35 passed。
- 既有 smoke runner、HotpotQA schedule、JointQA schedule 回归：63 passed，
  另有 8 个 subtests passed。
- 本报告中的 prepare-only 运行没有加载模型、调用 API、初始化 W&B 或使用 GPU。
