# HotpotQA：零更新批次恢复与 GPU6 续训

## 规范与已确认事实

本轮遵循两份用户 MD，冲突时以后面的 `LatentLoss_Implementation_Spec.md`
为准。动态 Combination Posterior、latent-risk readout、paired probe 和 Skill
生命周期沿用现有配置；任务学习仍为 Action-Masked One-Pass GRPO，TTB 禁用。
本次修改是执行边界与恢复适配，不代表所有 Skill held-out 验收已经完成。

2026-09-08 14:38 的退出不是用户停止，也不是 OOM。第 4 步生成 28 条自然
轨迹、7 个分组，3 组无奖励差异，4 组被 behavior log-prob 一致性检查拒绝。
learner 明确记录 optimizer_updates=0、grad_norm=0、update L2=0，未产生新
checkpoint；runner 原先把这种返回直接当作不可继续的异常。

当时最大 log-prob 偏差为 0.6895105，阈值为 0.25。发现原预检比较了完整
生成序列，而优化只使用 executed action span。第 4 步的 190 个 turn 都有
未执行后缀，其中 4 个有奖励差异的组含 647 个后缀 token。但原摘要没有偏差
位置，所以不能宣称已经证明后缀是本次拒绝的唯一原因。跨推理后端的数值差异
仍需后续真实运行检验；本轮不放宽 0.25 阈值。

## 最小修改与来源

| 修改 | 来源与边界 |
|---|---|
| `smoke_trainer.py::_preflight_partition` 仅比较 executed action span | 对齐 FlowSteer `train_interactive.py::compute_grpo_loss` 的 action mask、SkillFlow `training/gflownet_trainer.py::_compute_action_logprob_forward` 的 action teacher-forcing；最小切片参考主仓已有代码。完整 receipt shape 与 action 非有限值检查保留。没有移植主仓另一路降低 drift gate 的策略。 |
| `StepTransaction.recover_zero_update/retry_context` | 复用本项目已有 StepTransaction、learner summary 和单步 checkpoint 边界。仅对明确零更新的两类分组拒绝做归档，不是 SkillFlow 原论文定义的新算法。 |
| `train_hotpotqa_grpo.py` 的恢复接线 | 继续复用同一 runner、同一 backend 与 SkillFlow 来源的同一步并发/LoRA 同步；不另造训练层。自然、probe、canary、validation rollout 各自保留原命名规则并增加 attempt offset。 |
| `resume_hotpotqa_on_idle_gpu.sh` 的重试与资源等待 | 复用既有 GPU 空闲门控、flock 和原 CLI；只在已证明零更新的退出码 75 后重新进入资源等待。正常目标边界为 250，原 300-step scheduler horizon 不变。 |

## 自动恢复协议

1. learner summary 必须证明：optimizer_updates、trained_groups、trained_trajectories、
   grad norm、update L2、loss 均为零；无更新后 policy、无新 checkpoint/optimizer
   state，全部分组仅因零奖励方差或 behavior log-prob gate 被拒绝。
2. 旧 policy、adapter、continuation checkpoint 与 committed state 必须一致；
   journal 不得到达 gradient_complete 或之后。`absolute_update_step` 存在初始
   偏移，不得与 optimizer_updates_completed 混为一谈。
3. 原 step 目录与 current recovery 归档，原 journal、evidence、已提交 checkpoint
   和 run_state 保持不变。pending archive plan 支持中断后的幂等恢复。
4. 同一 optimizer step 从原 policy/ledger/Skill 快照重新采样，不采用失败批次的
   next-epoch 候选。每次 attempt 使用独立 rollout index/seed 命名空间：
   `attempt_index * 100_000_000 + 原 index`。旧批次不与新批次合并。
5. 只有真实 optimizer update、可恢复 checkpoint、权重发布、route switch、
   canary、validation monitor、W&B 全部完成才增加 step。失败尝试不计入目标。
6. 未知异常、已有参数更新但提交不完整、policy 不一致或恢复证据缺失不能盲目
   重放。用户 STOP_REQUESTED 仍有效。自动修复不等于忽略所有异常。

## 本次恢复现场与入口

- 失败 step4 归档：
  `artifacts/hotpotqa_dynamic_ledger_grpo_300step/recovery/zero_update_attempts/step_000004/attempt_000001/`。
- 当前 attempt_index=1，rollout_index_offset=100000000；已提交 optimizer step
  仍为 3，恢复 policy 为 `qwen35-9b-hotpotqa-dynamic-ledger-grpo-step-000003`。
- 新服务：`flowsteer-hotpotqa-grpo-gpu6-autorecover-20260908.service`。
- 启动入口仍为 `scripts/resume_hotpotqa_on_idle_gpu.sh 6`；
  `HOTPOTQA_STOP_AFTER_STEPS=250`，只有明确零更新返回 75 才自动重采。
- 此服务自身负责 250-step CLI 停止边界；旧 `stop250` watcher 绑定的是已失败的
  旧服务，仅作历史记录，不是当前服务的控制器。
- GPU6 当前其他进程必须先退出，保留显式确认的 798 MiB 残余分配；至少
  78000 MiB 空闲、连续三次检查、加载前复查。不能据此宣称集群级独占。
- 继续原 W&B run，不新建 run：
  https://wandb.ai/zhanghanwen6660909-dut/flowsteer-hotpotqa/runs/q78e3r39

实际开始/完成步数以当前 service、sampling_attempt、training_log、run_state
和 checkpoint receipt 为准，不把排队、采样或零更新重试冒充 optimizer step。

## 定向验证

- Action-span 预检：8 项 CPU 测试通过。
- 零更新归档/幂等恢复：15 项测试、35 个子测试通过。
- runner 恢复集成及原有顺序边界：6 项 CPU 测试通过，证明失败不提交、重采
  ID 不重叠、同一 policy 继续、成功才增加一次 step；未知/更新后异常拒绝重放。
- shell 语法与 Python 语法检查通过。没有用这些 CPU 测试冒充真实训练结果。

## 15:41 的第二次零更新与零 action mask 修正

第 4 step 的 attempt 1 再次未更新：4 组无奖励差异、1 组 action log-prob
差异超出 0.25（最大 0.2973239），另 2 组被新增的
`invalid_executed_action_span` 拒绝。自动恢复当时没有允许这个新原因，因此
退出；不能把这次退出说成已经自动成功重试。

已确认后两组属于工程误拒：199 个 turn 中有 5 个解析失败轮，真实 receipt
有效、mask 为 0，且所在轨迹有后续有效动作。MD §4.5、原 FlowSteer action
mask 和项目既有 `TrajectoryRecord.grpo_eligible` 只要求整条轨迹至少一个
有效 action token，不要求每轮都有。因此仅将 preflight 的非法范围从
`<=0` 改为 `<0`，保留完整形状校验、越界拒绝及实际 action token 的 0.25
一致性阈值。零 mask 轮仍是上下文，对其梯度为零；全零轨迹仍不进入训练。

已知 action-span 预检拒绝也加入“明确零更新后整批丢弃重采”的恢复列表；
这不允许非法 span 进入 loss，不放宽 checkpoint/非零更新/未知错误门控。
旧 attempt 完整归档到 `attempt_000002`，下一次采样 attempt_index=2、
rollout_index_offset=200000000，完整提交步数仍为 3。

验证：13 项 CPU action-mask 测试通过；20 项恢复及 runner 集成测试、35 个
子测试通过。修复后的分组仍须在真实运行中通过其余 action token 的检查，
不能预先把它们记为有效训练组。
