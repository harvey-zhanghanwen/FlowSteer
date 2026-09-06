# TriviaQA v16：W&B 训练绑定

本文件记录 `WANDB_BINDING_20260906_V1` 的执行边界。它是项目为设计文档
训练事务增加的可观测性与恢复要求，不改变训练目标，也不表示训练已经启动。

## 固定绑定

- entity：`zhanghanwen6660909-dut`
- project：`flowsteer-triviaqa`
- mode：`online`
- 凭据：仅由 W&B SDK 的标准凭据链读取；训练配置、manifest、日志和 artifact
  metadata 均不接受凭据字段。
- 启动门禁：`wandb.init` 成功且返回非空 run URL 后，才允许开始正式 rollout。
- 恢复：新 run 使用持久化 run ID；恢复只允许相同 run ID 且
  `resume="must"`。

## 每个 optimizer step 的记录

每次真实 `optimizer.step` 必须在进入下一轮 rollout 前记录：

- `global_step`、`dataset`、behavior/updated policy version、checkpoint version；
- terminal reward、同题同条件 group reward mean/std、有效 rollout 数；
- Action-Masked One-Pass GRPO loss、gradient norm、LoRA update L2；
- train 与固定 held-out validation 的 official metrics；
- GPU memory、rollout/train/publish/canary/step wall-clock；
- checkpoint、adapter publish、route switch、control-plane canary 与
  post-update rollout canary 状态。

本地完整 checkpoint 先保存并验证，再允许 adapter publish。checkpoint 包含
SkillFlow named `theta` LoRA adapter，以及 FlowSteer 恢复边界所需的 optimizer、
scheduler contract、Python/NumPy/Torch CPU/CUDA RNG、step/policy version 和训练
metadata。W&B model artifact 使用固定 collection `triviaqa-director-lora`，
每个 checkpoint 作为该 collection 的新 version，并至少使用 `latest` alias；
只有相同 split、相同 evaluator、相同 protocol 的 held-out validation 主指标刷新
历史最佳时才能增加 `best` alias。train reward、canary 或 transductive 指标不得
触发 `best`。

## Source map

| 边界 | 上游位置 | 本项目处理 |
| --- | --- | --- |
| W&B 初始化/step log | SkillFlow `training/gflownet_trainer.py` 的 `wandb.init` 与 `wandb.log` 调用位置 | 薄适配为 online-only、URL-required、fail-closed；不复用其 TTB 指标或 fail-open 行为 |
| run ID/resume/finish | FlowSteer `train_interactive.py` 的 W&B resume 与 finish 边界 | 薄适配为持久化 run ID、`resume="must"` 和显式 exit code |
| LoRA adapter checkpoint | SkillFlow named `theta` adapter 的 `save_pretrained` 路径 | 直接复用 |
| optimizer/scheduler/RNG checkpoint | FlowSteer `train_interactive.py` 的 `training_state.pt` 保存/恢复边界 | 最小适配到现有 one-pass GRPO trainer；当前无 scheduler，因此保存显式 constant-learning-rate disabled contract，不引入新 scheduler |
| checkpoint artifact、aliases、route/canary 状态 | 用户 MD 与 `WANDB_BINDING_20260906_V1` | 项目必要实现；不冒充 FlowSteer/SkillFlow 原始代码 |

## 当前状态

- `MD_FULL_COMPLIANCE_20260906_V2` 的严格 Phase 0 尚未整体通过。
- 当前任务命名空间没有可用 CUDA device。
- 尚未创建 online W&B run，因此 run URL 为 `null`。
- fail-closed runner envelope、run ID/URL receipt、逐 step telemetry 与可恢复
  model artifact 已通过 mock SDK 定向测试；正式 held-out validation/GPU
  telemetry provider 仍缺生产接线。
- rollout、backward、`optimizer.step`、adapter publish、route switch、canary 均未执行。
- TTB 未启用、未实例化、未运行。

因此当前只能验证配置、checkpoint/recovery schema 和 mock W&B SDK 接线；不得把
这些定向测试描述为 W&B 已连接或训练已启动。
