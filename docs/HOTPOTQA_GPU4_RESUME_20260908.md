# HotpotQA GPU4 恢复记录（2026-09-08）

用户指定将本任务从 GPU5 迁移到 GPU4。训练起点、数据、模型、loss、
rollout 并发、step 数和 W&B run 均保持原配置；任务目标仍为累计 300 个
真实 optimizer steps。当前可恢复 checkpoint 已完成 1 步。

## 最小适配与来源

- `config/training_hotpotqa_dynamic_ledger_grpo.yaml`：单 learner、rollout
  Supervisor、梯度设备统一绑定物理 GPU4；`cuda:4`，端口仍为 8016。
- `src/interactive/smoke_trainer.py::_restore_rng_state`：沿用 FlowSteer
  `train_interactive.py::run_full_training` 的 optimizer/scheduler/Python/
  NumPy/PyTorch RNG 恢复边界。旧项目 checkpoint 将唯一 learner 的 RNG
  记为 `cuda:5`，因此新增单 worker 按 learner 角色恢复到 `cuda:4` 的必要
  迁移适配；不重置种子、不重建优化器、不修改旧 checkpoint。相同设备不变；
  多 worker 不猜测设备排列；缺失、多余或非法设备状态仍拒绝。
- `cuda_rng_restore_device_map` 只作为恢复溯源写入 summary、policy metadata
  与新 checkpoint。下次 checkpoint 保存 `cuda:4`，后续恢复为 identity
  mapping。不据此声称跨 GPU 运行 bit-exact。
- SkillFlow `training/sglang_manager.py::SGLangSupervisorManager` 对应的
  项目服务、LoRA load/save/publish/canary 流程未更改；训练仍是 MD 指定的
  Action-Masked One-Pass GRPO，TTB 禁用。
- `scripts/resume_hotpotqa_gpu4.sh` 是本机运行入口，复用现有训练脚本和
  单卡生命周期，不引入新的训练算法。启动前检查无外部 compute process、
  空闲显存至少 78000 MiB，连续 3 次采样间隔 30 秒；最长等待 24 小时。
  这只是保守资源检查，不是跨项目资源预留。

## 定向验证

`bash -n scripts/resume_hotpotqa_gpu4.sh` 通过。
以下测试共 35 passed、17 subtests passed，均为离线测试：

- `tests/unit/test_smoke_trainer.py`
- `tests/unit/test_dynamic_ledger_training_config.py`
- `tests/unit/test_dynamic_hotpotqa_training_runner.py`

GPU4 配置、step-1 checkpoint 路径、冻结 worker catalog 的只读检查通过。
测试不代表完成新的 optimizer step；迁移后的真实 backward、保存、发布和
新 policy rollout 仍须以实际 step receipt 为准。

## 进程与恢复入口

使用独立 user systemd service：
`flowsteer-hotpotqa-grpo-gpu4-20260908.service`。
只管理该服务自己的进程树，不重启失败训练、不停止其他项目。
标准 W&B SDK 通过 `NETRC=/ssd1/iclr/1/.netrc` 读取既有凭据，不复制或打印密钥。
保持 Director 8016 / worker 8015；取消可能同时覆盖两个端口的环境变量。

```bash
systemctl --user status flowsteer-hotpotqa-grpo-gpu4-20260908.service
journalctl --user -u flowsteer-hotpotqa-grpo-gpu4-20260908.service -n 20
```

等待期间不启动模型、W&B run、rollout 或 optimizer。
GPU4 已于 08:49 被另一项目 `/ssd1/iclr/skillev-real4-gpu4-20260908`
使用，不能停止其进程或复用其 adapter。

真正恢复后沿用 W&B run `q78e3r39`：
https://wandb.ai/zhanghanwen6660909-dut/flowsteer-hotpotqa/runs/q78e3r39

训练日志：
`artifacts/hotpotqa_dynamic_ledger_grpo_300step/long_training_console_20260908_gpu4_resume.log`。
checkpoint / optimizer step 以现有 `run_state.json`、`training_log.jsonl`
和每步 receipt 为准，systemd 的 active 状态不等于训练已开始。
