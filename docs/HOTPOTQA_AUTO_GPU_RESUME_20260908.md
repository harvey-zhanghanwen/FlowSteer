# HotpotQA 自动选择空闲 GPU 恢复（2026-09-08）

用户最新要求不再限定 GPU1：选择一张空闲、可训练的卡继续原任务。
12:52 只读检查没有完全空闲的卡，因此关闭仅等待 GPU1 的本任务服务，
切换为单个自动选卡服务，不停止任何其他项目。

## 当前入口

`scripts/resume_hotpotqa_on_idle_gpu.sh auto`

由 `flowsteer-hotpotqa-grpo-auto-20260908.service` 执行。每 2 秒检查本机
GPU 列表，选择没有 compute process 且空闲显存至少 78000 MiB 的卡；
连续 3 次选中同一张后执行 checkpoint/config 前置检查，随即再检查资源。
准备期间若该卡被其他任务占用则不启动训练。最长等待 24 小时；不抢占、
不重启失败训练，不把这个本机检查称为跨项目资源预留。

仅通过已有环境变量展开机制 `HOTPOTQA_TRAIN_GPU` 绑定四个物理 GPU 字段
和两个 CUDA device 字段。其他训练配置不变；未指定变量时默认仍为 GPU1。
选卡完成后保持该卡用于现有单卡 rollout/learner 顺序生命周期，不在 batch
中途迁移。单实例锁避免 GPU1/GPU4/auto 启动重复本任务。

这只是此前运行入口的设备选择适配，继续使用 FlowSteer/MD 的任务学习、
SkillFlow 来源的 SGLang 服务与权重发布、现有单 learner RNG 恢复。
没有新增训练算法、修改样本/模型池、混入 TTB 或改写 checkpoint。

## 验证与真实状态

- 43 tests passed，25 subtests passed；包括默认设备、GPU0–7 环境覆盖及
  仅六个设备字段变化的配置对比、单 learner RNG 恢复和既有 runner 测试。
- `bash -n` 通过；这些是离线检查，不是实际 optimizer step。
- 已有真实更新仍为 1 步；原 checkpoint 与 W&B run `q78e3r39` 保留，
  目标累计 300 步。等待期间不采样、不创建 W&B run、不分配模型显存。
- 实际所选 GPU 记录在服务日志；恢复后日志按 GPU 编号写入
  `artifacts/hotpotqa_dynamic_ledger_grpo_300step/long_training_console_20260908_gpuN_resume.log`。
- 真正的完成数仍以 `run_state.json`、每步 checkpoint/receipt 和 W&B 为准。

```bash
systemctl --user status flowsteer-hotpotqa-grpo-auto-20260908.service
journalctl --user -u flowsteer-hotpotqa-grpo-auto-20260908.service -n 20
```
