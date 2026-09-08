# HotpotQA GPU1 恢复记录（2026-09-08）

用户指定 GPU1，并说明 Trivia 正在迁移；GPU1 释放后立即恢复 HotpotQA。
旧 GPU4 等待服务已在确认其仅运行 bash/sleep 后停止，未停止任何模型或
其他项目进程。当前只保留一个 HotpotQA 等待服务及原训练单实例锁。

- 配置：`config/training_hotpotqa_dynamic_ledger_grpo.yaml`，单卡顺序执行，
  learner / Supervisor / gradient 全部指向物理 GPU1、`cuda:1`。
- 入口：`scripts/resume_hotpotqa_on_idle_gpu.sh 1`，由既有 GPU4 运行入口
  做 GPU 参数化的最小运行适配，未引入训练算法或新的模型服务实现。
- user systemd service：`flowsteer-hotpotqa-grpo-gpu1-20260908.service`。
- 资源检查：每 2 秒查询 GPU1；连续 3 次没有 compute process 且空闲显存
  至少 78000 MiB 后，调用原训练脚本恢复。最长等待 24 小时；不抢占现有
  进程、不预先申请显存、不自动重启失败训练。此检查不是集群资源预留。
- 恢复进度：已有 1 个真实 optimizer step，目标累计 300 步；W&B 继续
  `q78e3r39`，等待期间不创建 run、不采样、不更新参数。
- CPU 读取真实 checkpoint 已确认格式、policy、step 与单 learner 状态；
  CUDA RNG 保存于 `cuda:5`，使用已实现的单 learner 角色映射恢复到
  `cuda:1`。源 checkpoint、optimizer/scheduler/RNG 数据均未改写。
- Director 仍为本地 Qwen3.5-9B、端口 8016；冻结 worker 仍为 8015。
  算法、数据、采样条件、LoRA 配置不变，TTB 仍禁用。

定向测试：35 passed、25 subtests passed；入口 `bash -n` 通过。
测试均为离线检查，不代表新的 optimizer update 或 GPU1 实训已通过。

```bash
systemctl --user status flowsteer-hotpotqa-grpo-gpu1-20260908.service
journalctl --user -u flowsteer-hotpotqa-grpo-gpu1-20260908.service -n 20
```

启动后日志写入
`artifacts/hotpotqa_dynamic_ledger_grpo_300step/long_training_console_20260908_gpu1_resume.log`。
真实 step 计数继续以该 run 的 `run_state.json` 和每步 receipt 为准。
