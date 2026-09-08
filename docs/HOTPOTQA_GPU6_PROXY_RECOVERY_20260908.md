# HotpotQA GPU6：代理环境与未提交批次恢复

## 方法边界

本次完整重读用户的设计 MD 和 `LatentLoss_Implementation_Spec.md`。
主目标保持 terminal-only、same-problem/same-condition、Action-Masked One-Pass
GRPO；TTB 仍禁用。动态后验、配对探针与 Skill 证据不进入 GRPO reward。
本次不是算法修改，也不代表尚未完成的 Skill held-out 验证已经完成。

## 实际故障与处理

- GPU6 的旧 HealthBench 服务已在用户确认后退出，数据和日志未删除。
- 首次 GPU6 续训的 systemd 服务未继承交互环境的 HTTP(S) 代理。
  训练进程的四个外部 worker 连接持续处于 SYN-SENT；直连失败，现有代理的
  无认证 `/v1/models` 请求返回 HTTP 401，说明代理路径可达，不是生成评测。
- 中断前核对 `run_state` 为 1 个已提交 optimizer step，step2 journal 只有
  `prepared`，step2 目录只有 `selected_tasks.jsonl`，没有 backward/optimizer
  入口或 learner 产物。原信号处理只请求在完整 step 后停止，不能取消在途
  网络调用，因此仅暂停并终止了这个 pre-backward 控制进程，由其 systemd
  服务回收所属子进程；没有修改其他项目的进程。
- 未提交恢复记录和 step2 目录已移至
  `artifacts/hotpotqa_dynamic_ledger_grpo_300step/failed_start_20260908_gpu6_missing_proxy/`。
  该目录另有中断前 `run_state`、manifest 和 journal 副本。
  已提交 step1、optimizer/scheduler/RNG、checkpoint、后验快照和准入记录未改变。
- 原 `evidence/` 目录原地完整保留。旧目录可能包含多次失败尝试的同编号轨迹，
  不能据其数量宣称本次 batch 已完成，不能与新 batch 拼接或直接喂给 learner。

## 最小实现与来源

| 改动 | 已有调用链 / 来源 | 原因与边界 |
|---|---|---|
| `storage.root` 支持 `HOTPOTQA_EVIDENCE_ROOT` | 本项目 `config_loader.load_yaml` 的既有环境插值；`LiveSmokeBackend.from_config` → `EvidenceStore`；Phase-E 已有独立 evidence root | 只隔离未提交尝试的存储，避免相同 rollout ID 的不同内容冲突；不改 task/seed/condition/policy/rollout 编号 |
| systemd 启动显式传入原有代理和 localhost NO_PROXY | SkillFlow 原码 `training/sglang_manager.py::_sglang_worker_entry`、`SGLangSupervisorManager.start` 使用父进程环境和 multiprocessing spawn | systemd 父环境与交互 shell 不同，是本项目部署适配，不是论文训练策略 |
| 明确 GPU 与小额既存占用的可选准入 | 原项目 `resume_hotpotqa_on_idle_gpu.sh` 的空闲检查 | 仅在固定 GPU、显式指定 PID、唯一既存占用不超过 1024 MiB、利用率为 0、剩余显存至少 78000 MiB 时准入；保留该进程，不推断归属，不保证集群级独占 |

LoRA 同步继续使用现有 publisher/route-switch/canary 链路；其上游参考为
SkillFlow `training/gflownet_trainer.py::_sync_lora_to_vllm`。没有换成 TTB，
没有用网络恢复替代真实 optimizer.step。

## 本次恢复入口

服务：`flowsteer-hotpotqa-grpo-gpu6-proxy-20260908.service`。
继续使用 `scripts/resume_hotpotqa_on_idle_gpu.sh 6`，其训练入口仍为：

```sh
python scripts/train_hotpotqa_grpo.py \
  --config config/training_hotpotqa_dynamic_ledger_grpo.yaml \
  --allow-md-grpo --resume --stop-after-optimizer-steps 300
```

仅本次进程的运行环境：

- HTTP_PROXY、HTTPS_PROXY、http_proxy、https_proxy：当前已存在的
  `http://127.0.0.1:38081`，不改代理服务或系统全局环境。
- NO_PROXY、no_proxy：`127.0.0.1,localhost,::1`。
- `HOTPOTQA_APPROVED_RESIDUAL_PID=23209`：本次准入时观察到的 798 MiB
  既存占用；后续运行必须重新核对，不能盲目复用这个 PID。
- `HOTPOTQA_EVIDENCE_ROOT=artifacts/hotpotqa_dynamic_ledger_grpo_300step/attempts/20260908_gpu6_proxy/evidence`。

新尝试从已提交 step1 恢复；W&B 继续原 run `q78e3r39`：
https://wandb.ai/zhanghanwen6660909-dut/flowsteer-hotpotqa/runs/q78e3r39

## 定向验证

- 默认 evidence 路径未设置/空值：1 test、2 subtests 通过。
- evidence 环境覆盖仅改变 storage.root，其余完整配置相同：1 test 通过。
- 启动 shell 的 `bash -n` 通过。
- 新 step 是否完成，仍必须以实际 checkpoint、optimizer 更新、发布、canary
  和 committed receipt 为准，不能以服务启动或 W&B 连接代替。

## 用户更新的停止边界：约 250 个完整 optimizer steps

用户要求继续当前训练，直到完整跑完约 250 steps。本次不重启训练、不重置
optimizer/scheduler/RNG，不修改已经加载的 300-step 学习率调度配置。实际停止
边界由独立进程 `scripts/stop_hotpotqa_at_step.py` 监视已有恢复记录来请求：

- 训练服务仍为 `flowsteer-hotpotqa-grpo-gpu6-proxy-20260908.service`，GPU6。
- 监控服务为 `flowsteer-hotpotqa-stop250-20260908.service`，目标为 250。
- `run_state.json.optimizer_updates_completed` 是已完整提交步数；
  `recovery/current_step.json` 是正在进行的事务，二者不可混为一谈。
- 当第 250 step 的 `prepared` 或后续状态已经落盘时，监控进程创建既有
  `STOP_REQUESTED` 标记。trainer 完成当前 step 的 checkpoint、发布、
  route switch、canary、validation monitor 和 W&B 记录后自行停止。
- 若轮询恰好跨过目标边界，则保留已经在途的完整 step，不强行中断 optimizer。
- 若训练异常退出或被已有 held-out gate 暂停，监控进程报告未达到目标并退出，
  不自动重启，也不绕过原有验收条件。此机制不是训练完成证据或聊天推送服务。

来源：本项目 `scripts/train_hotpotqa_grpo.py::run_hotpotqa_training` 已有循环入口和
完整提交后的 `STOP_REQUESTED` 检查，以及
`src/interactive/step_transaction.py::StepPhase`。新增脚本仅为本次长时间
运行的停止控制适配，不新增优化算法、rollout 层或权重同步协议；不宣称来自
FlowSteer/SkillFlow 论文的硬件或调度规定。

定向验证：`tests/unit/test_hotpotqa_stop_at_step.py` 的 69 个用例通过，覆盖
目标前不停止、目标 step 在途/已提交、全部已有事务阶段及无效状态拒绝。

查看后台监控（不启动模型）：

```sh
systemctl --user status flowsteer-hotpotqa-stop250-20260908.service
journalctl --user -u flowsteer-hotpotqa-stop250-20260908.service -n 10 --no-pager
```

这些配置只说明已安排的继续执行与停止条件；最终完成步数、指标和 checkpoint
必须以实际训练记录和 W&B run 为准，不能将 250 写成已经完成。
