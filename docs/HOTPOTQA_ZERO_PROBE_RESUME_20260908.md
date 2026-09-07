# HotpotQA 第 2 步恢复修正（2026-09-08）

## 已确认的失败

训练 run `q78e3r39` 已完成 1 个真实 optimizer step。2026-09-07
22:45 的第 2 步尝试在自然采样后、GRPO 更新前报错：
`dynamic training epoch produced no eligible paired probe site`。
该次尝试没有第二次权重更新，也没有新的 checkpoint。原始日志和任务清单
保留；由于旧代码直到探针完成后才写自然轨迹，不能追溯该批次零选中的
具体风险分布，也不能把未保存轨迹当作可恢复训练数据。

## 规范依据与最小修正

已完整读取项目 MD 和 `LatentLoss_Implementation_Spec.md`。
新规格 §6.1 的 10% 是条件性探针预算，§6.3 的 5% 是高风险位置的
审计采样概率；二者不保证每个 batch 都会选中位置。§6.4、§8 要求自然
轨迹独立进入 GRPO，不能为了填满探针数量而伪造或强制加入干预。

| 修改模块 | 来源与调用链 | 本轮适配 |
|---|---|---|
| `scripts/train_hotpotqa_grpo.py::run_hotpotqa_training` | FlowSteer `train_interactive.py::run_full_training` 的自然 rollout → reward → policy update；SkillFlow `GFlowNetTrainer::_sample_batch_for_step/_collect_episodes` 的同一步任务采样与并行边界；项目新规格 §6、§8 | 删除正式循环“零探针即失败”的额外限制，继续调用已有 `DynamicLedgerEpochCoordinator.collect_selected_probes` 的空集合路径 |
| 同一 runner 的 JSONL 写入顺序 | FlowSteer trajectory serialization、项目 MD §15 与新规格 §10 | 已完成自然轨迹、sidecar、ledger 在探针选择前落盘；不将落盘等同 optimizer commit，不自动复用不完整事务 |
| `tests/unit/test_dynamic_hotpotqa_training_runner.py` | 既有离线 runner 测试 | 同时覆盖有探针/零探针，及探针选择报错时保留自然证据 |

没有修改 Director 提示词、模型池、样本、seed、探针阈值、并发度、GRPO loss、
action mask、checkpoint、发布或 Skill gate。Phase E 的真实配对探针验收仍保留。
零探针 batch 的对比项新增观测为 0；自然轨迹仍按既有实现更新任务基线和
传感器。不会因此生成或发布未经 held-out 验证的 Skill。TTB 继续禁用。

## 验证

使用原环境运行下列四个定向测试文件，共 **12 passed**：

- `test_dynamic_hotpotqa_training_runner.py`
- `test_dynamic_epoch_transition_helpers.py`
- `test_dynamic_training_bridge.py`
- `test_dynamic_phase_e_runner.py`

测试为离线测试，不是额外真实 optimizer step，也不创建 W&B run。

## 恢复与资源边界

从现有 `run_state.json` 指向的 step-1 checkpoint 恢复，目标累计 300 步，
保持 W&B run `q78e3r39`。GPU5 的外部服务 PID 31958 / 33009 使用端口
18045，非本任务所有，不停止、不修改、不共享其 adapter。
当前约占 68 GiB，剩余约 11 GiB，不满足本任务已验证的单卡顺序执行需求。

只允许等待 GPU5 无其他 compute process 且空闲显存至少 78000 MiB 后启动；
这是保守的本机启动条件，不是论文给定阈值，也不是抢占或资源预留机制。
等待期间不加载模型、不创建 W&B run、不采样、不更新权重。

恢复时必须取消 `FLOWSTEER_SUPERVISOR_PORT` 环境覆盖，保留配置默认：
Director `8016`、冻结 worker `8015`。直接设置该环境变量为 `8016`
会同时修改 worker endpoint，破坏冻结模型目录条件。启动前比较现有版本 ID，
不修改 catalog、posterior 或 checkpoint 来绕过不一致。

本记录只证明代码修正及离线测试完成；实际训练阶段、optimizer 数量以
`artifacts/hotpotqa_dynamic_ledger_grpo_300step/run_state.json`、训练日志和
W&B 为准，排队不等于已经开始第 2 次更新。
