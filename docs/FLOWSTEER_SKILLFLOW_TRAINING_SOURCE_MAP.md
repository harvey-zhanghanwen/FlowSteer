# FlowSteer / SkillFlow 原始源码 Source Map

## 固定来源与方法边界

- FlowSteer：`beita6969/FlowSteer@a329b577a8e8c2f2d5492983b492473a06a8e718`，与论文 v4 对齐。
- SkillFlow：`beita6969/SkillFlow@74be52bb6bd9f0e9e68dacb72636b75649197983`，本地 commit-exact checkout 为 `/ssd1/iclr/owner/skillflow-iclr/06_GITHUB_RELEASE_20260816`。
- 项目规范：`FlowSteer_MACE_Bayesian_Skill_Design.md` v1.1。
- HotpotQA 起点：`backup/hotpotqa-compliant-best-round01-20260906@740e53ec6ccac635ecbe7f1f379b002bfb2574d1`。

`MD_FULL_COMPLIANCE_20260906_V2` 已确定唯一任务学习目标为 **Action-Masked One-Pass GRPO**。SkillFlow 的 Tempered Trajectory Balance、φ-LoRA、partition function Z 与 TTB residual 全部禁用。SkillFlow 仅为 Qwen3.5、SGLang、同一步 rollout 并行、LoRA、checkpoint 基础和权重同步提供源码参考。

完整逐章节矩阵见 `docs/MD_FULL_COMPLIANCE_20260906_V2.md`。

## FlowSteer 原始调用链

```text
train_interactive.py::run_full_training
  -> InteractivePromptBuilder / Director generation
  -> truncate after first action
  -> InteractiveWorkflowEnv.step
     -> ActionParser
     -> WorkflowGraph update
     -> AFlowExecutor
     -> execution feedback
  -> TurnRecord / Trajectory
  -> correctness / terminal reward
  -> compute_grpo_loss / backward
  -> optimizer.step
  -> LoRA save + service reload
  -> W&B / checkpoint
```

FlowSteer 发布源码的 graph 是有序 operator DSL，reward 含结构项，训练分组按 source，loss 也没有完整实现论文 Eq. 12 的 behavior ratio、PPO clipping 与 reference KL。因此只能直接复用原子编辑、Canvas 状态转移、执行反馈和训练循环边界；自由 AgentGraph、同题同条件分组、terminal-only reward 和 exact receipt 都是 MD 要求的最小适配。

## 模块级 source map

| 模块 | 原始代码出处 | 当前代码 | 分类与理由 | 状态 |
|---|---|---|---|---|
| Director generation | FlowSteer `train_interactive.py::create_generate_fn`；`src/interactive/prompt_templates.py::InteractivePromptBuilder` | `src/interactive/director.py` | 薄适配：XML operator action 改为自由 AgentGraph JSON atomic action；提示词保持简洁中性 | 已实现 |
| 首个原子动作 | FlowSteer `_truncate_after_first_action`；`ActionParser.parse` | `agent_action_parser.py`、`rollout_collector.py` | 语义直接复用、语法薄适配 | 已实现 |
| Progressive Canvas Editing | FlowSteer `InteractiveWorkflowEnv.step/_step_internal/_execute_workflow` | `agent_workflow_env.py::AgentWorkflowEnv.step` | 保留“编辑→校验→执行→反馈” | 已实现 |
| 自由 AgentGraph | FlowSteer `workflow_graph.py` 仅有 operator DSL | `agent_graph.py` | MD 必要项目新增：自由 contract、稳定 model ID、显式 relation、有限双向块和 DAG 校验 | 已实现 |
| Agent execution | FlowSteer `workflow_builder.py::create_aflow_executor_wrapper`；`aflow_executor.py` | `agent_runtime.py::AgentRuntime` | 必要适配：固定 operator executor 改为异构 Agent、显式通信与有限双向执行 | 已实现 |
| Trajectory / receipt | FlowSteer `workflow_builder.py::{TurnRecord,Trajectory}` | `records.py`、`rollout_collector.py` | 薄适配：加入 prompt/output IDs、逐 token behavior log-prob、policy/adapter/server version、execution 与 evaluator receipt | 已实现基础层 |
| Terminal reward | FlowSteer `trajectory_reward.py::TrajectoryRewardCalculator`、`_compute_correctness` | `task_evaluator.py`、`TrajectoryRecord.grpo_eligible` | 只复用终局回传边界；结构奖励、substring heuristic 和 LLM-judge override 不进入 MD GRPO | 已实现基础层 |
| 同题 advantage | FlowSteer `train_interactive.py::compute_grpo_loss` | `grpo_objective.py::same_condition_advantages` | 必要修正：严格按 `(task_id, condition_id, policy_version)` 分组 | 已实现 |
| Action-Masked One-Pass GRPO | FlowSteer `compute_grpo_loss` / `grpo_trainer.py::compute_grpo_loss` | `grpo_objective.py::{action_masked_one_pass_loss,torch_action_masked_one_pass_loss}` | 发布代码只提供 policy-gradient 外形；MD 补 exact receipt、真实 action mask、每组/轨迹归一化和一次消费 | 已实现；本分支未运行真实 step |
| 同一步并行 rollout | SkillFlow `GFlowNetTrainer::_sample_batch_for_step/_collect_episodes` | runner 的同一冻结 policy batch 并发 collect | 仅复用同一步并行；禁止原版跨 optimizer 的 async prefetch | 已实现基础边界 |
| Qwen3.5 θ-LoRA | SkillFlow `GFlowNetTrainer.setup`、`configs/skillflow.yaml` | `smoke_trainer.py::_load_models` | 上游对齐 r64/α128/q,k,v,o、PEFT、bf16、gradient checkpointing；`AutoModelForMultimodalLM`、dropout=0、两个独立 θ-only 副本和双副本 continuation 是 Qwen3.5/MD 必要兼容适配；不创建 φ/Z | 已实现基础层 |
| SGLang Supervisor | SkillFlow `training/sglang_manager.py::SGLangSupervisorManager` | `src/interactive/sglang_manager.py::SGLangSupervisorManager` | 薄适配 child process、ServerArgs 与 readiness；只终止自身 child。正式 Hotpot runner 尚未接入，GPU resource admission 仍由外部门禁完成 | 未接线、未启动 |
| Exact SGLang sampling | SkillFlow `batch_inference.py::supervisor_call` | `rollout_collector.py::SGLangReceiptDirectorClient` | 必要适配：原函数没有 MD 所需 prompt/output token IDs、逐 token log-prob 与 policy receipt | 代码存在；fresh 实证缺失 |
| 多卡 backward | SkillFlow `_batched_logprob_backward/_average_replica_grads` | `smoke_trainer.py` | 必要适配：只复用双副本并发与 gradient merge 边界；完整同题 group、token-cost 分区、GRPO loss/normalization 与 4→2→1 OOM backoff 均为 MD/项目实现 | 代码存在；本分支未运行 |
| θ checkpoint | SkillFlow `_save_checkpoint/resume`；FlowSteer `train_interactive.py::run_full_training` 的 optimizer/scheduler/RNG save/restore | `smoke_trainer.py::_save_training_state/_restore_training_state` 与 adapter save | adapter 保存/加载复用 SkillFlow；optimizer、cosine scheduler 与 RNG 边界复用 FlowSteer；两张任务 GPU 的 CUDA RNG、training step 与 frozen version metadata 是 MD 必要适配 | 已接线；待真实单步恢复验收 |
| LoRA transport / route | SkillFlow `_sync_lora_to_vllm`；`batch_inference.py::{set_current_adapter_name,_resolve_model}` | `policy_sync.py::SGLangPolicyPublisher` | 必要兼容适配：SkillFlow 用 `get_peft_model_state_dict`、`MultiprocessingSerializer` 和 `/load_lora_adapter_from_tensors`；当前服务边界使用已原子落盘的 PEFT adapter 目录和 `/load_lora_adapter`。pause、model-list verification 与 route selector 对齐上游；版本化 candidate、drain、chat canary、route switch、rollback 是项目工程补全 | 单元层存在；fresh 真实闭环缺失 |
| W&B | FlowSteer `run_full_training`；SkillFlow `GFlowNetTrainer.__init__/_log_step` | `scripts/train_hotpotqa_grpo.py::WandbTracker` | 薄适配为 online fail-closed，并补 reward/loss/gradient/update/policy/publish/canary/checkpoint 字段 | adapter 存在；环境和真实 run 未验收 |
| MACE baseline | FlowSteer/SkillFlow 无 | `exploration/mace.py`、`features.py` | MD 指定项目算法实现 | primitive 存在，Phase 1 未接线/未验收 |
| Joint Bayesian posterior / paired probe | FlowSteer/SkillFlow 无 | `posterior.py`、`paired_probe.py`、`records.py::ProbeRecord` | MD 指定项目算法实现 | primitive 存在，Phase 2 未接线/未验收 |
| UCB / Thompson / EVSI | FlowSteer/SkillFlow 无 | `policies.py`、`evsi.py` | MD 指定项目算法实现 | 数值组件存在，Phase 2–3 未接线/未验收 |
| Skill lifecycle | SkillFlow `training/skill_evolution.py`、`src/skills` 仅作工作区/生命周期参考 | `skills/{schema,validator,lifecycle,retrieval,store}.py` | MD 的 paired-effect、held-out calibration、harm gate 与版本约束是必要适配；Skill reward 不进入 GRPO | primitive 存在，Phase 4 未接线/未验收 |

## 明确禁用的 SkillFlow 训练路径

本任务不得调用或移植为主目标：

- `training/gflownet_trainer.py::compute_ttb_loss`；
- `PartitionFunctionHead`；
- `training/backward_policy.py::BackwardPolicy` 与 φ-LoRA；
- `_gradient_accumulation_step` 的 TTB balance、Z/φ update；
- `r_tilde`、flow entropy、process reward 或 Skill reward 对 GRPO loss 的参与；
- `configs/skillflow.yaml` 中的 `algorithm: gflownet_ttb`。

## SkillFlow 不能直接复用的训练顺序

SkillFlow `GFlowNetTrainer.train` 会在当前 optimizer update 前异步启动下一 step rollout，可能产生旧 θ 的下一批数据。本项目必须禁用该跨 step prefetch，采用：

```text
collect(policy_n, same problem + same condition)
  -> terminal evaluator receipt
  -> Action-Masked One-Pass GRPO
  -> backward
  -> optimizer.step
  -> save theta + optimizer + scheduler + RNG + metadata
  -> pause/drain
  -> publish candidate adapter
  -> route switch + canary
  -> admit rollout(policy_n+1) with exact receipt
```

## 当前门控

- 当前 Phase：Phase 0 数据可信性，未通过。
- optimizer steps：0。
- TTB training/config launch：禁用；遗留 TTB 文件只作为 disabled method reference。
- model service / GPU / W&B：未启动。
- 单步训练：未授权。
- 长训练：未授权。
