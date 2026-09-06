# FlowSteer / SkillFlow 训练 Source Map 与协议门控

## 结论

当前代码不允许启动正式训练，也不允许运行真实 optimizer step。原因不是工程实现尚未优化，而是主训练目标尚未统一：FlowSteer 论文、FlowSteer 发布源码、项目设计文档和 SkillFlow 原始源码给出了三种不能静默等同的训练口径。

- FlowSteer 论文：带 canvas token mask 的 clipped GRPO，包括 behavior-policy ratio、PPO clipping 和 reference KL。
- FlowSteer 发布源码：基于组归一化 advantage 的 action-masked policy-gradient；未实现论文公式中的 behavior ratio、clipping、reference KL 或 entropy 项。
- 项目设计文档：同题、同 condition、同 policy version 的 Action-Masked One-Pass GRPO；terminal-only task reward；一次使用 rollout；明确不使用 clipping、reference KL、entropy 或结构奖励。
- SkillFlow：Tempered Trajectory Balance（TTB），联合优化 forward policy θ、backward policy φ 和 partition function Z。

这些目标不能放入同一个 loss、optimizer step 或实验 condition 中。现有两条实现路径均保持禁用；没有创建第三套 trainer。

## 冻结来源

| 来源 | 精确版本 | 用途 |
|---|---|---|
| FlowSteer 论文 | `FlowSteer__Towards_Agents_Designing_Agentic_Workflows_via_Reinforced_Progressive_Canvas_Editing__3_(1).pdf` | 方法定义、Eq. 12、Algorithm 1、Appendix C/G |
| FlowSteer 原始代码 | `beita6969/FlowSteer@a329b577a8e8c2f2d5492983b492473a06a8e718` | 与 arXiv v4 对齐的发布实现；后续 `origin/main` 只改 README |
| SkillFlow 论文 | `SkillFlow.pdf` | TTB、θ/φ/Z、训练配置和实验范围 |
| SkillFlow 原始代码 | `beita6969/SkillFlow@74be52bb6bd9f0e9e68dacb72636b75649197983` | Qwen3.5、SGLang、rollout、TTB、LoRA、同步与 Skill 演化的发布实现 |
| SkillFlow 本地只读 checkout | `/ssd1/iclr/owner/skillflow-iclr/06_GITHUB_RELEASE_20260816` | 上述 revision 的 commit-exact 源码 |
| 项目设计文档 | `FlowSteer_MACE_Bayesian_Skill_Design.md` | 自由 AgentGraph、Action-Masked One-Pass GRPO、信号隔离和 MACE/Bayesian/Skill 设计 |
| HotpotQA 训练起点 | `backup/hotpotqa-compliant-best-round01-20260906@740e53ec6ccac635ecbe7f1f379b002bfb2574d1` | 不可改写的已备份推理基线 |

## FlowSteer 原始调用链

```text
train_interactive.py::main
  -> load_model_and_tokenizer
  -> run_full_training
     -> InteractivePromptBuilder
     -> Director generation
     -> first-action truncation
     -> InteractiveWorkflowEnv.step
        -> ActionParser
        -> WorkflowGraph update
        -> AFlowExecutor / fixed Executor
     -> TurnRecord / Trajectory
     -> correctness evaluator
     -> TrajectoryRewardCalculator
     -> compute_grpo_loss / backward
     -> optimizer.step / scheduler.step
     -> LoRA save + vLLM reload
     -> W&B log / checkpoint
```

## 模块级 source map

| 训练模块 | 首要原始实现 | 当前项目接线 | 分类 | 启动状态 |
|---|---|---|---|---|
| Flow-Director generation | FlowSteer `train_interactive.py::create_generate_fn`; `src/interactive/prompt_templates.py` | `src/interactive/director.py` | 薄适配：XML operator action 改为 AgentGraph atomic action | 已实现，训练禁用 |
| Progressive Canvas Editing | FlowSteer `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step`; `action_parser.py`; `workflow_graph.py` | `agent_workflow_env.py`; `agent_action_parser.py`; `agent_graph.py` | 必要薄适配：保留每轮一次原子编辑与 execution feedback，将有序 operator DSL 改为自由 AgentGraph | 已实现 |
| AgentGraph execution | FlowSteer `workflow_builder.py::create_aflow_executor_wrapper`; `aflow_executor.py` | `agent_runtime.py` | 必要适配：固定 operator executor 改为异构 Agent、显式 relation 与 communication | 已实现 |
| Rollout collection | FlowSteer `train_interactive.py::run_full_training`; `workflow_builder.py::Trajectory` | `rollout_collector.py`; `records.py` | 薄适配：增加 exact prompt/output IDs、behavior log-prob、condition、policy/adapter/server version 和 evaluator receipt | 已实现 |
| Terminal task reward | FlowSteer `trajectory_reward.py::TrajectoryRewardCalculator`; SkillFlow `training/reward.py` | `trajectory_reward.py` 与 benchmark evaluator | 项目必要适配：按 MD 只使用 evaluator-valid terminal task reward；不复用 FlowSteer 结构奖励 | 已实现基础层 |
| Action-Masked One-Pass GRPO | FlowSteer 发布源码 `train_interactive.py::compute_grpo_loss`; MD 第 4、11、12 节 | `grpo_objective.py::action_masked_one_pass_loss` | 项目必要适配；是候选主目标，不是 SkillFlow TTB | 未选择、禁用 |
| TTB objective | SkillFlow `training/gflownet_trainer.py::compute_ttb_loss` 与 `_gradient_accumulation_step`; `training/flow_metrics.py::edge_logprob_tilde` | 尚缺 AgentGraph trajectory adapter | 原始算法直接复用 + 最小 schema adapter；是另一候选主目标 | 未选择、未接线 |
| θ-LoRA | SkillFlow `GFlowNetTrainer.setup` 与 `_setup_optimizers` | 当前 Director LoRA loader / optimizer | 若选 TTB，复用 SkillFlow θ 配置；若选 GRPO，只训练 θ | 待方法选择 |
| φ-LoRA / backward policy | SkillFlow `training/backward_policy.py::BackwardPolicy` | 无 AgentGraph TTB 接线 | 仅 TTB 需要，不能加入 GRPO optimizer | 未接线 |
| Partition function Z | SkillFlow `PartitionFunctionHead`; `_compute_partition_function` | 无 AgentGraph TTB 接线 | 仅 TTB 需要，不能加入 GRPO optimizer | 未接线 |
| Qwen3.5 Supervisor service | SkillFlow `training/sglang_manager.py::SGLangSupervisorManager` | 项目 SGLang Supervisor 配置 | 直接复用启动参数语义，按本机资源薄适配 | 未启动 |
| LoRA publication | SkillFlow `gflownet_trainer.py::_sync_lora_to_vllm`; `batch_inference.py` adapter routing | `policy_sync.py`; `rollout_collector.py::RolloutGate` | 复用 tensor transport；版本化 candidate、drain、canary、fail-closed route 是项目工程补全 | 基础层已实现，未验收 |
| Parallel rollout | SkillFlow `_collect_episodes` | 候选 rollout worker pool | 可复用同批并发；禁止跨 optimizer step 的 async prefetch | 未启动 |
| Checkpoint / resume | FlowSteer `run_full_training`; SkillFlow `_save_checkpoint` / `resume` | 项目 versioned checkpoint contract | 薄适配：补 optimizer、φ、Z、step、RNG 和 publication receipt | 尚未完成 TTB 路径 |
| W&B | FlowSteer `run_full_training`; SkillFlow `GFlowNetTrainer.__init__` / `_log_step` | 项目 fail-closed tracking contract | 必要薄适配：补 policy/adapter version、TTB residual、gradient/update、sync/canary/checkpoint 字段 | 环境前置未通过 |
| MACE / Bayesian posterior / EVSI / Skill publication | 项目设计文档第 5–20 节 | 现有预留模块 | 项目算法新增，不属于 FlowSteer 或 SkillFlow 的直接复现 | 未接入本次训练 |

## 必须先解决的冲突

### 1. 主目标函数

- 设计文档要求 Action-Masked One-Pass GRPO。
- SkillFlow 主方法要求 TTB，并联合训练 θ、φ、Z。
- 二者的 probability factorization、trainable parameters、reward transformation 和 gradient 都不同。

可选的最小方案只有：

1. 选择 FlowSteer/MD one-pass GRPO 为主训练；SkillFlow 只提供 Qwen3.5、SGLang、LoRA、并发 rollout 和权重发布机制。
2. 选择 SkillFlow TTB 为主训练；明确标记为 HotpotQA 单数据集项目适配，不称为 MD 主训练。
3. 建立两个完全隔离、版本化的实验 condition，分别训练和比较；不共享 rollout batch，不混合 loss。

### 2. FlowSteer 论文与发布源码的 GRPO 不一致

论文 Eq. 12 包含 behavior ratio、PPO clipping 和 reference KL；发布源码 `compute_grpo_loss` 只计算 `-advantage * mean(current-policy log-prob)`。发布代码还按 dataset source 而不是同一 question 分组。设计文档选择的是严格同题、同 condition、同 policy version 的 one-pass group policy-gradient，并明确不称为 clipped GRPO。

因此，即使选择 GRPO，也必须进一步确认采用设计文档口径，而不是把发布代码实际 loss 或论文 Eq. 12 静默改名。

### 3. TTB 的 HotpotQA scalar reward

SkillFlow 论文对 multi-hop QA 描述 EM；原始 `training/reward.py` 将 `multi_hop_qa` 映射为 token F1。若选择 TTB，必须决定 terminal scalar 使用 official-compatible EM 还是 token F1。二者可同时记录为指标，但只能冻结一个进入 TTB residual。

### 4. action-token 定义

- MD GRPO mask：Canvas 实际消费的首个 atomic action 之前由 Director 采样的全部有效前缀可以获得 credit；environment feedback 只作为下一轮 context。
- SkillFlow TTB：`<think>` 被移入 context，只对结构化 action tail 求 edge log-prob，并按 action-token 数取边内平均。

两种 token contract 不能共享同一个 mask 名称。若选择 TTB，只能增加 exact receipt 到 SkillFlow edge schema 的薄 adapter。

### 5. 严格 on-policy 顺序

SkillFlow 原始 `train()` 会在当前 step optimizer update 前预取下一 step rollout。该实现可提高吞吐，但下一批可能使用旧 θ，不满足本项目“每个 optimizer step 后发布新权重，再允许下一批 rollout”的硬约束。

最小兼容顺序必须是：

```text
collect(theta_n)
  -> reward
  -> selected loss
  -> backward
  -> optimizer.step
  -> checkpoint theta_(n+1)
  -> publish/sync/canary theta_(n+1)
  -> admit next rollout with theta_(n+1) receipt
```

这要求禁用跨 step prefetch。publication barrier、version receipt 和 canary 属于项目工程补全，不能归因于 SkillFlow 论文。

### 6. 训练步数与配置口径

SkillFlow 论文主跑为 250 optimizer steps；原始发布配置写 300。用户给出的目标区间是 250–300，但正式 run manifest 必须冻结一个具体值，不能在运行后按曲线选择并称为预先设定。

## 单步验收门

用户选定协议后，只允许先执行一个真实 optimizer step，并随后执行一个使用新权重的 rollout/canary。验收必须同时满足：

1. rollout 由冻结且唯一的 behavior policy version 生成；
2. terminal reward 来自正式 evaluator receipt；
3. 所选 loss 与 manifest 完全一致；
4. `backward()` 与 `optimizer.step()` 真实执行；
5. 所选 trainable state 的 gradient norm 和 parameter delta 非零；
6. 更新后的 adapter 成功发布，server route 和 canary receipt 指向新版本；
7. 下一次 rollout 的 behavior-policy receipt 明确使用新版本；
8. W&B 在线记录完整 step、reward/loss、trajectory、GPU、gradient/update、checkpoint、publish/canary 字段。

若选择 GRPO，验收对象只有 θ；若选择 TTB，验收对象必须同时包括 θ、φ、Z。没有通过该门，不允许进入 250 或 300 step。

## 当前状态

- optimizer steps：0
- model service：未启动
- GPU task：未启动
- W&B run：未创建
- MACE / Bayesian posterior / EVSI / automatic Skill publication：未接入
- 长训：禁止启动
- 唯一下一步：由用户统一主目标函数、TTB reward scalar（仅在选择 TTB 时）及目标步数

