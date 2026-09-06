# MBPP+ 训练源码映射与目标函数冲突记录

## 当前状态

- 不可变训练起点：`backup/mbppplus-best-v6-20260906` @
  `66e66f02c908d16d097ceb75d65745d90a155d29`。
- 当前状态：`blocked_method_conflict`。
- optimizer update：0；尚未启动模型、rollout、W&B run 或训练。
- 启动约束：`config/training_mbppplus_method_decision_v1.yaml` 保持
  `method_decision.status=unresolved`。在主目标函数被明确接受之前，真实
  Step-1 和长训均被拒绝；长训还必须等待一个已接受的真实 Step-1 receipt。

本文件记录训练模块的真实来源和适配边界。它不把 TTB 与 GRPO 合成第三种
目标函数，也不把项目工程代码写成论文原始实现。

## 已核对的原始材料

| 材料 | 本次核对版本/位置 |
|---|---|
| 项目设计文档 | `FlowSteer_MACE_Bayesian_Skill_Design.md`，本次附件完整阅读 |
| FlowSteer 论文 | 本次附件完整阅读；策略目标见论文 Eq. 12，交互状态见 §3 |
| FlowSteer 原始代码 | `origin/main@1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9`；项目导入的最后一个代码对齐提交为 `a329b577a8e8c2f2d5492983b492473a06a8e718` |
| SkillFlow 论文 | 本次附件完整阅读；TTB 见 §4 与 Appendix B，主配置见 Appendix P |
| SkillFlow 原始代码 | `/ssd1/iclr/2/SkillFlow` @ `74be52bb6bd9f0e9e68dacb72636b75649197983` |

## FlowSteer 原始训练调用链

正式发布代码的主入口是 `train_interactive.py::run_full_training`
（1120–2050），不是 `src/interactive/grpo_trainer.py`：

```text
load_model_and_tokenizer
→ initial LoRA sync
→ InteractiveWorkflowEnv
→ Director generate
→ env.step
→ ActionParser.parse
→ Canvas edit
→ execute current workflow
→ execution feedback
→ TurnRecord / Trajectory
→ terminal evaluator
→ TrajectoryRewardCalculator
→ compute_grpo_loss (内部 backward)
→ gradient clipping
→ optimizer.step
→ LoRA sync
→ checkpoint / W&B
```

逐步编辑和执行的来源是
`src/interactive/workflow_env.py::InteractiveWorkflowEnv.step/_step_internal`
（405–627）。原始 action parser 位于
`src/interactive/action_parser.py::ActionParser.parse`（110–178）；基础 trajectory
和 mask 位于 `src/interactive/workflow_builder.py::TurnRecord/Trajectory`
（12–100）及 `create_action_mask`（106–124）。

原始 `train_interactive.py::compute_grpo_loss`（2052–2163）实际计算
`-advantage × masked mean current-policy log-prob`，没有 old-policy ratio、PPO
clipping、reference KL 或 entropy；主循环也按 source 而非同一题分组。因此它
既不是论文 Eq. 12 的完整 clipped GRPO，也不满足设计文档的同题同条件分组，
不能直接标为正式目标函数实现。

## SkillFlow 原始训练调用链

入口是 `run_training.py::main`（138–253）：

```text
GFlowNetTrainer.setup
→ collect on-policy episodes
→ fill forward/backward action-token log-probabilities
→ _gradient_accumulation_step
→ theta optimizer.step + Z optimizer.step + phi optimizer.step
→ local replica sync
→ LoRA publication / checkpoint / W&B
```

关键原始符号如下：

| 边界 | SkillFlow 原始实现 |
|---|---|
| Qwen3.5-9B 与 named LoRA | `training/gflownet_trainer.py::GFlowNetTrainer.setup`（256–419） |
| TTB residual | `training/gflownet_trainer.py::compute_ttb_loss`（138–152） |
| 正式 train-loop 更新路径 | `GFlowNetTrainer._gradient_accumulation_step`（1295–1446） |
| reasoning/action 分离 | `training/trajectory.py::split_think_and_action`（16–33） |
| edge/action-token normalization | `training/flow_metrics.py::edge_logprob_tilde/effective_paper_steps`（11–31） |
| 并行 rollout | `GFlowNetTrainer._collect_episodes`（639–684） |
| SGLang Supervisor | `training/sglang_manager.py::SGLangSupervisorManager`（50–255） |
| adapter route pause | `training/batch_inference.py::_wait_if_paused/get_current_adapter_name`（21–63） |
| LoRA hot-swap | `GFlowNetTrainer._sync_lora_to_vllm`（2545–2730） |
| MBPP public-test reward | `training/reward.py::code_test_pass_rate`（380–405） |
| checkpoint | `GFlowNetTrainer._save_checkpoint`（2516–2543）及 `resume`（2744–2773） |

SkillFlow 原始 `train()` 在第 N 步 optimizer update 之前异步启动第 N+1 批
rollout。其 pause flag 只阻止新请求，不等待在途请求排空；adapter 名固定为
`theta_live`，也没有严格的 behavior-policy/version receipt。因此原始顺序不能
直接满足“更新后的权重必须用于下一批 rollout”。关闭跨步预取、请求排空、唯一
adapter/policy version、canary 和 receipt 均属于必要工程适配。

## 逐模块 source map

| 当前模块/边界 | 首要原始来源 | 分类 | 说明 |
|---|---|---|---|
| Progressive Canvas Editing | FlowSteer `workflow_env.py::step/_step_internal` | 直接复用语义，薄适配数据结构 | 保留 action→edit→execute→feedback；自由 AgentGraph 替代旧 operator DSL。 |
| 自由 AgentGraph | 设计文档 `Agent = agent_id + model_id + free-text contract` | 项目必要适配 | FlowSteer 原始 `WorkflowGraph` 是有序/嵌套 DSL，不支持任意 relation 与每节点模型。 |
| AgentGraph rollout/trajectory | FlowSteer `workflow_builder.py`、`run_full_training` | 必要薄适配 | 补 task/condition/policy version、模型、relation、artifact、tool/evaluator receipt。 |
| terminal task reward | FlowSteer evaluator 边界；SkillFlow `code_test_pass_rate` | 必要数据集适配 | MBPP+ 只使用公开测试执行所得终局 reward；不加入结构奖励、MACE 或 Skill reward。 |
| Action-Masked One-Pass GRPO | 设计文档 §4；FlowSteer action-mask 边界 | 项目方法适配 | `src/interactive/grpo_objective.py` 实现同题同 condition/policy group；不是 FlowSteer 论文 Eq. 12 的 clipped GRPO。尚未通过真实 GPU Step-1。 |
| TTB 数学与 action-only target | SkillFlow `trajectory.py`、`flow_metrics.py`、`gflownet_trainer.py` | 必要薄适配 | `src/interactive/ttb_objective.py`/`ttb_trainer.py` 是本项目对 AgentGraph receipt 的重实现，不是直接调用上游 trainer。 |
| theta/phi/Z learner | SkillFlow `GFlowNetTrainer.setup/_gradient_accumulation_step` | 当前为项目重实现 | 当前 `ttb_trainer.py` 约 2k 行；若选择 TTB，应进一步收缩到上游调用或逐函数薄适配。 |
| Qwen3.5-9B / PEFT profiles | SkillFlow `GFlowNetTrainer.setup` | 配置语义复用 | theta r64/a128 q/k/v/o；phi 主表 r16/a32 q/v。论文 Appendix L 的 phi rank32 歧义必须保留。 |
| SGLang Supervisor lifecycle | SkillFlow `training/sglang_manager.py` | 必要薄适配 | `src/interactive/sglang_manager.py` 保留 spawned-process boundary；GPU 映射与 readiness 是项目配置。 |
| LoRA publication | SkillFlow `_sync_lora_to_vllm` 与 `batch_inference.py` | 项目工程适配 | 当前 `policy_sync.py` 的 drain、candidate、canary、route switch、rollback 和 receipt 超出上游实现，不得标为直接复用。 |
| 多 GPU backward | SkillFlow `_batched_logprob_backward`/replica 路径 | 项目工程适配 | 上游无 DDP/FSDP/ZeRO/TP/DP 实现；三卡角色和 token-cost partition 不是论文规定。 |
| checkpoint/resume | SkillFlow `_save_checkpoint/resume` | 必要扩展 | 上游 resume 未完整恢复 phi、optimizer、step 与 RNG；事务 checkpoint 是项目工程。 |
| W&B | SkillFlow 初始化及 `wandb.log` | 必要扩展 | policy/adapter version、gradient/update delta、filtered rollout、GPU、publish/canary/checkpoint receipt 是项目新增监测字段。 |
| MACE / Bayesian posterior / EVSI / Skill evolution | 设计文档规划章节 | 尚未实现且本轮禁用 | 不得写成已训练或已发布。 |

## 必须显式裁决的冲突

| 项目 | Action-Masked One-Pass GRPO | SkillFlow TTB |
|---|---|---|
| 训练目标 | 同题同条件组内终局 reward advantage 的 one-pass group policy gradient | `(log Z + log P_F - beta log r_tilde - log P_B)^2 / T^2` 回归 |
| token target | Canvas 实际消费的 Director 前缀；可包含 reasoning/think 与首个 atomic action | 仅 structured action tokens；reasoning 只作 context |
| 可训练参数 | Director theta LoRA | theta LoRA、backward-policy phi LoRA、Z |
| reward 使用 | 原始 terminal task reward | 经 epsilon 下界形成正的 `r_tilde` |
| 分组 | 相同 task、condition、policy version 的多条 trajectory | 7 questions × 4 trajectories 的 TTB batch |
| 本项目权威来源 | 设计文档主线，借用 FlowSteer Canvas/trajectory | SkillFlow 正式主方法 |

两者不能共用 token mask，不能联合成一个 loss，也不能把 theta/phi/Z 更新称为
GRPO。FlowSteer 论文的 clipped/KL objective、FlowSteer 发布代码的一次
REINFORCE-style update、设计文档的 Action-Masked One-Pass GRPO 也必须分别
命名，不能互相替代。

## 两条互斥方案

### A. 设计文档 / FlowSteer 主线（推荐）

- FlowSteer Progressive Canvas Editing、AgentGraph rollout、execution feedback
  和 terminal evaluator；
- 设计文档 Action-Masked One-Pass GRPO，仅更新 Director theta LoRA；
- SkillFlow 的 Qwen3.5-9B、PEFT 与 SGLang lifecycle 作为服务层来源；
- 禁用 TTB、phi、Z、MACE、Bayesian posterior 和 Skill evolution。

这最符合当前项目设计文档，但应表述为“FlowSteer 原始边界 + 设计文档必要
适配 + SkillFlow 服务层薄适配”，不是任一论文的 bit-exact reproduction。

### B. SkillFlow TTB 独立实验条件

- reasoning 仅作 context，structured action-only target；
- 联合更新 theta、phi 和 Z；
- 禁用 GRPO；MBPP+ 明确是 dataset adaptation；
- 关闭上游跨步 prefetch，并在 update→publish→drain/version/canary→next rollout
  顺序下验收严格 on-policy receipt。

该方案可保留为独立对照，但不能静默取代设计文档主线。

## 启动与验收边界

在用户明确接受 A 或 B 之前，状态必须保持 `blocked_method_conflict`。选择后也
只能先运行一个真实 Step-1，完整证明：

```text
on-policy rollout
→ official evaluator reward
→ selected objective
→ backward
→ optimizer.step
→ non-zero parameter delta
→ versioned adapter publication
→ updated-policy canary/rollout
→ W&B receipt
```

只有上述 receipt 被接受后，`long_run_allowed` 才能设为 true。约 20 分钟/step
和 100 多步收敛只是用户预期；W&B 曲线和 held-out evaluator 才是实际证据。
