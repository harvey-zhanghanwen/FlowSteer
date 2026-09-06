# TriviaQA v16：FlowSteer / SkillFlow 训练 source map 与方法门禁

本文件基于两篇论文全文、两套发布源码和项目设计文档逐项核对。它是训练前
的非执行性协议，不是新的训练器，也不表示训练已经开始。

## 1. 固定起点与当前状态

- 架构起点：`backup/triviaqa-fact-memory-v16-best-82p81-20260906`
- 固定提交：`114bd8b6378af55fbd1f40cdfdda91c79314a771`
- 独立训练分支：`train/triviaqa-v16-skillflow-300step-20260906`
- 当前状态：`blocked_method_conflict`
- 已完成 rollout / backward / optimizer.step / adapter publish / post-update
  rollout / W&B run：均为 **0**。

不得覆盖起点备份；任何真实 1-step 必须在本独立分支上进行。TriviaQA-only
训练是项目适配：FlowSteer 论文只把 TriviaQA 用作 OOD 评测，SkillFlow 主实验
是七个 IID 任务联合训练，因此本任务不能标为任一论文的 bit-exact reproduction。

## 2. 已核对的原始来源

| 来源 | 固定位置 / revision | 核对结论 |
| --- | --- | --- |
| FlowSteer 论文 | 用户附件 `a58ade6e-...pdf` | 全文已读；主目标是 canvas-masked clipped GRPO |
| FlowSteer 发布代码 | `/ssd1/iclr/icassp/code/FlowSteer` @ `1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9` | 训练核心自 `a329b577...`（arXiv v4 alignment）后未变化 |
| SkillFlow 论文 | 用户附件 `402fe9a0-.../SkillFlow.pdf` | 全文已读；正式主目标是 TTB，GRPO 仅为 baseline / ablation |
| SkillFlow 发布代码 | `/ssd1/iclr/2/SkillFlow` @ `74be52bb6bd9f0e9e68dacb72636b75649197983` | 已追踪正式入口、TTB、SGLang、LoRA、更新、同步和 checkpoint |
| 项目设计文档 | 用户附件 `2ab716b4-.../FlowSteer_MACE_Bayesian_Skill_Design.md` | 主目标明确为 terminal-only Action-Masked One-Pass GRPO；MACE/Bayesian/EVSI/Skill 闭环仍未实现 |

## 3. FlowSteer 发布代码真实调用链

| 模块 | 原始文件 / 类 / 函数 | 结论 |
| --- | --- | --- |
| 正式训练入口 | `train_interactive.py:3031 main` → `load_model_and_tokenizer:312` → `run_full_training:1120` | Qwen3-8B + PEFT LoRA |
| Flow-Director prompt | `src/interactive/prompt_templates.py:12 SYSTEM_PROMPT_TRAINING`; `InteractivePromptBuilder:200` | 发布 prompt 比论文 Table 2 长，并带复杂 workflow 先验；不能直接视为项目要求的简洁中性 prompt |
| Structured action | `src/interactive/action_parser.py:67 ActionParser`; `parse:110` | 可直接复用 action parsing 边界 |
| Workflow Canvas | `src/interactive/workflow_graph.py:128 WorkflowGraph`; `add_operator:170`, `add_parallel:209`, `add_conditional:271`, `add_loop:341` | 实际是顶层顺序列表加嵌套 control blocks，不是一般任意边 DAG |
| Progressive Canvas Editing | `src/interactive/workflow_env.py:57 InteractiveWorkflowEnv`; `step:405`, `_step_internal:433`, `_handle_add:628`, `_handle_prompt_input:796` | 两态 FSM：ADD 后请求 prompt；完整 edit 单元结束后执行当前 Canvas 并返回 feedback |
| Workflow execution | `workflow_env._execute_workflow:1876` → `workflow_builder.create_aflow_executor_wrapper:578` → `AFlowExecutor.execute_workflow:203` | `execute_each_step=true`，已完成 edit 单元后执行；node cache 支持增量复用 |
| Rollout / trajectory | `workflow_builder.py:12 TurnRecord`, `:27 Trajectory`, `:127 merge_trajectory_masks`; vectorized loop `train_interactive.py:1452-1672` | Director generation → Canvas step → execution feedback → continuation |
| Terminal correctness | `train_interactive.py:1626-1669` → `_compute_correctness` | 最终 evaluator 边界可复用 |
| Reward | `src/interactive/trajectory_reward.py:61 TrajectoryRewardCalculator`; `compute_reward:142-225` | `-1 + structure + gated correctness`；结构满分才释放 correctness |
| Loss / backward | `train_interactive.compute_grpo_loss:2052-2163`; `sample_loss.backward:2132` | 发布代码实际是一次 `-advantage * mean(masked current log-prob)` |
| Optimizer / LoRA sync | `train_interactive.py:1860-1875`; sync helpers `:114-194` | 更新后 unload/load；失败可继续 base model，非 fail-closed |
| Checkpoint | `train_interactive.py:2012-2027` | adapter/tokenizer/optimizer/scheduler/RNG 基础状态 |
| W&B | `train_interactive.py:1311-1340,1912-2008` | 默认关闭，初始化失败继续运行 |

### FlowSteer 论文与发布代码差异

FlowSteer 论文 Eq. 12 是带 behavior ratio、clipping、reference KL 的
canvas-masked GRPO，附录配置还列出 entropy coefficient；发布代码的正式
`compute_grpo_loss` 没有 `pi_old`、importance ratio、clipping、reference KL
或 entropy。配置虽然声明这些字段，loss 并未消费它们。

发布代码还存在四个不能忽略的差异：

1. 每题只采一条 trajectory，advantage 按 dataset/source 内不同问题分组，
   而不是论文要求的同一问题多 trajectory；
2. loss 重放只拼 `model_response + feedback`，没有保存完整真实 system/task/
   state prompt，因此不能精确恢复 behavior-policy conditional log-prob；
3. mask 覆盖完整 Director response（reasoning 和 action），但不会裁到 Canvas
   实际消费的首个原子动作边界；
4. LoRA 同步失败后可能继续以 base model rollout，没有严格 policy-version
   barrier。

所以发布脚本可以作为实现来源，但不能原样声称复现论文 Eq. 12。

## 4. SkillFlow 发布代码真实调用链

| 模块 | 原始文件 / 类 / 函数 | 结论 |
| --- | --- | --- |
| 正式入口 | `run_training.py:138-253` → `GFlowNetTrainer.setup()` → `train()` | 可直接复用入口骨架 |
| Qwen3.5 / named LoRA | `training/gflownet_trainer.py:256-353` | theta/phi 共用 base，以 named PEFT adapters 训练 |
| theta LoRA | `gflownet_trainer.py:278-286` | rank 64 / alpha 128 / q,k,v,o_proj |
| phi LoRA | `gflownet_trainer.py:289-298`; `training/backward_policy.py:18-59` | 发布配置 rank 16 / alpha 32 / q,v_proj；论文 Appendix L.3 另写 rank 32，歧义保留 |
| task-conditioned Z | `gflownet_trainer.py:113-135,342-346,1605-1636` | 具体 embedding + linear head 是发布实现细节 |
| frozen Executor | `src/executor/m_exec.py:24-114`; setup `gflownet_trainer.py:356-361` | API Executor 不进 optimizer |
| SGLang Supervisor | `gflownet_trainer.py:492-511`; `training/sglang_manager.py:214-231`; `training/batch_inference.py:591-702` | GPU ID、memory fraction、context、worker 数是发布配置，不是论文规定 |
| 7 x 4 rollout | `_sample_batch_for_step:429-464`; `_collect_episodes:639-684`; `_run_episode:686-800` | 7 questions × 4 trajectories，最大 12 action edges |
| action-token split | `training/trajectory.py:16-33`; `_prepare_turn_tokens:978-1004`; forward `:1529-1596` | 只有 structured action JSON tokens 进入 PF/PB；reasoning 只作 context |
| TTB | `_gradient_accumulation_step:1295-1446`; `training/flow_metrics.py:11-18` | `(delta/T)^2`，每边 action-token log-prob 取均值；theta/phi/Z 都更新 |
| Reward | `training/environment.py:451-508`; `training/reward.py:778-818` | outcome-only 模式，`R_tilde=max(R+epsilon_min, epsilon_min)` |
| Optimizers | `_setup_optimizers:1663-1702`; backward/step `:1295-1446` | 三个 AdamW；发布代码的 phi/Z LR multiplier 与论文主表不一致 |
| Multi-GPU gradient | `gflownet_trainer.py:318-339,1063-1169` | 手工 replica、token item 二分和梯度平均；不是论文规定的 TP/DP/ZeRO |
| KL reference | `gflownet_trainer.py:1006-1061,1171-1293` | adapter-disabled reference path |
| theta publish | `_sync_lora_to_vllm:2545-2730`; pause event `training/batch_inference.py:21-63` | tensor hot-load/unload/models verification 是发布代码工程实现 |
| Checkpoint / resume | `_save_checkpoint:2516-2543`; `resume:2744-2773` | checkpoint 缺 optimizer/RNG/current-step；resume 不恢复 phi/optimizer |
| W&B | `gflownet_trainer.py:242-254,2412-2433` | fail-open，且字段不足以满足本项目验收 |

### SkillFlow 严格 on-policy 顺序差异

发布代码 `gflownet_trainer.py:522-578` 的顺序是：收集当前 rollout，先同步
当前 theta，异步启动下一 step rollout，然后才执行当前 backward 和
`optimizer.step`。因此下一批可能使用旧 theta。论文只说明每 step 有 LoRA
hot-swap，没有规定允许一阶 staleness、barrier 或 route switch。

用户要求的严格顺序必须关闭跨 step `_next_future`：

```text
rollout(vN) → reward → selected loss → backward → optimizer.step
→ publish/verify theta(vN+1) → canary → rollout(vN+1)
```

这是对 SkillFlow 发布运行时的必要薄适配；version barrier、transaction receipt、
canary 和 fail-closed recovery 必须标为本项目工程补全，不能冒充论文原文。

## 5. 三方方法冲突（不得混 loss）

| 维度 | FlowSteer 论文 | FlowSteer 发布代码 | 项目设计 MD | SkillFlow 主方法 |
| --- | --- | --- | --- | --- |
| Objective | canvas-masked clipped GRPO | 一次 current-log-prob group policy loss；缺 ratio/clip/KL | Action-Masked One-Pass GRPO | TTB squared residual |
| Token support | Director 生成 token；Canvas feedback mask=0 | 完整 response mask=1，feedback=0 | 仅真实 prompt 下 Canvas 消费到首个原子动作的采样前缀；可含 think | structured action JSON only；reasoning=context |
| Reward | `-1 + structural diversity + gated answer` | 同左的近似硬编码实现 | terminal task reward only；禁止结构/format/Skill reward | outcome reward 后作正值变换供 TTB |
| Grouping | 同题多 trajectory | 按 dataset/source 的不同问题 | 同 `(task, condition, policy_version)` | 7题×4轨迹；TTB 逐轨迹回归，不用 group advantage |
| Trainable parameters | Director theta | Director theta | Director theta | theta + backward phi + Z |
| Main profile | 300 steps, batch36, T=20, LR1e-5 | 配置同论文但 loss 未完全接线 | 数值未锁死 | 250 steps, batch28, T=12, LR1e-4 |

因此不存在一个可以同时被称为“严格 FlowSteer”“严格 SkillFlow”和“严格 MD”
的单一目标。把 GRPO advantage 项和 TTB residual 同一步相加会构造第三种训练
算法，已明确禁止。

## 6. 待用户统一决定的互斥路径

### A. FlowSteer 论文主方法

保留 FlowSteer Canvas/rollout/reward 定义，实现论文缺失的 `pi_old/pi_ref`、
same-question rollouts、ratio、clip、KL/entropy 后训练 theta。这最接近 FlowSteer
论文，但不符合 MD 的 terminal-only one-pass objective，也不是发布代码原样运行。

### B. 项目设计 MD 主方法

复用项目现有 `src/interactive/grpo_objective.py` 和
`src/interactive/smoke_trainer.py` 的 exact-group Action-Masked One-Pass GRPO；
直接复用 FlowSteer progressive Canvas/terminal evaluator 边界；从 SkillFlow
薄适配 Qwen3.5 named LoRA、SGLang 和更新后权重发布。phi/Z/TTB 禁用。
这符合设计 MD，但不能称为 SkillFlow 正式主训练或 clipped GRPO。

### C. SkillFlow 正式 TTB

直接复用 `GFlowNetTrainer`、BackwardPolicy、Z、TTB、7×4 sampler 和
theta/phi named LoRA；只把 FlowSteer Canvas 原子 edit 映射成 SkillFlow
structured action，把 TriviaQA terminal evaluator 映射成 outcome reward。
GRPO 完全禁用。这最接近 SkillFlow 主方法，但会覆盖 MD 指定的主 objective，
必须得到明确决定。

如需先比较 B 与 C，必须是两个独立 entrypoint、optimizer、checkpoint、W&B
run 和 branch，各自只做 1-step；不能顺序混训、共享 optimizer 或合并 loss。
这是一项对照决策协议，不是第三个 objective。

## 7. 模块复用分类

| 项目模块 | 分类 | 依据 / 当前状态 |
| --- | --- | --- |
| ActionParser、Canvas FSM、每个完整 edit 单元后的 execution feedback | 直接复用 FlowSteer | 发布代码真实调用链已核对 |
| Structured parallel/conditional/loop blocks | 直接复用 FlowSteer | 任意自由 relation/DAG 不属于原实现能力 |
| TurnRecord/Trajectory 与 terminal evaluator 边界 | 直接复用 FlowSteer | exact context/token receipt 仍需薄适配 |
| Qwen3.5 model loading、named theta/phi LoRA | 直接复用 SkillFlow（按所选方法取子集） | B 只使用 theta；C 使用 theta/phi/Z |
| SGLang tensor load、pause primitive、model-list verification | 直接复用 SkillFlow primitives | 严格 step barrier 与 canary 是项目补全 |
| AgentGraph action → exact training token/context receipt | 必要薄适配 | MD mask 与 SkillFlow action-only mask 互斥，需先选方法 |
| `optimizer.step → publish → next rollout` 顺序 | 必要薄适配 | 关闭 SkillFlow 跨 step prefetch；sync failure fail-closed |
| policy/adapter version、parameter-delta、transaction/canary receipts | 项目工程新增 | 用户验收要求；非论文算法声明 |
| required/fail-closed W&B 字段 | 项目工程新增 | 两套发布代码都是 optional/fail-open |
| MACE、联合贝叶斯后验、EVSI、paired intervention、Skill 自动发布/撤销 | 尚未实现 | 设计 MD 明确如此；不得报告已运行 |

当前 `src/interactive/policy_sync.py` 的 transaction/route/canary 不能归类为
SkillFlow 直接复用。它使用 SkillFlow 的 tensor-load/pause 思路，但发布 revision
中不存在该文件注释曾引用的 `training/external_sglang.py` 和
`runtime/sglang_gateway.py`；正确上游位置是
`training/gflownet_trainer.py::_sync_lora_to_vllm` 与
`training/batch_inference.py`。transaction、route switch 和 canary 属于项目补全。

## 8. 真实 1-step 验收门禁

只有用户明确选择 A、B 或 C 后，才能把对应 loss、token mask、trainable
parameters 和超参数冻结到独立 executable config。随后还必须同时满足：

1. 只读核对 GPU 进程并获得不冲突的可见 CUDA allocation；
2. W&B client 与 online authentication 可用，初始化失败即停止；
3. 真实训练数据、evaluator 和 trajectory receipt 可用；
4. rollout 使用 `policy_vN` 并记录 exact context/token/policy version；
5. 真实 reward 进入所选且唯一的 loss；
6. `backward`、`optimizer.step`、非零 gradient 和非零 parameter delta 均有证据；
7. 新 theta 发布、校验和 canary 成功；
8. 下一次 rollout 的 behavior policy 明确为 `policy_vN+1`；
9. W&B 同步记录 step、policy/adapter version、loss、reward、trajectory length、
   valid/filtered rollout、GPU/throughput/error、gradient/update norm、checkpoint
   与 Skill phase 状态。

当前任务 namespace 没有可见 CUDA device，且 W&B client/auth 尚不可用；这些是
方法选择之后仍需解除的资源门禁。未通过上述 1-step 前，250–300 step 长训
保持禁止。
