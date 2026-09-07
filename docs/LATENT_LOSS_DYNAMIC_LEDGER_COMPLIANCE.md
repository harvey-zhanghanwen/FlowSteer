# LatentLoss 动态 Combination Posterior 合规矩阵

## 规范优先级与当前状态

- 任务学习目标仍为 `FlowSteer_MACE_Bayesian_Skill_Design.md` 指定的
  terminal-only、same-problem/same-condition、Action-Masked One-Pass GRPO。
- `LatentLoss_Implementation_Spec.md` 在账本、潜在损失、配对探针和 Skill
  生命周期存在冲突时优先。它没有新增可微损失；本文所称 latent loss 是写入
  Director observation 的 posterior risk readout。
- SkillFlow 的 Tempered Trajectory Balance（TTB）、backward policy φ 和
  partition function Z 均禁用，不能和 GRPO 混合。
- 当前版本化配置为
  `config/training_hotpotqa_dynamic_ledger_grpo.yaml`。该配置保持
  `experiment.training_enabled=false` 和 `gpu.training_enabled=false`。
- HotpotQA 原有 Phase 0 数据可信性 receipt 已存在；新规格的 Phase A–E 尚未
  形成完整运行 receipt。尤其 Phase E 要求的 50 questions × 4 natural
  trajectories 没有运行，因此 optimizer step 尚未获准，W&B run 也没有启动。

“代码符号存在”或“单元测试通过”不能替代 Phase acceptance receipt；只有 runner
验证全部门禁后才允许产生第一个 optimizer step。

## 冲突决议

| 冲突点 | 本配置采用的边界 |
|---|---|
| MD 的静态 MACE / joint Bayesian posterior / posterior UCB / particle EVSI 与新规格的动态账本 | 当前 profile 采用新规格的 dynamic Combination Posterior、straddle probe selection 和 posterior latent risk。旧 `MACE`、旧 joint posterior 及 particle EVSI 路径保持禁用，不在同一 evidence epoch 静默混用；它们仅可作为另行预注册的消融。 |
| SkillFlow TTB 与 MD GRPO | MD Action-Masked One-Pass GRPO 是唯一反向传播目标；TTB 禁用。SkillFlow 仅提供 Qwen3.5/SGLang、同一步 rollout、LoRA、checkpoint 和 adapter publication/synchronization 的源码边界。 |
| 新规格的二值终局变量 `R_T` 与 HotpotQA 的连续 token F1 | Combination Posterior、surface-signal sensor、paired probe 和 Skill evidence 使用 official-compatible `exact_match ∈ {0,1}`；GRPO 延续现有 terminal evaluator 的 `token_f1` reward。两者分别记录且互不回流，F1 不进入 Beta sensor。 |
| Phase B 第 4 条对 optimizer's curse 的文字方向 | 联合 posterior sampling 保留。数学上由 Jensen inequality 有 `E[max X] >= max E[X]`；实现和测试采用该方向，并将规格中相反措辞记录为歧义，不能为了通过门禁反转数学关系。 |
| SkillFlow 论文硬件与本机仅 GPU5 可用 | `single_gpu_sequential` 是本项目资源约束下的 time-multiplexing adaptation，不是 SkillFlow 论文声明的 GPU layout。执行顺序严格为 rollout → pause/drain → stop task-owned SGLang → train → restart → publish → canary → resume。 |
| held-out paired evidence 与常规 validation metric | discovery probe 使用 train；校准/确认只允许 problem-disjoint held-out paired evidence。配对分支不进入 GRPO 或标准 benchmark metrics。常规每步 7-task validation monitor 仍不用于 posterior/Skill；专用 held-out paired manifest 尚未冻结，因此 Phase D/E 仍为 pending。 |

## 数据流隔离

| 数据 | GRPO | Combination Posterior 对比项 | task baseline | surface-signal sensor | 标准指标 |
|---|---:|---:|---:|---:|---:|
| natural trajectory（evaluator-valid） | 是 | 否 | 是 | 是 | 是 |
| paired probe branch | 否 | `keep-switch` 差分 | 是 | 是 | 否 |
| warning audit branch | 否 | `keep-switch` 差分 | 是 | 是 | 否 |
| latent-risk readout | 否 | 冻结 posterior 的只读结果 | 否 | 否 | 否 |
| Skill candidate / active status | 否 | posterior 派生 | 否 | 否 | 否 |

同一自然 rollout group 必须共享 `(task_id, condition_id, policy_version)`，每批只
消费一次。forced probe、audit、fallback/manual repair、reconstructed context 和
无效 evaluator trajectory 均不能进入 GRPO。

## 新规格参数绑定

| 规格项 | 配置值 |
|---|---|
| DecisionKey role clusters | `solve, verify, plan, summarize, arbitrate, retrieve, code, test`；由 LLM strict-JSON 分类，禁止 keyword/regex fallback |
| relation | `independent, unidirectional, bidirectional` × `same_model_as_upstream` |
| stage | `before_output, other` |
| comparison posterior | Gaussian conjugate posterior；precision/information form；Cholesky solve |
| second-order activation | 5 probes |
| empirical Bayes | once per epoch；initial factor/interaction variance 0.05 |
| paired noise floor | 0.01 |
| task baseline | prior mean 0.5、prior count 4、输出 clip `[0.02,0.98]` |
| surface sensor | per-class Beta(1,1) sensitivity / false-positive-rate |
| latent risk | `tau=0.8`, `delta_min=0.05`, `Kc=5`, posterior samples `P=200` |
| probe | natural trajectory budget 的 10%，same snapshot，只改一个 DecisionKey field，每 arm `K=3`，下游由冻结 Director 重新采样 |
| audit | warning step 以 5% 概率配对审计，每 arm `K=3` |
| Skill | discovery pairs ≥10、independent held-out confirmation problems ≥20、alpha=0.05、BH-FDR=0.1、每步最多注入 3 条 active Skill、下一 epoch 可见 |
| policy refresh | normal decay 0.1；low-coverage decay 0.5；sensor counts ×0.9；baseline 保留最近 2 epochs；连续两轮低覆盖重置 |

## Phase A–E acceptance matrix

| Phase | 规格验收 | 当前实现证据 | 当前判定 |
|---|---|---|---|
| A Combination Posterior | 200 probes 的 first-order 90% coverage；第 5 次触及激活 interaction 并在 50 probes 后覆盖；sensor FPR 误差；paired fit 消除 selection confounding | `tests/unit/test_combination_ledger.py` 覆盖合成数值路径；正式 gate receipt 尚未由 runner 固化 | `pending_formal_gate_run` |
| B posterior latent risk | same-model self-check high risk；independent verifier low risk；empty ledger 不预警；joint posterior max 检查 | `tests/unit/test_latent_loss.py` 覆盖只读 risk calculation；optimizer's-curse 文本歧义按数学正确方向记录 | `pending_formal_gate_run` |
| C paired probe / audit | same snapshot、每 arm K=3、下游重新生成；straddle 比 random 更快收窄；branch 不进入 GRPO | `ledger_probe.py`、`probe_intervention.py`、`rollout_collector.py::collect_probe_branch` 及其单元测试提供 primitives；真实 cache/tool-state isolation 与同预算效率 receipt 尚无 | `pending_formal_gate_run` |
| D Skill lifecycle | 真效应规则经 discovery+independent confirmation 转 active；null rule 的 FDR；effect drift 后 suspended；条件匹配注入 Director observation | `ledger_miner.py`、现有 `skills` lifecycle/gate、`ledger_epoch.py` 及单元测试提供 primitives；真实 held-out paired manifest 与端到端发布 receipt 尚无 | `pending_formal_gate_run` |
| E integration | 数据去向断言；完整 50 questions × G=4 epoch；可解释方差、warning precision/recall 输出 | 配置冻结要求 200 natural trajectories；尚未运行 | `pending_not_run` |

## Source map

状态分类严格使用“直接复用 / 薄适配 / 项目算法新增 / 尚未接线”。

| 模块 | FlowSteer / SkillFlow 原始代码出处 | 当前项目符号 | 分类 | 状态 |
|---|---|---|---|---|
| Flow-Director generation | FlowSteer `train_interactive.py::create_generate_fn` | `src/interactive/director.py::DirectorPolicy` | 薄适配：XML operator action 改为 AgentGraph atomic action；minimal prompt 不变 | 基础调用链已实现 |
| first atomic edit | FlowSteer `train_interactive.py::_truncate_after_first_action`、`ActionParser.parse` | `src/interactive/agent_action_parser.py`、`src/interactive/rollout_collector.py::AgentGraphRolloutCollector` | 直接复用语义、JSON syntax 薄适配 | 已实现 |
| Progressive Canvas Editing | FlowSteer `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step` | `src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step` | 薄适配为自由 AgentGraph | 已实现 |
| terminal reward / one-pass update boundary | FlowSteer `train_interactive.py::{run_full_training,compute_grpo_loss}` | `src/interactive/task_evaluator.py`、`src/interactive/grpo_objective.py::{same_condition_advantages,action_masked_one_pass_loss,torch_action_masked_one_pass_loss}` | MD 必要适配：terminal-only、exact grouping、action-token mask、single consumption | primitive 已实现；新 profile 真实 optimizer step 未运行 |
| Qwen3.5 θ-LoRA | SkillFlow `training/gflownet_trainer.py::GFlowNetTrainer.setup`、`configs/skillflow.yaml` | `src/interactive/smoke_trainer.py::{SmokeTrainerConfig,Qwen35OnePassSmokeTrainer}` | LoRA r64/alpha128/q,k,v,o 与 PEFT load/save 对齐 SkillFlow；只训练 θ，不创建 φ/Z | 已实现；新 profile 未运行 |
| same-step rollout collection | SkillFlow `GFlowNetTrainer::_sample_batch_for_step/_collect_episodes` | Hotpot runner + `AgentGraphRolloutCollector.collect` | 直接复用同一步冻结 policy 的并行边界；跨 optimizer prefetch 禁用 | 基础实现存在 |
| SGLang Supervisor | SkillFlow `training/sglang_manager.py::SGLangSupervisorManager` | `src/interactive/sglang_manager.py::SGLangSupervisorManager` | 薄适配 child ownership、local readiness、Qwen3.5 server args | 已实现；本配置未启动 |
| exact behavior receipt | SkillFlow `training/batch_inference.py::supervisor_call` | `src/interactive/rollout_collector.py::SGLangReceiptDirectorClient` | MD 必要适配：prompt/output token IDs、per-token behavior log-prob、policy/adapter/server version | Phase 0 已验证；新 evidence epoch 未运行 |
| checkpoint / optimizer / RNG | SkillFlow `GFlowNetTrainer::_save_checkpoint`; FlowSteer `run_full_training` checkpoint boundary | `src/interactive/smoke_trainer.py::_save_training_state/_restore_training_state` | LoRA directory 对齐 SkillFlow；optimizer/scheduler/RNG 与 version metadata 为必要适配 | 代码存在；新 profile 真实恢复未验收 |
| adapter publication / route switch | SkillFlow `GFlowNetTrainer::_sync_lora_to_vllm`; `batch_inference.py::{set_current_adapter_name,_resolve_model}` | `src/interactive/policy_sync.py::SGLangPolicyPublisher` | 薄适配为 SGLang adapter directory load、route switch、canary receipt | 单元层存在；新 profile 未运行 |
| single-GPU lifecycle | SkillFlow 未规定该 GPU mapping | `src/interactive/sequential_gpu_runtime.py::SequentialGpuRuntime` | 项目工程适配；只管理本 runtime 启动的 SGLang child | 已实现；真实 cycle 未运行 |
| DecisionKey / Gaussian conjugate ledger / sensor | 无上游等价实现；由新规格定义 | `src/interactive/exploration/combination_ledger.py::{DecisionKey,SurfaceSignal,LedgerState,CombinationPosterior}` | 项目算法新增 | primitive 已实现；正式 gate pending |
| posterior latent risk / straddle | 无上游等价实现；由新规格 §5–6 定义 | `src/interactive/exploration/latent_loss.py::{LatentLossConfig,pre_execution_readout,post_execution_latent_loss,straddle_score}` | 项目算法新增；只读 observation，不是 loss/reward | primitive 已实现；正式 gate pending |
| paired intervention | FlowSteer Canvas snapshot/continuation 边界；新规格给出 same-snapshot protocol | `src/interactive/exploration/ledger_probe.py`、`probe_intervention.py::ProbeInterventionTranslator`、`rollout_collector.py::collect_probe_branch` | FlowSteer continuation 薄适配 + 项目算法新增 | primitive 已实现；Phase C real receipt pending |
| LLM role clustering / contract rewrite | 新规格固定 role clusters；无上游等价分类器 | `src/interactive/exploration/role_classifier.py::{RoleClassifier,ContractRoleRewriter}` | 项目必要适配，严格 JSON，无 keyword heuristic | primitive 已实现；真实 service receipt pending |
| frozen ledger/Skill epoch | 新规格 §8 | `src/interactive/exploration/ledger_epoch.py::LedgerEpoch` | 项目算法新增：冻结 policy/ledger/Skill version 与 data routing | primitive 已实现；完整 epoch 未运行 |
| calibrated Skill mining | SkillFlow `training/skill_evolution.py`、`src/skills` 仅提供 workspace/lifecycle 参考 | `src/interactive/skills/ledger_miner.py` 与 `skills/{schema,validator,lifecycle,retrieval,store}.py` | 新规格的 paired effect、calibration、BH-FDR 与 held-out gate 是项目算法新增 | primitive 已实现；Phase D/E pending |
| W&B | FlowSteer `run_full_training`; SkillFlow trainer step logging | `scripts/train_hotpotqa_grpo.py::WandbTracker` | 薄适配为 online fail-closed 和 checkpoint artifact | 配置已绑定；run 未启动、URL 不存在 |

## 单卡运行边界

本配置将 learner、rollout Supervisor 和唯一 gradient worker 绑定到物理 GPU5，端口
8016；三者不是并发常驻。`gradient_worker_count=1`，不加载第二个训练副本。每次
更新必须先暂停 admission 并 drain，然后只停止当前 runtime 自己启动的 SGLang
child；训练完成后重启、发布新 adapter、执行 canary，再恢复 admission。任何阶段
失败都保持 fail-closed。该布局是项目适配，不归因于 FlowSteer 或 SkillFlow。

## 启动前仍需完成

1. 为 Phase A–D 生成机器可读 acceptance receipt，而不只依赖测试名称。
2. 冻结 problem-disjoint held-out paired evidence manifest。
3. 运行 Phase E 的 50 questions × 4 natural trajectories，验证自然轨迹、probe 和
   audit 的数据去向，并输出可解释方差与 warning precision/recall。
4. 所有 A–E gate 通过后，才把 runtime admission 状态切为 authorized，并运行一个
   真实 `rollout → reward → Action-Masked One-Pass GRPO loss → backward →
   optimizer.step → checkpoint → publish → route switch → canary → new-policy
   rollout` 闭环。
5. 只有该闭环证明 non-zero gradient、non-zero LoRA update 以及下一轮实际使用新
   policy version 后，才允许 250–300 step 长训练。

当前没有启动模型服务、W&B run、rollout 或 optimizer step。
