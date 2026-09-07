# MD_FULL_COMPLIANCE_20260906_V2

## 训练算法与当前门控

- 规范来源：`FlowSteer_MACE_Bayesian_Skill_Design.md` v1.1，第 0–23 节。
- 唯一任务学习目标：terminal-only、same-problem/same-condition、Action-Masked One-Pass GRPO。
- SkillFlow Tempered Trajectory Balance（TTB）、backward policy φ 与 partition function Z：禁用。
- 当前阶段：Phase 0（数据可信性），尚未通过验收。
- 真实 optimizer step：本分支 0；单步训练与长训练均禁止。
- 三条数据流：任务学习流、探索学习流、经验固化流必须使用不同 record/condition 和更新入口；forced intervention 不进入 GRPO 或标准 benchmark 指标。

## MD 章节 compliance matrix 与 source map

状态含义：`已实现` 仅说明代码符号存在；`已接线` 表示进入真实调用链；`已验收` 必须具有该 Phase 规定的完整实验指标。占位、schema 和单元测试不能替代阶段验收。

| MD 章节 | 要求 | 当前代码符号 | FlowSteer / SkillFlow 原始出处 | 复用或适配边界 | 当前状态 |
|---|---|---|---|---|---|
| §0 文档边界 | 普通 QA、FlowSteer、GRPO、MACE、Bayesian、Skill；不得把规划当结果 | `records.py::{TaskRecord,TrajectoryRecord,ProbeRecord,PosteriorSnapshotRecord}` | FlowSteer `workflow_builder.py::{TurnRecord,Trajectory}` | 扩展为版本化 AgentGraph、probe、posterior records 是 MD 必要适配 | schema 已实现；完整闭环未验收 |
| §1 信号隔离 | 任务学习、探索学习、经验固化三流隔离 | `EvidenceStore::{trajectories,probes,posteriors}`；`TrajectoryRecord.grpo_eligible`；GRPO 配置中的零辅助奖励 | FlowSteer 无三流隔离；SkillFlow reward 会混合 answer/process/skill reward，不能直接复用 | 按 MD 独立存储和 eligibility gate | 基础隔离已实现；跨 Phase 调度未接线 |
| §2 总体架构 | question→Director→Canvas→AgentGraph→evaluator→GRPO，并旁路 posterior/Skill | `AgentGraphRolloutCollector.collect`、`AgentWorkflowEnv.step`、`evaluate_task`、`Qwen35OnePassSmokeTrainer.train` | FlowSteer `run_full_training`、`InteractiveWorkflowEnv.step` | 保留执行反馈闭环；自由 AgentGraph 是 MD 必要适配 | 基础任务学习链存在；探索/Skill 未接入 epoch loop |
| §2.2 QA schema | `task_id/question/ground_truth/split/metadata`，train/validation/test 隔离 | `TaskRecord`、`load_hotpotqa_training_pool` | FlowSteer 多数据源 loader；SkillFlow task environment | 统一为 MD TaskRecord；正式 split gate 仍需 Phase 0 实证 | 已实现；Phase 0 未验收 |
| §3 AgentGraph | 自由 Agent prompt、稳定 model ID、两比特 relation、有限双向语义、唯一 output | `agent_graph.py::{AgentNode,AgentGraph}`；`agent_runtime.py::AgentRuntime`；`agent_workflow_env.py` | FlowSteer `workflow_graph.py` 是有序 operator DSL，不能直接复用 | 原子编辑和 execute-on-edit 语义源自 FlowSteer；自由图结构来自 MD | 已实现并有定向单元测试；真实训练 rollout 待验收 |
| §4.1 Progressive Canvas | 每轮一个原子图编辑，Canvas 执行并反馈 | `agent_action_parser.py`、`AgentWorkflowEnv.step`、`AgentGraphRolloutCollector.collect` | FlowSteer `_truncate_after_first_action`、`ActionParser.parse`、`InteractiveWorkflowEnv.step` | XML operator action 薄适配为 AgentGraph JSON action | 已实现 |
| §4.2 Terminal reward | 只使用可信 evaluator 的终局任务奖励；结构、uncertainty、Skill reward 均为 0 | `task_evaluator.py`、`EvaluationReceipt`、`TrajectoryRecord.grpo_eligible`、`config_loader.validate_agent_graph_config` | FlowSteer `TrajectoryRewardCalculator` 只复用终局回传边界，不能复用结构奖励；SkillFlow `compute_full_reward` 不能用于此目标 | benchmark evaluator receipt 是必要替换 | 已实现基础层；fresh Phase 0 lineage 未验收 |
| §4.3 同题优势 | `(task_id, condition_id, policy_version)` 精确分组；零方差组零梯度 | `same_condition_advantages`、`group_eligible_trajectories` | FlowSteer 发布代码按 source 分组，不能直接复用 | 按 MD 修正分组并支持每题等权 | 已实现并有单元测试 |
| §4.4 One-Pass GRPO | 一批 rollout 只消费一次；不使用 PPO clipping、reference KL、entropy | `action_masked_one_pass_loss`、`torch_action_masked_one_pass_loss`；runner 每 sealed batch 一次 `backend.train` | FlowSteer `compute_grpo_loss` 只提供 policy-gradient 外形；论文 clipped GRPO 与发布代码不一致 | exact same-condition one-pass 是 MD 必要实现，不称 clipped GRPO | loss 已实现；本分支真实 step 未运行 |
| §4.5 Action mask | 真实 prompt/output IDs；只给 Canvas 消费的首个 action 前缀 credit；feedback 仅作下一轮 context | `TurnRecord::{prompt_token_ids,output_token_ids,executed_prefix_tokens,behavior_log_probs}`；`SGLangReceiptDirectorClient`；`trajectory_to_grpo` | FlowSteer 重新拼接 response+feedback，不可直接复用；SkillFlow `supervisor_call` 不返回所需 exact receipt | `/generate` receipt 和 prefix mask 是必要适配 | 已实现与定向测试；真实 SGLang receipt 待 Phase 0 验收 |
| §4.6 On-policy | 单一 behavior version、真实条件、无强制 probe/fallback/manual repair/reconstructed context | `RolloutGate`、`TrajectoryRecord.grpo_eligible`、`SmokeTrainerConfig`、`Qwen35OnePassSmokeTrainer.train` | SkillFlow 同一步 rollout 并行可复用；其跨 step async prefetch 禁用 | 增加 policy/adapter/server version receipt 与本地 log-prob 容差 gate | 已实现基础 gate；新训练闭环未验收 |
| §5 原始 MACE baseline | per-context/model LinUCB；terminal signal；same-budget 对比 random/greedy/uniform | `exploration/mace.py::{DisjointLinUCB,MACE}`；`features.py::MACEFeatureExtractor` | 不属于 FlowSteer/SkillFlow；按 MD 中原始 MACE baseline 实现 | 仅作为探索学习流，不修改 GRPO reward | 组件与单元测试存在；未进入 Phase 1 实验，未验收 regret |
| §6 条件价值 | 绑定 task/prefix/contract/model/relation/stage 与完整版本 | `VersionBundle`、`SelectionReceipt`、`exploration/records.py` | FlowSteer/SkillFlow 无该联合价值接口 | MD 项目算法数据结构 | 部分实现；候选全集与实际 condition-satisfied 接线未验收 |
| §7 联合贝叶斯后验 | Bayesian linear head、Cholesky、paired difference、每条普通 trajectory 一次 likelihood | `posterior.py::BayesianLinearPosterior`、`ProbeRecord`、`PosteriorSnapshotRecord` | FlowSteer/SkillFlow 无联合后验 | MD 项目算法新增；不能混入 GRPO optimizer | 数值组件与单元测试存在；低秩表示、真实 evidence ingestion、held-out calibration 未接线/未验收 |
| §8 UCB / Posterior Sampling / EVSI | epistemic UCB；每 rollout 单一 posterior draw；particle EVSI 与 common random numbers | `policies.py::{UCBPolicy,ThompsonSamplingPolicy,ThompsonRollout}`；`evsi.py` | FlowSteer/SkillFlow 无这些模块 | MD 项目算法新增 | 数值组件与单元测试存在；probe scheduler 与 empirical stop rule 未接线 |
| §9 三种关键性 | structural、decision(EVPI)、query(EVSI) 分离 | `evsi.py::estimate_particle_evpi/estimate_particle_evsi` | 无上游等价实现 | 需要 MD 定义的独立统计接口 | EVPI/EVSI 数值函数存在；structural criticality 和统一调度缺失 |
| §10 Skill | 结构化 evidence、held-out gate、candidate/active/suspended/retired、版本绑定 | `skills/schema.py`、`validator.py::SkillEvidenceGate`、`lifecycle.py::SkillLifecycleManager`、`retrieval.py`、`store.py` | SkillFlow `training/skill_evolution.py` 仅作生命周期/工作区实现参考；其 reward 不进入 GRPO | MD 的 paired-effect、calibrated interval、harm gate 是项目必要适配 | schema/gate/retrieval 单元层存在；candidate mining、confirmatory rollout、FDR、自动发布闭环未接线 |
| §11 严格隔离 | natural rollout、visible condition、forced intervention、confirmatory rollout 分开 | `TrajectoryRecord::{condition_satisfied,forced_probe}`、`ProbeRecord.task_split`、`EvidenceStore` | 原始两项目均无完整 MD 隔离 | MD eligibility 与分流规则 | 部分实现；完整 epoch orchestration 未实现 |
| §12 完整算法 | 冻结 epoch→natural rollout→probe→posterior→Skill→一次 GRPO→posterior version handling→β=0 metrics | 尚无单一 Phase 0–5 orchestrator | FlowSteer 仅提供任务学习 loop；SkillFlow trainer 是 TTB loop，禁止移植为主训练 | 必须用已有模块按 MD 顺序做最小集成，不能另建训练算法 | 缺失，长训练硬阻塞 |
| §13 校准/非平稳 | whole-problem held-out calibration；NLL/Brier/ECE/coverage；policy update 后 refit/inflate/restart | `PosteriorSnapshotRecord.coverage_metrics`、Skill evidence字段 | 无完整上游实现 | MD 项目算法新增 | schema 存在；calibrator、drift monitor、policy-update handler 未实现 |
| §14 计算预算 | top-K、EVSI stop、probe budget | `particle_evsi_many` 提供数值基础 | 无完整上游实现 | 调度策略必须在 development/validation 规则中预注册 | 未接线/未验收 |
| §15 持久化 | versioned trajectory/probe/posterior/snapshot，完整调用与 selection receipt | `AppendOnlyJsonlStore`、`EvidenceStore`、`GraphSnapshotEvent`、`SelectionReceipt` | FlowSteer trajectory 序列化；SkillFlow experience store 仅作存储参考 | 独立 streams 与 replay 是 MD 必要适配 | 存储与 replay 单元层存在；真实 terminal lineage 抽样验收未完成 |
| §16 代码映射 | 不把计划项描述成已实现/已运行 | 本矩阵及 `FLOWSTEER_SKILLFLOW_TRAINING_SOURCE_MAP.md` | 不适用 | 文档与代码状态同步 | 已纠正 |
| §17 Phase 0–5 | 严格按阶段晋级 | `md_compliance` 配置与 runner fail-closed gate | 无上游完整阶段机 | 只增加验收门控，不新增学习算法 | 当前停在 Phase 0 |
| §18 实验/消融 | random/greedy/MACE/UCB/Thompson/EVSI/Skill/memory-off 等同预算对照 | 各组件存在，未形成冻结实验 manifest | 无上游完整实现 | 按 MD 预注册后执行 | 未实现完整实验矩阵 |
| §19 失败模式 | 禁止 reward 污染、off-policy probe、future graph 特征、重复计数等 | eligibility/config gates 覆盖部分风险 | FlowSteer/SkillFlow 原流程不足 | MD 必要 fail-closed 校验 | 部分实现；cluster-aware posterior 统计与 deployment condition 检查缺失 |
| §20 表述边界 | 不声称 clipped GRPO、已完成 MACE–Bayesian–Skill、天然校准等 | 本文档、训练 manifest | 不适用 | 报告约束 | 已采用 |
| §21 公式 | 实现必须与 MD 公式一致 | `grpo_objective.py`、`mace.py`、`posterior.py`、`evsi.py` | 见各行出处 | 逐 Phase 验收 | 数值单元层覆盖，系统级未验收 |
| §22 当前验证状态 | 真实 GPU backward、hot-load、异构 rollout 与探索闭环需重新证明 | runner/测试/manifest | 不适用 | 历史 smoke 不替代本分支验收 | 本分支均未运行 |
| §23 参考边界 | 区分上游方法与项目扩展 | source map | commit-exact 两仓库 | 已标注直接复用、薄适配、项目算法新增 | 已完成文档核对 |

## Phase 0–5 当前验收状态

| Phase | MD 验收 | 当前判定 | 缺口 |
|---|---|---|---|
| Phase 0 数据可信性 | 任意 terminal reward 可追溯至完整图、每次真实调用和 evaluator receipt，并可 replay 同一 snapshot | **未通过** | 128 validation / 512 train、base_task_id 隔离及固定 7-task validation monitor 已静态验收；仍需在当前 policy/condition 下生成 fresh train trajectory，并验证完整 snapshot/tool state replay、每轮无缓存调用复制、全版本与 evaluator lineage |
| Phase 1 原始 MACE baseline | same-budget model/relation simple regret 优于 random | **未开始验收** | MACE 数值组件未接入真实 probe scheduler；无 random/greedy/uniform 对照结果 |
| Phase 2 联合后验与 paired probe | block-held-out NLL、coverage、top-model regret 优于 Phase 1 | **未开始验收** | snapshot fork/continuation 的完整 tool state、真实 paired repeats、low-rank feature、held-out calibrator 与 cluster-aware statistics 未形成闭环 |
| Phase 3 EVSI | 相同识别准确率所需额外 LLM probe 更少 | **未开始验收** | top-K scheduler、fantasy observation、ranking-stability 和 stop-rule 实验未接线 |
| Phase 4 Skill | 预注册 precision/coverage/held-out gain 达标，错误 Skill 自动撤销 | **未开始验收** | candidate miner、独立 confirmatory rollout、multiple-discovery control、memory-on/off/no-skill 与完整自动生命周期未接线 |
| Phase 5 部署评测 | β=0、无 forced intervention、仅 posterior mean 与 active Skill 的标准 benchmark | **未开始验收** | 必须等待 Phase 0–4；当前基础 QA inference 不能称完整 Phase 5 |

## 单步 GRPO 验收的前置缺口

配置已经选择 MD One-Pass GRPO 且 TTB 禁用，但以下项目完成前 `real_step_authorized=false`：

1. Phase 0 fresh trajectory lineage 与 replay 验收；
2. 训练 checkpoint 已按 FlowSteer 的边界保存并恢复 θ-LoRA、optimizer、cosine scheduler、Python/NumPy/PyTorch/两张训练 GPU 的 CUDA RNG、training step 与 frozen version fingerprint；需要真实单步验证；
3. W&B online fail-closed、必需字段、每步 checkpoint Artifact 及 `latest`/`best` alias 已接线；需要真实 run URL 验证；
4. 不冲突 GPU resource admission；
5. 一个冻结 behavior-policy batch 的完整 `rollout→terminal reward→one-pass loss→backward→optimizer.step→checkpoint→publish→route switch→canary→new-policy rollout` 验收。

单步验收通过前，`long_training_authorized=false`；不会进入 250–300 optimizer steps。

## WANDB_BINDING_20260906_V1

- entity：`zhanghanwen6660909-dut`。
- project：`flowsteer-hotpotqa`。
- mode：`online`。
- credential source：W&B SDK 默认凭据；代码不要求、读取或记录显式 API key 字段。
- run URL：尚无，因为当前没有创建 W&B run，也没有启动训练。
- 每步 required fields、固定 7-task held-out validation monitor、完整恢复 checkpoint Artifact、`latest`/`best` aliases 已接入 runner；validation trajectory 明确不进入 GRPO/posterior/Skill。
- 完整运行环境已确认可用：`/ssd1/iclr/gpf/venvs/skillflow/bin/python`。当前仍未创建 W&B run，且本机 GPU 全部被其他项目占用，因此不得描述为 W&B 已连接或训练已启动。

## Commit-exact 原始代码版本

- FlowSteer：`beita6969/FlowSteer@a329b577a8e8c2f2d5492983b492473a06a8e718`（与论文 v4 对齐）。
- SkillFlow：`beita6969/SkillFlow@74be52bb6bd9f0e9e68dacb72636b75649197983`，本地只读 checkout：`/ssd1/iclr/owner/skillflow-iclr/06_GITHUB_RELEASE_20260816`。
- HotpotQA 起点：`backup/hotpotqa-compliant-best-round01-20260906@740e53ec6ccac635ecbe7f1f379b002bfb2574d1`，不改写。
