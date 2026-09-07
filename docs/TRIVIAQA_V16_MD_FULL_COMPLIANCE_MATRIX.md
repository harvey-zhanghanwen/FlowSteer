# TriviaQA v16：MD_FULL_COMPLIANCE_20260906_V2 合规矩阵

本矩阵对应设计文档
`FlowSteer_MACE_Bayesian_Skill_Design.md` 的完整 1574 行内容。它只描述
当前源码和证据的真实状态，不把接口预留、单元测试或历史实验误报为阶段验收。

## 1. 固定边界

- 固定起点：`backup/triviaqa-fact-memory-v16-best-82p81-20260906` @
  `114bd8b6378af55fbd1f40cdfdda91c79314a771`
- 当前分支：`train/triviaqa-v16-md-full-compliance-20260906`
- 当前严格阶段：**Phase 0：数据可信性**
- 主训练目标：**terminal-only、same-problem/same-condition、
  Action-Masked One-Pass GRPO**
- TTB：**禁用**；theta/phi/Z TTB trainer 不得实例化
- 长训练：**禁止**，直到 Phase 0–5 按顺序通过，并完成本次版本的真实
  1-step 闭环验收
- 当前计数：rollout batch、backward、`optimizer.step`、adapter publish、
  post-update rollout、W&B run 均为 **0**
- 数据前置：512 条 train / 128 条 held-out validation 已物化，二者
  `base_task_id` overlap=0；train-only fact index 已在 CPU 构建（512 facts、
  BGE 768 维、L2 normalization、冻结 `top-k=3`）
- 1-step contract：一题同条件 4 trajectories 的 prepare config 已存在，但
  `execution_gate.execution_enabled=false`，不能执行 rollout 或训练

状态采用四种标准标签：`满足`、`部分满足`、`缺失`、`不适用`。其中“代码存在”
不等于“阶段满足”。

## 2. 训练算法与信号隔离

| MD 条款 | 当前代码符号 | 原始代码来源 | 复用分类 | 状态与缺口 |
| --- | --- | --- | --- | --- |
| §4.1 多轮图编辑轨迹 | `AgentGraphRolloutCollector.collect`；`TrajectoryRecord`；`TurnRecord` | FlowSteer `train_interactive.py:1452-1672`、`workflow_builder.py:12-41` | FlowSteer 直接复用 + exact receipt 薄适配 | 部分满足；当前版本尚无正式训练 trajectory |
| §4.2 只使用终局任务奖励 | `EvaluationReceipt.reward`；`GRPOTrajectory.reward`；`validate_smoke_bounds` | FlowSteer terminal evaluator 边界；MD 改写 reward 口径 | 必要适配 | 代码门禁满足；尚无本次 1-step 运行证据 |
| §4.3 同题相同条件组内优势 | `TrajectoryRecord.group_key`；`group_eligible_trajectories`；`same_condition_advantages` | FlowSteer group advantage 思路；发布实现按 dataset 分组，不能直接复用 | 必要适配 | 代码存在；尚未以本次真实 batch 验证 |
| §4.4 One-Pass loss | `action_masked_one_pass_loss`；`torch_action_masked_one_pass_loss` | FlowSteer 发布代码 `compute_grpo_loss:2052-2163` 的 one-pass 形式 | 必要适配 | 代码存在；未完成本次 backward/step |
| §4.5 真实 prompt 与 action mask | `TurnRecord` 的 prompt/output token IDs、log-probs、executed-prefix mask；`SGLangReceiptDirectorClient` | FlowSteer trajectory 边界；exact behavior receipt 为 MD 必需适配 | 必要适配 | 部分满足；需 Phase 0 receipt 验收及真实 step |
| §4.6 on-policy 条件 | `VersionBundle`；`TrajectoryRecord.grpo_eligible`；`RolloutGate`；`TransactionalAdapterPublisher` | SkillFlow named LoRA/hot-load/pause 原语；strict barrier 不在论文/上游 | 必要适配 + 工程补全 | 部分满足；尚无本次 publish 后下一轮新权重证明 |
| §11.1 正常 rollout 进入 GRPO | `grpo_eligible` | MD | 项目必要实现 | 代码存在；本次计数 0 |
| §11.2 forced probe 不进入 GRPO | `forced_probe`；`grpo_eligible`；`ProbeRecord` | MD | 项目必要实现 | 代码门禁满足；未运行本次 exploration epoch |
| §11.3 Skill validation 不进入 GRPO | `SkillEvidenceGate`；probe resolver 的 `grpo_eligible=false` 约束 | MD | 项目必要实现 | 代码存在；Phase 4 未验收 |
| 禁止 uncertainty/UCB/EVSI/structure/Skill reward 混入 GRPO | `validate_smoke_bounds` 检查 exploration/structure/skill reward 为 0 | MD；不同于 FlowSteer 的结构奖励 | 必要适配 | 配置门禁满足；需在 executable config 再冻结 |
| 禁止 TTB | `training_triviaqa_v16_upstream_method_decision.yaml`；兼容 redirect | SkillFlow TTB 仅作为不启用的来源对照 | 明确禁用 | 满足；TTB 训练次数 0 |

## 3. AgentGraph、Canvas、执行与终局评估

| MD 条款 | 当前代码符号 | 原始代码来源 | 复用分类 | 状态与缺口 |
| --- | --- | --- | --- | --- |
| §2 Flow-Director→Canvas→AgentGraph→Evaluator→Trajectory | `director.py`；`AgentWorkflowEnv`；`AgentRuntime`；`TaskEvaluator`；`AgentGraphRolloutCollector` | FlowSteer Director/Canvas/execution/trajectory 调用链 | 直接复用 + Qwen3.5/AgentGraph 薄适配 | 推理链路已存在；训练链路未验收 |
| §3.1 自由 Agent 节点 | `AgentNode`；`AgentExecutionMode` | FlowSteer operator 抽象 + MD AgentGraph 扩展 | MD 必要适配 | 代码存在并有测试；不等于探索已学习 |
| §3.2 两比特关系 | `RelationBits`；`AgentRelation` | MD 明确定义 | 项目算法新增 | 代码存在并有图校验 |
| §3.3 双向关系有限执行 | `AgentGraphValidator`；`AgentWorkflowEnv.execute` | MD 明确定义 | 项目算法新增 | 推理代码存在；尚无 Phase 2 paired intervention 验收 |
| §3.4 商图有向无环与可达性 | `AgentGraphValidator` | MD | 项目算法新增 | 代码存在并有定向测试 |
| Progressive Canvas Editing | `AgentWorkflowEnv.step`；ADD/MODIFY/REMOVE/SET_RELATION/SET_OUTPUT/FINISH | FlowSteer `InteractiveWorkflowEnv.step` 与完整 edit 后执行 | FlowSteer 直接复用 + action-space 薄适配 | 推理代码存在 |
| terminal evaluator 与 FINISH | `AgentWorkflowEnv` FINISH gate；`EvaluationReceipt`；`task_evaluator.py` | FlowSteer terminal evaluator boundary | 直接复用 + 数据集适配 | TriviaQA evaluator 存在；Phase 0 需证明 reward lineage |
| 冻结 train / held-out split | `data/agentgraph_v1/{train,validation}.jsonl`；`manifest.json` | 项目统一数据 schema；用户既定 128 held-out 后取 512 train 规则 | 项目必要适配 | **满足数据子门禁**：512/128 已物化，`base_task_id` overlap=0；不等于 Phase 0 整体通过 |
| train-only fact projection | `scripts/materialize_triviaqa_train_fact_subset.py::materialize_train_only_fact_subset` | 复用既有 TriviaQA fact-memory record/load/write 数据流 | 既有数据流薄适配 | **满足数据子门禁**：512 facts；Agent-facing 仅 `fact_text` 等事实记录字段，Q/A provenance 不由 index 加载，validation memory overlap=0 |
| embedding index | `build_triviaqa_fact_memory_index.py` 与 `triviaqa_embedding_index.py` | 项目既有 embedding index/Tool 数据流 | 直接复用 | **满足构建子门禁**：CPU 构建、BGE 768 维、L2-normalized dot product、`top-k=3`；尚无本次 worker Tool rollout receipt |

## 4. MACE、联合后验、paired intervention 与 EVSI

| MD 条款 | 当前代码符号 | 原始代码来源 | 复用分类 | 状态与缺口 |
| --- | --- | --- | --- | --- |
| §5 原始 MACE 9 维特征 | `MACEFeatureExtractor`；`MACEFeatureConfig` | 原始 MACE 特征定义，经 MD 限定仅作 baseline | 直接复用/薄适配 | 数值原语存在；未完成 Phase 1 同预算对照 |
| §5.2 ordered-pair LinUCB | `DisjointLinUCB`；`MACE` | 原始 MACE LinUCB | 直接复用 | 原语存在；当前 `update_from_scores` 仍使用原始 blended round score，不符合 terminal-effect 主闭环 |
| §6 探索对象包含上下文/模型/contract/关系/版本 | 现有 numeric `(querier, peer)` arm | MD 扩展 | 项目算法新增 | 缺失生产级复合 action key 与版本绑定 |
| §7.1 context feature | `RoundSnapshot` 与 MACE 特征；posterior 接受一般向量 | MD | 项目算法新增 | 部分存在；尚未冻结正式 feature schema |
| §7.2 low-rank main effect + sparse interaction | 无完整生产实现 | MD | 项目算法新增 | 缺失；现有 evidence runner 使用 one-hot，不等价 |
| §7.3 Bayesian linear head | `BayesianLinearPosterior`；`PosteriorSnapshot` | MD | 项目算法新增 | 数值原语存在；无正式 epoch/校准/漂移闭环 |
| §7.4 同前缀 paired intervention | `AgentWorkflowEnv.snapshot/restore/fork`；`PairedProbeRecord`；`run_joint_qa_mace_skill.py::_paired_probe` | MD | 项目算法新增 | 未满足；现 runner 两臂从空 Canvas 分别启动、K=1，不能证明相同任意前缀与 whole-rollout effect |
| §8.1 posterior UCB | `UCBPolicy` | MD | 项目算法新增 | 数值原语存在；未接生产 scheduler |
| §8.2 rollout-level posterior sampling | `ThompsonSamplingPolicy`；`ThompsonRollout` | MD | 项目算法新增 | 原语存在；未证明每条 rollout 只采一个 posterior particle |
| §8.3 EVPI 仅诊断 | `estimate_particle_evpi` | MD | 项目算法新增 | 数值原语存在；无正式 diagnostic artifact |
| §8.4–8.5 particle EVSI + common random numbers | `CommonRandomNumbers`；`estimate_particle_evsi`；`particle_evsi_many` | MD | 项目算法新增 | 数值原语存在；未接关键节点 scheduler/停止准则 |
| §9 结构关键性/决策关键性/查询价值分离 | 无完整 production metrics | MD | 项目算法新增 | 缺失 |
| §13 calibration/non-stationarity | posterior snapshot/restore；Skill version checks | MD | 项目算法新增 | 部分存在；无 coverage/calibration 和 policy-update posterior 处理验收 |

## 5. Skill 经验固化流

| MD 条款 | 当前代码符号 | 原始代码来源 | 复用分类 | 状态与缺口 |
| --- | --- | --- | --- | --- |
| §10.1 typed Skill record | `StructuredSkillCandidate`；`SkillRecord`；`SkillEvidence` | SkillFlow workspace 提供存取/检索思想；typed evidence schema 来自 MD | SkillFlow 薄适配 + MD 新增 | 代码存在 |
| candidate/active/suspended/retired | `SkillStatus`；`SkillStore`；`SkillLifecycleManager` | MD；SkillFlow 无四状态机 | 项目算法新增 | 状态机存在；当前 ACTIVE=0 |
| §10.2 independent held-out gate | `SkillEvidenceGate`；`SkillEvidencePipeline.confirm_and_publish` | MD；不能直接使用 SkillFlow 的文本 gate | 项目算法新增 | 代码存在；尚无正向 held-out gain；多候选 FDR/local false-sign 控制缺失 |
| §10.3 prompt/action/repair prior | `PromptSkillPrior`；`render_validated_skill`；retrieval pipeline | SkillFlow workspace retrieval/prompt formatting；可拒绝性来自 MD | 薄适配 | prompt prior 已接线；action proposal/repair prior 生产闭环未完整验收 |
| §10.4 suspension/retirement | `SkillLifecycleManager`；policy drift suspend | MD | 项目算法新增 | 代码存在；无本次 active→suspended→revalidate 实证 |
| Skill 不能直接注入 GRPO reward | `validate_smoke_bounds`；`grpo_eligible` | MD | 必要门禁 | 代码满足；尚无本次训练证据 |

## 6. 六阶段顺序验收

| Phase | MD 验收条件 | 当前证据 | 判定 |
| --- | --- | --- | --- |
| Phase 0 数据可信性 | 每次模型调用只记录一次；终局 reward 可追溯；稳定 problem/policy/prompt/evaluator/tool/model IDs；完整图快照可回放 | 512 train / 128 held-out、ID 隔离和 512-fact train-only index 子门禁已满足；128 条历史 trajectory 的 receipt/replay 子门禁也已通过，但它们来自 transductive profile，不能作为新 held-out 训练证据；ReAct provider sub-call receipt 与本次 same-condition training group 仍缺 | **进行中，未通过** |
| Phase 1 原始 MACE baseline | 9 维特征和 LinUCB；同预算 random/greedy/uniform 对照；遗憾与 reward 曲线 | 数值原语存在；无同预算真实对照；reward adapter 仍为 blended round score | **未通过** |
| Phase 2 联合后验与同前缀探针 | low-rank feature；随机臂序；相同前缀/续跑版本；whole-rollout effect；calibration/coverage | posterior、arm order、probe record 原语存在；真正同前缀和 K 次独立续跑未接通 | **未通过** |
| Phase 3 EVSI 与关键节点 | particle EVSI+CRN；UCB/TS/EVSI 同预算；三类关键性分离；停止准则 | EVSI/TS 数值原语存在；scheduler、指标与正式对照缺失 | **未通过** |
| Phase 4 Skill 发布闭环 | independent held-out positive gate；ACTIVE 使用；版本变化 suspension/revalidation；memory-on/off 增益 | 生命周期/gate/retrieval 原语存在；当前只有 candidate/retired，ACTIVE=0 | **未通过** |
| Phase 5 普通任务部署与评测 | exploration `beta=0`；posterior mean + validated ACTIVE Skill；无 forced probe；正式 held-out 评测 | deployment config checks 存在；受 Phase 1–4 前置条件阻塞 | **未通过** |

严格阶段规则：前一阶段未通过时，不启动或宣称后一阶段验收。后续阶段的现有
代码只保留为未激活原语，不作为越级执行依据。

## 7. checkpoint、权重发布与 W&B

| 要求 | 当前代码 / SkillFlow 来源 | 状态与最小适配 |
| --- | --- | --- |
| 真实非零 backward/step | `Qwen35OnePassSmokeTrainer.train`；SkillFlow replica/gradient merge 思路 | 主干存在；本次计数 0 |
| 每 step 保存 theta adapter | `smoke_trainer.py` 的 PEFT save | 存在 |
| 保存 optimizer/scheduler/RNG/训练元数据 | `Qwen35OnePassSmokeTrainer._save_training_state/_restore_optimizer_state`；FlowSteer `train_interactive.py:1371-1396,2012-2027` 的状态保存/恢复边界 | **部分满足**；完整 `training_state.pt` schema、恢复门禁和 CPU 定向测试已通过，当前无 scheduler 时保存显式 disabled contract；尚无真实 CUDA step 证据 |
| 保存完成后再 publish | `SmokeTrainingSummary.checkpoint_ready`；`TransactionalAdapterPublisher` | durable checkpoint completion gate 已实现并通过 CPU 定向测试；尚无真实 publish 事务证据 |
| pause/drain、load、route switch、canary | `RolloutGate`；`policy_sync.py`；SkillFlow `_sync_lora_to_vllm`/`batch_inference.py` 原语 | 部分满足；transaction/route/canary 是项目工程补全，不能称为 SkillFlow 原文 |
| 下一轮使用新权重 | post-update canary/trajectory version checks | 代码存在；本次尚无运行证据 |
| W&B 实时且 fail-closed | `WandbTrainingRun`、runner `_WandbLifecycle`、`MaterializedWandbStepEvidenceProvider`；两套上游均为 optional/fail-open；用户已固定 `zhanghanwen6660909-dut/flowsteer-triviaqa` online binding | **部分满足**；provider 可校验既有 held-out evaluator receipts 并读取 GPU telemetry，但 updated-policy held-out evaluation 的外层调度与 CLI provider 注入尚未接通；online run 尚未初始化，没有 run URL |
| 禁止 stale rollout | 禁用 SkillFlow `_next_future` 跨 step prefetch | 配置边界已确定；尚待 executable runner 验收 |
| 一题 × 4 trajectories 的 1-step contract | `training_triviaqa_v16_onepass_grpo_1step_prepare.yaml`；`triviaqa_training_schedule.py`；runner `_is_trivia_micro/_select_run_scope`；`_require_execution_gate` | SkillFlow frozen sequence/cursor + FlowSteer rollout/group 边界 + MD same-condition/one-pass 约束；TriviaQA binding 与 fail-closed gate 为必要薄适配/工程补全 | **prepare-only 已验证**；精确选中 1 个 train task 和 rollout ordinals 0–3，无静态 retrieval prefetch，cursor 未推进；TTB/Skill/exploration 均关闭，`execution_enabled=false` |

## 8. 当前允许的下一步

1. 保留已经完成的 512/128 split-isolation 和 train-only fact index；补齐
   branch-local training evidence、ReAct provider sub-call receipt 与一题同条件
   4 trajectories 的真实 group。当前 trajectory receipt/replay 只证明历史子门禁。
2. Phase 0 整体通过后，修正 Phase 1 的 terminal-effect 接口并准备同预算
   random/greedy/uniform/LinUCB 对照；未得到真实 acceptance evidence 前不进入
   Phase 2。
3. 将现有 materialized held-out/GPU evidence provider 接入 updated-policy
   held-out evaluation 外层调度和 runner CLI，并在初始化 online W&B run 后取得
   真实 run URL；实现存在不等于运行验收通过。
4. 只有该 step 证明 `rollout(vN) → terminal reward → one-pass loss → backward
   → optimizer.step → checkpoint → publish → route switch → canary →
   rollout(vN+1)`，才允许 250–300 step 长训练。

本次只读 preflight 的当前结果是：8 张 GPU 均有其他项目进程占用，现有
learner / rollout Supervisor / gradient replica 三卡分工没有安全的不冲突组合。
用户允许共用未用满的卡，但只有在新的资源门禁确认显存余量、并发负载、端口和
服务所有权后才可共享；当前不得打开 execution gate。训练 venv 已安装 W&B
client，用户确认标准凭据已配置，但尚未初始化 online run 或获得 run URL。
