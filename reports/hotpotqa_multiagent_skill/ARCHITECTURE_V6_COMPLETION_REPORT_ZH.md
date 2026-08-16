# Architecture v6 Completion Report

日期：2026-08-16。范围仅为 HotpotQA；没有进入其他数据集。

## 1. 已完成

- Flow-Director 固定为 GPU4 本地 Qwen3.5-9B；没有用远程 API 模型替代。
- 保留 FlowSteer 的逐轮单原子 Canvas 编辑、自由 Agent contract、两比特关系、唯一 Output、显式 FINISH 和终局 evaluator。
- Executor 目录扩大为 10 个等权、真实 evidence-backed 模型臂，覆盖 Qwen、DeepSeek、GPT、MiniMax、GLM、Kimi 六个家族；新增 GLM/Kimi 各一次 canary 均通过。
- Director 状态增加只读、无偏置的 `construction_progress`；提示只说明可选抽象关系形态，没有固定 Hotpot workflow、role enum、最少 Agent 数或 topology quota。
- 图诊断已明确 structural depth、reciprocal contraction、topology family/motif 与 graded effective dependency depth。普通 runtime delivery 最多记为 weak，不能伪称因果 verified。
- Skill 闭环已接通现有 Trajectory/Probe/EvidenceStore/SkillStore/Lifecycle/Retriever：`Trajectory → CANDIDATE → paired evidence → 独立 validation → gate → ACTIVE → rejectable retrieval prior`。
- 既有 smoke runner 已薄适配到冻结 Hotpot schedule/cursor：每次只执行一个预声明 task group，一次 optimizer update，成功同步和 post-update canary 后才提交新 cursor；旧 7×2 路径保留。
- 已物化真正未训练 Formal Step 0：`qwen35-9b-hotpot-step-000000 / theta_hotpot_step_000000`，64 个 LoRA tensor，optimizer updates 为 0。

## 2. 已验证模块

- 定向无模型测试：model catalog、Director renderer、AgentGraph、graph diagnostics、Skill pipeline、Hotpot evaluator runner、micro runner 均通过。
- Formal Step 0 已加载到任务自有 SGLang；adapter model-list 验证与本地 canary 成功。
- 两个固定 validation canary 都完整经过：

  `Question → Qwen3.5-9B Director → atomic Canvas → AgentGraph Runtime → Output → Hotpot evaluator → Trajectory`

- 两题均保存了完整 Director turn receipt、generation seed、Agent execution、Output inbox、explicit FINISH 和 evaluator receipt，Stable Zero 判定为通过。
- Canary 没有训练、backward、optimizer 或 policy publish。

## 3. 仅为接口/能力、尚未由真实收益验证

- 深图、fan-in/fan-out、parallel、reciprocal/mixed 都可表达且有结构测试，但 Formal Step 0 的前两条正常 trajectory 仍是 single；这不能算真实深图行为验证。
- Skill pipeline 的边界和证据门控已经端到端测试；尚未用新的 v6 真实、问题隔离 paired evidence 激活生产 Skill，因此当前正式 rollout 不注入 Skill。
- MACE/Bayesian 仍为项目方法接口，不进入本轮 terminal reward 或微训练主路径。
- `verified_dependency_depth` 需要独立 paired-intervention receipt；当前普通 transport 证据不会自动提升为 verified。

## 4. 已知问题

- 历史 v3/v4 的主坍缩是冻结策略在首个完整 singleton execution 后停止：v3 为 122/128 single，v4 为 11/12 single；不是 `max_rounds=20` 的硬限制。
- Formal Step 0 的 2 题 canary 仍为 2/2 single、0 个 depth 3+；必须由完整固定运行和后续终局奖励微更新判断 search distribution 是否可学。
- 两题 canary 不能用于声称 EM/F1、拓扑多样性或通信效用改善。
- 当前 Skill 正式库为空；不能将合成测试中的 ACTIVE Skill 当作真实 HotpotQA gain。

## 5. Stable Zero

```text
STABLE_ZERO = YES
FULL_CHAIN_CANARY = 2/2
FORMAL_POLICY_STEP = 0
OPTIMIZER_UPDATES = 0
DEEP_WORKFLOW_BEHAVIOR_VALIDATED = NO
```

因此架构运行链已达到 Stable Zero，可以继续同一固定 128-task HotpotQA Formal Step-0 development 评测。是否达到“深层多智能体已适配”仍必须由真实 trajectory、EM/F1、topology、异构路由、communication 和 Skill evidence 决定。
