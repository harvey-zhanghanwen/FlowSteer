# HotpotQA 正式 Step-0 静态预检报告（中文版）

## 一、执行结论

本阶段在现有架构上完成了训练前置能力补齐，但没有启动新的模型服务、rollout、
付费 API 请求、GRPO、反向传播、`optimizer.step()`、LoRA 权重更新或训练后策略发布。

本轮补齐了 architecture-v5 报告中指出的四项静态兼容性缺口：

1. 确定性物化从未更新过的 Qwen3.5-9B Director 初始 LoRA；
2. 仅包含 HotpotQA train 样本的不可变任务/rollout schedule 与精确 cursor；
3. 从正式 Step 1 开始保存 adapter 与 optimizer 状态，并在 Step 2+ 强制精确续接；
4. 在一次 pause/drain 事务内完成 adapter 验证、Director 路由切换、旧 adapter
   卸载、失败回滚和恢复接收 rollout。

代码实现严格依据 `docs/SOURCE_MAP.md` 中记录的 SkillFlow 与现有 FlowSteer
调用边界。没有另写第二套 workflow 框架，没有增加固定角色枚举、强制 topology、
Agent 数量奖励、图复杂度奖励、通信奖励或新的 Skill 发布捷径。

代码级训练前置条件已经通过测试，但正式 Step 0 尚未真正实例化：初始 adapter
没有写入磁盘，没有在 SGLang 上进行真实激活，也没有发布正式实验 schedule/cursor。
因此当前真实状态是：

```text
ARCHITECTURE_RUNTIME_CHAIN_COMPLETE = YES
STATIC_FORMAL_STEP0_PRECONDITIONS_IMPLEMENTED = YES
FORMAL_POLICY_STEP_000000_MATERIALIZED = NO
FORMAL_POLICY_STEP_000000_LIVE_ACTIVATED = NO
FORMAL_TRAINING_SCHEDULE_PUBLISHED = NO
GRPO_OR_WEIGHT_UPDATE_PERFORMED = NO
STEP0_TO_STEPN_LEARNING_VALIDATED = NO
SKILL_EVOLUTION_VALIDATED = NO
READY_FOR_GRPO = NO
READY_FOR_NEXT_DATASET = NO
```

## 二、本轮完成的架构改动

| 边界 | 修改文件 | 复用的上游实现 | 本项目必要适配 | 当前状态 |
| --- | --- | --- | --- | --- |
| 未训练正式初始策略 | `src/interactive/hotpot_step0.py`、`scripts/materialize_hotpotqa_step0.py` | SkillFlow 确定性初始策略构建器、保存前状态绑定门禁，以及项目已有 Qwen3.5 PEFT loader | SkillFlow 保存 forward/backward adapter 与 Z head；当前 Director 只暴露一个供 SGLang 使用的 `theta` adapter | 无模型 preflight 已通过；未物化 adapter |
| Hotpot 冻结训练顺序 | `src/interactive/hotpot_training_schedule.py`、`scripts/freeze_hotpot_training_schedule.py` | SkillFlow frozen sequence、ordered provider、exact cursor 和 attempt progress | 绑定现有 Hotpot train 顺序和题内 rollout ordinal，不重新切分数据 | 已在内存中解析真实 512 条 train 数据；未发布正式 schedule |
| optimizer 精确续接 | `src/interactive/smoke_trainer.py`、`scripts/train_agentgraph_smoke.py` | SkillFlow 不可变 policy+optimizer checkpoint 与 exact restore identity | 沿用本项目单 `theta` PEFT checkpoint，同时要求紧邻前一步的 policy/optimizer 状态 | 单测通过；本轮没有创建 optimizer 或执行更新 |
| 原子 serving/route 切换 | `src/interactive/policy_sync.py`、`scripts/train_agentgraph_smoke.py` | SkillFlow Supervisor 的 load → model list 验证 → canary → generation switch → old unload → rollback | 当前 Director 路由保存在 `SGLangReceiptDirectorClient`，所以 route switch/rollback 必须通过回调进入同一个 publisher gate | 模拟控制面单测通过；没有真实调用服务 |

正式续接模式是显式启用的。只有配置
`grpo.exact_optimizer_continuation=true` 时才进入该模式，历史 smoke 行为保持兼容。
正式模式要求：

- 明确提供 behavior adapter；
- 从 Step 1 开始保存 optimizer state；
- Step 2+ 必须提供紧邻前一步的 optimizer state；
- behavior adapter 元数据中的 committed step 与 policy version 必须和当前更新严格匹配。

## 三、初始策略的事实边界

现有 `theta_smoke_step_000001` 曾执行过真实的跨数据集 smoke optimizer update，
因此只保留为 warm-start diagnostic policy。它不会被重命名或重新包装成未经训练的
HotpotQA Step 0。

新物化器定义但尚未写入的正式初始状态为：

```text
policy_version    = qwen35-9b-hotpot-step-000000
adapter_name      = theta_hotpot_step_000000
policy_step       = 0
optimizer_updates = 0
```

默认命令只检查配置，不加载模型、不写文件。只有显式指定物化选项后，才允许加载
Qwen3.5-9B 并保存初始 `theta` adapter。未来生成的初始策略 receipt 必须明确记录：

```text
training_performed = false
optimizer_updates  = 0
```

未来激活该初始 adapter 时还必须记录：

```text
policy_published = false
```

因为加载一个已经存在、从未训练过的初始 adapter，不等同于发布训练后的新策略。

## 四、HotpotQA-only 冻结 schedule 边界

schedule 直接读取已经对齐的 `data/agentgraph_v1`，不会生成另一套数据切分。
其约束包括：

- 只接收 `dataset_key=hotpotqa` 且 `split=train` 的记录；
- 在任何结果驱动的执行前固定 train 顺序；
- 拒绝 validation/test task ID；
- 每个 optimizer step 绑定预先声明的 train position 与 task ID；
- 每个任务的 grouped rollout ordinal 固定为 `0..K-1`；
- schedule 与 cursor 都采用 write-once 语义；
- cursor 只能提交紧邻的下一步，并支持精确恢复。

真实数据的无写入 preflight 成功解析了 512 条 HotpotQA train 记录，并构造了一个
两任务、每题两个 rollout 的假设 schedule，共四个 rollout coordinate。这只验证
接口，没有选择或发布未来正式实验 schedule，也没有启动训练。

## 五、adapter 发布与 Director 路由的原子事务

旧 runner 在 adapter 发布成功后，会进入第二次 pause/drain 才更新 Director 路由，
导致“adapter 已发布”和“Director 已切换”不是同一个事务。

新路径在同一个 gate 内执行：

```text
暂停接收新 rollout
→ 等待已接收 rollout 完成
→ 加载、验证并 canary candidate adapter
→ 切换 Director policy + adapter route
→ 卸载旧 adapter
→ 恢复接收 rollout
```

如果在路由切换后失败，系统会先恢复旧 Director 路由，再清理 candidate adapter。
receipt 会分别记录：

- route switch 是否请求和成功；
- route rollback 是否成功；
- 该事务是训练策略发布，还是仅激活已有的未训练 Step-0 adapter。

## 六、Director search space 与当前行为证据

当前本地 Qwen3.5-9B Director 仍然自由选择：

```text
Agent 数量
× 自由文本 contract
× Executor 模型
× 有向 relation
× Output Agent
× continuation / FINISH
```

Runtime 已支持普通 DAG、fan-in、fan-out、并行独立 block，以及有限两阶段双向
block。自由 contract 可以表达 objective、expected input/dependency、artifact 和
completion condition。

通信 envelope 只记录 Runtime 确实知道的事实：source、target、message type、
artifact body、graph revision 和 target dependency。当前 Executor 没有产出可验证的
`confidence` 或 `evidence_refs`，因此没有凭空增加这些字段。

“合法 topology 已实现”不等同于“Director 已经学会使用 topology”。最近一次完整的
128 题 architecture-v3 运行得到：

- 122 个 single-Agent graph；
- 6 个双节点单向 chain；
- 0 个三节点及以上 graph；
- 0 个 workflow 内异构多模型 graph；
- AgentGraph：EM 67.97 / F1 80.23；
- 固定 Direct：EM 72.66 / F1 82.08。

architecture-v4 的 12 题比较受到 sampling coordinate 混淆，不能作为架构增益证据。
architecture-v5 修复了该坐标问题，但没有重新执行模型评测。

这些证据不支持通过强制多 Agent、固定 workflow template、固定 role enum 或 topology
reward 来制造复杂图。当前更符合证据的判断是 Director policy 尚未学会合理编排，
而不是继续手写 HotpotQA workflow 规则。

更早的 Training-ready Step-0 诊断属于独立历史证据：

- development-128：EM 73.44 / F1 81.62；
- untouched-32：EM 71.88 / F1 83.62。

但通信 masking 没有证明 upstream 的正向因果价值，并且仍存在 malformed Output。
本阶段没有生成新分数，因此不声明任何准确率增益。

## 七、模型与 Skill 边界

- Flow-Director 始终是本地 Qwen3.5-9B（`supervisor_theta`），API 模型不能替代。
- 已完成 canary 的八模型 Hotpot Executor catalog 和精确 model-list receipt 保留在
  `MODEL_CATALOG_AUDIT.md`。本轮没有发现、探测或加入新模型，也没有改变冻结 catalog。
- MACE、Bayesian posterior/EVSI、paired probe 和 Skill schema 仍是隔离的未来方法模块。
- 本轮没有发现、总结、验证、激活、检索或注入任何 Skill。
- MACE/Bayesian/Skill 信号没有进入 terminal reward。

## 八、验证结果

- 初始 policy、schedule/cursor、policy sync、runner 接线、optimizer 精确续接和
  trajectory collection 的定向测试：54 项通过；
- 排除无关可选 pandas 数据准备模块后的完整轻依赖 unit/regression suite：
  201 项通过；
- 所有修改 Python 文件的 Ruff 检查：通过；
- 初始策略 CLI 默认 preflight：通过，并明确记录
  `model_load_performed=false`、`optimizer_or_backward_performed=false`、
  `will_write=false`；
- 真实 Hotpot 对齐数据 schedule 无写入 preflight：成功解析 512 条 train 记录，
  没有写 artifact，`training_started=false`。

## 九、尚未执行的后续工作

以下工作必须获得后续明确训练授权，当前不能描述为已完成：

1. 物化未经更新的 `policy_step_000000` adapter；
2. 在本任务所属的 SGLang 服务上原子激活并 canary；
3. 选择并发布正式 Hotpot-only Step-1…N schedule/cursor；
4. 将冻结 schedule 接入正式 runner；
5. 收集 same-task/same-condition grouped rollout；
6. 按顺序执行 action-masked、terminal-only GRPO、backward、optimizer update、
   policy publish、更新后 canary 和 cursor commit；
7. 在固定任务上评测每个 checkpoint，生成完整 performance、workflow、collaboration、
   routing 和 stability 曲线；
8. 只有 Skill evidence gate 完整接通后，才能验证
   candidate → paired evidence → independent validation → ACTIVE → ON/OFF。

当前没有生成 `HOTPOTQA_STEP0_TO_STEPN_MULTIAGENT_SKILL_REPORT.md`，因为现在生成
该报告会错误暗示上述训练和 Skill 实验已经发生。

## 十、最终判定

当前完整推理链、静态训练前置代码与可观测性已经具备，但正式未训练 Step 0、
真实激活事务、正式 schedule 和 Step 0→N 学习均未执行。

```text
架构运行链完整：是
静态训练前置完成：是
正式 Step-0 policy 已物化：否
正式 Step-0 policy 已在线激活：否
Step 0→N 学习已验证：否
Skill 演化已验证：否
允许开始 GRPO：否
允许进入下一个数据集：否
```
