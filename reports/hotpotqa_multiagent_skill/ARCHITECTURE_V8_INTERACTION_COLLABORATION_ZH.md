# HotpotQA AgentGraph v8 交互编排与协作改进报告

## 1. 结论

本轮已经修复 v7 中“Director 几乎总是提前收敛到单 Agent、后续观察重复且带有过早终止暗示”的交互偏差。v8 在同一批固定 128 个 HotpotQA validation 样本、同一 Qwen3.5-9B Director policy/adapter、同一 Executor model catalog 和同一 evaluator 下完成：

- 128/128 显式 `FINISH`；
- 128/128 evaluator-valid；
- 0 个 max-round terminal failure；
- 0 个 operational/API/evaluator failure；
- 多 Agent 图从 5/128（3.91%）增至 77/128（60.16%）；
- 深度 3 图从 0 增至 9/128（7.03%）；
- 77/128 个任务发生真实、非空的运行时消息交付；
- 9 个任务形成真实的 `intermediate → intermediate → Output` 两跳交付链；
- 22 个任务发生跨模型真实交付，共 25 条跨模型边。

因此，v8 已经解决“几乎全是单 Agent、没有实际协作”的结构塌缩，不再属于过浅的单 Agent 编排。它仍以串行两跳图为主、最大深度为 3，尚未自然产生 fan-in、并行汇聚或 reciprocal 图；但 HotpotQA 本身主要是两跳问答，而且当前深度 3 子集没有显示更高准确率，所以没有证据支持继续强制加深或加入结构奖励。

## 2. 固定比较条件

| 项目 | v7 与 v8 条件 |
|---|---|
| 数据 | 同一批 128 个项目 held-out HotpotQA validation 样本 |
| Director | 本地 Qwen3.5-9B |
| Policy version | `qwen35-9b-hotpot-step-000003` |
| Adapter | `theta_hotpot_step_000003` |
| Agent model catalog | `config/model_catalog_hotpotqa_deep_v6.yaml`，冻结相同顺序 |
| Direct baseline | 同一批已保存的本地 Qwen3.5-9B Direct 结果 |
| Evaluator | 相同 strict EM / token F1 evaluator |
| 本轮训练 | 无训练、无 GRPO、无 backward、无 optimizer step、无权重或 LoRA 更新 |
| 结构奖励 | Agent 数、深度、关系、模型多样性和 Skill 使用奖励均为 0 |

本轮没有设置最低 Agent 数、最低深度、固定角色、固定 workflow 模板或“必须使用多个模型”的规则。图变深来自交互语义修复，不是配额或奖励驱动。

## 3. 改了什么

### 3.1 SkillFlow 风格的真实多轮 continuation

旧实现每轮重新构造一个单独的大 JSON prompt，并在其中重放重构后的 history。v8 改为持续的真实消息序列：

`system → initial user task/Canvas → sampled assistant action prefix → new user Canvas observation → ...`

只有 Canvas 实际消费的 action prefix 会进入下一轮，最近消息对按窗口截断；任务和冻结 catalog 保留在首个 user message。这对应 SkillFlow `training/environment.py` 与 `gflownet_trainer.py` 的 persistent messages / Supervisor continuation 边界，并通过本项目既有字符串 receipt 接口做薄适配。

实现位置：

- `src/interactive/director.py`：transcript schema、编码/解码、真实 continuation；
- `src/interactive/rollout_collector.py`：SGLang chat template 使用真实 messages，并按 `Canvas → continue` 顺序推进。

### 3.2 去掉会诱导过早结束的 observation 字段

从模型可见的中间 observation 中移除了：

- `remaining_rounds` / `max_rounds`；
- `topology_statistics`；
- `structurally_complete`；
- 重复的 reconstructed history；
- 渐进执行结果里的 `output_format`、`exact_single_answer_tag` 和 `answer_tag_count`。

显式 `FINISH` 仍然 fail-closed：如果最终格式或终局条件不合法，Canvas 会拒绝并返回具体错误。也就是说，格式检查没有被删除，只是不再在每个中间回合把“格式已经像答案”误当成“任务已经完成”。这沿用 FlowSteer 的原子 Canvas action → feedback → continuation 和显式 terminal semantics。

### 3.3 更直接但不扩张 search space 的有向关系观察

AgentGraph 的两比特 relation 仍是唯一关系表示和 mutation receipt。v8 只在已有边时额外给 Director 一个派生的 `directed_edges: [{from,to}]` 视图，避免 canonical endpoint 排序后让模型反向理解 source/target。

Search space 仍只有用户 MD 指定的六个原子动作：

`add_agent / modify_agent / remove_agent / set_relation / set_output / finish`

没有新增 topology action、Skill action 或隐藏 workflow 模板。

### 3.4 Executor 明确消费上游 artifact

中间节点和 Output 节点的执行协议现在把已路由 artifact 定义为该节点的声明依赖：除非 artifact 与原任务具体冲突，或 contract 明确要求验证，否则不应静默忽略并从头重做。

这复用 FlowSteer 的下游上下文传递以及 SkillFlow MExec 的 context 消费边界。它只是执行协议约束，不是硬编码答案合并器。

### 3.5 Skill 策略：阶段检索 + 严格证据门控

现有 `SkillEvidencePipeline / SkillStore / SkillQuery` 已接入 Director 每一轮，可根据图阶段动态检索：

- `empty_graph`；
- `construction`；
- `before_final_answer`。

检索模式参考 SkillFlow 的 `src/skills/workspace.py::retrieve` 与 `training/environment.py` 的 policy-visible catalog；ACTIVE 状态、独立验证证据、版本绑定、延迟激活和 paired-effect publication gate 来自用户 MD 的项目适配。

本轮没有版本兼容且通过证据门槛的 ACTIVE Skill，所以配置保持 `skills.enabled: false`、`retrieval_top_k: 0`。因此本轮只验证了 Skill 接线边界，没有测量 Skill gain，也没有把 Wrong Demo 临时写成未验证的 prompt recipe。

### 3.6 协作诊断补齐 reciprocal `peer_draft`

诊断现在同时统计普通 `upstream` artifact 和双向 revision 阶段的 `peer_draft`，并统一要求：

- request revision 等于终态 graph revision；
- message revision 等于终态 graph revision；
- source/target 是终态真实有向边；
- content 非空。

这些证据只能证明 transport，统一标为 `weak`；不能声称下游模型在因果上使用了消息。

## 4. 图深度与真实协作结果

### 4.1 图结构

| 指标 | v7 | v8 | 变化 |
|---|---:|---:|---:|
| 单 Agent | 123/128（96.09%） | 51/128（39.84%） | -72 |
| 双 Agent | 5/128（3.91%） | 68/128（53.13%） | +63 |
| 三 Agent | 0/128 | 9/128（7.03%） | +9 |
| 深度 1 | 124 | 51 | -73 |
| 深度 2 | 4 | 68 | +64 |
| 深度 3 | 0 | 9 | +9 |
| 终态有向边 | 6 | 86 | +80 |
| 平均 Director turns | 3.45 | 5.13 | +1.68 |

v8 拓扑分布为 `single=51`、`serial_2=68`、`serial_3_plus=9`。所有 86 条终态有向边都有对应的运行时非空交付，因此按 transport 证据计算的 effective depth 分布与 structural depth 相同：`1/2/3 = 51/68/9`。

### 4.2 实际通信

| 指标 | v7 | v8 |
|---|---:|---:|
| 有真实消息交付的任务 | 5/128 | 77/128 |
| 真实交付的终态有向边 | 6 | 86 |
| intermediate → intermediate 的任务 | 0 | 9 |
| intermediate → intermediate → Output 完整链 | 0 | 9 |
| 有跨模型真实交付的任务 | 0 | 22 |
| 跨模型真实交付边 | 0 | 25 |

旧 v7 的 `upstream`-only 口径是 4 个任务；补入 reciprocal revision 的 `peer_draft` 后是 5 个任务、6 条边。v8 没有 reciprocal 图，因此新旧口径的 v8 任务数均为 77。

## 5. HotpotQA 指标

| 条件 | Strict EM | Token F1 |
|---|---:|---:|
| 本地 Qwen3.5-9B Direct | 72.66（93/128） | 82.08 |
| v7 AgentGraph | 72.66（93/128） | 80.92 |
| v8 AgentGraph | 71.88（92/128） | 83.05 |
| v8 − v7 | -0.78 pp | +2.13 pp |
| v8 − Direct | -0.78 pp | +0.97 pp |

同题 v7 → v8 的 EM 翻转：

- correct → correct：81；
- wrong → correct：11；
- correct → wrong：12；
- wrong → wrong：24。

这说明结构塌缩已经修复，F1 也有改善，但不能声称 EM 有净提升。当前最合理的结论是：v8 获得了真实多步协作和更完整的答案覆盖，代价是 1 个净 EM 回退。

### 5.1 按深度切片（相关性，不是因果）

| Agent/有效深度 | n | EM | F1 |
|---|---:|---:|---:|
| 1 | 51 | 70.59 | 84.33 |
| 2 | 68 | 73.53 | 82.52 |
| 3 | 9 | 66.67 | 79.79 |

三 Agent 子集样本很小，而且任务难度并不随机分配；该表不能证明深度导致下降。但它足以否定“继续强制加深一定更好”的假设，因此本轮没有增加最低深度、固定三 Agent 模板或结构奖励。

## 6. 典型正确与错误链

### 6.1 正确的三 Agent 链

`hotpotqa:5ac1b7495542994ab5c67dd9`

- `extractor` 提取 Real Damage 的 solo artist；
- `reasoner` 消费 extractor artifact，连接到 Electropop 定义；
- `output_agent` 消费 reasoner artifact；
- 实际交付：`extractor → reasoner → output_agent`；
- 最终答案：`synth-pop`，EM/F1 均为 1。

这证明 v8 不只是把多个 Agent 名字写进图，而是执行了中间到中间、再到 Output 的真实 artifact 链。

### 6.2 Contract 语义漂移导致的错误

`hotpotqa:5ac2c3545542990b17b1548b`

- 原问题问：日本统治时期“以什么事件的结束而结束”；
- retriever artifact 已正确包含 `conclusion of World War II in 1945`；
- Director 后续 contract 把问题漂移成“提取 specific date”；
- analyzer 和 Output 正确执行了错误的 contract，最终输出 `1945`；
- Ground Truth 为 `World War II`。

错误起点不是通信丢失，而是 Director 在多轮重写 contract 时没有保持原问题的答案类型和语义不变量。

### 6.3 Output 边界过宽导致的错误

`hotpotqa:5ae64cab5542991bbc9760be`

- 三段链均找到了 `My Own Worst Enemy`；
- Output contract 同时要求纠正题干中的年份；
- 最终输出包含年份解释和正确歌名，F1 为 0.381，但 strict EM 失败；
- Ground Truth 只要求 `My Own Worst Enemy`。

错误起点是 Output contract 边界，而不是 AgentGraph transport。

## 7. 当前明确问题

1. **动作合法性仍可改善。** v8 有 90/656 个被拒回合（13.72%），高于 v7 的 28/441（6.35%）；parse failure 为 30，高于 v7 的 18。主要拒绝是引用尚不存在的 Agent、relation 两端相同、以及少量终局 wrapper/JSON/model 字段错误。
2. **Contract 语义守恒不足。** 多轮拆分时偶尔改变原问题问的实体类型、事件类型或期望 answer span。
3. **Output contract 偶尔承担额外解释任务。** 这会把已找到的正确短答案扩张为 strict EM 不接受的长答案。
4. **拓扑仍只有串行链。** 当前没有 fan-in、并行汇聚或 reciprocal；不过尚无证据表明 HotpotQA 需要强制增加这些结构。
5. **通信只有 transport 证据。** 86/86 边的 artifact 均到达目标，但未做同图 masking/paired intervention，所以不能声称每个下游输出都在因果上依赖 artifact。
6. **Skill gain 尚未测量。** 没有通过独立验证和版本门控的 ACTIVE Skill；Wrong Demo 只能成为待验证候选，不能直接发布。

## 8. 是否继续加深

本轮答案是：**不继续强制加深。**

v8 已把多 Agent 占比提高到 60.16%，有效深度 2/3 的任务达到 77/128，并产生 9 条真实三 Agent 链；“过浅”问题已经从系统性塌缩变成任务自适应选择问题。继续增加深度配额会违反当前 search-space 中性原则，而且深度 3 子集暂未显示收益。

下一轮最值得先验证的是两个最小、非结构奖励方向：

1. Director 合法动作顺序的 observation 表达是否可以进一步减少 unknown-agent / self-relation 拒绝；
2. contract 是否能显式保持原问题的 answer type / semantic target，而不引入固定角色模板。

这两个失败模式可以形成 CANDIDATE Skill 假设，但必须经过固定 held-out paired validation 后才允许变成 ACTIVE Skill。

## 9. 来源与实现状态

| 模块 | 状态 | 来源 |
|---|---|---|
| Canvas 原子动作、feedback、显式 FINISH | 直接复用/保持 | FlowSteer |
| 持续真实 messages 与 Supervisor continuation | 上游语义复用 + receipt 薄适配 | SkillFlow |
| 两比特 relation 与有向边薄视图 | 必要适配 | 用户 MD + 现有 AgentGraph |
| Executor 上游 artifact 消费协议 | 必要适配 | FlowSteer/SkillFlow 执行边界 |
| graph-stage Skill retrieval | 上游检索模式 + 项目适配 | SkillFlow + 用户 MD |
| ACTIVE/版本/独立验证/paired-effect gate | 项目算法适配 | 用户 MD |
| MACE/Bayesian/Skill 训练闭环 | 本轮未运行 | 预留已有接口 |
| GRPO/backward/optimizer/LoRA sync | 本轮未运行 | 不在本轮范围 |

## 10. 结果文件

- v8 完整机器可读报告：`reports/hotpotqa_multiagent_skill/interaction_v8_dev128.json`
- v8 简表：`reports/hotpotqa_multiagent_skill/interaction_v8_dev128.md`
- v8 完整 trajectory：`artifacts/hotpotqa_multiagent_skill/interaction_v8_dev128/agentgraph_trajectories.jsonl`
- v8 Wrong Demo：`artifacts/hotpotqa_multiagent_skill/interaction_v8_dev128/wrong_demos.jsonl`
- v8 run manifest：`artifacts/hotpotqa_multiagent_skill/interaction_v8_dev128/run_manifest.json`
- v7 对照报告：`reports/hotpotqa_multiagent_skill/director_observation_v7_dev128.json`
