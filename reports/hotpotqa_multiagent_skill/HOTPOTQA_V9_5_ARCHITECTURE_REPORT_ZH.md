# HotpotQA v9.5 编排架构验证报告

## 1. 结论

当前推理链路已经达到 Stable Zero：

`Question → Qwen3.5-9B Flow-Director → AgentGraph Canvas → incremental execution → artifact routing → Format Agent → HotpotQA evaluator → trajectory`

v9.5 在固定 16 条 architecture-development samples 上取得 **EM 93.75 / F1 99.11**，但在架构冻结后、未参与本轮调试的后续 32 条 confirmation samples 上仅取得 **EM 75.00 / F1 84.90**，与同批 Qwen3.5-9B Direct 的 **EM 75.00 / F1 85.42** 基本持平。因此，80%–90% 的 HotpotQA 泛化目标**尚未验收**；16 条开发样本上的 93.75 EM 不能作为 held-out 性能结论。

正式代码保留 v9.5。基于 confirmation Wrong Demo 构造的 v9.6 adaptation 只在 8 条 regression cases 中修复 1 条，EM 仍为 12.50%，且 8/8 都是 `serial_2`，因此按 evidence gate 拒绝，没有进入正式推理路径，也没有发布为 ACTIVE Skill。

## 2. Architecture Completion Report

| 模块 | 来源与实现方式 | 状态 |
|---|---|---|
| Qwen3.5-9B Flow-Director、SGLang/Supervisor 边界 | SkillFlow 运行边界的必要适配 | 已真实运行 |
| Progressive Canvas、单轮一个 atomic graph edit、显式 `FINISH` | FlowSteer 语义复用 | 已真实运行 |
| `add_agent` / `modify_agent` / `delete_agent` / `set_relation` / `set_output` / `finish` | FlowSteer search space 的 free-AgentGraph 适配 | 已验证 |
| 每次 accepted edit 后执行 | FlowSteer progressive execution | 已真实运行 |
| dirty closure、受影响子图重执行、未受影响 artifact reuse | SelfPlayGraphFlowSteer `MultiAgentGraph.dirty_closure` 语义适配 | 已真实运行 |
| Agent model routing、role、contract | SkillFlow provider boundary + AgentGraph Canvas | 已真实运行 |
| 单向、fan-in、fan-out、reciprocal communication | AgentGraph runtime | 单元测试通过；自然 HotpotQA rollout 尚未产生非链式 topology |
| 独立 Format Agent | FlowSteer `Format` Operator 语义适配 | 已真实运行 |
| Format contract isolation | 终端 Format invocation 不注入 free-text contract，只消费单一 routed artifact | 已真实运行 |
| Agent input/output、artifact envelope、Output Agent inbox | FlowSteer/SkillFlow 运行记录边界 | 已保存 |
| HotpotQA EM/F1 | HotpotQA 官方答案归一化规则的 answer-only 适配 | 已验证；不含 supporting-fact/joint 指标 |
| 完整 trajectory、policy/adapter/model/provider/token/latency receipt | FlowSteer trajectory + SkillFlow serving receipt 适配 | 已保存 |
| Skill query context | 复用 `SkillQuery.tags`，加入当前 prefix graph 的 task/stage/topology/role/model/relation/position 条件 | 已实现、单元测试通过 |
| ACTIVE Skill | 必须通过独立 evidence gate 后才可见 | 当前没有，不产生推理增益 |

Stable Zero 判定：**通过**。推理闭环不存在阻塞；253 项单元测试和 76 项 subtests 全部通过。已知问题是自然 rollout 的 topology 仍以串行 DAG 为主、answer-span selection 仍不稳定、Director 存在 parse failure/rejected edit，以及完整 MACE–Bayesian–Skill 闭环尚未接通。

## 3. 指标

指标均为 `hotpotqa.official.answer.v1` 的 **official-compatible answer-only EM/F1**。输入使用每题全部 10 个 passages；没有运行 training、backward、optimizer update、GRPO、MACE、Bayesian update 或 Skill publication。

| 运行 | 样本角色 | Direct EM/F1 | AgentGraph EM/F1 | 结论 |
|---|---|---:|---:|---|
| v9.2 train16 | architecture development | 75.00 / 90.91 | 75.00 / 83.42 | Format failure 明显 |
| v9.3 train16 | architecture development | 75.00 / 90.91 | 87.50 / 97.86 | FlowSteer Format rules 有效 |
| v9.4 train16 | architecture development | 75.00 / 90.91 | 75.00 / 95.08 | free-text Format contract 干扰 |
| v9.5 train16 | architecture development | 75.00 / 90.91 | **93.75 / 99.11** | 15/16 exact match |
| v9.5 confirm32 | 未参与本轮调试的 confirmation samples | **75.00 / 85.42** | **75.00 / 84.90** | 32/32 valid；未证明增益 |
| v9.6 regression8 | 已知 Wrong Demo，仅作回归诊断 | 12.50 / 45.83 | 12.50 / 52.08 | rejected adaptation |

v9.5 confirm32 的错误分布：24 correct、5 partial/overlong answer、2 shared reasoning/model failure candidates、1 architecture regression candidate；没有 operational failure 或 evaluator failure。

## 4. AgentGraph topology

| 运行 | Agent 数 | structural depth | topology family | fan-in / fan-out / reciprocal |
|---|---|---|---|---|
| v9.3 train16 | 2×15，3×1 | 2×15，3×1 | `serial_2`×15，`serial_3_plus`×1 | 0 / 0 / 0 |
| v9.5 train16 | 2×12，3×3，4×1 | 2×12，3×3，4×1 | `serial_2`×12，`serial_3_plus`×4 | 0 / 0 / 0 |
| v9.5 confirm32 | 2×27，3×5 | 2×27，3×5 | `serial_2`×27，`serial_3_plus`×5 | 0 / 0 / 0 |

运行时已经支持 quotient DAG、fan-in、fan-out 和有限 reciprocal communication，并有定向单元测试；但当前 Qwen3.5-9B Director policy 在自然 HotpotQA rollout 中仍然选择串行 DAG。Canvas 构建过程中短暂出现的 disconnected parallel components 没有 artifact exchange，不能计为并行协作。仅通过 prompt 扩展没有消除这一策略偏置；按用户 MD，下一步需要 GRPO 或经过 paired intervention 与独立验证的 Skill prior，而不是强制写死 topology。

## 5. 具体 Demo：逐编辑、逐执行、artifact routing

Task ID：`hotpotqa:5a7e567b55429949594199a0`

- Question：Who is the American internet entrepreneur who founded the company featured on 24 Hours on Craigslist?
- Ground Truth：`Craig Newmark`
- Final Answer：`<answer>Craig Newmark</answer>`
- Evaluator：`hotpotqa.official.answer.v1`
- EM / F1：`1 / 1`
- Flow-Director：本地 Qwen3.5-9B，policy `qwen35-9b-hotpot-step-000003`
- Executor：4 个 Agent 均为本地 Qwen3.5-9B
- Telemetry：9 Director turns，8 executor calls，17 API attempts，64,210 input tokens，660 output tokens，累计 latency 7,210 ms

最终 directed graph：

```text
reasoner → evidence_reader → verifier → format
```

这是 structural depth=4 的串行 DAG，不是非链式 topology。

| Round | Atomic action | 本轮执行 | artifact / feedback |
|---:|---|---|---|
| 0 | `ADD_AGENT evidence_reader` | 执行 `evidence_reader` | 输出 `Craig Newmark` |
| 1 | `ADD_AGENT reasoner` | 执行 `reasoner`；复用 `evidence_reader` | 输出 `Craig Newmark` |
| 2 | `SET_RELATION reasoner → evidence_reader` | dirty closure 只重执行 `evidence_reader`；复用 `reasoner` | `reasoner` 的 `Craig Newmark` 路由给 `evidence_reader`；输出 `Craigslist` |
| 3 | `ADD_AGENT verifier` | 执行 `verifier`；复用前两者 | 生成独立 verification artifact |
| 4 | `SET_RELATION evidence_reader → verifier` | 只重执行 `verifier` | `verifier` 实际消费 `Craigslist`，确认 Craig Newmark 是 Craigslist founder 且是 American Internet entrepreneur |
| 5 | `ADD_AGENT format` | 执行尚未连接的 `format`；复用前三者 | 中间输出 `Craig Newmark` |
| 6 | `SET_RELATION verifier → format` | 只重执行 `format` | `format` 消费 verifier artifact，输出 `Craig Newmark` |
| 7 | `SET_OUTPUT format` | 终端 Format protocol 重执行 `format` | Output inbox 记录 `verifier → format` artifact；输出 `<answer>Craig Newmark</answer>` |
| 8 | `FINISH` | 无新 executor call | 复用既有 artifacts，trajectory 正常终止 |

该 Demo 证明“一个 atomic edit 完成后立即执行”的语义已经实现：新增 Agent 会立即执行；新增 relation 后只重执行 dirty closure；`FINISH` 对未修改图复用既有结果。它也暴露一个真实问题：`reasoner → evidence_reader` 的边方向与 role 命名不够一致，虽然 artifact routing 正常，但 Director 的 role/relation alignment 仍需通过策略学习或 validated Skill 改善。

## 6. confirmation Wrong Demo

| Ground Truth | AgentGraph prediction | 类型 |
|---|---|---|
| `Presque Isle` | `Presque Isle State Park` | partial/overlong answer |
| `1984` | `1984年` | output-language/format failure；v9.6 仅此题修复 |
| `Barack Hussein Obama II` | `Barack Obama` | shortened formal name |
| `rock` | `alternative rock and indie rock` | shared-category synthesis failure |
| `Sarah Janet Maas` | `Sarah J. Maas` | alias/formal-name mismatch |
| `Uniondale, New York` | `Uniondale` | architecture regression candidate；Direct 为 exact match |
| `alternative rock virtual band Gorillaz` | `Gorillaz` | partial answer under strict reference |
| `Richard Ford` | `Robert E. Howard` | comparison reasoning failure |

这些错误不是 evaluator 算错：HotpotQA 官方归一化会处理大小写、标点、冠词和空白，但不会把别名、缩写、地理限定缺失、额外语言后缀或语义不同的答案自动视为 exact match。

## 7. 用户 MD 实现状态

不使用单一百分比，因为“接口存在”“数值单元测试通过”“真实端到端闭环完成”不是同一完成等级。

### 已实现并真实验证

- AgentGraph、free-text Agent contract、per-Agent model selection、two-bit relation encoding。
- Progressive Canvas、atomic graph editing、execution feedback、显式 terminal semantics。
- incremental execution、dirty closure、artifact reuse、Agent communication receipt。
- 独立 Format Agent、Output Agent inbox、answer-only EM/F1、完整 trajectory。
- Qwen3.5-9B/SGLang inference boundary、固定 policy/adapter receipt。

### 已实现原语或接口，但未完成真实闭环

- fan-in/fan-out/reciprocal runtime：单元测试通过，Director 自然行为未验证。
- Action-Masked One-Pass GRPO、LoRA update/sync：此前只完成小规模 smoke，本轮禁用。
- MACE/LinUCB 与 Bayesian posterior 原语：数值测试存在，未接入完整 exploration epoch。
- Skill schema、store、evidence gate、lifecycle、stage-conditioned retrieval：接口与测试存在，没有 ACTIVE Skill 或 held-out terminal gain。

### 尚未完成

- same-prefix paired intervention 的完整执行器。
- whole-rollout posterior sampling、EVSI probe scheduling 与双时间尺度训练循环。
- Skill automatic discovery、independent confirmatory validation、publication/suspension 的完整闭环。
- HotpotQA supporting-fact EM/F1 与 joint EM/F1。
- 80%–90% 的独立 held-out answer EM 验收。

因此，MD 中的**推理与编排主链主体已完成并达到 Stable Zero**；MACE–Bayesian–Skill 的探索、验证、发布和训练闭环尚未完成，不能表述为“MD 已全部实现”。

## 8. 证据位置

- v9.5 development report：`reports/hotpotqa_multiagent_skill/incremental_graph_v9_5_train16.{json,md}`
- v9.5 confirmation report：`reports/hotpotqa_multiagent_skill/incremental_graph_v9_5_confirm32.{json,md}`
- v9.6 rejected adaptation report：`reports/hotpotqa_multiagent_skill/incremental_graph_v9_6_regression8.{json,md}`
- 完整本地 trajectory：`artifacts/hotpotqa_multiagent_skill/incremental_graph_v9_5_train16/agentgraph_trajectories.jsonl`
- 完整 confirmation trajectory：`artifacts/hotpotqa_multiagent_skill/incremental_graph_v9_5_confirm32/agentgraph_trajectories.jsonl`

正式架构 commit：`f80b6ae` (`architecture: add incremental Format Agent graph execution`)。
