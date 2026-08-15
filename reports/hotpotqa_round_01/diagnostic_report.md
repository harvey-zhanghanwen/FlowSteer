# HotpotQA 第一轮架构验证与根因报告

## 执行结论

本轮已经完成用户要求的第一阶段闭环：

`Question → local Qwen3.5-9B Director → Canvas/AgentGraph → Agent execution`
`→ directed communication → Output Agent → evaluator → trajectory`

固定的 128 条 HotpotQA project-held-out validation 全部完成。Direct 与
AgentGraph 使用完全相同的 Task ID、完整十篇 passage 输入和同一个 normalized
EM/token-F1 evaluator。本轮没有训练、RL、backward、optimizer、权重更新、
MACE、Bayesian 或 Skill 注入；`optimizer_updates=0`。

Stable Zero 已通过：128/128 AgentGraph trajectory 都显式 `finish`，128/128
evaluator receipt 有效，所有 Director turn receipt 均已验证，所有 Output Agent
inbox 均已保存。正式分数是：

| 条件 | 有效/总数 | EM 正确 | Strict EM | Strict F1 |
| --- | ---: | ---: | ---: | ---: |
| Local Qwen3.5-9B Direct | 128/128 | 93 | 72.66 | 82.08 |
| 当前 AgentGraph | 128/128 | 96 | 75.00 | 84.44 |
| AgentGraph − Direct | — | +3 | **+2.34** | **+2.36** |

成对结果由 86 题两者都正确、10 题仅 AgentGraph 正确、7 题仅 Direct 正确、
25 题两者都不满足严格 EM 构成。AgentGraph 有真实但不大的净增益。

## Architecture Completion Report

架构来源和边界已经在 `docs/SOURCE_MAP.md` 逐项登记：

- 直接复用/薄适配 FlowSteer 的 progressive Canvas、单原子 action、step/feedback、
  graph snapshot、trajectory、evaluator 和显式 terminal 边界。
- 直接复用/薄适配 SkillFlow 的 Qwen3.5-9B/SGLang Supervisor 运行边界、短
  multi-hop contract、现有 LoRA evaluation-only load/list/verify/canary 和 bounded
  continuation history。
- 本项目只增加设计文档必需的自由文本 Agent contract、catalog 内模型选择、
  两比特关系、有限双向通信，以及固定 held-out paired evaluation driver。
- 初始 Director prompt 保持短、普通、中性，只声明合法动作、Canvas 状态/反馈和
  catalog 边界；没有注入 workflow 模板、固定角色套路或未验证 Skill。
- MACE、Bayesian posterior/EVSI、Skill 生命周期、GRPO/backward/optimizer 是
  预留方法边界，本轮全部关闭，不能据此声称已实现或已训练。
- Runtime 的有限双向 revision 路径保留了接口和单测，但本轮 128 题没有产生
  双向 relation，不能把它列为已经通过真实模型验证的能力。

架构提交前的验证为 161 个 unit tests 全部通过，相关改动的 Ruff 检查通过。
两条真实 canary 在进入 128 题正式运行前通过。详细报告见
`docs/ARCHITECTURE_COMPLETION_REPORT.md`。

## 固定样本与版本边界

- 数据集：HotpotQA。
- project split：validation；native source split：train。
- 样本数：128，Task ID 唯一且后续架构版本应继续复用同一批。
- 输入：每题保留原始对齐记录的完整十篇 passage；supporting facts 和 ground
  truth 不进入模型输入。
- Direct protocol：一次本地 Qwen3.5-9B 调用，绕过 Director/Canvas/AgentGraph，
  但复用项目 Agent gateway 的轻量 node system wrapper。
- Director：始终为本地 Qwen3.5-9B。
- policy version：`qwen35-9b-smoke-step-0001`。
- inference adapter：`theta_smoke_step_000001`。
- 全批次 policy/adapter/catalog condition 固定；没有批中换权重。

因此，本轮最可信的比较是同一批样本上的 Local Direct 与 AgentGraph 成对差值，
而不是跨论文协议的绝对分数。

## 与论文参考的差距

| 结果 | EM | F1 | 当前 AgentGraph 的差值 |
| --- | ---: | ---: | ---: |
| 当前 paired Local Direct | 72.66 | 82.08 | +2.34 / +2.36 |
| 用户给出的 Qwen3.5-9B Direct reference | 60.94 | 75.70 | +14.06 / +8.74 |
| 用户给出的 FlowSteer reference | 89.84 | 91.20 | -14.84 / -6.76 |
| 用户给出的 SkillFlow reference | 92.19 | 93.95 | -17.19 / -9.51 |

后三行来自论文参考设置，样本 split、prompt、工具、训练状态和 evaluator protocol
并不保证相同，不能解释为严格复现差值。它们只说明当前未训练架构仍明显低于公开
FlowSteer/SkillFlow reference；本地成对结果才隔离了本轮架构条件。

## Workflow 与通信检查

### Director/Canvas 行为

- 431 个 Director turn，平均 3.37 turn/题，范围 3–10。
- 最终图中 113 题为单 Agent，15 题为两 Agent 单向关系。
- 最终节点共 143 个：Qwen3.5-9B local 56、Qwen3.5 Flash 50、
  GPT-4o-mini 22、MiniMax-M2.5 15。
- 128/128 通过显式 `finish` 结束；没有把 `max_rounds` 冒充正常 finish。
- 431 个 turn 中有 7 个无效/被拒绝 action，分布在 5 题；Canvas 全部正确拒绝，
  整题最终都恢复并完成。问题是 Director 的 schema compliance/拒绝后恢复，
  而不是 validator 接受了非法图。
- 4 题在已经 `set_output` 后继续改图并再次执行，共产生 4 个额外 Executor call；
  最终都使用最新 revision，没有 stale result 混用。

### Runtime 与 Agent communication

- 共 147 个 Executor call；独立 block 按既有 Runtime 并发执行。
- 15/15 两 Agent 图的实际 upstream 来源与图中有向关系一致。
- 128/128 Output Agent 都有可定位 inbox；15/15 多 Agent 图都保存了实际传入的
  upstream 内容。
- 7 个两 Agent Wrong Demo 中，Output Agent 也都收到了方向正确的上游内容。
- 15/15 上游 Agent 已经直接输出 `<answer>`；12/15 下游 Agent 逐字复制上游输出，
  14/15 下游 answer span 与上游相同。这说明当前双 Agent 多为“先答后复述”，尚未
  稳定形成 evidence extraction → reasoning → answer 的职责分工。
- 未发现消息丢失、关系方向反转、错误 fan-in、Output Agent 取错节点输出或
  trajectory/evaluator receipt 缺失。
- 每个 Agent 同时收到完整原题和十篇上下文，因此“消息确实到达”不等同于“下游
  确实依赖 upstream”；本轮证据无法隔离下游答案来自原题还是来自通信。

这说明“通信通道是否工作”的答案是肯定的；但“下游 Agent 是否以正确 contract
使用收到的信息”仍有明确问题。

### 图规模的成对表现

| 最终图 | 题数 | Direct EM/F1 | AgentGraph EM/F1 | 成对观察 |
| --- | ---: | ---: | ---: | --- |
| 单 Agent | 113 | 73.45 / 83.42 | 77.88 / 86.60 | Graph 改写/路由总体增益 |
| 两 Agent | 15 | 66.67 / 72.00 | 53.33 / 68.19 | Graph 在同一子集回退 |

两 Agent 子集只有 15 题且是 Director 非随机选择的更复杂题，不能据此断言“增加
Agent 必然有害”。但成对 Direct 仍高出 13.33 EM，结合下述“正确中间答案被
Output Agent 改坏”的实例，这是最值得继续检查的系统性架构信号。

### Output 模型路由的成对观察

| Output Agent 模型 | 题数 | Direct EM | Graph EM |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B local | 51 | 72.55 | 78.43 |
| Qwen3.5 Flash | 43 | 74.42 | 81.40 |
| GPT-4o-mini | 20 | 60.00 | 50.00 |
| MiniMax-M2.5 | 14 | 85.71 | 78.57 |

这些任务不是随机分配给各模型，因此表格不能当作模型排行榜。它只支持一个需要
下一轮验证的 router 假设：候选模型的短答案/格式服从性可能应进入 Output Agent
选择证据，而不能仅依赖便宜、快速权重。

## Wrong Demo 分类

AgentGraph 的 32 个非 EM 样本分为：

- `architecture_regression_candidate`：7。
- `partial_or_overlong_answer`：16。
- `shared_reasoning_or_model_failure_candidate`：9。

其中 25/32 为单 Agent、7/32 为双 Agent。16 个 partial/overlong 中有若干是语义
正确但严格表面形式不一致，不能全部归因于架构。

### 六个代表性首错位置

| Task ID | 现象与首错位置 | 证据分类 |
| --- | --- | --- |
| `hotpotqa:5ac3ad225542995ef918c1da` | Gold/Direct=`no`。本地 Qwen Agent 首次已输出 `<answer>No</answer>`，Director 却继续 modify 并切换 GPT-4o-mini，结果变成 `false`。首错在 turn 2 的不必要 continuation。 | stopping/continuation 架构回归；非通信 Bug |
| `hotpotqa:5ae0fcb855429945ae959492` | reader 已给出精确 `Anna Kournikova`，且原样进入 answerer inbox；answerer contract 要求“clear, complete sentence”，将其扩写并丢失短 answer tag。 | Output role/contract 与全局短答案协议冲突 |
| `hotpotqa:5ae7a9c8554299540e5a5631` | Gold/Direct=`3000 metres steeplechase`。初始 contract 只提取年份、地点和项目数，未回答男女共同项目；后续非法 add 被正确拒绝后直接 finish。 | 初始 contract 偏题 + schema/recovery 弱；更像未训练 Director |
| `hotpotqa:5a8efd3c55429918e830d179` | Gold/Direct=`no`。MiniMax 判断事实正确，但把解释全文放进 `<answer>`，F1 仅 0.1429。evaluator 正确提取 tag，tag 内容本身过长。 | Executor 格式服从/Output router 假设 |
| `hotpotqa:5a89372855429951533612e6` | Gold=`6.213 km long`，Direct/Graph=`6.213 km`，二者 F1=0.8。 | 严格 EM 表面差异；不是 AgentGraph 独有失败 |
| `hotpotqa:5a84918e5542990548d0b2cf` | Gold=`Exeter Book`，二者回答中间实体 `Widsith`；没有完成“诗 → 所在书”的第二跳。 | 共享模型推理失败 + 单 Agent 浅编排 |

### 代表性架构增益

10 个仅 AgentGraph 正确的题说明 Director/contract/model routing 并非没有作用。
例如：`hotpotqa:5abd7ca05542993062266cab` 将 Direct 的 `USA` 收紧为
`Roseau, Minnesota, USA`；`hotpotqa:5ae732685542991e8301cbc3` 将 `Nassau`
收紧为 `Nassau County`；`hotpotqa:5ae27edc5542992decbdcd2d` 将 Direct 的
错误人物 `Frank Oz` 改为正确的 `Roger Christian`。

## 根因分层

### 已确认并修复的工程 Bug

部署中的 SGLang 0.5.15 native `/generate` 接口需要 `sampling_seed`，而不是
OpenAI 边界常用的 `seed`。两条首次 AgentGraph canary 因此得到 HTTP 500；项目
只做了这个具体版本兼容的最小适配，然后只补采失败项。`collection_failures.jsonl`
保留这 2 条已恢复的历史记录；最终没有缺题，也没有重复成功的付费 Direct 调用。

本轮没有发现其他已确认的 Runtime、Agent communication、trajectory 或 evaluator
代码 Bug。

### 有证据的 Architecture 问题/假设

1. Director 缺少“已有合规短答案时何时停止”的稳定判断，可能把正确答案继续改坏。
2. Output Agent contract 可与全局 concise-answer 协议冲突；通信正确仍会在末端失真。
3. 两 Agent 子集成对回退，表明 aggregation/Output role 目前没有稳定兑现额外 Agent
   的价值；不是传输层故障。
4. Director 偶尔生成非法 action，Canvas 能拒绝，但拒绝后的最小恢复策略不稳定。
5. 明确两跳题经常退化为单 Agent；当前自由 search space 没有稳定学会 decomposition
   或 verification。
6. 不同候选模型承担 Output role 时的格式服从差异值得作为 router 证据验证。
7. 当前统一节点提示鼓励每个中间 Agent 都直接输出 `<answer>`，与真正的中间证据
   contract 可能相互抵消，造成双 Agent“先答后复述”。

这些涉及 search space、terminal policy、role contract 和 model router，按用户要求
本轮不自动修改核心方法。

### 更可能来自模型能力、未训练或 evaluator 表面形式

- `Widsith → Exeter Book` 等错误在 Direct 与 Graph 同时出现，首先是多跳推理能力
  或未训练 Director 未能拆解的问题。
- `false` 对 `no`、`6.213 km` 对 `6.213 km long`、完整句对短实体等会被严格 EM
  判错；F1 保留了部分语义重合。不能用宽松语义判分替换正式 evaluator 来虚增结果。
- 当前 policy 是已有 smoke adapter，不是针对这一轮 128 题训练的结果；本轮也没有
  用 validation 更新权重。

## 下一步最值得人工判断的五个问题

1. 是否把“Output Agent 必须只转交/规范化一个 concise `<answer>` span”设为合法
   contract 约束，而不是在 prompt 中软性提醒？
2. Director 看见格式合规、事实自洽的现有输出后，是否应提高 `finish` 优先级，并
   禁止无证据的 model/contract continuation？
3. 两 Agent 图的 Output role 是否应默认保真转交上游实体，只有检测到冲突时才综合，
   从而避免正确中间答案被扩写破坏？
4. 是否在不预埋固定 workflow 模板的前提下，为明确关系链问题增加最小
   decomposition/verification action 证据？
5. 是否把实测的 concise-answer/schema compliance 纳入 Output Agent router；若做，
   如何避免用本轮 validation 直接调参造成泄漏？

## 调用、token 与 latency receipt

| 层 | 逻辑调用 | attempts | input/prompt tokens | output tokens | 累计调用 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct local | 128 | 128 | 209,485 | 1,361 | 69.23 s |
| local Director | 431 | 431 | 951,446 | 13,924 | 196.76 s |
| Workflow Executors | 147 | 147 | 238,807 | 151,479 | 1,588.29 s |
| 合计 | 706 | 706 | 1,399,738 | 166,764 | 1,854.28 s |

累计调用 latency 是各 receipt 的求和，不等于并发运行的墙钟时间。每题的 request
ID、generation seed、provider/model ID、attempt count、token 和 latency 均在相应
Direct record 或 trajectory execution receipt 中。

`qwen3.5-flash` 的 provider usage receipt 有明显计数语义异常：51 个 Executor
call 占 Executor 累计 latency 的 82.82%，并报告 145,394 completion tokens；其中
8 次报告值甚至超过请求记录的 `max_tokens=4096`，但保存的可见答案很短且
`finish_reason=stop`。这更像隐藏推理 token 或 provider usage 口径差异，不能据此
认定通信层截断了输出；因此表中的 provider-reported token 应按 receipt 原样理解。

## 完整 artifacts

- `artifacts/hotpotqa_round_01/selected_tasks.jsonl`：固定 Task ID、Question、Ground Truth。
- `artifacts/hotpotqa_round_01/direct_predictions.jsonl`：Direct answer、evaluator 与调用回执。
- `artifacts/hotpotqa_round_01/agentgraph_trajectories.jsonl`：Orchestrator 输出、每轮
  Canvas/graph、role/model、rendered Agent input、output、实际 upstream/peer message、
  Output inbox、完整 trajectory、token/latency/API receipts。
- `artifacts/hotpotqa_round_01/paired_results.jsonl`：同题 Direct/Graph 成对指标与分类。
- `artifacts/hotpotqa_round_01/wrong_demos.jsonl`：32 个 Wrong Demo。
- `artifacts/hotpotqa_round_01/collection_failures.jsonl`：2 个已恢复的历史 canary 失败。
- `artifacts/hotpotqa_round_01/preflight_receipt.json`、`run_manifest.json`：运行条件、
  固定版本、Stable Zero、进度与最终指标。
- `reports/hotpotqa_round_01/report.json`、`report.md`：机器可读和简版结果。

原始服务日志和重复 EvidenceStore 副本不作为必要结果提交。本报告结束后停止；不进入
TriviaQA 或其他数据集，不启动训练或自动修改方法级架构。
