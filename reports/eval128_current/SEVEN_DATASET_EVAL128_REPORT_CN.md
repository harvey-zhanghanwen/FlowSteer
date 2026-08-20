# 七数据集固定 128 样本评测报告

## 1. 报告状态与结论边界

本报告只汇总已经落盘且可由 runner receipt、trajectory 和 evaluator receipt 复核的结果。HotpotQA、TriviaQA、AIME family development-128、HealthBench Professional 与 WebShop 已完成固定 128 样本评测。ALFWorld 已对固定 128 任务完成第一遍尝试及一次只针对缺失 task ID 的 exact-resume；107 条取得有效 trajectory，21 条在两次尝试中均达到单任务 300 秒 wall-clock boundary。SWE-bench 已完成 regular-dev 128 任务、仓库和 Coding Agent Tool contract 的运行前适配，但官方 Docker harness 在 runtime preflight 阶段不可用，因此没有模型 Coding trajectory，也没有官方 `resolved` receipt。

当前条件没有执行 GRPO、backward、optimizer update、LoRA publication、Bayesian update 或 Skill publication；各条件均没有 ACTIVE Skill injection。Direct 与 AgentGraph 属于不同推理协议，差值仅作描述性比较，不构成配对因果效应估计。由于不同数据集主指标不可直接相加，且 SWE-bench 官方主指标不可测，本报告不计算跨数据集宏平均。

## 2. 七数据集结果总表

| 数据集 | 固定样本/任务 | Direct Local Baseline | AgentGraph | 有效 evaluator receipt | 当前状态 |
|---|---:|---:|---:|---:|---|
| HotpotQA | 128 | EM 70.31%；F1 78.68% | EM 72.66%；F1 82.05% | 128/128 | 已完成 |
| TriviaQA | 128 | EM 35.16%；F1 40.82% | EM 50.78%；F1 63.36% | 128/128 | 已完成 |
| AIME family development-128 | 128 | integer accuracy 5.47%（7/128） | integer accuracy 48.44%（62/128） | 128/128 | 已完成；不是官方 AIME 2026 |
| HealthBench Professional | 128 | strict mean raw score 0.1653（123/128 valid） | strict mean raw score 0.1709（124/128 valid） | Direct 123/128；AgentGraph 124/128 | 已完成；指标不是 accuracy |
| WebShop | 128 | native success 19.53%（25/128） | native success 16.41%（21/128） | Direct 127/128；AgentGraph 128/128 | 已完成；AgentGraph -3.125 pp |
| ALFWorld | 128 | native success 21.09%（27/128） | strict native success 20.31%（26/128）；completed-only 24.30%（26/107） | Direct 128/128；AgentGraph 107/128 | 已完成第一遍与一次 exact-resume；21 条 operational timeout |
| SWE-bench regular-dev | 128 | 不可测 | 不可测 | 0 个官方 `resolved` receipt | 官方 Docker harness runtime preflight 不可用；不是 0% |

HotpotQA 的 AgentGraph 相对 Direct 为 `+2.34` 个 EM 百分点和 `+3.37` 个 F1 百分点；TriviaQA 为 `+15.62` 个 EM 百分点和 `+22.55` 个 F1 百分点。AIME family development-128 的 integer accuracy 为 `+42.97` 个百分点。HealthBench Professional 的 strict mean raw score 为 `+0.0056`（`+0.5581` 个百分点）；该差值不是 accuracy 增量。ALFWorld 固定分母的 AgentGraph 相对 Direct 为 `-0.78125` 个百分点，但该差值包含 21 条 AgentGraph operational timeout，不能解释为纯任务能力差异。HotpotQA 与 TriviaQA 的 AgentGraph 各有 125 条显式 `FINISH` 和 3 条 terminal failure；AIME family 有 108 条显式 `FINISH` 和 20 条 `max_rounds` terminal failure；HealthBench AgentGraph 有 124 条有效 evaluator receipt、4 条上下文长度 operational failure。

## 3. 数据协议与官方主指标口径

| 数据集 | 当前固定评测条件 | 评价指标与判定边界 |
|---|---|---|
| HotpotQA | 项目 `validation` 128；模型输入为带上下文的 `TaskRecord.question`，ground truth 仅进入 evaluator | HotpotQA official answer normalization 下的 Exact Match 与 token-level F1；answer span 从 `<answer>...</answer>` 提取 |
| TriviaQA | 项目 `validation` 128；模型输入为 question-only，accepted answers 仅进入 evaluator | TriviaQA official answer normalization 下相对 accepted-answer set 的 Exact Match 与 token-level F1 |
| AIME family | development-128 为 AIME 2025 的 30 题加历史 AIME held-out 98 题；AIME 2026 官方 30 题保持为未运行 final test | 整数答案 Exact Match。development-128 不能标记为“官方 AIME 2026 准确率” |
| HealthBench Professional | 固定 development 128；rubric、reference answer 与 judge state 均为 evaluator-only | reference-judge mean raw score；不得改称 accuracy，也不得替代私有官方 leaderboard 指标 |
| WebShop | 固定 validation 128；每个 episode 由同一个 request-scoped environment session 执行 | 上游环境 terminal `success` 的 success rate；非终局文本相似度不计成功 |
| ALFWorld | 固定 validation 128；pinned game/seed 与原生 admissible-action observation | 上游环境 terminal success rate；不得以动作文本相似度替代环境成功 |
| SWE-bench | regular-dev 128；SWE-bench Verified 500 保留为 final-only | 官方 SWE-bench Docker harness 的 `resolved` / resolved rate；本地测试通过或非空 diff 均不能替代 `resolved` |

所有条件均要求 ground truth、aliases、rubric、environment reward/`won` 与 SWE-bench resolution 保持 evaluator-only。SWE-bench runner manifest 明确记录 `prompt_source=TaskRecord.question`、`ground_truth_role=evaluator_only` 和 `evaluator_payload_role=evaluator_only`。

## 4. 架构来源与本项目必要适配

本轮实现遵循 `SkillFlow > FlowSteer > 本项目必要适配` 的来源优先级：

1. **FlowSteer 直接复用边界**：progressive Canvas 的 atomic edit → execution → feedback 循环、显式 `FINISH`、action mask、turn/trajectory record，以及在每次被接受的 Canvas edit 后执行当前 graph 的交互编排语义。
2. **SkillFlow 直接复用边界**：本地 Qwen3.5-9B/SGLang Supervisor 边界、`StructuredAction`/Tool Registry、bounded ReAct execution、RetrievalIndex、MedRAG、RAGEN environment、SWE-bench detached worktree 与 official harness evaluation contract。
3. **必要兼容层**：在保留 FlowSteer scheduler 的前提下调度 `reasoning`、`react`、`coding` Agent；把 SkillFlow 的 Tool/environment/Coding observation 写入 FlowSteer trajectory；用 task-scoped Tool registry 约束工具可用性；把异构 Executor 的 provider/model receipt、token、latency、Tool receipt 和 environment revision 统一持久化。
4. **项目方法新增**：typed `CommunicationEnvelope`、free-form Agent contract、异构 Agent model selection、有限双 Agent reciprocal block，以及项目 MD 指定的 MACE/Bayesian/Skill evidence gate。后者在当前评测条件中没有执行或发布，不能据接口存在声称 Skill 已产生效果。

逐文件的上游类、函数、直接复用项和不兼容原因记录在 `docs/SOURCE_MAP.md`；能力与当前实现状态记录在 `DATASET_CAPABILITY_MATRIX.md`。

## 5. HotpotQA：自然拓扑与实际执行案例

HotpotQA 128 条最终 graph 的 topology family 分布为：`serial_2=100`、`serial_3_plus=25`、`single=2`、`reciprocal=1`。其中 structural depth ≥ 3 的 trajectory 为 25 条。以下案例来自自然 Flow-Director 轨迹，不是强制 topology probe。

### 5.1 正确案例：二节点串行图

- **Task ID**：`hotpotqa:5a7a06935542990198eaf050`
- **问题**：Which magazine was started first Arthur's Magazine or First for Women?
- **Ground Truth**：`Arthur's Magazine`
- **终图**：`verifier(qwen3.5-flash, verifier) → protocol_out(MiniMax-M3, format)`
- **执行顺序 receipt**：`[[verifier], [protocol_out]]`
- **实际通信 receipt**：`verifier → protocol_out` 传递一个 `json` artifact，包含 `Arthur's Magazine: 1844`、`First for Women: 1989` 和 `1844 < 1989`；Output Agent inbox 记录该 artifact，`tool_receipt_count=0`。
- **最终输出**：`<answer>Arthur's Magazine</answer>`
- **Evaluator**：EM `1.0`，F1 `1.0`，显式 `FINISH=true`。

这个案例证明执行器遵守有向依赖：上游 verifier 先执行，Format Agent 只接收一个 routed upstream artifact 并序列化答案。正确性证据来自 evaluator receipt；通信依赖仅由 delivery receipt 证明为 weak evidence，不能由答案正确反推为 causal dependence。

### 5.2 错误案例：reciprocal 结构被接受，但没有形成 reciprocal execution receipt

- **Task ID**：`hotpotqa:5ae0d91e55429924de1b7198`
- **问题**：The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee, who received a star on the Hollywood Walk of Fame in what year?
- **Ground Truth**：`2006`
- **早期有效执行**：在 turn 6 和 turn 12，单一 Format Agent `extractor(gpt-4o-mini)` 均实际输出 `<answer>2006</answer>`；这两次有 executor receipt，但 Director 没有在该状态完成终止。
- **后续结构编辑**：Director 添加 `reader(gpt-4o-mini, retriever)`，并将 `extractor ↔ reader` 改为 reciprocal relation。
- **执行失败 receipt**：`reader` 被配置为 `execution_mode=reasoning`，同时声明 `qa-retrieval.read`；Runtime 两次返回 `AgentRuntimeError: reasoning agent 'reader' cannot declare allowed_tools`，相应 turn 的 executor call 数为 0，没有 reciprocal message delivery。
- **终止失败 receipt**：后续 `FINISH` 被拒绝，原因为 `Format Agent must be a singleton terminal component`；trajectory 达到 `max_rounds`，`explicit_finish=false`，持久化 `final_answer=null`。
- **Evaluator**：EM `0.0`，F1 `0.0`。

该案例在 reciprocal block 阶段的首个执行失败点是 Agent execution mode 与 Tool capability contract 不一致；随后是 terminal semantics 违规。它不是模型不知道答案，也不能作为 reciprocal communication 有效执行的证据。最终 topology diagnostic 把结构记为 reciprocal，但 execution receipt 明确显示该 reciprocal block 从未运行。

## 6. TriviaQA：实际 reciprocal 通信与 fan-in 失败案例

TriviaQA 128 条最终 graph 的 topology family 分布为：`serial_2=100`、`serial_3_plus=27`、`fan_in=1`。此外，一条最终归类为 `serial_2` 的 trajectory 在其 strongly connected component 内实际执行了 reciprocal draft → revision；因此仅看最终 topology family 会漏掉这次双向通信。

### 6.1 正确案例：reciprocal draft → revision 后串行输出

- **Task ID**：`triviaqa:tc_39`
- **问题**：Which country does the airline Air Pacific come from?
- **Accepted answer**：包含 `Fiji` 的 26 个 accepted aliases。
- **Graph**：`agent-1(gpt-4o-mini, explorer, react) ↔ agent-2(gpt-4o-mini, dispatcher, reasoning) → agent-3(gpt-4o-mini, format)`。
- **执行顺序 receipt**：`[[agent-1, agent-2], [agent-3]]`；共 5 个 executor calls。
- **双向通信 receipt**：
  - draft：`agent-1` 产生 `final artifact`；`agent-2` 产生 `Air Pacific is the national airline of Fiji. DONE`；
  - revision：`agent-2 → agent-1` 以 `candidate` 传递 Fiji 事实，`agent-1` 修订为 `Air Pacific is the national airline of Fiji.`；`agent-1 → agent-2` 传递 candidate，`agent-2` 保留 Fiji 结论；
  - terminal edge：`agent-2 → agent-3` 传递 artifact，Format Agent 输出 `<answer>Fiji</answer>`。
- **Tool receipt**：`agent-1` 在 draft phase 以两轮 ReAct 调用 `qa-retrieval.search` 1 次；revision phase 直接 `complete`，该 phase 的 `tool_calls=0`。因此该案例同时证明一次检索 Tool 调用、reciprocal message delivery 与正确终局，但单题 receipt 不能隔离检索或 reciprocal communication 的因果贡献。
- **Evaluator**：EM `1.0`，F1 `1.0`，显式 `FINISH=true`。

### 6.2 错误案例：并行检索分支 → semantic fan-in → Format Agent

- **Task ID**：`triviaqa:tc_84`
- **问题**：On which Caribbean island did Princess Diana spend her first Christmas after her divorce was announced?
- **Ground Truth**：accepted aliases 对应 `Barbuda`。
- **Graph**：
  - `searcher(deepseek-v4-flash, executor, react)`；
  - `searcher_v2(gpt-4o-mini, executor, react)`；
  - 两个并行分支均指向 `reasoner(qwen3.5-flash)`，形成 fan-in；
  - `reasoner → formatter(gpt-4o-mini, format)`。
- **执行顺序 receipt**：`[[searcher_v2], [searcher], [reasoner], [formatter]]`；topology family=`fan_in`，structural depth=`3`，max width=`2`。
- **实际通信 receipt**：`searcher` 报告现有 passages 没有明确答案；`searcher_v2` 给出未被 passages 支持的 `Nevis`。`reasoner` 在 fan-in artifact 中明确标注 Nevis 未受检索证据支持，却仍把该候选保留在输出 artifact；Format Agent 随后抽取为 `<answer>Nevis</answer>`。
- **Evaluator**：EM `0.0`，F1 `0.0`，显式 `FINISH=true`。

该失败属于 evidence conflict resolution 与 semantic grounding 失败，不是通信丢失：两条上游 artifact、fan-in inbox 和 terminal artifact 均有 execution receipt。并行分支提供了互相矛盾的候选，但 semantic fan-in 没有把“unsupported”约束落实为 candidate rejection，Format Agent 又按合同只做 span extraction，最终传播了错误候选。

## 7. AIME family development-128：整数答案评测

该条件由 AIME 2025 的 30 题与历史 AIME 的 98 题构成，不包含官方 AIME 2026 final test。Direct 为 `7/128=5.47%`，AgentGraph 为 `62/128=48.44%`，两侧均有 128/128 有效 evaluator receipt。AgentGraph 最终 topology family 为：`single=8`、`serial_2=114`、`serial_3_plus=5`、`parallel=1`；没有 reciprocal、fan-in 或 fan-out。唯一自然产生的非链式 Canvas 没有完成执行，因此不能提供 AIME 正确非链式案例。

### 7.1 正确案例：异构三节点串行图

- **Task ID**：`aime-historical:2000:i:06`
- **Ground Truth**：`997`
- **Direct 输出**：`166165`，integer Exact Match `0`。
- **Graph**：`analyzer(gpt-4o-mini) → solver(Qwen3.5-9B local) → Format Agent(MiniMax-M3)`。
- **执行证据**：三个节点均有 executor receipt，两个有向边均有 artifact delivery receipt；Format Agent 输出整数 `997`。
- **Evaluator**：integer Exact Match `1`。

### 7.2 错误非链式案例：串行分量与孤立 Output Agent

- **Task ID**：`aime-historical:2001:i:13`
- **终图**：一条三节点串行分量，加一个孤立 Output Agent；diagnostic family=`parallel`。
- **失败点**：并非并行推理产生了错误答案，而是所有 Agent 未形成到唯一 Output Agent 的可达路径；20 轮后 `max_rounds`，没有 terminal artifact，也没有有效非链式 execution receipt。
- **Evaluator**：integer Exact Match `0`。

另一个完整执行但语义错误的串行案例为 `aime-historical:2000:ii:06`：错误中间结果沿三节点链传播，最终输出 `100`，Ground Truth 为 `181`。这说明 Format Agent 只能规范化已有候选，不能修复上游数学推导。

## 8. HealthBench Professional：reference-judge rubric

Direct 有效 receipt 为 123/128，strict fixed-denominator mean raw score 为 `0.1653`；AgentGraph 有效 receipt 为 124/128，strict mean raw score 为 `0.1709`。失败 receipt 持久化为 HTTP 400；对应运行日志将 Direct 的 5 条和 AgentGraph 的 4 条失败解析为 `input_tokens + max_tokens > 8192`。该数值是 public simple-evals reference judge 的 rubric raw score，不是 accuracy，也不是私有 leaderboard 指标。

124 条有效 Graph 的 topology family 为：`empty=1`、`single=34`、`serial_2=58`、`serial_3_plus=21`、`fan_in=2`、`mixed=8`；其中 10 条为实际执行的非链式 DAG。

### 8.1 高分非链式案例：fan-out、fan-in 与 MedRAG

- **Task ID**：`healthbench-professional:2142fe540982b1ca76ab54dd6280d831`
- **任务**：鉴别 halogenoderma 与 squamous cell carcinoma。
- **Graph**：`planner → retriever`，同时 `planner → synthesizer`、`retriever → synthesizer`；构成 fan-out 后的 fan-in。
- **执行证据**：MedRAG search receipt 成功；synthesizer inbox 实际收到 planner 与 retriever 两个 upstream artifact。
- **Evaluator**：reference judge `21/21`，raw score `1.0`。
- **因果边界**：Direct 同样为 `1.0`，且两份上游 artifact 内容存在重复，因此该案例证明非链式执行与通信可用，不证明 topology 带来净增益。

### 8.2 低分非链式案例：terminal output collapse

- **Task ID**：`healthbench-professional:bf9e5af57ef45233ad32ab7b57e0e2e6`
- **任务**：生成临床转诊信。
- **Graph**：validator 分别指向 drafter 与 terminal validator，drafter 再指向 terminal validator，形成 fan-out/fan-in。
- **执行证据**：两条上游完整 artifact 均送达 Output Agent；不存在 message delivery 丢失。
- **最终输出**：`<answer>Referral Letter Draft</answer>`，没有输出已生成的转诊信正文。
- **Evaluator**：reference judge `0/8`，raw score `0`。

首个失败点是 Output Agent 的 instruction following 与 terminal artifact selection，而不是图调度或 Agent communication。

## 9. WebShop：原生环境 success rate

Direct 的 native terminal success 为 `25/128=19.53%`，AgentGraph 为 `21/128=16.41%`，差值为 `-3.125` 个百分点。Direct 有 1 条 `environment_graph_callback_failed`，因此 evaluator-valid 为 127/128；AgentGraph 为 128/128。graded environment return 的 strict mean 分别为 `0.4561` 与 `0.5169`，它表示部分任务进展，不能与二值 success rate 混称准确率。

AgentGraph 最终 topology family 为：`single=90`、`serial_2=30`、`serial_3_plus=7`、`mixed=1`。201 次 Executor call 覆盖 7 个模型；20/128 条 trajectory 实际使用了两个或更多模型。Canvas 的 128 条 trajectory 均显式 `FINISH`，但 24 条环境 episode 达到 step limit；Canvas termination 与 environment termination 是两个不同状态。

### 9.1 正确案例：ReAct shopping policy → Format Agent

- **Task ID**：`webshop:00624`
- **任务**：购买低于 40 美元、无毒且易清洁的 resin DIY nail-art 产品。
- **Graph**：`researcher(gpt-4o-mini, ReAct) → format_agent(MiniMax-M3, format)`；topology family=`serial_2`。
- **原生 action trace**：`search[resin diy nail art non toxic easy to clean under 40 dollars] → click[b07fp5htcc] → click[buy now]`。
- **通信证据**：researcher 把包含 Purchased terminal observation 的 artifact 送达 format_agent；Output Agent 输出所购 ASIN、商品名与价格。
- **Evaluator**：native success `1`、environment return `1`、environment terminal `1`、steps `3`。
- **配对结果**：同题 Direct 达到 10 步上限且 success `0`。这是 `agentgraph_higher_success` 的 5 条样本之一，但单题结果不能推出总体方法增益。

### 9.2 错误非链式案例：fan-out/fan-in 后环境步数耗尽

- **Task ID**：`webshop:00530`
- **任务**：购买价格低于 60 美元的灰色无线充电立体声耳机。
- **Graph**：`sherpa(planner, qwen3.5-flash) → executor(operator/ReAct, qwen3.5-flash)`，同时 `sherpa → aggie(Format Agent, qwen3.5-flash)`、`executor → aggie`；构成 fan-out 与 fan-in。
- **执行证据**：三节点均有 executor receipt；aggie inbox 实际收到 planner strategy 与 executor environment observation 两个 artifact。
- **失败点**：ReAct executor 在 search、click 与 back 之间反复切换，10 个环境步内没有执行 `Buy Now`；terminal observation 未满足任务。
- **最终输出**：`<answer>click[b08m8p1zg9]</answer>`。
- **Evaluator**：native success `0`、environment return `0`、environment terminal `0`。

该批 128 条中不存在正确的自然非链式 WebShop trajectory。另有 `webshop:00607` 在中间 Canvas turn 执行过 parallel graph，同样失败；因此不能据此声称非链式 topology 在 WebShop 上产生了正增益。

## 10. ALFWorld：原生环境 success rate

Direct 的 native terminal success 为 `27/128=21.09%`。AgentGraph 在第一遍得到 83 条有效 trajectory；随后 exact-resume 只重试 45 个缺失 task ID，新增 24 条有效 trajectory，最终为 107/128。其 fixed-denominator strict success 为 `26/128=20.31%`，completed-only success 为 `26/107=24.30%`。剩余 21 个任务在两次尝试中均达到 300 秒 wall-clock boundary；这些记录是 operational failure，不是原生环境失败 evaluator receipt。第一遍及续跑累计保存 66 条 timeout receipt，其中 24 个 task ID 后续恢复成功、21 个 task ID 两次超时。

107 条有效 AgentGraph trajectory 全部显式 `FINISH`，没有 terminal failure。最终 topology family 为：`single=54`（11 success）、`serial_2=42`（10 success）、`serial_3_plus=11`（5 success）；最终图中不存在 fan-in、fan-out、reciprocal 或 disconnected parallel。progressive Canvas 历史中有 3 次真实执行的中间 `parallel_disconnected` graph（`alfworld:train:00039`、`00012`、`00194`），三题均失败，节点之间没有 relation 或 message delivery；全批次没有实际执行的 fan-in、fan-out 或 reciprocal graph。最终 graph 节点的模型分配为：`gpt-4o-mini=86`、`qwen3.5-9b-local=50`、`qwen3.5-flash=15`、`deepseek-v4-flash=15`、`MiniMax-M3=4`、`MiniMax-M2.5=1`。这些是 final graph node assignment，不是 provider call 次数。

### 10.1 正确案例：planner → ReAct executor → Format Agent

- **Task ID**：`alfworld:train:00028`
- **任务**：`put some candle on toilet.`
- **Graph**：`planner(qwen3.5-9b-local) → executor(deepseek-v4-flash, ReAct) → output(qwen3.5-9b-local, format)`；topology family=`serial_3_plus`。
- **执行顺序 receipt**：`[[planner], [executor], [output]]`。
- **原生 action trace**：`go to countertop 1 → take candle 1 from countertop 1 → go to toilet 1 → move candle 1 to toilet 1`。
- **通信证据**：planner artifact 送达 executor；executor 的 terminal environment observation 送达 output。四次环境 transition 均有 revision 与 Tool receipt。
- **最终输出**：`<answer>move candle 1 to toilet 1</answer>`。
- **Evaluator**：native success `1`、environment terminal `1`、steps `4`。
- **Direct 对照**：同题也成功，但使用 15 个环境步。

### 10.2 错误案例：disconnected parallel → 补边为串行图后仍失败

- **Task ID**：`alfworld:train:00039`
- **任务**：`heat some egg and put it in sidetable.`
- **初始约束失败**：首个三节点 graph 含两个 `alfworld.environment` owner，被 stateful-Tool ownership constraint 拒绝，没有执行。
- **中间非链式执行**：删除一个 Executor 后形成两个无 relation 的并行节点。environment-policy Agent 输出 `You move the pot 1 to the sidetable 1.`，Output Agent 独立输出 `heat_egg`；两节点均有 executor receipt，但没有 message delivery，因此这是 disconnected parallel execution，不是多 Agent 协作。
- **后续 progressive edit**：Director 添加 `environment-policy Agent → Output Agent`，最终 topology family 变为 `serial_2`。execution cache 复用上游环境 episode，只重新执行 Output Agent；其 inbox 实际收到上游 artifact 与 50 条 Tool receipt。
- **语义失败**：environment-policy Agent 先前往 stoveburner，却选择 pot 而不是 egg；随后在 stoveburner 与 sidetable 上反复 `take pot` / `move pot`，达到 50 个环境步。
- **最终输出**：`<answer>move pot 1 to sidetable 1</answer>`。
- **Evaluator**：native success `0`、steps `50`；step-limit termination 不能解释为任务成功。

该错误的首个任务语义失败点是 environment policy 的 object grounding 与 state tracking，不是 Agent communication 丢失。它同时证明 progressive Canvas 能从中间 disconnected parallel graph 补边为串行 graph 并复用已执行节点，但不能证明非链式协作有效：中间并行节点没有通信，最终结果仍失败。当前批次没有正确的自然非链式 ALFWorld 案例，也不据此声称该数据集不需要非链式 topology。

## 11. SWE-bench 适配状态

### 11.1 已完成的代码与数据边界

- regular-dev 固定 128 个 instance 已完成选择；`sqlfluff/sqlfluff=50`、`pvlib/pvlib-python=63`、`marshmallow-code/marshmallow=9`、`pylint-dev/astroid=6`。
- 四个仓库镜像均可用，128/128 个 `base_commit` 可解析，`unavailable_count=0`；运行前检查只使用 `instance_id`、`repo`、`base_commit`，没有读取 solution fields。
- Coding Agent Tool contract 复用/薄适配 SkillFlow 的 detached worktree、`list_files`、`search_code`、`view_file`、`bash`、`str_replace_editor`、AST file map、`run_tests` 和 workspace diff；多文件 patch 使用官方 Codex CLI `apply_patch` entry point。
- task-pinned no-model canary `sqlfluff__sqlfluff-4764` 已通过：`str_replace_editor create → Codex apply_patch → run_tests → diff → bash → cleanup`。该 canary 只证明 Tool/worktree contract 可执行，不是 benchmark task resolved。
- Coding completion freshness contract 要求：最后一次产生变更的 edit 之后，必须有后续 `run_tests` observation，再有后续 changed `diff`，最后才能 `complete`。
- Coding execution、repository Tools、detached worktree、SWE-bench adapter、v2 config 与 completion runner 的 43 项定向回归测试通过；这些测试不替代官方 benchmark harness。

### 11.2 当前阻塞与指标解释

runner 在调用任何模型 Coding Agent 之前以 `failed_runtime_preflight` 终止，错误为 `SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable`。因此：

- 模型 Coding trajectory：`0`；
- 官方 `resolved` receipt：`0`；
- resolved rate：**不可测**，不是 0%；
- 不允许用 Tool canary、本地单测通过或 non-empty diff 代理官方 `resolved`。

当前可以准确表述为“regular-dev 数据、仓库、Coding Agent Tool contract 与 fail-closed evaluator boundary 已适配；官方 harness 的运行权限是外部阻塞”。

## 12. 证据索引

- HotpotQA 汇总：`reports/qa_tool_react_exact_wire_v4_development/hotpotqa_report.json`
- HotpotQA trajectory：`artifacts/qa_tool_react_exact_wire_v4_development/hotpotqa/tool_agentgraph_trajectories.jsonl`
- TriviaQA 汇总：`reports/qa_tool_react_exact_wire_v3_stable_zero/triviaqa_report.json`
- TriviaQA trajectory：`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl`
- AIME family 汇总：`reports/eval128_current/aime_family_report.json`
- AIME family trajectory：`artifacts/eval128_current/aime_family/agentgraph_trajectories.jsonl`
- HealthBench 汇总：`reports/healthbench_professional_medrag_tool_stable_zero_v2/development_report.json`
- HealthBench trajectory：`artifacts/healthbench_professional_medrag_tool_stable_zero_v2/development/agentgraph_trajectories.jsonl`
- WebShop 汇总：`reports/webshop_ragen_environment_native_action_v4_stable_zero/development_report.json`
- WebShop trajectory：`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/agentgraph_trajectories.jsonl`
- ALFWorld 汇总：`reports/alfworld_ragen_required_actor_v2_stable_zero/evaluation_report.json`
- ALFWorld trajectory：`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl`
- ALFWorld timeout history：`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/collection_failures.jsonl`
- SWE-bench runner manifest：`artifacts/eval128_current/swebench_regular_dev/run_manifest.json`
- SWE-bench repository preflight：`artifacts/eval128_current/swebench_regular_dev/repository_preflight_receipt.json`
- SWE-bench Coding Tool canary：`artifacts/eval128_current/swebench_regular_dev/coding_tool_canary_receipt.json`
- 上游源码映射：`docs/SOURCE_MAP.md`
- 能力矩阵：`DATASET_CAPABILITY_MATRIX.md`
