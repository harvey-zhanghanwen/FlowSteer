# HealthBench Professional v2.39：同五题开发评估

## 结论

按本轮要求接入高质量来源、取消来源专用执行 profile、修复正文与历史证据
交接后，进行了**同五题的一次 AgentGraph 运行**。2026-09-06 10:35:55 UTC
已自然收尾：**3/5 FINISH 且 evaluator-valid；2/5 collect TimeoutError**。
没有训练、Direct 重跑、额外模型 canary、525 题扩展或挑选性付费重试。

**完整五题官方均分 N/A。**已完成三题 raw mean 为 **33.33%**，
length-adjusted mean 为 **33.00%**，仅是 completed-only 结果，不能与旧版
完整五题均分当作同分母对比，更不能宣称新版提升或达到 Stable Zero。
v2.39 保留为可恢复开发候选，不改写已有 best-profile 为“最佳”。

## 1. 实际修改与复用

- 原统一 AgentGraph、自由 contract、节点模型选择、双向关系和唯一 Output
  保留；Director 仍为本地 Qwen3.5-9B / minimal-neutral.v20。ReAct 是工作
  模式，不是角色；没有固定医疗角色或强制串行模板。
- 延用 SkillFlow bounded ReAct 和 FlowSteer Canvas/trajectory 接口。
  新增 NCBI Bookshelf（含 NCI PDQ、AHRQ EPC 集合）与 NLM MeSH 官方接口，
  沿用现有 search/read/receipt/index。Bookshelf 搜索命中只是元数据；
  可用开放正文需 source.read。MeSH 是术语表，不是临床疗效证据。
- 取消按来源绑定 Agent 的执行配置：仅 reasoning 无工具或 ReAct 全12工具。
  保留各工具来源标识与内部索引元数据，不强制 Director 先分类再检索。
- 修复公开任务需要正文时只提交标题的 ReAct completion：同 Agent 在既有
  预算内恢复，不靠最低长度拒绝合法短答，不读取 rubric。
- Canvas 历史来源通过原节点 historical_evidence 交给双向组件；不把旧答案
  当当前有效答案，不复制旧控制状态，不重放历史 Action 或重置工具预算。
- 234 项定向测试（另107 subtests）和 prepare-only 通过；这不等同于完整
  实测闭环通过。来源接口的官方协议及逐文件映射见 source_map/adaptation_log。
- NIH/HHS HIV Guidelines 当前直连403，未伪造接入；NICE 未接入。

## 2. 真实逐题结果

HealthBench Professional 采用官方 reference rubric grading，不是 EM/F1
或二元准确率。分数统一乘100展示。以下5题此前已用于开发诊断，**不是新的
独立 held-out 测试**，也不能代表完整525题。

| 问题 / task ID 后缀 | v2.38 raw | v2.39 raw | v2.39 length-adjusted | 终局 |
|---|---:|---:|---:|---|
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 50.00 | 0.00 | 2.86 | FINISH，有效评分 |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | 36.36 | 0.00 | 4.81 | FINISH，有效评分 |
| Varices + Barrett / 4f08ae480b16ef825cf098eca6530e68 | 43.75 | 100.00 | 91.34 | FINISH，有效评分 |
| IBD + HIV / dadbebd3dce1b5928cac5a44dde095d3 | 46.67 | N/A | N/A | collect TimeoutError，900秒 |
| MDT / 37101607e2947481e85e8fe3597a1acf | 0.00 | N/A | N/A | collect TimeoutError，900秒 |
| **完整5题均分** | **35.36** | **N/A** | **N/A** | **3/5有效** |

旧版完整5题 length-adjusted 为38.01%。三题已完成子集的新版33.33/33.00
不应替换五题均分。仅在这三题匹配子集上，旧 raw 为43.37%、新为33.33%；
该比较有完成样本选择偏差，不是全五题效果估计。

两个原始得分为0的短回答仍有正的 length-adjusted 值，来自既有官方长度
调整公式，不代表它们回答正确。Barrett 的 raw100只表示满足该题两项
rubric，不等于整篇医学建议全部正确、有证据或临床上可直接采用。

## 3. 逐类错误与代表案例

下面是**多标签、证据可见范围内的统计**，类别不能相加。两题缺完整轨迹，
因此不把未观察到的内部失败写成0。完整对话、各Agent输入输出、通信内容、
ReAct Action/Observation和rubric receipt保留在本地 evaluator_private 报告。

| 类型 | 已确认数量 / 可完整检查数 | 占比 | 代表 |
|---|---:|---:|---|
| 命名实体消歧失败、Director改写问题语义 | 2/3 | 66.67% | WATERFALL、ASTRONAUT |
| 检索相关性/证据支持不足 | 3/3 | 100% | WATERFALL检出其他设计/肿瘤研究；Barrett只有间接资料 |
| 工具预算耗尽后仍继续请求搜索 | 至少1/3 | 至少33.33% | Barrett，ReAct第4–6次动作被拒 |
| Agent provider错误，后续有恢复 | 3/3 | 100% | 三题均有HTTP429；WATERFALL还出现节点TimeoutError |
| 已确认通信传输截断 | 0/3 | 0% | 已完成案例未见；缺轨迹两题N/A |
| 仅标题式终局输出 | 0/3 | 0% | 旧Barrett标题变为完整正文；MDT未完成，不能称已修好 |
| 正式收集超时 | 2/5 | 40% | IBD/HIV、MDT |
| max_rounds终局 | 0/5 | 0% | 两题是时间超限，不是max_rounds |
| 终局grader无效 | 0/3已评分 | 0% | WATERFALL有一次grader provider重试，但最终有效 |
| 超时后完整轨迹未保存 | 2/5 | 40% | 仅剩progress、失败事件及局部资料索引 |

### Demo A：WATERFALL —— 编排一开始改变了查询对象

输入 `waterfall trial`。round0 Director 为 node_1 定义
“waterfall design trials … randomization timing schemes … post-hoc switch”。
**首个可观察语义失败发生于工具执行前的 contract**，把命名试验解释成设计方法。

实际链路：node_1（DeepSeek遇429后改Qwen/ReAct）→node_2
（MiniMax超时、DeepSeek429后改Qwen/ReAct）→FINISH。
node_1用 ClinicalTrials查询 `waterfall trial design randomization timing scheme`
得0条，再用Europe PMC查带 antibiotic resistance、biopsy monitoring等扩展词。
node_2延续“switching criteria”方向，实际共6次工具调用：literature4、trials1、
MedRAG1。检索命中不等于证据相关，新增限定不断强化最初误解。

最终回答1026字符，核心是“不存在独立、正式命名的Waterfall设计”，raw0。
这里更早的错误是问题解释，不能仅归因于恢复中的provider故障，也不是缺少Agent。

### Demo B：ASTRONAUT —— 错误解释传递畅通，非链式结构仍失败

输入 `omeprazole astronaut trial`。round0 Director 的 node_1 contract
明确写 `population (astronauts, cosmonauts, spaceflight crew)`，先把试验名称
误读成人群。429后只换模型、没有修正这个解释。

实际拓扑：node_1(Qwen/ReAct)→node_3(Qwen/ReAct)←node_2(MiniMax/reasoning)。
node_1在ClinicalTrials查 `omeprazole astronauts spaceflight cosmonaut` 得0条；
另一条偏离原始词锚的query被拒；authoritative随后取回3条间接教材片段、PubMed0条。
node_2输出一段“news search”JSON文本，但它没有工具权限，**并未真正执行新闻检索**。

round2 Director 又把“no registered trials exist … astronauts”写进输出contract。
node_3收到完整的617字符上游artifact、2个Tool receipts、另一个91字符query
artifact及未背书检索摘录，最后365字符回答继续声称没有宇航员试验，raw0。
没有发现这些输入被截断；错误是早期解释与过早结论在fan-in图中继续传播。
旧36.36也未正确识别试验：旧grader把检索局限当作研究局限的部分命中，
具体rubric与解释见私有receipt，不能把旧分当作正确解题证明。

### Demo C：Barrett —— 正文恢复，但检索与最终答案并未完全闭合

输入询问同时患食管静脉曲张和Barrett食管的管理。node_1先由MiniMax执行
两次Europe PMC搜索和一次knowledge.search，后者只是重取前两篇资料，
不是增加两份独立证据。后续3次额外搜索因预算而被拒，节点失败；换DeepSeek
后429，再改Qwen才完成。曾提交 supported + 空 evidence_items，被现有结构化
证据检查拒绝，随后以 insufficient 完成。

node_1(Qwen/ReAct)→node_2(Qwen/reasoning Output)。node_2真实收到949字符
摘要、uncertainties和4条未背书摘录（涉及两份文献），一次生成4947字符正文。
新分100/91.34，旧输出仅144字符标题、raw43.75。

**这不是正文guard触发后的重试：**本次node_2为reasoning，attempt_count=1；
三个完成案例均未观察到 `completion_artifact_requires_substantive_body` 返回。
因此只能说输出形态改善，不能把得分变化因果归于该修复或新增资料源。
满分也只覆盖该题有限rubric；其余具体临床建议仍缺可核对支持，不做医学正确性背书。

### Demo D/E：IBD/HIV、MDT —— 多轮修复未收敛且取消缺完整trace

IBD/HIV已提交14个Canvas步骤：ADD_SUBGRAPH1、MODIFY12、SET_RELATION1。
任务局部索引可确认单向阶段和后续双向DRAFT阶段仍保留PubMed38180722，
revision3、9以及15均可见相同来源；**不是所有历史资料都消失了**。
但DRAFT反复进行、未成功收束，最终900秒超时。保留来源的局部证据不等于
证明整个双向执行/最终回答已经正确；完整失败动作/模型返回未落盘，不能确诊
是哪一条completion校验或模型输出导致反复MODIFY。

MDT已提交9个步骤：ADD_SUBGRAPH3、MODIFY6；也在900秒后取消，没有完整
回答及官方评分，不能把旧标题问题报告为已经通过真实验证。

现有 `rollout_collector.collect()` 在CancelledError时只发取消progress、不
构造可评分trajectory。这避免给未完成任务伪造分数，但**没有保存独立的未完成
执行轨迹**，造成诊断证据缺口。局部 request_receipt/知识索引不能替代完整I/O。

## 4. 新增来源有没有提高分数？

当前**不能证明**。完整可检查的三题共11次实际工具调用，新增 Bookshelf/
PDQ/AHRQ/MeSH 和 source.read 的调用数均0。未完成两题的最终工具调用计数
N/A，其保留索引仅见原有PubMed/Europe PMC/MedRAG，不将其等同完整调用日志。
所以“已经接入”不能写成“已被有效使用”，也不能将Barrett提升归因于新来源。
这也不是分类检索的单因素消融：同时更改了工具可选边界和工程恢复行为。

## 5. 剩余优先问题与版本状态

1. 命名实体链接和查询改写要保留原问题对象；Director不能先给错误解释再让
   全图寻找佐证。需要通用消歧/证据反馈，而不是硬编码这两道题的正确答案。
2. ReAct预算反馈应让节点在无可用搜索动作时完成或交接已知证据，避免反复
   选择被屏蔽的动作；同时区分“没有检出”和“研究不存在”。
3. 在不改变官方评分和FINISH边界的前提下，补充取消/超时的独立未完成trace，
   先定位双向DRAFT和多轮MODIFY为何不收敛，再决定最小修改。
4. 校验同一来源经literature和knowledge传递后的语义去重；不能把重复摘录
   视为多份独立支持，也不能用正文长度掩盖无证据结论。

本轮不再自动付费补跑、训练或扩大数据集。架构提交 `9fb27de`，分支
`feature/healthbench-v2.39-guidelines-sources-20260906`；本地备份已建立。
GitHub push 当前认证失败，**尚未完成远端备份**。未破坏性回退旧版。

## 6. 证据入口

- `artifacts/healthbench_professional_clinical_reference_sources_v2_39_dev5/evaluation/run_manifest.json`
- 同目录 `agentgraph_trajectories.jsonl`（仅3个完整轨迹）、`rollout_progress.jsonl`、`collection_failures.jsonl`
- 同目录 `evaluator_private/agentgraph_development_demos.md`：完整可用输入、动作、通信、输出和评分；缺失明确标N/A。
- 本目录 `agentgraph_development_report.json`：逐题native metrics和receipt去重统计，不调用模型。
- `reports/healthbench_professional_external_medical_sources_v2_38/evaluation_summary.json`：旧五题。
- `docs/source_map.md`、`docs/adaptation_log.md`：SkillFlow/FlowSteer复用及必要适配。
