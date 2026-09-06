# v2.38 五题评测：分数上升，但仍有明确的回答与证据传递问题

已完成全部 5 题，执行代码 `a65f9f2`；5/5 evaluator-valid、5/5 显式 FINISH，
任务超时、最终 terminal failure、未恢复 grader error 均为 0。评测进程已退出。
本组是沿用 v2.37 的五道既有低分开发回归题，不是 525 题整体或独立测试成绩。

## 1. 官方指标与同题对照

沿用 OpenAI simple-evals HealthBench Professional reference evaluator
`652c89d@1`。先按每题 rubric 得分，再聚合，长度调整协议不变；不是 EM/F1
或二元正确率。没有为了新结果修改 rubric、聚合或答案回收方式。

| 同一组五题 | 原始 overall score | 长度调整后 overall score |
| --- | ---: | ---: |
| v2.35，5/5 有效 | 25.36% | 20.91% |
| v2.38，5/5 有效 | **35.36%** | **38.01%** |
| 差值 | **+10.00 个百分点** | **+17.10 个百分点** |

v2.36 未正式评分；v2.37 只有 3 道有效题、2 道运行失败，不能把它的三题
0% 当作同分母完整五题成绩。也不能拿 v2.35 全部 19 道的分数与本组五题
直接比较。没有重跑 Direct；旧 Direct 的工具条件不能冒充新版匹配对照。

| 题目 / ID 前缀 | v2.35 raw | v2.38 raw | v2.38 adjusted | 最终回答字符数 |
| --- | ---: | ---: | ---: | ---: |
| WATERFALL / f056cdb4 | 0.00% | 50.00% | 44.35% | 3923 |
| ASTRONAUT / ed8b3ca0 | 36.36% | 36.36% | 40.95% | 439 |
| Varices/Barrett / 4f08ae48 | 43.75% | 43.75% | 49.21% | 144，仅标题 |
| IBD/HIV / dadbebd3 | 46.67% | 46.67% | 49.90% | 900 |
| MDT / 37101607 | 0.00% | 0.00% | 5.62% | 89，仅标题 |

**原始分的全部增加来自 WATERFALL 一题，其余四题 raw 未提高。** adjusted
相对 raw 的额外增幅约 7.10 个百分点来自长度项变化，其中有回答退化为
标题的情况；不能把全部 17.10 个百分点说成临床内容质量的改善。
本版同时继承 v2.37 的未评测上下文修复，并改变工具集合，模型选择随之
变化，因此这是端到端版本比较，不是单独隔离数据库效果的因果消融。

## 2. 新知识源是否真的被使用

去重依据已落盘 Tool receipt 的工具 ID 与实际调用起止时间，不重复计入
continuation 中携带的旧 receipt：

| 实际工具 | 调用次数 |
| --- | ---: |
| 新 ClinicalTrials.gov `healthbench-trials.search` | 2 |
| 新 Europe PMC `healthbench-literature.search` | 1 |
| 既有 MedRAG `healthbench-medrag.search` | 2 |
| 既有 MedRAG + PubMed `healthbench-authoritative.search` | 8 |
| source.read / drug.lookup / calculator / knowledge.search | 0 |

本轮没有 Agent 调用全文阅读工具，不能说已经利用了大量论文全文。
Europe PMC 实际返回的三条记录在该问题上只有会议摘要合集的元数据，摘要
正文为空；接口成功不等于获得可支持回答的证据。ClinicalTrials.gov 在
WATERFALL 案例返回了相关注册研究，并出现了正确命名试验方向，但这一个
案例不能证明所有任务增加该源都会受益。

## 3. 自然形成的 topology

- 单 Agent：1/5（20%）。
- 多 Agent：4/5（80%）；其中有向链 3/5，双向协作再 fan-in 1/5。
- 最终图共有 11 个节点：本地 Qwen3.5-9B 9 个，MiniMax-M3 1 个，DeepSeek 1 个。
- 不预置角色或强制结构；Director 固定本地 Qwen3.5-9B，thinking 和原
  Canvas 功能单元执行边界保留，无训练或权重更新。

实际非链式案例为 IBD/HIV：`node_1 ↔ node_2`，两者共同 `→ node_3(Output)`。
该图正常执行，但发生跨阶段证据丢失，不能称为协作成功的证明。

## 4. 主要错误分类

按每题一个主要问题互斥归类，共 5 题；次要问题可同时存在，见下方 demo。

| 主要类别 | 数量 / 占比 | 代表 |
| --- | ---: | --- |
| 试验结局覆盖不完整 | 1 / 20% | WATERFALL |
| 命名实体消歧错误 | 1 / 20% | ASTRONAUT |
| 最终回复只有标题，completion/FINISH 未识别 | 2 / 40% | Varices/Barrett、MDT |
| 拓扑变化后的跨阶段证据传递缺口 | 1 / 20% | IBD/HIV |

最终 max_rounds、timeout、collection/parsing failure 均为 0。没有确认
Formatter 截掉正文的案例。检索相关性、contract scope drift 和 grader
解释偏宽是部分案例的次要问题，不重复计入上述互斥分类。

### A. WATERFALL：方向部分修正，结局未充分回答

Task：`healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a`。
目标是解释这个命名试验；评分信息仅在本次离线诊断时使用。

实际链路：Director 建立单个 MiniMax ReAct 节点并选为 Output；
`healthbench-trials.search` 检索原始试验词 → 相关注册记录 → 完整文字回复
→ Director FINISH → 两条 rubric 中满足一条，raw 50%。

最终回复识别到急性胰腺炎中的 WATERFALL，但仍讨论泛指试验/图表的其他
含义。grader 认可其液体负荷风险比较，未认可另一项具体临床结局比较。
首个可观察偏移在 Director contract：它要求泛化解释“waterfall trial”
及“没有通用注册覆盖”，使答案有较多偏离目标的说明。没有充分读取原试验
来源，不能凭返回了相关注册研究就认定证据链完整。

### B. ASTRONAUT：把试验名解释成宇航员

Task：`healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181`。
实际图：`node_1(Qwen ReAct，ClinicalTrials.gov) → node_2(DeepSeek，Output)`。

node_1 使用原问题词检索，没有获得匹配注册记录；node_2 最终说明没有
omeprazole 与 astronauts 人群匹配的试验。问题中的命名试验仍未被正确
识别；无注册命中不能排除其他数据库中的既有发表研究。

这题 raw 36.36% 不代表主要问题答对。grader 以回答谈及检索局限为理由，
接受了 rubric 中关于研究局限的宽泛选项；药物比较和研究分组两项均未满足。
这里应同时注明 evaluator 解释偏宽，保留原始官方评分，不自行改分。

### C. Varices/Barrett：Output 只提交标题

Task：`healthbench-professional:4f08ae480b16ef825cf098eca6530e68`。
实际最终链：`node_1(Qwen/MedRAG) → node_2(Qwen/Europe PMC) → node_3(Qwen/Output)`。

turn0 的 DeepSeek 429 通过改模型恢复；turn1 node_1 检索两次并产生带来源
artifact，node_2 的 Europe PMC 结果没有摘要正文；其总结把缺少检索结果
扩大为缺少标准指南。turn2 node_3 第一条 ReAct action 就 `complete`，值
只有一个 144 字符的英文标题；turn3 Director FINISH。

`react_trace.action_text.arguments.value == execution.output == final_answer`，
模型 `finish_reason=stop`，不是 length 截断，也没有 Formatter 删除正文。
提示词已要求完整回答，但执行结果检查和 FINISH 仍接受了标题。
上游原始证据在嵌套 provenance 和实际输入中仍可找到，本题不是消息完全丢失。

raw 43.75% 来自“提及资料有限”这一项；另一关键操作注意事项未满足。
短标题经官方长度项调整后为 49.21%，不能据此称为高质量回答。

### D. MDT：同样只有标题，未完成所要求的文档

Task：`healthbench-professional:37101607e2947481e85e8fe3597a1acf`。
实际链：`node_1(Qwen/MedRAG+PubMed) → node_2(Qwen/source.read capability，Output)`。

node_1 检索两次，指出来源不足以支持 contract 预设的方案；Director 又让
node_2 从来源中提取支持该方案的内容，存在结论预设。node_2 实际没有读
来源，而是直接提交 89 字符标题，随后 FINISH。四条 rubric 全部未满足，
raw 0%；adjusted 5.62% 只是短回答的长度项。与旧版相比未提高原始分。

### E. IBD/HIV：已找到证据，却在改单向为双向时丢失

Task：`healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3`。

```text
turn4：node_1 SINGLE 检索到 PubMed 39333960、38180722
       → 带来源的 supported artifact，含相关定量研究结果
turn7：把 node_1 → node_2 改成 node_1 ↔ node_2
       → 两节点 DRAFT 的 upstream=[]、own_draft=null、旧 Tool receipt 数=0
       → 重新检索未命中旧证据，生成 insufficient draft
       → 两节点互看新 draft 修订，继续说没有量化数据
turn8：两路 insufficient artifact → node_3(Output)
turn9：FINISH，grader raw 46.67%
```

高置信度首个传递缺口是 turn7 的阶段变化：旧成功来源没有作为可重新核验
的历史证据交给新的 reciprocal draft。Director 的历史 prompt 仍含旧 PMID，
但未把它重新提供给 Output，而接受了新的“证据不足”结论。

对应 `src/interactive/agent_runtime.py`：dirty 输出/metadata 失效（约1412），
reciprocal draft 只接当前 upstream（约2492/2505），同 component 不走普通
upstream（约2649），phase 不同的 continuation 被清空（约2734）。使旧结果
失效是合理的，但不应同时丢掉可引用和重新核验的既有来源证据。
本轮仅定位，没有中途更改运行条件或修改该 runtime。

## 5. 耗时、调用与保存位置

- 全流程约 699.71 秒（11 分 40 秒），最后一题多轮修改和协作耗时最长。
- 已落盘成功 execution 中去重后的 Agent model_calls：29；其输入 tokens
  146,583，输出 31,800。此口径不含 Director、grader 和可能未记入这些成功
  execution 的失败尝试，因此不是完整账单或全部 API 请求数。
- Grader 实际请求 17 次，包含 4 次 HTTP 500 后的恢复；最终五题均有效，
  共计已回传输入 tokens 12,082、输出 1,805。失败请求 token 使用未知。
- 原始完整输入输出、工具 Action–Observation、通信和 evaluator receipt：
  `artifacts/healthbench_professional_external_medical_sources_v2_38_dev5/evaluation/agentgraph_trajectories.jsonl`。
- runner 官方聚合：`reports/healthbench_professional_external_medical_sources_v2_38_dev5/evaluation_report.json`。
- 本报告的精确指标、图和调用统计：同目录 `evaluation_summary.json`。
- 代码和来源说明已按独立 v2.38 分支保存；不提交凭据或大型原始 trajectory。

结论：新工具确实接通，五题执行稳定性比 v2.37 的 3/5 有效更好，原始分
比同题 v2.35 提高 10 个百分点。但四题原始分没有改善，标题式输出和跨
阶段证据丢失是明确需要后续修复的问题。此次五题任务结束，不启动全量或训练。
