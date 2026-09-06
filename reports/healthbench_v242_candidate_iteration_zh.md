# HealthBench Professional v2.42：候选 Skill 迭代结果

本轮为固定五题开发回归，非独立 held-out 评测；没有训练、模型权重更新、
MACE、Bayesian 或 ACTIVE Skill 发布。源码为
`cdec2a9b3c0213fd918da805e0c81d81bf74dec0`，三个候选提示实际进入 Director
输入；`retrieved_skill_ids=[]` 不代表没有候选注入，它没有伪装成 ACTIVE 检索。

## 真实结果

官方 reference evaluator 的 raw score，不是二元准确率。保留负分、长度
调整与缺失值，不补零。固定分母 5，4 次显式 FINISH 且 evaluator-valid，
1 次 900 秒超时；完整五题 raw/length-adjusted 主指标均为 **N/A**。
已评分四题 raw 均值 **11.6667%**，length-adjusted **8.4503%**，仅为子集描述。

| 样本 | ID 后缀 | v2.41 候选 raw | v2.42 raw | v2.42 length-adjusted |
| --- | --- | ---: | ---: | ---: |
| WATERFALL | f056cdb489e3636b0b51afb8fd6b3a8a | 50.00% | 0.00% | 0.75558% |
| ASTRONAUT | ed8b3ca0a4dabfd0827c17a08513a181 | N/A：超时 | 0.00% | 0.54390% |
| Barrett | 4f08ae480b16ef825cf098eca6530e68 | 43.75% | 0.00% | −7.21770% |
| IBD/HIV | dadbebd3dce1b5928cac5a44dde095d3 | 100.00% | 46.66667% | 39.71945% |
| MDT | 37101607e2947481e85e8fe3597a1acf | −44.44444% | N/A：超时 | N/A |

有效子集不同，不能直接比较两个 completed-only 均值。同题中已有明显
退步，因此没有证据把 v2.42 称作最佳架构，也不据此启动完整 525 题。

## 首要失败类型与代表案例

按每题一个主要问题归类；这些是可观察执行原因，不等于已完成因果实验。

| 主要类型 | 数量 / 五题占比 | 代表与首个可观察问题 |
| --- | --- | --- |
| 任务解释及预设结论 | 2 / 40% | WATERFALL、ASTRONAUT：Director 在 round 0、取得证据前就指定实体解释或“不存在”的结论。 |
| 无依据结论及证据使用不足 | 1 / 20% | Barrett：上游返回 insufficient artifact；后续综合引入未被证据支持的临床细节，并产生语义失真。 |
| 回答范围与统计指标绑定 | 1 / 20% | IBD/HIV：把较宽泛疾病分类的统计结果用于回答特定疾病频率；实际来源与部分限定条件已送达，但问题没有充分回答。 |
| 执行超时 | 1 / 20% | MDT：未取得终局，900 秒达到任务上限；不能给缺失终局计算医学分数。 |

### 1. WATERFALL：检索前的错误解释被多 Agent 放大

输入仅为 `waterfall trial`。实际图：
`node_1 → {node_2, node_3} → node_4(Output)`。
round 0 的 node_1 contract 预先改成查找“clinical trial designs”；node_1
后续由 MiniMax 切换为本地 Qwen，工具包括 authoritative search 与 knowledge
search。两个下游核对节点沿着该错误解释生成总结。round 7 的 Output contract
又要求明确写“没有这类试验方法”，最终还声称检索了没有对应 receipt 的数据库。
有非链式图，但图结构没有自动解决任务理解错误。raw=0，正常 FINISH。

### 2. ASTRONAUT：证据已经送达，仍被预设答案覆盖

实际图：`node_1(检索) → node_3(Output) ← node_2(独立回答)`。
round 0 的 node_2 无上游、无工具，却被要求明确回答“不存在该试验”。
node_1 的真实 authoritative search 返回包含 ASTRONAUT Study Group 的
PubMed 9494148；没有 source.read，最终 evidence_items 为空。
node_3 实际输入含该文献标题与三条 Tool receipts，但 round 3 的 contract
仍指定“不存在”。这不是消息传输丢失，而是主张与证据冲突未被解决。
raw=0，正常 FINISH；较旧版只改善了完成状态。

### 3. Barrett：格式合法不代表临床内容正确

输入为同时存在两种食管问题的处理咨询。图为
`node_1 → node_2 → node_3(Output)`，前两节点遭遇 DeepSeek 429 后由 Director
改为本地 Qwen。真实恢复域同时保留 MiniMax，未强制整 provider 不可用。
上游 structured evidence 的 status 为 insufficient；终局仍加入大量缺少
支持的临床细节并出现语义失真。schema/FINISH 通过只代表流程条件满足。
raw=0，length-adjusted=−7.21770%。

### 4. IBD/HIV：统计口径未精确对应问题

输入：`How common is IBD in patients with HIV?`。
图为 `node_1(MiniMax) → node_2(Qwen) → node_3(Output, MiniMax)`。
node_1 有真实 literature/knowledge/source.read，保留来源、人群与引文；
node_2 交付 insufficient 总结，Output 收到上游信息并生成较完整长回答。
最终将广义 autoimmune disease 的频率与 IBD 频率混用，且核心频率仍未
充分回答。raw=46.66667%，说明信息送达并不自动保证 answer slot 对齐。

## 明确工程问题与下一版范围

MDT 的已保存 prefix 中，node_2/MiniMax 三次各运行六轮，共 18 次生成、
累计模型耗时 640.63 秒，单组分别 94.67、270.88、275.08 秒；不是单次
360 秒节点限时。已保存 Agent 请求去重后 22 条：19 条响应、2 条 DeepSeek
429、1 条被取消；不等于底层 HTTP 请求总量。真实 Tool 三次均成功。
12 次 parse error 分为：嵌套字段 5、Markdown 包装 2、stop 但非法 JSON 2、
字符串 null 1、重复 resource_id 后被 null 覆盖 1、8192 token length 终止且
空输出 1。另有三次准入问题：query_too_broad、query anchor 不保真、
state_action_not_admitted。当前 parser 没有重复键检测，不将该人工观察
伪装成系统已有能力。详细过程保存在 partial trajectory，不能补造终局。

- WATERFALL 五次、IBD/HIV 九次 ADD 声明被拒绝后，反馈笼统显示缺 relations。
  WATERFALL 的实际原始异常经 declaration validator 重放确认是 Output closure
  无法直接路由声明依赖，不是模型漏写关系字段。这导致错误修复目标失真。
- v2.43 仅沿用既有 Canvas 拒绝接口传递原始 phase/error，保持原始模型输出、
  Graph 与步数；不自动补边、不改变容量或 Output closure 规则。
- ReAct 的 catch 也丢掉现有 parser 的具体异常，只统一提醒禁止 wrapper；
  字段类型、缺少可执行 resource ID 等实际原因没有进入下一轮。
  下一版保留现有严格 parser，只将其真实异常通过 public Observation 和
  continuation 传递，不为模型补写 JSON，不增加重试或生成预算。
- 候选内容继续保持通用：证据前不把结论写入 contract、空检索不等于不存在、
  核对真实来源、绑定统计指标与人群/分母、不虚报工具调用。不加入医学答案。
- 本轮工具/Agent 失败与终局问题可能重叠；没有以“无报错”代替医学正确。
  schema receipt 仅确认客户端请求，不能证明 provider 严格执行 schema。

## 完整复现证据

- 配置：`config/evaluation_healthbench_professional_candidate_skill_v2_42_dev5.yaml`
- 五题汇总与调用统计：`reports/healthbench_professional_candidate_skill_v2_42_dev5/`
- 原始数据：`artifacts/healthbench_professional_candidate_skill_v2_42_dev5/evaluation/`
- 完整输入、Agent 输入/输出、通信、Tool receipts、rubric 及 evaluator receipts：
  上述 evaluation 目录下的 `evaluator_private/agentgraph_development_demos.md`、
  `agentgraph_development_evidence.jsonl`；超时中间过程另见 `partial_trajectories.jsonl`。
- 详细日志没有重新发送给模型；评分信息仍只在 evaluator/分析阶段使用。
