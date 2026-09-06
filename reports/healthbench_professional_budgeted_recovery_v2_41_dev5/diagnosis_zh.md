# HealthBench Professional v2.41 无 Skill 五题诊断

本报告仅分析已经收束的 `healthbench_professional_budgeted_recovery_v2_41_dev5`。未读取正在运行的候选 Skill 条件结果；未启动模型、评测、测试或训练。医学内容仅用于解释 benchmark 失败，不作为医疗建议。

## 1. 本轮结果与比较边界

源码版本为 `31597229802973d184a4c6949e97cb39a50bb812`，分支为 `feature/healthbench-v2.41-budgeted-recovery-skill-20260906`。最终 manifest 的收束时间是 `2026-09-06T13:15:40.313399+00:00`，状态为 `collection_completed_with_operational_failures`。`candidate_skill_evaluation.enabled=false`，`mode=memory_off`；没有 Skill 注入、发布或训练。

| 固定题目 | 原始评分 | 长度调整评分 | 最终状态 |
| --- | ---: | ---: | --- |
| WATERFALL，`f056cdb4…` | 0.0000% | -5.9770% | 有效评分，FINISH |
| ASTRONAUT，`ed8b3ca0…` | 36.3636% | 36.8340% | 有效评分，FINISH，但试验识别仍错误 |
| Varices＋Barrett，`4f08ae48…` | N/A | N/A | 900 秒任务级超时，无有效终局 |
| IBD/HIV，`dadbebd3…` | 0.0000% | 5.4302% | 有效评分，FINISH；只重试原回答的 grader 后成功 |
| MDT，`37101607…` | N/A | N/A | 900 秒任务级超时，无有效终局 |

固定分母为 5，**完整五题原始/长度调整均分均为 N/A**。实际完成的 3 题子集均分是 **12.1212% / 12.0957%**。不能将它与 v2.40 的另一个 2 题完成子集均分相减，并宣称整体提升或下降。共同已评分题可分别描述：WATERFALL 从 50% 到 0%，IBD/HIV 从 53.33% 到 0%；这只说明这两题本次重新采样的结果变差，不是完整五题平均差或单一改动的因果效应。

长度调整评分按已有官方 reference evaluator receipt 原样报告：它可以为负，也可能高于 raw；不能把 5.4302% 的短文本长度调整分理解为“IBD 回答命中了医学知识点”。IBD 的两项 rubric 实际均未满足。

## 2. 错误分类与数量

主失败层按每题的主要终局问题归类，互斥，总数为 5；不是对尚无终局的题目擅自判断医学答案对错。

| 主失败层 | 数量 / 5 | 占比 | 代表 |
| --- | ---: | ---: | --- |
| 实体消歧、任务语义与检索方向偏移 | 2 | 40% | WATERFALL、ASTRONAUT |
| 下游综合遗漏：已收到的信息未保留到最终回答 | 1 | 20% | IBD/HIV |
| 执行协议/证据修复未在预算内收束 | 2 | 40% | MDT、Varices＋Barrett |

下列为可重叠的伴随事件，不能再与上表相加：

- 已保存记录中，5 题均出现 StructuredAction 格式/字段相关的 `ValueError` 反馈。它不是统一的医学推理错误，也不能仅凭该代码断言每次都是 JSON 长度截断。
- WATERFALL、Varices＋Barrett、MDT 共 3/5 题出现 `structured_evidence_item_receipt_binding_invalid` 或 `structured_evidence_item_span_not_in_receipt`。这表明来源元数据/引用段落与实际 receipt 绑定失败，不等于没有资料可用。
- 三条完整 trajectory 都有 provider-identifiable runtime failure；两个 partial trace 也记录了 HTTP 429。故 provider 故障在 5/5 的已保存执行中出现，但不能将其当作三道已 FINISH 题医学失分的充分原因。
- IBD/HIV 的 `healthbench_grader_error` 为 1/5 的可恢复 evaluator 事件；已对原终局执行一次 grader 重试并获得有效分数，没有重新生成回答。它不是最终剩余缺口。
- 最终任务超时 2/5；`max_rounds` 终止 0。两个没有终局的 partial 不得补写 FINISH、答案或分数。
- 本轮没有以旧版“超过上下文窗口”的错误收尾；但五题的小样本观察不能证明所有长上下文路径已经完全解决。

## 3. Demo 一：WATERFALL——Director 先引入错误解释，多 Agent 放大偏差

**Task ID：**`healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a`。

**原始输入：**`waterfall trial`。

**评测目标（只在 evaluator 阶段读取）：**识别相应急性胰腺炎补液试验，说明其主要效果比较及液体过负荷差异。本轮两个 rubric 均未满足。

实际执行过程：

```text
原始短语
  → Director round 0：ADD_SUBGRAPH，建立 node_1 → node_2 → node_3
  → node_1：解释“临床药理学的 waterfall 试验结构/目的”
  → node_2：沿上游概念搜索临床试验
  → node_3：综合“Wasserfall / Author's Waterfall”方法解释
  → Director round 5：新增 node_4，node_3 → node_4，设为 Output
  → node_4：输出试验方法学、肿瘤 waterfall plot 等长篇解释
  → round 6 FINISH → 官方 raw 0，adjusted -5.9770%
```

首次可观察的语义失败在 **Director round 0 的 contract**：用户没有给出肿瘤或“试验方法学”限定，Director 却指定 cancer/drugs，并在 node_3 contract 中先写入 `Wasserfall`、`Modification of Author's Waterfall` 等未经证据确认的解释。这发生在该子图实际检索之前。

初始 node_1/node_2 使用 DeepSeek，node_3 使用 MiniMax；经实际 provider/执行错误后的 MODIFY，已完成的这条图最终四个节点均为本地 Qwen。模型切换保留了错误职责，因此切换本身没有修复语义偏差。

实际 Tool receipt 记录 9 次调用：authoritative 1、knowledge 2、literature 2、trials 3、MedRAG 1。首个短查询 `waterfall trial` 确实执行过；随后查询转为 oncology、clinical pharmacology design、Author's Waterfall 等方向。可见保留首个查询的保护不能约束后续由错误 contract 推导的查询。工具返回有 ClinicalTrials.gov、Europe PMC、PubMed、MedRAG 的来源信息，但不是命中正确试验的证明。

node_1、node_2、node_3 的最终中间 artifact 均将支持状态写为 `insufficient`；node_1 同时仍在 summary 中断言一种肿瘤分层试验概念。node_4 最终将局部未检索到定义升级为“没有正式 Waterfall Trial 这一方法”，并输出 4,033 字符。**传输链存在，错误的概念和否定结论沿链传播；增加节点没有纠正最初实体绑定。**

这里有真实证据格式修复反馈，但终局为 0 的直接原因是回答了错误问题；长度惩罚只是将 raw 0 进一步变为负数，并非主要医学失分原因。

## 4. Demo 二：IBD/HIV——证据已经送达，Output 仍丢弃必要信息

**Task ID：**`healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3`。

**原始输入：**`How common is IBD in patients with HIV?`

**评测目标（evaluator-only）：**回答该人群 IBD 风险/患病率的相对关系，同时交代患病率估计的局限或研究异质性。两项 rubric 本次均未满足。

```text
round 0 ADD_SUBGRAPH：node_1（研究并输出）与 node_2（检索）先独立执行
  → provider 失败，round 1 MODIFY node_1 为本地 Qwen
  → node_2（MiniMax ReAct）检索 PubMed/Europe PMC，并 read_source(38180722)
  → node_2 产生包含引用与不确定性的 structured evidence artifact
round 2 SET_RELATION：node_2 → node_1
  → node_1（Output，本地 Qwen ReAct）重新执行
  → round 3 FINISH → grader 重试后 raw 0，adjusted 5.4302%
```

来源 receipt 确实包含 PubMed `38180722` 的摘要及 `41677181` 的队列资料。node_2 artifact 保留了来源 ID、摘录和共同患病比例，并明确写出：研究异质性、双向共同患病比例不能无条件等同于特定人群患病率、地理和时期差异。

**传递是否丢失，已有直接证据：**round 2 的 `node_1.metadata.request.upstream` 内有来自 node_2 的 **2,976 字符**完整 artifact；其中确实包含比例范围与 `heterogeneous` 等不确定性说明。这不是仅有空壳 provenance，也不是上游结果没有传给 Output。

但 node_1 最终只输出 **153 字符**：

> Based on current medical literature, Inflammatory Bowel Disease (IBD) is uncommon in patients with HIV, though the combination is clinically encountered.

这句话既没有解释相对风险，也丢弃了已收到的研究异质性/估计范围局限。**最直接可定位的信息损失在 round 2 的 Output 综合，而不是 `node_2 → node_1` 的通信通道。**上游把共同患病比例用于单向患病率解读本身也有适用性问题，因此完整转发不是充分条件：还需要正确理解分母和问题询问的关系。

独立检索和最终综合两个 Agent 已经存在。把该失败归咎于“没有多 Agent”不符合实际轨迹；还不能证明再新增一名 Agent 就会修复。

## 5. Demo 三：MDT——已有资料和部分结果，但协议修复耗尽时间

**Task ID：**`healthbench-professional:37101607e2947481e85e8fe3597a1acf`。

**原始输入：**要求为拟保喉的喉鳞癌病例撰写 MDT board conclusion，并讨论用户给定的诱导化疗后同步放化疗方案。完整病例输入保存在 private demo，本报告不把其临床细节当作提示修改规则。

目标是完整的下一条 assistant response；本题没有有效 terminal/evaluator receipt，因此实际评分是 **N/A**，不是 0%。

```text
round 0：ADD_SUBGRAPH node_1 → node_2
  → node_1 的首个 ReAct 返回混入自然语言和 MiniMax tool-call 标记
  → StructuredAction ValueError；之后检索 Europe PMC 成功
  → complete 的 artifact 字段不合规，继续修复
round 1–5：修改 contract / model；保存 node_1，继续修复 node_2
round 6：新增综合 node_3 与独立 node_4
  → 实际关系：node_1 → node_2，node_1 → node_3，node_2 → node_3
  → node_2 后续 artifact 已保存，node_4 再次耗尽 ReAct 执行轮数
round 7 尚在 Canvas 执行 → 整题 900 秒取消
  → partial 保存；没有 Output Agent、没有 FINISH、没有评分
```

首个可观察的工程失败是 node_1 第一次 ReAct 输出不是单一 StructuredAction JSON：其中先有自然语言，再混入 `[minimax]`/`tool_call` 包装。后续记录还有 `structured_evidence_artifact_fields_invalid`、来源 metadata 不一致、`structured_evidence_item_span_not_in_receipt`。这是多种不同失败，不宜全部归为一个“重复输出”问题。

资料不是完全缺失：partial 内已有 Europe PMC `32871829`、`32548612`、`35303749` 等来源及 MedRAG 摘录。node_2 多次 complete 修复涉及引用连续片段和元数据绑定，最终曾保存有效 artifact；并没有因全部删除节点而丢掉已有成果。但新的 node_4 又出现执行耗尽，node_3 受到 cancellation，最后仍没有形成合法终局。

partial 显示 **7 个已提交 Director turn、正在执行 round 7**；最终图已有非链式的传递与 fan-in，另一个 node_4 尚未接入。这证明运行时允许非链式图，不证明该未完成图的综合质量好。没有唯一 Output 属于超时时仍未完成的状态，不能反推成“所有中间 Canvas 都必须完整”的 admission 要求。

配置仍是每 Agent 调用 180 秒、每题 900 秒、每调用最多 6 个 ReAct turn；partial 同时包含调用级 `TimeoutError`、HTTP 429、来源校验修复以及最后的任务取消。不能把已耗时间全部归给单一 API，或者通过简单加长 timeout 宣称修复。

## 6. 另外两题的关键证据

**ASTRONAUT（`healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181`）：**最终 `node_1 → node_2 → node_3`，Output 为 MiniMax。实际 Tool receipt 为两次 trials.search 和一次 calculator；calculator 的输入是注册号字符串，不能提供试验内容证据。检索返回的三个注册研究不对应目标试验，node_2 却沿“宇航员人群”的解释核验，Director 又在 Output contract 中要求输出“没有此类登记试验”。最终回答仍没有识别目标 ASTRONAUT 试验。36.36% 来自 grader 将回答中“检索局限”认作 rubric 所列“试验局限”；不能把这个分数称为语义上答对，也不能据此往提示加入 rubric 措辞。

**Varices＋Barrett（`healthbench-professional:4f08ae480b16ef825cf098eca6530e68`）：**首次 node_1 在检索成功后，ReAct 第 4–6 次输出连续收到 `ValueError`。其后出现元数据/引用绑定错误、查询范围限制、429 及调用超时。最后 partial 是 `node_1 → node_2` 与 `node_3 → node_4 → node_5` 两个未联通部分，没有 Output；已保存 node_1/node_3/node_4，仍在修复 node_2/node_5。10 个 Director turn 已提交、round 10 未完成，最终 900 秒超时。旧版上下文错误不再是此题的最终终止原因，但终局能力仍未解决。

## 7. 修复有效范围与尚未解决的问题

已经有本轮证据支持的进展：

- 本轮不再以旧的上下文超限错误结束；保存了有界反馈、完整终局及两个不可评分 partial。
- 有效结果在 repair 期间被保留，IBD 的完整 evidence artifact 实际送达下游。
- 三条终局均采用至少两个 Agent；两个未完成图的最新状态也为多 Agent。系统允许非链式关系，不是只能串行。
- evaluator-only 失败可复用同一回答重试，没有为了补 grader 而重复生成整题。

尚未解决且不能提前宣称有效：

1. **问题解释与后续查询偏移。** 首个查询保留原词，不足以阻止后续 contract 将不确定解释固定成事实。
2. **Output 语义覆盖。** 信息送达不等于被正确使用；IBD 显示最终压缩会主动丢失已存在的限定条件和不确定性。
3. **StructuredAction / structured evidence 的稳定性。** JSON 外包装、字段错误、引用段落与元数据绑定失败仍会使修复循环持续。
4. **执行预算与 provider 恢复。** 180 秒/900 秒边界仍被触发；多 Agent 的错误分支会阻塞或取消其他分支，整体完成率只有 3/5。
5. **evaluator 的语义误判。** ASTRONAUT 的局部得分存在可观察的判分误匹配，应该报告原始 receipt 和错误语义，而不是为了得分顺应误判。

特别更正：不能声称本轮 HealthBench ReAct “没有发送多动作 JSON Schema”是已确认根因。专用 Clinical/Authoritative 路径已有根级 `oneOf` 动作 schema；这些轨迹出现 JSON 错误是真实的，但 schema 是否完整送达并被远端服务执行，不能仅由错误文本判断。模型声明预算与实际请求预算的覆盖问题应按 request/finish receipt 单独检查，不能与缺 schema 混淆。公开/中间日志里的截断标记也不等于服务端 `finish_reason=length`。

本报告不修改 prompt、检索词、医学答案、grader 或任何运行条件；不评价尚在进行的候选 Skill 试验。

## 8. 可复现证据入口

- [本轮公开逐题评分与调用统计](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/reports/healthbench_professional_budgeted_recovery_v2_41_dev5/agentgraph_development_report.json)
- [收束 manifest](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/run_manifest.json)
- [三条完整 trajectory：逐步 Director、Agent 输入输出、Tool 与 evaluator receipt](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/agentgraph_trajectories.jsonl)
- [完整 private demo](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/evaluator_private/agentgraph_development_demos.md)
- [两条超时 partial：按 task_id 定位，不可评分](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/evaluator_private/partial_trajectories.jsonl)
- [collection failure 原始记录](/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v241/artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/collection_failures.jsonl)

Tool 调用总数 16 仅覆盖三条完整 trajectory；包含 partial 的全五题实际调用/计费总量没有在本报告补算或推测。`collection_failures.jsonl` 保留已恢复的 grader 失败历史，必须与最终 evaluator receipt 合读。
