# v2.37 首轮五题：没有证明提升

本轮已结束。实际执行代码为 `c4fbacd`，所有五题已尝试，3 题有效评分并
FINISH，2 题运行失败。**完整五题官方均分为 N/A**，不能把运行失败填成 0。
这是从旧版低分案例中挑选的开发回归集，不是随机或独立 held-out 样本。

| 题目 / 原始 ID | v2.35 raw | v2.37 raw | v2.37 length-adjusted | 状态 |
| --- | ---: | ---: | ---: | --- |
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 0.00% | 0.00% | -1.01% | FINISH |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | 36.36% | 36.36% | 37.85% | FINISH |
| Varices/Barrett / 4f08ae480b16ef825cf098eca6530e68 | 43.75% | N/A | N/A | 900 秒超时 |
| IBD/HIV / dadbebd3dce1b5928cac5a44dde095d3 | 46.67% | N/A | N/A | Director HTTP400 |
| MDT / 37101607e2947481e85e8fe3597a1acf | 0.00% | -44.44% | -44.60% | FINISH |

同一有效三题按官方聚合方法比较：旧版 raw **12.12%**、length-adjusted
**6.54%**；新版均为 **0.00%**。这是包括负分后对均值裁剪的分数，不是
“三题都没有输出”或二元准确率。不能把旧版五题 20.91% 直接和新版三题比。
没有运行新的 Direct，没有训练或 Skill evolution，也没有启动全量评测。

## 主要失败类型（五题互斥归类）

| 首要可观察问题 | 数量 / 占比 | 代表案例 |
| --- | ---: | --- |
| 命名实体消歧、contract 语义漂移 | 2 / 40% | WATERFALL、ASTRONAUT |
| 证据不足与方案关系推断错误 | 1 / 20% | MDT |
| Director 上下文预算错误 | 1 / 20% | IBD/HIV |
| 运行超时 | 1 / 20% | Varices/Barrett |

未发现可确认的消息丢失造成这三题低分；不把正常送达等同于内容正确。
此批 grader error 为 0，显式 max_rounds 终止为 0；两项运行失败不属于
有效 rubric 评分。没有足够证据单列格式化或 canonicalization 根因。

### WATERFALL：错误解释被传播

输入 `waterfall trial`；实际 `node_2 → node_1 → node_3(Output)`。
node_2 是无工具的 reasoning 节点，却被要求提供 authoritative evidence，
先把它解释成一般试验设计/图表。node_1 据此检索并引用其他肿瘤试验；node_3
继续输出“非标准化试验类别”。资料片段有真实来源，不等于支持所声称的
同名关系。首个错误在 node_2 的解释及 Director 对其职责/工具的配置。

### ASTRONAUT：已有线索仍未正确回答

输入 `omeprazole astronaut trial`；实际 `node_1 → node_3 ← node_2`。
node_1 的总结提到同名研究，但也把名称当作宇航员；node_2 的 contract
无依据引入 implants；Output 收到两路 artifact 后继续围绕航天研究回答。
这不是 fan-in 没送达，而是消歧和证据解释失误。部分 rubric 仍给分，并不
表示整篇回答正确。

### MDT：多 Agent 重复错误前提

实际 `node_2(DeepSeek) → node_1(Qwen) → node_3(Qwen Output)`，最终三节点
均为无工具 reasoning。node_2 先按用户要求的方案给出未经来源验证的结论，
后两节点主要转述。按该题 rubric，方案顺序与缺少替代处置条件触发负分。
此处只说明 benchmark 评分原因，模型生成的临床方案不构成医疗建议。

### IBD/HIV：没有得到可评测的最终回复

Director 第 3 轮请求：输入 29102 tokens + 生成预算 4096 = 33198，超过
SGLang 32768 上限，服务明确返回 HTTP400。这不是 grader 配额错误。
已完成一个独立的上下文预算补丁及离线测试，未重评，因此不存在补丁实测分数。

### Varices/Barrett：900 秒超时

任务在多轮编排过程中达到原任务超时阈值。原失败记录保留，不把中间
artifact 当 Final，不从未完成流程回收一个答案进行宽松评分。

## 证据与版本边界

- 原始结果：`artifacts/healthbench_professional_grounded_retrieval_v2_37_dev5/evaluation/agentgraph_trajectories.jsonl`。
- 完整 Canvas/Agent 输入输出、通信和工具 receipts 位于原始 trajectory；
  失败记录在同目录 `collection_failures.jsonl`，进度在 `rollout_progress.jsonl`。
- `evaluation_report.json` 的 strict 五题指标保持 null；没有改动官方 evaluator。
- `baseline_v235_dev5.json` 保存旧版五题真实指标和来源路径。
- `launch_receipt.json` 记录运行代码、GPU、时间与已通过的离线测试。
- 上下文补丁是评测后改动，不能声称这些结果已验证该补丁有效。

结论：v2.37 仍是候选，不适合据此推广到 525 题。来源索引和更完整 receipts
提供了证据访问条件，但尚不能保证 Director 选择正确职责/工具，也不能保证
Agent 正确理解实体和证据关系；本轮不能证明数据库本身导致了提升或下降。
