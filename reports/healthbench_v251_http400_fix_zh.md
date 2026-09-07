# 本地HTTP400修复与v2.51同五题验证

## 结论和边界

已修复本地Agent请求缺少精确上下文预算的问题。此次记录的26个本地Qwen
Agent请求，预检输入token数全部与服务端prompt_tokens一致，HTTP400为0。
已有Canvas节点引起的AttributeError也没有复现。但本轮最长本地输入为
19189，没有真实触发输出预算收缩；不能宣称所有长上下文或所有400均已验证。
边界收缩和输入单独超限的处理已通过定向测试，未额外调用付费API制造边界请求。

旧v248题2014…的三个400没有保存服务端错误正文或失败payload。最后成功
输入24123、请求输出8192，距离32768总窗口仅453；后续又加入长检索结果，
所以**上下文超限高度可疑，但不是已经精确复原的历史原因**。
400代表服务端拒绝请求，不可直接解释为OOM、模型能力或账号额度问题。

## 修改与来源

- Gateway直接复用现有SGLangReceiptDirectorClient._context_budget和
  rollout_collector._token_ids，实际Qwen聊天模板计数。只按剩余窗口调整
  输出上限，不裁原问题、证据或Agent消息，不修改thinking/模型/工具配置。
- 修复本地tokenizer实际返回BatchEncoding时直接len误计为2的问题。
- 输入自身已占满窗口时不发送必失败请求，保存明确预检receipt；其余HTTP
  错误保留有界provider type/code/message，以便区分原因。
- 修复非空AgentGraph.nodes为AgentNode tuple、而非ID集合的类型误用；
  沿既有Canvas admission读取node.id，不重写编排架构或引入固定角色。
- 本轮134项定向测试通过；包含本机真实tokenizer CPU测试和已有节点后的
  连续ADD/MODIFY，未训练、未下载模型。来源详见docs/source_map.md。

## 固定新五题结果

冻结源码：fafd551370e4520877a0454684ca0bf725036421。
条件：healthbench_professional_candidate_skill_v2_51_dev5。
仍为v248起固定新五题；模型池、seed、并发、工具、候选prior、evaluator不变。
候选仍是v248三条可拒绝建议，没有额外注入医学答案或rubric。

| Task ID 后缀 | 原生raw ×100 | 原生length-adjusted ×100 | 状态 |
|---|---:|---:|---|
| 3533d9bfd2d32f8c465e7af62aec9781 | -100.0000 | -94.6727 | FINISH、evaluator-valid |
| 9a160f86c59743692e46fab89aae42f2 | 100.0000 | 104.6423 | FINISH、evaluator-valid |
| fa30f3f57c7219130345f5c2e6d03d65 | N/A | N/A | 原900秒任务超时，partial保留 |
| 2014ab7a9d8865f0da483817843ccbc5 | 0.0000 | 5.2361 | FINISH、evaluator-valid |
| cd132a0c7cde74c0242aa8ef3850c9b9 | 0.0000 | 2.5754 | FINISH、evaluator-valid |

完整五题均分：**N/A**。已完成四题描述性均分：raw **0.0000%**，
length-adjusted **4.4453%**。原生指标不裁剪，所以可有负分和超过100的
长度调整值；它们不是EM或正确率。没有达到5/5有效且均分>60%的条件。
本轮未启动525，不能把HTTP400不再复现说成Skill或医学质量提升。

## 仍有的问题

已保存terminal与partial中按Agent request_id去重可见115条模型请求记录，
其中7条远端请求失败的provider正文为“当前分组上游负载已饱和”，code为
model_not_found，运行层HTTP429；不是本地HTTP400或已确认额度耗尽。
这些统计不含全部Director/Tool/grader请求，不能当作整轮账单调用数。

运行失败记录还有8次ReactExecutionError、3次CancelledError、1次TimeoutError
（去重后的记录数，不是互斥题数）；4条terminal均显式FINISH，另1题超时。
不能仅凭请求问题修复就宣布terminal和答案内容问题已解决。

HTTP400复现数：0；节点类型异常复现数：0；任务超时：1/5。
医学答案低分另按原生rubric receipt解释，本报告不依据评测答案继续改提示词。

## 证据与备份

- 公共指标/逐题模型、图、调用统计：
  reports/healthbench_professional_candidate_skill_v2_51_dev5/agentgraph_development_report.md
- 完整问题、各Agent输入输出、通信、Tool和rubric receipt：
  artifacts/healthbench_professional_candidate_skill_v2_51_dev5/evaluation/evaluator_private/agentgraph_development_demos.md
- 超时中间记录：同evaluation/evaluator_private/partial_trajectories.jsonl。
- 分支：feature/healthbench-v2.51-context-canvas-fix-20260907。
- 远端：backup，https://github.com/harvey-zhanghanwen/FlowSteer.git。
  源码fafd551已推送；本报告完成后追加独立报告提交，不覆盖历史或推送main。
