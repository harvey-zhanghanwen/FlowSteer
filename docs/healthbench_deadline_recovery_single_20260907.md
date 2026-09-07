# HealthBench 超时单题重跑报告

本轮只重跑曾超时的一个开发样本，不是525题结果或新一轮五题均分。

## 结果

| 项目 | 旧五题运行中的本题 | 修复后的单题运行 |
| --- | --- | --- |
| Task ID | healthbench-professional:38ed97e78292dfcf0805ac924311fa14 | 同一题 |
| Condition | healthbench_professional_semantic_skills_new5 | healthbench_professional_deadline_recovery_single |
| 原始分 overall_score | N/A | 50.000000% |
| 长度调整分 overall_score_length_adjusted | N/A | 24.704240% |
| Evaluator valid / FINISH | 未到达评分器 / 否 | 1/1；1/1 |
| 执行超时 / terminal failure | 超时900秒 | 0；0 |
| 采集开始到评分/轨迹完成 | 900秒取消 | 279.812秒 |

运行源码：`c5ae00310ca642ab22ac51ee7375f128365ae19b`。
运行 attempt：`run_attempt_fed4bc6fcde0611887d5e620`。
官方 evaluator：`openai-simple-evals-healthbench-professional-652c89d@1`；grader
`gpt-5.4-2026-03-05`，3个 rubric 调用，provider/grader error 为0。
分数从现有 evaluator receipt 读取，没有重新评分、失败置零或裁剪分数。

## 已完成的架构修复

1. Director 现在能看到同一900秒上限下的已用/剩余时间，以及前一步 Director/Canvas 的实际耗时。
   不再只有剩余轮数。默认配置保持关闭；本次单题显式开启。
2. 复用既有 v249 实现：若取消/超时，保存已完成 TurnRecord、Agent I/O、当前 Graph、
   未返回调用及失败信息；严格标为 non-scoreable，不把部分结果冒充 FINISH 或有效评分。
3. 原有无变化编辑拒绝、相同输入执行缓存、自由 Graph、合法 FINISH gate 均保留。
   没有加固定医疗角色、自动强制 FINISH 或扩大900秒上限。

源码来源和薄适配原因见 `docs/source_map.md` 与 `docs/adaptation_log.md`。
61项离线定向测试通过。未执行训练、权重更新、MACE/Bayesian/Skill evolution。
语义索引和3条 semantic.v1 candidate priors 未变，没有单独消融，不能认定候选 Skill 已产生净提升。

## 本题执行过程与剩余缺口

输入：用户提供约1万字符、有关痴呆症状的英文医学表格，并要求翻译成希腊语。
完整输入、输出及 rubric 仅保存在本地 evaluator-private demo。

实际生成的链路（并非预设模板）：

```text
node_1：翻译，Qwen3.5-9B，ReAct
  → node_2：检查译文，DeepSeek V4 Flash，reasoning
  → node_3：最终译文，Qwen3.5-9B，reasoning / Output
  → FINISH → 官方 grader
```

- Round 0，ADD_SUBGRAPH：加入 node_1、node_2 和 node_1→node_2，148.156秒。
  node_1 的4次 ReAct model calls中，前3次 complete 被
  `structured_evidence_artifact_requires_evidence` 拒绝，第4次接受 insufficient artifact。
  artifact 只有“将进行翻译”的说明，没有译文。实际 Tool calls=0。
  node_2 收到了这份 artifact，并明确报告没有译文，不能验证；不是消息没有传到。
- Round 1，ADD_SUBGRAPH：加入 node_3、node_2→node_3 并指定 Output，116.211秒。
  已有两个节点执行被复用，没有重跑。node_3 从原始公开任务生成10604字符的希腊语回答。
- Round 2，FINISH：7.815秒，合法完成。
  评分约7.26秒后返回。整个过程3个 Director turns，6个已记录 phase calls；
  Agent 已记录 model calls=6（本地Qwen5，DeepSeek1），Tool calls=0，grader calls=3。
  已记录调用数不等于包含所有未保存内部重试的完整计费账单。

**首个可观察失败点是 node_1 的任务产物接口与翻译 contract 不匹配。**
`healthbench_evidence_adapter.py::_completion_arguments_schema` 对启用结构化证据的所有
non-Output 节点施加同一医学 evidence schema，summary上限4000字符；
`_structured_evidence_artifact_error` 又要求 supported 状态带检索证据。
这不适合交付长译文的中间节点，造成 complete 反复被拒绝，再以 insufficient说明替代译文。
新时间反馈使本次未再次达到外层超时，但**没有消除这个遗留接口缺口**。
旧超时题没有完整中间轨迹，不能断言旧4次 MODIFY 全由这一原因导致。

node_2 并未检查后来 node_3 才生成的最终译文，因此图上存在检查节点不代表最终译文经过验证。
不应据此硬编码必选 Verifier；需要让实际检查消费实际任务产物。

## 错误分类（多标签；分母仅1题，类别可重叠）

| 类别 | 题数/占比 | 证据 |
| --- | --- | --- |
| Contract–artifact schema mismatch | 1/1，100% | node_1 被检索证据格式约束，3次 complete拒绝 |
| 不完整任务产物交接 | 1/1，100% | node_2收到说明但无译文；传输本身正常 |
| 最终产物验证缺口 | 1/1，100% | node_3的实际译文生成于node_2检查之后 |
| 医学术语翻译错误 | 1/1，100% | 原生grader的一条正分术语标准不满足 |
| 网络/检索失败 | 0/1，0% | 实际没有Tool调用；不可声称数据库提升或拖慢本题 |
| Collection timeout / max_rounds / terminal failure | 0/1，0% | 合法FINISH，有效评分 |
| Grader/provider failure receipt | 0/1，0% | 已保存失败receipt中未发生 |

每个非零类别都对应上述同一 Task ID，其全量可复现demo位于：
`artifacts/healthbench_professional_deadline_recovery_single/evaluation/evaluator_private/agentgraph_development_demos.md`。
该文件包含输入、任务目标、模型输出、动作、通信、ReAct及评分receipt；没有为零类别编造案例。

原始分50%对应2条正分标准中满足1条；1条负分标准未触发，不能按“3条中只对1条”计分。
长度调整按当前冻结reference协议为：

```text
0.5 − 0.0147 × (10604 − 2000) / 500 = 0.2470424
```

即长度项减少25.29576个百分点。原输入已很长，完整翻译自然会很长；
不能直接把这25.30点解释为重复输出，也不应通过删除请求翻译的内容来刷长度分。

## 解释边界与下一步

- 本次是由N/A恢复为有效评分，不存在可计算的同题旧→新分数提升幅度。
- 原先五题并发、本次仅单题；虽然配置仍concurrency4，实际服务负载不同，
  不能把缩短耗时全部归因于新时间反馈。
- 本次使用同一问题、同一配置seed/模型池/预算/evaluator/语义库，但冻结列表由5题变1题，
  scientific-sampling序列位置/命名空间变化可能改变派生seed，不是逐调用相同随机数的消融。
- 下一步应修复任务产物类型与证据附件的接口：通用文本产物可交接，检索证据作为可选独立附件；
  supported/insufficient用于证据状态，不应替代“译文是否真正交付”。需通用实现，不能把本题术语答案写进提示词。
- 本次到此停止，没有开始下一轮付费评测，也没有把本题成绩混入旧五题或525题均分。

## 备份

仓库：`https://github.com/harvey-zhanghanwen/FlowSteer.git`，远端 `backup`。
独立分支：`feature/healthbench-deadline-recovery-20260907`。
运行源码commit已推送；本报告及公开指标另行提交在同一分支。
原始对话、rubric、完整I/O、外部数据库及模型文件仅保留本地，未加入本次Git提交。
