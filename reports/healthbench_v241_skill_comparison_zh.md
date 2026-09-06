# HealthBench v2.41：候选 Skill 五题对照

两组已结束，模型权重均未更新。源码为
`31597229802973d184a4c6949e97cb39a50bb812`。两组使用同一已观察的五题、
同一模型目录、seed、工具/生成预算和官方 reference evaluator；唯一预声明
干预是三条未验证、可拒绝的候选编排提示。它们不是 ACTIVE Skill，
没有 MACE、Bayesian、训练或 Skill evolution。

## 逐题结果

以下均为官方 raw score 的百分数，不是二元准确率。

| 问题 / sample ID 后缀 | 无候选 Skill | 有候选 Skill |
| --- | ---: | ---: |
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 0.00% | 50.00% |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | 36.36% | N/A：900 秒超时 |
| Barrett / 4f08ae480b16ef825cf098eca6530e68 | N/A：900 秒超时 | 43.75% |
| IBD/HIV / dadbebd3dce1b5928cac5a44dde095d3 | 0.00% | 100.00% |
| MDT / 37101607e2947481e85e8fe3597a1acf | N/A：900 秒超时 | −44.44% |

有效 FINISH/评分：**3/5 → 4/5**。两组的完整五题主指标仍均为 **N/A**，
不能把缺失结果代填零或丢弃分母。无候选的 IBD/HIV 首次 grader 遇到
HTTP 500；仅复用冻结答案重试 grader 后得到上述有效 0 分，没有重做 AgentGraph。

仅供状态观察：无候选有效三题 raw 均值 12.1212%、length-adjusted
12.0957%；有候选有效四题 raw 37.3264%、length-adjusted 34.3879%。
**有效子集不同，这两个均值不能直接用于声称总体提升。**

## 已观察到什么

- WATERFALL 和 IBD/HIV 在同题上得分提高。
- Barrett 与 MDT 从未完成变为可评分，但 MDT 的临床内容仍触发负分。
- ASTRONAUT 反向退化为超时；无候选的 36.36% 还包含 grader 将检索局限
  误当作研究局限的情况，不能将其解读为准确识别了试验。
- 无候选 WATERFALL 的首个可观察问题在 Director contract：提前将短名称
  定义为某类试验方法。IBD/HIV 则是 Output 已收到丰富上游信息，却在
  终局过度压缩，不是通信链路没有送达。
- 候选 MDT 接受了用户建议中的未验证临床前提，最终输出还存在内容失真。
  下一版只提炼通用的前提核验与完整性检查，不把该题的标准临床结论写入 Skill。

## 结论与下一轮

候选提示有局部正向效果，但尚没有完整分母的整体提升证据，也没有
达到发布 ACTIVE Skill 的独立确认门槛。这五题已经用于开发，不是无偏
held-out 验证；一次模型生成也不能证明稳定因果效应。

v2.42 在独立工作树中继续修复生成预算与证据元数据绑定，并改进三个
通用候选条件。它同时改变工程和提示条件，不能将下一版的所有变化
单独归因于 Skill。完整 525 题尚未启动。

## 完整过程位置

- 无候选：`artifacts/healthbench_professional_budgeted_recovery_v2_41_dev5/evaluation/`
- 有候选：`artifacts/healthbench_professional_candidate_skill_v2_41_dev5/evaluation/`
- 两组各自包含 `run_manifest.json`、`agentgraph_trajectories.jsonl`、
  `collection_failures.jsonl`、`evaluator_private/partial_trajectories.jsonl`，
  以及离线展开的 `evaluator_private/agentgraph_development_demos.md`。
- 无候选细分诊断：`reports/healthbench_professional_budgeted_recovery_v2_41_dev5/diagnosis_zh.md`。
