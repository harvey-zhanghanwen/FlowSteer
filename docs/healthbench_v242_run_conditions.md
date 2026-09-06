# HealthBench v2.42：结构化证据、生成预算与候选 Skill 迭代

状态：工程候选，需定向测试、冻结配置和真实评测；本文不预报成绩。

## 已有证据与边界

v2.41 不加 Skill 的五题中，三题有效 FINISH、两题触及 900 秒。
本轮未再以 Director context overflow 终止，但并不证明所有输入永不超限。
Barrett 的去重记录包含 10 个 parse error、9 个非真实连续引文错误和
5 个元数据绑定错误；180 秒覆盖整个最多六轮 ReAct invocation。
这些问题不能通过把 N/A 改成零、回收中间答案或隐式 FINISH 解决。

诊断更正：HealthBench Clinical/Authoritative 已有完整动作域 `oneOf`
schema，不能将通用基类的单动作路径当作实际“不发送 schema”的证据。
本版保留该机制，没有另写 union schema，也不宣称代理服务已经严格执行它。

## 最小修改与来源

| 修改 | 来源及必要适配 |
| --- | --- |
| ReAct 按当前模型目录的 `max_tokens` 生成 | 复用既有 ModelSpec.metadata、Gateway 和 SkillFlow ModelRequest 预算边界；修复统一 Director 4096 覆盖 Executor 自身声明的现象。未声明才回落旧预算，thinking 配置不变。 |
| state-aware completion schema | 复用 SkillFlow `response_schema`、本项目 ToolRegistry 和已有 HealthBench `oneOf`；仅把真实 Tool Observation/routed receipt 的来源元数据组成合法候选，不改写模型 claim 或引文。 |
| 元数据来源约束 | document_id/source/title/date/url 必须来自同一个已观测来源，不能跨文档拼接。原严格连续 span validator 仍执行，schema 合法不代表医学结论正确。 |
| 请求格式 receipt | 从实际将发送的 Gateway payload 记录结构化输出参数的紧凑摘要，不重复传输完整 schema；只表示 client request，不声称 provider enforcement。 |
| Agent 时间预算 | 沿用 AgentRuntime 既有 `execution_timeout` 参数，180→360 秒，因为它覆盖整个六轮 invocation，而不是单次生成。每题仍 900 秒、四题并发、工具总数和 turn 数不增加。 |
| 429 恢复模型域 | 单次未知影响范围的 429 不证明同一 provider 的所有模型失效。启用既有恢复流程的薄适配，允许其它可用且执行模式兼容的同 provider 模型参与恢复；仍由 Director 选择，不改变 catalog 或自动切模型。 |
| 三条候选提示 | 使用同一候选 helper/collector 接口，新增 v2.42 profile；保持可拒绝、自由职责/模型/拓扑，不包含医学答案，不发布 ACTIVE，不训练。 |

v2.42 同时改变工程预算和候选内容。因此相对 v2.41 的差值是复合条件差异，
不能单独归因于 Skill；v2.41 同版本同五题的有/无候选对照另行完整保留。

## 运行计划

1. 同一五个已观察开发案例，配置 `evaluation_healthbench_professional_candidate_skill_v2_42_dev5.yaml`。
2. 执行闭环确认后，唯一完整条件 `evaluation_healthbench_professional_candidate_skill_v2_42_full525.yaml`。
3. 只运行 `--collection-arm agentgraph`，不重复 Direct，不训练/更新 LoRA，
   不启动 MACE、Bayesian 或 Skill evolution。原 v2.41 运行不混入本版目录。

保留官方 raw/length-adjusted score、原始负分与长度调整；报告有效样本分母、
未评分原因及逐题结果，不把完成子集均分说成完整五题或 525 题成绩。
公开 test 已用于反复开发，因此不能宣称新验证集上的无偏泛化提升。
