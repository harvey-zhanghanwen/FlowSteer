# HealthBench Professional v2.34：架构修复与评测交接

## 已完成的架构修正

本版修复的是 Director 的可观察信息，不是增加固定医疗角色或强制多 Agent。

1. **证据反馈。** 当前 Agent artifact、检索摘录、来源、限定条件和不确定性通过现有 v3 证据投影返回 Director；不再仅依赖极短的首尾预览。检索结果与 Agent 解释分别标注。
2. **失败后的信息保留。** 保留已有成功 artifact 与成功 Tool receipt，并将失败尝试的请求/输入版本和旧 artifact 区分。旧结果标为 stale，不把它冒充当前可提交输出。
3. **协作与通信。** 保留全部合法 fan-in 的生产者回执、有限双向协作的 draft/revision 版本、实际消费的上游/peer 证据；没有新增无关系节点之间的广播。
4. **职责与任务范围。** Director v20 要求修复 Tool 查询时保持原始职责，不能把完整回答任务缩成“只输出查询”；不能补造未提供的个案信息。依据公开证据发现冲突并自主修改/补充节点，不预设医疗工作流。

## 与 MD / 上游的一致性

- 仍是 `agent_id + model_id + free-text contract`，自由模型/节点/职责/拓扑，唯一 Output。
- ReAct 是每节点 execution mode，不是 role。
- 按 FlowSteer 执行一次已接受的功能子图编辑，再将该单元内有序的公开 Action–Observation 和当前状态返回 Director；没有新增每个 Tool 动作中途调用 Director 的调度器。
- 复用 SkillFlow 的公开 Observation 和 continuation、FlowSteer 的 Canvas/runtime/trajectory。详细来源见 `docs/source_map.md`。
- 不训练、不启用 GRPO、LoRA、MACE、Bayesian 或 Skill evolution。

## 已知边界

Director 当前 artifact/evidence 回执总预算为 24,000 字符，超限明确标记；完整原始内容仍在 trajectory。这不是无损全量上下文复制。更多证据可见不能保证模型正确使用证据。

近期 v2.33 低分样本中已经观察到：上游捏造个案信息后下游继续格式化；检索证据不足被解释为没有风险；完整检索 query 被当作最终回答。这些不全是消息丢失，仍涉及语义判断和指令遵循。新版的修复效果必须通过真实评测确定，不预测评分。

## 验证和评测条件

- 41 项 Director/证据投影/Direct-reference 检查通过。
- 127 项 runtime/Director/config/feedback 回归通过，另23项 subtests；其中包含13项新版反馈测试。
- 固定官方公开 test 的同一批525题 prepare-only通过。
- 原有模型池v7、所有既有thinking设置、seed、并发4、900秒任务预算、官方rubric grader及聚合方式不变。
- Direct复用原始459个有效回答和66个按原协议计零的失败，不再生成或评分；完整分母原始评分13.968705%、长度调整后16.756929%。
- 新版主指标为官方长度调整后的平均分，辅助报告原始平均分，不称为EM/F1或简单正确率。每题分数先按官方方式汇总，不能先把每题负分截成零。

## 运行及交接

旧版v2.33继续完整运行，不在途中修改其代码或条件。监控只读取新落盘记录，分类诊断仅用于开发报告，不输入模型。旧版有待评分项时沿用现有runner只补评分；有缺失任务时有限续跑。只有确认旧版525题已正式收束、锁释放且新版readiness通过后才启动新版。

新版配置：`config/evaluation_healthbench_professional_mixed_all_thinking_v2_34_full525_director_evidence_feedback.yaml`。

使用同一GPU5/8025本地Qwen3.5-9B Director服务，不启动第二套模型，不影响其他项目。新结果写入独立的v2.34 artifacts/reports目录，旧版原始trajectory与结果完整保留。

这些公开题已经用于开发和错误分析；完整重评也不是未接触测试集的泛化证明。rubric/reference仍只在evaluator与离线报告中出现，不得进入Director/Agent输入。

当前新版状态以运行manifest为准：`prepared`只表示完成数据及配置准备，不表示已开始推理，也不表示已有评分。
