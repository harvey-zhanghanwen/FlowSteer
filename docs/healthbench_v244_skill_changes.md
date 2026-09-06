# HealthBench v2.44：把候选 Skill 转化为可执行的编排操作

## 范围与来源

本轮只修改候选 profile、独立评测配置和说明，runtime、Director 基础提示、
action mask、模型目录、生成预算、工具和 evaluator 均沿用 v2.43 源码
`5fcfb04fb686e8943a47e18176dbe40de715f6eb`。没有训练或权重更新。

复用 SkillFlow `SkillEntry` 的 trigger/plan/pitfall/constraint 表达、
`SkillWorkspace.format_skills_for_prompt` 和 `Environment._handle_skill_invoke`
面向实际工具动作的策略表达；执行仍使用本项目已有
`healthbench_candidate_skill_profile.py`、`LiveSmokeBackend.collect(prompt_priors=...)`
和 FlowSteer 式 Canvas edit → execute → feedback。没有创建新的 Skill runtime。

用户 MD §§10–11 要求适用条件、可执行 model/contract/relation/repair 建议、
失败边界与独立确认。本轮是手工预声明的 candidate prompt priors，不是
自动 Skill evolution、ACTIVE 发布或完成 MACE/Bayesian 闭环。

## 三条候选相对 v2.43 的变化

1. **实体消歧与来源选择**：要求 Director 用 ADD/MODIFY 把原始表达、
   已有证据支持的身份与未解解释写成交付要求，并让 Agent 选择适合问题的
   已有 Tool。不能把相关教材段落等同于命中指定实体，不能把空检索当不存在。
2. **按失败层修复**：真实格式错误反复发生时，先修故障节点；必要时从
   当前准入域选择兼容模型或缩小该节点交付范围。保留有效证据与 Tool 能力，
   不在同一故障前置节点后继续添加无法执行的消费者。
3. **先修结论与证据的矛盾**：合法域允许时优先 MODIFY 不一致结论的节点。
   若收尾规则只允许新增消费者，则让该消费者评价而不是照抄上游结论。
   按合法域和实际需要加入独立核对；不强制固定角色、固定链或多 Agent 数量。

建议直接面向 Director 能执行的动作。没有将 Skill 通过隐藏通道注入所有
Agent，也不声称 Director 一定会采用。Agent 是否收到相应要求，以真实
contract、输入和轨迹为准，不以候选提示已经显示作为采纳证明。

## 不变条件与证据边界

- 使用与 v2.43 相同五个已观察开发题、模型池、seed 和 evaluator。
- Task deadline 900 秒、Agent invocation 360 秒、六轮 ReAct、三次 Tool
  dispatch 等原有限制不变；不增加付费重试，不重复 Direct。
- 每轮模型生成必须遵守原严格 schema；不补写 JSON，不回收中间答案冒充 FINISH。
- 五题已用于开发，不能用于独立确认 ACTIVE Skill，也不能声称无偏泛化提升。
- 本次不含题目 ID、具体临床答案、rubric 或参考回复。
- v2.43 的 Output 输入域限制单列记录；不通过修改 Skill 文案声称已经解除，
  也不在本轮暗改核心准入规则。
- 该限制有源码和测试依据：`_output_closure_sink_artifact_domains` 仅取
  quotient DAG sinks，`model_admissible_action_types` 在收尾状态只允许 ADD。
  候选不能承诺当轮可 MODIFY producer；若未来获准调整，另设版本，不混入本轮。
- 完整 525 的配置只是后续入口；未实际运行前不报告完整数据集成绩。
