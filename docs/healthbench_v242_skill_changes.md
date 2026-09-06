# v2.42 候选编排经验调整说明

状态：三个条目均是预声明、尚未验证的 candidate prompt condition，不是 ACTIVE Skill。只新增配置与本说明，不改变既有 schema/helper、Director、Canvas、Runtime、模型池或评分器；没有模型调用、训练或参数更新。

## 相对 v2.41 的内容调整

| 条目 | 本次增加的约束 | 触发边界 |
| --- | --- | --- |
| `preserve_goal_and_check_premises_v242` | 区分用户真正要完成的目标、范围与输出要求，以及用户建议/指定结论中尚未证实的前提。请求某个结论不等于该结论已经成立；有可靠冲突时解释冲突但仍回答原问题。 | 解释对话、规划/修订 contract、查询或依赖时。 |
| `repair_specific_failure_from_receipts_v242` | 根据已返回的字段、Tool、artifact、依赖或 provider 错误做具体局部修改。只换 contract 措辞不足以证明修复；保留仍有效的信息，不重复无变化请求；provider 故障时在已有可用模型域自主选替代。 | 当前实际反馈存在相应错误；不根据未来结果或评测信息触发。 |
| `verify_output_against_actual_findings_v242` | 对照 Output、原始任务和真正到达的上游 findings。保持必要限定与不确定性，不靠捏造事实补齐未知。合法 FINISH 或已有文本不保证内容正确。 | 结构完整、正在考虑终局；有具体未解冲突才考虑独立核对单元。 |

没有加入疾病、药物、试验、剂量、样本标识、参考结论或评分条目内容；没有预设角色或强制多 Agent。所有条目仍可被 Director 拒绝，模型选择、职责与关系均自由。

## 来源与必要适配

- 用户 MD §10.1–10.4：有适用条件和失效边界的编排规则、可拒绝 prompt prior、严格区分候选与发布。§11.2–11.3：候选干预与正常 rollout 分开，发现数据不能充当独立确认数据。
- 公开 SkillFlow `src/skills/workspace.py::SkillWorkspace.retrieve`、`format_skills_for_prompt` 与 `training/environment.py::_handle_skill_invoke`：复用“检索少量策略内容并保留实际暴露/调用边界”，不复制固定工作流。
- 本项目 `scripts/run_joint_qa_mace_skill.py::_prompt_condition` 和 `LiveSmokeBackend.collect`：继续使用既有 `application_mode=forced_probe_condition`、`rejectable=True` 与独立 condition 记录，不改写采样动作。
- 本项目 v2.41 helper `scripts/healthbench_candidate_skill_profile.py`：原样复用三个 candidate 的字段约束、文本长度限制、无训练/发布校验和 prior 构建。只换版本化配置。
- 新增措辞来自一般可观察的执行失败：未经验证的前提被当事实、具体字段错误重复出现、provider 故障、上游 findings 已送达但终局遗漏。它们是待验证的工程假设，不是已测的 Skill 效应或特定题答案。

## 使用与评测口径

配置为 `config/healthbench_candidate_skills_v242.yaml`；schema 与 v2.41 相同。前两项 graph stage 为 `*`，但正文各有真实触发条件；最后一项为 `before_final_answer`，正文明确结构合法不等于内容正确。静态注入不伪称新的自动阶段检索机制。

继续走 candidate-only evaluation 条件，不进入 ACTIVE Store，不填效应均值、置信区间或 gate receipt。已有公开 test 调试结果不能用于宣布这些规则已经发布。即使后续分数上升，也需与同时发生的架构修复区分；未做隔离对照前不能把所有变化归因于候选经验。

验证仅运行既有 helper 的离线 profile validate/build，已通过；不重跑旧单测、不启动评测或模型。三项 instruction 分别为 474、531、530 字符；构建后模型可见 content 分别为 798、825、922 字符，总计 **2,545 字符**，低于既有 4,000 字符限额。这是字符数，不是模型 token 数；没有估计实际收益或测试分数。
