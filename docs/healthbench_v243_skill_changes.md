# v2.43 候选编排经验变更

本子任务仅新增 `config/healthbench_candidate_skills_v243.yaml` 和本说明。完整复用 v2.42 的三个候选条目 schema、已有 profile helper 与 `forced_probe_condition` 注入边界；没有修改 v2.42 冻结文件、Director、Canvas、Runtime、模型池或 evaluator。

## 相对 v2.42 的细化

| 候选 | 细化内容 | 边界 |
| --- | --- | --- |
| `evidence_before_entity_commitment_v243` | contract 描述待完成工作，而非提前指定实体身份或肯定/否定结论；模糊专名先按原词查询，再根据实际资料消歧；空检索结果不证明不存在。继续区分用户目标与尚未证实的前提。 | 解释未解决的名称、前提或结论时；不改变用户实际问题。 |
| `reconcile_receipts_before_requery_v243` | 检索命中与上游主张矛盾时，先核对已得到的来源，必要时读取已经观测到的 source ID，而非不断发出同义查询。保留有效 artifact/关系；根据真实 receipt 修复字段、引用和 metadata。 | 只由实际来源冲突或 Tool/artifact/dependency/provider 错误触发；provider 替代仍由现有可用模型域决定。 |
| `ground_final_claims_in_observed_sources_v243` | 核对最终主张与实际来源及上下游输入；检索范围局限不是研究结论；没有对应 receipt 不能声称搜索过某数据库。不为填空捏造细节。 | 有具体未解冲突才自行决定是否增加独立核对或修复 Output 路径；合法 FINISH 不等于内容正确。 |

全部内容都是一般 procedural 建议，不包含具体题目、专名答案、临床建议、数值或评分信息。不定义固定医疗角色或拓扑，不强制多 Agent；三个条目均有 trigger，可被 Director 拒绝。

## 源码与方法依据

- 用户 MD §10：以条件、动作和失效边界表达 Skill；少量 prompt prior 只提供建议，不能改变开放 search space，也不能自行声明有效。
- 用户 MD §11：候选干预与正常 rollout 隔离；用于发现问题的公开 test 不能作为 Skill 的独立发布证据。
- SkillFlow `src/skills/workspace.py::{retrieve,format_skills_for_prompt}`、`training/environment.py::_handle_skill_invoke`：沿用少量策略内容及实际暴露/调用的界限。
- 本项目 `scripts/run_joint_qa_mace_skill.py::_prompt_condition`、`LiveSmokeBackend.collect`、`scripts/healthbench_candidate_skill_profile.py`：原样复用候选字段、可拒绝 prior、长度上限及无训练校验，不另造 runtime。
- 本轮调整依据是已观察到的公开执行问题类型：contract 先承诺结论、来源与主张不一致、已有 receipt 未被充分使用、同义检索重复、实际工具调用与最终叙述不一致。这里记录工程假设，不声称有独立确认的效应。

## 版本、发布与比较口径

版本为 `healthbench.orchestration-candidates.v2.43`，三个条目仍是 **candidate，不是 ACTIVE**。没有训练、MACE、Bayesian 或 Skill publication；没有写入效应均值、置信区间或 gate receipt。

本子任务相对 v2.42 只修改候选 prompt 内容。如果主线另行修复工程代码，应在该条件的 source map/manifest 中另记；不能把组合条件的变化称为纯 Skill 因果增益。静态注入仍按现有模式记录，不冒称新增了动态检索器或模型已经学会调用。

主线已用既有 helper 核验 schema 和注入路径，补充通用的数值—测量指标—人群—分母绑定提醒后，内容合计 2,511 字符。没有将特定样本的统计结论写入提示。

主线另有最小工程修复：把分阶段声明的真实 Output-closure 校验错误传回下一轮 Director，而非误报缺少 relations；把现有 ReAct parser 的实际错误通过 Agent continuation 和 Director feedback 传递，而非统一误报 wrapper。因此本轮是候选内容与错误反馈的组合条件，不单独估计 Skill 因果效应。
