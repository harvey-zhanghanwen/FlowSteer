# HealthBench v2.46：保持 ReAct 的工程迭代

## 范围和固定门槛

用户确认是另外 **5 道 HealthBench Professional 题**，不是五个数据集。
本轮无训练、GRPO、LoRA、backward、optimizer、MACE/Bayesian 更新或 Skill
evolution。候选 Skill 仍是可拒绝的 prompt priors，不是 learned/ACTIVE。

新五题从已准备的官方 public test 按原文件顺序选取，排除 v2.45 的旧五题；
选择时只读取任务 ID 和顺序，不查看参考回答、rubric 或历史分数。结果为
原文件 ordinal 0–4，固定如下，后续本轮迭代不换简单题：

1. `healthbench-professional:9566084de89c416408691006a6f06f9c`
2. `healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558`
3. `healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274`
4. `healthbench-professional:e339f34a3a35f3f067422b5768287f7c`
5. `healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc`

这只是与上一轮不同的开发批次；公开全集曾经评测过，不能称为此前从未
接触的 held-out 数据。调试后的成绩也不是无偏泛化估计。

进入全量的门槛：本批 **5/5 显式 FINISH、5/5 evaluator-valid**，且 official
`overall_score_length_adjusted` 的五题均值 **严格大于 0.60**。同时报告 raw。
任一缺失为 N/A，不置零、不回收草稿、不调整 rubric。通过后先提交并推送
通过版本，再按冻结版本运行完整525题；五题过线不保证完整525也过线。
未通过则分析最早可观察失败，做有来源的最小通用修复，再复评固定这五题。

## 保持 ReAct 和自由职责

本次显式复用已有 `execution_profile_allowlist`，只留下 ReAct profile。
Agent 可以从已有证据直接选择合法 complete，不强迫无意义的额外 Tool调用。
不能用切到无工具reasoning来跳过检索或证据格式错误。Director固定本地
Qwen3.5-9B，执行Agent从兼容当前ReAct profile的既有模型池自主选择。

职责仍是 free-text contract。候选 Skill 可建议实体识别、证据收集、独立
核对或综合等互补职责，但没有角色枚举、必选Agent、固定模型或串行骨架。
只为实际信息缺口调整职责/关系，不把模型相互同意当成证据。

## 必要修复及来源

- SkillFlow `training/environment.py::step` 的 Action–Observation 延续，
  与既有 FlowSteer `workflow_env.py::step` 的逐编辑执行/反馈是来源；
  AgentGraph 的 source/target/version 路由是用户 MD §3 的必要适配。
- 证据交接复用当前 `AgentRuntime._historical_evidence`、`UpstreamMessage`
  和 gateway 的有界证据投影；不新建全图广播、答案数据库或另一个memory。
  同节点公开检索观察应保留，而故障 completion、其他节点/任务和评分信息
  不得被当作事实混入。
- 恢复复用当前model domain、failure records与Canvas admission。在
  `allow_untried_react_model_repair=true` 下，仅为真实协议失败保留一次
  兼容且实际未尝试模型的修复机会。旧配置默认false；不无限重试、不增加
  Tool预算、不通过无意义检索重置恢复计数。
- Scope校验区分明确答案断言与普通输出修饰、保留原缩写的检索假设；不建立
  题目答案/医学别名白名单，不将缩写展开本身视为证据。
- 候选表达参考SkillFlow `SkillEntry` trigger/plan/pitfall/constraint，
  接线直接复用项目 candidate helper、collector 和暴露receipt。MD §§10–11
  的独立确认/ACTIVE门槛不因本轮五题均分而绕过。

## 不变条件

相同官方reference evaluator、现有数据库/检索工具、模型池与thinking设置，
并发4、Agent invocation360秒、task900秒、每Agent六轮ReAct/三次Tool。
新版本/condition各自独立目录，运行期间不改源码/样本/模型/seed/预算。
不重复Direct，不另做付费canary或模型池探测。每次运行和最终报告都保留
真实调用、失败、tool receipt、topology、完整trajectory和逐题官方评分。
