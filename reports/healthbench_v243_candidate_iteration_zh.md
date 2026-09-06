# HealthBench Professional v2.43：五题候选 Skill 评测

本轮已结束，源码 `5fcfb04fb686e8943a47e18176dbe40de715f6eb`。
四题显式 FINISH 且 evaluator-valid，一题 900 秒超时；完整五题官方 raw 与
length-adjusted 均为 **N/A**。有效四题的描述性均分分别为 **9.6465%** 与
**0.6552%**，不能代替固定五题或完整 525 的指标。无训练或权重更新。

| 任务 / sample ID 后缀 | v2.42 raw | v2.43 raw | v2.43 length-adjusted |
| --- | ---: | ---: | ---: |
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 0% | 0% | −5.33904% |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | 0% | 36.36364% | 22.39276% |
| Barrett / 4f08ae480b16ef825cf098eca6530e68 | 0% | N/A：超时 | N/A |
| IBD/HIV / dadbebd3dce1b5928cac5a44dde095d3 | 46.66667% | 46.66667% | 41.22767% |
| MDT / 37101607e2947481e85e8fe3597a1acf | N/A：超时 | −44.44444% | −55.66054% |

分数为官方 rubric 评分，不是二元准确率。ASTRONAUT 的部分得分不代表正确
识别了试验；其终局仍偏离了目标。两版本有效子集不同，不能直接比较子集
均分并宣称整体改善。没有启动 525 题，也未发布 ACTIVE Skill。

## 主要失败类别

每题按一个主要可观察问题归类；次要问题可以重叠。

| 主要类别 | 数量 / 五题占比 | 代表 |
| --- | --- | --- |
| 实体消歧和查询语义漂移 | 2 / 40% | WATERFALL、ASTRONAUT |
| 统计指标与问题范围不一致 | 1 / 20% | IBD/HIV |
| 结论与已有证据矛盾 | 1 / 20% | MDT |
| 格式修复失败导致超时 | 1 / 20% | Barrett |

次要问题包括错误来源归属、支持性检索偏置、Output 输入域限制，以及
在故障前置节点后继续扩图。不能把这些统一称为“Agent 数量不够”。

## 典型过程与首个可观察失败点

### WATERFALL：单节点也完成了循环，但解释仍错误

输入 `waterfall trial`。只有 node_1/MiniMax/ReAct，ADD 时就是 Output。
round 0 耗尽六轮；round 1 MODIFY contract，继承六条动作历史和三条 Tool
receipts，第八次生成完成；round 2 FINISH。raw=0。

初始 contract 自行引入 Phase III、Fail-safe drug trials、threat modeling。
第一次扩写查询被原词保真规则拒绝，之后 MedRAG 原词查询只返回凝血教材。
后续查询又偏向“trial design sequential treatment stages”，authoritative
search 的 PubMed 返回数为零，source.read 读的是教材。终局却把教材内容
归给 PubMed，并将相关概念当成了实体身份。这不是跨 Agent 消息丢失。

本题八次 MiniMax 调用含两次 parse error、两次准入错误、三次 Tool 与一次
complete；新的真实 error_message 已进入修复。流程变短不等于答案正确。

### ASTRONAUT：四节点链式重写没有纠正错误实体

输入 `astronaut trial`。实际链为本地 Qwen/ReAct → MiniMax/ReAct →
本地 Qwen/ReAct → MiniMax/ReAct(Output)。两个 DeepSeek 429 后由 Director
换模型。最早在 round 0 将原词解释为宇航员试验并加入原问题没有的 OASIS。
trial registry 返回零，Europe PMC 命中间接的太空药学资料，三次相近查询
又反复命中同一篇 Symposia。后续将“没找到”改写为“确认不存在”。
raw=36.36364%，但没有正确建立目标实体，不能称为答对。后两节点没有新的
Tool 调用，新增链式节点主要重写既有解释。

### IBD/HIV：有效完成，但问题没有被充分回答

输入询问 HIV 人群中 IBD 的频率。四节点链式图，本轮调用 knowledge、
authoritative、literature 等工具。raw=46.66667%，与 v2.42 相同；
较广泛疾病统计及其人群与目标指标没有充分区分，不能用更多文字替代
问题要求的统计结果。完整各节点输入/输出、引文与 evaluator 记录见下方私有报告。

### MDT：证据不足已经存在，主要推荐仍沿原计划

输入要求为指定病例写 MDT 结论并推荐指定治疗路径。实际图为
MiniMax/ReAct → MiniMax/reasoning → MiniMax/reasoning(Output)。
初始检索 contract 要求为用户指定方案寻找支持，并加入输入没有的前提。
原始 artifact 的 uncertainties 已说明检索证据未直接验证指定方案；下游
收到完整 7,823 字符 artifact 与两条 Tool receipts，却仍维持主要推荐。
raw=−44.44444%。

Director 不是完全没有尝试使用原始证据：round 2 新 Output contract 明确
要求读取 node_1 evidence，但被真实错误拒绝：
`Output closure cannot directly route declared dependencies [('node_1','node_3')]`。
当时 eligible_input_agent_ids 只有 node_2，随后只能退为读 node_2 总结。
这个限制不能归咎于模型不遵守 Skill，也不能靠更强提示声称已解除。

### Barrett：重复格式失败，扩图仍依赖失败节点

输入询问两种食管病变并存时的管理。round 0 建立 node_1→node_2，
node_1/DeepSeek 429；round 1 改 MiniMax 后成功检索两次、source.read 一次。
之后 round 2 只修改医学总结 contract，round 3 新增 node_3/4，却仍接
node_1→node_3→node_4。故障根节点持续阻塞消费者；pending round 4 双向
关系修改未完成执行，整题超时。无最终回复、无官方分数。

去重后的已保存 Agent 请求 19 条：18 次 MiniMax 返回、1 次 DeepSeek 429；
18 次均 stop，不是 length。13 次 parse error 包括嵌套字段三次、Markdown
围栏三次、未闭合/缺逗号六次、多余右括号一次；另有 query anchor 与 span
校验各一次。MiniMax 累计 569.94 秒；三次 Tool 总约 3.23 秒，均成功。
source.read 实际读出 1,016 字符，不能用摘要 read count=0 推断未读取。
新的错误消息已进入 round 2/3 的 Director prompt，但实际修复仍未针对格式层。

## 本轮确认的改进与仍有的限制

- 原始声明错误和 ReAct parser error 已真实传入后续 Agent/Director 输入；
  这属于工程反馈修复，不能推断医学得分必然增加。
- 当前 Output closure 是明确的 sink-only 输入域，且收尾状态只允许 ADD。
  源码 `_output_closure_sink_artifact_domains`、`_add_output_provenance_domain`、
  `model_admissible_action_types` 及既有 closure 单测锁定了该行为。
  “必须覆盖全部 sink”不等于“只能读取 sink”；后者是项目额外限制。
- v2.44 先做 candidate-only 迭代：把建议写成合法域内的 contract/model/repair
  操作，按信息需要选已有 Tool；不暗改核心准入，不固定医疗角色，不增加预算。
- 本批五题已经用于发现和调试，不能同时充当 Skill 发布的独立确认集。

## 完整证据与复现入口

配置：`config/evaluation_healthbench_professional_candidate_skill_v2_43_dev5.yaml`。
汇总、模型/工具/token/latency 统计位于
`reports/healthbench_professional_candidate_skill_v2_43_dev5/`。
全部原始结果位于
`artifacts/healthbench_professional_candidate_skill_v2_43_dev5/evaluation/`；
其中 `evaluator_private/agentgraph_development_demos.md` 和
`agentgraph_development_evidence.jsonl` 展开完整问题、Agent 输入/输出、通信、
Tool Action–Observation 与 evaluator receipt；超时完整已保存 prefix 位于
`evaluator_private/partial_trajectories.jsonl`。没有为报告重跑模型或评分器。
