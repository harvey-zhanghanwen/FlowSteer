# HealthBench Professional v2.45：完成率提高，但尚未验证 Skill 的得分增益

## 本轮结果

冻结源码/配置：`04ff2a96ad501a755dc5e97bbee80d69145a6de3`。
备份分支：`feature/healthbench-v2.45-early-protocol-repair-skills-20260906`；
远端 `backup`：`https://github.com/harvey-zhanghanwen/FlowSteer.git`。

同五个已观察开发题全部结束：**5/5 FINISH、5/5 evaluator-valid、0 terminal
failure、0 task timeout**。整批官方 raw **29.33333%**，length-adjusted
**24.92157%**。这是公开 test 中已用于开发的五题，不是 525 题最终成绩或
无偏 held-out 估计。不存在将 N/A 置零、从中间回答回收答案或修改 grader。

| 样本 / ID 后缀 | v2.44 raw | v2.45 raw | v2.45 length-adjusted | 最终图 |
| --- | ---: | ---: | ---: | --- |
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 50% | 0% | 2.57838% | 1→2→3 |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | N/A：超时 | 0% | −3.59268% | 1→2 |
| Barrett / 4f08ae480b16ef825cf098eca6530e68 | N/A：超时 | 100% | 83.52130% | 1 |
| IBD/HIV / dadbebd3dce1b5928cac5a44dde095d3 | 46.66667% | 46.66667% | 48.80405% | 1→2→3 |
| MDT / 37101607e2947481e85e8fe3597a1acf | N/A：超时 | 0% | −6.70320% | 1→2→3 |

v2.44 只有 2/5 有效结果，完整五题均分为 N/A，不能拿其有效子集 48.33%
与本轮完整五题 29.33%作整体升降比较。同题可确认 WATERFALL 内容下降、
IBD raw 不变；三个原先超时题现在能评分。MDT 相比 v2.43 的 −44.44% 上升
至 0%，但仍有负 rubric 被触发，不是回答正确。

**自然多 Agent 比例 4/5=80%**，最终均为串行链；单 Agent 1/5，非链式
0/5。未强制任何医疗角色、模型或 topology。这五题中唯一 raw 满分的是
单 Agent，因此不能用多 Agent 数量解释得分改善。

本轮已记录 Agent 模型请求 58 条、Director phase 请求 58 条、实际去重
Tool dispatch 19 次。存在中途 provider/schema 错误但均收束。WATERFALL
grader 有一次已恢复 provider error；五条最终评分全部有效。完整 tokens、
模型、延迟和错误统计见自动报告，不将服务时间相加冒充端到端耗时。

## 主要失败分类

每题只取一个主要内容问题；修复/传输问题在后表另列。

| 主要类别 | 数量 / 五题占比 | 代表 |
| --- | ---: | --- |
| 实体消歧失败 / query drift | 2 / 40% | WATERFALL、ASTRONAUT |
| 目标关系未充分回答 | 1 / 20% | IBD：限定为可量化患病率，遗漏 grader 要求的相对关系 |
| 对用户所提结论的确认偏差 / unsupported recommendation | 1 / 20% | MDT |
| 本题 rubric 全部满足 | 1 / 20% | Barrett；不保证所有额外医学内容均已核验 |

以下为可重叠的工程/执行现象：

| 类别 | 数量 / 五题占比 | 典型案例 |
| --- | ---: | --- |
| ReAct→reasoning 后原节点历史 Tool 证据未进入实际输入 | 2 / 40% | WATERFALL、MDT |
| 只有未来检索计划的中间 artifact 被接收 | 1 / 20% | IBD node_2 |
| 真实长度截断 | 1 / 20% | MDT node_1 第一次 reasoning，8192 tokens、空 visible completion |
| scope admission 的普通缩写展开误判 | 1 / 20% | IBD 前两次 ADD |
| task timeout / max-rounds termination / 最终 evaluator-invalid | 0 / 0% | 没有案例，不虚构 |

## 五个可复现 Demo

完整输入、Agent contract/输入/输出、每次通信、Tool Action–Observation、
来源 receipt 与 rubric 文本均保存在文末 evaluator-private 报告。下文是
便于阅读的实际过程摘要；原题与 grader 要求不进入任何候选 Skill。

### 1. WATERFALL：错误结论传得完整，检索证据没有跟随模式切换

输入：`waterfall trial`。目标是识别指定研究，不能回答成一般试验设计。
最早 node_1 查询擅自加入 `medical research methodology`，被拒后仍偏向
trial design/plots；工具获得的是邻近概念。Qwen/ReAct 两次 evidence
completion 缺字段，耗尽六轮。round 2 切 node_1 为 reasoning、tools=[]。

该次真实请求 `upstream=[]`、`own_draft=None`、`continuation_source_agent_id=None`；
rendered messages 没有原先查询及 PMID 30942559。随后 node_1 凭模型知识
声称 FDA/EMA 没有这种设计，并声称检查了 ClinicalTrials.gov，然而本次没有
trials Tool 调用。node_2 转述结论，node_3(Output) 的 contract 又写入该结论，
形成 1→2→3，round 6 FINISH，raw=0。

首个语义失败：原词未消歧就加入解释。首个证据传递问题：同节点切 execution
mode 后旧 Tool Observation 未进入新的模型请求。不是下游没收到上游文字，
而是它们完整接收了没有可靠证据支持的结论。

### 2. ASTRONAUT：相关文献不能绑定目标实体

输入：`astronaut trial`。最终链 node_1/MiniMax/ReAct→node_2/MiniMax/ReAct
(Output)。最初 DeepSeek 429，Director 改模型后完成执行。

node_1 初始 contract 接受了先消歧的建议，但首次查询仍加入 spaceflight。
真实原词检索没有建立实体身份；artifact 正确标为 insufficient，却把备选
解释都局限于太空/NASA。Director 随后要求下游定义 probable entity，
node_2 三次查询继续按这个假设展开，最终把相关综述当作“最可能解释”。
10 次 MiniMax 返回都是 stop，0 parse error，6 次 Tool 共约 7.58 秒；
round 3 FINISH，raw=0。首个失败在查询解释，随后 contract 固化了假设。
这里不是检索超时，也不是节点太少。

### 3. Barrett：一次实际来源读取，单题 rubric 满分

输入询问两种食管病变并存的管理。round 0 ADD 单个 node_1/DeepSeek/ReAct，
同时指定为 Output；429 后 round 1 改 MiniMax。随后 literature search、
authoritative search；source.read 第一次误传 query，收到真实参数错误，
改用 source_id 后读取 PMC12224665 的 16,000 字符片段，再 complete；
round 2 FINISH。

最终 7,605 字符回复引用实际检索文献，满足两条 rubric：活检需谨慎/避免
的条件、合并管理证据有限。权重 9+7，16/16，raw=100%，长度扣除16.4787
个百分点后为83.5213%。这只是本题两条 rubric 全满，不是全篇医学内容
逐项验证，也不是数据集达到100%。

相比 v2.44，没有中间 structured evidence artifact 的长解析循环；实际
模型切换是已有429恢复，不是“重复parse后早切换”被独立验证。因此不能
将该题提升完全归因于新候选 Skill。

### 4. IBD/HIV：修复模型成功，但消费者交付仍是计划

原题：`How common is IBD in patients with HIV?`。前两次 ADD 的正常缩写
展开被 scope guard 拒绝。round 2 建 node_1→node_2，node_1/MiniMax/ReAct
出现两次 parse、两次 source ID 错误。round 3 真正改 node_1 为本地 Qwen，
response receipt 记录继承了两条 Tool receipts，随后交付 insufficient
evidence artifact。这个实例确实发生了首次协议失败后的模型修复。

node_2/DeepSeek429后换MiniMax，仍发生格式失败；round 5切reasoning，
但 contract 还要求检索，输出只有221字符的“我将进一步检索”计划。
round 6新增node_3/Qwen/reasoning Output，round 7 FINISH。下游实际消息
仍含祖先检索摘要，包括 PMID8823573；不能说整条通信已经丢掉全部证据。

最终回答说明无法从检得资料计算明确比例，raw=46.66667%。grader确认
“没有统一具体估计”一项得7分，另一项相对风险/患病率关系未满足，得0，
合计7/15。不向候选补写该医学结论。首个工程问题是scope误判，最终问题是
目标关系覆盖不足，以及 Tool-free 节点仍执行“检索计划”contract。

### 5. MDT：模式切换完成了流程，没有纠正确认偏差

原题要求为指定喉癌病例写 MDT 意见，并建议指定治疗序列。round 0 contract
已要求“寻找支持该序列的证据”，而非评估其是否成立。DeepSeek429后改本地
Qwen/ReAct，取得Tool结果但completion字段仍错；round 2切reasoning。

真实reasoning请求无upstream且未含旧Tool观察，生成8192tokens后length，
visible completion为空；Canvas确实拒绝空artifact。round 3 Director把
其解释过的论文结论写入contract，再运行产出推荐；node_2/MiniMax综合。
round 4新增Output想读node_1却被sink-only域拒绝，round 5退为node_2→node_3
(Qwen3.5 Flash/reasoning)，round 6 FINISH。Director仍是本地Qwen3.5-9B，
Flash只是workflow内执行模型。

最终加入了按反应决策，但仍沿用有争议的默认后续治疗表述。rubric中正向
10分与负向−10分相抵，raw=0，length-adjusted=−6.7032%。首个可观察问题
是支持性contract；模式切换的信息缺失和把解释写成既定证据进一步传播错误。
与v2.43负分相比改善，不等于推荐正确。

## 这次 Skill 到底验证了什么

- 三个可拒绝候选真实进入Director输入；模型、自由contract、图关系仍自主选择。
- IBD真实出现早期MiniMax→Qwen协议修复；Barrett和ASTRONAUT的换模型主要
  是原有429恢复；不能把所有成功都归给新Skill。
- 消费者按需reasoning的想法，被Director用于未完成检索的生产者，且模式
  切换没携带旧证据。候选的前置条件没有被稳定满足，不可推广为有效Skill。
- 已知核心问题包括跨execution-mode的证据输入、仅计划artifact的完成判定、
  sink-only Output域及repair-exhaustion后的MODIFY限制。继续加长提示词
  不能代替这些接口问题的修复，也不能把“能FINISH”当作内容完整。
- 本版不替换历史最佳profile，不发布ACTIVE，不启用训练/MACE/Bayesian，
  不继续启动525或重复付费轮次。下一步应先修复证据交接，再重新验证候选
  前置条件；不得用测试题答案、rubric或固定医疗workflow补分。

## 文件与恢复入口

配置：`config/evaluation_healthbench_professional_candidate_skill_v2_45_dev5.yaml`。
候选：`config/healthbench_candidate_skills_v245.yaml`；来源/适配说明位于
`docs/source_map.md`、`docs/adaptation_log.md`、`docs/healthbench_v245_skill_changes.md`。
本轮只改这些配置和说明，没有修改共享runtime。

完整结果目录：`artifacts/healthbench_professional_candidate_skill_v2_45_dev5/evaluation/`。
其中`agentgraph_trajectories.jsonl`为原始轨迹；
`evaluator_private/agentgraph_development_demos.md`为五题完整展开案例；
`evaluator_private/agentgraph_development_evidence.jsonl`保存节点、通信、Tool和评分。
公开汇总与调用统计：`reports/healthbench_professional_candidate_skill_v2_45_dev5/`。
这些报告由现有文件离线生成，没有额外评分调用。必要报告与源码推送独立
备份分支；大型原始artifacts保留本地，不含在Git源码备份中。
