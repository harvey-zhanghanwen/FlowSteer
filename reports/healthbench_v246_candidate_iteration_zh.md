# HealthBench Professional v2.46：固定新五题未通过

源码：`8ca968750b08170c6c935d31e1350dc5e4553727`；分支
`feature/healthbench-v2.46-react-evidence-disambiguation-20260907`。
运行自2026-09-06 23:48 UTC至2026-09-07 00:03 UTC，GPU6/8026。
官方public test按原顺序选取另外五题，固定ID见协议；这些不是此前未接触的
独立held-out。ReAct-only、候选Skill、原模型池/thinking、每题900秒、并发4。
没有训练、权重更新、MACE、Bayesian、ACTIVE或Skill evolution。

## 真实结果

**3/5 FINISH且evaluator-valid，2/5收集超时；完整五题raw和
overall_score_length_adjusted均为N/A。未达到完整五题>60%的启动门槛，
未运行完整525题。** 已完成子集原始分0%，逐题length-adjusted算术均值
−3.59170%；子集不能替代固定五题或与上一轮不同五题作提升比较。
原runner进度里的completed-only reward裁剪值不是原生负分均值；本报告与
离线报告直接读取 `evaluation.metrics`，不裁剪、不回收草稿、不缺失置零。

| Task ID后缀 / 任务概述 | Raw | Length-adjusted | 完成状态 | 最终有效图 |
| --- | ---: | ---: | --- | --- |
| 9566084de89c416408691006a6f06f9c / 丹麦语儿童反复中耳炎、手术条件 | 0% | 5.15970% | FINISH / valid | node_1→node_2 |
| c19c2113ba68bb3c4a3e63836e31b558 / 面瘫伴耳道疱疹 | N/A | N/A | 900秒Timeout | 无终局，不用部分图替代 |
| a5778c7ecdb4eeccf9d252631e18a274 / 儿童听觉问卷分数解释 | 0% | −20.77110% | FINISH / valid | node_1→node_2→node_3 |
| e339f34a3a35f3f067422b5768287f7c / 多轮对话后儿童耳部处理 | N/A | N/A | 900秒Timeout | 无终局，不用部分图替代 |
| c42bd4fc760487ac7b5e70fbb41a8edc / 婴儿耳内用油及用户提供文献 | 0% | 4.83630% | FINISH / valid | node_1→node_2 |

已完成三题均为自然多Agent串行图；不据此推算全部五题拓扑，也不能认定
多Agent或候选Skill提升。已保存终局轨迹含51次Agent模型请求、38次Director
阶段请求及14次去重Tool dispatch；这些不覆盖两条超时partial，不是整轮费用。
局部provider 429和一次grader500已记录；grader500恢复，未导致该题无效。

## 分类与代表性Demo

主要结果分类（互斥，分母5）：内容失分3/5=60%，执行超时2/5=40%。
下列工程/内容问题可重叠，不能相加当作互斥分布。

| 已确定现象 | 题数 / 固定5题占比 | 代表 |
| --- | ---: | --- |
| 非英文task anchor的Unicode词误拒 | 1 / 20% | 9566… |
| 外部search的参数预检不一致，确定本地错误仍耗Tool次数 | 1 / 20% | a577… |
| contract将未经证实数值预设传成事实 | 1 / 20% | a577… |
| 文献标识符namespace混淆 | 1 / 20% | c42b… |
| 未命中被扩写成不存在或整体不能回答 | 3 / 60% | 9566…、a577…、c42b… |
| 已检查终局题中的上游证据未传到实际下游输入 | 0 / 0% | 三题均实际传入；不虚构丢失案例 |
| JSON结构/来源元数据重复被误当正文退化 | 1 / 20% | c19c…，有效中间产物被拒 |
| 裸文本/证据缺字段反复生成 | 2 / 40% | c19c…、e339…，导致未完成 |
| 非JSON终答真实重复退化 | 1 / 20% | e339…，原拒绝应保留 |
| 最终evaluator-invalid（已FINISH三题） | 0 / 0% | 无案例；另外两题没有终局评估 |

### 9566…：检索不足被升级成广泛否定

输入完整保留丹麦语对话，询问1.5岁儿童反复AOM及打鼾时的手术条件。
R0 Director预填数字范围被拒；R1 node_1查询包含损坏文本，之后合法共同词
`dræn`因ASCII切词丢失被误拒。R2查询仅靠一个共同词检到无关段落，却被算作
第2次相关成功查询，搜索域关闭；node_1最后提交空evidence/insufficient。
node_2实际收到完整628字artifact及全部六段上游来源；没有信息丢失。
其终答245字却声称可用指南/著作没有明确标准。R6 FINISH，尚余14轮。
grader两条rubric均未满足：缺参考依据、未表达年龄界限可变性；raw0。
这些评测要求仅用来解释失分，不写入候选提示词。

首错为检索职责缩窄/损坏查询；确定代码问题为Unicode误拒。后续为无来源
广泛否定和Director未识别答案不足，而不是“没有看到rubric”。

### a577…：无效查询耗尽预算，假设数值沿正常通信链传播

原题只提供Quiet=16、Noise=10、Overall=29。R0 node_1一次过长查询在
authoritative工具前被拒；换literature后同一类错误却在客户端抛ValueError，
消耗两次dispatch。第三次Bookshelf为空，后续查询被预算正常拒绝。
R1中性修复措辞unresolved alternatives被scope guard误当临床答案；R2获准
contract反而预填原题和来源都没有的分母/范围。node_1空证据摘要复述这些
假设；node_2后来实际检得文献，但未读截断摘要的后续内容，结论由Director
再次固定为不能分类。node_3正常收到全文artifact和Tool receipts，再扩写
未经证实的版本差异，输出9065字。官方raw0、length-adjusted−20.77110%。

首错为查询参数；确定接口问题是跨工具预检不一致。通信本身通畅，传播的
却包含未核验假设。缺少规范资料不自动证明分数无任何可解释含义；不能
反过来按rubric写定具体解释。

### c42b…：来源访问报告替代了完整回复

原对话讨论婴儿耳内用油，用户追问ScienceDirect链接。R0 node_1 contract
要求识别publisher PII，但artifact称其为PMID并报告无命中。R1新增node_2
综合，DeepSeek429后R2切Qwen；实际上游来源与完整原对话已传入，终答只说
不能访问文献。R3 FINISH，raw0；rubric要求的关键用油提醒未被包含。

首个实体错误是PII/PMID混淆；后续是局部访问失败替代整体回答。候选职责
应区分身份匹配、检索状态和答复覆盖，不预置该题的医学答案。

## 证据与后续边界

### c19c…：合法结构化证据被误判为重复，然后修复未收敛

输入为28岁患者面瘫伴耳道疱疹的诊断与处理请求。R0 provider429恢复后，
R1检索成功但经历引文/解析失败；R2/node_1/t9完成8230字结构化artifact。
Runtime却因24-token窗口出现3次拒绝；离线复核唯一重复3次的窗口是JSON
字段与同一本书的来源元数据，实际正文最多重复2次。这是确定工程误拒。
后续Qwen反复只生成五个来源字段，缺claim、qualifier和span，错误延续至
R11/t39；Director不断改contract/关系，未设置Output。900秒后取消，无
终局评分。旧中间产物没有被回收成正式答案。

### e339…：解析/证据协议循环，最后终答确实发生正文重复

输入是完整多轮对话后要求处理3岁儿童单侧耳部问题。R0/node_1/t1检索成功，
t2返回裸Markdown而非结构化动作；node_1累计43turn才完成，其中32次解析
错误、7次schema错误。后续节点又出现证据缺字段。R11–12 workers完成却
仍无Output，R13新增Output的标题措辞被scope guard误拒一次。R14/node_5
自然语言终答2117字存在斜杠/箭头乱码，24-token窗口真实重复7次，质量拒绝
是合理的；R15等Director返回时900秒取消。该题不应通过放宽正文重复阈值
取得形式完成，也没有可合法评分的终答。

两个超时不是“增加模型最大输出长度就解决”的统一问题：c19存在结构化
质量检查误拒，e339最后是真实生成退化。新版本必须保持这两个边界不同。

- 原生分数、调用统计：[离线报告](healthbench_professional_candidate_skill_v2_46_dev5/agentgraph_development_report.md)。
- 完整对话、所有Agent输入/输出、通信、ReAct与grader receipts：
  `artifacts/healthbench_professional_candidate_skill_v2_46_dev5/evaluation/evaluator_private/agentgraph_development_demos.md`。
- 两题中止前真实状态：同目录`partial_trajectories.jsonl`，明确non-scoreable。

v2.47在独立worktree准备上述通用接口修复和可拒绝职责建议。保持同五题、
工具总预算、官方evaluator与自由图结构，不做训练，也不把这轮标为最佳。
