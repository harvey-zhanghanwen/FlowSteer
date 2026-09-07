# HealthBench Professional v2.48：新五题结果与下一轮修复

源码69af58220355a0e123d500bcb6f33698d1880dca；独立分支
`feature/healthbench-v2.48-evidence-completion-skills-20260907`，已推送原GitHub
仓库remote backup。按用户要求另选五题，官方文件顺序排除此前dev5配置10个ID。
与v247不是同五题，不能用均分差异宣称回归或提升；全集曾历史评测，非独立盲测。

## 结果

4/5显式FINISH、4/5 evaluator-valid，一题900秒超时。**完整五题原始分与
长度调整均分均为N/A，未达验收、未启动525。** 已完成子集原始40.796703%、
长度调整29.038173%，仅为描述，不替代固定分母。无训练/权重更新/ACTIVE发布。

| Task ID后缀 | 公开任务摘要 | 原始分 | 长度调整分 | 终局图 |
| --- | --- | ---: | ---: | --- |
| 3533d9bfd2d32f8c465e7af62aec9781 | 将三点照护计划译成中文 | 0% | 5.62422% | Qwen node_1 |
| 9a160f86c59743692e46fab89aae42f2 | 餐后、平卧加重烧灼样胸痛的下一步及2026更新 | 100% | 93.92302% | MiniMax node_1 |
| fa30f3f57c7219130345f5c2e6d03d65 | 指定急性胰腺炎营养meta-analysis研究 | 34.61538% | 12.28902% | MiniMax node_1→node_2 |
| 2014ab7a9d8865f0da483817843ccbc5 | 指定静脉用药与溃疡再出血研究 | N/A | N/A | node_1→node_2，无Output |
| cd132a0c7cde74c0242aa8ef3850c9b9 | 指定肝硬化出血患者抗感染预防比较研究 | 28.57143% | 4.31643% | MiniMax→Qwen→MiniMax，三节点链 |

四条终局中2/4多Agent，均链式；未完成题另有两节点链，不当成功拓扑。
记录到42次Agent请求、40次Director阶段请求、15次实际Tool dispatch，仅涵盖
四条终局及其内嵌失败，不是完整费用。原始与长度调整指标直接来自官方reference，
0原始分的短答可能有正的长度调整值，不裁剪或改算准确率。

## 分类错误demo（固定分母5，类别可重叠）

### 1. 漏答与动作域过早收窄：1/5，20%

3533…原问题含三点计划并要求翻译。R0 Director创建Qwen Output node_1，
contract明确三点翻译。ReAct t1实际只输出87字符标题/病史，末尾还有无关短语。
完整输入已送达；provider正常stop，completion742 token，预算8192（visible4096），
不是超长截断。JSON value、execution.output和最终答复逐字一致，无通信丢失。

R1 Director收到完整原问题、完整Output和全部三条v248候选prior；但观察中的
动作域已经只有FINISH，尚余19轮、7节点容量。现成配置
finish_only_when_admissible=true在semantic_protocol=none时，把表面终局可接受
等同于只许结束。首个内容失败在node_1漏履责，随后配置阻断补救选择。
官方两条rubric均未命中，raw0。不向提示词加入rubric对应的临床答案。

### 2. 研究实体识别/证据覆盖不足：2/5，40%

fa30…R0检索职责把特定研究询问扩成多个综述的比较；DeepSeek429后换MiniMax
找到近期相关文献。R2补充节点的contract因“node 1's”和编号(3)被误拒，
R3重写成更广的综合职责，R4换MiniMax完成9594字符长文，R5被finish-only结束。
三条rubric仅一个一般结论命中，指定历史研究未涉及。来源有相关性不等于
目标研究已识别；长文也不能代替身份核对。下游收到来源，不是运输层丢数据。

cd13…初始两节点链；node_1未完成后，Director改contract预设“缺少直接比较
研究”。node_2沿此错误前提继续，后加node_3检索终于找到了直接比较研究，
终答明确纠正上游“没有”的断言，但10250字符仍缺部分评测要求，raw28.57。
首个可观察错误是把检索未找到提升为既定事实并写入contract，产生错误传播；
后续来源交接可用，Output能读到并纠正部分结论，不等于整体临床内容已验证。

### 3. 过长综合而未聚焦：2/5，40%

同fa30…和cd13…分别9594、10250字符；length adjustment分别降低约22.33、
24.26个百分点。它们同时存在实体/回应范围问题，不能只靠删字解决。下一版
应恢复可选编辑选择，让已有覆盖核对先验有机会生效；不手工截断终答来刷分。

### 4. 操作性编号/节点引用被误拒：1/5，20%

fa30…R2本来要求读取上游来源，但Canvas把任务编号及node 1所有格当临床数值。
这是确定的admission工程问题。下一版沿原literal提取修复，仅排除实际Canvas
引用与连续任务编号；不放宽真实剂量、年龄等未依据公开输入的数值。

### 5. Runtime/provider与收集超时：1/5未恢复，20%

2014…R0 DeepSeek429，R1/R2/R5 Qwen400，R3/R6 MiniMax协议循环，R4又429。
R7具体来源ID被拒，R8待Director时达到900秒；两节点均未形成最终Output，
不评分partial。400的原始响应正文未写入现有失败receipt，不能仅凭状态码
定性上下文超限或JSON Schema错误；正在只读核对本地服务日志，不重复API探测。

以上只把未恢复超时计作1/5；其它题也有已恢复429/解析失败，不冒充无故障。
已完成后的evaluator-invalid为0；grader有两条provider错误receipt但最终恢复。
已核对终局中的明确网络传输/输出裁断丢失为0，不虚构此类案例。

## Skill、来源与下一轮

v248候选先验真实启用，仍是可拒绝prompt prior，不是学习后的ACTIVE Skill。
从v247已知错误修的证据状态schema、局部原文反馈通过离线回归，但不能将本轮
新题评分归因于Skill独立收益。本轮最强证据是确定的finish-only配置阻断复核。

v249已独立准备：复用原有false开关恢复FINISH与合法编辑共存，修编号误拒；
候选文本目前保持v248，样本/模型池/生成/工具预算/evaluator不变。不开训练。
完整过程、每节点输入输出、通信、Tool receipts和逐条rubric见
`artifacts/healthbench_professional_candidate_skill_v2_48_dev5/evaluation/evaluator_private/agentgraph_development_demos.md`。
原生计分与统计见[自动报告](healthbench_professional_candidate_skill_v2_48_dev5/agentgraph_development_report.md)。
源码来源逐项见docs/source_map.md，单测与条件说明见docs/adaptation_log.md。
