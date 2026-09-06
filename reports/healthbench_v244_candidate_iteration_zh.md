# HealthBench Professional v2.44：五题候选 Skill 结果与失败分析

## 结论

源码/配置版本 `385ed219e8de2a6989ca44378f01bde0c3772b9b`。
固定五个已观察开发样本，2 题显式 FINISH 且 evaluator-valid，3 题达到
900 秒任务时限而取消。**完整五题 raw 和 length-adjusted 均为 N/A**。
两题有效子集的描述性均分为 raw **48.3333%**、length-adjusted **47.9100%**，
不能当作全五题、525 题或无偏泛化成绩，不能与 v2.43 的四题子集均分比较。

| 任务 / ID 后缀 | v2.43 raw | v2.44 raw | v2.44 length-adjusted |
| --- | ---: | ---: | ---: |
| WATERFALL / f056cdb489e3636b0b51afb8fd6b3a8a | 0% | 50% | 45.60764% |
| ASTRONAUT / ed8b3ca0a4dabfd0827c17a08513a181 | 36.36364% | N/A：超时 | N/A |
| Barrett / 4f08ae480b16ef825cf098eca6530e68 | N/A：超时 | N/A：超时 | N/A |
| IBD/HIV / dadbebd3dce1b5928cac5a44dde095d3 | 46.66667% | 46.66667% | 50.21231% |
| MDT / 37101607e2947481e85e8fe3597a1acf | −44.44444% | N/A：超时 | N/A |

完成率由 4/5 下降为 2/5。本轮局部内容改善仅能确认 WATERFALL；IBD raw
不变，调整后评分增加来自长度项，不代表内容正确率提高。官方指标不是
二元准确率。没有 Direct 重跑、训练、权重更新、ACTIVE 发布或完整 525。

## 失败类别

以下按五题统计，可重叠；不是把所有低分都归为一种原因。

| 可观察类别 | 题数 / 五题占比 | 代表及解释 |
| --- | ---: | --- |
| 运行未完成 / task timeout | 3 / 60% | ASTRONAUT、Barrett、MDT；没有终局分数 |
| 重复动作/证据 JSON 解析或校验失败 | 4 / 80% | 上述三题及 WATERFALL；后者恢复完成 |
| 故障前置节点后继续扩图 | 3 / 60% | 三个超时题；新节点多不能执行 |
| 实体解释或 contract scope 提前缩窄 | 3 / 60% | WATERFALL、ASTRONAUT、Barrett 初始动作 |
| 当前可见证据未充分回答目标 | 2 / 40% | WATERFALL 仍为间接来源；IBD 仍缺目标统计 |
| 实际 Agent token-length 截断 | 1 / 20% | MDT 最后一条已保存 ReAct 调用 |
| Provider 429 | 1 / 20% | ASTRONAUT 首次 DeepSeek；随后已换模型 |
| 工具 dispatch 超时/失败 | 0 / 0% | 本次已保存、去重的 Tool dispatch 均成功 |
| Scope admission 词面误判 | 1 / 20% | IBD 的 `patients with citations` |
| Grader error | 0 / 0% | 两条实际终局 evaluator 均有效；其余未进入评分 |

不将没有终局回答的样本虚构为“医学推理错误已评分”，不将未保存的调用
推定成功。真实 JSON 错误是执行协议问题，不等于最终回答排版问题。

## 典型过程

### WATERFALL：找回正确主题，但仅获得间接证据

输入：`waterfall trial`。目标是识别并解释指定试验，不能把相关术语当作
目标实体。round 0 ADD 两节点，先 node_1→node_2，node_1 为 Output；初始
contract 仍误入 piecewise-exponential/prognostic 解释，node_1/MiniMax/ReAct
失败。round 1 修改 node_1 contract，使用真实 registry 结果所指向的主题；
node_1 完成，但 node_2 耗尽 ReAct 轮次。round 2 将 node_2 改成已有
reasoning profile；round 3 改关系为 node_2→node_1，重新执行；round 4 FINISH。

最终识别了急性胰腺炎液体复苏方向，但主要依据另一试验注册条目对 WATERFALL
的引述，没有取到原始试验正文。输出保留了间接证据限制，raw=50%，不是全对。
首个失败是 Director 的实体假设，不是传输中丢掉原始论文。20 次 Agent
模型调用、7 次实际 Tool dispatch；完整输入/输出及通信见私有展开报告。

### ASTRONAUT：双向图存在，但故障节点反复执行

输入：`astronaut trial`。round 0 未消歧就写入 astronauts/spaceflight，
并加入无关检索目标；构图 node_1↔node_2→node_3。node_1/DeepSeek 429 后，
round 1 改 MiniMax；round 2 只扩写研究 contract。round 3 新增
node_1→node_4→node_5，round 4 增加 node_4→node_3；新检索节点仍等待故障
node_1。一次 draft 已合法，但 reciprocal revision 再次失败，不能把 draft
当成完整终局答案。

29 次 MiniMax ReAct 返回均 stop，累计 722.30 秒；17 次 parse、7 次其他
schema/admission 错误，4 次 Tool dispatch 成功合计 7.26 秒。另有 node_2
的 reasoning 调用，不能与 ReAct 重复计数。首个语义失败是实体假设；主要
运行阻塞是协议修复失败和后续依赖。无 final answer、无 grader receipt，N/A。

### Barrett：唯一可修改窗口被用于改写内容

输入询问两种食管病变并存的管理。round 0 因无证据的额外限定被拒绝；
round 1 ADD node_1→node_2。node_1/MiniMax/ReAct 取得三条成功 Tool 结果，
却未产出合法 completion。round 2 本可修改模型，但只改写医学职责。之后
round 3/4 新增 node_3/4/5，仍挂在故障前置链后；它们没有实际执行。

24 次 MiniMax 返回均 stop，共 719.61 秒；17 次 parse、2 次 query anchor、
2 次 evidence-span 错误；3 次工具成功仅 5.37 秒。真实错误已进入下一次
Agent/Director 输入，不是反馈缺失。后续输入增加了错误历史，但没有新有效
证据或有效模型切换。首个运行失败是序列化，错误传播由故障 prerequisite
阻塞消费者。任务超时，final answer / evaluator 均 N/A。

### IBD/HIV：流程更短，但内容分数不变

输入询问 HIV 人群中 IBD 的频率。round 0 contract 中 `patients with
citations` 被 scope guard 将 citations 误识别为新增临床实体，动作被拒。
round 1 ADD node_1→node_2，round 2 ADD node_2→node_3(Output)，均本地 Qwen，
round 3 FINISH。实际运行是三节点链，不是独立证据验证的因果证明。

最终回答表示未检得可靠的目标定量结果，并列举参考教材和间接论文。raw
仍为 46.66667%。回答变短后 length-adjusted 为 50.21231%，不能称为
找到了正确统计。首个工程问题是 scope 词面误判；终局内容问题是目标证据
不足。5 次实际 Tool dispatch、10 次 Agent 模型调用。

### MDT：检索已成功，长结构化回答反复失败

输入要求为指定病例生成 MDT 意见与指定治疗路径。round 0 三节点单向图，
node_1 取得三条 Tool receipt 后发生格式错误。round 1 更换为更长、更细的
报告 contract；round 2 新增节点仍依赖故障 node_1。另一个独立节点产出了
文本，但没有上游证据，不能当成全链完成。

18 次 MiniMax ReAct 返回，共 692.19 秒；14 次 parse、1 次 span 校验错误。
其中绝对 turn 18 确有一次 length=8192/8192，其余 17 次是 stop。三次 Tool
调用共 3.98 秒；PDQ metadata 也不能等同已读取正文。首个失败是执行协议，
而非断言已完成回答医学错误。最终任务超时，没有官方评分。

## 候选暴露、限制和下一步

v2.44 三条 candidate 逐轮进入 Director prompt；`visible_skill_ids=[]`
对应非 ACTIVE 路径，不表示没有注入。但不要在故障节点后新增消费者、
不要只改措辞等建议，未稳定落实到实际动作。

源码 `_repair_exhausted_agent_ids`（agent_workflow_env.py:7236）会在一次
bounded 修复后未新增 Tool receipt 时将节点移出 MODIFY 域；之后换模型
也被禁止。因此“再次失败后才考虑换模型”可能错过合法时机。v2.45 候选
将协议修复提前到首次已有多次 parser/action 错误的反馈；同一动作仅修改
允许字段或耦合执行 profile。资料已齐的消费者可选择 reasoning，缺证据
的检索节点不能无条件关闭 Tool。这仍是假设，不保证得分或完成率提高。

本版未改 sink-only Output 域、修复耗尽域或 scope guard；不靠候选文本
声称已经修好了这些核心限制。不为清零恢复计数额外调用 Tool，不绕过
证据、JSON 或 FINISH 校验。

## 复现与完整证据

入口：`config/evaluation_healthbench_professional_candidate_skill_v2_44_dev5.yaml`。
公开汇总：`reports/healthbench_professional_candidate_skill_v2_44_dev5/`。
原始证据：`artifacts/healthbench_professional_candidate_skill_v2_44_dev5/evaluation/`。

`evaluator_private/agentgraph_development_demos.md` 展开已完成题的完整输入、
参考目标、最终回复、各 Agent 输入/输出、communication、ReAct Action–
Observation、terminal 与 rubric receipt；`partial_trajectories.jsonl`
保留三个超时题的真实完整已落盘前缀。公开汇总的调用统计只覆盖终局轨迹；
本报告的超时统计来自对 partial receipts 的独立去重，不混称全轮计费总数。
报告完全从现有文件生成，没有重跑模型、检索或 grader。
