# HealthBench：v2.33＋现有医学知识源＋候选 Skill 五题结果

## 结论
本轮已经结束，没有训练或权重更新。配置中的三个候选 Skill 实际进入 Director
输入；现有医学知识源已注册并发生真实检索。**4/5 题获得有效官方评分，
四题原始均分 26.9231%，长度调整后均分 26.4791%；完整五题均分 N/A。**
剩余一题在 Director 输入 32,799 tokens 超过 32,768 上限时被发送前检查拦截，
没有终局答案，不能填零或当作已完成。未追加补跑或第二个付费对照条件。

- condition：`healthbench_professional_v233_all_sources_candidate_dev5`
- 源码：`f1c97db902119098d32bbe11055989c1ecb48567`
- 分支：`feature/healthbench-v233-all-sources-dev5-20260907`
- 基础模型 policy：`qwen35-9b-base-healthbench-professional-mixed-all-thinking-v2-33`
- adapter：无；ACTIVE Skill：0；optimizer update：0。
- 评测时间：2026-09-07；同一组五题开发样本，不是盲测或全量 benchmark。
- evaluator：`openai-simple-evals-healthbench-professional-652c89d@1`；
  grader：`gpt-5.4-2026-03-05`。
- Direct 本轮未重跑；不能引用其他 tool condition 的 Direct 声称配对提升。
- [配置与来源说明](healthbench_v233_all_sources_candidate_setup_zh.md)
- [原生评分汇总](healthbench_professional_v233_all_sources_candidate_dev5/evaluation_report.json)
- [逐题结构与调用统计](healthbench_professional_v233_all_sources_candidate_dev5/agentgraph_development_report.md)
- [完整本地输入、输出、Director/Canvas、通信、Tool 和 rubric receipt](../artifacts/healthbench_professional_v233_all_sources_candidate_dev5/evaluation/evaluator_private/agentgraph_development_demos.md)
- [未完成题的错误记录](../artifacts/healthbench_professional_v233_all_sources_candidate_dev5/evaluation/collection_failures.jsonl)

## 逐题结果

HealthBench 的分数是加权 rubric score，不是普通 Accuracy。
原生负分条目可使 raw 为负，长度调整可以使分数超过 100%，本报告不截顶。

| Task ID 后缀 | 任务简述 | 原始分数 | 长度调整后 | 终局 | 实际末次图 |
| --- | --- | ---: | ---: | --- | --- |
| 3533…9781 | 医疗计划中文翻译 | 100.0000% | 104.6158% | FINISH | node_1 → node_2 |
| 9a16…42f2 | 餐后胸部不适对话 | N/A | N/A | Director 上下文超限 | 无可评分终局图 |
| fa30…3d65 | 胰腺炎 PN/EN 文献比较 | 69.2308% | 58.6615% | FINISH | node_1 → node_4；node_2 → node_3 → node_4 |
| 2014…cbc5 | 静脉奥美拉唑研究 | −61.5385% | −62.0912% | FINISH | node_1 → node_2 → node_3 |
| cd13…c9b9 | 两种抗菌药物预防感染研究 | 0.0000% | 4.7305% | FINISH | node_1 → node_2 |
| 有效四题均分 | 4/5 有效 | 26.9231% | 26.4791% | 4 FINISH | |
| 完整五题均分 | 固定分母 5 | N/A | N/A | 1 个运行失败 | |

四个完整图都由多 Agent 组成：2 Agent 两题、3 Agent 一题、4 Agent 一题。
链式 3/4，带分支汇合的 DAG 1/4，没有双向关系；不能据此声称模型已经学会
普遍的非链式编排。第四节点的实际模型选择包括 MiniMax-M3，不是全部 Qwen：
PN/EN 题 node_1、node_2 为本地 Qwen3.5-9B ReAct；
node_3、node_4 为 MiniMax-M3 reasoning，其余完成题为本地 Qwen。
ReAct 是每节点 execution_mode，不是 role。

## 与历史 v2.33 同题对照

历史备份的 525 行评分快照中，这五题均有原生评分。仅比较本轮完成的共同四题：
历史 raw **8.6538%**、adjusted **7.1963%**；
本轮 raw **26.9231%**、adjusted **26.4791%**，
分别高 **18.2692**、**19.2828** 个百分点。
该比较是**共同完成子集的描述性结果**，有本轮缺失项造成的选择偏差，不是固定五题通过结果。

| Task ID 后缀 | 原 v2.33 raw | 本轮 raw |
| --- | ---: | ---: |
| 3533…9781 | −100.0000% | 100.0000% |
| fa30…3d65 | 34.6154% | 69.2308% |
| 2014…cbc5 | 100.0000% | −61.5385% |
| cd13…c9b9 | 0.0000% | 0.0000% |

变化并不一致，有改善也有严重退化。两轮除候选 Skill 外还改变了工具集合和
必要适配层，且未重跑同期无 Skill 对照，**无法隔离 Skill 的净增益**。
历史全五题 raw 6.9231%、adjusted 5.7094%，本轮缺一题，不做全五题差值。
参考：[历史同题评分快照](../backups/healthbench_professional_v2_33_strict_high460/receipts/agentgraph_score_rows_460_snapshot.jsonl)。

## 失败分类与代表案例

下表按固定五题的主要可观察失败归类，互斥计数；
“首个可观察失败”不等于已证明的唯一因果机制。

| 主要类别 | 题数／5 | 占比 | 代表 |
| --- | ---: | ---: | --- |
| Context-window exhaustion／运行未完成 | 1 | 20% | 9a16…42f2 |
| Task-scope drift＋相关文献替代目标研究 | 1 | 20% | 2014…cbc5 |
| Evidence utilization／回答不完整 | 1 | 20% | cd13…c9b9 |
| Evidence coverage 不足／部分正确 | 1 | 20% | fa30…3d65 |
| 未触发本题 rubric 扣分 | 1 | 20% | 3533…9781 |

附属问题非互斥：已完成轨迹中 2 个任务存在 Agent runtime failure 后恢复；
1 题有 source.read LookupError；1 题出现过两次 grader HTTP 500 后成功恢复。
没有最终 grader-invalid 题；没有 max_rounds 终局；没有可证实的关系反向或消息未送达案例。
这不表示所有临床事实均已验证，也不表示无其他 latent bug。

### A. Director 上下文超限
完整 ID：`healthbench-professional:9a160f86c59743692e46fab89aae42f2`。
原始输入和当前运行进度保存在本地 demo/selected_tasks 中。
round 3 报错：
`Director context exhausted before generation: input_tokens=32799, max_context_tokens=32768`。
此次不是本地 HTTP 400：请求在发送前被拒绝；输入没有被静默截断。
来源为 Director 上下文容量边界，不能归为答案内容得零。
v2.33 旧采集路径没有落盘该任务完整中间 trajectory，报告仅保留实际存在的
progress/failure，不重建缺失 Agent 输入输出，也无法按组件精确拆分超限贡献。
下一步需要基于既有上下文预算/结构化反馈实现边界内的消息投影，而不是增加医学答案提示。

### B. 三 Agent 顺利执行，但研究对象错位
完整 ID：`healthbench-professional:2014ab7a9d8865f0da483817843ccbc5`。
round 0 Director 将问题表述为“寻找相关研究或系统综述即可”，生成
node_1 检索 → node_2 阅读并归纳；round 1 增加 node_3 Output，round 2 FINISH。
实际 literature.search/source.read 检索到相关的新研究，证据进入后续节点。
最后输出变成广泛研究综述，未覆盖 rubric 对应的目标研究，并触发负分条件。
首个可观察偏差是 contract 将研究识别任务扩大为一般主题检索；下游忠实使用了
**错误对象**的证据，而非通信丢失。多 Agent 本身不能纠正这一偏差。
完整原题、输出、逐条 rubric 判定和 Tool source ID 均见私有 demo。

### C. 检索和数据库实际返回信息，但 Output 只有开场白
完整 ID：`healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9`。
round 0 node_1 使用 Europe PMC、AHRQ 搜索；其中一条 query 把原药名拼为
`orofloxacin`，且主要得到相关综述，未正确定位特定研究。
它把“这次没找到”扩大为较强的证据缺失总结。
round 1 node_2 调用 knowledge.search，收到药物耐药相关内容，但最终只写
“Below is a summary…”的引言，没有真正给出 summary；round 2 Director FINISH。
该终局模型调用 `finish_reason=stop`，不是 max_tokens 截断。
首个偏差在 node_1 的检索/证据缺失推断；node_2 未使用已经可见的相关事实，
Director 又没有识别回答不完整。此次存在真实数据库调用，不能解释为数据库未接入。

### D. 非链式汇合提高本题分数，但仍缺文献并过长
完整 ID：`healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65`。
实际图：
```text
node_1（Qwen3.5-9B，ReAct）─────────────────┐
                                           ↓
node_2（Qwen3.5-9B，ReAct）→ node_3（MiniMax）→ node_4（MiniMax，Output）
```
round 0/2 node_1 曾耗尽六次 ReAct turn；随后 Director 增加 node_2→node_3，
保留原节点；恢复后 round 5 将两分支汇入 node_4，round 6 FINISH。
支持性来源与结论进入最终长回答，命中两个正向 rubric 条目，另一个仍缺失。
raw 69.2308%，长度调整扣约 10.57 个百分点，adjusted 58.6615%。
这证明本轮能形成并运行非链式 DAG，不证明这种结构普遍优于链式。
此外 round 1 的 scope 校验把普通编号 `1)` 中的 `1` 误判为预填临床答案，
拒绝了修复 contract：这是原 v2.33 admission 的明确误拦截，随后改写才通过。
为保持冻结条件，本轮没有中途修补这一代码问题。

### E. 高分例也有评价边界
完整 ID：`healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781`。
node_1 初次 reasoning 产生空 Artifact，Director 将其改为 ReAct 后获得包含风险提示的
中间内容；node_1→node_2 后完成中文输出，rubric raw=1。
本题没有实际 Tool 调用；提高来自本次输出内容变化，不能归因于新增数据库。
其参考 rubric 很少，满分不意味着翻译忠实度或所有医学细节都已完全验证。

## Skill 注入证据、调用量与已知缺口
四条完整轨迹共 17 个 Director turn，每个 turn 的保存 prompt 都含三个候选 ID；
trajectory 标记 forced_probe=true、active_skill_ids=[]。这是候选 prompt 条件，
不是 ACTIVE 技能发布，训练依然关闭。

已完成四题记录 51 条 Agent model receipt、34 条 Director phase receipt、14 次
去重 Tool 调用；完整失败题的调用未全部保留，所以这些**不是整轮账单总数**。
实际使用的工具包括 authoritative.search、literature.search、source.read、
AHRQ search、knowledge.search 和 calculator。十二工具都可选不代表十二工具都使用；
也不代表已把所有远端数据库全文下载。

在已保存终局轨迹和 collection failure 中未见本地 HTTP 400，但上下文超限造成的
运行失败仍未解决。两个 Agent TimeoutError 发生在同一最终完成任务中；
本轮没有 task-level TimeoutError、没有全量五题结果。后续优先修复：
上下文投影边界、编号误拦截、研究身份定位、完成回答的语义完整性。
不为这些问题硬编码医学事实、固定角色模板或 Ground Truth。
