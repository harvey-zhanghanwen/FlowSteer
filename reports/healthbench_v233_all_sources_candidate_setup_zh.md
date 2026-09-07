# v2.33＋现有医学知识源＋候选 Skill：五题开发评测

## 版本与口径
本轮基于可恢复的 v2.33 源码提交 `3b18f7a866b189254b387dd2ccc406888744b9ad`；
工作树起点为备份提交 `5aed2903e7a5f8e3eef46c92bf92b94dd2d33e45`。
历史备份分支为 `backup/healthbench-v233-strict-high460-of525-20260906`。
历史 strict length-adjusted 27.4559%、raw 29.5213%，但 evaluator-valid 为 460/525。
这是历史报告的原始口径，并非 525 个有效评分；不能把本轮五题与之当作同样本提升比较。
本轮缺失项不会填零制造完整均分。

独立分支：`feature/healthbench-v233-all-sources-dev5-20260907`。
配置：`config/evaluation_healthbench_professional_v233_all_sources_dev5.yaml`。
实际 condition：`healthbench_professional_v233_all_sources_candidate_dev5`。
保留 v2.33 Director minimal-neutral.v19、Canvas incremental execution、communication v3、
自由 contract/节点数量/关系、唯一 Output、原有 FINISH 和恢复策略。
没有引入后续 v2.51 的编排策略或 communication v4。

## 必要适配和已有来源

| 模块 | 来源与本轮边界 |
| --- | --- |
| 多工具适配 | 现有项目 `9fb27de` 的 clinical ReAct/knowledge adapter；不是新写执行循环。其底层沿用 SkillFlow BoundedAgent 的 Action–Observation 和 FlowSteer incremental Canvas。 |
| 12 个工具 | 现有项目 `8359561` 的医学来源模块；知识索引复用 SkillFlow DocumentPassage/build_retrieval_index。 |
| Runtime 工具组合 | 仅移植 `9fb27de` 的 allowlist 两处修改，使已经注册的多个工具可以作为一个 ReAct profile；未移植其调度/历史 evidence/v4 改动。 |
| 搜索强制条件 | 原 v2.33 search-only 层强制初次检索和后续 query refinement，与 source.read/drug.lookup/calculator 等多工具层不兼容。按现有 clinical adapter 将两项改为 false；预算不变，工具仍由模型自主选择。 |
| Evidence 传递 | 扩展 v3 receipt 对新工具 ID、页码和来源元数据的识别；不改变 v3 消息投影方式。原始 receipts 留在 trajectory。 |
| HTTP 400 修复 | 保留 `8359561` 的实际 tokenizer 输入计数、输出预算边界和失败 receipt；不截断输入，不修改模型权重。Director 使用已有 `9fb27de` 上下文预算接口。 |
| 候选 Skill | 直接复用 `8359561` 的 candidate profile loader 和 `healthbench_candidate_skills_v248.yaml`；通过已有 collect(prompt_priors, forced_probe=True) 传入。此标志记录候选提示实验，不强制动作、不启用 MACE。 |
| 单臂评测 | 复用 `9fb27de` 的 collection-arm；本轮仅 AgentGraph，不重复 Direct 或额外五题对照。 |
| 评分 | 保留原 official simple-evals HealthBench Professional evaluator、grader 和 length adjustment；rubric/reference evaluator-only。 |

## 知识来源与模型边界
工具为 authoritative.search、medrag.search、source.read、drug.lookup、
computation.calculator、knowledge.search、literature.search、trials.search、
bookshelf.search、pdq.search、ahrq.search、terminology.search。
对应已有 MedRAG textbooks、PubMed、DailyMed、Europe PMC、ClinicalTrials.gov、
NCBI Bookshelf/PDQ/AHRQ、MeSH。source.read 和 calculator 是工具，不是额外数据库。
MedRAG 为已有本地语料；其余远端按需查询，未宣称全部数据库已经下载。
任务内 FTS5 索引只接收当前对话与实际来源结果，不索引 rubric、参考回答或 Agent 生成结论。

保留历史 v7 模型选择边界；本地 Qwen3.5-9B 支持 ReAct，远端混合模型仍按既有
tool_capable 声明提供 reasoning。未对未探测模型猜测工具能力。
Director 为 GPU6/8026 已有本地 Qwen3.5-9B；所有 thinking 配置与原模型目录一致。

三个候选建议是：保持实体与任务 scope、修复参数并保留 ReAct、
区分证据缺口与回答完整性。均可拒绝、无固定角色拓扑、无样本答案；
不是 ACTIVE Skill，不宣称已验证增益。训练、GRPO、LoRA、MACE、Bayesian、
Skill evolution 和 policy sync 均关闭。

## 固定样本与复现
五题沿用上一轮的 ID：3533d9bfd2d32f8c465e7af62aec9781、
9a160f86c59743692e46fab89aae42f2、fa30f3f57c7219130345f5c2e6d03d65、
2014ab7a9d8865f0da483817843ccbc5、cd132a0c7cde74c0242aa8ef3850c9b9。
这些任务已经用于开发，不是未接触的盲测集。不得据此宣称全量泛化提升。

在项目已有环境中提供 provider 凭据，不把凭据写入本报告或配置。
复现入口：
```bash
FLOWSTEER_SUPERVISOR_PORT=8026 FLOWSTEER_ROLLOUT_GPU=6 \
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_v233_all_sources_dev5.yaml \
  --collection-arm agentgraph
```
配置冻结后不在运行中修改；真实指标另行保存到本 condition 的 evaluation_report.json。
