# HealthBench 暂停、架构修复与数据库优化

## 当前状态与真实成绩

用户要求暂停后，已停止本任务的525题采集进程，保留GPU6本地模型服务。
暂停时保存25题有效评分、2条超时记录；部分轨迹诊断保留，不当成有效评分。
25题原始均分 **6.310924%**，长度调整均分 **6.935968%**。
这不是525题成绩；先前22题的9.58%是更早的子集快照。
没有重新评分，也没有把超时记零。新修复版本尚未运行解题评测，无新成绩。

## 数据库是否起作用

对暂停前全部25题，仅使用既有外部Tool receipts统计，包含失败执行已保存的调用：

- 21题调用工具并收到外部证据；知识库查询11次，外部资料原文读取5次。
- 返回的证据条目中，MedRAG 114、PubMed 20、Europe PMC 23、Bookshelf 2、DailyMed 2；
  是条目出现次数，不是去重文献数，也不是正确证据覆盖率。
- 有高分题使用了检索，也有零分题收到多条证据；没有同题/同配置数据库开关对照，
  所以不能给数据库计算因果提分。
- 旧库已有 `ASTRONAUT` 标题的PubMed记录，但错误轨迹向查询添加了太空相关解释。
  这是实体消歧和查询偏移，不是该研究完全不在库中。此事实只用于诊断，不写成别名规则。
- 原4422条记录中3923条摘录不足600字符；多数为教材短片段，不能把“有命中”当成
  “找到目标试验/完整方案”。旧PubMed记录中的文献ID没有统一放在PMID字段，影响精确检索。

## 完成的最小修复

1. **文本产物兼容性**：直接复用此前已实现的v4接口修复。中间ReAct节点可以交付实际
   文本，或继续使用逐条绑定真实Tool receipt的结构化证据；不再要求每项文本工作先检索。
   保留完整、去重、有界的文本交接。旧v3仍可重建。本次不把历史v4零分运行提升为最佳架构。
2. **研究名称的初次检索**：短独立研究请求在首次查询时保持原名，限制同时出现在模型
   可见JSON Schema和实际Tool admission；成功的原名查询允许后续细化。上游已有真实
   查询receipt可复用，不重复付费查询。知识库查询也不能绕过此检查；对话库检索不受限。
3. **检索结论的范围**：明显的无范围“没有任何研究”等表述通过既有ReAct反馈要求修正。
   不把局部检索空结果升格为不存在；正常临床否定、有限范围的不确定性与短答案仍允许。
4. **任务完成声明**：纯“已完成/待验证”不当成任务产物，通过同Agent有限续执行修复。
   新候选说明要求核对原对话、患者属性冲突及真实Output；不固定Verifier或三角色顺序。
5. **文献匹配**：复用SkillFlow标题加权FTS5，将真实标识/标题完整词匹配与相关性结果
   区分；后者仍使用原BM25+BGE混合检索。精确匹配仅表示候选文献匹配，不表示临床结论已验证。

## 新数据库

独立目录：`artifacts/healthbench_public525_evidence_corpus_v2/`。
旧库与索引不覆盖。

| 来源 | 新版记录数 |
| --- | ---: |
| MedRAG/textbooks | 3966 |
| NCBI PubMed | 511 |
| Europe PMC | 26 |
| NLM DailyMed | 2 |
| 合计 | 4505 |

3446条记录由短摘录补为本地真实原文passage，3966条教材记录可对应本地原始文档，
没有缺失的教材文档ID；511条补齐PMID及原文读取标识。
这里只是本地passage完整，不声称整本书或整篇论文全文完整。
外部537条医学文献记录保留“已检索摘录，全文完整性未确认”的状态；另有2条药品记录。
不从常识填造剂量、年份、研究结果或缺失摘要，不使用逐题rubric构建答案库。

语义索引另存 `artifacts/healthbench_public525_semantic_v2/`；复用本地BGE，CPU编码，
保持上游混合检索权重与阈值。没有训练Qwen、调用付费模型或运行525题问答。

## 复现入口

```bash
python scripts/refine_healthbench_evidence_corpus.py \
  --source-manifest <旧公开证据库manifest.json> \
  --medrag-path <现有MedRAG目录>/all_chunks.jsonl \
  --receipts artifacts/healthbench_professional_failure_skills_full525/evaluation/agentgraph_trajectories.jsonl \
  --output <新的空目标目录>
python scripts/build_healthbench_semantic_index.py \
  --source-manifest <新版证据库manifest.json> --output <新的空索引目录> \
  --model-path /ssd1/iclr/.private/skillflow-resources/bge-base-en-v1.5 \
  --config config/evaluation_healthbench_public_task_repair_full525.yaml --skip-query-probes
```

下一轮配置：`config/evaluation_healthbench_public_task_repair_full525.yaml`。
保持同525题和官方evaluator，但修改架构/候选/检索条件，必须使用新的运行目录；不能续入旧25条结果。
用户尚未要求恢复，因此本轮只做离线构建、定向测试和prepare-only，不运行该配置的真实评测。

## 必须保留的限制

- 源码格式检查不能判定全部临床正确性；背景段落被误当成完整回答等语义错误仍需实际评测。
- 英文短研究名和明显完成声明只是本次有边界的适配，不声称覆盖全部语言和任意别名。
- 这批公开题及错误已参与开发，后续是开发后复评，不是未使用测试集的泛化成绩。
- 候选Skill仍是可拒绝的提示先验，未通过独立消融，不发布ACTIVE、不进行训练。
- 公开通用评价维度可指导资料组织；逐题rubric及参考回答只留给evaluator，不用于资料筛选或索引。
