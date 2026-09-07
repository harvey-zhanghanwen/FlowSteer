# HealthBench：525 题问题感知证据库与架构修复

## 执行范围与评测条件
基于用户要求，用 525 条官方公开对话构建问题感知的外部医学证据库。
不是题号到答案的查表库；不使用 rubric、参考回答、最终答案或分数选择/生成库内容。
全部题目适用同一建库算法，不按高分/低分挑选记录。
评测称为“问题感知检索增强适配条件下的五题复测”，保留官方评分公式。
建库看过全部 525 条公开对话，因此不是未接触测试问题的 inductive evaluation；
不是答案泄漏，但不能省略这个条件。五题仍沿用上一轮固定 ID，不按结果重选。
配置内部 stage=development 继续防止把重复调试样本误标为 final benchmark。

## 建库实现与来源

| 部分 | 复用来源／必要适配 |
| --- | --- |
| 525 条公开对话 | 现有 HealthBench Professional TaskRecord.question + parse_model_visible_conversation；只提取 task_id/question。 |
| 医学教材检索 | 直接复用 FrozenMedRAGBM25Corpus.search，来源 SkillFlow training/environment.py::_load_external_corpus/_search_external_corpus；不重写 BM25。每题使用最后用户消息及完整对话两种公开 query（相同时去重）。 |
| 历史外部文献缓存 | 现有 report_multidataset_stable_zero._tool_receipts 和 openai_gateway._healthbench_search_candidates；只读取实际外部 Tool Observation，不使用 Agent 总结或 evaluator。缓存有无评分均同样处理。 |
| 数据库和索引 | 现有 HealthBenchKnowledgeStore.ingest_evidence/search，底层 SkillFlow DocumentPassage/build_retrieval_index/SQLite FTS5；保留文献 ID、来源、摘录、日期、页码。 |
| 本轮新增接口 | build_healthbench_question_corpus.py 仅串联已有检索/投影/索引模块；load_frozen_public_evidence 将冻结来源记录注入原 knowledge.search，保留每次执行与图上通信的隔离。 |
| 实际检索路径 | 原 ReAct knowledge.search 查询冻结外部来源及当前节点实际接收的 receipts，不能访问其他题的对话或回答。未把全部文献正文放入 Director prompt。 |

建库复用既有外部资料，没有新增模型、grader 或远端 API 调用。本地 CPU 检索不是
训练。question_retrieval_receipts 单独保存用于复现，不进入模型可查询库。
“525 题均有返回”若发生，只表示词法检索非空，不代表证据覆盖或答案正确率。

## 架构修复
1. **FINISH 可选而非强制**：沿用已有 AgentWorkflowEnv 配置，设
   finish_only_when_admissible=false。旧值 true 会在输出形式合法后只允许 FINISH，
   硬性阻断候选 Skill 提议的修改。这是上一轮“Director 忽略完整性建议”的更具体原因。
   现在保留 FINISH，也保留合法 ADD/MODIFY/关系修复，不强制添加 Verifier 或固定角色。
2. **Director 上下文投影**：直接移植现有提交 8359561 的
   AgentGraphOrchestrator._context_feedback_projection/_encode_context_bounded_transcript。
   精确 tokenizer 预算 24000；保留原对话、当前图、合法动作、当前有效证据和候选建议；
   压缩重复历史反馈与 Director 不执行的完整 Tool schema。Executor schema、原始
   Tool receipts 和 trajectory 不改。保留原 2048 reasoning + 4096 action 预算。
3. **列表编号误拦截**：基于同一提交的 scope admission 词法豁免，
   薄适配支持实际出现的 1) … 2) …，不只支持 (1) … (2)。
   只豁免完整递增操作编号；实际剂量/年份等仍须来源支持，原 contract 不重写。
4. **候选 Skill**：继续相同 v248 三条候选，仍可拒绝、非 ACTIVE、不训练；
   本轮未按五题参考答案改写 Skill。修复的是工具访问和执行可选动作，不是医学模板。
5. **仍有边界**：这些改动不能保证模型识别正确研究或充分利用证据。
   本轮也没有加入基于 rubric 的终局检查，不能把合法 FINISH 等同于答案正确。

上述改动遵循 MD §§2–3 的自由 AgentGraph、§10.3 的可拒绝 prompt/repair prior、
§11 的训练与候选信号隔离；保留 SkillFlow BoundedAgent 的逐步 Action–Observation
和 FlowSteer InteractiveWorkflowEnv._step_internal 的增量 Canvas 执行。
不做新训练、GRPO、LoRA、MACE、Bayesian 或 Skill evolution。

## 冻结配置与复现
分支：feature/healthbench-question-corpus-dev5-20260907。
基点：9c5222366adf2bdace74601cf3e15fa28d1b287d。
配置：config/evaluation_healthbench_question_corpus_dev5.yaml。
condition：healthbench_professional_question_corpus_candidate_dev5。
冻结库：artifacts/healthbench_public525_evidence_corpus_v1/manifest.json。
保留同样五题、模型目录、seed、并发4、900秒单题预算和官方 evaluator。
Director 使用已有 GPU6/8026 本地 Qwen3.5-9B；不启动其他模型服务。

入口：
```bash
python scripts/build_healthbench_question_corpus.py \
  --config config/evaluation_healthbench_professional_v233_all_sources_dev5.yaml \
  --output artifacts/healthbench_public525_evidence_corpus_v1 \
  --receipts /ssd1/iclr/1/.tmp/FlowSteer-healthbench-v233/artifacts/healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context/evaluation/agentgraph_trajectories.jsonl \
  --receipts /ssd1/iclr/1/.tmp/FlowSteer-healthbench-v233-all-sources/artifacts/healthbench_professional_v233_all_sources_candidate_dev5/evaluation/agentgraph_trajectories.jsonl

FLOWSTEER_SUPERVISOR_PORT=8026 FLOWSTEER_ROLLOUT_GPU=6 \
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_question_corpus_dev5.yaml --collection-arm agentgraph
```
已存在的冻结库不覆盖。复现需要机器已有 MedRAG/SkillFlow/模型/官方数据及列明的
外部 Tool 缓存；代码备份不把原始测试 rubric、密钥或模型权重加入 Git。
