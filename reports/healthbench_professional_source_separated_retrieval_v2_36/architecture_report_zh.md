# HealthBench Professional v2.36：按来源分类的检索索引

状态：独立候选版本，基于已备份的 v2.35（a3dd5e6）。代码与离线接线验证完成，
固定公开 525 题已完成 prepare-only。**尚无本版正式评分，未替换正在运行或排队的评测。**

## 数据从哪里来

| 数据库 | 内容与来源 | 用途与边界 |
| --- | --- | --- |
| `conversation` | 当前样本原始健康对话，保留 speaker、轮次和完整原文 | 回查患者已提供的信息、问题与限制；不是经过证实的外部医学证据。 |
| `medical_references` | 当前节点通过现有 MedRAG、PubMed 搜索及原文读取获得的片段；或沿实际图关系收到的同类 Tool receipt | 检索医学资料与论文摘要，保留文献 ID、URL、日期、原始 excerpt 和分页。PubMed 摘要不能称为论文全文。 |
| `drug_labels` | 当前节点或实际通信上游取得的 NLM DailyMed SPL 标签 | 药品资料独立查询，保留 SETID、版本和日期；不等于完整的药物相互作用检查器。 |

外部库在尚未取得资料时明确为空，不能用模型编造的资料填充。既有的
MedRAG/textbooks 125,847 个片段与 BM25 检索保持不变，没有下载另一套完整语料库。
查询工具为 `healthbench-knowledge.search(database, query)`，底层直接复用
SkillFlow `DocumentPassage`、`build_retrieval_index`、`RetrievalIndex.search/read`
和项目已有的线程亲和 SQLite FTS5 生命周期。

真实外部 Tool 成功返回后自动追加公开证据记录，不增加一次模型/API 请求。
查询时按需建立不可变版本索引；正文、来源、版本和分页均参与记录去重，
后续新页或新版本不会覆盖旧证据。数据库查询也计入原来的每次 Agent 调用
最多 3 次工具调用预算；同一数据库内容未变化时，重复相同查询被拒绝。
其他数据库新增条目不会使无变化的查询重新合法。

## 信息如何传给其他 Agent

每次 Agent invocation 使用独立目录，不读取其他任务或无关系节点的数据库。
可索引的内容只包括原始对话、本节点 continuation 保留的 Tool receipt、
现有单向/双向关系实际传来的 Tool receipt 及其公开 provenance。
Agent 的自由文本总结不是外部文献，不据此生成“医学事实”。

完整公开检索记录追加保存到 `evidence_indexes/request-*/<database>/records.jsonl`；
原始 Tool receipt 继续进入 trajectory。工具结果返回索引状态、数量和路径，
不把完整对话反复复制进每次 Observation。下游和 Director 的消息仍遵守原有
上下文预算，不能把数据库保留完整记录描述为模型上下文无损、无限传递。

ReAct 的每次 Tool Action–Observation 仍由既有执行器处理；Director 在一次
Canvas 功能单元执行结束后接收有序反馈，**没有新增每次 Tool 调用中途暂停并
重新调用 Director 的调度机制**。本次不改自由角色、图关系、唯一 Output、
增量执行顺序、Director 提示词或终局规则，也不预置医疗工作流。

## 已验证与待验证

- 数据库模块：10 项测试及 4 个子项通过，使用真实 SkillFlow FTS5、合成文档。
- 新工具接线、并发隔离、实际关系信息传递、原有 ReAct 与评测协议：81 项测试、4 个子项通过。
- 索引失败时仍保留已成功获取的原始 Tool 结果，不把数据库故障变成证据丢失。
- 525 题 prepare-only 通过，未启动模型、外部检索或 grader。
- 模型能力沿用 v2.35 已保存的兼容记录，不重复付费 canary。Qwen3.5-Flash
  仍为 reasoning-only；本地 Qwen3.5-9B、DeepSeek-V4-Flash、MiniMax-M3 保留
  ReAct 能力声明，并增列本地查询工具。新数据库查询在真实模型上的使用效果尚未验证。
- 所有模型保留既有 thinking 配置；Director 固定本地 Qwen3.5-9B。

没有训练、GRPO、backward、optimizer、LoRA、MACE、Bayesian 更新或 Skill evolution。
rubric、参考回答和 evaluator 状态不会传给数据库适配层。

## 恢复与评测边界

配置：`config/evaluation_healthbench_professional_source_separated_retrieval_v2_36.yaml`。
模型目录：`config/model_catalog_healthbench_professional_source_separated_thinking_v9.yaml`。
索引实现与接线：`src/interactive/healthbench_knowledge_store.py`、`healthbench_knowledge_tools.py`。
来源与适配记录：`docs/source_map.md`、`docs/adaptation_log.md`。

恢复时使用现有 SkillFlow Python 环境、公开 HealthBench Professional 数据路径、
既有 MedRAG 资源和 provider 环境配置；不从 Git 读取任何凭据。最小无 API 检查：

```bash
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_source_separated_retrieval_v2_36.yaml \
  --prepare-only
```

正式运行时 Direct 与 AgentGraph 使用同一批 525 题、同一套六项工具和官方
rubric evaluator；不可把旧工具条件的 Direct 当作匹配对照。保留原样本顺序、
seed、并发 4、生成参数和任务预算。新增目录与旧结果隔离，未自动替换 v2.35 队列。
模型池异构与多 Agent 预算差异仍存在，Direct 对照不是仅改变 topology 的因果消融。
该公开测试集合已用于开发，因此后续分数属于开发后重评测，不是 untouched held-out。

当前不能宣称数据库提高了官方评分，需等独立条件实际评测后报告。
