# HealthBench：语义检索与候选 Skill 精简

## 本轮边界

继续基于全部 525 条公开对话的外部证据库，增加语义检索，然后换另外五题评测。
不使用 rubric、参考回答或最终答案建索引；不做训练、GRPO、LoRA、MACE、
Bayesian 更新或 Skill 自动发布。Director、自由 AgentGraph、ReAct 执行方式、
既有工具预算和官方 evaluator 不变。本次不是 525 题答案生成或全量评分。

本轮向量化的是已经保存的 4422 条外部证据，而不是给全部 125847 条 MedRAG
片段建立向量索引，也没有声称新增了文献来源。先前词法召回漏掉的资料可能仍然
不在此库中。因此该改动改善的是已有资料的检索方式，不能预先宣称证据质量或
官方评分已经提高。全部 525 个公开问题会分别留下语义检索 receipt；命中率不是
答案正确率。历史对话和 Agent 回答不会变成共享证据。

## 来源和必要适配

| 部分 | 具体来源与修改边界 |
| --- | --- |
| 模型 | 直接沿用 SkillFlow `training/environment.py::_get_embed_model` 的本地 BGE-base-en-v1.5 和 CPU 配置；本机已有权重，无下载。 |
| 向量相似度 | `environment.py::_embed_score` 与 `training/tools.py::_search_context` 的 normalized embeddings / dot product；不是重写嵌入模型。 |
| BM25 | 直接移植 `_extract_query_terms`、`_bm25_score`，保留 k1=1.5、b=0.75。避免导入包含 task gold / reward / 训练依赖的整个 Environment。 |
| 混合评分 | 沿用 `_search_passages`：0.4 × 归一化 BM25 + 0.6 × dense cosine，阈值 0.15；没有自行调权重追逐五题分数。 |
| 索引与长文适配 | 必要适配：按本地编码器 512-token 限制，以最多 480 tokens、重叠 64 tokens 切片，保存字符位置；查询和文献均避免静默丢失尾部。聚合取该来源各窗口的最大相似度，原始 excerpt 不重写。 |
| Runtime | 原 `HealthBenchKnowledgeStore.search` 外部来源分支接入向量评分；conversation 仍用原 SkillFlow FTS5。原 ReAct Tool Action–Observation、source receipt、图上通信和 Canvas 调度不变。新收到的外部 Tool 证据可即时编码；不能返回未进入当前 request 的来源。 |
| 候选 Skill | 直接复用 `healthbench_candidate_skill_profile.py` 和 collector 的可拒绝 prompt-prior 接口，不改 ACTIVE gate，不增加模型调用做 Skill 生成。 |

BGE 查询指令和文献不加指令的用法依据
[模型作者说明](https://huggingface.co/BAAI/bge-base-en-v1.5)。该模型主要用于英语，
不能据此保证跨语言或医学实体消歧；相似度不是事实验证或证据等级。

## 旧候选停用与替代

新配置不再加载 `healthbench_candidate_skills_v248.yaml`。其中三条旧 condition ID
全部退出本轮注入，原文件保留供历史复现，并非删除运行证据。
旧“Keep ReAct”、反复 refinement / repair 与过多交接建议没有经配对实验独立验证；
旧五题出现了低分和两个 900 秒采集超时，但不能把失败单独归因到某条 Skill。

新三条候选更短，分别是：

1. 来源匹配：保持实体和关系，用语义检索找到候选后区分相关来源与目标来源。
2. 证据交接与错误修复：保留成功证据，只修真实反馈指出的错误，不重复无变化操作。
3. 及时完成：基于当前证据补具体缺口，不抑制重要的证据支持提醒，不强制额外 Agent
   或检索；预算不足时保留已支持内容并表达不确定性。

ReAct 仍是 execution mode，但不等于每个节点必须检索。没有预设医疗角色、固定
Reasoner→Verifier→Formatter 顺序或增加 Agent 数量的奖励。三条仍是候选，不能
称为“已证实高分 Skill”；本轮也不能分离语义检索与 Skill 的净增益。

## 固定另外五题

从官方原顺序中排除上一轮五题，用 `random.Random(20260907).sample(..., 5)`
选择另外五题，选择时只读 task_id，不读问题、rubric 或分数。
ID 与排除列表见 [选择记录](healthbench_semantic_new5_selection.json)。
“另五题”不代表过去所有实验从未接触；也不与旧五题均分直接做配对提升比较。
与旧条件保持同一模型池、seed、并发 4、900 秒/题和官方 grader；没有新 Direct。

配置：`config/evaluation_healthbench_semantic_skills_new5.yaml`。
候选：`config/healthbench_candidate_skills_semantic_v1.yaml`。
向量索引：`artifacts/healthbench_public525_semantic_v1/manifest.json`。
输出：`artifacts/healthbench_professional_semantic_skills_new5/evaluation`。
分支：`feature/healthbench-semantic-skills-dev5-20260907`；基点 `db24b14`。

## 验证与复现

36 项定向测试及 4 个子测试通过：同义表述召回、输入隔离、原始来源保留、
长文尾部覆盖、新 Tool 证据、真实 ReAct 接口、旧配置不变、旧 Skill 停用与新五题。
其中模型可控的测试使用合成数据，不是 HealthBench 准确率；正式五题另行记录。

实际索引已发布：4422 个来源记录、4433 个窗口、768 维；全程 CPU。
三次使用真实本地 BGE 的检索检查用时分别为 0.232、0.225、0.221 秒
（模型载入完成后测量，不含模型冷启动）。查询包括 heart attack diagnosis、
medication interactions in older adults、研究证据与建议的匹配问题。
第一项实际返回 Harrison 和 First Aid 教材片段；第二项第三名仍返回与药物相互
作用无直接对应的老年孤独症干预试验，因此检索相关性并未被“语义”自动解决。
这些是接口和检索行为检查，不是临床核验、召回率评估或答案准确率。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
python scripts/build_healthbench_semantic_index.py \
  --source-manifest /ssd1/iclr/1/.tmp/FlowSteer-healthbench-question-corpus/artifacts/healthbench_public525_evidence_corpus_v1/manifest.json \
  --output artifacts/healthbench_public525_semantic_v1 \
  --model-path /ssd1/iclr/.private/skillflow-resources/bge-base-en-v1.5 \
  --config config/evaluation_healthbench_semantic_skills_new5.yaml

FLOWSTEER_SUPERVISOR_PORT=8026 FLOWSTEER_ROLLOUT_GPU=6 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_semantic_skills_new5.yaml --collection-arm agentgraph
```

运行需要已有模型、官方数据、来源库、服务和凭据环境；不把凭据或原始 rubric
放入代码备份。已存在的索引目录不覆盖，复现应使用新输出目录。
