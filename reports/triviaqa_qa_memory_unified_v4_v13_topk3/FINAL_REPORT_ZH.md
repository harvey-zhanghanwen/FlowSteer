# TriviaQA QA-memory v13 正式评测报告

## 结论

本版本已经按冻结的原始 TriviaQA `rc.nocontext` 数据构建本地 QA-memory embedding index，并移除了静态 prefetch 与 Web Search 路径。Qwen3.5-9B Director 只负责 progressive Canvas editing；只有 Canvas 中 `role_family=evidence_retriever`、`execution_mode=react` 的 worker Agent 可以调用 `triviaqa.qa_memory`。worker 执行 `search(top_k=3)` 和按 rank 的 `read(memory_id)`，再把完整 top-k evidence artifact 通过显式 AgentGraph relation 发送给下游 Reasoner。

正式 128 条 held-out validation 已完整收束，但该条件没有获得正确的 terminal answer：AgentGraph 严格 EM 为 **0.00%**，严格 token F1 为 **0.00%**；同一固定样本、同一 `triviaqa.official.answer.v1` evaluator 下，Qwen3.5-9B Direct baseline 为 **35.16% EM / 40.82% F1**。这说明 Tool ownership 与数据隔离已经成立，但当前 512 条互斥训练 QA-memory 对 held-out 问题的 evidence coverage 不足，且 progressive Canvas execution 仍有 relation routing、answer-slot binding 和 terminal convergence 问题。本结果不能作为准确率提升版本。

## 数据库构建

- 原始数据：TriviaQA `rc.nocontext`，本地冻结源 `/home/test/datasets/TriviaQA_HF/rc.nocontext`。
- 固定切分：原始前 128 条作为 held-out validation；随后 `[128:640]` 的 512 条作为 train QA-memory。
- 隔离结果：512 条记录、512 个唯一 `source_train_task_id`、0 条循环补齐、train/validation `base_task_id` overlap 为 0，`validation_content_indexed=false`。
- 文档内容：`paraphrase_question` 与 `paraphrase_answer_statement`；专名、数字、日期等 canonical span 保留。没有写入 validation question、validation answer、accepted aliases、supporting facts 或 evaluator receipt。
- 同义改写：512/512 通过 `triviaqa.qa_memory.semantic_admission.v13`；从前一 checkpoint 保留 493 条通过项，重新生成并修复 19 条语义漂移项。record-level generation provenance 仍诚实记录为 paraphrase/prompt v12；v13 表示 semantic-admission、index 与 evaluation release。
- embedding：BGE `bge-base-en-v1.5`，768 维，L2 normalization，normalized dot product。
- top-k：只用 train self-retrieval coverage 选择；K=1 为 511/512，K=3 与 K=5 均为 512/512，因此冻结最小最优 K=3。validation 未参与选择，也未被读取。

## AgentGraph 与 Tool 边界

- Director：本地 Qwen3.5-9B；`allowed_tools=[]`；实测 Tool call 为 0；canonical Director request 中 QA-memory provenance-bearing payload exposure 为 0。
- Retriever worker：仅 worker 可使用 `triviaqa.qa_memory`；正式运行共 1,856 次 Tool call，ownership violation 为 0。
- Reasoner/Verifier/Formatter：未分配 QA-memory Tool；Reasoner 只消费显式 relation 上游 artifact。
- 静态 prefetch：关闭。
- Web Search：未配置、未调用；配置中的 `passage_source=external_corpus` 是现有 runtime 对本地独立索引的枚举值，不表示 HTTP 或 Web Search。
- ordered top-k artifact：452/455 个 projection 完全满足 1 次 search + 3 次有序 read；3 个 projection 有额外 ReAct Tool action，因而未通过严格 exact-batch 断言。
- relation routing：125 个发生检索的任务中，124 个把 QA-memory receipt artifact 通过 `upstream` 或双向 exchange 的 `peer_draft` 沿显式 relation 发送给下一 Agent；唯一真实未完成路由的是 `triviaqa:tc_26`（`max_rounds`，没有 Output Agent）。119 个任务满足“所有历史 receipt 均进入 Output inbox”的强断言；该断言还把后续已被替换的旧 batch 计为缺失，因此不等同于 6 个 runtime 路由 bug。

## 正式准确率

| 条件 | 样本数 | evaluator 有效 | EM | token F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 128 | 128 | 35.16% | 40.82% |
| QA-memory AgentGraph v13 | 128 | 128 | 0.00% | 0.00% |
| AgentGraph − Direct | 128 | 128 | -35.16 pp | -40.82 pp |

AgentGraph terminal failure 为 128/128，其中 `canvas_action_domain_exhausted` 121 条、`max_rounds` 7 条；collection failure 与 evaluator operational failure 均为 0。运行报告的任务级 failure type 为：`knowledge_base_coverage_failure` 33、`relation_or_answer_slot_binding_failure` 87、`structured_output_or_format_failure` 8。

## 验证

- 定向单元与端到端测试：205 passed，39 subtests passed。
- 关键断言：`director_tool_calls=0`、`director_requests_toolless=true`、`director_data_plane_isolated=true`、`retrieval_tool_calls_by_worker>0`、`worker_ownership_violation_count=0`、`reasoner_qamemory_tool_unassigned=true`。
- 失败断言按真实结果保留：`native_top_k_batches_complete=false`、`retrieval_artifact_routed_via_relation=false`（124/125，唯一真实失败为 `tc_26`）、`output_inbox_receipt_lineage=false`（119/125 的 all-historical 强断言）。
- 本轮为 inference-only：没有 GRPO、LoRA、backward、optimizer update、MACE 或 Skill 训练。

## 证据文件

- 数据 materialization：`data/triviaqa_qa_memory_v13/materialization_manifest.json`
- top-k 选择：`data/triviaqa_qa_memory_v13/top_k_selection.json`
- index manifest：`data/triviaqa_qa_memory_v13/index/manifest.json`
- 运行配置：`config/evaluation_triviaqa_qa_memory_unified_v4_v13.yaml`
- 正式 run manifest：`artifacts/triviaqa_qa_memory_unified_v4_v13_topk3/run_manifest.json`
- 指标报告：`reports/triviaqa_qa_memory_unified_v4_v13_topk3/report.md`
- 完整离线链路分析与 3 个真实 Wrong Demo：`reports/triviaqa_qa_memory_unified_v4_v13_topk3/formal_result_analysis.md`

完整 trajectory 与 evidence artifacts 保留在本地运行目录；由于体积较大，它们不作为代码备份的一部分推送到 GitHub。GitHub 备份包含可重建数据库/index 的源码、配置、冻结 manifest、最小正式结果与本报告。
