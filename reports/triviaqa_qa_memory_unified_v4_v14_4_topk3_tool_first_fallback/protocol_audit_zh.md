# TriviaQA QA-memory Tool-first 协议核对

## 正式评测结果

- 固定 held-out validation：128 条；同一 `triviaqa.official.answer.v1` evaluator。
- Qwen3.5-9B Direct：EM 35.16%，token F1 40.82%。
- AgentGraph：EM 27.34%，token F1 33.62%。
- AgentGraph 相对 Direct：EM -7.81 个百分点，token F1 -7.20 个百分点。
- 103/128 显式 `FINISH`；25/128 terminal failure，其中 14 条为 `canvas_action_domain_exhausted`，11 条为 `max_rounds`；operational/evaluator failure 为 0。
- 本轮仅做 inference；未运行 GRPO、LoRA、backward、optimizer update、MACE、Bayesian update 或 Skill 注入。

## QA-memory 数据合同

- 主索引由冻结的 TriviaQA train slice 构建：512 条 memory、512 个唯一训练源、0 条 cycle。
- 每条 memory 保留一个不可拆分的 Q-A 语义记录：`paraphrase_question`、`paraphrase_answer_statement`、`canonical_answer` 和 `source_train_task_id`。
- embedding 文本模板同时编码问题和答案：`Question: {paraphrase_question}\nAnswer: {paraphrase_answer_statement}`。
- 512/512 canonical span 保留在对应 answer statement 中；512/512 与冻结训练源的答案及 source binding 一致。
- embedding 为 BGE-base-en-v1.5、768 维、L2 normalization、dot-product similarity；train-only 选择后冻结 top-k=3。
- 128 条 validation 与 512 条 train 的 `base_task_id` 交叉为 0；validation question 精确重合和 validation Q-A pair 精确重合均为 0。validation answer、accepted aliases 和 evaluator metadata 未进入索引 payload。

## Tool-first 与 fallback 执行核对

- Director/Supervisor Tool 调用为 0；Director request 的 `allowed_tools=[]`，且不接收 QA-memory top-k 正文。
- Web Search 调用为 0；唯一检索资源是 worker Agent 的本地 `triviaqa.qa_memory`。
- 193 次已落盘 Reasoner execution 全部接收到由显式 AgentGraph relation 传来的 QA-memory Tool receipt 或 typed retrieval status；未发现 Reasoner 在检索状态之前直接回答。
- 193 个已完成 top-k batch 全部执行 `search -> read(rank 1) -> read(rank 2) -> read(rank 3)`，incomplete batch 为 0。
- 正常 `FINISH` 的 103 条全部先得到 `knowledge_base_coverage_failure`，随后才由 Tool-less child Reasoner 执行 parametric fallback，并由 Verifier 标记 `parametric_fallback`。
- 25 条 terminal failure 没有越过该门控直接回答。其中特别是 3 条没有形成可接受 retrieval artifact：`triviaqa:tc_81` 与 `triviaqa:tc_115` 已成功 search/read top-k，但 completion artifact 连续违反结构化字段约束；`triviaqa:tc_140` 的 worker query 连续丢失原问题实体 anchor，未产生成功 Tool receipt。三者均以 terminal failure 结束，没有非法启用 fallback。

## 低分的直接原因

1. 512 条训练 QA-memory 对开放域 TriviaQA 的 128 条 held-out 问题覆盖不足。最终 live lineage 中 117 条被判定为 `knowledge_base_coverage_failure`；只有 4 条产生 `evidence_found`，而这 4 条都未完成合法 terminal path。
2. 因此，103 条正常完成的 AgentGraph 任务全部退化为 Qwen3.5-9B 的 parametric fallback；本地向量库没有为任何正常完成任务提供可计分的事实增益。
3. AgentGraph 还引入 25 条 terminal failure，以及 reasoning、answer-slot binding、structured output 等额外失败面，所以最终 EM/F1 低于 Direct。
4. 同义改写只改变表述，不增加训练库没有的事实。dense retrieval 可以找语义相近的问答，但当实体不一致时，严格 entity/relation gate 必须拒绝，不能把相似但错误实体的答案当作 evidence。

## 结论

v14.4 已实现用户要求的执行顺序：worker local Tool retrieval 在先，完整 top-k 经 AgentGraph relation 传递；仅在 typed coverage failure 后允许 child Reasoner parametric fallback。当前瓶颈是 train QA-memory 的事实覆盖率与 terminal reliability，不是 Q-A 分离、Director 代查或 Web Search。
