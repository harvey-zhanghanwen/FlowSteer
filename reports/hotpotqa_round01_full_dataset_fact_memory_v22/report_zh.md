# HotpotQA 全量陈述性事实记忆库：128 条运行报告

## 1. 结论

本轮在固定的 128 条 HotpotQA 样本上完成了真实推理与官方兼容答案评估。AgentGraph 得到 **EM 79.6875%（102/128）**、**F1 89.5533%**；128 条轨迹均可评估，其中 **127 条显式执行 `FINISH`，1 条因 `max_rounds` 终止**。

本结果严格标记为 **`in_database_transductive`**：按照用户要求，HotpotQA 原始 `train` 与 `validation` 的全部 97,852 条来源记录都被转换为陈述性事实并进入索引，因此当前 128 条评测样本的来源事实也在库内。它使用同一官方兼容 EM/F1 evaluator，但**不是 held-out 检索结果，也不能替代官方 held-out 指标**。

| 条件 | 样本数 | EM | F1 | 显式 `FINISH` |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct 本地基线 | 128 | 72.6563% | 81.7521% | N/A |
| Round-01 AgentGraph（同 evaluator 离线复算） | 128 | 75.0000% | 83.9453% | 128 |
| 本轮 AgentGraph + 动态事实检索 | 128 | **79.6875%** | **89.5533%** | 127 |

相对 Direct 提升 **+7.0313 EM、+7.8012 F1 个百分点**；相对 Round-01 同 evaluator 复算提升 **+4.6875 EM、+5.6080 F1 个百分点**。历史 Round-01 报告中的 F1 84.4401% 来自旧报告口径；本表使用同一 `hotpotqa.official.answer.v1` evaluator 对保存输出重新计算，以保证可比性。

## 2. 数据库约束与构建结果

- 来源：HotpotQA `distractor/train` 90,447 条 + `distractor/validation` 7,405 条，共 97,852 条。
- 所有来源问题均进行了语义保持改写：97,852/97,852，覆盖率 100%，`fallback=0`，与原问题归一化后完全相同的改写为 0。
- 索引记录严格只有 `{memory_id, fact_text}`；`fact_text` 是自包含的陈述性事实，不使用 `Question/Answer` 拼接格式。
- 原始问题、原始标准答案、accepted aliases、supporting-fact label、evaluator receipt 都不参与向量化，也不通过 Tool 暴露。原始 QA 仅保留在数据库外的原始数据集与评测/溯源映射中。
- embedding：`BAAI/bge-base-en-v1.5`，768 维，归一化 cosine similarity，冻结 `top_k=2`。
- 索引中无 Web Search 结果；运行期间也未调用 Web Search。

三个构建例子：

1. `hotpotqa:5a7a06935542990198eaf050`
   - 原始问题（库外溯源）：`Which magazine was started first Arthur's Magazine or First for Women?`
   - 原始答案（库外溯源）：`Arthur's Magazine`
   - 语义保持改写（不参与向量化）：`Which of the two publications, Arthur's Magazine or First for Women, had its initial launch date earlier?`
   - 索引事实：`Arthur's Magazine was the first publication to be started between Arthur's Magazine and First for Women.`
2. `hotpotqa:5a879ab05542996e4f30887e`
   - 原始问题：`The Oberoi family is part of a hotel company that has a head office in what city?`
   - 原始答案：`Delhi`
   - 语义保持改写：`In which city is the headquarters of the hotel company that includes the Oberoi family situated?`
   - 索引事实：`The Oberoi family is part of a hotel company whose head office is located in Delhi.`
3. `hotpotqa:5a8d7341554299441c6b9fe5`
   - 原始问题：`Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?`
   - 原始答案：`President Richard Nixon`
   - 语义保持改写：`Who did Matt Groening name the 'The Simpsons' character Milhouse after, a subject of a song written by musician and satirist Allie Goertz?`
   - 索引事实：`Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who Matt Groening named after President Richard Nixon.`

## 3. AgentGraph 与 Tool 调用边界

本轮保持 progressive Canvas editing，不固定 Reasoner→Verifier→Formatter 串行模板。ReAct 仅是具备检索能力的 worker Agent 的 execution mode。实际链路为：

`Director Canvas action → worker Agent ReAct search/read → evidence artifact → AgentGraph relation → downstream/output Agent → evaluator`

边界断言全部满足：

- `director_tool_calls=0`：控制面的 Qwen3.5-9B Director 请求中 `allowed_tools=[]`，没有检索正文或 top-k payload。
- `retrieval_tool_calls_by_worker=390`：128/128 题均由 worker Agent 动态调用 Tool；`search=130`、`read=260`，成功 390、失败 0。
- `retrieval_artifact_routed_via_relation=true`：127 个正常 `FINISH` 的任务均通过显式 AgentGraph relation 将检索 artifact 传给下游节点。
- 检索到 257 个不同的 `memory_id`；2 题的原始问题 query 在 Tool dispatch 前被拒绝并由 worker 改写，实际发送的 search query 不等于原问题。

worker 的 `read` action 现在只能从当轮 search 返回且尚未读取的候选 `memory_id` 中选择。该约束薄适配 FlowSteer `qa_tool_adapter.py::_condition_qa_action_response_schema` 的 state-conditioned action schema，修复了 worker 反复读取 rank-1、耗尽 ReAct turn budget 的问题。

## 4. 错误分类

严格 EM 错误共 26 条。以下分类按“首个因果失败点”互斥统计：

| 错误类型 | 数量 | 错误样本占比 | 全部样本占比 |
|---|---:|---:|---:|
| 答案表述与 evaluator canonicalization 不匹配 | 19 | 73.08% | 14.84% |
| worker 证据充分性/证据选择错误 | 3 | 11.54% | 2.34% |
| semantic binding / contract scope 错误 | 3 | 11.54% | 2.34% |
| Director `max_rounds` / terminal semantics | 1 | 3.85% | 0.78% |
| top-k 检索完全缺失且为唯一首因 | 0 | 0% | 0% |

26 条错误中，worker artifact 为 `supported` 的有 16 条、`unsupported` 的有 10 条；其中约 9 条 `unsupported` 的 top-k 实际已经包含答案事实，说明主要剩余检索问题是 evidence sufficiency 判定，而不是 embedding top-k 召回。

## 5. 典型错误 Demo

### A. 答案 canonical span 被缩短

- Task ID：`hotpotqa:5a8d7341554299441c6b9fe5`
- 参考答案：`President Richard Nixon`
- 最终输出：`Richard Nixon`
- 指标：EM 0，F1 0.8
- 链路：`ADD_AGENT(retrieval worker) → ADD_AGENT(answer worker) → SET_RELATION → SET_OUTPUT → FINISH`
- Tool：rank-1 `hotpotqa-fact-000002` 明确包含 `President Richard Nixon`；worker 正确标记 `supported`，artifact 经 relation 到达输出节点。
- 首个失败点：输出节点将有证据支持的 canonical span 缩短为 `Richard Nixon`。检索、Agent communication 和 terminal 均正常。

### B. worker 对已检索事实作出 false-negative sufficiency 判断

- Task ID：`hotpotqa:5ab93287554299753720f78f`
- 问题：Kniphofia 与 Baptisia 共有的 vegetation type 是什么？
- 参考答案：`plant`
- 最终输出：`None`
- 指标：EM 0，F1 0
- Tool：原始 query 被拒绝，worker 改写为 `Kniphofia Baptisia common vegetation type`；rank-1 `hotpotqa-fact-000121` 已明确说明二者共有 `plant` 类型，`read` 成功。
- 首个失败点：worker 仍输出 `unsupported/null`；该 receipt 经 relation 传递后，输出节点回答 `None`。

### C. contract scope 被错误改写

- Task ID：`hotpotqa:5ab698885542995eadef002a`
- 参考答案：`Nevada`
- 最终输出：`N/A`
- 指标：EM 0，F1 0
- Tool：rank-2 `hotpotqa-fact-000019` 明确支持 Catherine Cortez Masto 与 Nevada 的关系。
- 首个失败点：Director 将原问题中的 `Attorney General` 改成 `Solicitor General`，造成 semantic scope mutation；worker 按错误 contract 标记 `unsupported`。输出节点在收到该 artifact 前曾生成 `Nevada`，接收错误 contract 的 receipt 后改为 `N/A`，体现了错误通过 Agent communication 继续传播。

### D. 正确答案已生成但没有执行 FINISH

- Task ID：`hotpotqa:5add85b65542997545bbbd61`
- 参考答案：`Tantallon Castle`
- 最终输出：`null`
- terminal：`max_rounds`，`explicit_finish=false`
- Tool：worker rank-1 已检索到支持 `Tantallon Castle` 的 `hotpotqa-fact-000113`。
- 链路：输出节点在 turn 9 已产生 `<answer>Tantallon Castle</answer>`；Director 随后继续执行冗余/no-op relation edit，直到最后一轮才补足 output reachability，已经没有剩余 turn 执行 `FINISH`。
- 首个失败点：Director 的 terminal control，而不是答案生成能力或检索召回。

## 6. 当前结论与后续优先级

本轮表明动态事实检索相对同样本 Direct 和 Round-01 AgentGraph 都有真实增益；检索调用边界、Tool receipt ownership、relation 路由和无 Web Search 约束均已验证。剩余主要问题依次是：

1. 在不泄露 evaluator metadata 的前提下保持 evidence-supported canonical span，避免不必要的删词、后缀丢失和全名/简称漂移；
2. 改进 worker 的 evidence sufficiency 判定，使已命中的答案事实不会被错误标记为 `unsupported`；
3. 强化 contract 的 semantic fidelity，禁止 Director 改写原问题的实体、属性、角色或比较范围；
4. 当 output artifact 已存在且 reachability 已满足时优先 `FINISH`，避免冗余 Canvas edit 消耗 `max_rounds`。

本轮是 inference-only 架构验证：未执行 GRPO、LoRA、MACE、Skill training、backward、optimizer step 或权重更新。

## 7. 证据文件

- `reports/hotpotqa_round01_full_dataset_fact_memory_v22/report.json`
- `reports/hotpotqa_round01_full_dataset_fact_memory_v22/error_demos.jsonl`
- `artifacts/hotpotqa_round01_full_dataset_fact_memory_v22/agentgraph_trajectories.jsonl`
- `artifacts/hotpotqa_round01_full_dataset_fact_memory_v22/paired_results.jsonl`
- `artifacts/hotpotqa_round01_full_dataset_fact_memory_v22/run_manifest.json`
- `artifacts/hotpotqa_round01_full_dataset_fact_memory_v22/fact_memory_index_smoke_receipt.json`
- `data/hotpotqa_full_dataset_fact_memory_v1/materialization_manifest.json`
- `data/hotpotqa_full_dataset_fact_memory_v1/index_topk2/manifest.json`
