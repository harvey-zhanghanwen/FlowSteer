# HotpotQA 架构报告

## Stable Zero

- 能力边界：闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
- Raw receipts：manifest=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/run_manifest.json`; paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl`
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- optimizer update：**0**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | exact_match | token_f1 |
|---|---:|---:|
| Direct/Simple Baseline | 100.00% | 100.00% |
| AgentGraph | 100.00% | 100.00% |

以上是当前 evidence scope 中 2 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Evidence scope 与协议限制

- Evidence scope：exposed development canary；不是 unseen held-out 或 benchmark estimate
- Protocol：Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。

- 当前 2 题曾进入多轮架构诊断；2/2 只记录任务行为与 Stable Zero 链完整性，不能报告为 100% benchmark accuracy。
- HotpotQA distractor protocol 的给定 passages 正常包含回答所需事实；这与 Ground Truth 字段进入模型 prompt 不同。

### 明确排除的历史结果

- v1 exact-schema 前的自然策略 canary 仅保留为历史结果；v2 因 Director schema/tool_id 边界含混导致 1/2 max_rounds，仅作为 v3 修复前失败诊断。


## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | exact_match=100.00%; token_f1=100.00% | Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。 |
| AgentGraph Stable Zero | exact_match=100.00%; token_f1=100.00% | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | exact_match=100.00%; token_f1=100.00% | current v3 condition after exact Director field/resource-ID clarification |
| Tool/ReAct/Coding-enabled AgentGraph | exact_match=100.00%; token_f1=100.00% | declared capability; actual Tool receipts=1 |

同一 receipt 同时代表多个 stage 时不会重复解释为独立实验，也不会把 protocol-separated 条件的差值解释为因果增益。

## Workflow 分布

互斥 `topology_family` 计数（未观察到的合法结构显式记 0）：

- `single`: 0
- `serial_2`: 1
- `serial_3_plus`: 1
- `parallel`: 0
- `fan_in`: 0
- `fan_out`: 0
- `reciprocal`: 0
- `verification`: 0
- `mixed`: 0
- `other`: 0
- `unknown`: 0

可重叠的执行/协作 motif（来自最终图与实际 execution receipt）：

- `parallel execution`: 0/2 tasks
- `fan-in`: 0/2 tasks
- `fan-out`: 0/2 tasks
- `reciprocal`: 0/2 tasks
- `verification`: 0/2 tasks
- `ReAct`: 2/2 tasks
- `Tool-using`: 1/2 tasks
- `Coding`: 0/2 tasks
- `mixed execution modes`: 2/2 tasks

- 平均 structural depth：**2.50**
- 平均 effective dependency depth：**2.50**
- 平均 Agent 数：**2.50**
- 平均 relation 数：**1.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

- `gpt-4o-mini`: 5 Agent nodes

- Model family：Qwen=0, DeepSeek=0, Gemini=0, GPT=5, MiniMax=0, Grok=0, GLM=0, Other=0
- Multi-model workflow 比例：**0/2**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| qwen3.5-9b-local | 2 | 2 | 2982 | 0.32 |

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| gpt-4o-mini | 7 | 7 | 13783 | 14.86 |

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`qwen35-9b-hotpot-step-000000`；prompt=`agentgraph.director.minimal.v3`
- Director calls/attempts：**5/5**；tokens=**19948**；latency=**6.79s**



## Tool / ReAct 使用情况

- Tool call：**1**；成功：**1**；失败：**0**
- Tool call task rate：**1/2**
- Tool useful rate：**不可测**；当前 receipt 没有独立的 causal usefulness annotation
- Tool wasted rate：**不可测**；Tool error 单独报告，不能等同于无效信息价值
- Environment transition receipt：**0**
- Coding action receipt：**0**
- AgentGraph 原生 environment action：**0**；invalid action：**0**
- Direct 原生 environment action：**0**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 2 | 2982 | 0.32 |
| AgentGraph | 12 | 33731 | 21.65 |


## Exact-schema Tool forced probe（不计入 benchmark）

- Receipt：`artifacts/tool_exact_schema_canary/hotpotqa_exact_wire_v7_20260820.json`
- Controls：`diagnostic_only=true`、`forced_probe=true`、`grpo_eligible=false`、`skill_evidence_eligible=false`
- Overall status：`failed`
- StructuredAction schema compliance：`true`
- Tool backend compliance：`true`；successful receipts=`4`
- Model action/termination compliance：`false`
- Observed action sequence：`search → read → read → read → read → read`

该 receipt 只回答 exact `StructuredAction`、真实 backend dispatch 和有界 ReAct termination 是否可执行；不含 evaluator、Ground Truth、benchmark metric、Skill evidence 或训练数据。forced probe 失败不覆盖同条件自然策略成绩，反之亦然。

## Correct Demo

### Correct Demo: `hotpotqa:5a7a06935542990198eaf050`

- Raw receipts：paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl`

- Task：Based on the following passages, answer the question. [[Radio City (Indian radio station)] Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The R…
- Ground Truth：Arthur's Magazine
- Final Answer：<answer>Arthur's Magazine</answer>
- Evaluator：`hotpotqa.official.answer.v1`; metrics=`{"exact_match": 1.0, "token_f1": 1.0}`
- Trajectory ID：`trajectory_9da342b94a0ec5d46fa3f1a3`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v3`; tool=`skillflow.qa-retrieval.search-read.exact-structured-action.v2`
- Output Agent：`answerer`
- AgentGraph: `reader → answerer`

Agent 配置：

- `answerer` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Compare the start dates of Arthur's Magazine and First for Women and output the magazine that started first
- `reader` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`reader`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read the passages containing 'Arthur's Magazine' and 'First for Women' to find their launch/start dates

Director atomic edit 序列：`add_subgraph → add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`1`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['reader'], ['answerer']]`; executed=`['answerer', 'reader']`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_result={"output_agent_id":"answerer","output":"<answer>Arthur's Magazine</answer>","executed_agent_ids":["answerer","reader"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["reader"],"sink_agent_ids":["answerer"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"answerer"},"output_inbox":[{"source_agent_id":"reader","target_agent_id":…
- round=`2`; action=`finish`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['reader'], ['answerer']]`; executed=`['answerer', 'reader']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `reader` → `answerer`; artifact_type=`text`; dependency=Compare the start dates of Arthur's Magazine and First for Women and output the magazine that started first; graph_revision=`4`; environment_revision=`None`; tool_receipts=`0`; body=Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.

Executor ReAct trace（公开 StructuredAction/observation）：

- agent=`reader`; turn=`1`; action={"arguments": {"value": "Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first."}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null

### Correct Demo: `hotpotqa:5a879ab05542996e4f30887e`

- Raw receipts：paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl`

- Task：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.] [[Ishqbaaaz] Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast …
- Ground Truth：Delhi
- Final Answer：<answer>Delhi</answer>
- Evaluator：`hotpotqa.official.answer.v1`; metrics=`{"exact_match": 1.0, "token_f1": 1.0}`
- Trajectory ID：`trajectory_490c70719d4aea00f60f9e46`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v3`; tool=`skillflow.qa-retrieval.search-read.exact-structured-action.v2`
- Output Agent：`format`
- AgentGraph: `reasoner → format; next_hop → reasoner`

Agent 配置：

- `format` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Output the exact city name as the final answer.
- `next_hop` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`retriever`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`passage_id`; contract: Identify the passage containing information about the Oberoi family's hotel company head office based on the question and provided texts.
- `reasoner` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`researcher`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read the retrieved passage to extract the city where the Oberoi Group head office is located.

Director atomic edit 序列：`add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['next_hop'], ['reasoner'], ['format']]`; executed=`['format', 'next_hop', 'reasoner']`; reused=`[]`; feedback=accepted add_subgraph at revision 6; execution_result={"output_agent_id":"format","output":"<answer>Delhi</answer>","executed_agent_ids":["format","next_hop","reasoner"],"reused_agent_ids":[],"topology":{"agent_count":3,"relation_count":2,"directed_edge_count":2,"quotient_directed_edge_count":2,"reciprocal_pair_count":0,"component_count":3,"max_depth":3,"structural_depth":3,"max_width":1,"topology_family":"serial_3_plus","topology_motifs":["serial_3_plus"],"root_agent_ids":["next_hop"],"sink_agent_ids":["format"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"format"},"output_inbox":[{"source_agent_id":"reasoner","target_age…
- round=`1`; action=`finish`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['next_hop'], ['reasoner'], ['format']]`; executed=`['format', 'next_hop', 'reasoner']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `reasoner` → `format`; artifact_type=`text`; dependency=Output the exact city name as the final answer.; graph_revision=`6`; environment_revision=`None`; tool_receipts=`0`; body=Delhi
- `next_hop` → `reasoner`; artifact_type=`passage_id`; dependency=Read the retrieved passage to extract the city where the Oberoi Group head office is located.; graph_revision=`6`; environment_revision=`None`; tool_receipts=`1`; body=Delhi

Executor ReAct trace（公开 StructuredAction/observation）：

- agent=`next_hop`; turn=`1`; action={"arguments": {"limit": 1, "query": "Oberoi Group head office location"}, "kind": "tool", "name": "search", "resource_id": "qa-retrieval.search", "skill_id": null}; observation_status=`success`; observation={"completed": true, "executed_action": {"arguments": {"limit": 1, "query": "Oberoi Group head office location"}, "kind": "tool", "name": "search", "resource_id": "qa-retrieval.search", "skill_id": null}, "observation_status": "success", "result": {"hits": [{"document_id": "sha256:f2ad8e87c027cb0295a2dc2bbe0e203d24096808215ebfd33567d4401af4ec29", "passage_id": "atlas-dpr-wikipedia:000012729919", "rank": 1, "snippet": "The Oberoi Group The Oberoi Group is a hotel group with its head office in Delhi. Founded in 1934, the company owns and/or operates 35 luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands. The foundatio…
- agent=`next_hop`; turn=`2`; action={}; observation_status=`parse_error`; observation=null
- agent=`next_hop`; turn=`3`; action={"arguments": {"value": "Delhi"}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null
- agent=`reasoner`; turn=`1`; action={"arguments": {"value": "Delhi"}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null

Tool receipts：

- receipt=1; tool=`qa-retrieval.search`; version=`sha256:b9a491d7cdaa2a107e8a12d579bfabd37ce496ad92feb9a1a5c69527f9e026e8`; status=`completed`; latency_ms=`3765.840709209442`; request={"action": "search", "arguments": {"limit": 1, "query": "Oberoi Group head office location"}}; result={"completed": true, "value": {"hits": [{"document_id": "sha256:f2ad8e87c027cb0295a2dc2bbe0e203d24096808215ebfd33567d4401af4ec29", "passage_id": "atlas-dpr-wikipedia:000012729919", "rank": 1, "snippet": "The Oberoi Group The Oberoi Group is a hotel group with its head office in Delhi. Founded in 1934, the company owns and/or operates 35 luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands. The foundations of the Oberoi Group date back to 1934 w…", "title": "The Oberoi Group"}], "operation": "search", "passage_ids": ["atlas-dpr-wikipedia:000012729919"], "query": "Oberoi Group head office location", "retrieval_index": …
## Wrong / Failure Demo

当前 AgentGraph 2 题均成功；以下是相同任务或固定条件中保留的真实适配前 failure receipt。该结果只用于 root-cause 对照，不混入当前 Stable Zero 指标。

- Preserved condition：`qa_tool_react_exact_wire_v2_stable_zero (pre-v3 schema clarification)`
- Failure classification：`Director action-schema / Tool resource identifier / terminal control`

### Preserved Wrong Demo: `hotpotqa:5a879ab05542996e4f30887e`

- Raw receipts：paired=`artifacts/qa_tool_react_exact_wire_v2_stable_zero/hotpotqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v2_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl`

- Task：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.] [[Ishqbaaaz] Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast …
- Ground Truth：Delhi
- Final Answer：
- Evaluator：`hotpotqa.official.answer.v1`; metrics=`{"exact_match": 0.0, "token_f1": 0.0}`
- Trajectory ID：`trajectory_c2fd8beb3d15b195d1bcf69e`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v2`; tool=`skillflow.qa-retrieval.search-read.exact-structured-action.v2`
- Output Agent：`answerer`
- AgentGraph: `reader → answerer`

Agent 配置：

- `answerer` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Analyze the retrieved passages to determine the head office city of the Upperoi family's hotel company and output the answer.
- `reader` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`processor`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read passages relevant to the Oberoi family and its hotel company's head office location.

Director atomic edit 序列：`invalid → invalid → invalid → invalid → invalid → add_subgraph → finish → set_output → modify_agent → delete_agent → set_output → invalid → invalid → delete_agent → add_subgraph → add_subgraph → add_subgraph → set_relation → finish → modify_agent`

Progressive Canvas turn receipts：

- round=`0`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: unknown action fields: output_agent_id
- round=`1`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: unknown action fields: output_agent_id
- round=`2`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: unknown action fields: output_agent_id
- round=`3`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: missing action fields: agents
- round=`4`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: unknown action fields: output_agent_id
- round=`5`; action=`add_subgraph`; graph_revision=`3`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 3; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'search_agent' cannot declare allowed_tools"}
- round=`6`; action=`finish`; graph_revision=`3`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=cannot finish: output_agent_count: exactly one output agent is required
- round=`7`; action=`set_output`; graph_revision=`3`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`8`; action=`modify_agent`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted modify_agent at revision 4; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'search_agent' cannot declare allowed_tools"}
- round=`9`; action=`delete_agent`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['answer_agent']]`; executed=`['answer_agent']`; reused=`[]`; feedback=accepted delete_agent at revision 5; execution_result={"output_agent_id":null,"output":null,"executed_agent_ids":["answer_agent"],"reused_agent_ids":[],"topology":{"agent_count":1,"relation_count":0,"directed_edge_count":0,"quotient_directed_edge_count":0,"reciprocal_pair_count":0,"component_count":1,"max_depth":1,"structural_depth":1,"max_width":1,"topology_family":"single","topology_motifs":[],"root_agent_ids":["answer_agent"],"sink_agent_ids":["answer_agent"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":null},"output_inbox":[],"agent_artifacts":[{"agent_id":"answer_agent","model_id":"qwen3.5-flash","role_family":"format"…
- round=`10`; action=`set_output`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Format Agent must consume exactly one upstream semantic-answer artifact; received 0
- round=`11`; action=`invalid`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Expecting ',' delimiter: line 1 column 365 (char 364)
- round=`12`; action=`invalid`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Expecting ':' delimiter: line 1 column 359 (char 358)
- round=`13`; action=`delete_agent`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted delete_agent at revision 6
- round=`14`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`15`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Format Agent must consume exactly one upstream semantic-answer artifact; received 0
- round=`16`; action=`add_subgraph`; graph_revision=`10`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 10; execution_error={"type":"AgentRuntimeError","message":"agent 'reader' references unknown tool 'read'"}
- round=`17`; action=`set_relation`; graph_revision=`10`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: action made no graph change; modify an Agent contract/model or another graph field before expecting a new execution
- round=`18`; action=`finish`; graph_revision=`10`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=cannot finish: execution_error={"type":"AgentRuntimeError","message":"agent 'reader' references unknown tool 'read'"}
- round=`19`; action=`modify_agent`; graph_revision=`11`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['reader'], ['answerer']]`; executed=`['answerer', 'reader']`; reused=`[]`; feedback=accepted modify_agent at revision 11; execution_result={"output_agent_id":"answerer","output":"<answer>Delhi</answer>","executed_agent_ids":["answerer","reader"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["reader"],"sink_agent_ids":["answerer"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"answerer"},"output_inbox":[{"source_agent_id":"reader","target_agent_id":"answerer",…

实际 CommunicationEnvelope：

- `reader` → `answerer`; artifact_type=`text`; dependency=Analyze the retrieved passages to determine the head office city of the Upperoi family's hotel company and output the answer.; graph_revision=`11`; environment_revision=`None`; tool_receipts=`0`; body=Delhi

Executor ReAct trace（公开 StructuredAction/observation）：

- agent=`reader`; turn=`1`; action={"arguments": {"value": "Delhi"}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null

FIRST ERROR：前五个 Director action 把 top-level output_agent_id 放入 Agent object，随后又把 action_name `search`/`read` 当成 allowed_tools 的 resource_id；Canvas 均 fail closed。第 20 轮得到正确输出后已无剩余显式 FINISH turn，因此以 max_rounds 终止。v3 只澄清字段层级和 exact tool_id 边界。

## HotpotQA Tool availability OFF/ON 配对诊断

| Scope | Pair | Trajectory | OFF EM/F1 | ON EM/F1 | ΔEM/ΔF1 | ON Tool invocation |
|---|---:|---:|---:|---:|---:|---:|
| exposed development forced probe | 2 | 4 | 100.00% / 100.00% | 100.00% / 100.00% | +0.00 / +0.00 | 0/2 |

OFF 未暴露 Tool catalog，ON 精确暴露 `qa-retrieval.search/read`，2/2 pair 的 non-treatment observation projection 相同；但 ON 未产生 Tool call 或 ToolReceipt。该结果只属于 Tool availability assignment ITT 诊断，不计入自然策略 Stable Zero、ToolReceipt 总数、Skill evidence、GRPO 或 benchmark accuracy。完整报告见 `reports/hotpotqa_tool_availability_pair_v1/report.md`。
