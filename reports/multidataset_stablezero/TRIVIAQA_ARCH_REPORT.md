# TriviaQA 架构报告

## Stable Zero

- 能力边界：仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
- Raw receipts：manifest=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/run_manifest.json`; paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl`
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- optimizer update：**0**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | exact_match | token_f1 |
|---|---:|---:|
| Direct/Simple Baseline | 50.00% | 50.00% |
| AgentGraph | 100.00% | 100.00% |

以上是当前 evidence scope 中 2 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Evidence scope 与协议限制

- Evidence scope：exposed development canary；不是 unseen held-out 或 benchmark estimate
- Protocol：Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。

- 当前 2 题来自已用于架构开发的 development block；结果不能外推为 TriviaQA benchmark accuracy。

### 明确排除的历史结果

- v1 exact-schema 前的自然策略 canary 只保留为历史结果，不进入当前 v3 指标。


## Receipt-distinct Evaluation Conditions

| Receipt-distinct condition | Result | Protocol note |
|---|---|---|
| Simple Baseline | exact_match=50.00%; token_f1=50.00% | Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。 |
| Current AgentGraph Stable Zero | exact_match=100.00%; token_f1=100.00% | fixed tasks, explicit FINISH, native evaluator |

`architecture-final` 与 Tool/ReAct/Coding-enabled 是当前 AgentGraph condition 的版本/能力属性，不是额外运行的实验条件；因此不再重复列出同一 graph metric。protocol-separated 条件的差值也不解释为因果增益。

## Runtime / Search-space capability 与 Director natural policy adoption

| Layer | Receipt-backed statement |
|---|---|
| Runtime / search-space capability | `仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力` 已在当前配置/Runtime 边界中暴露；只有对应 execution receipt 才能证明某条轨迹实际执行该能力。 |
| Director natural policy adoption | observed topology: serial_2=2; actual ToolReceipt=1 across 1/2 tasks。这是当前 fixed-task trajectory 的观测采用情况，不等同于 Runtime 能力上限。 |

## Workflow 分布

互斥 `topology_family` 计数（未观察到的合法结构显式记 0）：

- `single`: 0
- `serial_2`: 2
- `serial_3_plus`: 0
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
- `ReAct`: 1/2 tasks
- `Tool-using`: 1/2 tasks
- `Coding`: 0/2 tasks
- `mixed execution modes`: 1/2 tasks

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

- `glm-4.5-flash`: 1 Agent nodes
- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-9b-local`: 2 Agent nodes

- Model family：Qwen=2, DeepSeek=0, Gemini=0, GPT=1, MiniMax=0, Grok=0, GLM=1, Other=0
- Multi-model workflow 比例：**1/2**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| qwen3.5-9b-local | 2 | 2 | 478 | 0.32 |

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| glm-4.5-flash | 1 | 1 | 489 | 4.72 |
| gpt-4o-mini | 1 | 1 | 198 | 2.02 |
| qwen3.5-9b-local | 3 | 3 | 4770 | 0.95 |

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`qwen35-9b-hotpot-step-000000`；prompt=`agentgraph.director.minimal.v3`
- Director calls/attempts：**6/6**；tokens=**17364**；latency=**8.65s**



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
| Direct | 2 | 478 | 0.32 |
| AgentGraph | 11 | 22821 | 16.34 |


## Exact-schema Tool forced probe（不计入 benchmark）

- Receipt：`artifacts/tool_exact_schema_canary/triviaqa_exact_wire_v2_20260820.json`
- Controls：`diagnostic_only=true`、`forced_probe=true`、`grpo_eligible=false`、`skill_evidence_eligible=false`
- Overall status：`passed`
- StructuredAction schema compliance：`true`
- Tool backend compliance：`true`；successful receipts=`2`
- Model action/termination compliance：`true`
- Observed action sequence：`search → read → complete`

该 receipt 只回答 exact `StructuredAction`、真实 backend dispatch 和有界 ReAct termination 是否可执行；不含 evaluator、Ground Truth、benchmark metric、Skill evidence 或训练数据。forced probe 失败不覆盖同条件自然策略成绩，反之亦然。

## Correct Demo

### Correct Demo: `triviaqa:tc_1`

- Raw receipts：paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl`

- Task：Which American-born Sinclair won the Nobel Prize for Literature in 1930?
- Ground Truth：Sinclair Lewis | (Harry) Sinclair Lewis | Harry Sinclair Lewis | Lewis, (Harry) Sinclair | Grace Hegger
- Final Answer：<answer>Sinclair Lewis</answer>
- Evaluator：`triviaqa.official.answer.v1`; metrics=`{"exact_match": 1.0, "token_f1": 1.0}`
- Trajectory ID：`trajectory_862dfdf06b8b13b068ee1c71`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v3`; tool=`skillflow.qa-retrieval.search-read.exact-structured-action.v2`
- Output Agent：`output_agent`
- AgentGraph: `investigative_analyst → output_agent`

Agent 配置：

- `investigative_analyst` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`analyst`, allowed_tools=`[]`, artifact_type=`text`; contract: Identify the American-born Sinclair who won the Nobel Prize in Literature in 1930 and draft a concise rationale
- `output_agent` — model=`glm-4.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the final answer from [investigative_analyst]'s text artifact: 'Name. Reason'. Output the exact string 'Name. Reason' as the only content

Director atomic edit 序列：`add_subgraph → modify_agent → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'investigative_analyst' cannot declare allowed_tools"}
- round=`1`; action=`modify_agent`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['investigative_analyst'], ['output_agent']]`; executed=`['investigative_analyst', 'output_agent']`; reused=`[]`; feedback=accepted modify_agent at revision 5; execution_result={"output_agent_id":"output_agent","output":"<answer>Sinclair Lewis</answer>","executed_agent_ids":["investigative_analyst","output_agent"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["investigative_analyst"],"sink_agent_ids":["output_agent"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"output_agent"},"output_inbox":[{"s…
- round=`2`; action=`finish`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['investigative_analyst'], ['output_agent']]`; executed=`['investigative_analyst', 'output_agent']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `investigative_analyst` → `output_agent`; artifact_type=`text`; dependency=Extract the final answer from [investigative_analyst]'s text artifact: 'Name. Reason'. Output the exact string 'Name. Reason' as the only content; graph_revision=`5`; environment_revision=`None`; tool_receipts=`0`; body=The American-born Sinclair who won the Nobel Prize in Literature in 1930 is Sinclair Lewis. He was recognized for his significant contributions to literature, particularly for his critical portrayal of American society and culture.

### Correct Demo: `triviaqa:tc_3`

- Raw receipts：paired=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/protocol_separated_results.jsonl`; trajectory=`artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl`

- Task：Where in England was Dame Judi Dench born?
- Ground Truth：York | Park Grove (1895) | York UA | Yorkish | UN/LOCODE:GBYRK | York, UK | Eoforwic | Park Grove School | York Ham | The weather in York | City of York | York, England | York, Yorkshire | York ham | County Borough of York | YORK | Eoferwic | Park Grove Primary School | York, North Yorkshire | Yoisk | York (England)
- Final Answer：<answer>York</answer>
- Evaluator：`triviaqa.official.answer.v1`; metrics=`{"exact_match": 1.0, "token_f1": 1.0}`
- Trajectory ID：`trajectory_0f43b55c221ed66a1d7eff75`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v3`; tool=`skillflow.qa-retrieval.search-read.exact-structured-action.v2`
- Output Agent：`formatter`
- AgentGraph: `searcher → formatter`

Agent 配置：

- `formatter` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the specific location of Dame Judi Dench's birth from the provided search results and output it as a final textual answer.
- `searcher` — model=`qwen3.5-9b-local`, execution_mode=`react`, role_family=`searcher`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`search_results`; contract: Formulate a search query about Dame Judi Dench's birthplace, execute the search, and provide the retrieved passages.

Director atomic edit 序列：`add_subgraph → add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`1`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['searcher'], ['formatter']]`; executed=`['formatter', 'searcher']`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_result={"output_agent_id":"formatter","output":"<answer>York</answer>","executed_agent_ids":["formatter","searcher"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["searcher"],"sink_agent_ids":["formatter"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"formatter"},"output_inbox":[{"source_agent_id":"searcher","target_agent_id":"fo…
- round=`2`; action=`finish`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['searcher'], ['formatter']]`; executed=`['formatter', 'searcher']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `searcher` → `formatter`; artifact_type=`search_results`; dependency=Extract the specific location of Dame Judi Dench's birth from the provided search results and output it as a final textual answer.; graph_revision=`4`; environment_revision=`None`; tool_receipts=`1`; body=Dame Judi Dench was born in York, England.

Executor ReAct trace（公开 StructuredAction/observation）：

- agent=`searcher`; turn=`1`; action={"arguments": {"limit": 5, "query": "Dame Judi Dench birthplace England"}, "kind": "tool", "name": "search", "resource_id": "qa-retrieval.search", "skill_id": null}; observation_status=`success`; observation={"completed": true, "executed_action": {"arguments": {"limit": 5, "query": "Dame Judi Dench birthplace England"}, "kind": "tool", "name": "search", "resource_id": "qa-retrieval.search", "skill_id": null}, "observation_status": "success", "result": {"hits": [{"document_id": "sha256:d5769d4c3e2357c33a77a2a4c5e3222aae4afc3a9e7f5d95b992786f37caffbc", "passage_id": "atlas-dpr-wikipedia:000017426401", "rank": 1, "snippet": "Judi Dench filmography Dame Judi Dench is an English actress who has worked in theater, television, and film. Dench made her professional debut in 1957 with the Old Vic Company. Over the following few years she played in several of Shakespeare's plays in such roles as Ophelia …
- agent=`searcher`; turn=`2`; action={"arguments": {"value": "Dame Judi Dench was born in York, England."}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null

Tool receipts：

- receipt=1; tool=`qa-retrieval.search`; version=`sha256:b9a491d7cdaa2a107e8a12d579bfabd37ce496ad92feb9a1a5c69527f9e026e8`; status=`completed`; latency_ms=`1246.8277914449573`; request={"action": "search", "arguments": {"limit": 5, "query": "Dame Judi Dench birthplace England"}}; result={"completed": true, "value": {"hits": [{"document_id": "sha256:d5769d4c3e2357c33a77a2a4c5e3222aae4afc3a9e7f5d95b992786f37caffbc", "passage_id": "atlas-dpr-wikipedia:000017426401", "rank": 1, "snippet": "Judi Dench filmography Dame Judi Dench is an English actress who has worked in theater, television, and film. Dench made her professional debut in 1957 with the Old Vic Company. Over the following few years she played in several of Shakespeare's plays in such roles as Ophelia in \"Hamlet\", Juliet in \"Romeo and Juliet\" a…", "title": "Judi Dench filmography"}, {"document_id": "sha256:15271d52fc252ec1fb8a4e156530dff1ec22884688753287e2714912d58f47ce", "passage_id": "atlas-dpr-wikipedia:000000…
## Wrong / Failure Demo

当前 AgentGraph 2 题均成功；以下是相同任务或固定条件中保留的真实适配前 failure receipt。该结果只用于 root-cause 对照，不混入当前 Stable Zero 指标。

- Preserved condition：`qa_tool_react_stable_zero (pre-v3 output-span clarification)`
- Failure classification：`Output Agent contract / answer-span serialization`

### Preserved Wrong Demo: `triviaqa:tc_3`

- Raw receipts：paired=`artifacts/qa_tool_react_stable_zero/triviaqa/wrong_demos.jsonl`; trajectory=`artifacts/qa_tool_react_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl`

- Task：Where in England was Dame Judi Dench born?
- Ground Truth：York | Park Grove (1895) | York UA | Yorkish | UN/LOCODE:GBYRK | York, UK | Eoforwic | Park Grove School | York Ham | The weather in York | City of York | York, England | York, Yorkshire | York ham | County Borough of York | YORK | Eoferwic | Park Grove Primary School | York, North Yorkshire | Yoisk | York (England)
- Final Answer：<answer>York, North Yorkshire, England</answer>
- Evaluator：`triviaqa.official.answer.v1`; metrics=`{"exact_match": 0.0, "token_f1": 0.8571428571428571}`
- Trajectory ID：`trajectory_abe8e2bec7a2a2441c00b516`
- Policy version：`qwen35-9b-hotpot-step-000000`; prompt=`agentgraph.director.minimal.v2`; tool=`skillflow.qa-retrieval.search-read.v1`
- Output Agent：`Solver`
- AgentGraph: `Planner → Executor; Executor → Solver`

Agent 配置：

- `Executor` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`executor`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`text`; contract: Execute the search query to find information about Dame Judi Dench's birthplace.
- `Planner` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`planning`, allowed_tools=`[]`, artifact_type=`text`; contract: Reason the location of Judi Dench's birth in England using available trivia tools.
- `Solver` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Synthesize the search results and reasoning to determine the specific English town where she was born and then output the semantic answer using the format protocol.

Director atomic edit 序列：`invalid → add_subgraph → invalid → add_subgraph → add_subgraph → modify_agent → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: unknown action fields: completition_condition
- round=`1`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`2`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: add_subgraph supports at most three Agents
- round=`3`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: unknown agent_id: format
- round=`4`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 6; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'Planner' cannot declare allowed_tools"}
- round=`5`; action=`modify_agent`; graph_revision=`7`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['Planner'], ['Executor'], ['Solver']]`; executed=`['Executor', 'Planner', 'Solver']`; reused=`[]`; feedback=accepted modify_agent at revision 7; execution_result={"output_agent_id":"Solver","output":"<answer>York, North Yorkshire, England</answer>","executed_agent_ids":["Executor","Planner","Solver"],"reused_agent_ids":[],"topology":{"agent_count":3,"relation_count":2,"directed_edge_count":2,"quotient_directed_edge_count":2,"reciprocal_pair_count":0,"component_count":3,"max_depth":3,"structural_depth":3,"max_width":1,"topology_family":"serial_3_plus","topology_motifs":["serial_3_plus"],"root_agent_ids":["Planner"],"sink_agent_ids":["Solver"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"Solver"},"output_inbox":[{"source_agent_id"…
- round=`6`; action=`finish`; graph_revision=`7`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['Planner'], ['Executor'], ['Solver']]`; executed=`['Executor', 'Planner', 'Solver']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `Planner` → `Executor`; artifact_type=`text`; dependency=Execute the search query to find information about Dame Judi Dench's birthplace.; graph_revision=`7`; environment_revision=`None`; tool_receipts=`0`; body=**Evidence Artifact:** * **Subject:** Dame Judi Dench * **Birth Date:** 9 December 1934 * **Birth Location:** York, North Yorkshire, England * **Verification Status:** Confirmed via multiple authoritative biographical sources (including official profiles and encyclopedic records). * **Reasoning:** Standard biographical data consistently identifies York as the specific city of birth within England.
- `Executor` → `Solver`; artifact_type=`text`; dependency=Synthesize the search results and reasoning to determine the specific English town where she was born and then output the semantic answer using the format protocol.; graph_revision=`7`; environment_revision=`None`; tool_receipts=`0`; body=York, North Yorkshire, England

Executor ReAct trace（公开 StructuredAction/observation）：

- agent=`Executor`; turn=`1`; action={"arguments": {"value": "York, North Yorkshire, England"}, "kind": "complete", "name": "complete", "resource_id": null, "skill_id": null}; observation_status=`completed`; observation=null

FIRST ERROR：Output Agent returned `York, North Yorkshire, England` even though the task asks for the English town and the Direct arm returned `York`. The official TriviaQA answer evaluator therefore recorded EM=0 and token F1=0.8571428571. The receipt does not support attributing this recovered semantic answer to retrieval or model reasoning; the first evaluator-visible error is the over-specified terminal answer span.
