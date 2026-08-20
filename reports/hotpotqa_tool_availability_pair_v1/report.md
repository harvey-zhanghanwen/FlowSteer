# HotpotQA Tool availability OFF/ON 配对诊断

## 结论

两个已暴露 development task 上完成 2 个随机化 OFF/ON pair 和 4 条 evaluator-valid trajectory。Tool-OFF 与 Tool-ON 均为 EM=100%、F1=100%，availability assignment ITT 为 ΔEM=0、ΔF1=0。两个 Tool-ON arm 均未实际调用检索 Tool，因此只验证 treatment exposure 与 paired runtime，不能证明 Tool usefulness、retrieval usefulness 或 Skill effect。

该结果不是 held-out accuracy、benchmark estimate 或 SOTA 证据；`forced_probe=true`、`grpo_eligible=false`。

## Receipt 与协议

- Raw manifest：`artifacts/hotpotqa_tool_availability_pair_v1/development/run_manifest.json`
- Pair receipt：`artifacts/hotpotqa_tool_availability_pair_v1/development/paired_results.jsonl`
- 完整 trajectory：`artifacts/hotpotqa_tool_availability_pair_v1/development/tool_availability_trajectories.jsonl`
- Direct reuse：`artifacts/hotpotqa_tool_availability_pair_v1/development/direct_reused_v3.jsonl`
- Direct 严格复用 2 条，新增 Direct API call=0；EM=100.00%，F1=100.00%
- OFF catalog：空；ON catalog：`qa-retrieval.search`、`qa-retrieval.read`
- non-treatment observation projection equal：2/2
- ACTIVE/retrieved/invoked Skill ID：全部为空
- training/backward/optimizer update/policy publication：均未执行

## 指标

| Condition | n | Evaluator valid | Explicit FINISH | EM | F1 | Tool-invoked task | ToolReceipt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tool-OFF | 2 | 2 | 2 | 100.00% | 100.00% | 0 | 0 |
| Tool-ON | 2 | 2 | 2 | 100.00% | 100.00% | 0 | 0 |

## Workflow 与模型采用

- Tool-OFF：topology={'serial_2': 1, 'serial_3_plus': 1}，depth=[2, 3]，width=[1, 1]
- Tool-ON：topology={'serial_2': 1, 'serial_3_plus': 1}，depth=[2, 3]，width=[1, 1]
- 4 条终图均为 serial DAG；parallel、fan-in、fan-out、reciprocal 均未采用。
- Final graph model IDs：OFF=['qwen3.5-flash', 'qwen3.5-9b-local', 'glm-4.5-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']；ON=['qwen3.5-flash', 'qwen3.5-flash', 'deepseek-v4-flash', 'deepseek-v4-flash', 'deepseek-v4-flash']
- Multi-model workflow：OFF=2/2，ON=0/2，合计=2/4。

## 已持久化成本

| Layer | Accepted calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| Flow-Director | 21 | 21 | 93933 | 24.18 |
| Executor | 11 | 12 | 27229 | 320.19 |

失败 provider call 没有完整 token/latency receipt，因此不能重建为精确总调用成本；上表仅汇总已持久化记录，并保留中间恢复阶段已经保存的 Executor 调用。

## 四条真实 Demo

### `hotpotqa:5a7a06935542990198eaf050` / `tool_on`

- Question：Based on the following passages, answer the question. [[Radio City (Indian radio station)] Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recent…
- Ground Truth：`Arthur's Magazine`
- Final Answer：`<answer>Arthur's Magazine</answer>`
- Official evaluator：EM=1.00, F1=1.00
- Trajectory：`trajectory_24ebf95e33c6adaa180a1e3b`；explicit FINISH=`true`
- Director atomic edit：`add_subgraph → modify_agent → finish → delete_agent → add_subgraph → add_subgraph → finish`
- Final topology：`serial_2`；depth=2；width=1

Agent：

- `Emitter`: role=`format`, model=`qwen3.5-flash`, mode=`reasoning`, tools=`[]`; contract=Present the final answer identifying which magazine was started first based on the analysis.
- `analyzer`: role=`reasoning`, model=`qwen3.5-flash`, mode=`reasoning`, tools=`[]`; contract=Compare the start dates of 'Arthur's Magazine' and 'First for Women' from the provided passages and identify which one started first.

实际 CommunicationEnvelope：

- `analyzer` → `Emitter`; artifact_type=`text`; graph_revision=7; ToolReceipt=0; body=Evidence & Facts: - **Arthur's Magazine**: Start year is 1844 (inferred from title range "Arthur's Magazine (1844–1846)"). - **First for Women**: Start year is 1989 (explicitly stated: "The magazine was started in 1989"). Reasoning: - Comparing the start years: 1844 vs 1989. - 1…

公开 ReAct trace：

- 无公开 ReAct action

真实恢复事件：

- round=0, action=`add_subgraph`: accepted add_subgraph at revision 4; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'analyzer' cannot declare allowed_tools"}
- round=1, action=`modify_agent`: edit rejected: action made no graph change; modify an Agent contract/model or another graph field before expecting a new execution
- round=2, action=`finish`: cannot finish: execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'analyzer' cannot declare allowed_tools"}
- round=4, action=`add_subgraph`: edit rejected: add_subgraph Agent already exists: Emitter

### `hotpotqa:5a7a06935542990198eaf050` / `tool_off`

- Question：Based on the following passages, answer the question. [[Radio City (Indian radio station)] Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recent…
- Ground Truth：`Arthur's Magazine`
- Final Answer：`<answer>Arthur's Magazine</answer>`
- Official evaluator：EM=1.00, F1=1.00
- Trajectory：`trajectory_23b95c8a32bcc9e54b5ae51a`；explicit FINISH=`true`
- Director atomic edit：`invalid → add_subgraph → add_subgraph → finish`
- Final topology：`serial_2`；depth=2；width=1

Agent：

- `agent_format`: role=`format`, model=`qwen3.5-flash`, mode=`reasoning`, tools=`[]`; contract=Extract the exact answer from the upstream agent's reasoning and articulate the final terminal response: 'First for Women was started first in [Year]' or 'Arthur's Magazine was started first in [Year]'.
- `agent_semantic`: role=`reasoner`, model=`qwen3.5-9b-local`, mode=`reasoning`, tools=`[]`; contract=Read the provided passages, identify the start years for 'Arthur's Magazine' and 'First for Women', compare them, and resolve the question 'Which magazine was started first?'. Output the final answer clearly.

实际 CommunicationEnvelope：

- `agent_semantic` → `agent_format`; artifact_type=`text`; graph_revision=4; ToolReceipt=0; body=Evidence from passages: - "Arthur's Magazine (1844–1846) was an American literary periodical..." - "First for Women is a woman's magazine... The magazine was started in 1989." Comparison: - Arthur's Magazine started in 1844. - First for Women started in 1989. Resolution: 1844 is…

公开 ReAct trace：

- 无公开 ReAct action

真实恢复事件：

- round=1, action=`add_subgraph`: edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent

### `hotpotqa:5a879ab05542996e4f30887e` / `tool_on`

- Question：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family…
- Ground Truth：`Delhi`
- Final Answer：`<answer>Delhi</answer>`
- Official evaluator：EM=1.00, F1=1.00
- Trajectory：`trajectory_d9d84e66a9d5f12e5ed5de7d`；explicit FINISH=`true`
- Director atomic edit：`add_subgraph → add_subgraph → finish → modify_agent → modify_agent → modify_agent → finish`
- Final topology：`serial_3_plus`；depth=3；width=1

Agent：

- `output`: role=`format`, model=`deepseek-v4-flash`, mode=`reasoning`, tools=`[]`; contract=Extracts the exact city name from the solver's artifact and presents it as the final answer.
- `searcher`: role=`retriever`, model=`deepseek-v4-flash`, mode=`react`, tools=`['qa-retrieval.search']`; contract=Uses qa-retrieval.search to find the specific passage containing 'head office' information for the company associated with the Oberoi family.
- `solver`: role=`reasoner`, model=`deepseek-v4-flash`, mode=`reasoning`, tools=`[]`; contract=Analyzes the retrieved passage and the provided context to determine the city where the Oberoi Group's head office is located.

实际 CommunicationEnvelope：

- `solver` → `output`; artifact_type=`text`; graph_revision=9; ToolReceipt=0; body=The Oberoi Group, the hotel company associated with the Oberoi family, has its head office in Delhi.
- `searcher` → `solver`; artifact_type=`text`; graph_revision=9; ToolReceipt=0; body=Delhi

公开 ReAct trace：

- agent=`searcher`; action=`{}`; observation_status=`completed`

真实恢复事件：

- round=0, action=`add_subgraph`: edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=1, action=`add_subgraph`: accepted add_subgraph at revision 6; execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'searcher' during single: provider request failed for vectorengine: HTTP 500"}
- round=2, action=`finish`: cannot finish: execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'searcher' during single: provider request failed for vectorengine: HTTP 500"}
- round=3, action=`modify_agent`: accepted modify_agent at revision 7; execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'output' during single: provider request failed for vectorengine: HTTP 500"}
- round=4, action=`modify_agent`: accepted modify_agent at revision 8; execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'solver' during single: provider request failed for vectorengine: HTTP 500"}

### `hotpotqa:5a879ab05542996e4f30887e` / `tool_off`

- Question：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family…
- Ground Truth：`Delhi`
- Final Answer：`<answer>Delhi</answer>`
- Official evaluator：EM=1.00, F1=1.00
- Trajectory：`trajectory_737f2db740b9066ac3f1659b`；explicit FINISH=`true`
- Director atomic edit：`add_subgraph → modify_agent → finish`
- Final topology：`serial_3_plus`；depth=3；width=1

Agent：

- `format_agent`: role=`format`, model=`glm-4.5-flash`, mode=`reasoning`, tools=`[]`; contract=Output the final answer in the format: 'The city is [City Name].'
- `reasoner`: role=`reasoner`, model=`deepseek-v4-flash`, mode=`reasoning`, tools=`[]`; contract=Based on the retrieved facts, deduce the specific city where the Oberoi group's head office is located.
- `retriever`: role=`retriever`, model=`deepseek-v4-flash`, mode=`reasoning`, tools=`[]`; contract=Read provided context passages and extract facts related to 'Oberoi family' and 'hotel company head office'.

实际 CommunicationEnvelope：

- `reasoner` → `format_agent`; artifact_type=`text`; graph_revision=7; ToolReceipt=0; body=The Oberoi Group’s head office is in Delhi.
- `retriever` → `reasoner`; artifact_type=`text`; graph_revision=7; ToolReceipt=0; body=Delhi

公开 ReAct trace：

- 无公开 ReAct action

真实恢复事件：

- round=0, action=`add_subgraph`: accepted add_subgraph at revision 6; execution_error={"type":"AgentRuntimeError","message":"agent 'format_agent' requires unregistered execution adapter 'react'"}


## Wrong Demo 与诊断

- Terminal Wrong Demo：0；四条 trajectory 均由原生 evaluator 判定正确，不制造失败样本。
- Tool adoption：0/2 Tool-ON task。只能记录为当前 policy 没有采用检索，不能由两题推出检索无效。
- Non-chain adoption：0/4。search space/runtime 能力与 policy adoption 分开报告。
- `QA_TOOL_USE_VALIDATED = NO`；`SKILL_END_TO_END_READY = NO`。
