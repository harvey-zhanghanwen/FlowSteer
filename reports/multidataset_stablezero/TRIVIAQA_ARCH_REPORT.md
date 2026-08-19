# TriviaQA 架构报告

## Stable Zero

- 能力边界：仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | exact_match | token_f1 |
|---|---:|---:|
| Direct/Simple Baseline | 50.00% | 50.00% |
| AgentGraph | 50.00% | 92.86% |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

- `serial_2`: 1
- `serial_3_plus`: 1

- 平均 structural depth：**2.50**
- 平均 effective dependency depth：**2.50**
- 平均 Agent 数：**2.50**
- 平均 relation 数：**1.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `qwen3.5-flash`: 5 Agent nodes

- Model family：Qwen=5
- Multi-model workflow 比例：**0/2**

## Tool / ReAct 使用情况

- Tool call：**0**；成功：**0**；失败：**0**
- AgentGraph 原生 environment action：**0**；invalid action：**0**
- Direct 原生 environment action：**0**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 2 | 478 | 0.70 |
| AgentGraph | 16 | 46560 | 355.27 |

## Correct Demo

### Correct Demo: `triviaqa:tc_1`

- Task：Which American-born Sinclair won the Nobel Prize for Literature in 1930?
- Ground Truth：Sinclair Lewis | (Harry) Sinclair Lewis | Harry Sinclair Lewis | Lewis, (Harry) Sinclair | Grace Hegger
- Final Answer：<answer>Sinclair Lewis</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `agent-1 → agent-2`

Agent 配置：

- `agent-1` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`researcher`, allowed_tools=`[]`, artifact_type=`text`; contract: Identify the American-born Sinclair who won the Nobel Prize for Literature in 1930 by using the search tool.
- `agent-2` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Provide the full name of the Nobel Laureate.

Director atomic edit 序列：`add_subgraph → modify_agent → finish`

Output Agent 实际 inbox：

- `agent-1` → `agent-2`; artifact_type=`text`; body=**Search Query:** "1930 Nobel Prize Literature winner American" **Retrieved Evidence:** * **Award Year:** 1930 * **Category:** Nobel Prize in Literature * **Recipient Name:** Sinclair Lewis * **Birthplace:** Sauk Centre, Minnesota, United States * **Nationality:** American * **Notable Work:** *Main Street*, *Babbitt*, *Elmer Gantry* **Verification Notes:** * The recipient's first name is "Sinclair", matching the query descriptor "American-born Sinclair". * The surname is "Lewis", not "Sinclair"…
## Wrong Demo

### Wrong Demo: `triviaqa:tc_3`

- Task：Where in England was Dame Judi Dench born?
- Ground Truth：York | Park Grove (1895) | York UA | Yorkish | UN/LOCODE:GBYRK | York, UK | Eoforwic | Park Grove School | York Ham | The weather in York | City of York | York, England | York, Yorkshire | York ham | County Borough of York | YORK | Eoferwic | Park Grove Primary School | York, North Yorkshire | Yoisk | York (England)
- Final Answer：<answer>York, North Yorkshire, England</answer>
- Evaluator: `{"exact_match": 0.0, "token_f1": 0.8571428571428571}`
- AgentGraph: `Planner → Executor; Executor → Solver`

Agent 配置：

- `Executor` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`executor`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`text`; contract: Execute the search query to find information about Dame Judi Dench's birthplace.
- `Planner` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`planning`, allowed_tools=`[]`, artifact_type=`text`; contract: Reason the location of Judi Dench's birth in England using available trivia tools.
- `Solver` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Synthesize the search results and reasoning to determine the specific English town where she was born and then output the semantic answer using the format protocol.

Director atomic edit 序列：`? → add_subgraph → ? → add_subgraph → add_subgraph → modify_agent → finish`

Output Agent 实际 inbox：

- `Executor` → `Solver`; artifact_type=`text`; body=York, North Yorkshire, England

FIRST ERROR：terminal answer serialization 与 accepted answer span 不一致，但 token overlap 非零；错误位于 Format boundary。
