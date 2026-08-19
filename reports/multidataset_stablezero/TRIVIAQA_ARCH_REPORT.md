# TriviaQA 架构报告

## Stable Zero

- 能力边界：仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
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

- Evidence scope：fixed validation canary
- Protocol：Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。

- 无额外限制记录。

### 明确排除的历史结果

- v1 exact-schema 前的自然策略 canary 只保留为历史结果，不进入当前 v3 指标。


## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | exact_match=50.00%; token_f1=50.00% | Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。 |
| AgentGraph Stable Zero | exact_match=100.00%; token_f1=100.00% | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | exact_match=100.00%; token_f1=100.00% | current v3 condition after exact Director field/resource-ID clarification |
| Tool/ReAct/Coding-enabled AgentGraph | exact_match=100.00%; token_f1=100.00% | declared capability; actual Tool receipts=1 |

同一 receipt 同时代表多个 stage 时不会重复解释为独立实验，也不会把 protocol-separated 条件的差值解释为因果增益。

## Workflow 分布

- `serial_2`: 2

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `glm-4.5-flash`: 1 Agent nodes
- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-9b-local`: 2 Agent nodes

- Model family：GLM=1, GPT=1, Qwen=2
- Multi-model workflow 比例：**1/2**

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

- Task：Which American-born Sinclair won the Nobel Prize for Literature in 1930?
- Ground Truth：Sinclair Lewis | (Harry) Sinclair Lewis | Harry Sinclair Lewis | Lewis, (Harry) Sinclair | Grace Hegger
- Final Answer：<answer>Sinclair Lewis</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `investigative_analyst → output_agent`

Agent 配置：

- `investigative_analyst` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`analyst`, allowed_tools=`[]`, artifact_type=`text`; contract: Identify the American-born Sinclair who won the Nobel Prize in Literature in 1930 and draft a concise rationale
- `output_agent` — model=`glm-4.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the final answer from [investigative_analyst]'s text artifact: 'Name. Reason'. Output the exact string 'Name. Reason' as the only content

Director atomic edit 序列：`add_subgraph → modify_agent → finish`

Output Agent 实际 inbox：

- `investigative_analyst` → `output_agent`; artifact_type=`text`; body=The American-born Sinclair who won the Nobel Prize in Literature in 1930 is Sinclair Lewis. He was recognized for his significant contributions to literature, particularly for his critical portrayal of American society and culture.

### Correct Demo: `triviaqa:tc_3`

- Task：Where in England was Dame Judi Dench born?
- Ground Truth：York | Park Grove (1895) | York UA | Yorkish | UN/LOCODE:GBYRK | York, UK | Eoforwic | Park Grove School | York Ham | The weather in York | City of York | York, England | York, Yorkshire | York ham | County Borough of York | YORK | Eoferwic | Park Grove Primary School | York, North Yorkshire | Yoisk | York (England)
- Final Answer：<answer>York</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `searcher → formatter`

Agent 配置：

- `formatter` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the specific location of Dame Judi Dench's birth from the provided search results and output it as a final textual answer.
- `searcher` — model=`qwen3.5-9b-local`, execution_mode=`react`, role_family=`searcher`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`search_results`; contract: Formulate a search query about Dame Judi Dench's birthplace, execute the search, and provide the retrieved passages.

Director atomic edit 序列：`add_subgraph → add_subgraph → finish`

Output Agent 实际 inbox：

- `searcher` → `formatter`; artifact_type=`search_results`; body=Dame Judi Dench was born in York, England.

Tool receipts：

- tool=`qa-retrieval.search`, status=`completed`; request={"action": "search", "arguments": {"limit": 5, "query": "Dame Judi Dench birthplace England"}}
## Wrong / Failure Demo

该 2 题样本中没有 AgentGraph 错例，也没有适用的 preserved failure receipt；不进行虚构。
