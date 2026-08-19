# HotpotQA 架构报告

## Stable Zero

- 能力边界：闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
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

- Evidence scope：fixed validation canary
- Protocol：Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。

- 无额外限制记录。

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

- `serial_2`: 1
- `serial_3_plus`: 1

- 平均 structural depth：**2.50**
- 平均 effective dependency depth：**2.50**
- 平均 Agent 数：**2.50**
- 平均 relation 数：**1.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `gpt-4o-mini`: 5 Agent nodes

- Model family：GPT=5
- Multi-model workflow 比例：**0/2**

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

- Task：Based on the following passages, answer the question. [[Radio City (Indian radio station)] Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The R…
- Ground Truth：Arthur's Magazine
- Final Answer：<answer>Arthur's Magazine</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `reader → answerer`

Agent 配置：

- `answerer` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Compare the start dates of Arthur's Magazine and First for Women and output the magazine that started first
- `reader` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`reader`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read the passages containing 'Arthur's Magazine' and 'First for Women' to find their launch/start dates

Director atomic edit 序列：`add_subgraph → add_subgraph → finish`

Output Agent 实际 inbox：

- `reader` → `answerer`; artifact_type=`text`; body=Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.

### Correct Demo: `hotpotqa:5a879ab05542996e4f30887e`

- Task：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.] [[Ishqbaaaz] Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast …
- Ground Truth：Delhi
- Final Answer：<answer>Delhi</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `reasoner → format; next_hop → reasoner`

Agent 配置：

- `format` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Output the exact city name as the final answer.
- `next_hop` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`retriever`, allowed_tools=`['qa-retrieval.search']`, artifact_type=`passage_id`; contract: Identify the passage containing information about the Oberoi family's hotel company head office based on the question and provided texts.
- `reasoner` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`researcher`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read the retrieved passage to extract the city where the Oberoi Group head office is located.

Director atomic edit 序列：`add_subgraph → finish`

Output Agent 实际 inbox：

- `reasoner` → `format`; artifact_type=`text`; body=Delhi

Tool receipts：

- tool=`qa-retrieval.search`, status=`completed`; request={"action": "search", "arguments": {"limit": 1, "query": "Oberoi Group head office location"}}
## Wrong / Failure Demo

当前 AgentGraph 2 题均成功；以下是相同任务或固定条件中保留的真实适配前 failure receipt。该结果只用于 root-cause 对照，不混入当前 Stable Zero 指标。

- Preserved condition：`qa_tool_react_exact_wire_v2_stable_zero (pre-v3 schema clarification)`
- Failure classification：`Director action-schema / Tool resource identifier / terminal control`

### Preserved Wrong Demo: `hotpotqa:5a879ab05542996e4f30887e`

- Task：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.] [[Ishqbaaaz] Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast …
- Ground Truth：Delhi
- Final Answer：
- Evaluator: `{"exact_match": 0.0, "token_f1": 0.0}`
- AgentGraph: `reader → answerer`

Agent 配置：

- `answerer` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Analyze the retrieved passages to determine the head office city of the Upperoi family's hotel company and output the answer.
- `reader` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`processor`, allowed_tools=`['qa-retrieval.read']`, artifact_type=`text`; contract: Read passages relevant to the Oberoi family and its hotel company's head office location.

Director atomic edit 序列：`? → ? → ? → ? → ? → add_subgraph → finish → set_output → modify_agent → delete_agent → set_output → ? → ? → delete_agent → add_subgraph → add_subgraph → add_subgraph → set_relation → finish → modify_agent`

Output Agent 实际 inbox：

- `reader` → `answerer`; artifact_type=`text`; body=Delhi

FIRST ERROR：前五个 Director action 把 top-level output_agent_id 放入 Agent object，随后又把 action_name `search`/`read` 当成 allowed_tools 的 resource_id；Canvas 均 fail closed。第 20 轮得到正确输出后已无剩余显式 FINISH turn，因此以 max_rounds 终止。v3 只澄清字段层级和 exact tool_id 边界。
