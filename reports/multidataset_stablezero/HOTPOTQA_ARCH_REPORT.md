# HotpotQA 架构报告

## Stable Zero

- 能力边界：闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力
- Protocol：Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | exact_match | token_f1 |
|---|---:|---:|
| Direct/Simple Baseline | 100.00% | 100.00% |
| AgentGraph | 100.00% | 100.00% |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

- `serial_2`: 2

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `deepseek-v4-flash`: 2 Agent nodes
- `gpt-4o-mini`: 2 Agent nodes

- Model family：DeepSeek=2, GPT=2
- Multi-model workflow 比例：**2/2**

## Tool / ReAct 使用情况

- Tool call：**0**；成功：**0**；失败：**0**
- AgentGraph 原生 environment action：**0**；invalid action：**0**
- Direct 原生 environment action：**0**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 2 | 2982 | 1.04 |
| AgentGraph | 11 | 36109 | 78.81 |

## Correct Demo

### Correct Demo: `hotpotqa:5a7a06935542990198eaf050`

- Task：Based on the following passages, answer the question. [[Radio City (Indian radio station)] Radio City is India's first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The R…
- Ground Truth：Arthur's Magazine
- Final Answer：<answer>Arthur's Magazine</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `reasoner → format`

Agent 配置：

- `format` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Format Directives: ## Objective Take the semantic answer from the Reasoner Agent and output it as the final system response. ## Inputs - A single text artifact from the Reasoner Agent containing the definitive answer. ## Execution 1. Receive the artifact. 2. Output the content of the artifact exactly as the final answ…
- `reasoner` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`reasoner`, allowed_tools=`[]`, artifact_type=`text`; contract: Reasoning Directives: ## Objective Compare the start years of Arthur's Magazine and First for Women to determine which one was started first. ## Inputs - A single text artifact from the Extraction Agent, strictly in the format: "Initial Year [Year1], End Year [Year2]". The order in this string corresponds to the input…

Director atomic edit 序列：`add_subgraph → add_subgraph → add_subgraph → finish`

Output Agent 实际 inbox：

- `reasoner` → `format`; artifact_type=`text`; body=Initial Year [1844], End Year [1989]

### Correct Demo: `hotpotqa:5a879ab05542996e4f30887e`

- Task：Based on the following passages, answer the question. [[Ritz-Carlton Jakarta] The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005.] [[Oberoi family] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.] [[Ishqbaaaz] Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast …
- Ground Truth：Delhi
- Final Answer：<answer>Delhi</answer>
- Evaluator: `{"exact_match": 1.0, "token_f1": 1.0}`
- AgentGraph: `reasoner → format`

Agent 配置：

- `format` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the single city name from the reasoner's text artifact and format it as the final executable answer.
- `reasoner` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`solver`, allowed_tools=`[]`, artifact_type=`text`; contract: Analyze the passages to find the hotel company associated with the Oberoi family and identify its head office city.

Director atomic edit 序列：`add_subgraph → add_subgraph → finish`

Output Agent 实际 inbox：

- `reasoner` → `format`; artifact_type=`text`; body=The Oberoi family is associated with The Oberoi Group, which has its head office in Delhi.
## Wrong Demo

该 2 题样本中没有 AgentGraph 错例；不进行虚构。
