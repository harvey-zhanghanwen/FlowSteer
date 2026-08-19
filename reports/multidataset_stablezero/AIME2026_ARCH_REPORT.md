# AIME 2026 架构报告

## Stable Zero

- 能力边界：推理 + 有界 calculator/Python execution 能力
- Protocol：Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | exact_match |
|---|---:|
| Direct/Simple Baseline | 50.00% |
| AgentGraph | 100.00% |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

- `serial_2`: 2

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `gpt-4o-mini`: 2 Agent nodes
- `qwen3.5-flash`: 2 Agent nodes

- Model family：GPT=2, Qwen=2
- Multi-model workflow 比例：**0/2**

## Tool / ReAct 使用情况

- Tool call：**0**；成功：**0**；失败：**0**
- AgentGraph 原生 environment action：**0**；invalid action：**0**
- Direct 原生 environment action：**0**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 2 | 1623 | 24.17 |
| AgentGraph | 29 | 90566 | 149.94 |

## Correct Demo

### Correct Demo: `aime-2026:01`

- Task：Patrick started walking at a constant rate along a straight road from school to the park. One hour after Patrick left, Tanya started running along the same road from school to the park. One hour after Tanya left, Jose started bicycling along the same road from school to the park. Tanya ran at a constant rate of $2$ miles per hour faster than Patrick walked, Jose bicycled at a constant rate of $7$ miles per hour faster than Tanya ran, and all three arrived at the park at the same time. The distance from the school to the park is $\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.
- Ground Truth：277
- Final Answer：<answer>277</answer>
- Evaluator: `{"accuracy": 1.0, "exact_match": 1.0}`
- AgentGraph: `router → format`

Agent 配置：

- `format` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the final numeric answer (value of m+n) from the upstream computation and format it strictly as the required output without additional explanation.
- `router` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`director`, allowed_tools=`[]`, artifact_type=`text`; contract: Receive the computed m+n value from solver and route it to the format agent via relation.

Director atomic edit 序列：`add_subgraph → add_subgraph → modify_agent → delete_agent → finish → ? → add_subgraph → add_subgraph → finish → modify_agent → finish → delete_agent → finish → add_subgraph → add_subgraph → modify_agent → delete_agent → finish`

Output Agent 实际 inbox：

- `router` → `format`; artifact_type=`text`; body=Let \( d \) be the distance from the school to the park in miles, \( r \) be Patrick's walking rate in miles per hour, \( t \) be the time in hours that Patrick walked until he reached the park, and \( t - 1 \) be the time in hours that Tanya ran until she reached the park. Since Tanya runs at a rate of \( r + 2 \) miles per hour, we can express the distance \( d \) in terms of Patrick's rate and time: \[ d = r \cdot t \] For Tanya, who starts one hour later, the distance can be expressed as: \…

### Correct Demo: `aime-2026:02`

- Task：Find the number of positive integer palindromes written in base $10$ with no zero digits, and whose digits add up to $13$. For example, $42124$ has these properties. Recall that a palindrome is a number whose representation reads the same from left to right as from right to left.
- Ground Truth：62
- Final Answer：<answer>62</answer>
- Evaluator: `{"accuracy": 1.0, "exact_match": 1.0}`
- AgentGraph: `solver → output_agent`

Agent 配置：

- `output_agent` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Extract the final count from the solver's text.
- `solver` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`solver`, allowed_tools=`[]`, artifact_type=`text`; contract: Find the count of positive integer palindromes in base 10 with no zero digits whose digits sum to 13.

Director atomic edit 序列：`? → add_subgraph → set_output → finish`

Output Agent 实际 inbox：

- `solver` → `output_agent`; artifact_type=`text`; body=To determine the count of positive integer palindromes in base 10 with no zero digits whose digits sum to 13, we analyze the structural properties of such numbers. **1. Structural Constraints** Let the palindrome be denoted by the sequence of digits $d_1 d_2 \dots d_n$. Since the number contains no zero digits, $d_i \in \{1, 2, \dots, 9\}$ for all $i$. The sum of the digits is $\sum_{i=1}^n d_i = 13$. Since the maximum value of a single digit is 9, the number of digits $n$ cannot be 1 (as $13 >…
## Wrong Demo

该 2 题样本中没有 AgentGraph 错例；不进行虚构。
