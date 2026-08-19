# WebShop 架构报告

## Stable Zero

- 能力边界：request-scoped SkillFlow/RAGEN environment ReAct
- Protocol：Direct 与 AgentGraph 使用相同原生环境、task lock、action budget 和 evaluator。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | success |
|---|---:|
| Direct/Simple Baseline | 100.00% |
| AgentGraph | 100.00% |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

- `single`: 2

- 平均 structural depth：**1.00**
- 平均 effective dependency depth：**1.00**
- 平均 Agent 数：**1.00**
- 平均 relation 数：**0.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `gpt-4o-mini`: 2 Agent nodes

- Model family：GPT=2
- Multi-model workflow 比例：**0/2**

## Tool / ReAct 使用情况

- Tool call：**6**；成功：**6**；失败：**0**
- AgentGraph 原生 environment action：**6**；invalid action：**0**
- Direct 原生 environment action：**6**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 6 | 9025 | 2.71 |
| AgentGraph | 11 | 18887 | 19.75 |

## Correct Demo

### Correct Demo: `webshop:00000`

- Task：i need a long clip-in hair extension which is natural looking, and price lower than 30.00 dollars
- Ground Truth：environment_success
- Final Answer：Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B08BZM24XR [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 1.0 [SEP] Reward Details [SEP] None
- Evaluator: `{"environment_return": 1.0, "steps": 3.0, "success": 1.0, "terminal": 1.0}`
- AgentGraph: `agent_1`

Agent 配置：

- `agent_1` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`customer_service`, allowed_tools=`['webshop.environment']`, artifact_type=`string`; contract: Shopping agent for lightweight hair extensions. This agent must factor 'long clip-in' and 'natural looking' into its shopping criteria, then keep price under $30 using the webshop tool.

Director atomic edit 序列：`add_subgraph → finish`

原生 ReAct trace（3 个 action）：`search[long clip-in hair extension natural looking under 30 dollars] → click[b08bzm24xr] → click[buy now]`

Tool receipts：

- tool=`webshop.environment`, status=`completed`; request={"action": "search[long clip-in hair extension natural looking under 30 dollars]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[b08bzm24xr]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[buy now]", "arguments": {}}

### Correct Demo: `webshop:00001`

- Task：i want a noise cancelling cosycost usb microphone, and price lower than 70.00 dollars
- Ground Truth：environment_success
- Final Answer：Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B0972Q1T8T [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 1.0 [SEP] Reward Details [SEP] None
- Evaluator: `{"environment_return": 1.0, "steps": 3.0, "success": 1.0, "terminal": 1.0}`
- AgentGraph: `shopper_agent`

Agent 配置：

- `shopper_agent` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`shopping`, allowed_tools=`['webshop.environment']`, artifact_type=`string`; contract: Act as a shopper agent to find a noise-cancelling USB microphone on a budget under $70 using the WebShop environment.

Director atomic edit 序列：`add_subgraph → ? → finish`

原生 ReAct trace（3 个 action）：`search[noise cancelling cosycost usb microphone under 70.00] → click[b0972q1t8t] → click[buy now]`

Tool receipts：

- tool=`webshop.environment`, status=`completed`; request={"action": "search[noise cancelling cosycost usb microphone under 70.00]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[b0972q1t8t]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[buy now]", "arguments": {}}
## Wrong Demo

该 2 题样本中没有 AgentGraph 错例；不进行虚构。


## 最小架构适配

保留的 v1 receipt 显示连续 10 次 `parse_error` transition：Executor 输出 JSON action object，而 WebShop 只接受原生 `search[...]` / `click[...]` action。修正后，executor action grammar 对自由文本 Canvas contract 具有执行优先级；在相同 2 个固定任务上，AgentGraph success 从 **1/2 变为 2/2**。这是接口修正，不是 benchmark-specific workflow template。
