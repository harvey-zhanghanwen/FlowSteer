# WebShop 架构报告

## Stable Zero

- 能力边界：request-scoped SkillFlow/RAGEN environment ReAct
- Protocol：Direct 与 AgentGraph 使用相同原生 WebShop validation 环境、task lock、action budget 和 evaluator。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- optimizer update：**0**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | success |
|---|---:|
| Direct/Simple Baseline | 50.00% |
| AgentGraph | 50.00% |

以上是当前 evidence scope 中 2 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Evidence scope 与协议限制

- Evidence scope：native validation indices 500..627；2-task canary when executed
- Protocol：Direct 与 AgentGraph 使用相同原生 WebShop validation 环境、task lock、action budget 和 evaluator。

- 只有完整原生 environment transition receipt 与 terminal success 才计入 success。

### 明确排除的历史结果

- artifacts/webshop_ragen_environment_native_action_v2_stable_zero/evaluation：旧结果取自 native test 范围，属于 test-contaminated adaptation evidence，不进入 v4 development 指标；v3 development 因未传递 SkillFlow max_action_tokens 导致本地 Direct 上下文超限，仅保留为失败诊断。


## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | success=50.00% | Direct 与 AgentGraph 使用相同原生 WebShop validation 环境、task lock、action budget 和 evaluator。 |
| AgentGraph Stable Zero | success=50.00% | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | success=50.00% | current v4 development condition after native-action, split-isolation and action-token-budget adaptation |
| Tool/ReAct/Coding-enabled AgentGraph | success=50.00% | native environment replay; actual actions=14 |

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

- `deepseek-v4-flash`: 4 Agent nodes
- `qwen3.5-9b-local`: 1 Agent nodes

- Model family：DeepSeek=4, Qwen=1
- Multi-model workflow 比例：**1/2**

## Tool / ReAct 使用情况

- Tool call：**14**；成功：**14**；失败：**0**
- Tool call task rate：**2/2**
- Tool useful rate：**不可测**；当前 receipt 没有独立的 causal usefulness annotation
- Tool wasted rate：**不可测**；Tool error 单独报告，不能等同于无效信息价值
- Environment transition receipt：**14**
- Coding action receipt：**0**
- AgentGraph 原生 environment action：**14**；invalid action：**0**
- Direct 原生 environment action：**19**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 19 | 54333 | 3.61 |
| AgentGraph | 22 | 54021 | 69.11 |

## Correct Demo

### Correct Demo: `webshop:00501`

- Task：i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars
- Ground Truth：environment_success
- Final Answer：<answer>add_to_cart</answer>
- Evaluator: `{"environment_return": 1.0, "steps": 6.0, "success": 1.0, "terminal": 1.0}`
- AgentGraph: `planner → selector; selector → terminal`

Agent 配置：

- `planner` — model=`deepseek-v4-flash`, execution_mode=`react`, role_family=`operator`, allowed_tools=`['webshop.environment']`, artifact_type=`product_options`; contract: deduction
- `selector` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`operator`, allowed_tools=`[]`, artifact_type=`validated_pro`; contract: selection
- `terminal` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: terminal

Director atomic edit 序列：`add_subgraph → finish`

Output Agent 实际 inbox：

- `selector` → `terminal`; artifact_type=`validated_pro`; body=1.0

原生 ReAct trace（6 个 action）：`search[heavy duty dust proof tempered glass iPhone 13 Pro Max 6.7 inch case+4 red black under 50] → click[b09p572dp9] → click[case+4 protectors] → click[redblack] → click[redblack] → click[buy now]`

Tool receipts：

- tool=`webshop.environment`, status=`completed`; request={"action": "search[heavy duty dust proof tempered glass iPhone 13 Pro Max 6.7 inch case+4 red black under 50]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[b09p572dp9]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[case+4 protectors]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[redblack]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[redblack]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[buy now]", "arguments": {}}
## Wrong Demo

### Wrong Demo: `webshop:00500`

- Task：i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars
- Ground Truth：environment_success
- Final Answer：<answer>B09GF9SSQN</answer>
- Evaluator: `{"environment_return": 0.6, "steps": 8.0, "success": 0.0, "terminal": 1.0}`
- AgentGraph: `env_exec_a1 → out_a1`

Agent 配置：

- `env_exec_a1` — model=`deepseek-v4-flash`, execution_mode=`react`, role_family=`reactor`, allowed_tools=`['webshop.environment']`, artifact_type=`env_observation`; contract: Execute the selected WebShop environment action.
- `out_a1` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`final_answer`; contract: Extract the final product match from the terminal observation.

Director atomic edit 序列：`add_subgraph → delete_agent → finish`

Output Agent 实际 inbox：

- `env_exec_a1` → `out_a1`; artifact_type=`env_observation`; body=Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09GF9SSQN [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 0.6 [SE…

原生 ReAct trace（8 个 action）：`search[small end table easy assemble blue steel frame under 70] → click[b096xnxgjq] → click[back to search] → search[blue coated steel frame end table under 70 easy assembly] → click[b09gf9ssqn] → click[description] → click[< prev] → click[buy now]`

Tool receipts：

- tool=`webshop.environment`, status=`completed`; request={"action": "search[small end table easy assemble blue steel frame under 70]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[b096xnxgjq]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[back to search]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "search[blue coated steel frame end table under 70 easy assembly]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[b09gf9ssqn]", "arguments": {}}
- tool=`webshop.environment`, status=`completed`; request={"action": "click[description]", "arguments": {}}

FIRST ERROR：原生 environment episode 结束时未达到 terminal success。

## 最小架构适配

保留的 v1 receipt 显示连续 10 次 `parse_error` transition：Executor 输出 JSON action object，而 WebShop 只接受原生 `search[...]` / `click[...]` action。executor action grammar 对自由文本 Canvas contract 具有执行优先级。当前 v4 使用 native validation indices 500..627，并在两题 canary 上完成 2/2 full-chain Stable Zero；Direct 与 AgentGraph success 均为 1/2。旧 v2 native-test 结果明确排除；v3 development 因未传递 SkillFlow `max_action_tokens` 导致本地 Direct 上下文超限，仅保留为失败诊断。以上是接口与数据隔离修正，不是 benchmark-specific workflow template。
