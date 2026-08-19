# WebShop 架构报告

## Stable Zero

- 能力边界：request-scoped SkillFlow/RAGEN environment ReAct
- Protocol：Direct 与 AgentGraph 使用相同原生 WebShop validation 环境、task lock、action budget 和 evaluator。
- 固定 validation task：**2**
- Raw receipts：manifest=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/run_manifest.json`; paired=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/agentgraph_trajectories.jsonl`
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
- `Tool-using`: 2/2 tasks
- `Coding`: 0/2 tasks
- `mixed execution modes`: 2/2 tasks

- 平均 structural depth：**2.50**
- 平均 effective dependency depth：**2.50**
- 平均 Agent 数：**2.50**
- 平均 relation 数：**1.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

- `deepseek-v4-flash`: 4 Agent nodes
- `qwen3.5-9b-local`: 1 Agent nodes

- Model family：Qwen=1, DeepSeek=4, Gemini=0, GPT=0, MiniMax=0, Grok=0, GLM=0, Other=0
- Multi-model workflow 比例：**1/2**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| qwen3.5-9b-local | 2 | 2 | 7323 | 0.34 |

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| deepseek-v4-flash | 16 | 16 | 41955 | 64.79 |
| qwen3.5-9b-local | 1 | 1 | 384 | 0.12 |

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`qwen35-9b-webshop-ragen-environment-v4-stable-zero-step-000000`；prompt=`agentgraph.director.progressive_subgraph.v1`
- Director calls/attempts：**5/5**；tokens=**11682**；latency=**4.19s**



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

- Raw receipts：paired=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/agentgraph_trajectories.jsonl`

- Task：i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars
- Ground Truth：environment_success
- Final Answer：<answer>add_to_cart</answer>
- Evaluator：`skillflow.ragen_adapter.v2`; metrics=`{"environment_return": 1.0, "steps": 6.0, "success": 1.0, "terminal": 1.0}`
- Trajectory ID：`trajectory_0f64fb6d386909b65ff8311f`
- Policy version：`qwen35-9b-webshop-ragen-environment-v4-stable-zero-step-000000`; prompt=`agentgraph.director.progressive_subgraph.v1`; tool=`skillflow.ragen-environment-react.native-validation.v4`
- Output Agent：`terminal`
- AgentGraph: `planner → selector; selector → terminal`

Agent 配置：

- `planner` — model=`deepseek-v4-flash`, execution_mode=`react`, role_family=`operator`, allowed_tools=`['webshop.environment']`, artifact_type=`product_options`; contract: deduction
- `selector` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`operator`, allowed_tools=`[]`, artifact_type=`validated_pro`; contract: selection
- `terminal` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: terminal

Director atomic edit 序列：`add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['planner'], ['selector'], ['terminal']]`; executed=`['planner', 'selector', 'terminal']`; reused=`[]`; feedback=accepted add_subgraph at revision 6; execution_result={"output_agent_id":"terminal","output":"<answer>add_to_cart</answer>","executed_agent_ids":["planner","selector","terminal"],"reused_agent_ids":[],"topology":{"agent_count":3,"relation_count":2,"directed_edge_count":2,"quotient_directed_edge_count":2,"reciprocal_pair_count":0,"component_count":3,"max_depth":3,"structural_depth":3,"max_width":1,"topology_family":"serial_3_plus","topology_motifs":["serial_3_plus"],"root_agent_ids":["planner"],"sink_agent_ids":["terminal"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"terminal"},"output_inbox":[{"source_agent_id":"selector"…
- round=`1`; action=`finish`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['planner'], ['selector'], ['terminal']]`; executed=`['planner', 'selector', 'terminal']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `planner` → `selector`; artifact_type=`product_options`; dependency=selection; graph_revision=`6`; environment_revision=`6`; tool_receipts=`6`; body=Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09P572DP9 [SEP] options [SEP] {"color": "redblack", "size": "case+4 protectors"} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward […
- `selector` → `terminal`; artifact_type=`validated_pro`; dependency=terminal; graph_revision=`6`; environment_revision=`None`; tool_receipts=`0`; body=1.0

原生 environment ReAct trace（6 个 action）：`search[heavy duty dust proof tempered glass iPhone 13 Pro Max 6.7 inch case+4 red black under 50] → click[b09p572dp9] → click[case+4 protectors] → click[redblack] → click[redblack] → click[buy now]`
- step=1; action=`search[heavy duty dust proof tempered glass iPhone 13 Pro Max 6.7 inch case+4 red black under 50]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=WebShop [SEP] Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Search
- step=2; action=`click[b09p572dp9]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09H4PKRQZ [SEP] Lanhiem Metal Case for iPhone 13 Pro Max (6.7 inch), Heavy Duty Shockproof [Tough Armour] Rugged Case with Built-in Glass Screen Protector, 360 Full Body Dust Proof Protective Cover, Silver [SEP] $21.99 [SEP] B09GG4P4MD [SEP] iPhone 13 Pro Max Case, Neon Turtle iPhone 13 Pro Max Cases, Tempered Glass Back+Soft Silicone TPU Shock Protective Case for Apple iPhone 13 Pro Max [SEP] $14.99 [SEP] B09M9LZS2M [SEP] Compatible w…
- step=3; action=`click[case+4 protectors]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shockproof Drop Protective Cover for iPhone 13 Pro Max 6.7 Inch (WineRed+Lens…
- step=4; action=`click[redblack]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shockproof Drop Protective Cover for iPhone 13 Pro Max 6.7 Inch (WineRed+Lens…
- step=5; action=`click[redblack]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shockproof Drop Protective Cover for iPhone 13 Pro Max 6.7 Inch (WineRed+Lens…
- step=6; action=`click[buy now]`; reward=`1.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shockproof Drop Protective Cover for iPhone 13 Pro Max 6.7 Inch (WineRed+Lens…

Tool receipts：

- receipt=1; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`65.58891013264656`; request={"action": "search[heavy duty dust proof tempered glass iPhone 13 Pro Max 6.7 inch case+4 red black under 50]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 1, "observation": "Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09H4PKRQZ [SEP] Lanhiem Metal Case for iPhone 13 Pro Max (6.7 inch), Heavy Duty Shockproof [Tough Armour] Rugged Case with Built-in Glass Screen Protector, 360 Full Body Dust Proof Protective Cover, Silver [SEP] $21.99 [SEP] B09GG4P4MD [SEP] iPhone 13 Pro Max Case, Neon Turtle iPhone 13 Pro Max Cases, Tempered Glass Back+Soft Silicone TPU Shock Protective Case for…
- receipt=2; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`20.532548427581787`; request={"action": "click[b09p572dp9]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 2, "observation": "Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shoc…
- receipt=3; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`18.166267313063145`; request={"action": "click[case+4 protectors]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 3, "observation": "Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shoc…
- receipt=4; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`18.342374823987484`; request={"action": "click[redblack]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 4, "observation": "Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shoc…
- receipt=5; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`17.95380748808384`; request={"action": "click[redblack]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 5, "observation": "Instruction: [SEP] i need a heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch and its size is case+4 protectors with redblack color, and price lower than 50.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] black [SEP] pinkblue [SEP] purpleblue [SEP] redblack [SEP] wine red pink [SEP] black+clip [SEP] wine red pink+clip [SEP] black+lens [SEP] blackblue+lens [SEP] lightbluepink+lens [SEP] wineredpink+lens [SEP] size [SEP] case+2 protectors [SEP] case+2 protectors+clip [SEP] case+4 protectors [SEP] GREWORD for iPhone 13 Pro Max Case with 2 Screen Protector 2 Camera Protector Heavy Duty Shoc…
- receipt=6; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`25.8687911555171`; request={"action": "click[buy now]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 6, "observation": "Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09P572DP9 [SEP] options [SEP] {\"color\": \"redblack\", \"size\": \"case+4 protectors\"} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 1.0 [SEP] Reward Details [SEP] None", "terminal": true}}
## Wrong Demo

### Wrong Demo: `webshop:00500`

- Raw receipts：paired=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/agentgraph_trajectories.jsonl`

- Task：i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars
- Ground Truth：environment_success
- Final Answer：<answer>B09GF9SSQN</answer>
- Evaluator：`skillflow.ragen_adapter.v2`; metrics=`{"environment_return": 0.6, "steps": 8.0, "success": 0.0, "terminal": 1.0}`
- Trajectory ID：`trajectory_ef9a4a007ee5e72fba11da04`
- Policy version：`qwen35-9b-webshop-ragen-environment-v4-stable-zero-step-000000`; prompt=`agentgraph.director.progressive_subgraph.v1`; tool=`skillflow.ragen-environment-react.native-validation.v4`
- Output Agent：`out_a1`
- AgentGraph: `env_exec_a1 → out_a1`

Agent 配置：

- `env_exec_a1` — model=`deepseek-v4-flash`, execution_mode=`react`, role_family=`reactor`, allowed_tools=`['webshop.environment']`, artifact_type=`env_observation`; contract: Execute the selected WebShop environment action.
- `out_a1` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`final_answer`; contract: Extract the final product match from the terminal observation.

Director atomic edit 序列：`add_subgraph → delete_agent → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`7`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 7; execution_error={"type":"AgentRuntimeError","message":"stateful tool 'webshop.environment' requires one graph Agent owner; found 2"}
- round=`1`; action=`delete_agent`; graph_revision=`8`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['env_exec_a1'], ['out_a1']]`; executed=`['env_exec_a1', 'out_a1']`; reused=`[]`; feedback=accepted delete_agent at revision 8; execution_result={"output_agent_id":"out_a1","output":"<answer>B09GF9SSQN</answer>","executed_agent_ids":["env_exec_a1","out_a1"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["env_exec_a1"],"sink_agent_ids":["out_a1"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"out_a1"},"output_inbox":[{"source_agent_id":"env_exec_a1","target_agent_id":…
- round=`2`; action=`finish`; graph_revision=`8`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['env_exec_a1'], ['out_a1']]`; executed=`['env_exec_a1', 'out_a1']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `env_exec_a1` → `out_a1`; artifact_type=`env_observation`; dependency=Extract the final product match from the terminal observation.; graph_revision=`8`; environment_revision=`8`; tool_receipts=`8`; body=Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09GF9SSQN [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 0.6 [SE…

原生 environment ReAct trace（8 个 action）：`search[small end table easy assemble blue steel frame under 70] → click[b096xnxgjq] → click[back to search] → search[blue coated steel frame end table under 70 easy assembly] → click[b09gf9ssqn] → click[description] → click[< prev] → click[buy now]`
- step=1; action=`search[small end table easy assemble blue steel frame under 70]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=WebShop [SEP] Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Search
- step=2; action=`click[b096xnxgjq]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B096XNXGJQ [SEP] 15 Inch Modern Side Table, Industrial Small End Table with Thickened Top Plate, Accent Table, Wood Panel and Metal Frame, Nightstand Sofa Table for Living Room, Kitchen, Bedroom, Easy Assembly (Black) [SEP] $39.99 [SEP] B09S8WCJGK [SEP] New TV Tray Table Set Steel Stand End Home Furniture Serve Snack Dinner HOT, High and Narrow, Hallway, Living Room, Metal, Easy Assembly, Clean, Simple Space-Saving Design [SEP] $35.55 [SEP] B094VZTX4K [SEP] gen…
- step=3; action=`click[back to search]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] 15 Inch Modern Side Table, Industrial Small End Table with Thickened Top Plate, Accent Table, Wood Panel and Metal Frame, Nightstand Sofa Table for Living Room, Kitchen, Bedroom, Easy Assembly (Black) [SEP] Price: $39.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
- step=4; action=`search[blue coated steel frame end table under 70 easy assembly]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=WebShop [SEP] Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Search
- step=5; action=`click[b09gf9ssqn]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09J29KTKP [SEP] KOTEK Mobile Sofa Side Table, Height Adjustable C Shaped End Table with Rolling Casters, Portable Snack Table Laptop Desk Coffee Tray for Living Room, Bedroom [SEP] $49.99 [SEP] B09CZ41DST [SEP] Height Adjustable Side Table with Rolling Wheel Bedside Table,Slide Under Bedside End Table Laptop Desk Computer Stand Workstation for Home Office Bedroom (Red) [SEP] $42.99 [SEP] B09GF9SSQN [SEP] CSBYBD Table Sofa Side Table, Adjustable C Shaped End Ta…
- step=6; action=`click[description]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] CSBYBD Table Sofa Side Table, Adjustable C Shaped End Table with 2-Tier Storage Shelves for Sofa Couch, Mobile Desk with Wheels, Easy Assembly - Blue [SEP] Price: $38.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
- step=7; action=`click[< prev]`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] Home Office Desk Can Be Lifted And LoweBlue Mobile Computer Desk Bedside Table Feature: 100% brand new and high quality . Quantity: 1pcs Color:Blue Weight:16.09Ib Materials:MDF + steel frame Sheet thickness:0.59 in Desktop size:L:80cm×40cm/31.5×15.7inch Desktop lifting range:65cm-85cm/25.6-33.5inch Load bearing:50kg/110pound This product requires manual assembly, the requiBlue parts are included in the product, there are installation instructions, easy to operate Contain: 【Panel Material】 15mm…
- step=8; action=`click[buy now]`; reward=`0.6`; terminal=`未记录`; parse_error=`False`; observation=Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] CSBYBD Table Sofa Side Table, Adjustable C Shaped End Table with 2-Tier Storage Shelves for Sofa Couch, Mobile Desk with Wheels, Easy Assembly - Blue [SEP] Price: $38.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

Tool receipts：

- receipt=1; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`80.00550512224436`; request={"action": "search[small end table easy assemble blue steel frame under 70]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 1, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B096XNXGJQ [SEP] 15 Inch Modern Side Table, Industrial Small End Table with Thickened Top Plate, Accent Table, Wood Panel and Metal Frame, Nightstand Sofa Table for Living Room, Kitchen, Bedroom, Easy Assembly (Black) [SEP] $39.99 [SEP] B09S8WCJGK [SEP] New TV Tray Table Set Steel Stand End Home Furniture Serve Snack Dinner HOT, High and Narrow, Hallway, Living Room, Metal, Easy Assembly, …
- receipt=2; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`16.977129504084587`; request={"action": "click[b096xnxgjq]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 2, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] 15 Inch Modern Side Table, Industrial Small End Table with Thickened Top Plate, Accent Table, Wood Panel and Metal Frame, Nightstand Sofa Table for Living Room, Kitchen, Bedroom, Easy Assembly (Black) [SEP] Price: $39.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now", "terminal": false}}
- receipt=3; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`5.43949194252491`; request={"action": "click[back to search]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 3, "observation": "WebShop [SEP] Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Search", "terminal": false}}
- receipt=4; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`60.264838859438896`; request={"action": "search[blue coated steel frame end table under 70 easy assembly]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 4, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09J29KTKP [SEP] KOTEK Mobile Sofa Side Table, Height Adjustable C Shaped End Table with Rolling Casters, Portable Snack Table Laptop Desk Coffee Tray for Living Room, Bedroom [SEP] $49.99 [SEP] B09CZ41DST [SEP] Height Adjustable Side Table with Rolling Wheel Bedside Table,Slide Under Bedside End Table Laptop Desk Computer Stand Workstation for Home Office Bedroom (Red) [SEP] $42.99 [SEP] …
- receipt=5; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`17.050351947546005`; request={"action": "click[b09gf9ssqn]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 5, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] CSBYBD Table Sofa Side Table, Adjustable C Shaped End Table with 2-Tier Storage Shelves for Sofa Couch, Mobile Desk with Wheels, Easy Assembly - Blue [SEP] Price: $38.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now", "terminal": false}}
- receipt=6; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`104.19476497918367`; request={"action": "click[description]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 6, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] Home Office Desk Can Be Lifted And LoweBlue Mobile Computer Desk Bedside Table Feature: 100% brand new and high quality . Quantity: 1pcs Color:Blue Weight:16.09Ib Materials:MDF + steel frame Sheet thickness:0.59 in Desktop size:L:80cm×40cm/31.5×15.7inch Desktop lifting range:65cm-85cm/25.6-33.5inch Load bearing:50kg/110pound This product requires manual assembly, the requiBlue parts are included in the product, there are …
- receipt=7; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`12.819693423807621`; request={"action": "click[< prev]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 7, "observation": "Instruction: [SEP] i need to find a small end table that is easy to assemble; pick a blue-coated steel frame that won't rust, and price lower than 70.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] CSBYBD Table Sofa Side Table, Adjustable C Shaped End Table with 2-Tier Storage Shelves for Sofa Couch, Mobile Desk with Wheels, Easy Assembly - Blue [SEP] Price: $38.99 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now", "terminal": false}}
- receipt=8; tool=`webshop.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`27.337574400007725`; request={"action": "click[buy now]", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 8, "observation": "Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B09GF9SSQN [SEP] options [SEP] {} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 0.6 [SEP] Reward Details [SEP] None", "terminal": true}}

FIRST RECORDED ENVIRONMENT-OWNERSHIP FAULT：round=0, graph_revision=7, feedback=accepted add_subgraph at revision 7; execution_error={"type":"AgentRuntimeError","message":"stateful tool 'webshop.environment' requires one graph Agent owner; found 2"} The Director recovered by deleting one owner; the terminal reward was 0.6, so the receipt does not establish a unique semantic cause for the non-success.

## 最小架构适配

保留的 v1 receipt 显示连续 10 次 `parse_error` transition：Executor 输出 JSON action object，而 WebShop 只接受原生 `search[...]` / `click[...]` action。executor action grammar 对自由文本 Canvas contract 具有执行优先级。当前 v4 使用 native validation indices 500..627，并在两题 canary 上完成 2/2 full-chain Stable Zero；Direct 与 AgentGraph success 均为 1/2。旧 v2 native-test 结果明确排除；v3 development 因未传递 SkillFlow `max_action_tokens` 导致本地 Direct 上下文超限，仅保留为失败诊断。以上是接口与数据隔离修正，不是 benchmark-specific workflow template。
