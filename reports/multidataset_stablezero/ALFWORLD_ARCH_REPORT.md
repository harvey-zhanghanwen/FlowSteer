# ALFWorld 架构报告

## Stable Zero

- 能力边界：request-scoped SkillFlow/RAGEN environment ReAct
- Protocol：Direct 与 AgentGraph 使用相同原生游戏、task lock、50-step budget 和 evaluator。
- 固定 validation task：**2**
- Raw receipts：manifest=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/run_manifest.json`; paired=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/paired_results.jsonl`; trajectory=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl`
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- optimizer update：**0**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | success |
|---|---:|
| Direct/Simple Baseline | 50.00% |
| AgentGraph | 100.00% |

以上是当前 evidence scope 中 2 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Evidence scope 与协议限制

- Evidence scope：fixed validation canary
- Protocol：Direct 与 AgentGraph 使用相同原生游戏、task lock、50-step budget 和 evaluator。

- 无额外限制记录。

### 明确排除的历史结果

- 无需隔离的旧结果。


## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | success=50.00% | Direct 与 AgentGraph 使用相同原生游戏、task lock、50-step budget 和 evaluator。 |
| AgentGraph Stable Zero | success=100.00% | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | success=100.00% | current v2 condition after receipt-driven minimal adaptation |
| Tool/ReAct/Coding-enabled AgentGraph | success=100.00% | native environment replay; actual actions=10 |

同一 receipt 同时代表多个 stage 时不会重复解释为独立实验，也不会把 protocol-separated 条件的差值解释为因果增益。

## Workflow 分布

互斥 `topology_family` 计数（未观察到的合法结构显式记 0）：

- `single`: 1
- `serial_2`: 1
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
- `ReAct`: 2/2 tasks
- `Tool-using`: 2/2 tasks
- `Coding`: 0/2 tasks
- `mixed execution modes`: 1/2 tasks

- 平均 structural depth：**1.50**
- 平均 effective dependency depth：**1.50**
- 平均 Agent 数：**1.50**
- 平均 relation 数：**0.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-flash`: 2 Agent nodes

- Model family：Qwen=2, DeepSeek=0, Gemini=0, GPT=1, MiniMax=0, Grok=0, GLM=0, Other=0
- Multi-model workflow 比例：**0/2**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| qwen3.5-9b-local | 2 | 2 | 1597 | 0.95 |

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| gpt-4o-mini | 4 | 4 | 2769 | 7.02 |
| qwen3.5-flash | 7 | 7 | 22307 | 138.68 |

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`qwen35-9b-alfworld-ragen-environment-stable-zero-step-000000`；prompt=`agentgraph.director.progressive_subgraph.v1`
- Director calls/attempts：**4/4**；tokens=**7761**；latency=**9.26s**



## Tool / ReAct 使用情况

- Tool call：**10**；成功：**10**；失败：**0**
- Tool call task rate：**2/2**
- Tool useful rate：**不可测**；当前 receipt 没有独立的 causal usefulness annotation
- Tool wasted rate：**不可测**；Tool error 单独报告，不能等同于无效信息价值
- Environment transition receipt：**10**
- Coding action receipt：**0**
- AgentGraph 原生 environment action：**10**；invalid action：**0**
- Direct 原生 environment action：**54**；invalid action：**1**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 54 | 40463 | 19.68 |
| AgentGraph | 15 | 32837 | 154.96 |

## Correct Demo

### Correct Demo: `alfworld:train:00000`

- Raw receipts：paired=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/paired_results.jsonl`; trajectory=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl`

- Task：put a handtowel in garbagecan.
- Ground Truth：environment_success
- Final Answer：You move the handtowel 1 to the garbagecan 1.
- Evaluator：`skillflow.ragen_adapter.v2`; metrics=`{"environment_return": 10.0, "steps": 4.0, "success": 1.0, "terminal": 1.0}`
- Trajectory ID：`trajectory_e12ed08ce175fb615e29ea4e`
- Policy version：`qwen35-9b-alfworld-ragen-environment-stable-zero-step-000000`; prompt=`agentgraph.director.progressive_subgraph.v1`; tool=`skillflow.ragen-environment-react.required-actor.v2`
- Output Agent：`lf_router`
- AgentGraph: `lf_router`

Agent 配置：

- `lf_router` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`operator`, allowed_tools=`['alfworld.environment']`, artifact_type=`react_step`; contract: Evaluate current ALFWorld state and determine the next executable action to reach the goal state (handtowel in garbagecan).

Director atomic edit 序列：`add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`2`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['lf_router']]`; executed=`['lf_router']`; reused=`[]`; feedback=accepted add_subgraph at revision 2; execution_result={"output_agent_id":"lf_router","output":"You move the handtowel 1 to the garbagecan 1.","executed_agent_ids":["lf_router"],"reused_agent_ids":[],"topology":{"agent_count":1,"relation_count":0,"directed_edge_count":0,"quotient_directed_edge_count":0,"reciprocal_pair_count":0,"component_count":1,"max_depth":1,"structural_depth":1,"max_width":1,"topology_family":"single","topology_motifs":[],"root_agent_ids":["lf_router"],"sink_agent_ids":["lf_router"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"lf_router"},"output_inbox":[],"agent_artifacts":[{"agent_id":"lf_router","mod…
- round=`1`; action=`finish`; graph_revision=`2`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['lf_router']]`; executed=`['lf_router']`; reused=`[]`; feedback=workflow finished

原生 environment ReAct trace（4 个 action）：`go to handtowelholder 1 → take handtowel 1 from handtowelholder 1 → go to garbagecan 1 → move handtowel 1 to garbagecan 1`
- step=1; action=`go to handtowelholder 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a countertop 1, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1. Your task is to: put a handtowel in garbagecan.
- step=2; action=`take handtowel 1 from handtowelholder 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at handtowelholder 1. On the handtowelholder 1, you see a handtowel 1.
- step=3; action=`go to garbagecan 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You pick up the handtowel 1 from the handtowelholder 1.
- step=4; action=`move handtowel 1 to garbagecan 1`; reward=`10.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at garbagecan 1. On the garbagecan 1, you see nothing.

Tool receipts：

- receipt=1; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`32.31111727654934`; request={"action": "go to handtowelholder 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 1, "observation": "You arrive at handtowelholder 1. On the handtowelholder 1, you see a handtowel 1.", "terminal": false}}
- receipt=2; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`31.923177652060986`; request={"action": "take handtowel 1 from handtowelholder 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 2, "observation": "You pick up the handtowel 1 from the handtowelholder 1.", "terminal": false}}
- receipt=3; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`33.98862108588219`; request={"action": "go to garbagecan 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 3, "observation": "You arrive at garbagecan 1. On the garbagecan 1, you see nothing.", "terminal": false}}
- receipt=4; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`31.112930737435818`; request={"action": "move handtowel 1 to garbagecan 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 4, "observation": "You move the handtowel 1 to the garbagecan 1.", "terminal": true}}

### Correct Demo: `alfworld:train:00006`

- Raw receipts：paired=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/paired_results.jsonl`; trajectory=`artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl`

- Task：examine the pillow with the desklamp.
- Ground Truth：environment_success
- Final Answer：<answer>You turn on the desklamp 1.</answer>
- Evaluator：`skillflow.ragen_adapter.v2`; metrics=`{"environment_return": 10.0, "steps": 6.0, "success": 1.0, "terminal": 1.0}`
- Trajectory ID：`trajectory_63d47d45d1faf296cb9ad967`
- Policy version：`qwen35-9b-alfworld-ragen-environment-stable-zero-step-000000`; prompt=`agentgraph.director.progressive_subgraph.v1`; tool=`skillflow.ragen-environment-react.required-actor.v2`
- Output Agent：`report_final`
- AgentGraph: `env_diagnostic → report_final`

Agent 配置：

- `env_diagnostic` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`exploration`, allowed_tools=`['alfworld.environment']`, artifact_type=`observation_text`; contract: An ALFWorld diagnostic agent that executes a single step in the given environment to observe the pillow and desklamp states.
- `report_final` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Read the diagnostic observation and output the final answer.

Director atomic edit 序列：`add_subgraph → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['env_diagnostic'], ['report_final']]`; executed=`['env_diagnostic', 'report_final']`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_result={"output_agent_id":"report_final","output":"<answer>You turn on the desklamp 1.</answer>","executed_agent_ids":["env_diagnostic","report_final"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["env_diagnostic"],"sink_agent_ids":["report_final"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"report_final"},"output_inbox":[{"so…
- round=`1`; action=`finish`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['env_diagnostic'], ['report_final']]`; executed=`['env_diagnostic', 'report_final']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `env_diagnostic` → `report_final`; artifact_type=`observation_text`; dependency=Read the diagnostic observation and output the final answer.; graph_revision=`4`; environment_revision=`6`; tool_receipts=`6`; body=You turn on the desklamp 1.

原生 environment ReAct trace（6 个 action）：`go to bed 1 → take pillow 1 from bed 1 → examine pillow 1 → go to shelf 1 → go to dresser 1 → use desklamp 1`
- step=1; action=`go to bed 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1. Your task is to: examine the pillow with the desklamp.
- step=2; action=`take pillow 1 from bed 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at bed 1. On the bed 1, you see a book 3, a book 2, a book 1, a laptop 1, a pillow 2, and a pillow 1.
- step=3; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You pick up the pillow 1 from the bed 1.
- step=4; action=`go to shelf 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=5; action=`go to dresser 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at shelf 1. On the shelf 1, you see nothing.
- step=6; action=`use desklamp 1`; reward=`10.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at dresser 1. On the dresser 1, you see a cd 1, a cellphone 2, a creditcard 1, a desklamp 1, a mug 1, a pen 2, a pen 1, and a pencil 1.

Tool receipts：

- receipt=1; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`66.0429298877716`; request={"action": "go to bed 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 1, "observation": "You arrive at bed 1. On the bed 1, you see a book 3, a book 2, a book 1, a laptop 1, a pillow 2, and a pillow 1.", "terminal": false}}
- receipt=2; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`51.8827848136425`; request={"action": "take pillow 1 from bed 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 2, "observation": "You pick up the pillow 1 from the bed 1.", "terminal": false}}
- receipt=3; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`59.65535342693329`; request={"action": "examine pillow 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 3, "observation": "There's nothing special about pillow 1.", "terminal": false}}
- receipt=4; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`64.61669784039259`; request={"action": "go to shelf 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 4, "observation": "You arrive at shelf 1. On the shelf 1, you see nothing.", "terminal": false}}
- receipt=5; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`56.509800255298615`; request={"action": "go to dresser 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 5, "observation": "You arrive at dresser 1. On the dresser 1, you see a cd 1, a cellphone 2, a creditcard 1, a desklamp 1, a mug 1, a pen 2, a pen 1, and a pencil 1.", "terminal": false}}
- receipt=6; tool=`alfworld.environment`; version=`skillflow.ragen_adapter.v2`; status=`completed`; latency_ms=`51.857282407581806`; request={"action": "use desklamp 1", "arguments": {}}; result={"completed": true, "value": {"environment_revision": 6, "observation": "You turn on the desklamp 1.", "terminal": true}}
## Wrong / Failure Demo

当前 AgentGraph 2 题均成功；以下保留同一 fixed task 的真实 Direct failure contrast，不把它计为 AgentGraph wrong case。

### Direct Failure Contrast: `alfworld:train:00006`

- Task：examine the pillow with the desklamp.
- Ground Truth：environment_success
- Direct Final Answer：<action>examine pillow 1</action>
- Evaluator：`{"environment_return": 0.0, "steps": 50.0, "success": 0.0, "terminal": 0.0}`
- Failure classification：`ReAct action selection / state tracking / stopping`

Direct ReAct trace（50 个 action）：`go to bed 1 → take pillow 1 from bed 1 → examine pillow 1 → <INVALID> → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1 → examine pillow 1`
- step=1; action=`go to bed 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1. Your task is to: examine the pillow with the desklamp.
- step=2; action=`take pillow 1 from bed 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You arrive at bed 1. On the bed 1, you see a book 3, a book 2, a book 1, a laptop 1, a pillow 2, and a pillow 1.
- step=3; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=You pick up the pillow 1 from the bed 1.
- step=4; action=`<INVALID>`; reward=`0.0`; terminal=`未记录`; parse_error=`True`; observation=There's nothing special about pillow 1.
- step=5; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=6; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=7; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=8; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=9; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=10; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=11; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=12; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=13; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=14; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=15; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=16; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=17; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=18; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=19; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=20; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=21; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=22; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=23; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=24; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=25; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=26; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=27; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=28; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=29; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=30; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=31; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=32; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=33; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=34; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=35; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=36; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=37; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=38; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=39; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=40; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=41; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=42; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=43; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=44; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=45; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=46; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=47; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=48; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=49; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.
- step=50; action=`examine pillow 1`; reward=`0.0`; terminal=`未记录`; parse_error=`False`; observation=There's nothing special about pillow 1.

FIRST ERROR：Direct arm 在 zero-based step=3（第 4 个 environment action）首次产生 <INVALID>，随后重复 examine pillow 1，最终触发 50-step environment_step_limit；同题 v2 AgentGraph 用 6 个原生 action 成功。


## 最小架构适配

保留的 v1 receipt 中有一个 graph 在缺少 environment replay trace 时到达 FINISH，因此被拒绝。现在 FlowSteer terminal validation 对 interactive condition 要求且仅要求一个 ReAct environment actor，同时 model、role、relation 与 topology 仍由 Director 决定。v2 AgentGraph 完成了两个固定游戏；第二个游戏用 6-step 原生 episode 完成，而 Direct 达到 50-step limit。
