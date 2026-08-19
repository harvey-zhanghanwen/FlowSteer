# AIME-2025 Development（AIME 2026 目标适配） 架构报告

## Stable Zero

- 能力边界：推理 + 有界 calculator/Python execution 能力
- Protocol：开发阶段使用 AIME 2025 官方题目与整数 exact match；Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。
- 固定 validation task：**2**
- Raw receipts：manifest=`artifacts/aime2026_computation_tool_stable_zero/development/run_manifest.json`; paired=`artifacts/aime2026_computation_tool_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/aime2026_computation_tool_stable_zero/development/agentgraph_trajectories.jsonl`
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- optimizer update：**0**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | exact_match |
|---|---:|
| Direct/Simple Baseline | 0.00% |
| AgentGraph | 50.00% |

以上是当前 evidence scope 中 2 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Evidence scope 与协议限制

- Evidence scope：AIME 2025 development canary；不是 AIME 2026 benchmark 成绩
- Protocol：开发阶段使用 AIME 2025 官方题目与整数 exact match；Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。

- 仅 2 题 Stable Zero canary，不是正式 benchmark 估计。
- 可选计算 Tool 未被自然选择时，只能报告 capability 已接线，不能声称 Tool 已验证有效。

### 明确排除的历史结果

- artifacts/aime2026_computation_tool_stable_zero/evaluation：使用 AIME 2026 official test 的旧结果，仅保留为历史诊断，不进入开发指标。


## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | exact_match=0.00% | 开发阶段使用 AIME 2025 官方题目与整数 exact match；Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。 |
| AgentGraph Stable Zero | exact_match=50.00% | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | exact_match=50.00% | current Stable Zero condition; no later architecture version was run |
| Tool/ReAct/Coding-enabled AgentGraph | exact_match=50.00% | declared capability; actual Tool receipts=0 |

同一 receipt 同时代表多个 stage 时不会重复解释为独立实验，也不会把 protocol-separated 条件的差值解释为因果增益。

## Workflow 分布

互斥 `topology_family` 计数（未观察到的合法结构显式记 0）：

- `single`: 0
- `serial_2`: 2
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
- `ReAct`: 0/2 tasks
- `Tool-using`: 0/2 tasks
- `Coding`: 0/2 tasks
- `mixed execution modes`: 0/2 tasks

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

- `deepseek-v4-flash`: 1 Agent nodes
- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-9b-local`: 1 Agent nodes
- `qwen3.5-flash`: 1 Agent nodes

- Model family：Qwen=2, DeepSeek=1, Gemini=0, GPT=1, MiniMax=0, Grok=0, GLM=0, Other=0
- Multi-model workflow 比例：**2/2**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| qwen3.5-9b-local | 2 | 2 | 619 | 0.33 |

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
| deepseek-v4-flash | 1 | 1 | 683 | 6.14 |
| gpt-4o-mini | 1 | 1 | 4350 | 2.72 |
| qwen3.5-9b-local | 1 | 1 | 4432 | 105.53 |
| qwen3.5-flash | 2 | 2 | 6419 | 76.19 |

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`qwen35-9b-aime2025-computation-tool-development-step-000000`；prompt=`agentgraph.director.progressive-subgraph.stage-conditioned-skill.v3`
- Director calls/attempts：**24/24**；tokens=**80820**；latency=**128.04s**



## Tool / ReAct 使用情况

- Tool call：**0**；成功：**0**；失败：**0**
- Tool call task rate：**0/2**
- Tool useful rate：**不可测**；当前 receipt 没有独立的 causal usefulness annotation
- Tool wasted rate：**不可测**；Tool error 单独报告，不能等同于无效信息价值
- Environment transition receipt：**0**
- Coding action receipt：**0**
- AgentGraph 原生 environment action：**0**；invalid action：**0**
- Direct 原生 environment action：**0**；invalid action：**0**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 2 | 619 | 0.33 |
| AgentGraph | 29 | 96704 | 318.62 |


## Exact-schema Tool forced probe（不计入 benchmark）

- Receipt：`artifacts/tool_exact_schema_canary/aime_2026_exact_wire_v3_20260820.json`
- Controls：`diagnostic_only=true`、`forced_probe=true`、`grpo_eligible=false`、`skill_evidence_eligible=false`
- Overall status：`failed`
- StructuredAction schema compliance：`false`
- Tool backend compliance：`true`；successful receipts=`2`
- Model action/termination compliance：`false`
- Observed action sequence：`python_exec → calculator`

该 receipt 只回答 exact `StructuredAction`、真实 backend dispatch 和有界 ReAct termination 是否可执行；不含 evaluator、Ground Truth、benchmark metric、Skill evidence 或训练数据。forced probe 失败不覆盖同条件自然策略成绩，反之亦然。

## Correct Demo

### Correct Demo: `aime-2025:i:01`

- Raw receipts：paired=`artifacts/aime2026_computation_tool_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/aime2026_computation_tool_stable_zero/development/agentgraph_trajectories.jsonl`

- Task：Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.
- Ground Truth：70
- Final Answer：<answer>70</answer>
- Evaluator：`skillflow.protocol-v10.static.integer.v1`; metrics=`{"accuracy": 1.0, "exact_match": 1.0}`
- Trajectory ID：`trajectory_9ae004f72264346853f5eca1`
- Policy version：`qwen35-9b-aime2025-computation-tool-development-step-000000`; prompt=`agentgraph.director.progressive-subgraph.stage-conditioned-skill.v3`; tool=`skillflow.training-tools.computation-react.v1`
- Output Agent：`output`
- AgentGraph: `solver → output`

Agent 配置：

- `output` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Receive the calculated sum and output it as the final answer.
- `solver` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`reasoner`, allowed_tools=`[]`, artifact_type=`text`; contract: Solve the equation: find sum of all integer bases b>9 such that 17_b divides 97_b.

Director atomic edit 序列：`invalid → add_subgraph → add_subgraph → set_output → invalid → invalid → modify_agent → finish → delete_agent → invalid → add_subgraph → modify_agent → invalid → add_subgraph → add_subgraph → delete_agent → add_subgraph → finish → modify_agent → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`invalid`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Invalid \escape: line 1 column 123 (char 122)
- round=`1`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`2`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'solver' cannot declare allowed_tools"}
- round=`3`; action=`set_output`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: action made no graph change; modify an Agent contract/model or another graph field before expecting a new execution
- round=`4`; action=`invalid`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Expecting ',' delimiter: line 1 column 1230 (char 1229)
- round=`5`; action=`invalid`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Invalid \escape: line 1 column 157 (char 156)
- round=`6`; action=`modify_agent`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted modify_agent at revision 5; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'solver' cannot declare allowed_tools"}
- round=`7`; action=`finish`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=cannot finish: execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'solver' cannot declare allowed_tools"}
- round=`8`; action=`delete_agent`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['output']]`; executed=`['output']`; reused=`[]`; feedback=accepted delete_agent at revision 6; execution_result={"output_agent_id":"output","output":"<answer></answer>","executed_agent_ids":["output"],"reused_agent_ids":[],"topology":{"agent_count":1,"relation_count":0,"directed_edge_count":0,"quotient_directed_edge_count":0,"reciprocal_pair_count":0,"component_count":1,"max_depth":1,"structural_depth":1,"max_width":1,"topology_family":"single","topology_motifs":[],"root_agent_ids":["output"],"sink_agent_ids":["output"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"output"},"output_inbox":[],"agent_artifacts":[{"agent_id":"output","model_id":"qwen3.5-flash","role_family":"format",…
- round=`9`; action=`invalid`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: the first JSON object is malformed at character 0: Invalid \escape: line 1 column 160 (char 159)
- round=`10`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: add_subgraph Agent already exists: output
- round=`11`; action=`modify_agent`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: unknown agent_id: solver
- round=`12`; action=`invalid`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=invalid action: add_subgraph relation must contain at least one directed edge
- round=`13`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: add_subgraph Agent already exists: output
- round=`14`; action=`add_subgraph`; graph_revision=`6`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: add_subgraph Agent already exists: output
- round=`15`; action=`delete_agent`; graph_revision=`7`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted delete_agent at revision 7
- round=`16`; action=`add_subgraph`; graph_revision=`11`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 11; execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'solver' during single: provider request failed for vectorengine: HTTP 500"}
- round=`17`; action=`finish`; graph_revision=`11`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=cannot finish: execution_error={"type":"AgentRuntimeError","message":"gateway failed for agent 'solver' during single: provider request failed for vectorengine: HTTP 500"}
- round=`18`; action=`modify_agent`; graph_revision=`12`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['solver'], ['output']]`; executed=`['output', 'solver']`; reused=`[]`; feedback=accepted modify_agent at revision 12; execution_result={"output_agent_id":"output","output":"<answer>70</answer>","executed_agent_ids":["output","solver"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["solver"],"sink_agent_ids":["output"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"output"},"output_inbox":[{"source_agent_id":"solver","target_agent_id":"output","message_type…
- round=`19`; action=`finish`; graph_revision=`12`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['solver'], ['output']]`; executed=`['output', 'solver']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `solver` → `output`; artifact_type=`text`; dependency=Receive the calculated sum and output it as the final answer.; graph_revision=`12`; environment_revision=`None`; tool_receipts=`0`; body=The base-\( b \) notation means: \[ 17_b = 1 \cdot b + 7 = b + 7 \] \[ 97_b = 9 \cdot b + 7 = 9b + 7 \] We need \( b + 7 \) to divide \( 9b + 7 \), with integer \( b > 9 \). Let \[ 9b + 7 = k(b + 7) \] for some integer \( k \). Solve for \( b \): \[ 9b + 7 = kb + 7k \] \[ (9 - k)b = 7k - 7 = 7(k - 1) \] \[ b = \frac{7(k - 1)}{9 - k} \] Since \( b > 9 \) and positive, \( 9 - k \) must have the same sign as \( 7(k - 1) \). Also \( b \) integer. Alternatively, use modular arithmetic: \[ 9b + 7 \eq…
## Wrong Demo

### Wrong Demo: `aime-2025:i:02`

- Raw receipts：paired=`artifacts/aime2026_computation_tool_stable_zero/development/paired_results.jsonl`; trajectory=`artifacts/aime2026_computation_tool_stable_zero/development/agentgraph_trajectories.jsonl`

- Task：On $\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$.
- Ground Truth：588
- Final Answer：<answer>576</answer>
- Evaluator：`skillflow.protocol-v10.static.integer.v1`; metrics=`{"accuracy": 0.0, "exact_match": 0.0}`
- Trajectory ID：`trajectory_29835d6b1aa2f622f73882b2`
- Policy version：`qwen35-9b-aime2025-computation-tool-development-step-000000`; prompt=`agentgraph.director.progressive-subgraph.stage-conditioned-skill.v3`; tool=`skillflow.training-tools.computation-react.v1`
- Output Agent：`output`
- AgentGraph: `solver → output`

Agent 配置：

- `output` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Receive the final computed area from 'solver' and output it exactly as requested in the terminal protocol.
- `solver` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`operator`, allowed_tools=`[]`, artifact_type=`text`; contract: Execute geometric and algebraic calculations. Use tools to compute lengths, intersections, or areas. Check for collinearity issues that might make quadrilateral DEGF degenerate; if so, consider alternative interpretations or assume non-degeneracy as implied by the area 288. Route clear text artifacts to 'finish'.

Director atomic edit 序列：`add_subgraph → add_subgraph → modify_agent → finish`

Progressive Canvas turn receipts：

- round=`0`; action=`add_subgraph`; graph_revision=`0`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent
- round=`1`; action=`add_subgraph`; graph_revision=`4`; receipt_verified=`True`; communication_condition=`not_executed`; blocks=`[]`; executed=`[]`; reused=`[]`; feedback=accepted add_subgraph at revision 4; execution_error={"type":"AgentRuntimeError","message":"reasoning agent 'solver' cannot declare allowed_tools"}
- round=`2`; action=`modify_agent`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['solver'], ['output']]`; executed=`['output', 'solver']`; reused=`[]`; feedback=accepted modify_agent at revision 5; execution_result={"output_agent_id":"output","output":"<answer>576</answer>","executed_agent_ids":["output","solver"],"reused_agent_ids":[],"topology":{"agent_count":2,"relation_count":1,"directed_edge_count":1,"quotient_directed_edge_count":1,"reciprocal_pair_count":0,"component_count":2,"max_depth":2,"structural_depth":2,"max_width":1,"topology_family":"serial_2","topology_motifs":["serial_2"],"root_agent_ids":["solver"],"sink_agent_ids":["output"],"root_component_count":1,"sink_component_count":1,"fan_in_agent_ids":[],"fan_out_agent_ids":[],"output_agent_id":"output"},"output_inbox":[{"source_agent_id":"solver","target_agent_id":"output","message_type…
- round=`3`; action=`finish`; graph_revision=`5`; receipt_verified=`True`; communication_condition=`normal`; blocks=`[['solver'], ['output']]`; executed=`['output', 'solver']`; reused=`[]`; feedback=workflow finished

实际 CommunicationEnvelope：

- `solver` → `output`; artifact_type=`text`; dependency=Receive the final computed area from 'solver' and output it exactly as requested in the terminal protocol.; graph_revision=`5`; environment_revision=`None`; tool_receipts=`0`; body=To find the area of the heptagon $AFNBCEM$, we first analyze the geometry of the given points and the specific quadrilateral $DEGF$. **1. Analyze the positions of points on the sides:** * **Side $AB$:** Points $A, D, E, B$ are collinear in that order. * $AD = 4$ * $DE = 16$ * $EB = 8$ * Total length $AB = AD + DE + EB = 4 + 16 + 8 = 28$. * Let's set up a coordinate system or vector representation along line $AB$. Let $A$ be the origin for this line segment. * $A = 0$ * $D = 4$ * $E = 4 + 16 = 2…

FIRST RECORDED CANVAS/RUNTIME FAULT：round=0, graph_revision=0, feedback=edit rejected: Output Agent must be a distinct Format Agent with role_family='format'; keep semantic-answer computation in its upstream Agent This fault was later recovered; the receipt does not prove it caused the wrong terminal answer.
