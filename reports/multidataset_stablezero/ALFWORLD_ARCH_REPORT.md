# ALFWorld 架构报告

## Stable Zero

- 能力边界：request-scoped SkillFlow/RAGEN environment ReAct
- Protocol：Direct 与 AgentGraph 使用相同原生游戏、task lock、50-step budget 和 evaluator。
- 固定 validation task：**2**
- 显式 FINISH：**2/2**
- 有效原生 evaluator receipt：**2/2**
- `STABLE_ZERO = PASS`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | success |
|---|---:|
| Direct/Simple Baseline | 50.00% |
| AgentGraph | 100.00% |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

- `serial_2`: 1
- `single`: 1

- 平均 structural depth：**1.50**
- 平均 effective dependency depth：**1.50**
- 平均 Agent 数：**1.50**
- 平均 relation 数：**0.50**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-flash`: 2 Agent nodes

- Model family：GPT=1, Qwen=2
- Multi-model workflow 比例：**0/2**

## Tool / ReAct 使用情况

- Tool call：**10**；成功：**10**；失败：**0**
- AgentGraph 原生 environment action：**10**；invalid action：**0**
- Direct 原生 environment action：**54**；invalid action：**1**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | 54 | 40463 | 19.68 |
| AgentGraph | 15 | 32837 | 154.96 |

## Correct Demo

### Correct Demo: `alfworld:train:00000`

- Task：put a handtowel in garbagecan.
- Ground Truth：environment_success
- Final Answer：You move the handtowel 1 to the garbagecan 1.
- Evaluator: `{"environment_return": 10.0, "steps": 4.0, "success": 1.0, "terminal": 1.0}`
- AgentGraph: `lf_router`

Agent 配置：

- `lf_router` — model=`gpt-4o-mini`, execution_mode=`react`, role_family=`operator`, allowed_tools=`['alfworld.environment']`, artifact_type=`react_step`; contract: Evaluate current ALFWorld state and determine the next executable action to reach the goal state (handtowel in garbagecan).

Director atomic edit 序列：`add_subgraph → finish`

原生 ReAct trace（4 个 action）：`go to handtowelholder 1 → take handtowel 1 from handtowelholder 1 → go to garbagecan 1 → move handtowel 1 to garbagecan 1`

Tool receipts：

- tool=`alfworld.environment`, status=`completed`; request={"action": "go to handtowelholder 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "take handtowel 1 from handtowelholder 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "go to garbagecan 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "move handtowel 1 to garbagecan 1", "arguments": {}}

### Correct Demo: `alfworld:train:00006`

- Task：examine the pillow with the desklamp.
- Ground Truth：environment_success
- Final Answer：<answer>You turn on the desklamp 1.</answer>
- Evaluator: `{"environment_return": 10.0, "steps": 6.0, "success": 1.0, "terminal": 1.0}`
- AgentGraph: `env_diagnostic → report_final`

Agent 配置：

- `env_diagnostic` — model=`qwen3.5-flash`, execution_mode=`react`, role_family=`exploration`, allowed_tools=`['alfworld.environment']`, artifact_type=`observation_text`; contract: An ALFWorld diagnostic agent that executes a single step in the given environment to observe the pillow and desklamp states.
- `report_final` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Read the diagnostic observation and output the final answer.

Director atomic edit 序列：`add_subgraph → finish`

Output Agent 实际 inbox：

- `env_diagnostic` → `report_final`; artifact_type=`observation_text`; body=You turn on the desklamp 1.

原生 ReAct trace（6 个 action）：`go to bed 1 → take pillow 1 from bed 1 → examine pillow 1 → go to shelf 1 → go to dresser 1 → use desklamp 1`

Tool receipts：

- tool=`alfworld.environment`, status=`completed`; request={"action": "go to bed 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "take pillow 1 from bed 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "examine pillow 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "go to shelf 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "go to dresser 1", "arguments": {}}
- tool=`alfworld.environment`, status=`completed`; request={"action": "use desklamp 1", "arguments": {}}
## Wrong Demo

该 2 题样本中没有 AgentGraph 错例；不进行虚构。


## 最小架构适配

保留的 v1 receipt 中有一个 graph 在缺少 environment replay trace 时到达 FINISH，因此被拒绝。现在 FlowSteer terminal validation 对 interactive condition 要求且仅要求一个 ReAct environment actor，同时 model、role、relation 与 topology 仍由 Director 决定。v2 AgentGraph 完成了两个固定游戏；第二个游戏用 6-step 原生 episode 完成，而 Direct 达到 50-step limit。
