# AIME-2025 Development（AIME 2026 目标适配） 架构报告

## Stable Zero

- 能力边界：推理 + 有界 calculator/Python execution 能力
- Protocol：开发阶段使用 AIME 2025 官方题目与整数 exact match；Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。
- 固定 validation task：**2**
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

- `serial_2`: 2

- 平均 structural depth：**2.00**
- 平均 effective dependency depth：**2.00**
- 平均 Agent 数：**2.00**
- 平均 relation 数：**1.00**
- 平均 parallel execution width：**1.00**

## Model 使用情况

- `deepseek-v4-flash`: 1 Agent nodes
- `gpt-4o-mini`: 1 Agent nodes
- `qwen3.5-9b-local`: 1 Agent nodes
- `qwen3.5-flash`: 1 Agent nodes

- Model family：DeepSeek=1, GPT=1, Qwen=2
- Multi-model workflow 比例：**2/2**

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

- Task：Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.
- Ground Truth：70
- Final Answer：<answer>70</answer>
- Evaluator: `{"accuracy": 1.0, "exact_match": 1.0}`
- AgentGraph: `solver → output`

Agent 配置：

- `output` — model=`qwen3.5-flash`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Receive the calculated sum and output it as the final answer.
- `solver` — model=`deepseek-v4-flash`, execution_mode=`reasoning`, role_family=`reasoner`, allowed_tools=`[]`, artifact_type=`text`; contract: Solve the equation: find sum of all integer bases b>9 such that 17_b divides 97_b.

Director atomic edit 序列：`? → add_subgraph → add_subgraph → set_output → ? → ? → modify_agent → finish → delete_agent → ? → add_subgraph → modify_agent → ? → add_subgraph → add_subgraph → delete_agent → add_subgraph → finish → modify_agent → finish`

Output Agent 实际 inbox：

- `solver` → `output`; artifact_type=`text`; body=The base-\( b \) notation means: \[ 17_b = 1 \cdot b + 7 = b + 7 \] \[ 97_b = 9 \cdot b + 7 = 9b + 7 \] We need \( b + 7 \) to divide \( 9b + 7 \), with integer \( b > 9 \). Let \[ 9b + 7 = k(b + 7) \] for some integer \( k \). Solve for \( b \): \[ 9b + 7 = kb + 7k \] \[ (9 - k)b = 7k - 7 = 7(k - 1) \] \[ b = \frac{7(k - 1)}{9 - k} \] Since \( b > 9 \) and positive, \( 9 - k \) must have the same sign as \( 7(k - 1) \). Also \( b \) integer. Alternatively, use modular arithmetic: \[ 9b + 7 \eq…
## Wrong Demo

### Wrong Demo: `aime-2025:i:02`

- Task：On $\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$.
- Ground Truth：588
- Final Answer：<answer>576</answer>
- Evaluator: `{"accuracy": 0.0, "exact_match": 0.0}`
- AgentGraph: `solver → output`

Agent 配置：

- `output` — model=`gpt-4o-mini`, execution_mode=`reasoning`, role_family=`format`, allowed_tools=`[]`, artifact_type=`text`; contract: Receive the final computed area from 'solver' and output it exactly as requested in the terminal protocol.
- `solver` — model=`qwen3.5-9b-local`, execution_mode=`reasoning`, role_family=`operator`, allowed_tools=`[]`, artifact_type=`text`; contract: Execute geometric and algebraic calculations. Use tools to compute lengths, intersections, or areas. Check for collinearity issues that might make quadrilateral DEGF degenerate; if so, consider alternative interpretations or assume non-degeneracy as implied by the area 288. Route clear text artifacts to 'finish'.

Director atomic edit 序列：`add_subgraph → add_subgraph → modify_agent → finish`

Output Agent 实际 inbox：

- `solver` → `output`; artifact_type=`text`; body=To find the area of the heptagon $AFNBCEM$, we first analyze the geometry of the given points and the specific quadrilateral $DEGF$. **1. Analyze the positions of points on the sides:** * **Side $AB$:** Points $A, D, E, B$ are collinear in that order. * $AD = 4$ * $DE = 16$ * $EB = 8$ * Total length $AB = AD + DE + EB = 4 + 16 + 8 = 28$. * Let's set up a coordinate system or vector representation along line $AB$. Let $A$ be the origin for this line segment. * $A = 0$ * $D = 4$ * $E = 4 + 16 = 2…

FIRST ERROR：terminal evaluator metric 未达到满分；现有 receipt 不能证明更窄的原因。
