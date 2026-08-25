# ALFWorld 初版适配报告

## 1. Architecture Completion Report

ALFWorld Initial Adapter v1 已完成。该版本只在统一架构外接入 Dataset /
Environment Adapter、`alfworld.act(command)` Tool、task-scoped environment
runtime、native evaluator 和 paired evaluation；没有修改统一 AgentGraph 的
Agent 定义、Canvas action space 或固定任何具身任务 workflow。

端到端链路已经由真实 environment receipt 验证：

```text
ALFWorld task
  -> Qwen3.5-9B Flow-Director
  -> progressive Canvas editing
  -> current AgentGraph execution
  -> Agent communication artifact
  -> alfworld.act(command)
  -> shared task-scoped world state
  -> native observation / admissible_commands
  -> official terminal won / episode score
  -> evaluator receipt / trajectory
  -> FINISH or max_rounds
```

完成并验证的模块：

- protocol-v10 Dataset Adapter：独立 train preflight、完整 `valid_seen=140`、
  完整 `valid_unseen=134`；
- SkillFlow ALFWorld Tool schema：`resource_id="alfworld"`、`act`、
  `arguments={"command": native_command}`；
- 每条 rollout 一个独立 session，Canvas 多次执行共享一个串行 world state；
- SkillFlow 20-turn ReAct policy budget 与官方 TextWorld 50-step hard limit；
- 原生 terminal `won` Success Rate evaluator 和 `episode_score`；
- Direct / AgentGraph 同 task、environment、action budget、model/tool condition、
  evaluator 的 paired evaluation；
- 完整 environment transition、Tool receipt、Agent communication、Canvas
  feedback、trajectory 和 Wrong Demo；
- `max_rounds` trajectory 的 complete native replay：不再丢失已执行 action、
  observation、score 或 terminal state。

保留但未启用的接口：GRPO、LoRA、backward、optimizer update、MACE、Bayesian
posterior、Skill retrieval/injection/evolution。本轮全部为零更新。

已知限制：当前 model-admissible action mask 尚未把
`execution_mode=react + allowed_tools=[alfworld]`、唯一 stateful Tool owner、
Tool owner 不得进入 reciprocal block 编码成联合参数约束；这些合法性约束目前
由 runtime feedback 执行。因此 Director 在大量 episode 中先生成非法 execution
profile，再消耗 Canvas turn 修复，最终造成较多 `max_rounds`。

Stable Zero 结论：独立 train-split canary 已完整通过。Direct 与 AgentGraph
均成功完成 `put a handtowel in garbagecan.`，环境结果均为 `won=true`、
`episode_score=1`、4 个原生 action、0 invalid、0 repeated，并保存了显式
`FINISH` 和完整 trajectory。因此初版 adapter 达到受限 Stable Zero；正式分数
仍以完整官方 split 为准。

## 2. 实现来源与兼容边界

实现优先级为项目 MD、SkillFlow 实际 ALFWorld 源码、官方 ALFWorld、FlowSteer。
完整逐项映射见
[source map](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/docs/ALFWORLD_INITIAL_ADAPTER_V1_SOURCE_MAP.md)。

| 本地边界 | 上游来源 | 状态 |
|---|---|---|
| Dataset / split / pinned game identity | SkillFlow protocol v10 与官方 ALFWorld inventory | 必要薄适配 |
| reset、step、observation、admissible actions、state transition、terminal `won` | SkillFlow official bridge / RAGEN adapter / `AlfredTWEnv` | 直接复用语义 |
| `alfworld.act(command)` | SkillFlow public embodied Tool contract | 直接复用接口 |
| progressive Canvas、execute-on-edit、feedback、trajectory、`FINISH` | FlowSteer | 直接复用 |
| free-text Agent contract、Director 自主 Agent/model/relation/Output Agent/topology | 项目 MD 与统一 AgentGraph | 现有 core，不变 |
| task-scoped session 与统一 runtime/evaluator receipt 转换 | 本项目 | 必要薄适配 |

没有预设 `Navigator`、`Manipulator`、`Planner`、`Verifier`，没有要求固定 chain、
parallel 或 reciprocal topology。ReAct 只作为单个 Agent 的 execution mode。

## 3. 正式 evaluation 结果

Primary metric 为官方 environment terminal Success Rate。所有分母均为完整官方
split，所有 episode 均 evaluator-valid；Agent 自述和 LLM judge 不参与 reward。

| Official split | Condition | Success / Total | SR | Episode score（sum / mean） | Evaluator valid |
|---|---|---:|---:|---:|---:|
| `valid_seen` | Qwen3.5-9B Direct | 43 / 140 | 30.71% | 43 / 0.3071 | 140 / 140 |
| `valid_seen` | AgentGraph | 46 / 140 | 32.86% | 46 / 0.3286 | 140 / 140 |
| `valid_unseen` | Qwen3.5-9B Direct | 28 / 134 | 20.90% | 28 / 0.2090 | 134 / 134 |
| `valid_unseen` | AgentGraph | 38 / 134 | 28.36% | 38 / 0.2836 | 134 / 134 |

- `valid_seen`：AgentGraph 相对 Direct 为 **+2.14 percentage points**。
- `valid_unseen`：AgentGraph 相对 Direct 为 **+7.46 percentage points**。
- `valid_seen` AgentGraph：`FINISH=46`、`max_rounds=94`、episode-level
  operational/evaluator failure `=0`。
- `valid_unseen` AgentGraph：`FINISH=38`、`max_rounds=96`、episode-level
  operational/evaluator failure `=0`。
- `valid_unseen` 曾有 3 次 SGLang connection failure；三条均已在相同冻结条件下
  checkpoint-resume 成功，当前 unresolved failure 为 0，历史 attempt receipt 保留。

逐 split 机器可读报告：

- [valid_seen JSON](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_seen_report.json)
- [valid_seen Markdown](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_seen_report.md)
- [valid_unseen JSON](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_unseen_report.json)
- [valid_unseen Markdown](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_unseen_report.md)

## 4. Environment telemetry

| Split / Condition | Policy turns | Native actions | Parse errors | Invalid | No effect | Immediate repeats | Terminal episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| seen Direct | 2,331 | 2,250 | 81 | 0 | 0 | 134 | 43 |
| seen AgentGraph | 2,259 | 2,176 | 83 | 0 | 0 | 114 | 46 |
| unseen Direct | 2,354 | 2,227 | 127 | 0 | 0 | 204 | 28 |
| unseen AgentGraph | 339 | 333 | 6 | 0 | 0 | 9 | 38 |

`invalid=0` 表示未把不可执行命令提交给 environment；不可解析输出单独计入
`parse errors`。`repeated` 只统计相邻完全相同 action，不包含 `A -> B -> A -> B`
oscillation。

首个可观察 failure layer：

- seen：`tool_interface=87`、`director_canvas=7`；
- unseen：`tool_interface=86`、`director_canvas=10`。

这些是 trajectory 内可恢复或终局前的失败层，不等同于 episode-level provider /
environment operational failure。

## 5. 自然 AgentGraph topology

`valid_seen` 的最终/evaluated topology：

```text
empty 6, single 58, serial_2 31, serial_3_plus 5,
parallel 18, fan_in 13, mixed 6, reciprocal 3
```

`valid_unseen`：

```text
empty 1, single 45, serial_2 39, serial_3_plus 4,
parallel 25, fan_in 13, fan_out 2, mixed 4, reciprocal 1
```

Director 确实生成了 chain、parallel、fan-in、fan-out、mixed 和 reciprocal graph，
说明 adapter 没有把 search space 固定为链式结构。但在本初版中，复杂 topology
episode 多数先违反 stateful Tool ownership / execution-mode legality，因此不能从
当前结果推断复杂 topology 本身无效，也不应根据测试集失败硬编码固定 workflow。

## 6. 代表性 Wrong Demo

### 6.1 `valid_seen:00026` — Tool capability 未绑定

Task：`heat some tomato and put it in fridge.`

Director 添加 React Agent，但没有设置结构化 `allowed_tools=["alfworld"]`。
runtime 首轮明确返回：

```text
environment Agent must allow exactly its request-scoped environment tool
```

后续 Director 把 `allowed_tools=[:alfworld]` 写入 free-text contract，而不是 Agent
的结构化 Tool capability。最终虽形成 3-Agent `serial_3_plus`，environment action
仍为 0，`score=0`，终止于 `max_rounds`。首个 failure layer 为
`tool_interface`。

### 6.2 `valid_seen:00007` — object grounding 错误

Task：`put a cool apple in microwave.`

Tool owner 修复后，environment 正常执行。关键 transition：

```text
open fridge 1
-> fridge contains bowl 2, bowl 1, egg 1, potato 1

take potato 1 from fridge 1
-> cool potato 1 with fridge 1
-> move potato 1 to microwave 1
```

任务目标为 apple，但 action policy 在检索未完成时改为处理 potato。20 个 action
均合法，仍因 entity grounding 错误得到 `score=0`。后续 Canvas 编辑又删除已执行
节点，terminal graph 为 `empty`。

### 6.3 `valid_seen:00009` — model catalog 与目标实体错误

Task：`clean some tomato and put it in countertop.`

Director turn 0 使用 catalog 外的 `model_id="via-hotswap"`，Canvas 以
`unknown_model_id` 拒绝。后续 environment action 又转向 `fork 2`：

```text
take fork 2 from sinkbasin 1
-> clean fork 2 with sinkbasin 1
-> sinkbasin / countertop oscillation
```

目标 tomato 未完成，最终 `20 actions / score 0 / max_rounds`。首个 failure layer
为 `director_canvas`。

### 6.4 `valid_unseen:00000` — stateful Tool owner 缺失

Task：`cool some lettuce and put it in countertop.`

React Agent 没有结构化 `alfworld` Tool ownership，turn 0 即收到 request-scoped
Tool 错误。最终为 3-Agent `parallel`、0 relation、0 environment action、
`score=0 / max_rounds`。

### 6.5 `valid_unseen:00038` — 无效 Canvas 修复

Task：`clean some knife and put it in countertop.`

Director 对 Agent 重复设置相同 model，Canvas 返回 `action made no graph change`；
随后又给 reasoning Agent 添加 `alfworld` Tool，runtime 拒绝该 execution profile。
最终两个节点同时成为 stateful Tool owner，违反单一 owner 约束，environment action
仍为 0。

### 6.6 `valid_unseen:00128` — pseudo-action 与多 Tool owner

Task：`clean some pan and put it in countertop.`

reasoning Agent 先输出非原生 pseudo-action `move_to(pan)`；新增 React Agent 又缺少
Tool 绑定。Director 一度修复一个 owner，随后再次增加第二个 `alfworld` owner。
最终为 3-Agent `mixed`，environment ledger 仍为 0 action、`score=0 / max_rounds`。

## 7. 结论与下一步边界

本轮目标已经完成：ALFWorld 只新增 task/environment adapter，没有改变统一
orchestration core；Stable Zero、完整 official `valid_seen`、完整 official
`valid_unseen` 和 Wrong Demo 报告均有真实 receipt。

当前最明确的架构瓶颈不是 evaluator 或 Agent communication 丢失，而是 Director
Canvas action 的联合参数合法性：React execution mode、唯一 stateful Tool owner、
结构化 Tool capability 和 reciprocal relation 之间尚未进入 action mask 的联合
约束。object grounding、搜索 oscillation 和 no-progress recovery 则属于后续 policy /
Skill 学习问题。本轮按要求不根据 evaluation error 固定 ALFWorld workflow，也不
启动训练或 Skill evolution。

