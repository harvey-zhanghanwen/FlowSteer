# ALFWorld receipt-causal 错误分类报告

## 1. 统计口径

本报告只统计 `alfworld_valid_seen_unified_architecture_v1` 与
`alfworld_valid_unseen_unified_architecture_v1` 中 evaluator-valid、native
environment `success=0` 的 AgentGraph episode。每条失败 episode 只分配一个
互斥的 primary failure class。判定使用同一 task-scoped episode 中最长、最新的
完整 environment ledger；已在后续 Canvas turn 修复的早期 Tool/Canvas 异常仍保留
在 trajectory 中，但不再误记为终局 root cause。

`max_rounds`、缺少 `FINISH` 是 terminal manifestation，不自动作为 primary cause。
ALFWorld 的 reward 只来自 native environment terminal `won`；不使用 LLM judge、
Formatter、文本答案 canonicalization 或 Agent 自述。

机器可读分类位于两份正式 JSON 的
`alfworld_receipt_causal_failure_taxonomy` 字段：

- [valid_seen report](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_seen_report.json)
- [valid_unseen report](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_unseen_report.json)

## 2. 分类统计

### `valid_seen`

失败分母为 `94`（46/140 native success）。

| Primary failure class | 数量 | 占失败 episode |
|---|---:|---:|
| Environment exploration/search | 37 | 39.36% |
| Object grounding/affordance | 32 | 34.04% |
| Subgoal sequencing/action policy | 23 | 24.47% |
| Tool/execution-profile | 1 | 1.06% |
| Director/Canvas construction | 1 | 1.06% |
| Native action parser | 0 | 0% |
| Agent communication | 0 | 0% |
| Agent runtime | 0 | 0% |
| Environment runtime | 0 | 0% |
| Terminal control（primary cause） | 0 | 0% |
| Evaluator | 0 | 0% |
| Provider/unresolved collection | 0 | 0% |

### `valid_unseen`

失败分母为 `94`（40/134 native success）。

| Primary failure class | 数量 | 占失败 episode |
|---|---:|---:|
| Object grounding/affordance | 40 | 42.55% |
| Environment exploration/search | 35 | 37.23% |
| Subgoal sequencing/action policy | 18 | 19.15% |
| Tool/execution-profile | 1 | 1.06% |
| Director/Canvas construction | 0 | 0% |
| Native action parser | 0 | 0% |
| Agent communication | 0 | 0% |
| Agent runtime | 0 | 0% |
| Environment runtime | 0 | 0% |
| Terminal control（primary cause） | 0 | 0% |
| Evaluator | 0 | 0% |
| Provider/unresolved collection | 0 | 0% |

### 两个 official split 合并描述

该合并表只描述 error composition，不把两个 official split 合并为一个官方分数。

| Primary failure class | 数量 | 占 188 条失败 |
|---|---:|---:|
| Environment exploration/search | 72 | 38.30% |
| Object grounding/affordance | 72 | 38.30% |
| Subgoal sequencing/action policy | 41 | 21.81% |
| Tool/execution-profile | 2 | 1.06% |
| Director/Canvas construction | 1 | 0.53% |
| 其余七类 | 0 | 0% |

补充口径：seen AgentGraph 共记录 83 个、unseen 共记录 153 个
`parse_error` policy turn。这些 candidate 均不在当时的 native
`legal_actions` 中，parser 正确地拒绝了它们，所以属于 grounding/affordance 或
action-policy failure，不是 native parser defect。历史 3 次 provider/collection
失败均已按相同冻结条件恢复，unresolved 数量为 0。

不适用类别：检索/数据库（观测与 admissible actions 来自 stateful environment
Tool）、final-answer formatting/canonicalization、LLM judge validation。

## 3. 典型可复现 Wrong Demo

### 3.1 Environment exploration/search — `alfworld:valid_seen:00003`

- 输入：`clean some soapbar and put it in cabinet.`
- 目标：native environment `won=true`。
- 系统输出与指标：`final_answer=null`，`success=0`，`episode_score=0`，
  evaluator-valid，`20` environment policy turns，`max_rounds`，无 `FINISH`。
- Director/Canvas：D0 添加 `agent_0` 为 React，但未绑定 Tool，runtime 返回
  request-scoped Tool ownership error；D1 通过结构化
  `allowed_tools=["alfworld"]` 修复，随后该 Agent 独占 task-scoped session。
- Agent 输入/输出：`agent_0` 收到原任务、当前 observation、完整
  admissible-action list 和已执行 Action--Observation history；没有 upstream
  artifact。它的输出被解析为 native ALFWorld action。
- Agent communication：`upstream=[]`，因此本例不存在 transport、edge direction
  或 artifact delivery failure。
- ReAct Tool Action--Observation receipt：前 7 步依次检查 cabinet、countertop、
  sinkbasin、bathtubbasin 等 receptacle；从第 8 个 policy turn 开始持续
  `cabinet -> countertop -> cabinet -> countertop`，environment observation 未出现
  soapbar，也没有 `take/clean/move soapbar`。
- Terminal/evaluator receipt：最后 trace `done=false`、`won=false`、`score=0`；
  evaluator `skillflow.ragen_adapter.v2` 返回 valid success=0。
- 首个决定性失败点：进入已经检查过的 receptacle 并形成 A-B-A-B
  no-progress oscillation，而没有扩展 exploration frontier。
- 错误传播：`incomplete exploration -> repeated world state -> no object grounding
  -> no transformation/placement -> action budget exhausted -> max_rounds`。

可复核 trajectory：[line 4](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/artifacts/alfworld_initial_adapter_v1/valid_seen/agentgraph_trajectories.jsonl:4)。

### 3.2 Object grounding/affordance — `alfworld:valid_unseen:00000`

- 输入：`cool some lettuce and put it in countertop.`
- 目标：cool lettuce，再放到 countertop；native `won=true`。
- 系统输出与指标：`final_answer=null`，`success=0`，`score=0`，
  evaluator-valid，20 policy turns，`max_rounds`。
- Director/Canvas：D0 添加 React `alfworld_agent`，因无 Tool 绑定失败；D1 设置
  `allowed_tools=["alfworld"]` 后恢复并开始真实 episode；D2 设置 Output Agent；
  D3 添加 reasoning `alfworld_agent_2`；D4 建立
  `alfworld_agent_2 -> alfworld_agent`。
- Agent 输入/输出：environment owner 收到原任务、observation、admissible actions；
  reasoning Agent 输出 `{"action":"cool_lettuce"}`，作为无 Tool receipt 的
  upstream artifact 送达 owner。
- Agent communication：transport 成功，artifact 确实进入 owner inbox；但
  `environment_revision=null`、`tool_receipts=[]`，所以它不是已冷却 lettuce 的
  environment evidence。
- ReAct Tool Action--Observation receipt：

  ```text
  go to fridge 1
  open fridge 1
  -> bowl, bread, cup, egg, mug, plate, potato, tomato；没有 lettuce
  take lettuce 1 from fridge 1
  -> parse_error：不在当前 admissible-action list
  后续 17 个 turn 继续输出同一 off-list lettuce action
  ```

- Terminal/evaluator receipt：world state 停留在 fridge observation，
  `done=false/won=false/score=0`，native evaluator valid success=0。
- 首个决定性失败点：在 observation 已否定 fridge 中存在 lettuce 后，仍将
  target entity-location binding 固定到 fridge，并选择不可执行 affordance。
- 错误传播：`wrong entity-location binding -> off-list action -> no state change ->
  repeated action -> budget exhausted -> max_rounds`。

可复核 trajectory：[line 1](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/artifacts/alfworld_initial_adapter_v1/valid_unseen/agentgraph_trajectories.jsonl:1)。

### 3.3 Subgoal sequencing/action policy — `alfworld:valid_unseen:00080`

- 输入：`heat some apple and put it in garbagecan.`
- 目标：先满足 heat predicate，再满足 garbagecan placement predicate。
- 系统输出与指标：`final_answer=null`，`success=0`，`score=0`，
  evaluator-valid，20 policy turns，`max_rounds`。
- Director/Canvas：D0 创建 reasoning `agent_1`；D1 改为 React 但未绑定 Tool；
  D4 绑定 `alfworld` 后恢复。随后创建 `agent_2/3/4`，D7/D9 形成三路 fan-in
  到 environment owner；D13 又把 Output pointer 改到无 Tool 的 `agent_4`。
- Agent 输入/输出：三个 reasoning Agent 输出 generic
  `{"action":"move_to(0, 0)"}`；这些 artifact 被实际路由到 owner，但均无
  environment receipt。owner 根据真实 observation 执行 native actions。
- Agent communication：没有消息丢失；问题是上游 artifact 未包含可验证的
  environment state/progress，且后续 Output pointer 与 Tool artifact lineage 不一致。
- ReAct Tool Action--Observation receipt：

  ```text
  find apple on countertop -> take apple 1
  go to microwave 1 -> open microwave 1
  legal_actions 已包含 heat apple 1 with microwave 1
  但选择 move apple 1 to microwave 1
  close -> open -> take apple 1
  move 未加热 apple 1 to garbagecan 1
  最后在 microwave/countertop 间循环
  ```

- Terminal/evaluator receipt：placement 已发生，但 heat predicate 未满足，
  `done=false/won=false/score=0`，native evaluator valid success=0。
- 首个决定性失败点：在 heat action 已处于 admissible domain 时选择 placement，
  破坏了正确的 subgoal ordering。
- 错误传播：`missed transformation predicate -> destination predicate alone cannot
  satisfy goal -> repeated navigation -> budget exhausted -> max_rounds`。

可复核 trajectory：[line 81](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/artifacts/alfworld_initial_adapter_v1/valid_unseen/agentgraph_trajectories.jsonl:81)。

### 3.4 Tool/execution-profile — `alfworld:valid_unseen:00075`

- 输入：`find two soapbar and put them in cabinet.`
- 目标：native `won=true`，且 count predicate 为 two soapbar。
- 系统输出与指标：`final_answer=null`，`success=0`，`score=0`，0 native action，
  evaluator-valid，`max_rounds`。
- Director/Canvas：D0--D7 连续添加 `agent_0` 至 `agent_7`，D8
  `set_output(agent_7)`，D9--D19 继续编辑 relations；D11、D15 的 no-op edit
  被 Canvas 拒绝。
- Agent 输入/输出：所有节点实际均为
  `execution_mode=reasoning, allowed_tools=[]`。输出主要为
  `{"action":"move_to","target":"kitchen"}` 或坐标式 pseudo-action。
- Agent communication：D9 构成 `agent_1 <-> agent_7` bounded reciprocal
  exchange，后续还有 fan-out；通信 transport 工作正常，但传播的只是没有
  environment grounding 的 reasoning artifact。
- ReAct Tool Action--Observation receipt：0。Director 把
  `execution_mode: react, allowed_tools: [alfworld]` 写进 free-text contract，
  没有写入 typed Canvas fields，因此没有 Agent 获得 stateful session。
- Terminal/evaluator receipt：trace 为空，reset observation 保留，evaluator reason
  为 `environment_rollout_closed_before_terminal`，valid success=0。
- 首个决定性失败点：free-text contract 与 typed execution profile 不一致。
- 错误传播：`no valid Tool owner -> no world-state transition -> topology grows but
  remains non-executable -> max_rounds`。

可复核 trajectory：[line 76](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/artifacts/alfworld_initial_adapter_v1/valid_unseen/agentgraph_trajectories.jsonl:76)。

### 3.5 Director/Canvas construction — `alfworld:valid_seen:00093`

- 输入：`put some creditcard on drawer.`
- 目标：pickup creditcard 并放入 drawer，native `won=true`。
- 系统输出与指标：`final_answer=null`，`success=0`，`score=0`，0 native action，
  evaluator-valid，`max_rounds`。
- Director/Canvas：D0/D1 添加 `agent_0/agent_1`，但实际都是 reasoning 且
  `allowed_tools=[]`；D2 设 `agent_1` 为 Output；D3 执行
  `set_relation(agent_1, alfworld)`，Canvas 明确拒绝 `unknown agent_id: alfworld`；
  后续添加 `agent_2` 并在 reciprocal、serial、fan-in relations 间反复重排。
- Agent 输入/输出：最初两个 Agent 输出
  `{"action":"put creditcard in drawer"}`；之后输出
  `{"action":"put creditcard on drawer"}`、generic `move_to`，或明确说明没有
  environment observation/admissible actions。
- Agent communication：relation transport 可把上述 reasoning artifacts 在
  Agent 间传递；receipt 中 `environment_revision=null`、`tool_receipts=[]`，
  Output inbox 也没有 native environment artifact。
- ReAct Tool Action--Observation receipt：0。没有节点被改为 typed React Tool
  owner；`alfworld` 被当成 Agent ID，而不是 Tool capability。
- Terminal/evaluator receipt：trace 为空，terminal observation 仍为 reset 房间，
  `done=false/won=false/score=0`，native evaluator valid success=0。
- 首个决定性失败点：D3 将 Tool resource identifier 用作 Canvas relation endpoint，
  之后继续 relation construction 而没有修复 execution profile。
- 错误传播：`invalid Canvas target -> relation churn among Tool-free nodes ->
  pseudo-action communication -> no environment transition -> max_rounds`。

可复核 trajectory：[line 94](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/artifacts/alfworld_initial_adapter_v1/valid_seen/agentgraph_trajectories.jsonl:94)。

## 4. 零计数类别与结论

- Native action parser：0。存在 action-domain/admissibility error，但没有证据表明
  parser 拒绝了一个当时合法的 native action。
- Agent communication：0 个 primary cause。抽查的 upstream artifact 均实际到达；
  主要问题是 artifact 缺少 environment evidence、Director 未充分使用 relations，
  或 Output pointer 与 artifact lineage 不一致，这些是次生结构问题。
- Agent runtime、environment runtime、evaluator：均为 0 个当前 unresolved
  primary cause。两条旧 evaluator-trace propagation mismatch 已通过完整 frozen
  ledger 的确定性 native replay 修正为 success，全程模型调用数为 0。
- Provider/collection：0 个 unresolved failure；3 次历史 connection failure 已恢复，
  receipt 保留但不计入 Wrong Demo。
- Terminal control：0 个 primary cause；两个 split 的 188 条 native 失败均以
  `max_rounds` 呈现，但这是上游 failure 的终止结果。另有 2 条 unseen native
  success 也没有 `FINISH`，所以 `max_rounds` 不能代替 task reward。

当前主要瓶颈是 environment exploration、object grounding/affordance 与 subgoal
sequencing，而不是 evaluator、provider 或消息 transport。该结论只用于诊断；本轮
没有据此硬编码 ALFWorld workflow，也没有启动训练或 Skill evolution。
