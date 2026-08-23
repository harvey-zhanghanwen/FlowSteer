# HotpotQA 中性编排架构 128 样本重评报告

## 1. 评测条件

- 数据：固定 development/validation 切分，共 128 个唯一 `task_id`；`selected_tasks`、AgentGraph trajectory 与 paired result 的任务集合完全一致。
- Director：本地 Qwen3.5-9B，policy version `qwen35-9b-qa-free-agent-neutral-step-000000`，adapter version `theta_jointqa_progressive_step_000000`。
- 编排约束：`minimal-neutral.v10`、action mask v2、`semantic_protocol=none`、`recovery=default`、`require_format_agent=false`。不存在固定 Reasoner→Verifier→Formatter 模板；ReAct 仅作为 Agent 的 `execution_mode`。
- 指标：`hotpotqa.official.answer.v1`，使用 HotpotQA 官方答案归一化后的 Exact Match 与 token-level F1，`metric_scope=answer_only`。
- 本轮未训练：无 GRPO、backward、optimizer update、LoRA 更新、Bayesian update 或 Skill 注入；`optimizer_updates=0`，Skill 为 `memory_off`。
- Direct 使用同一批 128 个任务的已封存 Local Baseline receipt；两种协议不同，因此差值是描述性对照，不是因果效应估计。

## 2. 真实结果

| 条件 | 有效样本 | EM | F1 | 显式 FINISH | terminal failure |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128/128 | 90/128 = **70.3125%** | **78.6792%** | 128/128 | 0 |
| AgentGraph | 128/128 | 88/128 = **68.7500%** | **79.3217%** | 127/128 | 1 |

AgentGraph 相对 Direct：EM **-1.5625 个百分点**，F1 **+0.6426 个百分点**。配对 EM 矩阵为：两者都对 82 题、仅 AgentGraph 对 6 题、仅 Direct 对 8 题、两者都错 32 题。AgentGraph 的 40 个 EM 错误中，18 题 F1=0，22 题为 partial F1。

128 条 AgentGraph 与 128 条 Direct receipt 均通过 evaluator validity 检查，collection failure 为 0。运行状态为 `completed_with_terminal_failures`。此前 15 题 Stable Zero canary 为 15/15；正式 128 题中因 1 个 terminal failure，严格的全量 Stable Zero 检查为 127/128，不能标记为通过。

## 3. AgentGraph 与执行统计

- 最终 Agent 数：1 Agent 41 题、2 Agents 64 题、3 Agents 23 题。
- 最终 topology family：`single` 41、`serial_2` 50、`serial_3_plus` 13、`reciprocal` 16、`mixed` 7、`parallel` 1。
- 最终结构深度：depth 1 为 57 题、depth 2 为 51 题、depth 3 为 20 题。
- 含任一非线性 motif 的任务为 26/128（20.31%）；按互斥 topology family 统计，非链式结构为 24/128（18.75%）。
- 共 929 个 Director turns，平均 7.26，最大 28；成功执行 444 次。
- 193 次 invalid action，主要为 relation self-loop 94 次与重复 endpoint pair 61 次；另有 195 次 edit rejected。action-domain 合法性仍是明显的编排开销来源。
- ReAct 成功执行 135 次，覆盖 51 题；Tool Action/receipt 16 次，均为 `qa-retrieval`，覆盖 4 题。
- 最终图中 238 个 Agent 节点里，235 个为 `qwen3.5-9b-local`；余下 3 个外部模型节点均属于唯一 terminal failure。外部候选调用出现 177 次 HTTP 403，其中 176 次通过 model-only repair 回退到本地模型。

## 4. 唯一 terminal failure

任务 `hotpotqa:5a8efd3c55429918e830d179`：问题询问 The New Pornographers 与 Kings of Leon 是否都是美国摇滚乐队，ground truth 为 `no`。

失败不是 QA 推理错误，而是 failure recovery 的 liveness deadlock：`agent_3` 的外部 provider 返回 HTTP 403 后，provider-repair gate 要求只修改该节点的 `model_id`；同时 terminal-reachability gate 要求先添加一条能严格减少 unreachable Agent 的 relation。前者的 `modify_agent` 被后者拒绝，其他 relation/edit 又被前者拒绝，Director 在两个 mandatory gate 之间循环至 28 rounds，最终没有显式 `FINISH`，严格计 0 分。

## 5. 代表性错误

1. **answer-slot / entity-type 错配**：`hotpotqa:5a84918e5542990548d0b2cf`。问题问“哪本书包含该诗”，ground truth 为 `Exeter Book`，输出为诗名 `Widsith`。错误从首个语义 artifact 开始，下游继续传递。
2. **semantic lineage 被 Output 重写**：`hotpotqa:5adf732a5542993a75d264e9`。上游两个 Agent 已得到 `Kelli Ward`，Output Agent 重执行后改写成 `Sue Donahue`，导致正确 artifact 被覆盖。
3. **comparison relation 判断错误**：`hotpotqa:5ac2a912554299218029dae8`。The Wolfhounds 形成于 1985，Hole 形成于 1989，ground truth 为 `The Wolfhounds`，系统输出 `Hole`；没有独立的数值比较/evidence verification。
4. **answer span serialization 过长**：`hotpotqa:5a7a02235542996c55b2dcd3`。语义判断正确，但输出 `Aleksander Ford was born first.`，而 ground truth 为 `Aleksander Ford`；官方 EM 为 0、F1 为 0.5714。
5. **failure-recovery deadlock**：上述 `hotpotqa:5a8efd3c55429918e830d179`，正确类别应为 `no`，但未产生 terminal answer。

## 6. 非链式结构示例

- 正确：`hotpotqa:5a8ce5c7554299441c6b9f64`，图为 `topic → search → read` 且 `topic → read`，属于 3-Agent、depth-3、`mixed` topology，包含 fan-in/fan-out；输出与 ground truth 均为 `The Saimaa Gesture`，EM/F1 均为 1。
- 错误：`hotpotqa:5ab5141a5542991779162d70`，图为 `agent_0 → agent_1 → agent_2` 且 `agent_0 → agent_2`，同样属于 3-Agent、depth-3、`mixed` topology。通信链路正常，但三个同质 QA contract 一致保留了“奖项名称”，未恢复问题要求的“颁奖组织” answer slot；输出 `Academy Award for Best Director`，ground truth 为 `Academy of Motion Picture Arts and Sciences`。

## 7. 结论

本次重评完整、可复核，没有把训练样本 reward、无效 evaluator 或 terminal failure 排除出严格分母。中性 search space 已能生成单节点、串行、并行、reciprocal 与 mixed topology，但当前 AgentGraph **没有超过 Direct 的 EM**；其 F1 略高，说明部分错误更接近正确答案，但 answer-slot grounding、Output semantic preservation、comparison verification、terminal serialization 和 repair-gate precedence 仍是主要问题。正式结果不支持“达到 90%”或“Stable Zero 全量通过”的结论。

