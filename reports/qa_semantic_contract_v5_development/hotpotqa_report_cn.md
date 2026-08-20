# HotpotQA 语义契约 v5 开发集验证报告

## 评测条件

- 数据：固定的 HotpotQA development validation-128；v4 与 v5 的 `task_id`、问题和标准答案逐条一致。
- 输入：每题完整 10 个 passages。
- 指标：`hotpotqa.official.answer.v1` 的 answer-only Exact Match（EM）与 token-level F1；本轮未生成 supporting-fact prediction，因此没有 supporting-fact/joint 指标。
- Flow-Director：本地 Qwen3.5-9B；Executor 由既有 heterogeneous model catalog 选择。
- 本轮没有训练、backward、optimizer update、LoRA 更新、MACE、Bayesian update 或 Skill 注入。

## 本轮最小架构适配

1. 在 Canvas commit 前复用 AgentRuntime 的 execution-contract validation；非法的 `reasoning + allowed_tools` 不再写入 Canvas，也不会先触发失败执行。
2. Flow-Director 的 contract 增加 relation、qualifier、comparison criterion 和 answer type fidelity，并要求多跳答案具有 source-grounded evidence；没有固定 Agent 数量、角色模板或拓扑。
3. Format Agent 保持独立 terminal sink，只消费一个 semantic-answer predecessor；对 uncertainty 或 unresolved aliases fail closed，不把 unknown 映射为 `no`。
4. Intermediate Agent 在 verification contract 下独立重建 evidence，并报告 agreement、conflict 或 insufficiency。

## 最新结果

| Condition | 有效样本 | EM | F1 |
|---|---:|---:|---:|
| Qwen3.5-9B Direct local baseline | 128/128 | 70.31%（90/128） | 78.68% |
| AgentGraph v4 | 128/128 | 72.66%（93/128） | 82.05% |
| AgentGraph v5 | 128/128 | **73.44%（94/128）** | **83.40%** |

- v5 相对 Direct：EM `+3.12` 个百分点，F1 `+4.72` 个百分点。
- v5 相对 v4：EM `+0.78` 个百分点，F1 `+1.35` 个百分点；paired outcome 为 7 条 wrong-to-right、6 条 right-to-wrong。
- 显式 `FINISH`：126/128；另 2 条为 `max_rounds`。最终 128 条均有合法 evaluator result，但由于这 2 条未显式终止，完整 Stable Zero 判定仍不成立。

## 正确 Demo 1：多跳实体消歧

- `task_id`: `hotpotqa:5a82171f5542990a1d231f4a`
- 问题：What nationality was James Henry Miller's wife?
- Ground Truth：`American`
- Direct：`British`（EM 0 / F1 0）
- AgentGraph：`American`（EM 1 / F1 1）

AgentGraph：

`Retrieval Agent (DeepSeek-V4-Flash) → Reasoning Agent (Qwen3.5-9B local) → Format Agent (GLM-4.5-Flash)`

执行过程：

1. Round 0：Canvas 拒绝把非 Format Agent 设为 Output Agent；没有执行。
2. Round 1：Canvas 拒绝 model catalog 中不存在的 model ID；没有执行。
3. Round 2：接受三节点子图并执行。Retrieval Agent 找到 James Henry Miller 即 Ewan MacColl，并检索 Peggy Seeger；Reasoning Agent消除同名人物歧义，确认 Peggy Seeger 是 American；Format Agent 输出 `<answer>American</answer>`。
4. Round 3：Flow-Director 发出 `finish`。

关键通信：

`Retrieval Agent → Reasoning Agent: American`

`Reasoning Agent → Format Agent: James Henry Miller (Ewan MacColl) was married to Peggy Seeger; the passage explicitly identifies Peggy Seeger as an American folksinger.`

## 正确 Demo 2：比较关系与答案类型保持

- `task_id`: `hotpotqa:5ac3e8c65542997ea680c993`
- 问题：Are the New Orleans Outfall Canals the same length as the Augusta Canal?
- Ground Truth：`yes`
- Direct：`No`（EM 0 / F1 0）
- AgentGraph：`yes`（EM 1 / F1 1）

AgentGraph：

`Reasoning Agent (Qwen3.5-Flash) → Format Agent (Qwen3.5-Flash)`

关键通信：Reasoning Agent 分别提取 Augusta Canal 为约 13 miles、New Orleans Outfall Canals 为 13 miles，并将比较结论 `Yes` 发送给 Format Agent；Format Agent 输出 `<answer>yes</answer>`。轨迹以显式 `finish` 终止。

## 错误 Demo 1：答案别名未消歧

- `task_id`: `hotpotqa:5a84dd955542997b5ce3ff79`
- 问题：Cadmium Chloride is slightly soluble in this chemical, it is also called what?
- Ground Truth：`alcohol`
- Direct：`alcohol`（EM 1 / F1 1）
- AgentGraph：`ethanol`（EM 0 / F1 0）

AgentGraph：

`Reasoning Agent (GPT-4o-mini) → Format Agent (GPT-4o-mini)`

首个错误发生在 Flow-Director 生成的 semantic contract：contract 已把答案空间引导为 `Ethanol or ethyl alcohol`。Reasoning Agent 发送的 artifact 同时包含 `alcohol / ethanol / ethyl alcohol`，Format Agent随后选择 `ethanol`。消息完整到达，因此这不是 communication transport error，而是 semantic contract 与 answer alias resolution 错误。

## 错误 Demo 2：正确答案已生成但没有进入终局

- `task_id`: `hotpotqa:5a7e36045542991319bc9440`
- 问题：Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?
- Ground Truth：`Jonathan Stark`
- Direct：`Jonathan Stark`（EM 1 / F1 1）
- AgentGraph 正式结果：`null`（EM 0 / F1 0）

轨迹中先后出现 execution contract rejection、本地模型 HTTP 400、ReAct completion exhaustion 和不可达 Output Agent。Round 18 的 Reasoning Agent 已重新核对 passages，得到 Henri Leconte 1 个、Jonathan Stark 2 个；Format Agent也实际生成了 `<answer>Jonathan Stark</answer>`。但 Flow-Director 在 Round 19 删除上游 Agent，且没有在 20-round budget 内发出 `finish`，因此正式结果按 `max_rounds` 记为 `null`。决定性失败属于 terminal control，而不是答案推理或消息传输。

## 结构诊断与结论

- Agent 数量分布：1/2/3 Agents = 1/87/40。
- topology family：single 1、serial-2 88、serial-3-plus 39；depth ≥ 3 为 39。
- 本轮没有自然选择 fan-in 或 reciprocal topology。运行时已通过定向单元测试证明 `Reader ⇄ Verifier → Format` 可执行，但未训练的 Flow-Director 尚未在这 128 题中选择这种拓扑。
- rejected-turn rate 为 48.34%，parse failure 为 85。v5 提升了答案质量和执行边界，但编排效率、terminal control 和 answer alias resolution 仍是主要问题。
- 首轮服务的 context ceiling 为 8192，完成 126 条；两条 SGLang HTTP 400 后按相同 task、prompt、catalog、policy、adapter、seed 和 evaluator 在 32768 context ceiling 下精确续跑缺失两条，没有重复 126 条成功调用。历史失败 receipt 被保留，最终 operational/evaluator failure 为 0。

