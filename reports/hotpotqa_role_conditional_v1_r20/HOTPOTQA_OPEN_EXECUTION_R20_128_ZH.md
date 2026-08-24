# HotpotQA 开放编排 r20：128 题正式评测报告

## 结论

本轮已按“开放编排、角色按需选择”的要求实现并完成正式评测。`Reasoner`、`Verifier`、`Formatter` 仅是 Director search space 中的候选职责，不是必选角色；Runtime、action mask、Output 与 `FINISH` 均不要求三者齐备，也不要求 `Reasoner -> Verifier -> Formatter` 的固定先后依赖。ReAct 仅作为 Agent 的 `execution_mode`，没有定义名为 ReAct 的 Agent role。失败恢复采用 `preserve -> diagnose -> repair -> augment`；本轮 2,355 次 Director action 中没有一次 `delete_agent` 尝试。

架构约束已经解除，但本轮不能判定为 Stable Zero：128 题严格口径 EM 为 **38.28%**、F1 为 **41.22%**，并有 **67 个自然 terminal failure** 和 **1 个 operational failure**。主要问题是 Director 未能及时收敛并提交已经生成的答案，而不是固定三角色模板。

## 冻结评测条件

- 数据集：HotpotQA validation，固定顺序的同一批 128 题。
- Director：本地 Qwen3.5-9B，GPU0，SGLang。
- seed：`20260815`；concurrency：`2`；`max_rounds=28`。
- 输入：每题提供的全部 10 个 passages；Retriever 仅使用配置的 provided-context retrieval runtime。
- evaluator：HotpotQA answer-only official-compatible EM / token F1；本轮没有 formal supporting-fact prediction，因此不报告 supporting-fact 或 joint 指标。
- Direct comparator：同一批 128 题、同一 answer evaluator、同一本地 Qwen3.5-9B 的已冻结 receipt。
- 无训练：未运行 backward、optimizer step、policy update、GRPO、MACE 或 Bayesian update。
- 无 Skill 注入：`skills.enabled=false`，因此本报告不宣称 Skill 增益。

## 正式指标

严格口径把 1 个 operational failure 计为 0，分母固定为 128；不能把它描述为 128 条均有效。

| 条件 | 有效样本 | 显式 FINISH | EM | F1 | 自然 terminal failure | operational failure |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local | 128/128 | 不适用 | 70.31% | 78.68% | 不适用 | 0 |
| 旧 `qa_tool_react_exact_wire_v4_development` | 128/128 | — | 72.66% | 82.05% | 3 | 0 |
| 开放 AgentGraph r20（严格 128 题） | 127/128 | 60 | **38.28%** | **41.22%** | **67** | **1** |
| 开放 AgentGraph r20（仅 127 条有效轨迹，诊断口径） | 127/127 | 60 | 38.58% | 41.55% | 67 | 0 |

- 相对 Direct Local：EM **-32.03** 个百分点，F1 **-37.46** 个百分点。
- 相对旧条件：EM **-34.38** 个百分点，F1 **-40.83** 个百分点，natural terminal failure 从 3 增至 67。
- 仅看 60 条显式 `FINISH`：EM **81.67%**、F1 **87.94%**。这只是诊断指标，不能代替 128 题正式结果。
- 67 条 non-FINISH 轨迹全部没有 `final_answer`，因此正式 EM/F1 均为 0。
- 其中 46/67 条 non-FINISH 轨迹曾在 Agent execution output 中生成与 Ground Truth 规范化后完全一致的 candidate，但没有在轮数耗尽前完成 Output routing 与 `FINISH`。

## 开放 topology 的实测证据

127 条有效终态 AgentGraph 的统计如下：

| topology / execution 指标 | 结果 |
|---|---:|
| 严格单链 | 39/127（30.71%） |
| 分支图 | 82/127（64.57%） |
| 存在 fan-in | 65/127 |
| 存在 fan-out | 66/127 |
| 存在 reciprocal relation | 50/127 |
| 平均节点数 | 4.73 |
| 平均有向边数 | 4.46 |
| 平均 structural depth | 3.16 |
| 终态严格 `Reasoner -> Verifier -> Formatter` | 7/127（5.51%） |
| 同时使用 reasoning 与 ReAct execution mode | 94/127 |

角色出现次数也证明它们不是必选：`Verifier` 仅出现在 53/127 条轨迹，`Formatter` 仅出现在 37/127 条轨迹；12 条终态甚至没有设置 Output。7 条严格三角色链中，5 条正确 FINISH、1 条错误 FINISH、1 条 `max_rounds`，因此该拓扑既不是必要条件，也不保证正确。

## Recovery policy 的实测证据

- `preserve`：412 turns，257 accepted。
- `diagnose_repair`：209 turns，153 accepted。
- `augment`：1,050 turns，651 accepted。
- `delete_agent`：0 次尝试，0 次 accepted。
- 所有 action 的总 acceptance rate：1,555/2,355 = 66.03%。

这说明本轮没有继续采用“Agent 出错就删除”的策略。已有 artifact 和 semantic lineage 被保留，Director 通过 `modify_agent`、`add_subgraph`、`set_relation` 和 `set_output` 尝试修复或扩展；当前缺陷是这些 continuation 经常不能在 28 rounds 内收敛。

## 主要错误类型

1. **Director continuation / terminal semantics**：66 条 `max_rounds`，1 条 `no_admissible_action`。复杂图的平均 turns、节点数和边数均高于 FINISH 轨迹，说明 Director 在诊断、修复和扩增间反复编辑，没有及时提交可用答案。
2. **semantic answer 丢失**：46 条 non-FINISH 轨迹曾生成正确 candidate，却没有完成 semantic candidate 到 Output 的 routing 和 `FINISH`。这是当前最主要的 architecture regression。
3. **answer-slot 与 entity--attribute binding**：例如 GT `Crambidae` 却输出 `Nepita`；GT `Roman Catholic` 却输出 `University of Providence`。
4. **multi-hop completeness**：例如 GT `The Joshua Tree` 却输出 `The Chimes`；GT `3000 metres steeplechase` 却输出 `Asian Junior Athletics Championships`。
5. **specificity / serialization**：例如 `March 28, 1941` 退化为 `1941`，`16-year-old` 退化为 `16`，或输出过长的地点描述。Formatter 若被选择，只能序列化已经确定的 semantic answer；这些错误不能由 Formatter 重新选答案来掩盖。
6. **operational receipt failure**：selection index 83（GT `Train to Busan`）在三次精确续跑中均因 `ADD_SUBGRAPH` role-selection JSON 无效而失败，保留为 1 个真实 operational failure，没有伪造补齐。

## Architecture Completion Report

- 已完成并验证：开放 role search space、role 与 execution mode 正交、execute-on-edit、directed / reciprocal AgentGraph、provided-context retrieval、Output routing、official-compatible answer evaluator、完整 trajectory/receipt、`preserve -> diagnose -> repair -> augment` recovery。
- 未固定：Reasoner、Verifier、Formatter 的存在、数量、先后顺序和 topology；它们不是 `FINISH` 的全局前置条件。
- 仅预留且本轮未执行：GRPO、LoRA optimizer update、policy sync、MACE、Bayesian posterior、Skill gate / ACTIVE Skill。
- 已知阻塞：Director continuation 不能稳定收敛到 `FINISH`；部分正确 semantic candidate 在 Output routing 之前丢失；1 条 role-selection JSON operational failure。
- Stable Zero：**未达到**。执行链已跑通，但完整 128 题的 completion rate 和正式 EM/F1 均不满足稳定推理要求。

## 后续修改边界

下一轮应优先修正通用的 continuation 与 terminal policy：当已有 evidence-grounded semantic candidate 时，限制无效重复编辑，允许最小 Output routing 后 `FINISH`；同时保留 scope、entity binding、multi-hop evidence 和 answer-slot validation。不得把 `Reasoner -> Verifier -> Formatter` 重新写成固定模板，也不得针对 task ID 或 Ground Truth hard-code。
