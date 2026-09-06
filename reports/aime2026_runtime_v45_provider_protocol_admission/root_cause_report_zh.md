# AIME 2026 v45 同 30 题评测与根因报告

## 1. 评测状态

- 条件：`aime2026_runtime_v45_provider_protocol_admission`
- 数据：AIME 2026 固定 30 题，同一正式分母
- 正式指标：经目标不可见 answer extraction / canonicalization 后的 Exact Match Accuracy
- AgentGraph：`24 / 30 = 80.00%`
- AgentGraph evaluator-valid 子集：`24 / 27 = 88.89%`
- Qwen3.5-9B Direct：`19 / 30 = 63.33%`
- 同题差值：`+5` 题，严格 Accuracy `+16.67` percentage points
- 显式 `FINISH`：`27 / 30`
- terminal failure：`3 / 30`
- AgentGraph parsing failure：`0 / 30`
- collection / operational failure：`0 / 30`
- 训练、GRPO、MACE、Bayesian posterior、Skill retrieval/evolution：均未启用

`protocol_equivalent_to_direct=false`，因此 Direct 与 AgentGraph 的差值是同题、同 evaluator
下的描述性 paired comparison，不应解释为严格因果效应。Direct 有 11 个
`finish_reason=length` 并形成 11 个 terminal-output parsing failure；AgentGraph 没有
parsing failure。

## 2. 本轮架构修复

本轮保留统一自由 AgentGraph、自由文本 contract、异构模型目录、逐次 Canvas edit、
执行反馈和显式 `FINISH`。没有加入固定 Solver/Verifier 角色或固定数学 workflow。

1. Artifact completeness：不完整或被截断的输出不能成为可终止 artifact；成功的
   upstream artifact 保持不可变并复用。
2. Provenance-aware assessment：fan-in 保留 `source_agent_id + artifact_id + raw_output`；
   assessment 必须绑定确切 artifact，不用 ground truth 仲裁 candidate。
3. Recovery ordering：对 provider/runtime failure，优先暴露同一 Agent 的显式
   model/profile repair，再进入 provenance-protocol repair；runtime 不静默换模型。
4. Parameter-level action masking：constrained Director domain 与 raw Canvas admission
   使用同一可执行 parameter domain，避免 schema 允许但 Canvas 再拒绝的组合。
5. Artifact consumption 与 termination lookahead：已有 fresh、complete、parseable
   artifact 时先尝试消费；`SET_OUTPUT` 只改 Output pointer，`FINISH` 只消费当前
   fresh Output artifact，不重新执行 Agent。
6. Role-neutral AIME：Agent 仍为 `agent_id + model_id + free-text contract`，AIME action
   domain 不携带预定义 role enum；coding/ReAct 仅是可选 execution mode。

定向回归覆盖 provider repair precedence、exact fan-in consumer repair、v44/v45 同 30
题条件和禁用训练边界；Stable Zero canary 的 task10 得到 `156`，合法显式 `FINISH`，
且 evaluator-valid。

## 3. 图结构统计

| 指标 | 结果 |
|---|---:|
| 单 Agent | 15 / 30 |
| 2 Agent | 10 / 30 |
| 3 Agent | 1 / 30 |
| 4 Agent | 3 / 30 |
| 5 Agent | 1 / 30 |
| multi-Agent | 15 / 30 |
| `single` | 15 |
| `serial_2` | 10 |
| `fan_in` | 2 |
| `mixed` | 3 |
| structural depth >= 3 | 3 / 30 |
| rejected Director turns | 10 / 218（4.59%） |
| mean Director turns | 7.27 |
| max Director turns | 18 |

实际 Executor 调用分布为 Qwen3.5-9B local `79`、DeepSeek-V4-Flash `22`、
MiniMax-M3 `15`。因此并非所有 Agent 都使用 Qwen3.5-9B；Director 固定为本地
Qwen3.5-9B，workflow Agent 从冻结目录中选择。

## 4. 六个失败样本的专业分类

| Failure taxonomy | 数量 | 占全部 30 题 | Task |
|---|---:|---:|---|
| mathematical reasoning error（合法终止但答案错误） | 3 | 10.00% | 09, 15, 29 |
| termination/control-plane failure（正确 candidate 已产生） | 2 | 6.67% | 17, 22 |
| runtime/artifact failure（未形成合法 candidate） | 1 | 3.33% | 28 |
| output extraction/canonicalization failure | 0 | 0% | — |
| evaluator failure | 0 | 0% | — |
| collection failure | 0 | 0% | — |

### 4.1 Task 09：条件概率推理错误

- 问题：贴纸覆盖条件下，求“恰有一面空白”在偶数贴纸都可见条件下的条件概率。
- 参考答案：`29`；正式 prediction：`16`；Accuracy：`0`。
- 链路：
  1. Round 0 `ADD_SUBGRAPH`：node_1，Qwen3.5-Flash，coding + calculator；
     `ReactGenerationError`。
  2. Round 1 `MODIFY_AGENT`：改为 reasoning。
  3. Round 2 `MODIFY_AGENT`：显式切到 Qwen3.5-9B local；形成完整 artifact，
     candidate=`16`。
  4. Round 3 `SET_OUTPUT(node_1)`。
  5. Round 4 `FINISH`；evaluator 将 `Final Answer: 16` 规范化为 `16`。
- 首个因果失败点：恢复后 node_1 的条件样本空间计数错误，将所求条件概率的分母
  错当成“恰有一面空白”的序列总数；这是 mathematical reasoning error。
- 传播：错误 candidate 被合法解析、设为 Output 并显式终止。不存在 communication、
  terminal 或 evaluator bug。

### 4.2 Task 15：组合几何计数错误

- 问题：将 `10 x 10` 网格分成 5 个 cell loops 的方案数。
- 参考答案：`83`；正式 prediction：`2`；Accuracy：`0`。
- 链路：
  1. Round 0 `ADD_SUBGRAPH`：node_1，Qwen3.5-Flash，ReAct + Python；执行失败。
  2. Round 1 改 reasoning；Round 2 显式切到 Qwen3.5-9B local。
  3. Round 2 形成完整 artifact，candidate=`2`。
  4. Round 3 `FINISH`。
- 首个因果失败点：node_1 将 perimeter identity 当作足以决定平面 tiling 数量，遗漏
  cell-loop 的嵌套/分割结构，属于 combinatorial reasoning error。
- 传播：单一错误 artifact 被合法消费；没有 parsing 或 provenance 丢失。

### 4.3 Task 29：递推计数错误

- 问题：对奇偶相关二元运算，计数和为 12 且从左到右运算结果为 0 的正整数序列。
- 参考答案：`157`；正式 prediction：`28`；Accuracy：`0`。
- 链路：
  1. Round 0 ReAct + Python 失败。
  2. Round 1 改 reasoning；Round 2 Qwen3.5-9B local 输出不完整。
  3. Round 3 改 Qwen3.5-Flash，provider failure。
  4. Round 4 显式恢复 Qwen3.5-9B local，形成完整 artifact，candidate=`28`。
  5. Round 5 `FINISH`。
- 首个因果失败点：最终 artifact 把每个“奇数计数为奇数”的区间都误当成必须各自
  承担和 6，而实际约束是这些区间的总和为 6；属于 recurrence/counting error。
- 传播：runtime recovery 最终成功，但恢复后的数学结论错误；终止与 evaluator 正常。

### 4.4 Task 17：正确 candidate 存在，但 assessment action 被拒绝后预算不可达

- 问题：10 个方格图上的受限 trail 数，求 `sqrt(N)`。
- 参考答案：`243`；正式 prediction：`null`；terminal failure。
- 链路摘要：两节点 serial AgentGraph；node_1 经 429、incomplete artifact 和多次显式
  profile/model repair 后成功；node_2 在 Round 16 形成 fresh、complete artifact，
  candidate=`243`，且 Output pointer 已为 node_2。
- Round 17：Director 增加 assessment consumer，但把未验证 candidate `243` 写入
  pre-execution contract；Canvas 以 `unverified_candidate_in_contract` 拒绝。
- 首个决定性失败点：这个 rejected action 消耗了临界轮次；随后 `remaining_rounds=2`，
  而 provenance-bound terminal path 的下界为 3 个 action，形成
  `canvas_action_domain_exhausted`。
- 结论：数学答案已算对；主要缺口是 rejected-action horizon accounting 与 Director
  contract discipline，而不是 answer extraction。

### 4.5 Task 22：四个一致的正确 candidate，但 Output pointer 与图 sink 不一致

- 问题：Alice/Bob 在 Carol 首次得币前各至少得两枚币的概率，求 `100m+n`。
- 参考答案：`754`；正式 prediction：`null`；terminal failure。
- 链路摘要：最终是 4-Agent mixed topology。node_1、node_2、node_3 均形成
  candidate=`754`；node_4 接收三路 exact fan-in，形成 candidate=`754`，并对三个
  upstream artifact 给出 provenance-bound `supported` assessment。
- 决定性失败点：Output pointer 仍在 node_3，而 node_4 是新 sink。Graph validation
  报告 `output_not_sink` 与 `cannot_reach_output`。同时旧的 protocol failure state
  仍把 node_1/node_2 标成 active failure，使 `next_progress_action_types=[]`，尽管
  `SET_OUTPUT(node_4) -> FINISH` 在语义上已经足够。
- 结论：这是 Output-pointer/recovery-state consistency 缺口。fan-in 信息并不少；
  provenance、raw output 和 assessment 都已到达 node_4。

### 4.6 Task 28：结构化动作与 artifact 不完整导致无合法 candidate

- 问题：含 4040 个 cousins 的有限整数集的最小基数。
- 参考答案：`107`；正式 prediction：`null`；terminal failure。
- 链路摘要：初始 MiniMax ReAct + calculator 出现 structured-action serialization
  failure；改 Python 后仍失败。Director 在 unresolved node_1 尚未修复时扩为 5-Agent
  图，产生 blocked downstream。node_2 曾成功，但只输出 4 个字符且 candidate 超出
  AIME 整数范围；node_3/node_5 又经历 429、continuation exhaustion 和 incomplete
  artifact。
- 决定性失败点：没有任何 fresh parseable candidate，且没有合法 Output Agent；
  最终 `candidate_count=0`、`admissible_output_agent_ids=[]`、
  `canvas_action_domain_exhausted`。
- 结论：主因是 structured-action/runtime artifact failure，Director 的
  augment-before-repair 又放大了阻塞传播；不是 evaluator 或格式化错误。

## 5. 当前结论与下一步边界

v45 已把 parsing、collection、静默模型回退、fresh artifact 重采样等工程噪声从
正式指标中基本剥离，但仍有两个明确 control-plane 缺口：

1. rejected action 不应把一个本来可达的 assessment + FINISH 路径推到 horizon 外；
2. 成功 fan-in consumer 已吸收并评估旧 protocol-failed artifacts 后，应清除相应
   active-failure state，并立即暴露 `SET_OUTPUT(consumer)`。

这两点可在下一版做定向 runtime 修复；task09/15/29 更像模型 mathematical reasoning
能力不足，若后续改善应依赖真实 trajectory、工具执行或训练，而不是把这三题的解法
写入初始 Director prompt。按本轮范围，正式同 30 题只运行一次，不再追加付费重跑。

## 6. 可复现证据

- 机器可读总报告：`evaluation_report.json`
- 自动报告：`evaluation_report.md`
- 固定任务与逐题 paired 结果：`selected_tasks.jsonl`、`paired_results.jsonl`
- 完整 AgentGraph trajectory：`agentgraph_trajectories.jsonl`
- 六个 Wrong Demo：`wrong_demos.jsonl`
- collection failures：`collection_failures.jsonl`（0 条）
- manifest 与 preflight：`run_manifest.json`、`preflight_receipt.json`

