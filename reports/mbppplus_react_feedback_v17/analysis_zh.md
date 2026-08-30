# MBPP+ 逐 Action ReAct 反馈评测报告（v17）

## 结论

固定 100 题、官方 EvalPlus 0.3.1 完整覆盖下，Qwen3.5-9B Direct 的
Base pass@1 / Plus pass@1 为 **83% / 72%**；v17 AgentGraph 为
**81% / 69%**。AgentGraph 相对 Direct 分别为 **-2 / -3 个百分点**。
本轮没有训练、LoRA、GRPO、MACE、Bayesian update、Skill retrieval 或
Skill evolution。

## ReAct 反馈闭环

每次 Agent execution 只产生一个公开 Action--Observation transition，随后
返回 FlowSteer Canvas。下一次 Director 决策收到：

- 原始 `task_objective`；
- Agent 的 `final_objective` 与 free-text contract；
- `execution_status`；
- `latest_action`；
- `latest_observation`；
- Tool call budget 与 remaining budget。

完整 trajectory 另行保留无损 `react_events`；Director prompt 使用合并后的
紧凑 `react_state`，不重复注入同一个 event list。公开测试 Observation 可见，
EvalPlus Base/Plus hidden inputs、expected outputs 与 evaluator outcomes 不进入
Director prompt、Agent input、Agent communication 或 public Tool receipt。

全量审计：

- trajectory：100；
- ReAct Action--Observation event：809；
- 存在下一次 Director 决策的 event：801；
- 被下一次 Director 完整观察的 event：801（100%）；
- Director prompt 中重复 `react_events`：0；
- 最后一个 Director round 产生、因 `max_rounds` 而没有下一次决策的 event：8。

最后一类 event 仍完整保存在 terminal trajectory；它们没有被标记为通信丢失。

## action mask 修正

v16 首次把同一步完成、失败与 pending 的所有并行 ReAct Agent 合并到
`react_state`。随后发现 action mask 错误地使用所有 Agent 的
`observation_status` 判定是否开放 `modify_agent`；已完成 sibling 的合法
`completed` 状态因此被当作 pending failure。v17 保持全部状态可见，但只用
pending Agent 的最新 Observation 决定 pending boundary 的 repair admission。

定向测试确认：一个 pending Agent 的 Tool Observation 为 `success`，即使同一步
另一个 sibling 已 `completed`，下一动作域仍只有 `continue`，不会错误开放
`modify_agent`。

## 正式指标

| Condition | Base passed | Base pass@1 | Plus passed | Plus pass@1 | Evaluator coverage |
|---|---:|---:|---:|---:|---:|
| Direct | 83/100 | 83% | 72/100 | 72% | 100/100 |
| AgentGraph v17 | 81/100 | 81% | 69/100 | 69% | 100/100 |

AgentGraph 的 evaluator status pair 为：

- Base PASS / Plus PASS：69；
- Base PASS / Plus FAIL：12；
- Base FAIL / Plus FAIL：19；
- Base FAIL / Plus PASS：0。

终止与运行状态：

- explicit FINISH：91；
- max_rounds：9；
- final operational failure：0；
- parse failure：0；
- 首次 collection timeout：1（`Mbpp/160`），随后只续跑该缺口并补齐正式
  100/100 evaluator coverage；旧 timeout receipt 被保留。

## 与先前条件的区别

| Version | Base pass@1 | Plus pass@1 | FINISH | max_rounds | 反馈条件 |
|---|---:|---:|---:|---:|---|
| v11 | 90% | 75% | 92 | 8 | 审计中只有 311/391 对应 event 出现在下一次 Director prompt，不满足本轮逐 Action 完整回传要求 |
| v16 | 85% | 71% | 88 | 12 | 回传完整，但 completed sibling 错误影响 pending repair admission |
| v17 | 81% | 69% | 91 | 9 | 回传完整，pending-only repair admission |

v17 将 `modify_agent` 从 v16 的 249 次降到 137 次，将 rejected-turn rate 从
8.79% 降到 4.36%，并把 max_rounds 从 12 降到 9。这说明 action-mask bug 已被
修复；但正式 Base/Plus 指标没有提高。因此不能声称逐 Action 状态回传本身已经
改善代码正确性。

主要剩余问题是 public assertion 对 hidden edge cases 的约束不足，以及 Director
生成的 AgentGraph/contract 在不同采样条件下仍不稳定。12 个 Base PASS / Plus
FAIL 样本直接表明：通过公开测试不等于通过 Plus hidden tests。后续若要提高
Plus pass@1，应在不泄漏 hidden tests 的前提下改进从公开 specification 推导
边界条件的代码推理；不能通过把 evaluator 结果回传给 Director 来优化。

## 自然 topology

v17 的最终 topology 分布：single 28、serial_2 31、serial_3_plus 16、fan_in 9、
parallel 2、mixed 14。Director 没有被固定为 Coder--Reviewer--Tester 或固定链式
模板；ReAct 仍是 Agent 的 execution mode，而不是 role。

## 证据

- `artifacts/mbppplus_react_feedback_v17/evaluation/run_manifest.json`
- `artifacts/mbppplus_react_feedback_v17/evaluation/agentgraph_trajectories.jsonl`
- `artifacts/mbppplus_react_feedback_v17/evaluation/paired_results.jsonl`
- `artifacts/mbppplus_react_feedback_v17/evaluation/wrong_demos.jsonl`
- `reports/mbppplus_react_feedback_v17/evaluation_report.json`
- `reports/mbppplus_react_feedback_v17/evaluation_report.md`
