# AIME 2026 AgentGraph Runtime v3 报告

## 结论

- 固定官方测试集：30 题，task ID 为 `aime-2026/01` 至
  `aime-2026/30`。
- Qwen3.5-9B Direct：`6/30 = 20.00%` Accuracy。
- AgentGraph v2.3：`10/30 = 33.33%` Accuracy。
- AgentGraph v3：`14/30 = 46.67%` strict Accuracy。
- v3 相对 Direct：`+26.67` percentage points。
- v3 相对 v2.3：`+4/30`，即 `+13.33` percentage points。

主指标始终使用固定分母 30。v3 有 29 个 evaluator-valid trajectory，
completed-only Accuracy 为 `14/29 = 48.28%`，但不作为正式主指标。

## v3 架构修改

1. parameter-level action masking：每轮只向 Director 提供当前状态下合法的
   action type 与精确 parameter domain；标量 `ADD_AGENT` 只约束下一个中性
   `agent_id`、可用 `model_id`、注册的 execution profile，contract 仍为自由文本。
2. candidate agreement/conflict、artifact freshness 与 provenance：仅从当前
   fresh artifact 做 target-blind candidate extraction，并保留 source Agent、
   artifact ID 与直接 upstream provenance；不暴露 ground truth、reward 或 winner。
3. artifact-consumption ordering：存在 fresh parseable artifact 时，优先执行能
   严格缩短 terminal path 的 relation、`SET_OUTPUT` 与显式 `FINISH`，而不是继续
   无关扩图。
4. termination lookahead：复用 `AgentGraph.construction_progress()` 计算完成
   显式终止所需的结构性 action lower bound；剩余轮数达到该边界时，只保留能
   严格降低 lower bound 的合法原子动作。
5. output parsing gate：`FINISH` 只消费当前 fresh Output artifact；不可解析输出
   记录为 `output_parsing_failure`，不重新求解、不使用 ground truth 修正。
6. SGLang runtime receipt：当 `/server_info` 中配置值
   `max_running_requests=null` 时，只读取所有 DP state 一致的
   `effective_max_running_requests_per_dp`，不猜测调度上限。

这些修改没有增加固定数学角色、固定 Agent 数量、固定 chain/parallel
topology 或数学 workflow 模板。

## 运行完整性

- explicit `FINISH`：29/30。
- `max_rounds`：0。
- terminal failure：0。
- output parsing failure：0。
- operational failure：1（`aime-2026/28` collection timeout）。
- rejected Director turn：2，rate `0.98%`；v2.3 为 30，rate `18.87%`。
- `SET_OUTPUT` 早于最后 relation：0；v2.3 为 5。
- topology：single 16、serial-2 11、serial-3-plus 1、reciprocal 1。
- Agent 数量：1 Agent 16 题、2 Agents 11 题、3 Agents 2 题。

`aime-2026/28` 在首轮及两次只补该缺口的 resume 中均达到未修改的
600-second task timeout。它保持 `final_answer=null`、Accuracy 0，不从历史
candidate 回收答案。首轮另外两个 timeout（03、24）均通过 checkpoint resume
补齐，已经成功的 trajectory 没有重采。

## 配对变化

相对 AgentGraph v2.3：

- repaired：6 题（03、06、12、23、25、27）；
- regressed：2 题（07、14）；
- unchanged correct：8 题；
- unchanged incorrect：14 题。

## 验证

- Stable Zero canary：2/2 完整通过 Direct、Director、Canvas、AgentGraph、
  explicit `FINISH`、evaluator 与 full-turn receipt。
- unit tests：1065 passed，197 subtests passed。
- `git diff --check`：通过。
- evaluator：SkillEval-compatible target-blind canonical integer Exact Match /
  Accuracy；Direct 与 AgentGraph 使用同一 extraction、canonicalization 与
  evaluator。

## 实验边界与已知限制

- 本轮没有训练、backward、optimizer update、LoRA、GRPO、MACE、Bayesian
  posterior、Skill retrieval 或 Skill evolution。
- 当前 GPU0 SGLang receipt 显示 `enable_deterministic_inference=false`。任务、
  seed、模型目录、catalog 与 evaluator 已冻结，但当前服务不能保证 bitwise
  deterministic replay。
- v3 消除了本轮的 `max_rounds`、terminal 与 parsing failure，但 AgentGraph
  API attempts 和 token 使用高于 v2.3；Task 28 仍显示长路径 runtime 问题。
- 本轮实际 topology 仍以 single 与 serial-2 为主；v3 保留自由 AgentGraph
  search space，但不以人工先验强制生成复杂 topology。
