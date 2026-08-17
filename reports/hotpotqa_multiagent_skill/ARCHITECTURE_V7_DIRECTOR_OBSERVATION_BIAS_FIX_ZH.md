# HotpotQA Director 多轮观察偏置检查与修正报告

## 结论

确认存在会把 Flow-Director 推向单 Agent、浅结构和过早 `finish` 的输入偏置。
问题不在 AgentGraph 是否支持深图，而在完整单 Agent 第一次执行后，下一轮
Director 同时收到多组语义高度一致的“可以结束”信号。

本轮已修改观察协议和中性提示词，但没有启动模型、API、rollout、训练、
backward、optimizer 或权重更新。因此只能确认输入偏置已从代码中移除，不能
宣称真实图深度分布已经改善。

## 真实 Demo 证据

固定 HotpotQA Step 0--3 共 512 条轨迹中：

- 单 Agent：476/512（92.97%）；
- 双 Agent：34/512（6.64%）；
- 三 Agent 及以上：2/512（0.39%）。

Step 3 的 128 条轨迹中：

- 127 条明确 `finish`；
- 127 次出现格式合法的 progressive execution；
- 其中 126 次下一动作直接为 `finish`；
- 119/128 条在第一次执行后的下一轮立即结束；
- 99/128 条严格遵循 `add_agent → set_output → finish`。

这说明最大轮数 20 并不是主要限制；策略通常主动在第 3 轮左右结束。

## 修正前下一轮输入中的叠加信号

当一个 Agent 被设为 Output 后，模型会同时看到：

1. `topology_family: single` 和结构深度 1；
2. `structurally_finishable_now: true`；
3. `minimum_remaining_actions: 1`，且 breakdown 中唯一动作是 `finish`；
4. `complete_validation.valid: true`；
5. progressive execution 被命名为 `final_answer`；
6. `exact_single_answer_tag: true`；
7. 同一 execution result 同时出现在 `canvas_feedback` 与最近历史中；
8. system prompt 明确写着 `prefer finish`；
9. system prompt 还使用了 `not to make the graph larger` 的非对称表述。

其中第 2--9 项叠加后，把“结构合法、格式合法”误强化成“任务已经充分解决”。
但按照项目 MD，只有明确 `finish` 后的 evaluator 才产生终局任务质量；格式
检查本身不是正确性或分解充分性的证据。

## 已完成修改

### 1. 对齐 SkillFlow 的历史/当前观察边界

`src/interactive/director.py` 不再把每个 history entry 的 post-action feedback
原样写回。现在历史保存“动作前观察 + 动作 + 接受状态”，最新 Canvas feedback
只作为当前观察出现一次。

依据是 SkillFlow `training/environment.py::_build_react_prompt` 与
`training/react_prompts.py` 的边界：历史 observation/action 与 current observation
分开渲染。

### 2. 移除 Director 可见的最短终止距离

下一轮 prompt 不再包含 `construction_progress`、
`minimum_remaining_actions`、`minimum_remaining_breakdown` 和
`structurally_finishable_now`。`AgentGraph.construction_progress()` 仍保留给离线诊断，
没有删除诊断能力。

### 3. 分离结构合法、格式合法和任务充分性

- `complete_validation.valid` 改为更明确的
  `graph_validation.structurally_complete`；
- progressive execution 内的 `final_answer` 改为中性的 `output`；
- `answer_protocol` 改为 `output_format`；
- 内部 `AgentRuntimeResult.final_answer`、明确 `finish`、trajectory terminal answer
  和 evaluator 均未改变。

### 4. 修正提示词中的非对称早停引导

删除：

- `prefer finish`；
- `not to make the graph larger`。

替换为中性规则：图结构由任务依赖决定，大小本身既不是收益也不是成本；
格式合法只是 terminal protocol 检查，不代表答案或分解充分；完整 singleton
可以充分，也可能仍隐藏尚未覆盖的独立依赖。

## 明确没有做的事情

- 没有规定最少 Agent 数；
- 没有要求最小深度；
- 没有固定 Researcher/Verifier/Writer 等角色模板；
- 没有强制串行、并行、fan-in/out 或双向关系；
- 没有增加结构奖励、Agent 数量奖励或模型多样性奖励；
- 没有改变六个原子动作、模型池、Executor、evaluator 或终局 reward；
- 没有使用训练或 API 调用来制造“改善”结论。

## 静态验证

- Director/AgentGraph/graph-diagnostics 定向测试：33 passed；
- 完整 `tests/unit`（排除本环境缺少 pandas 的可选数据准备测试）：227 passed；
- Ruff：通过；
- 回归测试确认最新 `execution_result` 在下一轮 prompt 中只出现一次；
- 回归测试确认 prompt 不再出现 `construction_progress`、
  `complete_validation`、Director-visible `final_answer` 或 `prefer finish`。
- 新增的 `evaluation_hotpotqa_director_observation_v7_dev128.yaml` 已通过配置
  校验，保持相同固定样本、policy/adapter、catalog 顺序和 Direct comparator，
  且训练、GRPO、optimizer、policy sync 均关闭；本轮没有执行该配置。

## 尚未验证

真实 Qwen3.5-9B Director 的单/双/三 Agent 分布是否改变，必须在相同固定任务、
相同 policy/adapter、相同 catalog 顺序、相同采样条件下，用新的 prompt/condition
版本重新跑受控对照后才能回答。旧 Step 0--3 数据保持为历史基线，不能与新输入
协议混写或续跑。
