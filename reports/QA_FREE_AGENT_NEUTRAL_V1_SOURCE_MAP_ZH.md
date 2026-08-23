# HotpotQA / TriviaQA 自由 AgentGraph 编排源映射

本条件复用 FlowSteer 的 Progressive Canvas Editing 和项目已有的通用
`minimal-neutral.v10` Director policy，不引入固定的 Agent 角色链或工作流模板。

## 共同原则

- Flow-Director 固定为 Qwen3.5-9B，但每个 Executor 的模型由 Director 从模型目录选择。
- Director 每轮只产生一个 Canvas 原子编辑；Canvas 接受后立即执行，并在下一轮返回校验与执行反馈。
- Agent 的 `contract` 是自由文本；`role_family` 只是可选元数据，不用于规定 Operator 类型或固定依赖边。
- `execution_mode` 在每个 Agent 上独立选择。`react` 只表示该 Agent 使用有界的 Tool Action--Observation 执行策略，不表示整体 AgentGraph 拓扑。
- `<answer>...</answer>` 只属于终局输出语法，不再要求独立的 Format Agent。

## 激活配置

- HotpotQA：`config/evaluation_hotpotqa_unified_architecture_v1.yaml`
- TriviaQA：`config/evaluation_triviaqa_unified_architecture_v2.yaml`

两个配置共同使用：

- `prompt_version: agentgraph.director.minimal-neutral.v10`
- `sampling_schema_version: agentgraph.model-admissible-action-mask.v2`
- `semantic_protocol_by_source: <dataset>: none`
- `recovery_policy: default`
- `require_format_agent: false`

旧的 HotpotQA/QA semantic-lineage protocol 保留用于历史实验复现，但不进入本条件。

## 代码映射

- 简短通用 Director prompt：`src/interactive/director.py`
- Canvas 原子编辑与动态 AgentGraph：`src/interactive/agent_workflow_env.py`
- 自由 Agent 节点、两比特关系和节点级 `execution_mode`：`src/interactive/agent_graph.py`
- 节点级 reasoning/react/coding 分派：`src/interactive/agent_runtime.py`
- SkillFlow-compatible Search/Read ReAct adapter：`src/interactive/qa_tool_adapter.py`
- 终局语法与 Format Agent 解耦：`scripts/train_agentgraph_smoke.py`
- 配置约束：`src/interactive/config_loader.py`

## 回归约束

- Director prompt 长度不超过 1200 字符，且不得包含固定 Reasoner、Verifier、Formatter 模板。
- 通用 Canvas action domain 不得暴露固定 `role_constraints`、固定角色枚举或固定输出角色。
- 一个图中可以同时存在 reasoning Agent 和 react Agent；只有 `execution_mode=react` 的节点进入 ReAct adapter。
- 启用 exact answer tag 时，任意 Director 选定的终端 Agent 都可以直接结束，不要求 Format Agent。

## HotpotQA terminal wiring r1

首次 15 条 neutral canary 完整执行了 Canvas 和 AgentGraph，但 15/15 均在
`max_rounds` 结束。代表轨迹在第 2--4 轮已经生成与参考答案一致的纯文本
Output artifact；由于 `exact_single_answer_tag` 没有从 Canvas 传入任意 Output
Agent 的执行请求，`finish` 始终不可接受。该批次保存为
`artifacts/qa_free_agent_neutral_v1/hotpotqa_failed_stable_zero_terminal_wire_v0_20260823`，
不得计入修订条件结果。

修订条件 `hotpotqa_free_agent_neutral_terminal_wire_r1` 只补齐通用终局边界：

- `AgentWorkflowEnv -> AgentRuntime -> AgentRequest` 传递
  `require_exact_answer_tag`，且只有当前 Output Agent 收到该标志；
- reasoning Output 将最终 artifact 序列化为一个非空
  `<answer>...</answer>` wrapper；ReAct/Coding Output 仍返回 SkillFlow
  `StructuredAction`，wrapper 只允许出现在 `complete.arguments.value`；
- Canvas 在成功和失败状态都公开 `finish_admissibility`，使 Director 能看到
  terminal protocol 的实测失败原因；
- `role_family=react` 被 Runtime/Canvas admission 拒绝；`execution_mode=react`
  仍可由任意职责的 Agent 独立选择。

来源边界保持不变：FlowSteer `FORMAT_PROMPT` 提供“终局序列化而不重新求解”的
边界；SkillFlow `StructuredAction.COMPLETE` 与 `BoundedAgent._validate_completion`
提供 `complete.arguments.value` 的终局提交语义。这是必要的通用 terminal
contract 接线，不是固定 Format Agent，也不规定 Reasoner、Verifier 或 Formatter
拓扑。
