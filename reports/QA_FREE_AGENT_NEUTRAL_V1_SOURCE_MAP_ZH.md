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
