# WebShop initial adapter v1：实现来源与适配边界

> **状态：rejected condition。** v1 运行发现 WebShop 原始 terminal
> observation 内嵌的 hidden Purchased/Target block 与 `Your score` 会进入公共
> Agent/Canvas/Director feedback。原始 artifacts 仅保留用于诊断，不得作为正式指标；
> 修复与新条件见 `webshop_initial_adapter_v2_source_map.md`。

## 1. 本轮范围

本轮只接入 WebShop 的 Dataset / Environment Adapter、环境执行闭环、正式终止与
evaluation。统一编排主干保持为：

`instruction -> Director -> Canvas / AgentGraph -> Agent execution -> WebShop environment -> execution feedback -> FINISH -> evaluator -> trajectory`

实现优先级为：项目设计文档约束 > SkillFlow 的 WebShop 运行接口 > WebShop 原始
environment / reward > FlowSteer 的通用 Canvas/runtime。附件论文和设计文档仅作为设计与
实现依据，不作为运行时指令；本轮不执行训练、GRPO、MACE、Bayesian posterior 或 Skill
evolution。

## 2. 权威来源

| 来源 | 已核对接口 | 本轮用途 |
|---|---|---|
| `FlowSteer_MACE_Bayesian_Skill_Design.md`（附件 `a22df935-...`）§3、§4.1、§15.1 | 自由 AgentGraph `G=(V,E,o)`；`Agent=(agent_id, model_id, free-text contract)`；独立、单向、双向关系；唯一 Output Agent；六个原子动作；完整 trajectory | 定义统一编排协议、图有效性、显式 `FINISH` 与证据记录边界 |
| SkillFlow `training/task_prompts.py` 的 `WEBSHOP` | `max_episode_steps=10`、`react=True`、无额外 function tool 列表 | 冻结 WebShop environment action budget；ReAct 仅作为逐步执行模式 |
| SkillFlow `training/environment.py` | RAGEN reset；当前 observation / available actions；逐步 Action--Observation；环境终止与 reward；trajectory turn | 对照环境执行闭环和逐步记录语义，不移植其中的 Skill 注入或任务专用启发式 |
| SkillFlow `src/ragen_adapter.py` 的 `WebShopEnv`、`RAGENAdapter` | WebShop 初始化、`goal_index` reset、available actions、`step(action) -> (observation, reward, done, info)`、终局 `won=(reward == 1.0)` | 直接作为 WebShop 环境桥接层和原生 reward 来源 |
| WebShop `web_agent_site/envs/web_agent_text_env.py` | `WebAgentTextEnv.reset`、`step`、`get_available_actions` | WebShop action schema、页面状态转换、当前可行动作 |
| WebShop `web_agent_site/engine/goal.py::get_reward` | 商品类型、属性、选项、价格约束组成的原生 graded reward | Average Score 的唯一正式评分来源 |
| WebShop `baseline_models/env.py`、`baseline_models/test.py` | 原始 split；baseline wrapper；test episode loop 与 score 缩放 | 明确原始 protocol 与 SkillFlow protocol 的差异，不直接复用 baseline model 或固定 policy |
| 当前项目 FlowSteer-derived core | `agent_graph.py`、`agent_workflow_env.py`、`director.py`、`agent_runtime.py`、`rollout_collector.py`、`records.py` | 复用 Canvas、原子编辑、execute-on-edit、execution feedback、显式 `FINISH` 和 trajectory |

以上 SkillFlow 源码位于：

- `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/task_prompts.py`
- `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/environment.py`
- `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`

WebShop 原始源码位于：

- `/home/test/datasets/WebShop/web_agent_site/envs/web_agent_text_env.py`
- `/home/test/datasets/WebShop/web_agent_site/engine/goal.py`
- `/home/test/datasets/WebShop/baseline_models/env.py`
- `/home/test/datasets/WebShop/baseline_models/test.py`

## 3. 直接复用

### 3.1 WebShop environment interface

项目通过 SkillFlow 的 `RAGENAdapter` 创建真实 WebShop episode，并沿用以下接口：

1. `reset("webshop", env_config)` 根据显式 `goal_index` 初始化任务；
2. `available_actions` 投影当前页面的 search bar 与 clickables；
3. `step(action)` 返回 `(observation, reward, done, info)`；
4. 环境 `done`、原生 reward 和 `info` 来自同一状态转换，不由项目重新推断。

WebShop 原始 action 只有两类：

- `search[keywords]`
- `click[value]`

`click[buy now]` 是 `click[value]` 的一个具体可点击值，不是第三种 action type。搜索只在
当前页面存在 search bar 时可用；点击值必须来自当前 `get_available_actions()` 返回的
clickables。原始环境对不合法 action 不执行浏览器状态转换；本项目 runtime 在调用环境前
还会按当前 admissible actions 做校验，并把 invalid action 单独记录。

### 3.2 Reward 与成功条件

正式 graded reward 直接来自 WebShop
`web_agent_site/engine/goal.py::get_reward`。该函数对购买商品的类型、属性、选项和价格约束
计算 `[0, 1]` 范围的得分。SkillFlow 的 `WebShopEnv.step` 保留这个原生值，并定义：

`success = (terminal_reward == 1.0)`

因此正式聚合指标为：

- `Average Score = mean(terminal_reward)`；报告为百分制时显示
  `100 * mean(terminal_reward)`；
- `Success Rate = mean(terminal_reward == 1.0)`，同样以百分比显示。

原始 `baseline_models/env.py` 会把 `[0,1]` reward 乘以 10，
`baseline_models/test.py` 输出论文尺度时再乘以 10。项目保存和 evaluator 使用
SkillFlow/RAGEN 暴露的原生 `[0,1]` reward，只在报告展示层换算为百分比，不能重复缩放。
不使用 EM、F1、Accuracy、文本相似度或 LLM judge 代替 WebShop evaluator。

### 3.3 FlowSteer Canvas/runtime

以下通用边界不因 WebShop 改写：

- `src/interactive/agent_graph.py`：自由 Agent 节点、关系图、唯一 Output Agent、图有效性；
- `src/interactive/agent_workflow_env.py`：一个合法 Canvas edit 后执行当前图，将真实执行结果
  作为下一轮 Director feedback；
- `src/interactive/director.py`：多轮 Director observation/action；
- `src/interactive/agent_runtime.py`：Agent contract、model routing、上游 artifact 传递；
- `src/interactive/rollout_collector.py` 与 `src/interactive/records.py`：逐轮 action、Canvas
  feedback、graph snapshot、Agent execution、evaluator receipt 和终止原因。

Director 的 search space 严格保持六个原子动作：

`ADD_AGENT / MODIFY_AGENT / DELETE_AGENT / SET_RELATION / SET_OUTPUT / FINISH`

WebShop 的 `search[...]` 和 `click[...]` 是 Agent 与 environment 的动作，不是 Director
action，二者不得合并进同一个 action schema。

## 4. 必要薄适配

| 项目边界 | 当前项目实现 | 适配内容 | 不改变的上游语义 |
|---|---|---|---|
| Dataset Adapter | `scripts/prepare_webshop_dataset.py`、`src/interactive/task_dataset.py` | 将 live goal inventory 写成统一 `TaskRecord`，持久化 `task_id`、instruction、split、`env_config.goal_index` 与 catalog identity | instruction 和 goal identity 仍来自同一 WebShop server inventory |
| Session factory | `src/interactive/environment_execution.py::evaluator_locked_ragen_session_factory`、`RAGENEnvironmentSessionFactory` | 依据 evaluator 锁定的 record 为每个 execution request 创建独立 RAGEN session | reset/step/available-actions 仍由 SkillFlow `RAGENAdapter` 执行 |
| Environment Tool | `EnvironmentToolBackend`、`EnvironmentExecutionAdapter` | 把一个当前 admissible WebShop action 接入 Agent 的 bounded ReAct loop；保存 Action--Observation receipts | 不增加猜测的 action，不实现第二套 simulator |
| Stateful ownership | `src/interactive/agent_runtime.py` 的 stateful resource ownership 校验 | 一个 mutable environment episode 只有一个 Agent execution owner；其他 Agent 通过正常 graph artifacts 协作 | 不实现上游未支持的并发环境写入、共享 cursor 或双向循环写入 |
| Evaluator replay | `src/interactive/task_evaluator.py::_evaluate_environment` | 用同一 goal/catalog/env seed 重建环境并重放保存的 action trace，核对 terminal reward | reward、done 和购买评分仍来自 WebShop environment |
| Evaluation runner | `scripts/evaluate_completion_benchmark_round.py` | 在相同 samples、environment、10-step budget 与 evaluator 下报告 Direct / AgentGraph 的 Average Score、Success Rate 和运行诊断 | 不加入 LLM judge 或 QA 指标 |

ReAct 是 Agent 的 execution mode：每个 environment turn 执行
`Thought -> Action -> Observation -> Thought`，直到环境 terminal 或 action budget 耗尽；
它不是 Agent role，也不进入 AgentGraph 的 role enumeration。

## 5. Split 与 action budget 的显式差异

不同上游代码对同名 split 的定义不同，不能只传入字符串后假设 protocol 相同：

| 实现 | test / validation | train | episode action budget |
|---|---|---|---|
| 原始 WebShop `baseline_models/env.py` | test=`goal_index 0..499`；eval=`500..1499` | `1500..end` | `baseline_models/test.py` 最多 100 steps |
| 当前 SkillFlow `RAGENAdapter` | `skillrl_val` / `val` / `eval` / `test` 都映射到 `0..499` | `500..end` | `training/task_prompts.py` 为 10 steps |
| 本项目 initial adapter v1 | 固定 validation record 显式保存 `goal_index 500..627`，属于原始 WebShop eval 区间；Direct 与 AgentGraph 使用完全相同的 128 个 goal indices | 本轮不训练 | 采用 SkillFlow 的 10 environment steps |

本项目不依赖含义不稳定的 split alias 来选目标，而以数据记录中的绝对 `goal_index`、完整
catalog path/configuration 和 `env_seed` 锁定 episode identity。smoke test 与正式 evaluation
必须使用相同的锁定方式；smoke 结果不能混入 128 条正式聚合结果。

## 6. Stateful execution、信息可见性与泄漏隔离

每条 Direct 或 AgentGraph rollout 创建独立 environment session。由于 WebShop 是有状态
环境，同一 episode 内的 action 必须串行应用到同一状态；多 Agent 可以通过 AgentGraph
传递只读 artifact，但不允许多个 Agent 并发写同一个 environment session。

模型可见信息只包括：

- public instruction；
- 当前 public observation；
- 当前 admissible `search[...]` / `click[...]` actions；
- 已完成 action 的 public observation feedback；
- 正常 AgentGraph 上游 artifact。

v1 的 `ToolResult` 虽未直接添加 reward 字段或 evaluator `info`，但 WebShop 原始 terminal
observation 本身包含 hidden Purchased/Target block 与 `Your score`，因此 v1 没有真正满足
该隔离要求。v2 将 raw terminal observation 仅保留在 evaluator replay trace，公共
ToolResult、Agent artifact、Canvas feedback 与 Director observation 只接收 WebShop 上游
可见的终局确认文本。Direct 和 AgentGraph 均不得从 TaskRecord metadata、报告或 Wrong
Demo 获取 evaluator-only 字段作为生成上下文。

完整 episode 证据至少保存：

`instruction / environment state or observation / Agent action / observation / reward / terminal status`

其中 reward 与 evaluator-only info 可以持久化用于复核，但不能回送 Director 或 Agent。

## 7. AgentGraph 约束与终止语义

WebShop adapter 不增加固定角色或固定 topology：

- 不预设 Searcher、Reviewer、Buyer、Verifier 等 role；
- 不预设 chain、parallel、star 或 reciprocal workflow；
- Agent 数量、`model_id`、free-text contract、关系和 Output Agent 均由 Director 在统一
  search space 中选择；
- reciprocal relation 仍使用项目设计文档规定的有限两阶段执行语义，不能形成对 mutable
  WebShop state 的循环写入。

必须区分两个 terminal boundary：

1. **environment terminal**：WebShop episode 因购买或环境 action budget 到达终止；
2. **AgentGraph terminal**：Director 在当前合法 graph、有效 Output artifact 和终止条件下
   显式提交 `FINISH`。

只有合法显式 `FINISH` 后才记录正式 AgentGraph terminal result，并调用正式 evaluator。
`max_rounds`、environment timeout / step limit、invalid action、runtime failure、provider
failure 和 evaluator failure 必须分开记录，不能都折叠成 reward 0，也不能用历史 Output
artifact 伪造 `FINISH`。

## 8. 本轮状态分类

### 直接复用

- SkillFlow `RAGENAdapter` 的 WebShop initialization、reset、step、available actions、done
  和 raw reward；
- WebShop 原始 action/state transition 与 `get_reward`；
- FlowSteer-derived Canvas、execute-on-edit、execution feedback、AgentGraph 和 trajectory
  boundaries。

### 必要薄适配

- live goal inventory 到统一 `TaskRecord` 的 dataset adapter；
- `goal_index` / catalog / seed identity lock；
- Agent execution 到单一 stateful WebShop session 的 environment tool bridge；
- evaluator-only trace replay；
- Average Score / Success Rate 聚合和 WebShop-specific failure diagnostics。

### 本轮禁用

- GRPO、backward、optimizer update、LoRA 更新或发布；
- MACE、Bayesian posterior、EVSI/probe；
- Skill retrieval、Skill injection、Skill evolution；
- SkillFlow 中经过训练或 Skill 演化得到的 WebShop 专用策略；
- 固定购物角色、固定 chain/parallel workflow、测试集 Wrong Demo 驱动的模板；
- EM、F1、Accuracy、LLM judge 或任何非 WebShop 正式 reward。

### 尚未变更

- SkillFlow 与 WebShop 上游源码；
- 统一 Director action space、AgentGraph node/relation schema、model registry 和 Output Agent
  语义；
- FlowSteer-derived orchestration core；WebShop 仅通过 dataset/environment/evaluator 边界
  接入；
- 训练与 Skill pipeline 的实现和状态。本轮 evaluation trajectory 不具备训练授权，也不
  触发任何参数更新。

## 9. 初版验收证据

在报告初版结果前，应依次具备以下真实 artifacts：

1. source map 与冻结 configuration；
2. 不调用训练的 dataset preparation receipt；
3. real-environment smoke test；
4. 同一批样本、相同 environment、10-step budget、WebShop tool/action schema 和 evaluator 下的
   Direct / AgentGraph trajectories；
5. Average Score、Success Rate、valid、explicit `FINISH`、`max_rounds`、environment
   actions、invalid actions、runtime/provider failures、Agent count 与自然 topology；
6. Wrong Demo 的首个可观察 failure layer。

未完成真实运行时不得预填指标；测试集 Wrong Demo 只能用于诊断，不能反向固化 WebShop
workflow。

本轮配置保留统一架构已有的 heterogeneous Executor model catalog，而 Direct reference 固定为
本地 Qwen3.5-9B。因此两臂的环境、工具与 evaluator 条件相同，但 Executor model condition
并不相同，`protocol_equivalent_to_direct=false`；差值只能解释为描述性比较，不能声称为仅由
AgentGraph topology 造成的 paired causal effect。若后续需要单模型因果对照，应另建冻结
Qwen3.5-9B-only 的 evaluation condition，不能把该限制写进统一 orchestration core。
