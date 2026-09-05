# WebShop v51：v16 基线、公开证据保留与多 Agent 协作接线

## 1. 基线与版本含义

本轮的效果基线仍是已完整评测、官方主指标 Average Score 最高的 v16：

| 条件 | 固定样本数 | Average Score /100 | Success Rate | evaluator-valid / FINISH |
|---|---:|---:|---:|---:|
| `webshop_stepwise_director_v16` | 128 | 63.72265625 | 34.375%（44/128） | 128 / 128 |
| `webshop_v16_recovery_v50_128` | 128 | 55.37 | 29.6875%（38/128） | 128 / 128 |

v16 评测源码是 `3ca1b3443a97dc10f0bf2da16ad1e37d8b683da9`；
附报告版本是 `921486fbbad48adb792219a302a44374399c5018`，独立分支
`feature/webshop-stepwise-director-v16-20260830` 保留不变。
v50 的最终本地提交为 `32626c8`；其分数下降，**未提升为最佳版本**。
v51 延续 v16 源码谱系及 v50 已验证的必要接口修复，再做下述有边界适配，
不能将该实现描述为未修改的 v16，也不能预先声称优于 v16。

现有实测依据：

- `reports/webshop_stepwise_director_v16/development_report.json`
- `reports/webshop_stepwise_director_v16/FINAL_REPORT_ZH.md`
- `reports/webshop_v16_recovery_v50_128/validation_report.json`
- `reports/webshop_v16_recovery_v50_128/FINAL_REPORT_ZH.md`

v50 的 30 题预算耗尽未购买、50 题未读所购商品详情但非满分、10 题读过所购商品
详情仍非满分，是观察到的行为分类，不自动等同于已证明的单一因果原因。
不能把未强制多 Agent 当成一个已经得到实验验证的低分原因。

## 2. MD 约束

依据用户 `FlowSteer_MACE_Bayesian_Skill_Design.md` 第 2.1 节、第 3.1–3.3 节：

- Agent 的核心定义仍为 `agent_id + model_id + free-text contract`。
- Director 自行决定数量、职责和关系；不添加固定 Searcher/Reviewer/Buyer 模板，
  不预置 chain、parallel、fan-out/fan-in 或 reciprocal 拓扑。
- 保留独立、单向通信，以及无状态分析节点的两 Agent 有限双向交换。
- ReAct 是 execution mode，不是角色；每个环境动作后立即交还 Director。
- 仅明确合法 FINISH 的终局结果进入原生 evaluator。
- 本轮不训练、不做 backward/optimizer update/LoRA 发布，不启用 GRPO、MACE、
  Bayesian 更新、强制 exploration 或 Skill evolution。

因此本轮可以验证“协作能力的状态接线是否可用、Director 是否自然选择使用”，
但**不能宣称模型已经通过参数学习学会多 Agent 编排**。

## 3. 实际上游调用链与适配原因

### 3.1 SkillFlow：单动作返回、完整目标和 Action–Observation 历史

实际源码：
`/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/environment.py`

- `_react_step`（约 8437 行）调用 `_ragen_adapter.step(action_str)`，保存 Turn、
  上一步 observation/action，更新 `_current_obs`，再构造下一轮输入。
- `_build_react_prompt`（约 9753 行）对 WebShop 使用全部 `_react_history`，调用
  `_render_webshop_prompt` 时传入 `task_description`、`current_observation`、
  `available_actions`、`action_history` 和 step count。上游会保留先前页面正文，
  而不是只留下“曾看过 Features”的标记。
- `_build_webshop_visible_state_block`（约 8142 行）从公开 task/page/history
  提取页面类型、价格、选项、已点击值、查询及打开过的 ASIN；不使用隐藏目标或 reward。
  该 state block 受上游开关控制，不应把薄适配写成上游默认开启。
- `_append_webshop_neutral_env_feedback`（约 8109 行）将重复查询和 Back 的结果
  作为可见历史反馈，不直接更换 query 或选择商品。该开关上游也默认关闭。

**必要适配原因：** SkillFlow 的上述输入属于一个绑定环境的 ReAct 调用；本项目
另有 Canvas Director 和无 Tool 的协作 Agent。v16/v50 的摘要加局部 continuation
无法保证先前详情正文在后续购买决策中仍可访问。因此在已有公开 receipt 边界保留
同商品的详情证据，并将当前公开状态送入已存在的推理节点请求，而不是重写购物环境。

### 3.2 SkillFlow/WebShop：动作、状态和官方评分

实际源码：
`/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`

- `RAGENAdapter.reset/step` 维护独立 episode。
- `WebShopEnv.reset`（约 496 行）初始化指定 goal session；
  `WebShopEnv.step`（约 524 行）直接执行原生环境动作并获取
  `get_available_actions()`，终局原生 reward 决定 `graded_score` 和 `won`。
- 原生 `web_agent_site/envs/web_agent_text_env.py::WebAgentTextEnv.step` 和
  `get_available_actions` 继续负责动作/可选项；环境不新增本项目的购物动作。

保留 v50 的原生重复查询执行、公开 radio group/value receipt 和最后一页 Next
兼容性修复。公开 radio 信息独立保存在 transition/receipt，**不写入原生 info**，
以保持原生 evaluator replay 的接口不变。价格和可见选项检查只是局部前置条件，
不是完整自然语言目标满足度的证明；也不增加“预算少则只准 Buy Now”的过滤。

### 3.3 FlowSteer：Canvas 执行反馈、Trajectory 与通信

实际复用的仓库文件：

- `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop`：
  `model_response -> env.step -> feedback -> TurnRecord -> next prompt`。
  原有 `TurnRecord` 同时保存模型动作、反馈、图快照和执行结果。
- `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step`：返回
  feedback/success/active，保持 progressive Canvas edit–execute–feedback 边界。
- `src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step`、
  `public_environment_state`、`model_admissible_action_types`、
  `model_admissible_action_targets`：本项目按 MD 对自由 AgentGraph 的必要适配。
- `src/interactive/agent_runtime.py::AgentRuntime._execute_block`、`_upstream`、
  `_request`：现有有向依赖、upstream artifact、有限双向草稿/修订交换，继续复用。

**必要适配原因：** 工具无关推理节点原来主要靠图前驱 artifact 获取上下文；当环境
通过 `continue` 变化、图结构却不变时，节点可能没有最新公开页面，或复用上一个
environment revision 的分析结果。v51 的受控改动是公开状态投影及相关分析缓存失效，
不是创建另一套 AgentGraph 调度器或向模型注入答案。

### 3.4 Stateful environment 的并发边界

`AgentRuntime._validate_stateful_resource_ownership` 明确要求一个环境 Tool 写者。
SkillFlow 绑定一个 episode，FlowSteer 双向块会执行两个节点的草稿和修订；若两个
节点同时写同一环境，会改变动作顺序和计数，不是上游已有的执行语义。

因此只读分析节点可以共同处理同一公开状态，并通过 Director 自行设置的关系向
环境执行节点提供证据分析；**不解除单环境写者限制，不把环境写者放进双向并发块**。
单环境写者不等于只能有一个 Agent。是否增加协作节点、怎样连接仍由 Director 决定。

## 4. v51 的有边界实现范围

1. 同一个商品的 Description/Features 等真实公开 observation 随 receipt 保留，
   区分商品身份和页面类型，避免另一件商品的证据被当成当前商品属性。
2. 每一步向 Director 返回原始完整目标、实际动作及结果、当前公开状态、下一步
   可用动作、剩余预算和可追溯历史证据；不传入隐藏目标属性、终局 reward 或 evaluator 字段。
3. 不带 Tool 的分析 Agent 可以获得当前公开状态；环境 revision 变化时，受影响的
   分析缓存失效，防止旧结论在新商品/新选项状态下继续指导购买。
4. 保持真实 relation 路由；不凭空补造 Agent communication，不强制增加 Agent 数量，
   不让每次分析额外消耗一个购物环境动作。
5. Director `minimal-neutral-scalar-stepwise.v3` 在 v2 基础上只增加公开能力说明：
   无 Tool Agent 可以分析共享公开状态并经有向关系通信；只有环境 Tool owner 能改变
   状态。仍由 Director 选择职责和关系，不规定角色数量、执行顺序或 shopping 模板。
6. 环境执行提示词保留原问题 scope，允许使用已获得的同商品证据和经图关系路由的
   消息，不要求每一步重复检索已读证据。没有将“少检索”改成未经取证直接购买的固定规则。

以上为本条件的实现范围，不是测试已通过或已取得新分数的声明。定向测试结果、
实际运行源码版本和完整 128 条指标由主线在完成后写入本条件报告。

## 5. 冻结评测条件与可恢复入口

执行配置：`config/evaluation_webshop_v16_collaboration_v51_128.yaml`。
condition：`webshop_v16_collaboration_v51_128`。

- 固定样本 `webshop:00500..00627`，`development / validation / sequential / 128`；
  相同 `env_seed=1000`、原生环境数据和 evaluator 协议，不能换成其他 split。
- 本地 Qwen3.5-9B Director，GPU4 / `http://127.0.0.1:8016/v1`；不使用 API 替代。
- Director context 32768、action tokens 1024、20 rounds、temperature 1.0、
  top_p 1.0、top_k -1；seed 20260825、concurrency 1。
- 沿用 `config/model_catalog_webshop_v16_recovery_v50.yaml`，保留实际模型目录 ID
  与 `webshop_stepwise_v16_local_catalog` 顺序条件，不为改版本名重写 catalog。
  Executor context 8192、max_tokens 4096、thinking=false；不扩充模型池。
- 相同环境动作上限 10、每题一条 rollout、task timeout 900 秒。
- 本条件明确使用 `minimal-neutral-scalar-stepwise.v3`，是 v2 加上述公开能力说明；
  v2 保持可恢复，不能把 v51 描述为提示词完全未变的条件。采样参数和预算仍不变。
- Direct 仅复用 v16 当时真正采用的同批 128 条记录：
  `/ssd1/iclr/1/.tmp/FlowSteer-webshop-stateful-action-v15/artifacts/webshop_stepwise_director_v16/development/direct_predictions.jsonl`。
  不重复模型调用，也不用后来改条件的 Direct。

新数据独立保存于：

- `artifacts/webshop_v16_collaboration_v51_128/validation/`
- `reports/webshop_v16_collaboration_v51_128/`

主线使用现有环境变量与依赖执行：

```sh
python scripts/evaluate_completion_benchmark_round.py --config config/evaluation_webshop_v16_collaboration_v51_128.yaml --prepare-only
python scripts/evaluate_completion_benchmark_round.py --config config/evaluation_webshop_v16_collaboration_v51_128.yaml
```

正式结果只用全 128 分母报告 Average Score、Success Rate，同时区分 evaluator-valid、
FINISH、max_rounds、环境预算耗尽、invalid action 和 provider/runtime failure。
自然 Agent 数量、拓扑及真实消息输入/输出从轨迹统计；没有实际发生的多 Agent 协作
不得虚构 demo。这个面板已反复用于架构开发，不能冒充未见 held-out test 的泛化分数。

## 6. 冻结前定向验证

2026-09-05 分组执行的 162 项相关单测通过：Director/gateway/profile 69 项，
runtime/environment/collaboration/evidence 93 项。旧测试中“WebShop 不返回动作域”
的断言已按本次用户要求改为原生及模型可用动作均可见，并单独复测通过。
未为单测启动评测模型调用。其他数据集不自动激活此次只读协作状态注入。

真实多 Agent 通信是否自然出现、分数是否提高，需要本轮固定 128 题结果，
不能以合成两 Agent 单测代替。轨迹现在额外保存环境内层实际 rendered_messages，
便于核实工具动作模型真正收到的上下文，而不仅是 Runtime 外层请求。

已知边界：保留详情正文不等于自动验证全部自然语言约束；分析输出不是官方证据；
单环境写者限制、原生 reward 与 FINISH 检查均未放宽。
