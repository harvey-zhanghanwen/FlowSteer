# WebShop initial adapter v2：source map 与正式条件边界

本文件是
`docs/webshop_initial_adapter_v1_source_map.md` 的纠正补充；v1 中列出的
Dataset Adapter、RAGEN session、原生 action schema、10-step budget、WebShop
reward/evaluator 和 FlowSteer Canvas/runtime 来源均保持不变。

## v1 rejection

旧条件 `webshop_initial_adapter_v1` 在 128 条 Direct 和 17 条 AgentGraph 后停止，
不得作为正式评测。原因不是 WebShop evaluator 错误，而是原始终局 observation 的
模型可见边界不正确：

1. WebShop `web_agent_site/templates/done_page.html:20-44` 把 Purchased、Target、
   Reward 放入隐藏 HTML block，并显示 `Your score`；
2. `web_agent_text_env.py::convert_html_to_text` 将该页面展平为文本，隐藏 block 与
   score 一并进入 terminal observation；
3. SkillFlow `src/ragen_adapter.py::WebShopEnv.step` 与
   `RAGENAdapter.step` 原样转发 observation/reward/done/info；
4. v1 又把 raw terminal observation 同时用于 evaluator replay 和公共 Agent artifact，
   因而 score 进入下游 Agent、Canvas feedback 和下一轮 Director prompt。

原始 v1 artifacts 保留为 rejected diagnostic evidence，不与 v2 合并。

## v2 必要薄适配

`src/interactive/environment_execution.py` 在 WebShop environment boundary 将同一次
transition 分成两种已有语义：

- **raw observation**：逐字保留在 `evaluator_environment_trace`，供
  `src/interactive/task_evaluator.py::_evaluate_environment` 重新创建同一 goal 并精确
  replay `observation / next_observation / reward / done / info`；
- **public observation**：非终局页面保持原样；真实 WebShop 终局页只向 Agent、Tool、
  routed artifact、Canvas 和 Director 返回上游可见确认文本
  `Thank you for shopping with us!`。

该投影直接依据 WebShop 的真实 `done_page.html`，不修改 environment state、action、
done、reward 或 evaluator，也不包含样本特定条件。Direct evaluator 在 terminal transition
后立即退出，因此 raw terminal page 不会进入下一次 Direct policy call。

## Evaluation receipt 修正

`scripts/evaluate_completion_benchmark_round.py` 分开报告：

- **formal evaluator episode**：唯一决定 Average Score / Success Rate 的 replay trace；
- **full rollout environment execution**：FlowSteer execute-on-edit 期间所有 request-scoped
  episode 的 action attempts、state-advancing actions、invalid actions 和 terminal episodes。

二者不得合并：后者用于执行成本与 failure diagnosis，不能替代或重复计算官方 reward。
Direct provider receipt 也从完整 `executions[]` 聚合，不再只读取最后一个 policy call。

## 保持不变

- Director 仍只有 `ADD_AGENT / MODIFY_AGENT / DELETE_AGENT / SET_RELATION /
  SET_OUTPUT / FINISH` 六个 scalar Canvas actions；
- WebShop action 仍只有原生 `search[keywords]` 与当前 admissible `click[value]`；
- `Agent = agent_id + model_id + free-text contract`，Agent 数量、模型、独立/单向/双向
  relation、唯一 Output Agent 均由 Director 选择；
- 不预设 Searcher、Reviewer、Buyer 或固定 chain/parallel topology；
- ReAct 只是 environment Agent 的逐步 execution mode；
- GRPO、backward、optimizer、LoRA、MACE、Bayesian posterior、Skill retrieval/evolution
  全部禁用。

正式 v2 配置为 `config/evaluation_webshop_initial_adapter_v2.yaml`，输出目录为
`artifacts/webshop_initial_adapter_v2/development`，不得恢复或复用 v1 checkpoint。
