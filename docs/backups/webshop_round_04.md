# WebShop Round04 独立备份说明

## 范围与状态

本备份对应 WebShop 的 Round04 推理期架构适配，不包含训练、LoRA/GRPO
反向传播、optimizer update、adapter 发布或权重同步。训练相关配置均为关闭状态：
`training_enabled: false`、`grpo.enabled: false`、`max_optimizer_updates: 0`、
`policy_sync.enabled: false`。

Round04 的开发集与固定 held-out 运行仍在进行或尚未完成。本说明不预填开发集
或 held-out 的 success rate；本快照只保存当时的架构和可恢复入口，不把未完成
运行表述为结果。

- 发布分支：`backup/webshop-round04-stable-zero-arch-clean-20260819`
- 发布 commit：由本分支的 Git 历史确定；本说明不自引用 commit ID
- 开发集正式指标：运行中/未完成，不在此预填
- 固定 held-out 正式指标：未完成，不在此预填

## 固定数据划分与正式指标

数据目录由 `config/datasets_webshop.yaml` 定义，使用 live WebShop goal inventory：

- validation：按 live `server.goals` 的顺序取前 128 个 goal index；
- train：从余下候选按顺序取 512 个；仅在训练候选不足时在训练部分循环，不允许
  validation 目标进入首次训练候选；
- 当前全量 live inventory 足以提供 512 个唯一训练目标时，不产生循环样本；
- 每条记录固定保存 `env_config.goal_index` 和 `task_id=webshop:<goal_index>`。

正式主指标是 WebShop episode **success rate**。仅接受上游环境在 episode terminal
给出的 reward：`success = (environment_return >= 1.0)`。环境、协议或 evaluator 不可用
时记录为 invalid，不使用文本相似度、LLM judge 或其他代理指标替代。

## 上游复用与必要适配

### 直接复用

1. **WebShop 环境和目标身份**

   - SkillFlow：
     `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`
     - `_check_webshop`（144--153）
     - `WebShopEnv.reset`（497--523）
     - `WebShopEnv.step`（525--538）
     - `RAGENAdapter.reset` / `RAGENAdapter.step`（556--596）
     - `RAGENAdapter._reset_webshop`（624--663）
   - WebShop：
     `/home/test/datasets/WebShop/web_agent_site/envs/web_agent_text_env.py`
     - `SimServer.__init__`（276--335）：构建并固定 shuffle 后的 `goals`
     - `WebAgentTextEnv.reset`（240--258）：以 session index 重置任务
     - `WebAgentTextEnv.step`（86--128）：执行 `search[...]` / `click[...]`
     - `get_available_actions`：公开当前 search bar 与可点击项
   - 目标文本生成：
     `/home/test/datasets/WebShop/web_agent_site/engine/goal.py:get_human_goals`
     （22--65）。

   项目通过部署的 `RAGENAdapter` reset 读取同一 `server.goals`，而非从独立静态
   文件推断目标身份：`scripts/prepare_webshop_dataset.py:157-225`。

2. **SkillFlow ReAct 的交互边界**

   - `training/environment.py:8247-8331`：RAGEN reset、任务描述与可行动作读取；
   - `training/environment.py:8437-8463`：动作解析失败不推进环境、保留状态并继续；
   - `training/react_prompts.py:17-28`：WebShop 有历史动作 prompt 模板。

   项目对应执行边界在 `src/interactive/task_evaluator.py:843-893`、
   `1196-1328`，并保存逐步 observation、legal actions、action、feedback、reward 与
   terminal 信息。

3. **FlowSteer 式渐进 Canvas 执行**

   - `src/interactive/agent_workflow_env.py:278-420`：每次已接受 Canvas edit 后在
     `execute_on_edit=true` 下执行当前图；
   - `src/interactive/agent_workflow_env.py:437-508`：将 execution result、Output Agent
     inbox 和 agent artifacts 作为 Canvas feedback；
   - `src/interactive/director.py:27-37`、`376-507`：`add_subgraph` action space 与
     多轮 Canvas observation/continuation。

### 必要适配

1. **128/512 对齐记录。** `scripts/prepare_webshop_dataset.py:254-335` 将上游 live
   `server.goals` 枚举顺序写成项目统一 TaskRecord/JSONL 格式，并使用共享的
   held-out-first 切分器；这不是 WebShop 或 SkillFlow 原生数据切分接口。

2. **随机数边界。** `prepare_webshop_dataset.py:203-206` 与
   `task_evaluator.py:1026-1038` 在首次 RAGEN reset 前设置固定 `env_seed`。原因是
   WebShop `get_human_goals` 会采样价格上界，而上游 RAGEN adapter 没有在该边界显式
   设置 Python RNG。该适配只用于保证 prepared goal index 与 live reset 一致。

3. **协议与目标锁定。** `task_evaluator.py:1106-1194` 核对 `goal_index`、完整 catalog
   配置、路径、`env_seed` 和 goal instruction；不匹配即 invalid。它是项目评测的
   identity adapter，不替换上游环境或终局回报。

4. **AgentGraph 到 ReAct 的逐步桥接。**
   `scripts/train_agentgraph_smoke.py:361-388` 只向 Flow-Director 说明必要运行接口；
   `rollout_collector.py:940-995` 保留不可变 TaskRecord，但允许该 runtime context；
   `train_agentgraph_smoke.py:1793-1849` 在 Director `finish` 后以同一 finalized graph
   处理每一个环境 step。SkillFlow 原生路径是单一 ReAct policy 的逐步调用，而非
   AgentGraph，因此这是 FlowSteer Canvas 与 SkillFlow RAGEN 之间的最小兼容层；它不
   固定 Agent role、拓扑、模型或 Skill。

5. **环境 prompt/action parser。** `task_evaluator.py:932-1004` 是对 SkillFlow
   `WEBSHOP_TEMPLATE` 和 ReAct action syntax 的薄移植，不是直接调用
   `_render_webshop_prompt`。它额外接受单个 `<action>...</action>` 封装，以兼容 Output
   Agent 的终端文本；不得表述为逐字复用上游模板。

## Skill gate

Round04 配置为 `skills.enabled: false`、`library_version: none`。当前没有随机化 paired
intervention 与确认性证据，因此没有 ACTIVE Skill；开发集 Direct 与 AgentGraph 的比较
只能作为架构证据，不能作为 Skill activation 的因果证据。任何后续 Skill 只能在项目
既有 Bayesian posterior / evidence gate 给出满足条件的 paired evidence 后激活。

## 恢复入口与外部运行时依赖

恢复前需提供下列已有外部资源；本备份不包含它们：

- SkillFlow deployed adapter：
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`
- WebShop checkout：`/home/test/datasets/WebShop`
- WebShop full catalog：`items_shuffle.json`、`items_ins_v2.json`、
  `items_human_ins.json`
- WebShop Lucene search index（由 `WEBSHOP_SEARCH_INDEX_PATH` 指向）
- 本地 Qwen3.5-9B Supervisor/SGLang 服务（Round04 配置默认
  `http://127.0.0.1:8015/v1`，served model `supervisor_theta`）
- 项目 `.env` 与现有 provider/model router 配置；凭据不写入本备份。

以下命令是恢复入口示例，均不启动训练：

```bash
cd /ssd1/iclr/1/FlowSteer
export SKILLEV_FORMAL_RUNTIME=1
export SKILLRL_WEBSHOP_PATH=/home/test/datasets/WebShop
export WEBSHOP_FILE_PATH=/home/test/datasets/WebShop/data/items_shuffle.json
export WEBSHOP_ATTR_PATH=/home/test/datasets/WebShop/data/items_ins_v2.json
export WEBSHOP_HUMAN_ATTR_PATH=/home/test/datasets/WebShop/data/items_human_ins.json
export WEBSHOP_SEARCH_INDEX_PATH=/ssd1/iclr/.private/skillflow-resources/webshop-search/indexes
export PYTHONPATH=/ssd1/iclr/1/FlowSteer

# 仅准备固定数据记录；不调用模型或 API。
python scripts/prepare_webshop_dataset.py --catalog config/datasets_webshop.yaml

# 使用冻结的 Round04 配置。先执行 canary，再按主线批准执行开发/held-out。
python scripts/evaluate_completion_benchmark_round.py \
  --config config/development_webshop_round_04.yaml --canary-only
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_webshop_round_04.yaml --canary-only
```

注意：部署的 SkillFlow `RAGENAdapter` 将 WebShop server 缓存在进程内，其 cache key 不含
`env_seed`。因此同一进程中不得混用不同的 WebShop catalog 或 `env_seed`；Round04 已冻结
为一个 catalog/version 和 `env_seed`，恢复后也必须保持一致。

## 包含与不包含

本独立备份应包含：WebShop Round04 的 catalog、开发/评测配置、数据 preparer、
AgentGraph/RAGEN 适配代码、相关单元测试、此说明及完成后的聚合报告。

不包含：`.env`、API key、模型权重/adapter、原始数据与 search index、运行中的 raw
trajectory/artifact、中间缓存，以及任何训练产物。
