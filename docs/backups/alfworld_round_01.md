# ALFWorld Round01 独立备份说明（草稿）

## 范围与发布状态

本备份对应 ALFWorld 的 Round01 推理期架构适配。它不包含训练、LoRA/GRPO
反向传播、optimizer update、adapter 发布或权重同步；配置中
`training_enabled: false`、`grpo.enabled: false`、`max_optimizer_updates: 0`、
`policy_sync.enabled: false`，且 `skills.enabled: false`。

开发集与固定 held-out 尚未完成正式运行，因此本说明不预填分数。

- 发布分支：待发布
- 发布 commit：待发布
- 开发集正式 success rate：未完成，不预填
- 固定 held-out 正式 success rate：未完成，不预填

## 固定数据划分与正式指标

`config/datasets_alfworld.yaml` 和 `scripts/prepare_alfworld_dataset.py` 使用部署的
SkillFlow `RAGENAdapter` 的 train inventory。候选 game file 依官方六类 task type
顺序做 family round-robin；每个被选任务均在写入前实际 reset 一次，以固定 canonical
instruction、`game_file`、game index/seed 与 50-step 上限。

- validation：从确定性候选顺序先取 128 个；
- train：从 validation 之后的候选顺序取 512 个；若训练候选不足，仅在训练候选内部循环；
- validation 样本不进入首次训练候选；
- 旧的 `data/agentgraph_v1/selected_tasks` 不作为本 Round01 的任务身份来源，已由
  `data/alfworld_v2` 的 live-inventory 对齐记录替代并归档；
- 正式主指标为 ALFWorld episode **success rate**：仅在 simulator terminal 且
  `info["won"]` 为布尔值时计为成功。环境、身份或 evaluator 不可用时为 invalid，
  不使用文本相似度、reward 大小或 LLM judge 代理。

## 上游复用与必要适配

### 直接复用

1. **RAGENAdapter 的环境生命周期**

   - `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`
     - `AlfredEnvConfig`（86--94）：读取 ALFWorld 配置；
     - `ALFWorldEnv.__init__`（185--221）：按 `(config_file, mode)` 载入官方
       `AlfredTWEnv` 的 `game_files` inventory；
     - `ALFWorldEnv.reset`（223--313）：按 `seed % num_games` 选择 game file，
       reset 后读取 canonical task instruction 与 `admissible_commands`；
     - `ALFWorldEnv.step`（338--388）：执行 action、更新可行动作，并返回 `done`、
       `won` 与环境信息；
     - `RAGENAdapter.reset` / `step`（556--596）及 `_reset_alfworld`（598--623）：
       管理 reset/step 调用及 `available_actions`。

   项目 evaluator 直接加载该部署的 adapter：
   `src/interactive/task_evaluator.py:_lock_alfworld_task`（814--839）和
   `_evaluate_ragen` 的 reset/step 边界（1040--1376）。未重新实现 ALFWorld
   simulator 或 action semantics。

2. **SkillFlow 官方 episode 边界**

   - `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/alfworld_official.py`
     - `OfficialALFWorldTask`（84--100）：固定 task/game/seed/max_steps；
     - `OfficialALFWorldResetResult`（104--126）与
       `OfficialALFWorldStepResult`（129--155）：公开 observation、admissible commands
       以及仅在 terminal 可用的成功信号；
     - `_create_pinned_env`（267--292）：验证 game identity、seed、step limit、reset
       instruction、initial observation 与 admissible commands；
     - `_OfficialEpisodeState.execute`（245--264）：逐步执行并只在 terminal 保存
       success；
     - `_OfficialOutcomeView.final_success`（340--345）和
       `OfficialALFWorldEpisodeFactory.create`（358--374）：拒绝非 terminal success，
       严格核对 public/private task 的 seed 与 max_steps。

   当前 `task_evaluator.py` 遵循相同的 terminal-only success 语义：终局只读取
   boolean `won`，无终局的 step-budget 耗尽是有效的 zero success，而非伪造终局。

3. **FlowSteer 渐进 Canvas 执行**

   - `src/interactive/agent_workflow_env.py:278-420`：每个已接受 Canvas edit 后，
     在 `execute_on_edit=true` 下执行当前 compiled graph；
   - `src/interactive/agent_workflow_env.py:437-508`：将 execution result、Output
     Agent inbox 与 agent artifacts 写回 Canvas feedback；
   - `src/interactive/director.py:376-507`：`add_subgraph` 等 action space 的多轮
     Canvas observation/continuation。

### 必要适配

1. **统一 128/512 切分。**
   `scripts/prepare_alfworld_dataset.py` 将 live inventory 转为项目统一 TaskRecord/JSONL，
   复用共享 held-out-first splitter。这是项目数据契约适配，不改变上游任务或 simulator。

2. **单 game identity 锁定。** 上游 RAGEN `reset(seed)` 有至多五次可用 game 的 retry；
   为避免 retry 换题，`task_evaluator.py:_lock_alfworld_task` 先用 inventory 将
   `game_file` 映射为唯一 index，再在 reset 后核验 `current_game_file`。这对应
   SkillFlow 官方 bridge 的 pinned game/seed 检查。

3. **canonical instruction 与 step budget 对齐。** evaluator 要求 immutable
   TaskRecord.question 等于 reset 后的 canonical instruction，且 record 中 `max_steps`
   等于评测配置的 50。原因是 SkillFlow 官方 bridge 在 `_create_pinned_env` 中同样拒绝
   instruction 或 max_steps 不一致。

4. **AgentGraph 到 ReAct 的逐步桥接。**
   `scripts/train_agentgraph_smoke.py:_workflow_problem` 和
   `src/interactive/task_evaluator.py` 仅把 observation、history 与当前 admissible-action
   list 作为每一步执行上下文交给已完成的 Canvas graph；这使 FlowSteer graph 可作为
   SkillFlow RAGEN 的 step policy。它不规定 Agent role、拓扑、模型或 Skill。

## Skill gate

Round01 的 `skills.enabled: false`、`library_version: none`。当前没有随机化 paired
intervention 与确认性证据，因而没有 ACTIVE Skill。任何未来 Skill 的激活必须经过项目既有
Bayesian posterior / evidence gate；开发集的 Direct/AgentGraph 比较不能作为 Skill 的因果
激活证据。

## 恢复入口与外部依赖

本备份不包含以下已有外部资源：

- SkillFlow deployed adapter：
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py`；
- ALFWorld repository：`/home/test/datasets/ALFWorld/repo`；
- ALFWorld data：`/home/test/datasets/ALFWorld/data`；
- 官方 `base_config.yaml`；
- 本地 Qwen3.5-9B Supervisor/SGLang 服务（默认 `127.0.0.1:8015`）及项目已有
  `.env` / model router 配置。

以下为恢复入口示例，均不启动训练：

```bash
cd /ssd1/iclr/1/FlowSteer
export SKILLEV_FORMAL_RUNTIME=1
export SKILLRL_ALFWORLD_PATH=/home/test/datasets/ALFWorld/repo
export ALFWORLD_DATA=/home/test/datasets/ALFWorld/data
export ALFWORLD_CONFIG_FILE=/home/test/datasets/ALFWorld/repo/configs/base_config.yaml
export PYTHONPATH=/ssd1/iclr/1/FlowSteer

# 仅 materialize 固定数据记录；不调用模型或 API。
python scripts/prepare_alfworld_dataset.py --catalog config/datasets_alfworld.yaml

# 使用冻结配置先执行 Stable Zero canary；完整开发/held-out 运行须由主线批准。
python scripts/evaluate_completion_benchmark_round.py \
  --config config/development_alfworld_round_01.yaml --canary-only
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_alfworld_round_01.yaml --canary-only
```

## 包含与不包含

独立备份应包含：ALFWorld catalog、开发/评测配置、live-data preparer、
AgentGraph/RAGEN 适配代码、相关单元测试、此说明及完成后的聚合报告。

不包含：`.env`、API key、模型权重/adapter、原始 ALFWorld 数据、运行中 raw
trajectory/artifact、中间缓存与任何训练产物。
