# AgentGraph Qwen3.5 smoke training

本轮只验证最小真实训练闭环，不是大规模实验：从七个对齐训练源中各按顺序取第 1、2 条样本，共 14 个 task；同题由同一冻结版本的 Qwen3.5-9B Director 采样 2 条 rollout，共请求 28 条；过滤后最多执行 1 次 optimizer update。唯一配置入口是
`config/training_agentgraph_smoke.yaml`。

## 固定边界

- Flow-Director 固定为本地 Qwen3.5-9B；模型目录与 tokenizer 目录分别配置。
- GPU 3 加载 learner，GPU 4 运行 SGLang rollout，GPU 5 加载梯度副本。
- 反向 micro-batch 依次尝试 `4 -> 2 -> 1`；不能把一次 smoke run 自动扩大为更多 task、rollout 或 update。
- 目标是 terminal-only Action-Masked One-Pass GRPO。结构、探索和 Skill 奖励均为 0；每批数据只使用一次，不称为 clipped GRPO。
- MACE、贝叶斯后验、paired probe、EVSI 和 Skill 均关闭。它们不能影响此次采样、奖励或梯度。
- Director 初始提示词保持简洁中性：只给出六个合法原子动作、当前图、Canvas 反馈和允许选择的模型 ID。不加入固定角色、workflow 模板、复杂示例、未验证 Skill、evaluator rubric 或答案。
- 每次 optimizer update 后必须把版本化 LoRA 发布到 SGLang，并用明确选择该新 adapter 的至少 1 条 canary rollout 验证。同步或 canary 失败时，checkpoint 可保留用于诊断，但本次 update 不能标记为已发布。

`max_optimizer_updates: 1` 是上限，不是伪造更新的承诺。如果有效同题组不足两条、同组奖励没有差异、receipt 校验失败或没有可信 evaluator，应保存排除原因并得到 0 次更新。

## 来源与实现状态

| 状态 | 模块 | 边界 |
| --- | --- | --- |
| 直接复用 | FlowSteer progressive Canvas、逐轮轨迹/动作记录、同题分组和 GRPO loss 基础 | 保留真实 Director prompt/response、action mask、policy version 和终局回报；不复用旧 Operator 结构奖励。 |
| 直接复用 | SkillFlow 的 Qwen3.5-9B SGLang、LoRA rank/target、梯度 checkpoint、双副本与 micro-batch 思路 | 启动参数来自其 Supervisor 路径；本项目只把物理卡映射到 3/4/5。 |
| 必要适配 | Qwen3.5 本地模型与独立 tokenizer | `start_qwen35_director_server.sh` 增加 `--tokenizer-path` 和 `--enable-multimodal`；不使用 FlowSteer 的 Qwen3-8B/vLLM 启动路径。 |
| 必要适配 | 七源顺序采样、异构 Agent 模型目录和 evaluator gate | 所有 14 个 task 都可采 rollout；只有 evaluator 有效且满足 on-policy receipt 的轨迹能进入梯度。 |
| 项目算法新增 | 自由文本 Agent contract、每节点模型选择、两比特关系、terminal-only action-masked one-pass 目标 | 来自项目设计文档的 AgentGraph/信号隔离设计，不宣称是 SkillFlow 的 TTB/GFlowNet。 |
| 尚未实现/本轮关闭 | MACE、联合贝叶斯后验、同前缀干预、EVSI、Skill 发布/撤销 | 不为这些模块生成占位结果，也不把随机模型路由冒充探索算法。 |
| 尚未实现 | 大规模分布式训练、正式 benchmark 报告 | smoke 结果不能用作七数据集正式成绩。 |

## Update、LoRA 同步与 canary

初始 28 条 rollout 必须记录冻结的 `behavior_policy_version`。完成至多一次 optimizer update 后，checkpoint 使用新的 `adapter_checkpoint_version`，并按 `theta_smoke_step_` 前缀生成不复用的 adapter 名称。训练前版本、checkpoint 版本、SGLang 返回值、重试次数和发布时间写入 `sync_receipt.json`，两者不能用同一个模糊的 `latest` 标签代替。

本机 SGLang 的原生接口已经由安装源码确认：

1. `POST /load_lora_adapter` 接收 `lora_name`、文件系统 `lora_path` 和可选 `pinned`；
2. native `POST /generate` 用请求字段 `lora_path` 选择 adapter，但字段值是**已注册的 `lora_name`**，不是再次传 checkpoint 路径；
3. canary 必须显式携带 `{"lora_path": "theta_smoke_step_..."}`。只调用 base model、只在 OpenAI model 字符串中写未验证别名，或省略选择字段，都不能证明新权重已生效。

配置固定同步 timeout 为 120 秒、最多 3 次尝试、重试间隔 2 秒，并设置 `fail_run_on_sync_error: true`。本轮只有一次 update，SGLang 的两个 adapter slot 足够保留版本化发布；不得通过覆盖同名 adapter 隐藏版本变化。

## 模型 search space

Director 自身始终只使用 GPU 4 的本地 `supervisor_theta`；远端 API 模型只能作为 AgentGraph Executor，不能替代 Director。模板记录下面六个账号 `/v1/models` 已返回的 ID：

- 本地 `qwen3.5-9b-local`；
- `qwen3.5-flash`；
- `deepseek-v4-flash`；
- `gpt-4o-mini`；
- `grok-4-1-fast-non-reasoning`；
- `minimax-m2.5`（上游请求 ID 为 `MiniMax-M2.5`）。

Gemini 没有出现在本账号实测模型列表中，因此不进入当前目录。目录中的 `selection_weight`、`cheap_weight` 和 `fast_weight` 只是可复现的路由偏好，不是价格或延迟测量值，也不固定任何 Agent 角色。

2026-08-15 这次 frozen smoke batch 在开始前完成最小生成兼容探测，并固定使用四个 Executor 候选：本地 Qwen3.5-9B、`qwen3.5-flash`、`gpt-4o-mini`、`MiniMax-M2.5`。`deepseek-v4-flash` 当次探测超时，Grok 当次限流，所以没有在批次中途加入。它们只能在下一批重新探测并冻结一个新的 catalog version 后使用。

认证只从 `VECTOR_ENGINE_API_KEY` 环境变量读取，不写入 YAML、轨迹或说明文件。VectorEngine 的 OpenAI-compatible API base 是 `https://api.vectorengine.ai/v1`，不是 console 页面。

## 七个数据集与 evaluator gate

| 数据源 | smoke 数量 | 终局 evaluator | GRPO 规则 |
| --- | ---: | --- | --- |
| HotpotQA | 2 task / 4 rollout | QA token F1 | evaluator receipt 有效时进入。 |
| TriviaQA | 2 / 4 | alias-aware token F1 | evaluator receipt 有效时进入。 |
| AIME 2026 view | 2 / 4 | 规范化精确答案 | evaluator receipt 有效时进入。 |
| HealthBench Professional | 2 / 4 | rubric LLM judge | 可用于 smoke 信号，但当前 judge 不是官方可比配置，不能报告官方 HealthBench 分数。 |
| WebShop | 2 / 4 | 真实环境 success | 仅环境实际终止并返回有效 receipt 时进入；静态文本代理不得进入。 |
| ALFWorld | 2 / 4 | 真实环境 success | 仅环境实际终止并返回有效 receipt 时进入。 |
| SWE-bench | 2 / 4 | 官方容器测试 harness | harness 暂不可用时仍保存 rollout，但 receipt 标为无效并排除出 GRPO。不得用文本相似度代替测试。 |

以上“排除”只作用于梯度。原始 task、完整 rollout 和排除理由必须保留在训练数据中，方便后续接好 evaluator 后重跑；不能把无效轨迹静默当作 0 分样本。

## 训练数据产物

配置约定的采集目录为 `artifacts/agentgraph_smoke/data/`：

- `selected_tasks.jsonl`：14 条冻结的输入记录及源内顺序；
- `trajectories.jsonl`：最多 28 条完整 `TrajectoryRecord`，含真实 token receipt、图快照、执行调用、终局评测和排除状态；
- `grpo_groups.jsonl`：按 `(task_id, condition_id, policy_version)` 分组后的优势与 eligibility；
- `training_manifest.json`：模型/数据/evaluator 版本、请求数、有效数、失败与排除汇总。
- `sync_receipt.json`：behavior policy、checkpoint/adapter version、动态加载响应、重试和 canary 结果；无 update 时应明确记录 `not_attempted_no_optimizer_update`。
- `post_update_trajectories.jsonl`：至少 1 条明确指定新 adapter 的 canary 完整轨迹；只有发布成功时生成有效记录。

trainer 在 output 根目录另外写出：

- `grpo_batch.jsonl`：传入 one-pass 目标的逐轨迹摘要与 advantage；
- `training_summary.json`：实际 optimizer update 数、loss、grad norm、receipt 最大偏差和排除理由；
- `checkpoint_final/supervisor_lora/step_.../theta/`：仅在确实完成 1 次更新时存在的版本化 Director LoRA。

数据文件是结果证据，checkpoint 是模型产物，两者不能混称。未经真实执行时，配置中的路径只是产物契约，不代表文件已经生成。

## 启动与静态验证

先复制模型目录模板并只通过环境变量提供凭据：

```bash
cp config/model_catalog.yaml.example config/model_catalog.yaml
export VECTOR_ENGINE_API_KEY='...'
```

查看 Qwen3.5 启动参数不会占用 GPU：

```bash
scripts/start_qwen35_director_server.sh --help
```

确认 GPU 4 空闲后才能启动 Director；默认本地路径可分别用 `QWEN35_9B_MODEL_PATH` 和 `QWEN35_9B_TOKENIZER_PATH` 覆盖：

```bash
scripts/start_qwen35_director_server.sh
```

只做静态配置检查、不启动训练：

```bash
python3 scripts/validate_agentgraph_setup.py \
  --config config/training_agentgraph_smoke.yaml \
  --allow-example-catalog
python3 -m unittest tests.unit.test_config_loader -v
```

运行固定的 7×2 smoke transaction：

```bash
python3 scripts/train_agentgraph_smoke.py \
  --config config/training_agentgraph_smoke.yaml
```

入口会严格拒绝扩大 task、rollout 或 optimizer update 边界。下一批不能直接把本脚本再次当作 base-policy step 1 使用；必须先把 live behavior adapter、previous adapter、policy version 与 step 2 参数化，再冻结新的 Executor catalog，避免混用 base 和更新后的 policy。

## 2026-08-15 实际运行结果

本次不是离线 loss 模拟，完整 transaction 已成功结束：

- 14 个 task、28 条 frozen behavior rollout 全部落盘，七源各 4 条；22 条 evaluator receipt 有效，18 条通过完整 GRPO gate。
- 11 个精确可训练组中，1 组的两条 rollout 产生非零同题相对优势；其余 10 组同奖，按 zero-information 排除。SWE-bench 的 4 条轨迹因无容器 harness 保留但排除；WebShop 和 ALFWorld 各有 1 条环境 callback 失败并保留为 invalid。
- GPU 3/5 完成 1 次真实 LoRA `optimizer.step()`：loss `0.0455208346`，grad norm `1.3975842`，LoRA 参数更新 L2 `0.0270686074`；behavior log-prob 最大复算偏差 `0.1116111`，低于 `0.25` gate；micro-batch 为 4，未发生 OOM。
- checkpoint 发布为 `theta_smoke_step_000001`，逻辑 policy 从 `qwen35-9b-base-step-0000` 更新到 `qwen35-9b-smoke-step-0001`。SGLang pause/drain、adapter load、server canary 均一次成功。
- 额外 1 条完整 post-update trajectory 的所有 Director turn 都明确记录新 policy、`theta_smoke_step_000001` 和 SGLang server weight receipt；runner 最终状态为 `completed`。

实际证据位于 `artifacts/agentgraph_smoke/`。其中 `data/trajectories.jsonl` 是 28 条训练前 rollout，`grpo_batch.jsonl` 是进入目标计算的逐轨迹摘要，`training_summary.json` 证明 optimizer update，`data/sync_receipt.json` 与 `data/post_update_trajectories.jsonl` 证明热同步及更新后采样。该 smoke 结果只证明训练闭环，不作为七数据集正式 benchmark 成绩。
