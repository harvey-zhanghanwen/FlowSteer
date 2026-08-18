# HotpotQA + TriviaQA 联合架构与小规模训练报告

## 1. 实验范围

本轮只覆盖 HotpotQA 与 TriviaQA，不包含其余五个数据集，不进行大规模训练。

- 每个数据集固定使用 32 条 held-out validation 样本。
- 两个 policy step 使用完全相同的 task ID、官方 answer evaluator、Qwen3.5-9B Director、Model Catalog、Director prompt、采样参数和终局协议。
- HotpotQA evaluator：`hotpotqa.official.answer.v1`
- TriviaQA evaluator：`triviaqa.official.answer.v1`
- Direct Local Baseline 在所有 policy step 之间复用，不重复采样。
- EM/F1 均使用严格 32 题分母；无效或未终局的 trajectory 不从分母中删除。
- 这些是项目固定 validation-32 结果，不是数据集官方 test set 成绩。

## 2. 架构适配

### 直接复用

- SkillFlow：Qwen3.5-9B/SGLang/Supervisor 边界、LoRA adapter 加载与发布、public `RetrievalIndex.search/read`。
- FlowSteer：progressive Canvas editing、atomic actions、`execute_on_edit`、AgentGraph runtime、trajectory 与 evaluator receipt。
- 现有项目：Format Agent、`<answer>...</answer>` 终局协议、official answer evaluator、action-masked GRPO、optimizer state continuation、OOM micro-batch backoff。

### 必要适配

- HotpotQA/TriviaQA 各一题组成一个固定 GRPO batch，每题 8 条 rollout。
- TriviaQA 在 Direct 与 AgentGraph 间共享相同的、答案不可见的 public retrieval context。
- write-once 联合训练 schedule/cursor。
- 双数据集固定 task ID、evaluator receipt、policy version 与逐 turn adapter receipt 的联合曲线聚合。
- policy 更新后分别用 HotpotQA 与 TriviaQA 做 post-update canary。

### 尚未启用

- `skills.enabled=false`：没有 Skill Library retrieval、Skill induction 或 Skill deployment。
- 没有启用 MACE 或 Bayesian optimization。
- `structural_reward=0`、`exploration_reward=0`、`skill_usage_reward=0`：没有拓扑奖励，也没有强制复杂图。
- TriviaQA 当前是确定性 retrieval prefetch；尚未达到论文 Protocol 10 的交互式 search/read parity。

Stable Zero 推理链已打通：

`Question → Qwen3.5-9B Director → progressive Canvas/AgentGraph → execute-on-edit → inter-agent artifact routing → Format Agent → Final Answer → official evaluator → trajectory`

Step 0 两个数据集共 64/64 条 trajectory 均完成且 evaluator receipt 有效，因此推理与评测边界达到本项目的 Stable Zero；这不等于方法已达到目标精度。

## 3. 固定验证集结果

| Policy / 方法 | HotpotQA EM | HotpotQA F1 | TriviaQA EM | TriviaQA F1 | 宏平均 EM | 宏平均 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 68.75 | 78.18 | 50.00 | 57.29 | 59.38 | 67.74 |
| Step 0 AgentGraph，zero LoRA | 71.88 | 82.81 | 40.63 | 54.36 | 56.25 | 68.58 |
| Step 1 AgentGraph，1 次联合 GRPO update | 68.75 | 79.69 | 43.75 | 53.13 | 56.25 | 66.41 |
| Step 1 − Step 0 | -3.13 | -3.13 | +3.13 | -1.23 | 0.00 | -2.18 |

主要结论：

1. Step 0 AgentGraph 在 HotpotQA 上优于 Direct，但在 TriviaQA 上低于 Direct，存在明显的跨数据集不对称。
2. Step 1 令 TriviaQA 净增加 1 个 EM，同时令 HotpotQA 净减少 1 个 EM；宏平均 EM 不变，宏平均 F1 下降。
3. 每个数据集只有 32 题，一题等于 3.125 个 EM 百分点；本轮不足以支持稳定泛化结论。
4. Step 1 之后两个数据集的联合成绩没有改善，因此不能把这次更新描述为成功优化。

权威曲线文件：

- `reports/joint_qa_curve/final/joint_qa_curve.json`
- `reports/joint_qa_curve/final/joint_qa_curve.csv`
- `reports/joint_qa_curve/joint_qa_curve_input.json`

曲线生成器已验证固定 task ID、官方 evaluator、policy version 和逐 turn adapter。环境未安装 Matplotlib，所以没有写入 PNG；JSON/CSV 是权威结果。

## 4. 真实训练闭环

### Step 1

- behavior policy：`qwen35-9b-hotpot-step-000000`
- 训练任务：HotpotQA 1 题 + TriviaQA 1 题
- rollout：每题 8 条，共 16 条
- evaluator receipt：16/16 有效
- exact GRPO groups：2
- informative groups：1
- HotpotQA group：8 条 reward 全为 1，零组内方差，按规则排除
- TriviaQA group：存在 0、2/3、1 的组内奖励差异，7 条 eligible trajectory 参与更新
- `optimizer_updates=1`
- loss：`-0.042232`
- gradient norm：`0.649876`
- `trainable_update_l2=0.027050`
- OOM backoff：0
- optimizer state：已保存
- updated policy：`qwen35-9b-jointqa-step-000001`
- adapter：`theta_jointqa_step_000001`
- SGLang route switch：成功
- post-update canary：HotpotQA/TriviaQA 各 1 条，2/2 成功，并记录新 policy 与新 adapter

需要保留的诊断：provider log-probability comparison 中 3 个 token 超过 0.25 tolerance；mean absolute delta 为 0.00991，P95 为 0.06754。它不否定本次更新，但继续训练前应复核。

### Step 2 尝试

Step 2 收集了新的 HotpotQA/TriviaQA 各 8 条 rollout，但没有执行 optimizer step：

- HotpotQA：8 条 reward 全为 2/3
- TriviaQA：8 条 reward 全为 0
- informative groups：0
- `optimizer_updates=0`
- `trainable_update_l2=0`
- 未生成 Step 2 checkpoint
- 未发布 adapter，也未运行 post-update canary
- manifest：`failed_no_optimizer_update`

这是 GRPO 对 zero-information group 的 fail-closed 行为。Step 2 不能作为新的 policy point，也不能写入训练曲线。

## 5. AgentGraph 与通信

| 诊断项 | HotpotQA Step 0 → Step 1 | TriviaQA Step 0 → Step 1 |
|---|---:|---:|
| 二 Agent | 23 → 26 | 29 → 27 |
| 三 Agent | 6 → 4 | 1 → 5 |
| 四 Agent | 3 → 2 | 2 → 0 |
| `serial_3_plus` | 8 → 5 | 2 → 5 |
| `fan_in` | 1 → 1 | 0 → 0 |
| 显式 `FINISH` | 32 → 31 | 27 → 29 |
| `max_rounds` | 0 → 1 | 5 → 3 |
| action parse failure | 4 → 5 | 7 → 7 |
| rejected action rate | 9.87% → 9.50% | 21.07% → 15.48% |

- 每条 trajectory 都发生了 inter-agent artifact routing。
- 代表样本中没有发现 upstream artifact 丢失或路由方向错误。
- 两个数据集均未产生 reciprocal relation 或 peer-draft communication。
- effective dependency status 均为 `weak`，拓扑仍以浅层二节点串行图为主。
- Step 1 后 HotpotQA 进一步变浅，TriviaQA 的三节点串行图增加；没有证据说明更深 topology 本身带来稳定收益。
- Format role 在每条验证 trajectory 中均存在，但 Format Agent 无法修复上游错误事实。

推理成本明显高于 Direct：

| Policy step | 数据集 | AgentGraph input tokens | Direct input tokens | 倍数 | AgentGraph 平均累计 latency/题 |
|---|---|---:|---:|---:|---:|
| Step 0 | HotpotQA | 1,238,021 | 54,993 | 22.51× | 12.48 s |
| Step 0 | TriviaQA | 1,177,128 | 38,055 | 30.93× | 13.73 s |
| Step 1 | HotpotQA | 1,254,403 | 54,993 | 22.81× | 15.03 s |
| Step 1 | TriviaQA | 1,077,251 | 38,055 | 28.31× | 12.65 s |

这里的 latency 是各题内部调用累计值；并发运行的墙钟时间更短。

## 6. 典型 Wrong Demo 与最早失败环节

### HotpotQA：错误 contract 被通信链传播

Task：`hotpotqa:5a76a401554299373536012b`

- Ground Truth：`Carol Lawrence`
- Step 0：`Carol Lawrence`
- Step 1：`Eartha Kitt`
- Director 在首个 Agent contract 中预置了错误实体 `Eartha Kitt`。
- Verification Agent 已发现文本中明确被描述为 “American actress” 的是 Carol Lawrence，但后续 Synthesis/Format 仍受错误 contract 锚定。
- 分类：Director 语义解析与错误传播，不是检索失败或通信丢失。

### HotpotQA：正确 artifact 已到达 Output Agent，但未终局

Task：`hotpotqa:5ae3b4d05542992f92d82349`

- Ground Truth：`Todd Phillips`
- Step 0：`Todd Phillips`
- Step 1：空
- Reader、Reasoning Agent 和 Format Agent inbox 均已包含正确答案。
- 首次 `FINISH` 被终局协议拒绝后，Director 继续 `modify_agent/delete_agent/add_agent`，最后达到 `max_rounds`。
- 分类：答案序列化与 termination recovery；检索、推理和 artifact routing 正确。

### TriviaQA：retrieval recall 不足后发生无证据推理

Task：`triviaqa:tc_9`

- Ground Truth：`Chicago`
- Step 0：`Chicago`
- Step 1：`Canada`
- top-5 passages 没有包含出生城市。
- Director/Executor 将 “Canadian context” 写入 contract，并错误推断 Augustana College 位于加拿大。
- 分类：retrieval recall failure，随后是无证据推理；通信忠实传递了错误 artifact。

### TriviaQA：答案粒度得到修复

Task：`triviaqa:tc_22`

- Ground Truth：`Ballet`
- Step 0：`dance`
- Step 1：`ballet`
- 分类：answer specificity 与 Format Agent contract 改善。

### TriviaQA：榜单歧义得到消解

Task：`triviaqa:tc_43`

- Ground Truth：`Bo Donaldson and The Heywoods`
- Step 0：`Paper Lace and Bo Donaldson and The Heywoods`
- Step 1：`Bo Donaldson and The Heywoods`
- 分类：evidence disambiguation 与 answer span extraction 改善；但问题未明确限定美国榜单，需警惕 benchmark-specific answer selection。

## 7. 是否属于 HotpotQA 过拟合

当前证据不支持“TriviaQA 初始低分是 HotpotQA 训练过拟合”的因果结论：

1. Step 0 使用 zero-initialized LoRA；adapter 名称包含 `hotpot`，但它没有经过 HotpotQA optimizer update。
2. Step 0 的跨数据集差异更符合 retrieval coverage、问题语义和答案规范化边界不同。
3. Step 1 的实际梯度完全来自一个 TriviaQA informative group；之后 HotpotQA 降低、TriviaQA EM 增加，与 negative transfer 相容，但样本与更新次数过少。
4. 没有持续多步退化、重复 seed 或更大固定验证集证据，因此不能称为 catastrophic forgetting。

更准确的结论是：平衡的 rollout 数量没有带来平衡的梯度信号；一次单任务 informative group 的 GRPO 更新产生了高方差的跨数据集泛化结果。

## 8. 下一步建议

在继续训练前，优先人工判断以下问题：

1. 参考 SkillFlow 的交互式 search/read boundary，提高 TriviaQA retrieval recall，而不是把固定 top-5 prefetch 当作完整 SkillFlow retrieval policy。
2. 参考 FlowSteer 的 terminal semantics，为已存在正确 Output Agent artifact、但 `FINISH` 被拒的情况核对既有 recovery 路径；不要新增未经来源支持的终局捷径。
3. 限制 Director contract 中预置未经证据支持的答案实体；保持提示词简洁中性，不加入固定 workflow template。
4. 继续使用 official terminal reward，但需要按上游训练策略处理 zero-information group 导致的 schedule stall；不能强行赋予 structural reward。
5. 在两个数据集都获得 informative groups 后，再使用更大的固定 held-out validation 与重复 seed 判断 generalization；当前不应扩大到大规模训练。

## 9. 关键 artifacts

- Step 1 training manifest：`artifacts/joint_qa_micro/step_000001/training_manifest.json`
- Step 1 training summary：`artifacts/joint_qa_micro/step_000001/checkpoint/training_summary.json`
- Step 1 sync receipt：`artifacts/joint_qa_micro/step_000001/sync_receipt.json`
- Step 1 post-update canaries：`artifacts/joint_qa_micro/step_000001/post_update_trajectories.jsonl`
- Step 2 zero-information groups：`artifacts/joint_qa_micro/step_000002/grpo_groups.jsonl`
- Step 2 failure manifest：`artifacts/joint_qa_micro/step_000002/training_manifest.json`
- Step 0/1 evaluation reports：`reports/joint_qa_curve/step_000000/`、`reports/joint_qa_curve/step_000001/`
- 联合训练曲线：`reports/joint_qa_curve/final/`

