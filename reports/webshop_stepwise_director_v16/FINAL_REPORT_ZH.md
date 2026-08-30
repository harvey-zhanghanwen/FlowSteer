# WebShop stepwise Director v16 正式评测报告

## 评测口径

- 数据：固定且同序的 WebShop validation 任务 `webshop:00500`–`webshop:00627`，共 128 条。
- 模型：本地 Qwen3.5-9B；Direct 与 AgentGraph 使用同一任务、环境、action budget 和 WebShop 原生 evaluator。
- 正式指标：WebShop **Average Score** 与 **Success Rate**；不使用 EM、F1、Accuracy 或 LLM judge。
- 本轮为 inference/evaluation only：未运行训练、GRPO、backward、optimizer update、LoRA、MACE、Bayesian update 或 Skill injection/evolution。

## 架构变更

v16 保留 FlowSteer 的 progressive Canvas `edit -> execute -> feedback`，并薄适配 SkillFlow/WebShop 的单次 `Action -> Observation` 环境执行边界。每个 WebShop action 后，下一次 Director observation 都包含：

- 原始任务目标；
- 最新 Agent action 及其 public observation；
- 当前 public environment state；
- 剩余 action budget；
- 当前商品及历史候选的公开 title、price、option、selected option 和 inspected tab；
- public progress、purchase precondition 与 terminal status。

ReAct 只作为 Agent 的 `execution_mode`，不是固定角色。统一 AgentGraph search space、自由 Agent 数量、free-text contract、模型选择、关系和唯一 Output Agent 均未被替换成固定购物 workflow。对已执行的重复非零 search 使用 typed `precondition_failed` receipt：不调用 WebShop、不推进环境状态；evaluator 只在 action、observation、next observation、reward、terminal、reason、feedback 和 info 全部一致时重放该 receipt，随后继续使用 WebShop 原生 transition 和 terminal reward。

## 正式结果

| 条件 | Evaluator valid | Average Score (/100) | Success Rate | 成功数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 128/128 | 33.8659 | 14.8438% | 19/128 |
| AgentGraph v13 | 128/128 | 52.7005 | 26.5625% | 34/128 |
| AgentGraph v14 | 128/128 | 53.2214 | 25.0000% | 32/128 |
| **AgentGraph v16** | **128/128** | **63.7227** | **34.3750%** | **44/128** |

v16 相比 Direct：Average Score **+29.8568** 个百分点，Success Rate **+19.5313** 个百分点。v16 相比 v14：Average Score **+10.5013** 个百分点，Success Rate **+9.3750** 个百分点，净增 12 个成功任务。

同任务 paired comparison（v16 vs v14）：Average Score 为 **41 胜 / 74 平 / 13 负**；17 个任务由失败转成功、5 个由成功转失败。WebShop environment terminal episode 从 96 增加到 113，step-limit episode 从 32 降到 15，`Back to Search` 从 88 降到 29。

## 执行闭环

- Stable Zero：通过。
- AgentGraph trajectory：128/128。
- Direct receipt：128/128。
- Paired result：128/128。
- Explicit `FINISH`：128/128。
- Collection/provider/runtime/evaluator failure：0。
- Evaluator retry：0。
- AgentGraph environment action：622；其中 581 次推进状态。
- Typed repeated-search precondition：41 次，分布于 21 个任务；均未调用 WebShop，均由修复后的 evaluator 正常重放。
- Native terminal episode：113；action-budget step limit：15。

## AgentGraph 结构

Director 自主生成 127 个 single-Agent graph 和 1 个 two-Agent serial graph。v16 的主要提升来自 stepwise Director feedback、public state retention、candidate evidence retention 和 option binding，而不是强制增加 Agent 数或 topology depth。该结果不能表述为模型已经通过训练学会复杂多 Agent topology；本轮没有训练或 Skill 注入。

## 剩余问题

128 条中 44 条满分，84 条低于满分：15 条在 action budget 内没有完成购买；69 条完成购买但商品或属性/option 未全部匹配。15 条 step-limit 中有 14 条出现 typed `precondition_failed`，其中 9 条在收到反馈后仍重复失败动作；另 1 条持续覆盖错误 option。69 条购买错误仅凭终局 reward 无法可靠自动拆分出完整的 attribute-level 类别，因此不虚构细分类数量。可直接观察的主要问题是：

1. 候选商品属性证据不足时过早 `Buy Now`，尤其是只能在 Description/Features 中确认的软属性；
2. 复合规格、尺寸、pack/count、颜色变体仍可能发生 entity/option binding 错误；
3. repeated search 被 typed precondition 拒绝后，Director 有时没有形成有效的 query reformulation 或 candidate recovery，导致剩余 action budget 耗尽；
4. stateful WebShop 的单环境写入语义使 Director 大多选择 single Agent；当前不应为追求图深度而强制并发 Agent 操作同一 environment session。

下一版应在不固定角色/chain 的前提下，将 repeated-search diagnosis 映射到可执行的 query reformulation/candidate recovery，并在 `Buy Now` 前基于公开 evidence 做 candidate/option consistency check；不能使用隐藏目标、reward 或 evaluator 信息。

### 典型 Wrong Demo

- `webshop:00500`，Average Score `0.60`：目标要求 small/easy-to-assemble/blue-coated steel/non-rusting/低于 70 美元。实际链路为 `search[...] -> click[B09GF9SSQN] -> click[Features] -> click[< Prev] -> click[Buy Now]`。候选只满足部分公开属性，Director 在缺少 coated-steel/non-rusting 的明确 evidence 时完成购买；首个失败层是 candidate attribute verification。
- `webshop:00586`，Average Score `0.075`：目标要求 medium gray long-sleeve hoodie；商品页同时公开 `color=gray` 与 `size=medium`。实际链路为 `search[...] -> click[B09FJ7C929] -> click[gray] -> click[Buy Now]`，只绑定 color，未选择可用的 `medium` size；首个失败层是 required option binding 不完整。
- `webshop:00524`，Average Score `0.00`、environment step limit：目标要求 18-ounce、low-carb、sugar-free BBQ marinade。实际链路为 `search[...] -> click[B00VEJ4N22] -> click[12 ounce (pack of 1)] -> Back to Search -> <INVALID>×4 -> click[Search] -> <INVALID>`。首个语义错误是选择与 18 ounce 明确冲突的 12 ounce option；随后重复 query 被 typed precondition 连续阻止，但 recovery 没有及时改变 Action，最终耗尽 action budget。

## 证据

- 正式机器可读报告：`reports/webshop_stepwise_director_v16/development_report.json`
- 正式 Markdown 报告：`reports/webshop_stepwise_director_v16/development_report.md`
- 本地完整 manifest：`artifacts/webshop_stepwise_director_v16/development/run_manifest.json`
- 本地完整 trajectories：`artifacts/webshop_stepwise_director_v16/development/agentgraph_trajectories.jsonl`
- 本地 paired results：`artifacts/webshop_stepwise_director_v16/development/paired_results.jsonl`
- 配置：`config/evaluation_webshop_stepwise_director_v16.yaml`
- 上游 source map：`docs/SOURCE_MAP.md`

完整 artifacts 因体积较大保留在本地 versioned artifact directory；Git 备份包含代码、配置、source map、测试以及正式汇总报告，不包含 API key 或大型 rollout 文件。
