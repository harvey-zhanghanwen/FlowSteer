# WebShop v37 全量评测与错误分析

## 1. 评测结论

本轮使用固定的 WebShop validation 面板 `webshop:00500`—`webshop:00627`，共 128 个任务。WebShop 的官方指标是 **Average Score** 与 **Success Rate**，不是 EM、F1 或静态问答 Accuracy。

| 条件 | 样本 | evaluator-valid | Average Score | Success | Success Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 128 | 128 | 32.69 / 100 | 18 | 14.06% |
| AgentGraph v37 | 128 | 128 | **62.82 / 100** | **46** | **35.94%** |
| AgentGraph − Direct | — | — | **+30.13** | +28 | **+21.88 pp** |

相对此前相同任务面板、相同 evaluator 的 AgentGraph v34（Average Score 60.71、Success Rate 35.16%，45/128），v37 的 Average Score 提高 **2.11**，Success Rate 提高 **0.78 个百分点**，净增 1 个完全成功任务。逐任务对照为 29 个分数提高、27 个分数下降、72 个分数不变；因此这是小幅净提升，不应解释为所有任务均获得稳定改进。

## 2. 完整性与终止语义

- AgentGraph：128/128 完成、128/128 evaluator-valid、128/128 显式 `FINISH`。
- WebShop 环境自然终止 110 个，环境动作预算耗尽 18 个。
- 744 个正式环境动作全部推进状态；invalid action、environment timeout、provider failure、runtime failure、evaluator failure、collection failure 均为 0。
- `terminal failure=0`、Director `max_rounds=0`。18 个 environment step-limit 有合法 `FINISH` 和 evaluator receipt，不属于 AgentGraph terminal failure。
- Direct 复用 v34 同一固定面板的完整回执，没有重复推理；Direct 共 894 个动作，其中 18 个 invalid action、65 个 environment step-limit。

## 3. AgentGraph 结构

| 结构 | 任务数 | Average Score | Success Rate |
| --- | ---: | ---: | ---: |
| single Agent | 52 | 59.95 | 32.69%（17/52） |
| multi-Agent | 76 | 64.79 | 38.16%（29/76） |
| serial-2 | 69 | 66.63 | 42.03%（29/69） |
| serial-3-plus | 2 | 63.33 | 0% |
| mixed | 3 | 66.67 | 0% |
| fan-in | 2 | 0.00 | 0% |

这些数字是 Director 自主选择结构后的描述性统计，不是随机对照实验。当前 multi-Agent 的总体指标高于 single Agent，但 69/76 个 multi-Agent 任务仍是 `serial-2`；非串行 topology 只有 5 个任务，样本不足，且不能据此得出非串行结构本身降低分数的因果结论。

## 4. v37 的架构修复

v37 保留 FlowSteer 的 Canvas `edit → execute → feedback`、directed relation、progressive output、unique Output Agent 与 trajectory receipt，并保留 SkillFlow/WebShop 的逐动作 `Action → Observation → 下一次决策`、原生环境终止、reward 和 evaluator。本轮只对 WebShop environment adapter 做必要兼容修复：

1. measurement 硬约束只绑定到对应的 measurement-bearing option dimension；`for our 60 inch TV` 一类 compatibility measurement 不再错误绑定到无关的 `style name`。
2. color requirement 只抽取紧邻 `color`、`colour`、`colored` 或 `coloured` 的有界短语，并保留同一可见 option 内的 exact-first 匹配。
3. `purchase admissible` 同时要求保留 product context 且当前页面可见 `Buy Now`；跨页面可达性仍由 `minimum_actions_to_purchase` 表达。

这些逻辑只使用原始 instruction、当前公开 observation、native admissible actions 与既有公开 Action–Observation receipts；不向 Director 或 Agent 提供 hidden target、reward 或 evaluator state。未预设固定购物角色、Agent 数量或 topology。

## 5. 非满分任务分类

128 个任务中有 82 个非满分任务。以下分类按首个可从正式 evaluator trace 与公开 receipts 复核的主要失败层互斥归类：

| 失败类型 | 数量 | 占非满分任务 |
| --- | ---: | ---: |
| option / variant binding | 49 | 59.76% |
| environment action-budget exhaustion，未购买 | 18 | 21.95% |
| primary evidence 检查后仍发生 candidate–attribute mismatch | 8 | 9.76% |
| requirement coverage 缺失导致 premature purchase | 7 | 8.54% |

option / variant binding 内部涉及 color 23 个、size/measurement 30 个、style 11 个；这些子标签可以重叠，不能相加作为分母。

### 代表案例 A：option / color binding

- Task：`webshop:00500`
- 指令：寻找易安装、蓝色涂层钢架、防锈且低于 70 美元的小边桌。
- AgentGraph：`ADD_SUBGRAPH → SET_OUTPUT → SET_RELATION → CONTINUE × 4 → FINISH`，`serial-2`。
- 环境链路：`search[small end table blue coated steel frame easy assemble under 70] → click[next >] → click[B09GF9SSQN] → click[features] → click[< prev] → click[buy now]`。
- 公开 Observation 已确认标题中的 Blue，Features 也支持 steel frame；但 public requirement ledger 只物化了 `steel frame`，没有建立 blue 的 option/attribute binding，最终购买 receipt 为 `options={}`。
- 官方结果：Average Score 0.60，Success 0。
- 首个可观察失败层：instruction constraint grounding / option binding；后续 purchase gate 在缺少 blue binding 的情况下放行。

### 代表案例 B：environment action budget exhaustion

- Task：`webshop:00505`
- AgentGraph：`ADD_AGENT → SET_OUTPUT → CONTINUE × 9 → FINISH`，single Agent。
- 环境链路：`search[shoes…] → candidate A → description → back → size 7.5 → features → back/results → candidate B → size 7.5`。
- 第 10 个环境动作后仍未执行 `buy now`；末态仍显示 camo option absent、non-slip evidence unverified。
- 官方结果：Average Score 0，Success 0，environment step-limit。
- 首个可证明失败层：environment action-budget exhaustion。仅凭终局 reward 不能把某个更早动作标成唯一因果错误。

### 代表案例 C：evidence ledger 与 purchase readiness 不一致

- Task：`webshop:00509`
- AgentGraph：`ADD_SUBGRAPH → SET_OUTPUT → CONTINUE × 4 → FINISH`，`serial-2`。
- 环境链路：`search[bookcase steel frame] → candidate → features → < prev → buy now`。
- 同一末态同时出现 `steel frame=unverified` 与 `Purchase readiness: admissible=True`，随后完成购买。
- 官方结果：Average Score 0.50，Success 0。
- 首个可观察失败层：public evidence ledger / purchase-readiness gate 不一致。

### 代表案例 D：requirement coverage 缺失导致 premature purchase

- Task：`webshop:00556`
- AgentGraph：`ADD_AGENT → SET_OUTPUT → CONTINUE × 2 → FINISH`，single Agent。
- 环境链路：`search[stainless steel tongue cleaners] → candidate B002YTTVAU → buy now`。
- 原指令中的 stainless steel 与 rid of bad breath 没有形成 attribute-evidence obligation，Agent 未检查 Description/Features 即购买。
- 官方结果：Average Score 0.667，Success 0。
- 首个可观察失败层：instruction requirement coverage；错误继续传播到 purchase readiness。

其他可复核残留包括：`00503` 已抽取 width=52 inch，但选择 `52\"W×63\"L` 而非目标 variant；`00596` 的 width=2 inch、height=64 inch 仍为 unverified，却被判定可购买；`00546` 已正确绑定 color 与 neck/sleeve，但第三个公开 option `big` 未物化为 option group，最终 Average Score 0.833。

## 6. 报告层缺陷与修复

首次生成的 82 条 `wrong_demo_diagnosis` 被错误标为 `canvas_action_rejected`，而全部对应 trajectory 的 `rejected_turn_count=0`。原因是离线诊断器在完整序列化 Canvas feedback 中搜索单词 `rejected`，误命中了嵌套字段 `tool_action_output_rejected=false`。

现已将判定改为只识别 Canvas 顶层拒绝回执前缀，并增加回归测试。修复后 82 条派生 Wrong Demo receipt 为：64 条 `environment_outcome`、18 条 `environment_termination`，`canvas_action_rejected` 为 0。该修复没有更改 trajectory、环境 reward 或正式指标，也没有发起新的模型调用。

## 7. 训练与 Skill 状态

本轮 `training_enabled=false`，GRPO、backward、optimizer update、LoRA、policy sync、MACE、Bayesian update、Skill retrieval/injection/evolution 均未执行。当前结果是 Stable Zero 架构评测，不是训练后结果。

## 8. 结论

v37 解决了 v36 诊断出的 measurement scope、自然语言 color parsing 与跨页面 purchase admissibility 三项明确 adapter 缺陷，并把 Average Score 从 v34 的 60.71 提高到 62.82；但 Success Rate 只从 35.16% 提高到 35.94%。全量结果表明当前主要瓶颈已集中到 option/variant binding、instruction requirement coverage、evidence ledger 与 purchase readiness 的一致性，以及有限环境动作预算内的搜索—验证—购买闭环。由于完整 128 题已经收束，本报告不基于同一 validation Wrong Demo 自动继续修改并重跑，以避免反复适配固定评测面板。
