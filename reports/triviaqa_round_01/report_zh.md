# TriviaQA 第一轮 Progressive AgentGraph 架构验证报告

## 1. 结论

本轮已在独立分支 `dataset/triviaqa-progressive-round-01` 完成 TriviaQA 的最小必要适配，并保存固定 128 题 validation、Direct Local Baseline、Progressive AgentGraph trajectory、Agent communication、终局 evaluator receipt、Wrong Demo，以及 3 个 train 样本上的串行、fan-in 和 complex mixed topology 对照。本轮没有执行 GRPO、backward、optimizer update、LoRA 发布或 Skill 更新。

固定 128 题的严格结果为：

| 条件 | 有效样本 | Exact Match | Token F1 |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128/128 | 51.56% | 57.90% |
| Qwen3.5-9B Progressive AgentGraph | 128/128 | 52.34% | 61.80% |
| AgentGraph − Direct | — | +0.78 pp | +3.90 pp |

AgentGraph 的 F1 有明确增益，但离 80%–90% 目标仍有较大距离。当前结果只代表固定项目 validation 上、确定性 retrieval prefetch 条件下的第一轮架构验证，不等同于 TriviaQA 官方完整 dev，也不等同于 SkillFlow Formal Protocol 10 的完整模型驱动 `search/read/complete` 交互。

## 2. Architecture Completion Report

当前推理链已经接通：

`Question → SkillFlow RetrievalIndex → Qwen3.5-9B Flow-Director → Progressive Canvas Editing → AgentGraph Runtime → Agent communication → Format Agent → FINISH terminal validation → TriviaQA evaluator → Trajectory`

已完成并验证：

- 固定 128 题 validation；Direct 与 AgentGraph 使用同一批问题、同一 Qwen3.5-9B、同一公开检索观察和同一 evaluator。
- FlowSteer 风格的一次一个原子 Canvas edit；每次 accepted edit 后立即执行受影响的 subgraph，并将 execution feedback 返回下一轮 Director 输入。
- 自由文本 Agent contract、模型选择、directed relation、fan-in、fan-out、有限 reciprocal block、Output Agent 和显式 `FINISH`。
- 独立 Format Agent；终局必须是唯一、非空的 `<answer>...</answer>`。
- Agent input/output、实际 upstream message、Output Agent inbox、graph snapshot、token、latency、API call 和 evaluator receipt 持久化。
- TriviaQA accepted-answer normalized exact match 和最大 token F1。

预留或本轮禁用：

- Skill retrieval/injection、Skill evidence gate、Skill lifecycle 与 Skill update。
- MACE、Bayesian posterior、Thompson sampling、EVPI/EVSI 和 paired intervention。
- One-Pass GRPO、LoRA learner、backward、optimizer update 和 SGLang adapter synchronization。

Stable Zero 状态：预运行小样本 canary 已验证端到端链路可工作；但完整 128 题中只有 116 题显式 `FINISH`，12 题达到 `max_rounds`，因此按“所有固定样本均完成完整链”的强判据，当前完整轮次**未达到 Stable Zero**。这不影响 EM/F1 的严格分母：12 个未显式终止样本仍按 0 计入 128 题。

## 3. 上游复用与必要适配

| 模块 | 来源 | 本轮处理 |
|---|---|---|
| Retrieval | SkillFlow `RetrievalIndex.search/read` | 直接复用公共检索实现；增加 TriviaQA 的薄适配与 answer-free receipt |
| TriviaQA evaluator | SkillFlow `PrivateStaticBenchmarkEvaluator` 的答案归一化、alias 最大 F1 和 normalized EM | 按相同规则接入项目终局 evaluator |
| Qwen3.5-9B / OpenAI-compatible serving boundary | SkillFlow 的 Supervisor/serving 边界 | 使用本地 Qwen3.5-9B；未替换为外部 API Director |
| Progressive Canvas Editing | FlowSteer `step → execute → feedback → next edit → finish` | 复用项目现有 `AgentWorkflowEnv.step` 与 rollout collector |
| AgentGraph Runtime / Trajectory | FlowSteer/SkillFlow 现有执行与 receipt 边界 | 复用现有 runtime、communication artifact 和 trajectory schema |
| Format Agent | FlowSteer Format Operator 的终端格式职责 | 最小适配为自由 AgentGraph 的独立 terminal Agent |
| TriviaQA runner | HotpotQA 最新适配架构 | 在独立 TriviaQA 分支中只替换数据、retrieval、evaluator、配置与报告 |

当前 retrieval 使用一次确定性 keyword query 预取 5 个 passage。它复用了 SkillFlow retrieval implementation，但没有实现模型驱动、多轮、可继续选择 query 的完整 Protocol 10，因此 retrieval recall 是本轮的重要上限。

## 4. 128 题自然编排统计

| 统计项 | 结果 |
|---|---:|
| 1 Agent | 1/128 |
| 2 Agents | 102/128（79.69%） |
| 3 Agents | 20/128（15.63%） |
| 4 Agents | 5/128（3.91%） |
| Structural depth = 2 | 101/128（78.91%） |
| Structural depth ≥ 3 | 25/128（19.53%） |
| Fan-in topology | 3/128（2.34%） |
| Reciprocal relation | 0/128 |
| Explicit FINISH | 116/128（90.63%） |
| `max_rounds` termination | 12/128（9.38%） |

自然 Flow-Director 仍明显偏向两节点串行结构：`serial_2` 101 题，`serial_3_plus` 22 题，fan-in 3 题，parallel 1 题，single 1 题。当前不能据此断言 TriviaQA 不需要非链式 topology；更准确的结论是，未训练的 Qwen3.5-9B Director 在当前 search space 和 feedback 条件下尚未自然选择 reciprocal relation，且 fan-in 使用率很低。

## 5. 3 题 Progressive Canvas Topology 对照

三题来自 train split，仅用于 forced diagnostic probe，不进入 held-out 指标、GRPO 或 Skill evidence。每个 topology 都通过 `AgentWorkflowEnv.step` 逐个添加 Agent、关系和 Output Agent；每个 accepted non-FINISH edit 后都立即执行。9 个运行的 `each_edit_executed` 均为 `true`。

### 5.1 结构控制

| Topology | Agents | Relations | Directed edges | Structural depth | Reciprocal pairs | Fan-in |
|---|---:|---:|---:|---:|---:|---|
| Serial | 4 | 3 | 3 | 4 | 0 | 无 |
| Fan-in | 4 | 3 | 3 | 3 | 0 | `synthesis` |
| Complex mixed | 7 | 8 | 9 | 5 | 1 | 3 个 fan-in Agent |

### 5.2 具体结果

| Question | Topology | Final Answer | EM | F1 |
|---|---|---|---:|---:|
| Which British general was killed at Khartoum in 1885? | Serial | Charles George Gordon | 0 | 0.50 |
| 同题 | Fan-in | Charles George Gordon | 0 | 0.50 |
| 同题 | Complex mixed | Charles George Gordon | 0 | 0.50 |
| On the border of which two countries is Victoria Falls? | Serial | Zimbabwe and Zambia | 0 | 1.00 |
| 同题 | Fan-in | Zimbabwe and Zambia | 0 | 1.00 |
| 同题 | Complex mixed | Zimbabwe and Zambia | 0 | 1.00 |
| What is the name of the volcanic valley that runs from the Sinai peninsula to central Mozambique? | Serial | Great Rift Valley | 1 | 1.00 |
| 同题 | Fan-in | The Great Rift Valley | 1 | 1.00 |
| 同题 | Complex mixed | No such volcanic valley exists | 0 | 0.25 |

聚合结果：

| Topology | EM | F1 | Model calls | Prompt tokens | Completion tokens | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 33.33% | 83.33% | 26 | 30,666 | 1,690 | 12.94 s |
| Fan-in | 33.33% | 83.33% | 24 | 28,372 | 1,558 | 12.41 s |
| Complex mixed | 0% | 58.33% | 57 | 75,375 | 9,292 | 58.20 s |

第三题的 serial topology 首次产生 `<answer></answer>`，`FINISH` 被 terminal validation 拒绝；随后按 FlowSteer continuation 执行一个 `modify_agent` 原子编辑，立即重执行 `verification → format` dirty subgraph，再次 `FINISH` 后得到 `Great Rift Valley`。完整 rejected action、feedback、修改和两次 execution 均保存在 trajectory。

Complex mixed topology 在第三题失败的主要原因不是 communication 没有执行，而是共同 retrieval miss 经过多级 reasoning、reciprocal revision 和 synthesis 后形成 error amplification。Fan-in 的 independent candidate branch 使用模型已有事实知识恢复了答案；Complex mixed 的多个下游节点共享同一个错误前提。该结果说明 reciprocal communication 可以提高一致性，但不能替代独立证据，也不能保证正确性。

## 6. Failure Analysis

重新生成报告后，配对分类为：

| Failure type | 数量 |
|---|---:|
| `architecture_gain` | 6 |
| `architecture_regression_candidate` | 5 |
| `correct`（Direct 与 AgentGraph 均 EM=1） | 61 |
| `director_max_rounds` | 12 |
| `partial_or_overlong_answer` | 17 |
| `shared_reasoning_or_model_failure_candidate` | 27 |

最典型 Wrong Demos：

1. `triviaqa:tc_5`：检索干扰项使首个 Evidence Agent 的 contract 锚定到错误年代，错误沿串行路径传播。
2. `triviaqa:tc_30`：retrieval miss 后语义 Agent 输出 `Unknown`，Format Agent 产生空答案；Director 连续修改仍未在 `max_rounds` 内通过 terminal validation。
3. `triviaqa:tc_82`：上游事实为 Suwon Airfield，但 Format Agent 输出 `Battle of Suwon Airfield`；多余事件前缀使 EM 从 1 降为 0。
4. `triviaqa:tc_150`：Direct 输出正确的 `60 feet`，Evidence Agent 因 passage 缺失而过早 abstain，AgentGraph 最终输出 `not found`。
5. `triviaqa:tc_184`：retrieval 没有奖杯颜色事实，Reasoning artifact 明示证据不足，但终端仍产生 unsupported numeric answer，形成 evidence-to-answer consistency failure。

主要原因按证据强度排序：

- **Retrieval recall**：确定性单查询预取无法覆盖部分 TriviaQA 事实；这是 shared failure、abstention 和 `max_rounds` 的主要来源。
- **Answer canonicalization**：Format Agent 仍会保留 `RMS`、`USS`、`Battle of` 等非必要前缀；17 题只有 partial F1。
- **Termination policy**：12 题没有显式 `FINISH`，说明 terminal feedback 到下一次 Canvas edit 的恢复策略尚不稳定。
- **Search policy**：自然图 79.69% 是双 Agent，reciprocal relation 为 0；当前 policy 未充分探索 graph topology。
- **Error propagation**：更深 graph 在缺少独立证据时可能放大共同错误，复杂 topology 不能作为固定默认模板。

## 7. MD 实现覆盖

已端到端接通的是推理和评测阶段：Progressive Canvas、Flow-Director continuation、AgentGraph execution、communication receipt、Format Agent、terminal semantics、TriviaQA evaluator、trajectory 和 Wrong Demo analysis。

代码中已有但本轮关闭的是 One-Pass GRPO、LoRA trainer/policy sync、MACE/LinUCB 数值组件、Bayesian posterior/Thompson sampling/EVPI/EVSI 组件，以及 Skill schema/evidence/store/lifecycle 组件。

尚未端到端接通的是 MACE–Bayesian–Skill 双时间尺度闭环、真实同前缀 paired intervention executor、自动 probe scheduling、EVSI stop rule、held-out calibration、跨 epoch posterior 更新和自动 Skill publish。因此不能宣称 MD 已全部完成，也不能把预留配置描述为已实现训练。

## 8. Artifacts

- 固定样本：`artifacts/triviaqa_round_01/selected_tasks.jsonl`
- Retrieval receipts：`artifacts/triviaqa_round_01/retrieval_receipts.jsonl`
- Direct predictions：`artifacts/triviaqa_round_01/direct_predictions.jsonl`
- 完整 AgentGraph trajectories：`artifacts/triviaqa_round_01/agentgraph_trajectories.jsonl`
- Paired results：`artifacts/triviaqa_round_01/paired_results.jsonl`
- Wrong Demos：`artifacts/triviaqa_round_01/wrong_demos.jsonl`
- Run manifest：`artifacts/triviaqa_round_01/run_manifest.json`
- 三题 topology 对照：`artifacts/triviaqa_round_01/topology_probe3/progressive_topology_comparison.json`
- 逐条件 checkpoint：`artifacts/triviaqa_round_01/topology_probe3/progressive_topology_comparison.checkpoint.json`
- 机器可读报告：`reports/triviaqa_round_01/report.json`

## 9. 备份边界

HotpotQA 已在 TriviaQA 改动之前固定为本地分支 `backup/hotpotqa-adapted-20260817`，commit 为 `19fb4ec`，并生成独立可恢复 bundle：`/ssd1/iclr/1/FlowSteer-hotpotqa-adapted-20260817.bundle`。TriviaQA 基于该 commit 创建独立分支，没有改写 HotpotQA 备份；本轮结果另行保存为 `/ssd1/iclr/1/FlowSteer-triviaqa-round-01-20260817-v2.bundle`。GitHub remote push 同时尝试了两个独立分支，但仍受当前主机非交互认证缺失阻塞；本地 branch、commit 和 bundle 不受影响。
