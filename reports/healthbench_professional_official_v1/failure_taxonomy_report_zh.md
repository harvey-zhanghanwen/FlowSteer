# HealthBench Professional Failure Taxonomy 与脱敏 Wrong Demo

本报告完全来自冻结的 paired-result、trajectory 与 evaluator receipt；没有重新调用模型、Tool 或 grader，也没有训练。完整 conversation、rubric、physician response、candidate output 和逐轮 prompt 只保存在 evaluator-private 本地报告，不进入 Git 或模型输入。Wrong Demo 的固定定义是 `agentgraph.overall_score_length_adjusted < 1.0`，不是 Direct-vs-AgentGraph 高低。

## 互斥分类

Terminal failure 优先分类，因此 terminal case 不会再次计入 graph/director 类。百分比分别以 496 个 wrong demo 和 525 个 public-test task 为分母。

| 类别 | 数量 | Wrong Demo 占比 | 全体占比 |
| --- | ---: | ---: | ---: |
| Rubric / response quality | 358 | 72.1774% | 68.1905% |
| Terminal response length adjustment | 81 | 16.3306% | 15.4286% |
| 已 FINISH 的 Canvas / graph / relation edit anomaly | 34 | 6.8548% | 6.4762% |
| 已 FINISH 的 Director action parsing / recovery anomaly | 1 | 0.2016% | 0.1905% |
| Terminal / max_rounds | 22 | 4.4355% | 4.1905% |
| **合计** | **496** | **100.0000%** | **94.4762%** |

## 子类分布

- Rubric / response quality：`{"triggered_negative_only": 25, "unmet_positive_and_triggered_negative": 62, "unmet_positive_only": 271}`。
- 已 FINISH 的 graph/relation anomaly 首个 rejected action：`{"add_agent": 2, "modify_agent": 5, "set_relation": 27}`。
- Terminal/max_rounds 首个可观察 layer：`{"director": 1, "graph": 21}`。

## 适用性为 0 或 N/A 的类别

| 类别 | 数量 | 依据 |
| --- | ---: | --- |
| Retrieval / Tool | 0 | 官方 public base condition 为 no-Tool；全部 final node 的 allowed_tools=[]，无 Tool 或 ReAct Action–Observation receipt。 |
| Agent communication transport/runtime | 0 | 没有 Agent runtime failed turn、execution error 或 message transport failure；relation construction anomaly 单列，不与 transport failure 混算。 |
| Agent communication semantic use | N/A | receipt 能证明 artifact 经过 relation 传输，但不能证明下游模型在语义上正确使用，因此不伪报因果失败数。 |
| 隐藏 reasoning step | N/A | rubric receipt 能确认终局 response-quality shortfall，不能反推未记录的内部推理步骤。 |
| 独立 Verifier Agent | N/A | 统一 search space 未强制 Verifier role，不能把 rubric miss 追溯为不存在的固定验证节点故障。 |
| Formatter / terminal output parsing | 0 | require_format_agent=false，terminal parsing failure=0；长度校正不是格式解析失败。 |
| Final evaluator / canonicalization | 0 | 最终 operational/evaluator failure=0；HealthBench 使用 rubric score，不使用 QA canonicalization。 |
| Final provider / collection | 0 | 历史 provider/collection attempts 已恢复并单列，不能计作最终 task failure。 |

上述 0/N/A 类别没有对应真实 failure receipt，因此不生成 demo。

## 历史已恢复 attempts

append-only `collection_failures.jsonl` 保留 124 个历史 attempt：`{"collect": 48, "terminal_evaluator": 24, "terminal_evaluator_retry": 52}`。这些 attempt 已被最终 receipt 取代，最终 provider/collection/evaluator operational failure 为 0。
valid terminal grader receipt 内另记录已恢复的 provider error attempts：Direct=161，AgentGraph=169。它们是物理调用 retry，不是最终 task failure。

## 各类代表样本（脱敏）

### Rubric / response quality

- task_id：`healthbench-professional:5ce259604fbfef1118840478e256e126`
- subcategory：`unmet_positive_only`
- Direct / AgentGraph length-adjusted：`0.790378 / -0.2775948`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`pharm_info_agent`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

- task_id：`healthbench-professional:21b7e2c4ac5982729923429bc25c780d`
- subcategory：`triggered_negative_only`
- Direct / AgentGraph length-adjusted：`0.9346732 / -0.3650265333333333`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`EHR-CCC`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

- task_id：`healthbench-professional:3b22335d21170ba8dfc661926255c449`
- subcategory：`unmet_positive_and_triggered_negative`
- Direct / AgentGraph length-adjusted：`1.017052 / -2.308087685714286`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`diet_expert`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

### Terminal response length adjustment

- task_id：`healthbench-professional:6daac09cf85897e495539f3255bcf14c`
- subcategory：`terminal_response_length_adjustment`
- Direct / AgentGraph length-adjusted：`1.014406 / 0.8766082`
- first observable layer：`terminal_response_length_adjustment`；turn=`None`；action=`healthbench_reference_grade`；agent=`vaccine_safety_expert`
- terminal：`finish`；explicit_finish=`True`；nodes=`2`；relations=`1`
- failure boundary：raw rubric score 已满，首个可观察损失发生在 Professional character-length adjustment；不是 Formatter、canonicalization 或 terminal parsing failure。

### 已 FINISH 的 Canvas / graph / relation edit anomaly

- task_id：`healthbench-professional:7e885f8fa08be5f5540c0f20a5791d38`
- subcategory：`set_relation`
- Direct / AgentGraph length-adjusted：`1.0312816 / -1.072177`
- first observable layer：`graph`；turn=`4`；action=`set_relation`；agent=`None`
- terminal：`finish`；explicit_finish=`True`；nodes=`4`；relations=`3`
- failure boundary：round 4 的 set_relation Canvas edit 首次被拒；workflow 后续恢复并 FINISH，因此该 rejection 是最早 fault receipt，但不自动证明它导致最终 rubric 回退。

- task_id：`healthbench-professional:5c15466f11c7b3588ad54e198a56022f`
- subcategory：`modify_agent`
- Direct / AgentGraph length-adjusted：`0.0393078 / -0.9591634`
- first observable layer：`graph`；turn=`1`；action=`modify_agent`；agent=`reader`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：round 1 的 modify_agent Canvas edit 首次被拒；workflow 后续恢复并 FINISH，因此该 rejection 是最早 fault receipt，但不自动证明它导致最终 rubric 回退。

- task_id：`healthbench-professional:35cb554c59c8313f09594c0fddb8b445`
- subcategory：`add_agent`
- Direct / AgentGraph length-adjusted：`-0.04448219999999999 / -0.07117739999999999`
- first observable layer：`graph`；turn=`5`；action=`add_agent`；agent=`clinical_synth`
- terminal：`finish`；explicit_finish=`True`；nodes=`6`；relations=`5`
- failure boundary：round 5 的 add_agent Canvas edit 首次被拒；workflow 后续恢复并 FINISH，因此该 rejection 是最早 fault receipt，但不自动证明它导致最终 rubric 回退。

### 已 FINISH 的 Director action parsing / recovery anomaly

- task_id：`healthbench-professional:b8fcd39ceb6161426de40c6508d742f1`
- subcategory：`finished_director_action_parsing_anomaly`
- Direct / AgentGraph length-adjusted：`-0.026607 / -0.9638568000000001`
- first observable layer：`director`；turn=`4`；action=`None`；agent=`None`
- terminal：`finish`；explicit_finish=`True`；nodes=`2`；relations=`1`
- failure boundary：round 4 的 Director action 首次无法解析；后续 Canvas 恢复并 FINISH，语义影响只作为 causal hypothesis。

### Terminal / max_rounds

- task_id：`healthbench-professional:ec01cbca0e677185ae31af9a89ca7bea`
- subcategory：`graph`
- Direct / AgentGraph length-adjusted：`0.6714882666666666 / 0.0`
- first observable layer：`graph`；turn=`5`；action=`set_relation`；agent=`None`
- terminal：`max_rounds`；explicit_finish=`False`；nodes=`3`；relations=`2`
- failure boundary：首个可观察 fault 位于 round 5 的 set_relation；后续持续 graph editing，最终 20 turns 内没有合法 FINISH，formal evaluator 未调用。terminal failure 为确定结果。

- task_id：`healthbench-professional:3e700f616cccc0c3cbeea24244544f27`
- subcategory：`director`
- Direct / AgentGraph length-adjusted：`1.0373968 / 0.0`
- first observable layer：`director`；turn=`8`；action=`None`；agent=`None`
- terminal：`max_rounds`；explicit_finish=`False`；nodes=`4`；relations=`5`
- failure boundary：首个可观察 fault 位于 round 8 的 None；后续持续 graph editing，最终 20 turns 内没有合法 FINISH，formal evaluator 未调用。terminal failure 为确定结果。

## 解释边界

- HealthBench Professional 没有单一 reference answer；正式 target 是 signed `rubric_items`。`physician_response` 是 evaluator-only reference material，不直接参与该 public scorer。
- `first observable failure` 来自实际 receipt。若 workflow 后续恢复，不能仅凭时间顺序宣称该 fault 唯一导致终局分数变化。
- Rubric failure 是 evaluator-visible response-quality shortfall；没有保存证据时，不把它凭空细分成某个隐藏 reasoning 或 verification step。
- 完整 private demo 报告是 evaluator-side diagnostic artifact，禁止拼入 Director/Agent prompt 或作为训练样本。
