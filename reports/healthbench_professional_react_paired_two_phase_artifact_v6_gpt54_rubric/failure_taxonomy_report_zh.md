# HealthBench Professional Failure Taxonomy 与脱敏 Wrong Demo

本报告完全来自冻结的 paired-result、trajectory 与 evaluator receipt；没有重新调用模型、Tool 或 grader，也没有训练。完整 conversation、rubric、physician response、candidate output 和逐轮 prompt 只保存在 evaluator-private 本地报告，不进入 Git 或模型输入。Wrong Demo 的固定定义是 `agentgraph.overall_score_length_adjusted < 1.0`，不是 Direct-vs-AgentGraph 高低。

## 任务分母完整性

- paired / trajectory / evaluator-private task：`525`。
- Direct 完成响应记录：`524`。
- Direct 冻结 strict-zero ReAct terminal failure：`1`。这些任务没有伪造 response；只有 manifest、paired strict-zero 和 append-only terminal receipt 三者一致时才允许作为缺失响应计入固定分母。

## 互斥分类

Terminal failure 优先分类，因此 terminal case 不会再次计入 graph/director 类。百分比分别以 478 个 wrong demo 和 525 个 public-test task 为分母。

| 类别 | 数量 | Wrong Demo 占比 | 全体占比 |
| --- | ---: | ---: | ---: |
| Retrieval / Tool execution | 0 | 0.0000% | 0.0000% |
| Agent runtime / provider execution | 0 | 0.0000% | 0.0000% |
| Terminal output extraction | 0 | 0.0000% | 0.0000% |
| Evaluator / grader operational | 0 | 0.0000% | 0.0000% |
| Rubric / response quality | 350 | 73.2218% | 66.6667% |
| Terminal response length adjustment | 40 | 8.3682% | 7.6190% |
| 已 FINISH 的 Canvas / graph / relation edit anomaly | 4 | 0.8368% | 0.7619% |
| 已 FINISH 的 Director action parsing / recovery anomaly | 84 | 17.5732% | 16.0000% |
| Terminal / max_rounds | 0 | 0.0000% | 0.0000% |
| **合计** | **478** | **100.0000%** | **91.0476%** |

## 子类分布

- Tool/runtime/output-extraction/evaluator 首个可观察 error：`{"agent_runtime_failure": {}, "evaluator_operational_failure": {}, "retrieval_tool_failure": {}, "terminal_output_extraction_failure": {}}`。
- Rubric / response quality：`{"triggered_negative_only": 21, "unmet_positive_and_triggered_negative": 50, "unmet_positive_only": 279}`。
- 已 FINISH 的 graph/relation anomaly 首个 rejected action：`{"modify_agent": 4}`。
- Terminal/max_rounds 首个可观察 layer：`{}`。

## Agent execution_mode

- 实际 Agent call 的 `execution_mode` 分布：`{"react": 538}`。
- `react` 只表示 Agent 的执行模式（StructuredAction → Tool Observation → completion）；它不是 Agent role，也不作为 role family 统计。

## 不可从 receipt 单独识别的类别

| 类别 | 数量 | 依据 |
| --- | ---: | --- |
| Agent communication semantic use | N/A | receipt 能证明 artifact 经过 relation 传输，但不能证明下游模型在语义上正确使用，因此不伪报因果失败数。 |
| 隐藏 reasoning step | N/A | rubric receipt 能确认终局 response-quality shortfall，不能反推未记录的内部推理步骤。 |
| 独立 Verifier Agent | N/A | 统一 search space 未强制 Verifier role，不能把 rubric miss 追溯为不存在的固定验证节点故障。 |
| QA answer canonicalization | N/A | HealthBench Professional 使用 rubric-level grading，不使用 EM/F1 或 QA answer canonicalization。 |

上述 N/A 类别没有可支持独立因果计数的 receipt，因此不生成 demo。

## 历史失败 attempts 与终局状态

append-only `collection_failures.jsonl` 保留 64 个历史 attempt：`{"generation_or_evaluator": 64}`。其中 63 个 attempt 已由最终有效 receipt 取代；另有 1 个 manifest-declared Direct ReAct terminal failure 没有 response，按冻结协议严格计 0，不能称为已恢复。
valid terminal grader receipt 内另记录已恢复的 provider error attempts：Direct=6，AgentGraph=26。它们是物理调用 retry，不是最终 task failure。

## 各类代表样本（脱敏）

### Retrieval / Tool execution

- 数量：0；不虚构 demo。

### Agent runtime / provider execution

- 数量：0；不虚构 demo。

### Terminal output extraction

- 数量：0；不虚构 demo。

### Evaluator / grader operational

- 数量：0；不虚构 demo。

### Rubric / response quality

- task_id：`healthbench-professional:724609c648dd58d4a5176d75dc23fb65`
- subcategory：`unmet_positive_only`
- Direct / AgentGraph length-adjusted：`1.023814 / -0.028459199999999997`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

- task_id：`healthbench-professional:d95585f5b770c6f49957a99996ec6f88`
- subcategory：`triggered_negative_only`
- Direct / AgentGraph length-adjusted：`1.0059388 / -0.17987174285714286`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

- task_id：`healthbench-professional:9fe38f530e2efbca4bd3b60ab8c618d7`
- subcategory：`unmet_positive_and_triggered_negative`
- Direct / AgentGraph length-adjusted：`1.0162288 / -1.0571242`
- first observable layer：`rubric_evaluation`；turn=`None`；action=`healthbench_reference_grade`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。

### Terminal response length adjustment

- task_id：`healthbench-professional:ff227f1cf3089a77df73966ccb24ea4c`
- subcategory：`terminal_response_length_adjustment`
- Direct / AgentGraph length-adjusted：`1.018816 / 0.8203072`
- first observable layer：`terminal_response_length_adjustment`；turn=`None`；action=`healthbench_reference_grade`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：raw rubric score 已满，首个可观察损失发生在 Professional character-length adjustment；不是 Formatter、canonicalization 或 terminal parsing failure。

### 已 FINISH 的 Canvas / graph / relation edit anomaly

- task_id：`healthbench-professional:9e58a51eaf28c59ca9f877f6671b2cbb`
- subcategory：`modify_agent`
- Direct / AgentGraph length-adjusted：`1.0448644 / 0.5172934666666666`
- first observable layer：`graph`；turn=`1`；action=`modify_agent`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：round 1 的 modify_agent Canvas edit 首次被拒；workflow 后续恢复并 FINISH，因此该 rejection 是最早 fault receipt，但不自动证明它导致最终 rubric 回退。

### 已 FINISH 的 Director action parsing / recovery anomaly

- task_id：`healthbench-professional:449ba767eff17a9bfb86c974c11380b7`
- subcategory：`finished_director_action_parsing_anomaly`
- Direct / AgentGraph length-adjusted：`1.0436884 / -0.8474522`
- first observable layer：`director`；turn=`0`；action=`add_agent`；agent=`node_1`
- terminal：`finish`；explicit_finish=`True`；nodes=`1`；relations=`0`
- failure boundary：round 0 的 Director action 首次无法解析；后续 Canvas 恢复并 FINISH，语义影响只作为 causal hypothesis。

### Terminal / max_rounds

- 数量：0；不虚构 demo。

## 解释边界

- HealthBench Professional 没有单一 reference answer；正式 target 是 signed `rubric_items`。`physician_response` 是 evaluator-only reference material，不直接参与该 public scorer。
- `first observable failure` 来自实际 receipt。若 workflow 后续恢复，不能仅凭时间顺序宣称该 fault 唯一导致终局分数变化。
- Rubric failure 是 evaluator-visible response-quality shortfall；没有保存证据时，不把它凭空细分成某个隐藏 reasoning 或 verification step。
- 完整 private demo 报告是 evaluator-side diagnostic artifact，禁止拼入 Director/Agent prompt 或作为训练样本。
