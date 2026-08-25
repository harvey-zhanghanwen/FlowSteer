# AIME 2026 初版适配与全量评测报告

## 1. 结论

AIME 2026 已接入现有统一 orchestration core，没有建立独立 AIME
架构。以下推理链路已在真实模型和 evaluator 上执行并保存完整 receipt：

`AIME problem → Qwen3.5-9B Director → Canvas/AgentGraph → Agent execution/communication → explicit FINISH → AIME evaluator → trajectory`

固定 2 题 canary 的 Stable Zero 为 **2/2 通过**。30 题全量运行的
Stable Zero 为 **未通过**：29/30 题有 AgentGraph trajectory，25/30 题
合法执行 `FINISH`，4 题达到 `max_rounds`，1 题采集超时，另有 5 题
在合法 `FINISH` 后发生终局答案解析失败。

正式 Accuracy：

| Condition | Correct | Denominator | Accuracy |
|---|---:|---:|---:|
| Qwen3.5-9B Direct | 9 | 30 | 30.00% |
| AgentGraph | 4 | 30 | 13.33% |
| AgentGraph − Direct | -5 | 30 | -16.67 percentage points |

本轮没有执行 GRPO、backward、optimizer update、LoRA、MACE、Bayesian
posterior/EVSI、Skill retrieval、Skill evolution 或任何训练。

## 2. 数据

- 数据集：`MathArena/aime_2026`
- 固定 revision：`d2de22f3c656b4f56cf8981212186377d1e23bc3`
- SkillEval production source：
  `/home/test/datasets/AIME_2026/data/train-00000-of-00001.parquet`
- 原始字段及顺序：`problem_idx, answer, problem`
- 实际样本数：30，`problem_idx=1..30`
- 项目切分：train=0、validation=0、test=30
- 项目最小 schema：`task_id, problem/question, ground_truth, split, metadata`
- 任务 ID：`aime-2026/01..30`

本地 30 条题面、题号与答案已逐行对齐 production Parquet。没有复制题目、
改写题目、补齐到 128 条或混入历史 AIME。模型侧只接收题目和合法 public
metadata；`ground_truth`、`accepted_answers` 与 `evaluator_payload` 只进入
离线 evaluator。

## 3. Evaluator

正式 evaluator 复用 downstream SkillEval
`PrivateStaticTarget.score` 的 integer rule：

1. 从模型输出中接受 bare integer，或解包恰好一个完整
   `<answer>...</answer>`；
2. 多重、残缺或嵌套异常的 answer boundary fail closed；
3. 对 candidate 执行 `str(int(candidate.strip()))`；
4. 与同样 canonicalized 的 hidden ground truth 做 Exact Match；
5. 输出单题 `accuracy ∈ {0,1}`。

没有使用 HotpotQA token-F1、last-number fallback、`\boxed{}` 宽松提取、
数值容差、符号等价、LLM judge 或 ground-truth-aware repair。Direct 与
AgentGraph 使用同一 extraction、canonicalization 和 evaluator。

`evaluator_valid` 表示 evaluator 成功返回正式结果，不表示输出一定成功解析；
解析失败仍会得到合法的 0 分 receipt。

## 4. 上游复用与项目适配

### SkillFlow / SkillEval reused

- AIME 2026 数据 source plan、Parquet row schema 和 task identity；
- public/private target separation；
- static task 的空 Tool catalog；
- integer canonicalization、Exact Match 和 Accuracy。

### FlowSteer reused

- progressive Canvas 的 `edit → execute → feedback` 边界；
- graph execution、dirty closure、Agent communication 与 trajectory；
- bounded reciprocal execution；
- terminal evaluator timing；
- Direct comparator 的 `AnswerGenerate`、`ANSWER_GENERATION_PROMPT`、
  `AnswerGenerateOp` 与 `XmlFormatter.prepare_prompt`。

### Project-specific thin adaptation

- production Parquet row 到现有 `TaskRecord` 的转换；
- 单一 `<answer>` envelope 到 SkillEval `{"answer": str}` 的兼容层；
- 统一 `EvaluationOutcome` receipt；
- AIME 配置与 completion benchmark runner 接线；
- explicit-`FINISH` evaluator admission；
- `--direct-only`：只重采 Direct，并从 frozen AgentGraph checkpoint 重建
  paired report，不重复 AgentGraph 请求；
- 将 predecessor-identity PRESERVE guard 限定于 verified semantic-lineage
  protocol，使自由 AgentGraph 恢复 FlowSteer 的 progressive relation editing。

## 5. 架构完成度

| Boundary | Status | Verification |
|---|---|---|
| AIME Dataset Adapter | 完成 | 固定 30 题、顺序、ID、字段与 split 测试 |
| task-specific input/output protocol | 完成 | public/private boundary 与严格解析测试 |
| free AgentGraph `G=(V,E,o)` | 完成 | free-text contract、unique Output Agent、reachability |
| scalar Director action space | 完成 | `ADD_AGENT/MODIFY_AGENT/DELETE_AGENT/SET_RELATION/SET_OUTPUT/FINISH` |
| progressive Canvas | 完成 | 每个 accepted edit 后执行 dirty closure 并返回 feedback |
| directed / reciprocal communication | 完成 | 真实 directed、reciprocal、fan-in、parallel、mixed receipts |
| recovery | 完成 | `PRESERVE → DIAGNOSE → REPAIR → AUGMENT` runtime boundary |
| terminal semantics | 完成 | 仅 explicit `FINISH` 进入 evaluator；`max_rounds` 不回收历史 candidate |
| trajectory / Wrong Demo | 完成 | action、graph、execution、communication、tokens、latency、errors、evaluation |
| Python/calculator/symbolic Tool | 未启用 | SkillEval static AIME task 的 tool catalog 为空 |
| ReAct Tool execution | 预留但未启用 | 当前没有 admissible Tool action |
| GRPO/MACE/Bayesian/Skill/LoRA | 未启用 | 配置为 evaluation-only，optimizer updates=0 |

Director 初始 prompt 保持简洁、中性；没有加入 `Plan → Solve → Verify`、
Solver/Verifier role enum、固定三 Agent、parallel solvers、debate、voting、
self-consistency、mandatory Python 或数学类型到 topology 的人工映射。

## 6. Direct 结果

Direct 使用 GPU0 上的本地 `qwen3.5-9b-local / supervisor_theta`，30 题均为
单模型、单调用，协议为
`flowsteer_answer_generate_xml_to_skillev_integer_v1`。

- API calls：30
- `finish_reason=stop`：10；其中 9 对、1 错
- `finish_reason=length`：20；全部达到 4096 output-token limit
- Direct parsing failures：20；全部为 `integer_conversion_failed`
- input tokens：18,884
- output tokens：107,243
- aggregate request latency：758,737.15 ms

因此 Direct 的严格结果是 9/30，而不是 9/10。20 条截断输出没有完整
`<answer>`，不能从未完成的 `<thought>` 末尾抽取数字代替正式答案。旧的
bare-integer Direct collection 和只导入基础 prompt、未接入 XmlFormatter 的
collection 均已隔离保存，只作为 protocol diagnosis，不作为最终 baseline。

## 7. AgentGraph 结果

- 正确题：`01, 03, 11, 16`
- trajectory：29/30
- evaluator-valid explicit `FINISH`：25
- `max_rounds` terminal failures：4（`05, 15, 17, 28`）
- collection timeout：1（`13`）
- AgentGraph parsing failures：5（`06, 08, 09, 12, 14`）
- Director attempts：351
- Executor calls：188
- provider attempts：188
- aggregate API attempts（Director + Executor）：539
- input tokens：943,361
- output tokens：266,014
- aggregate request latency：2,868,931.72 ms

### Graph 结构分布

| Distribution | Counts |
|---|---|
| Agent count | `1:12, 2:12, 3:2, 4:1, 5:1, 7:1` |
| Relation count | `0:12, 1:12, 2:2, 3:2, 6:1` |
| Topology | `single:12, serial_2:8, reciprocal:4, fan_in:2, parallel:1, mixed:1, serial_3_plus:1` |

这说明 Director 确实生成并执行了非链式 topology，但初始未训练策略仍以
single 和 two-node graph 为主。报告的节点模型分布覆盖 25 条 evaluator-valid
graph，以及 4 条无 `FINISH` 的 terminal Canvas graph。

### Model routing

| Model ID | Terminal Canvas nodes | Executor calls |
|---|---:|---:|
| `gpt-4o-mini` | 30 | 87 |
| `qwen3.5-9b-local` | 16 | 52 |
| `deepseek-v4-flash` | 6 | 22 |
| `qwen3.5-flash` | 4 | 16 |
| `MiniMax-M3` | 2 | 11 |

Executor provider calls：VectorEngine 136，本地 Qwen3.5-9B 52。AIME 初版
不是“所有 Agent 都使用 Qwen3.5-9B”。

## 8. Wrong Demo 分类

以下分类只描述 receipt 中第一个可观察 failure；“后续传播”表示其后的
execution span，不宣称未经观测验证的因果关系。

| Failure layer | Count | Receipt interpretation |
|---|---:|---|
| Agent | 13 | 没有可观察 runtime failure，Output Agent 提交错误整数 |
| output extraction | 5 | 正式终局输出不是可 canonicalize 的十进制整数 |
| runtime | 5 | 1 个 collection timeout、2 个 graph execution failure、2 个 provider failure |
| graph | 2 | Canvas 拒绝非法 relation 或未知 model ID |
| Director | 1 | 非法 self-loop，后续未能合法终止 |
| dataset / evaluator / tool | 0 | receipt 中未观察到该层 failure |

### 26 个错误任务

| Task | GT | Formal final | First observable failure | Subsequent receipt / terminal result |
|---|---:|---|---|---|
| 02 | 62 | 56 | t4 `SET_RELATION`：3-Agent reciprocal block 超过上限 | 后续 8 turns；`FINISH`，Accuracy=0 |
| 04 | 70 | 69 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 05 | 65 | ∅ | t1 `ADD_AGENT`：`miniMax-M3` 不是 catalog 中的 exact model ID | 后续空内容执行失败；`max_rounds`，未评测 |
| 06 | 441 | `\boxed{441}` | t4 `FINISH`：不能 canonicalize 为纯整数 | 正确 semantic candidate 因 parsing failure 得 0 |
| 07 | 396 | 540 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 08 | 244 | `N = 288` | t2 `FINISH`：非规范终局格式 | parsing failure，Accuracy=0 |
| 09 | 29 | 长文本 | t2 `FINISH`：非规范终局格式 | parsing failure；可见 candidate 也不等于 GT |
| 10 | 156 | 108 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 12 | 161 | 长文本 | t2 `FINISH`：非规范终局格式 | parsing failure；可见 candidate 不等于 GT |
| 13 | 39 | ∅ | AgentGraph collection `TimeoutError` | 无 trajectory，`collection_failed` |
| 14 | 681 | 长文本 | t2 `FINISH`：非规范终局格式 | parsing failure；可见 candidate 不等于 GT |
| 15 | 83 | ∅ | t3 `SET_RELATION`：two-Agent block 返回空内容 | 后续 16 turns；`max_rounds` |
| 17 | 243 | ∅ | t6 `SET_RELATION`：two-Agent block 返回空内容 | 后续 13 turns；`max_rounds` |
| 18 | 503 | 14 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 19 | 279 | 49 | t0 Executor provider HTTP 500 | 恢复后 `FINISH`，最终整数仍错误 |
| 20 | 190 | 240 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 21 | 50 | 400 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 22 | 754 | 106 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 23 | 245 | 125 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 24 | 669 | 499 | t0 Executor provider HTTP 429 | 恢复后 `FINISH`，最终整数仍错误 |
| 25 | 850 | 425 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 26 | 132 | 999 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 27 | 223 | 13 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 28 | 107 | ∅ | t4 Director 生成 `U → U` self-loop | 后续非法 reciprocal relations；`max_rounds` |
| 29 | 157 | 105 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |
| 30 | 393 | 729 | 无结构化 runtime failure；Output Agent 输出错误整数 | `FINISH`，Accuracy=0 |

### 代表案例

1. **任务 02：正确 artifact 已传到下游，但 Output Agent 仲裁错误。**
   `task_analyst` 给出 62，另一条 `combinatorics_solver` artifact 给出 56；
   `final_validator` 收到两份冲突信息后选择了 56。可以确认 communication
   已传递正确 candidate，但不能仅凭 receipt 证明先前 relation rejection
   导致最终仲裁错误。
2. **任务 05：正确历史 candidate 不能代替正式终局。** t0 artifact 已包含
   65，但后续 model ID admission 和空内容执行失败使 Director 达到
   `max_rounds`。按照正式 terminal semantics，历史 65 不进入 evaluator。
3. **任务 06：明确的终局格式错误。** AgentGraph 得到正确 semantic candidate
   441，但提交 `\boxed{441}`；SkillEval integer rule 正确判 parsing failure。
4. **任务 19：provider failure 后恢复，但答案仍错。** t0 HTTP 500；系统恢复
   并合法 `FINISH`，最终输出 49。receipt 只能确认二者同时存在，不能断言
   provider failure 是错误答案的原因。
5. **任务 28：Director action 与 termination failure。** Director 先生成
   self-loop，之后多次生成超出上限的 reciprocal relation，并最终达到
   `max_rounds`。

## 9. 根因判断

当前主要问题按可观察证据排序：

1. **数学推理和冲突仲裁**：13 个错误没有结构化 runtime failure，终局为
   错误整数；任务 02 还显示正确 candidate 已进入 Output Agent inbox，但
   冲突选择失败。
2. **终局输出协议执行**：AgentGraph 5 条格式失败；Direct 20 条因 4096-token
   截断没有产生完整 `<answer>`。
3. **Director termination**：4 条 `max_rounds`；非法 relation、未知 model ID
   或 execution failure 后没有及时恢复到合法 `FINISH`。
4. **runtime 可诊断性**：33 个 turn 标记 `execution_status=failed`，其中 31 个
   没有结构化 `failure_record`；这会限制精确 recovery attribution。
5. **operational failure**：1 条全题超时，以及可恢复的 HTTP 500/429。

没有证据支持 dataset 或 evaluator 算错，也没有启用 Tool，因此本轮不存在
retrieval/tool-selection failure。当前 Accuracy 低于 Direct，不能描述为
FlowSteer/SkillFlow 论文效果的复现；这是统一架构的未训练 Stable Zero 初版。

## 10. 已知问题与后续实验边界

- 全量 30 题 Stable Zero 未通过，不能把 2 题 canary 结论外推到全量。
- 若后续修改 terminal output contract、Director action admission、timeout 或
  runtime failure receipt，必须建立新的 frozen condition 并重新做 paired
  evaluation；不得离线改写本轮结果。
- 不应通过宽松解析、历史 candidate 回收、搜索 AIME 答案、题解数据库或
  ground-truth-aware repair 提高本轮 Accuracy。
- 不应根据这些 30 题手工写入固定 Solver/Verifier topology 或题型解法。
- MACE、Bayesian posterior/EVSI、Skill、GRPO 与 LoRA 仍属于后续单独授权阶段。

## 11. Reproducibility

- Branch：`feature/aime2026-initial-adaptation-20260825`
- Pre-task backup：`backup/pre-aime2026-adaptation-20260825`
- Config：`config/evaluation_aime2026_unified_initial_v1.yaml`
- Selected tasks：`artifacts/aime2026_unified_initial_v1/evaluation/selected_tasks.jsonl`
- Direct receipts：`artifacts/aime2026_unified_initial_v1/evaluation/direct_predictions.jsonl`
- AgentGraph trajectories：`artifacts/aime2026_unified_initial_v1/evaluation/agentgraph_trajectories.jsonl`
- Paired results：`artifacts/aime2026_unified_initial_v1/evaluation/paired_results.jsonl`
- Wrong Demos：`artifacts/aime2026_unified_initial_v1/evaluation/wrong_demos.jsonl`
- Collection failures：`artifacts/aime2026_unified_initial_v1/evaluation/collection_failures.jsonl`
- Machine-readable report：`reports/aime2026_unified_initial_v1/evaluation_report.json`
- Manifest：`artifacts/aime2026_unified_initial_v1/evaluation/run_manifest.json`

Final targeted regression：**94 passed，13 subtests passed，1 upstream
Pydantic deprecation warning**。AIME Canvas correction后的更广回归记录为
**363 passed，60 subtests passed，1 warning**。
