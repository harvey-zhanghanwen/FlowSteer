# HealthBench Professional 统一 AgentGraph 第一轮正式评测报告

## 结论

本轮已完成 HealthBench Professional 的 task adapter、统一 AgentGraph
执行闭环、reference-compatible evaluator 接线、两条 Stable Zero canary
和完整 525 条 public test 请求。没有训练，没有启用 Tool、GRPO、LoRA、
MACE、Bayesian posterior 或 Skill evolution，也没有加入固定医疗角色或固定
医疗 workflow。

Direct 为 525/525 evaluator-valid；AgentGraph 为 503/525 evaluator-valid，
其余 22 条均在 20 个 Director turn 内未产生合法 `FINISH`，属于
`max_rounds` terminal failure。正式状态为
`completed_with_terminal_failures`，当前 operational/evaluator failure 为 0。
两条 smoke canary 的 Stable Zero 已确认，但 525 条正式集的 all-task Stable
Zero criterion 未通过，原因是 22 个 workflow 没有完成终局提交。

## 数据、条件与 evaluator

- 数据：`openai/healthbench-professional` 唯一公开 `test`，按源顺序固定
  525 条；没有从 test 构造 train 或 development。
- 模型可见输入：完整 `conversation.messages`；gateway 恢复原生 role。
- evaluator-only：`rubric_items`、`physician_response`、canary 和 slice
  metadata，只在 private task-ID join 后进入 grader。
- Direct 与 AgentGraph：相同本地 Qwen3.5-9B、相同 generation setting、
  相同空 Tool condition、相同 grader condition。
- evaluator：OpenAI `simple-evals` revision `652c89d` 的 HealthBench
  Professional reference path；grader 为 `gpt-5.4-2026-03-05`、low
  reasoning effort。
- 主指标：signed rubric score 的 Professional character-length adjustment，
  然后对 dataset mean 做 `[0,1]` clipping；不是 EM、F1 或 Accuracy。
- 限定：OpenAI 内部 production evaluator 未公开，因此本结果称为
  **HealthBench Professional reference-compatible score**；每题一次采样，
  不等价于论文 repeated-sampling protocol。

## 正式分数

| Condition | 请求数 | Evaluator valid | Strict raw | Strict length-adjusted | Valid-only length-adjusted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 525 | 525 | 18.9721% | **19.1728%** | 19.1728% |
| AgentGraph | 525 | 503 | 22.6451% | **20.2395%** | 21.1247% |

- AgentGraph − Direct strict raw：**+3.6730 percentage points**。
- AgentGraph − Direct strict length-adjusted：**+1.0667 percentage points**。
- 21.1247% 是 503 条有效样本均值，不能当作 525 条 strict score。
- 22 条 terminal failure 没有被伪造成 valid grade；strict score 使用完整
  请求分母 525，valid-only score 单独给出。
- Paired outcome：AgentGraph 更高 197，Direct 更高 302，相等 4，
  terminal failure 22。

## 执行闭环与 topology

- Graded explicit `FINISH`：503。
- `max_rounds`：22。
- reportable terminal failure：22。
- terminal parsing failure：0。
- current evaluator operational failure：0。
- Agent runtime failed turn：0；Executor error：0。
- 525 条 trajectory 中记录 4 次 Director action parse failure。

全部 525 个 AgentGraph terminal receipt 的自然分布如下：

| 统计 | 分布 |
| --- | --- |
| Agent 数量 | 1: 347；2: 103；3: 40；4: 27；5: 5；6: 2；8: 1 |
| Relation 数量 | 0: 347；1: 106；2: 33；3: 23；4: 7；5: 8；6: 1 |
| Topology | single: 347；serial-2: 80；serial-3-plus: 17；reciprocal: 39；fan-in: 18；fan-out: 4；parallel: 4；mixed: 16 |
| Structural depth | 1: 386；2: 101；3: 31；4: 7 |
| Effective dependency depth | 1: 387；2: 123；3: 14；4: 1 |

其中 178/525 是 multi-Agent，包含 serial、reciprocal、fan-in、fan-out、
parallel 和 mixed
topology。所有 contract/Agent ID/topology 均由 Director 在开放 search space
中产生；代码没有 Doctor→Researcher→Reviewer 或其他固定医疗 workflow。
`effective_dependency_status=weak` 只证明发生了真实消息传输，不证明下游
模型在语义上充分利用了该消息。

22 个 `max_rounds` workflow 全部是 3–8 Agent 的 multi-Agent graph，分布为
mixed 9、fan-out 4、parallel 4、serial-3-plus 4、serial-2 1；它们共消耗
1,195 次 generation API attempt。终止长尾集中在复杂 graph，而不是
single-node graph。

## Wrong Demo 与首个可观察 failure layer

`wrong_demos.jsonl` 收录 496 个低于逐题满分或 terminal failure 的 case；
它不是“AgentGraph 全部回退”的集合。首个可观察 failure layer 分布为：

下表保留原始首个可观察 layer 视图。为避免 `max_rounds` 与早期
graph/director anomaly 重复解释，terminal-first 的互斥专业分类、每个非零
子类的确定性代表样本以及 0/N/A 类别见
`failure_taxonomy_report_zh.md`；完整链路只保存在 evaluator-private artifact。

| Failure layer | 数量 | AgentGraph 更差 | 解释 |
| --- | ---: | ---: | --- |
| `rubric_evaluation` | 358 | 230 | 最终回答未充分满足 positive rubric，或触发 negative rubric；主要属于模型回答质量 |
| `terminal_response_length_adjustment` | 81 | 44 | raw rubric 已满但回答冗长，Professional 长度校正降低分数 |
| `graph` | 55 | 20 个有效配对回退，另有 21 个 terminal failure | Canvas action rejection、relation/communication 编排问题；21 条最终未 `FINISH` |
| `director` | 2 | 1 个有效配对回退，另有 1 个 terminal failure | Director action 无法解析；其中 1 条最终未 `FINISH` |

以下示例只保留 task ID、分数和结构化 receipt，不公开 conversation、rubric
文本、physician response、grader explanation 或完整回答。逐题 score 未做
dataset-level clipping，因此可以大于 1 或小于 0。

1. **回答质量 / rubric failure**
   - task：`healthbench-professional:3b22335d21170ba8dfc661926255c449`
   - Direct / AgentGraph length-adjusted：`1.017052 / -2.308088`
   - 首个可观察层：`rubric_evaluation`
   - receipt：3 个 rubric，触发 2 个 negative rubric，未满足 1 个 positive
     rubric；Output Agent 为 Director 自主生成的 `diet_expert`。

2. **Graph relation / communication failure**
   - task：`healthbench-professional:7e885f8fa08be5f5540c0f20a5791d38`
   - Direct / AgentGraph length-adjusted：`1.031282 / -1.072177`
   - 首个可观察层：`graph`，第 4 个 Director turn 的 `SET_RELATION`
     后出现 Canvas rejection；2 个 rubric 中触发 1 个 negative，且 1 个
     positive 未满足。

3. **Director action parse failure**
   - task：`healthbench-professional:b8fcd39ceb6161426de40c6508d742f1`
   - Direct / AgentGraph length-adjusted：`-0.026607 / -0.963857`
   - 首个可观察层：`director`，第 4 个 turn action 无法解析；后续仍合法
     `FINISH`，但 2 个 rubric 中触发 1 个 negative 且遗漏 1 个 positive。

4. **终局长度校正**
   - task：`healthbench-professional:6daac09cf85897e495539f3255bcf14c`
   - Direct / AgentGraph length-adjusted：`1.014406 / 0.876608`
   - 2 个 rubric 全部满足、未触发 negative；AgentGraph 回答更长，损失来自
     Professional length adjustment，而不是 termination failure。

5. **Terminal failure**
   - task：`healthbench-professional:3e700f616cccc0c3cbeea24244544f27`
   - Direct length-adjusted：`1.037397`；AgentGraph grade 不可用。
   - 第 8 个 Director turn 首次出现无法解析的 action；其后继续执行，但在
     第 20 个 turn 到达 `max_rounds`，`explicit_finish=false`、终局答案为空。
   - 最新 receipt：`not_evaluated_without_explicit_finish`，
     `formal_evaluator_called=false`；没有把空提交送入 rubric grader。

### Root-cause 排序

1. 最大误差来源是最终回答的 rubric coverage 与 negative-rubric avoidance，
   更接近基础模型回答质量而不是统一 runtime bug。
2. 81 个 length-adjustment case 表明部分 AgentGraph 输出过长；这是终局响应
   控制问题，但本轮不能通过 task-specific Formatter 或强制压缩改写回答。
3. 34 个已评分 graph-layer case 中有 20 个回退，另有 21 个 graph-layer
   terminal failure；2 个 director-layer case 中有 1 个已评分回退和 1 个
   terminal failure。这说明开放 search space 中仍有 action admission、
   relation construction 和及时 `FINISH` 的问题；可以在独立 development
   condition 上研究，但不能依据 public test Wrong Demo 写死医疗 topology。
   22 个 terminal failure 的首个被拒动作均为 `SET_RELATION`：10 次没有改变
   graph、7 次 bidirectional block 超过限制、3 次形成 quotient cycle、1 次
   source/target 相同、1 次引用未知 `agent_id`。这是 Canvas feedback 中直接
   可观察的 failure evidence，不等同于已证明的单一因果根因。
4. 22 个未评分项是 workflow terminal failure，不是当前 evaluator/provider
   failure。append-only evidence 仍保留修正前空回答误入 grader 的历史错误和
   retry receipt；它们不能当作当前 operational failure 数量。

## Token、调用与延迟 receipt

| Condition | 模型 API attempts | Input tokens | Output tokens | 累计调用 latency |
| --- | ---: | ---: | ---: | ---: |
| Direct | 525 | 429,781 | 271,133 | 1,987,969 ms |
| AgentGraph | 6,819 | 13,368,650 | 2,259,412 | 25,540,806 ms |

AgentGraph 的 6,819 次包含 Director 4,264 次与 Executor 2,555 次。两者均
使用本地 `qwen3.5-9b-local` / served model `supervisor_theta`。

Reference grader telemetry：

- Direct：1,296 API calls，161 次已记录 provider error，
  `invalid_grade_count=0`，2,122,417 total tokens；
- AgentGraph：1,250 API calls，169 次已记录 provider error，
  `invalid_grade_count=22`，2,263,966 total tokens。

provider error 数是 append-only 的物理调用级 receipt，其中包含后续恢复的
调用；AgentGraph 的 `invalid_grade_count=22` 是修正前空提交误入 grader 的
历史本地校验记录。它们都不能等同于当前 operational failure；当前值为 0。

## 源码复用与必要适配

- **直接复用**：统一 `AgentGraph`、FlowSteer progressive Canvas 的
  edit→execute→feedback 边界、relation communication、唯一 Output Agent、
  explicit `FINISH`、runtime、trajectory、model interface。
- **SkillFlow/SkillEval 语义复用**：Supervisor/Executor 的 bounded execution
  与 private evaluator separation；本轮没有启用 Skill 或训练 session。
- **HealthBench 必要适配**：525-row schema→`TaskRecord`、完整 conversation
  role round-trip、task-ID keyed private evaluator join、pinned `simple-evals`
  rubric grader、Direct/AgentGraph paired report 与 evaluator-only retry。
- **未实现/未启用**：固定医疗角色或 topology、医学检索 Tool、GRPO、LoRA、
  MACE、Bayesian update、Skill retrieval/injection/evolution。

具体上游文件、类、函数和不兼容原因记录在 `docs/source_map.md`；实施与
评测边界记录在 `docs/adaptation_log.md`。

## 已知问题与下一步边界

- 22 条 `max_rounds` terminal failure 阻止 525 条正式集通过 all-task Stable
  Zero；本报告同时给出 strict 与 valid-only 指标，未填造缺失 grade。
- 22 条 terminal failure 全部来自至少 3 Agent 的复杂 graph，说明当前开放
  search space 的主要终局风险是 relation editing 持续占用 turn budget，未能
  及时提交 `FINISH`；这是一项 development-condition 假设，不据 public test
  直接写入固定 topology 或 task-specific prompt。
- 66.1%（347/525）的 recorded graph 为 single-node。开放 search space 已能
  产生多种非链式 topology，但未训练 Director 时仍明显偏向最小图；本轮不得
  据 test failure 写死医疗编排。
- AgentGraph strict 主指标只提升 1.0667 个百分点，同时推理调用显著增加；
  需要未来在独立 development condition 上验证 topology 与分数的因果关系。
- 后续若授权训练、Skill 或 Tool，需要建立新的 condition/version，不能覆盖
  本轮无 Tool、无训练、memory-off 的 Stable Zero baseline。
