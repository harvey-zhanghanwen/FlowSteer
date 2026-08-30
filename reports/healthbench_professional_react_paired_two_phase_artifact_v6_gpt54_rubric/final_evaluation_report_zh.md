# HealthBench Professional V6 正式评测报告

## 结论

统一 AgentGraph 单侧执行闭环达到 Stable Zero：525/525 个任务均产生有效
AgentGraph response、有效 rubric grading 和显式 `FINISH`，Graph terminal
failure、`max_rounds` 与终局解析失败均为 0。整项 paired Stable Zero 没有
通过，因为 Direct 有 1 个 ReAct terminal failure（524/525）。本轮架构也没有获得质量增益：
AgentGraph 的官方/reference 主指标为 **17.72%**，低于 Single-Agent ReAct
对照的 **23.81%**，下降 **6.09 个百分点**。

本轮没有训练、GRPO、LoRA、backward、optimizer update、MACE、Bayesian
update、Skill injection 或 Skill evolution。

## 固定协议

- 数据：HealthBench Professional `public_test`，固定 525 个任务。
- 主指标：OpenAI simple-evals HealthBench Professional reference rubric 的
  `overall_score_length_adjusted`。
- 辅助指标：未做字符长度调整的 `overall_score`。
- Executor / Direct 模型：本地 Qwen3.5-9B，thinking 开启。
- Director：本地 Qwen3.5-9B，两阶段 `REASONING -> ACTION`；只有 ACTION
  进入 Canvas。
- Agent execution profile：`execution_mode=react`，Tool 为
  `healthbench-authoritative.search`；ReAct 不是 Agent role。
- Tool：冻结 SkillFlow MedRAG BM25 corpus + NCBI PubMed E-utilities；两侧
  使用同一 Tool condition。
- Grader：现有 `gpt-5.4` alias，复用 pinned OpenAI simple-evals
  HealthBench Professional reference implementation。
- AgentGraph 仍保持自由 Agent 数量、自由 free-text contract、自由关系、
  唯一 Output Agent 与增量 Canvas editing；没有固定医疗 role 或 workflow。

## 正式结果

| 条件 | 固定分母 | 完成 / grader valid | Raw score | Length-adjusted score | Terminal failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-Agent ReAct + authoritative search | 525 | 524 / 524 | 22.14% | **23.81%** | 1 |
| Free AgentGraph + authoritative search | 525 | 525 / 525 | 17.59% | **17.72%** | 0 |

- AgentGraph − Direct：raw `-4.54pp`；length-adjusted `-6.09pp`。
- 逐题：AgentGraph 较高 219 题，Direct 较高 305 题；另 1 题 Direct 是
  冻结 strict-zero ReAct terminal failure。
- Direct 的 1 条失败没有伪造 response：manifest、paired strict-zero 与
  append-only ReAct exhaustion receipt 三者一致，严格按 0 分进入 525 分母。
- paired generation identity 仅验证 524/525，因此整体 `verified=false`。
- 两侧任务、Executor 模型、Tool 与 evaluator 相同，但 AgentGraph 额外使用
  Director 且可增加 Executor call，所以正式报告将差值标为
  `separate_protocol_descriptive_comparison`，不是 compute-matched causal
  estimate。

## AgentGraph 搜索行为

| Topology | 数量 | 占比 |
| --- | ---: | ---: |
| `single` | 522 | 99.43% |
| `serial_2` | 3 | 0.57% |
| 三 Agent 或非链式 topology | 0 | 0.00% |

- 507 个任务的 action sequence 是
  `ADD_AGENT -> SET_OUTPUT -> FINISH`。
- 15 个任务包含一次 `MODIFY_AGENT`；3 个任务形成二节点单向关系。
- FINISH 在 singleton output 后不是唯一合法 action，Director 仍选择在
  522/525 个任务终止。因此这是未训练 Director 的 observed policy collapse，
  不是 action mask 强制单 Agent。
- 三个 `serial_2` 任务的 length-adjusted score 均低于各自 Direct：
  `-0.0662 vs 0.0328`、`0.0539 vs 0.6446`、`0.9404 vs 1.0208`。
- 三条二节点轨迹都在 `SET_OUTPUT` 之后才 `SET_RELATION`，dependency
  evidence 为 `weak`。其中两条把检索职责节点设为 Output，而临床综合职责在
  另一个节点，形成明确的 contract—relation—Output Agent 不一致。

## Wrong Demo 分类

AgentGraph length-adjusted score 小于 1.0 的任务共有 478 个；以下类别互斥：

| 首个可观察 failure layer | 数量 | Wrong Demo 占比 | 全体占比 |
| --- | ---: | ---: | ---: |
| Rubric / response quality | 350 | 73.22% | 66.67% |
| Terminal response length adjustment | 40 | 8.37% | 7.62% |
| 已恢复的 Director action parsing / recovery anomaly | 84 | 17.57% | 16.00% |
| 已恢复的 Canvas / relation edit anomaly | 4 | 0.84% | 0.76% |
| Tool execution / Agent runtime / output extraction / evaluator / max_rounds | 0 | 0.00% | 0.00% |

Rubric / response-quality 子类为：279 个只漏掉 positive rubric，21 个只触发
negative rubric，50 个同时漏 positive 并触发 negative。Receipt 没有提供证据
把这些终局语义缺失唯一归因到某个隐藏 reasoning、communication 或不存在的
固定 Verifier Agent，因此报告不伪造更细的因果结论。

## 资源开销

| 条件 | API attempts | Input tokens | Output tokens | 累计 latency |
| --- | ---: | ---: | ---: | ---: |
| Direct | 1,364 | 2,981,621 | 1,526,975 | 12,934,245 ms |
| AgentGraph | 6,781 | 10,888,011 | 2,066,907 | 35,865,013 ms |

AgentGraph 使用约 3.65 倍 input tokens 和 4.97 倍 API attempts，却没有产生
rubric score 增益；这是当前最重要的架构效率问题。

## 当前判断

1. 执行闭环已经稳定，问题不再是 terminal、Tool provider 或 evaluator
   operational failure。
2. 当前未训练 Director 没有学会按任务复杂度选择多 Agent topology；仅放开
   search space 不会自然产生有效协作。
3. 当前冻结 model catalog 只有 `qwen3.5-9b-local`，因此本轮没有发生
   heterogeneous model routing；这是协议选择，不是多模型实验结果。
4. 主要可观察短板是完整 response 对 signed rubric 的覆盖与负面陈述控制，
   其次是字符长度调整；不能仅靠增加 Agent 数量解决。
5. 继续改进时应先在 development split 做 topology-selection、artifact
   sufficiency 与 response coverage 的架构实验；不能根据 public test rubric
   手工加入固定医疗 workflow。

完整公开指标见 `evaluation_report.md/json`；脱敏分类与代表 task ID 见
`failure_taxonomy_report_zh.md`。完整 conversation、rubric、candidate output、
逐轮 Director/Canvas/Agent/Tool receipt 仅保存在 evaluator-private artifact，
不会提交 Git 或进入模型输入。
