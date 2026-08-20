# TriviaQA v6.2 架构开发报告

## 1. 评测边界

- 数据：`joint_qa_v2` 中固定的 TriviaQA development 128 样本；这是已暴露的架构开发集，不是 held-out test。
- Direct：本地 Qwen3.5-9B，复用同一批 question-only Direct receipt，不重复生成。
- Flow-Director：本地 Qwen3.5-9B；Executor 从冻结的异构 `model_catalog_multidataset_tool_v2.yaml` 选择。
- Evaluator：`triviaqa.official.answer.v1`；对官方 accepted-answer aliases 计算最大 Exact Match 与 token F1。
- 本轮未执行 GRPO、LoRA、backward、optimizer update、Skill 检索或 Skill 注入。

机器可读汇总见 `triviaqa_report.json`；完整 Canvas、Executor、通信和 Tool receipt 见 `artifacts/qa_orchestration_tool_v6_2_development/triviaqa/tool_agentgraph_trajectories.jsonl`。

## 2. 完整 fixed-128 结果

| 条件 | 有效样本 | EM 正确数 | EM | token F1 | 显式 FINISH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 128 | 45 | 35.1563% | 40.8160% | 不适用 |
| AgentGraph v6.2 | 128 | 66 | 51.5625% | 60.1044% | 116/128 |
| AgentGraph - Direct | — | +21 | +16.4063 pp | +19.2884 pp | — |

运行状态为 `completed_with_terminal_failures`：128/128 evaluator receipt 有效，0 collection failure，12 条轨迹以 `max_rounds` 结束。Direct 与 AgentGraph 的上下文和可用 Tool 不同，因此上表差值是 separate-protocol descriptive comparison，不是单变量因果效应。

旧 v3 在同一 128 个 task ID、问题、accepted answers 和 evaluator 上得到 EM 50.7813% / F1 63.3649%，125/128 FINISH。v6.2 相对 v3 为 EM `+0.7813` pp、F1 `-3.2605` pp、FINISH `-9`；两者的 Director condition、sampling purpose、派生 seed、Tool domain 和 handoff prompt 不同，所以该差值同样只能做描述性比较。v6.2 没有形成整体优于 v3 的证据。

## 3. 架构与上游来源

执行链完整：

`Question -> Qwen3.5-9B Flow-Director -> progressive Canvas -> AgentGraph Runtime -> Executor communication/Tool -> Format Agent -> FINISH -> TriviaQA evaluator -> trajectory`

- Progressive Canvas：沿用 FlowSteer 的一次 Canvas edit 后立即执行、返回 feedback、再决定下一次 edit；`add_subgraph` 是一个含 1–3 个 Agent 的原子功能子图，不等同于“每次只能添加一个 Agent”。
- Format：直接导入 FlowSteer `scripts/prompts/prompt.py::FORMAT_PROMPT`，使用 `problem + computed solution` 的 extraction-only 调用边界。
- Tool/ReAct：沿用 SkillFlow `QARetrievalEnvironment` 的单一 `qa-retrieval` resource、`search/read` StructuredAction、public Action/Observation continuation 和显式 `complete`。
- AgentGraph 必要适配：异构 Executor、typed communication envelope、canonical `passage_id` 的 search-to-read admission、直接 Format predecessor 的 semantic-answer handoff，以及 current-revision FINISH admission。
- Director 提示词保持简洁、中性，不包含固定 workflow、强制 Agent 数、强制 topology 或未验证 Skill。

## 4. AgentGraph、模型和 Tool

- 最终 topology：`empty=4`、`single=3`、`serial_2=76`、`serial_3_plus=45`；没有自然提交的 fan-in 或 reciprocal topology。
- 最终 Agent 数：0 个 4 题、1 个 3 题、2 个 76 题、3 个 44 题、4 个 1 题。
- Executor 共 367 次调用：GPT-4o-mini 135、Qwen3.5-Flash 75、本地 Qwen3.5-9B 69、DeepSeek-V4-Flash 59、MiniMax-M3 13、MiniMax-M2.5 9、GLM-4.5-Flash 7。实际执行不是单一 Qwen3.5-9B 模型池。
- 实际 Tool dispatch receipt 共 186 条：137 次 search、49 次 read；186/186 backend completion 成功，0 backend error。
- Director 有 188 次 action parse failure、391 次 rejected turn，rejected-turn rate 46.16%。这是 terminal failure 和搜索效率的主要工程信号之一。

上述 topology 与分数关系没有随机 paired intervention；不能据描述性均值强制非链式结构，也不能把结构深度写入 reward。

## 5. 通信和错误归因

- 128/128 轨迹保存完整 Director turn receipt，0 reconstructed context。
- 所有 Executor 通信记录合计 192 条 upstream message，artifact body 均非空。
- 124/128 轨迹保存 Output inbox；116 条显式 FINISH 轨迹全部保存 Output inbox、恰有一个非空 semantic predecessor artifact。

因此没有发现系统性的 message routing、artifact transport 或 Output Agent 可见性丢失。主要失败来自：

1. retrieval query breadth 和 evidence relevance validation 不足；
2. ReAct Agent 在 6-turn budget 内未完成，随后 Director repair/delete/rebuild；
3. Director malformed JSON/action repair 导致 `max_rounds`；
4. semantic predecessor 的 relation、answer type、alias 或 answer granularity 错误；
5. 少量旧 artifact 在后续成为 Format predecessor 时未采用 `Candidate answer / Evidence` handoff。

## 6. 真实 Demo

### 正确：ISDN / Japan

Task：`triviaqa:tc_15`。问题询问 1988 年 ISDN 广泛使用始于哪个国家。

链路：

`Solver (DeepSeek-V4-Flash, ReAct, qa-retrieval) -> Extractor (GPT-4o-mini, Format) -> FINISH`

Solver 执行 `search(limit=10)`，从第 4 个 passage 定位到日本 NTT 在 1988 年开始提供 nationwide ISDN services，再对 canonical passage ID 执行 `read`；handoff 为 `Candidate answer: Japan` 加证据。Format 输出 `<answer>Japan</answer>`。EM=1，F1=1。

### 错误：Prince Henry / windshield wiper

Task：`triviaqa:tc_18`。

链路：

`Researcher (GPT-4o-mini, ReAct, qa-retrieval) -> Answer Agent (GPT-4o-mini, Format) -> FINISH`

Researcher 使用 `search(limit=1)` 后读取 rank-1 的 `Vauxhall Prince Henry` passage；该 passage 只匹配人名和年份，没有支持问题要求的 innovation relation。Researcher 过早完成为 `Candidate answer: The Vauxhall Prince Henry...`，Format 忠实输出 `<answer>The Vauxhall Prince Henry</answer>`；Ground Truth 为 `windshield wiper`，EM=0，F1=0。错误起点是 retrieval relevance 和 semantic relation，不是通信。

### 错误：Director JSON / terminal failure

Task：`triviaqa:tc_31`。问题询问 Algeria 的国际车辆注册字母，Ground Truth 为 `DZ`。Director 在 20 个 round 中有 19 次 malformed JSON/action parse failure，未形成有效 Output graph，最终 `max_rounds`、空答案，EM=0，F1=0。

### 正确事实被错误序列化

Task：`triviaqa:tc_94`。Researcher 已得到 “Eighteenth Amendment ... prohibition in 1920”，但其 artifact 是整句而不是 semantic-answer handoff；Format 输出整个句子，官方 accepted answers 为 `18th | 18 | eighteen`，因此 EM=0，F1=0。这不是答案或 evidence 丢失，而是 progressive artifact reuse 与 answer-span serialization 的边界问题。

## 7. 被拒绝的 v6.3 修正

开发轨迹显示，一些 Agent 在 Format Agent/edge 建立前已经执行；后续新增 Format edge 时，FlowSteer dirty-closure 复用了旧 artifact，而本项目新增的 `is_format_predecessor` 会改变模型输入。针对这一必要适配候选，曾在 dirty set 中加入“Format-predecessor execution role 发生变化的 Agent 及其 descendants”，并运行 13 个精确受影响样本的固定 panel。

| 条件 | EM | F1 | FINISH |
| --- | ---: | ---: | ---: |
| v6.2 panel | 69.2308% | 71.4286% | 13/13 |
| v6.3 candidate | 69.2308% | 73.6264% | 12/13 |
| 变化 | 0.0000 pp | +2.1978 pp | -1 |

候选修正了 `tc_94`（整句 -> `18th`）和 `tc_109`（Michael Nesmith -> Peter Tork），但使 `tc_36` 从 Pisces 回归为 Gemini，并使原本正确的 `tc_99` 变为 terminal failure。由于新增 terminal failure，候选未通过无回归条件，源码已回退；配置、trajectory 和报告保留在 `triviaqa_format_predecessor_v6_3_panel` 目录作为 rejection evidence。有效版本仍是 v6.2。

## 8. 当前结论

TriviaQA v6.2 通过 2-task Stable Zero canary，但 full development-128 只有 116/128 显式 FINISH，因此不能称为完整 terminal-stable。当前完成的是 inference、progressive orchestration、heterogeneous Executor、Tool/ReAct、Format、official evaluator 和 trajectory 接线；MACE、Bayesian posterior、ACTIVE Skill 和 LoRA/GRPO training 均未在本轮执行。

下一阶段优先信号应是 Director action-schema compliance、ReAct completion/terminal control、retrieval evidence relevance 和 answer granularity。现有证据不支持强制非链式 topology，也不支持把本轮被拒绝的 v6.3 修正直接合并。
