# HotpotQA v6.2 架构开发报告

## 1. 评测边界

- 数据：`joint_qa_v2` 中固定的 HotpotQA development 128 样本；这是已暴露的架构开发集，不是 held-out test。
- Direct：本地 Qwen3.5-9B，复用同一批 v4 `direct_closed_context_predictions.jsonl`，不重复生成。
- Flow-Director：本地 Qwen3.5-9B；Executor 由冻结的异构 `model_catalog_multidataset_tool_v2.yaml` 选择。
- Evaluator：`hotpotqa.official.answer.v1`，按 HotpotQA 官方 normalization 计算 Exact Match 与 token F1。
- 本轮未执行 GRPO、LoRA、backward、optimizer update、Skill 检索或 Skill 注入。

完整机器可读结果见 `hotpotqa_report.json`；完整 Canvas、Agent 调用、通信与 Tool receipt 见 `artifacts/qa_orchestration_tool_v6_2_development/hotpotqa/tool_agentgraph_trajectories.jsonl`。

## 2. 结果

| 条件 | 有效样本 | 正确数 | EM | token F1 | 显式 FINISH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 128 | 90 | 70.3125% | 78.6792% | 不适用 |
| AgentGraph v6.2 | 128 | 91 | 71.0938% | 81.4453% | 126/128 |
| AgentGraph - Direct | — | +1 | +0.7813 pp | +2.7662 pp | — |

运行状态为 `completed_with_terminal_failures`：128/128 evaluator receipt 有效，0 collection failure，2 条轨迹因 `max_rounds` 截断。

同一 sampling schedule 下，v6.2 相对 v6.1 从 EM 67.1875% / F1 77.2507% 提升到 EM 71.0938% / F1 81.4453%。v5 的描述性结果为 EM 73.4375% / F1 83.4037%，但 v5 使用不同 sampling purpose 与派生 Director seed，不能作为严格 paired causal comparison。

## 3. 架构实现与来源

当前链路完整执行：

`Question -> Qwen3.5-9B Flow-Director -> progressive Canvas -> AgentGraph Runtime -> Executor communication/Tool -> Format Agent -> FINISH -> evaluator -> trajectory`

- Progressive Canvas：保留 FlowSteer 的一次 `add_subgraph|modify_agent|delete_agent|set_relation|set_output` 后立即执行并返回 feedback，再进行下一次编辑；不是先构建完整图再统一执行。
- Format：直接复用 FlowSteer `FORMAT_PROMPT` 的 `problem + computed solution` 调用边界；AgentGraph 仅增加直接 predecessor 标记和 `<answer>...</answer>` terminal protocol。
- Tool/ReAct：复用 SkillFlow 的单一 `qa-retrieval` resource、`search/read` action domain、public Action/Observation continuation 与显式 `complete`；`read` 只接受同次 execution 中 `search` 返回的 canonical `passage_id`。
- FINISH：只向 Director 暴露当前 graph revision 的正向 admissibility；仍必须由 Director 显式提交 `FINISH`。
- Communication：typed envelope、directed edge、graph revision、artifact body 与 Tool receipt 全部写入 trajectory。

定向测试结果：162 passed，1 warning，43 subtests passed。

## 4. AgentGraph、模型与 Tool

- 最终图：`single=1`、`serial_2=94`、`serial_3_plus=33`；没有自然提交的 fan-in 或 reciprocal topology。
- 三 Agent 图的描述性 EM/F1 为 75.76%/87.88%，二 Agent 图为 70.21%/80.05%；任务难度与模型选择未控制，不能据此宣称非链式 topology 具有因果增益，也不应强制 topology。
- Executor 确实使用异构模型池，包括本地 Qwen3.5-9B、Qwen3.5-Flash、GPT-4o-mini、DeepSeek-V4-Flash、GLM 与 MiniMax，不是所有 Agent 都使用 Qwen3.5-9B。
- 83 条最终 Canvas 含 Tool-capable Agent，78 条最终使用 ReAct；共 21 个实际 Tool dispatch receipt（18 search、3 read），全部成功，无 backend error。
- 仍有 11 次 `qa_read_requires_successful_search` 与 5 次 StructuredAction parse error。根因是 Executor 未遵守 `search -> canonical passage_id -> read`，不是 retrieval backend 故障。

## 5. 通信与错误归因

126 条显式完成轨迹中，113 条 terminal predecessor 使用 `Candidate answer` handoff，111/113 的 Format 输出在 HotpotQA normalization 后与 candidate 一致。128/128 保存完整 turn receipt，127/128 保存 Output inbox；唯一缺失 inbox 的任务没有建立 Output Agent。因此没有发现系统性的 artifact 丢失、错误路由或下游不可见。

37 条 EM 错误的人工 operational taxonomy：

- 20 条：answer span 过长、过短或 canonical form 不匹配，token F1 大于 0；
- 5 条：surface/alias mismatch；
- 10 条：semantic relation 或 answer type 错误；
- 2 条：terminal control failure。

主要错误起点是 semantic-answer predecessor 的 relation selection、answer type 与 lexical span，而不是 Format 或 Agent 间通信。

## 6. 真实 Demo

### 正确：Nassau County

Task：`hotpotqa:5ae732685542991e8301cbc3`。问题询问 Guwe Secondary School 的 sister school 位于纽约哪个 county。

Canvas 过程：

1. 拒绝 `reasoning Agent + allowed_tools`；
2. 拒绝非 Format Output Agent；
3. 接受并执行 `Searcher (ReAct, Qwen3.5-Flash) -> Answerer (Qwen3.5-Flash) -> Format (Qwen3.5-Flash)`；
4. Searcher 提供证据，Answerer 生成 `Candidate answer: Nassau County`，Format 输出 `<answer>Nassau County</answer>`；
5. Director 提交 `FINISH`。

Ground Truth / Final：`Nassau County`；EM=1，F1=1。

### 错误：Widsith / Exeter Book

Task：`hotpotqa:5a84918e5542990548d0b2cf`。

链路：`Solver (ReAct, GPT-4o-mini) -> Format (GPT-4o-mini)`。问题询问包含 poem 的 book；Tool search 返回相关 passage，但 Solver 把 poem `Widsith` 当作 container book，Format 完整接收并输出 `<answer>Widsith</answer>`。Ground Truth 为 `Exeter Book`，EM=0，F1=0。错误属于 semantic relation 与 answer type selection，不是通信故障。

### 错误：答案出现但未显式终止

Task：`hotpotqa:5a7e36045542991319bc9440`。前 6 轮包含 4 次 action parse/schema failure、execution-contract rejection 和 Output Agent rejection。Round 19 的 Format execution 已产生 `<answer>Jonathan Stark</answer>`，但 20-round episode horizon 没有留下下一轮显式 `FINISH`，正式结果为 `None`。

## 7. 被拒绝的开发干预

- 通用 `ANSWER_GENERATION_PROMPT`：在与 full128 前 16 条具有相同 task_id、sequence position 与 Director seed 的 paired panel 中，v6.2 为 EM 56.25% / F1 71.25%，干预后降至 EM 37.50% / F1 60.625%。该代码已回退。
- 22-round horizon：六条 paired panel 的显式 FINISH 从 5/6 增至 6/6，但目标任务输出 `Neither (tie)`，未恢复正确答案；panel EM 从 66.67% 降至 50%。该配置未被接受为 accuracy improvement。
- 非链式 topology：当前没有 paired evidence 支持强制 fan-in 或 reciprocal communication，未加入 topology quota、structural reward 或固定 workflow template。

## 8. 当前结论

v6.2 达到 2-task Stable Zero canary，但 full development-128 仍有两条 `max_rounds`，因此不能把完整 fixed128 描述为 100% terminal-complete。当前架构已完成推理、progressive orchestration、异构 Executor、Tool/ReAct、Format、evaluator 与 trajectory 接线；MACE、Bayesian posterior、ACTIVE Skill 与 LoRA/GRPO 训练不属于本轮执行结果。后续最优先的训练信号应来自 semantic relation/answer-type selection 与 action-schema compliance，而不是把通信层或 topology complexity 当作无证据的奖励。
