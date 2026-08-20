# HotpotQA + TriviaQA v6.2 联合架构开发报告

## 1. 最终有效结果

以下均为固定 development-128，不是 held-out test；每个数据集 128/128 evaluator receipt 有效。

| 数据集 | 条件 | EM 正确数 | EM | token F1 | 显式 FINISH |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | Qwen3.5-9B Direct | 90/128 | 70.3125% | 78.6792% | 不适用 |
| HotpotQA | AgentGraph v6.2 | 91/128 | 71.0938% | 81.4453% | 126/128 |
| TriviaQA | Qwen3.5-9B Direct | 45/128 | 35.1563% | 40.8160% | 不适用 |
| TriviaQA | AgentGraph v6.2 | 66/128 | 51.5625% | 60.1044% | 116/128 |

相对各自 Direct 的描述性差值：HotpotQA 为 EM `+0.7813` pp、F1 `+2.7662` pp；TriviaQA 为 EM `+16.4063` pp、F1 `+19.2884` pp。Direct 与 AgentGraph 的上下文和 Tool 条件不同，因此这些不是单变量 paired causal effects。

## 2. 当前有效架构

`Question -> local Qwen3.5-9B Flow-Director -> FlowSteer progressive Canvas -> heterogeneous AgentGraph -> typed communication / SkillFlow Tool-ReAct -> FlowSteer Format -> explicit FINISH -> official answer evaluator -> trajectory`

- Flow-Director 始终是本地 Qwen3.5-9B；没有使用 API 模型替代 Director。
- Executor 实际覆盖本地 Qwen3.5-9B、Qwen3.5-Flash、GPT-4o-mini、DeepSeek-V4-Flash、GLM 和 MiniMax。
- Canvas 按一个原子 edit unit 执行；`add_subgraph` 可包含 1–3 个 Agent，整个 subgraph 接受后执行一次并返回 feedback。
- Format 是独立 terminal sink，直接复用 FlowSteer `FORMAT_PROMPT`。
- QA Tool 是 SkillFlow 单一 `qa-retrieval` resource 下的 `search/read` StructuredAction；不存在 benchmark answer 访问。
- FINISH 必须由 Director 显式提交；evaluator 只在终局后运行。

## 3. 架构诊断

HotpotQA v6.2 的主要剩余问题是 semantic relation、answer type 和 answer-span canonicalization；通信 receipt 未显示系统性 artifact 丢失。TriviaQA v6.2 的主要问题是 Director action parse failure、ReAct exhaustion、retrieval relevance、answer granularity 和 terminal control；同样没有系统性 message routing 或 artifact transport loss。

HotpotQA 最终 topology 为 `single=1, serial_2=94, serial_3_plus=33`；TriviaQA 为 `empty=4, single=3, serial_2=76, serial_3_plus=45`。当前没有自然 fan-in/reciprocal 样本足以形成 paired causal evidence，因此有效源码没有加入 topology quota、结构奖励或固定非链式 workflow。

## 4. 迭代结论

- HotpotQA：v6.2 相对同 sampling schedule 的 v6.1 提升 EM `3.9063` pp、F1 `4.1946` pp。通用 `ANSWER_GENERATION_PROMPT` 和 22-round horizon 两个候选均因 paired panel 回归而被拒绝。
- TriviaQA：v6.2 相对旧 v3 为 EM `+0.7813` pp、F1 `-3.2605` pp、FINISH `-9`，不能称为整体优于 v3。针对 stale Format-predecessor artifact 的 v6.3 candidate 在 13 题 panel 上保持 EM、F1 `+2.1978` pp，但 FINISH `-1`，因此已回退。

本轮保留的原则是：只有 fixed-panel 证据同时支持 accuracy/terminal boundary，才进入有效源码；局部增益伴随明确回归时保留 trajectory 作为 rejection evidence，不进行 benchmark-specific answer rule、强制 topology 或 evaluator-aware repair。

## 5. 实现状态

| 模块 | 状态 |
| --- | --- |
| FlowSteer progressive Canvas、edit -> execute -> feedback、显式 FINISH | 已实现并运行 |
| 本地 Qwen3.5-9B Flow-Director | 已实现并运行 |
| 异构 AgentGraph Runtime、Agent communication、Format | 已实现并运行 |
| SkillFlow QA search/read、public continuation、Tool receipt | 已实现并运行 |
| HotpotQA / TriviaQA official answer evaluator 与完整 trajectory | 已实现并运行 |
| MACE -> Bayesian posterior -> Skill evidence gate | 项目已有接口/历史实验；本轮未运行 |
| ACTIVE Skill 检索与注入 | 本轮未启用 |
| LoRA/GRPO、backward、optimizer update、policy sync | 本轮未执行 |

## 6. 可复核文件

- HotpotQA：`reports/qa_orchestration_tool_v6_2_development/HOTPOTQA_ARCHITECTURE_REPORT_ZH.md`
- TriviaQA：`reports/qa_orchestration_tool_v6_2_development/TRIVIAQA_ARCHITECTURE_REPORT_ZH.md`
- HotpotQA trajectory：`artifacts/qa_orchestration_tool_v6_2_development/hotpotqa/tool_agentgraph_trajectories.jsonl`
- TriviaQA trajectory：`artifacts/qa_orchestration_tool_v6_2_development/triviaqa/tool_agentgraph_trajectories.jsonl`
- TriviaQA v6.3 rejection evidence：`artifacts/triviaqa_format_predecessor_v6_3_panel/`

所有报告分数均来自落盘 evaluator receipt，不使用 behavior reward、训练样本循环补齐结果或人工文本相似度替代官方 answer metric。
