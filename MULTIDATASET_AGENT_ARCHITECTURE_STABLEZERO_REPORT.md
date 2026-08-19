# 多数据集 Agent 架构 Stable Zero 报告

## 架构完成情况

控制路径保持为：本地 Qwen3.5-9B Flow-Director、one-atomic-edit progressive Canvas、execute-after-edit feedback、dynamic AgentGraph、显式 FINISH、数据集原生 evaluator 与完整 trajectory receipt。统一 AgentRuntime 分发 `reasoning`、Tool/ReAct、environment ReAct 和 `coding` execution adapter。Tool assignment、model selection、自由文本 contract、dependency、artifact type 与 completion condition 仍属于 Director search space。

当前所读 manifests 的 optimizer update 总数：**0**。本轮未执行大规模训练、GRPO、backward、LoRA publication 或新的 Skill activation。

## 实现来源分类

- `DIRECT_REUSE`：FlowSteer progressive Canvas 的 edit→execute→feedback、显式 FINISH、action mask、trajectory；SkillFlow 的 StructuredAction/Tool Registry、RetrievalIndex、bounded computation、RAGEN environment、MedRAG corpus、SWE-bench worktree 与 evidence/library contract。
- `NECESSARY_ADAPTATION`：异构 `reasoning|react|coding` dispatch、task-scoped Tool registry、typed evaluator receipt、WebShop 原生 action grammar、ALFWorld interactive FINISH 的 environment actor invariant、SWE-bench worktree ownership。
- `PROJECT_ALGORITHM_ADDITION`：typed `CommunicationEnvelope`、`ToolCapability`、measured `ToolReceipt` 与既有 same-prefix paired AgentGraph posterior/evidence gate。
- `NOT_IMPLEMENTED_OR_NOT_EXECUTED`：SWE-bench 官方-harness-valid Coding trajectory、evidence-gated `ACTIVE` Skill 注入以及本轮 micro-training/optimizer/policy synchronization。

逐文件的上游类/函数与不兼容原因记录在 `docs/SOURCE_MAP.md`。

## Stable Zero 结果

| Dataset | Stable Zero | n | 满分/成功 | 错误 | 原生指标 | 能力边界 |
|---|---:|---:|---:|---:|---|---|
| HotpotQA | PASS | 2 | 2 | 0 | exact_match: Direct=100.00%, AgentGraph=100.00%; token_f1: Direct=100.00%, AgentGraph=100.00% | 闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力 |
| TriviaQA | PASS | 2 | 2 | 0 | exact_match: Direct=50.00%, AgentGraph=100.00%; token_f1: Direct=50.00%, AgentGraph=100.00% | 仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力 |
| AIME-2025 Development（AIME 2026 目标适配） | PASS | 2 | 1 | 1 | exact_match: Direct=0.00%, AgentGraph=50.00% | 推理 + 有界 calculator/Python execution 能力 |
| HealthBench Professional（reference-judge diagnostic） | PASS | 2 | 0 | 2 | raw_score: Direct=0.2000, AgentGraph=0.2000 | 临床推理 + 冻结教材语料 MedRAG search 能力 |
| WebShop | PASS | 2 | 1 | 1 | success: Direct=50.00%, AgentGraph=50.00% | request-scoped SkillFlow/RAGEN environment ReAct |
| ALFWorld | PASS | 2 | 2 | 0 | success: Direct=50.00%, AgentGraph=100.00% | request-scoped SkillFlow/RAGEN environment ReAct |
| SWE-bench Regular Dev | NOT_RUN | 0 | 0 | 0 | resolved: Direct=不可测, AgentGraph=不可测 | detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness |

只有存在当前 evidence scope 下的 paired result 与原生 evaluator receipt 时才显示数值；缺失项显示“不可测”，不填 0。AIME 数值来自 AIME-2025 development canary，**不是 AIME 2026 benchmark 成绩**；WebShop v4 只报告当前 native-validation paired receipts。SWE-bench regular-dev 尚无 paired result，不以代理零分替代。

## Runtime receipts

- 显式 FINISH：**12** 条 trajectory
- ToolReceipt（含 environment action）：**26**
- QA / computation / MedRAG 自然策略 ToolReceipt：**2**
- Environment transition receipt：**24**
- Coding action receipt：**0**

## Exact-schema Tool forced probe（不计入 benchmark）

| Dataset | Overall | Schema | Backend | Model/termination | Successful ToolReceipt |
|---|---:|---:|---:|---:|---:|
| HotpotQA | failed | PASS | PASS | FAIL | 4 |
| TriviaQA | passed | PASS | PASS | PASS | 2 |
| AIME-2025 Development（AIME 2026 目标适配） | failed | FAIL | PASS | FAIL | 2 |
| HealthBench Professional（reference-judge diagnostic） | passed | PASS | PASS | PASS | 1 |

这些 receipt 均为 `diagnostic_only=true`、`forced_probe=true`，没有 evaluator、Ground Truth、benchmark metric、Skill evidence、GRPO 或 optimizer update；不能与自然策略 Tool adoption 混合计数。

## Protocol audit

- HotpotQA、TriviaQA 与 AIME-2025 development：Direct 与 Tool-capable AgentGraph 分别报告；未把 protocol-separated delta 解释为 architecture causal effect 或 SOTA improvement。
- HealthBench Professional v2：只报告 openai/simple-evals-compatible **reference-judge diagnostic**；不是私有官方评测服务或 leaderboard 成绩。
- WebShop v4：只接受 native validation indices 500..627 的原生环境结果；旧 v2 native-test 结果作为 test-contaminated adaptation evidence 排除，v3 仅保留为上下文预算失败诊断。
- ALFWorld：Direct/Simple ReAct 和 AgentGraph 使用相同 task lock、原生环境、action budget 与 evaluator，可进行同条件描述性比较。
- SWE-bench：架构开发只使用 regular dev；完整 Verified 保留给最终评测。没有官方 Docker harness receipt 时 resolved/resolved_rate 不可测。
- 所有原生 evaluator 都是唯一 terminal metric source；LLM self-judgement、文本相似度或 local proxy test 均未替代正式指标。

## Skill evidence gate

- `hotpotqa`: Skill=`jointqa.hotpotqa.exact_answer_handoff`, status=`candidate`, effective_pairs=40, paired_effect_mean=-0.0035714285714285726, calibrated_interval=[-0.10357142857142858, 0.1], harm_probability=0.51185; gate reasons=['calibrated lower bound does not exceed delta_min', 'negative-transfer probability exceeds harm limit', 'effect direction is inconsistent across task slices']
- `triviaqa`: Skill=`jointqa.triviaqa.exact_answer_handoff`, status=`candidate`, effective_pairs=40, paired_effect_mean=0.05361111111111111, calibrated_interval=[-0.04083333333333333, 0.15638888888888886], harm_probability=0.1322; gate reasons=['calibrated lower bound does not exceed delta_min', 'negative-transfer probability exceeds harm limit']

最新 evidence-gated `ACTIVE` Skill 数量：**0**。因此不存在可注入新多数据集 Director condition 的版本兼容 `ACTIVE` Skill，也不满足 Skill-on micro-training 的触发条件。`CANDIDATE` instruction 仍是候选，不作为已验证 Skill。

## 剩余问题分类

- SWE-bench regular-dev 当前尚无 official Docker harness evaluator receipt，因此 resolved_rate 不可测，不能提前归因为 Coding Agent 或 patch quality。
- `SKILL_EVIDENCE_INSUFFICIENT`：最新独立 paired evidence 未满足 calibrated lower-bound/harm gate；`ACTIVE` Skill 数为 0。
- HotpotQA 与 TriviaQA v3 的当前 Stable Zero trajectories 各自然产生 1 条成功 retrieval `ToolReceipt`；AIME-2025 development 与 HealthBench v2 未自然选择其可选 Tool。两题 canary 只能验证自然工具调用链已经出现，不能估计 tool-use policy 的总体采用率、收益或 Skill effect。
- HealthBench 的 2 题 reference-judge diagnostic raw_score 较低；该 canary 不足以证明模型能力或架构的单一原因。
- WebShop v4 已形成 native-validation paired receipt；旧 v2 native-test success 仍不进入当前指标，v3 仅保留为上下文预算失败诊断。

## 最终判定

```text
FLOWSTEER_CORE_PRESERVED = YES

MODEL_POOL_EXPANDED = YES
MULTI_MODEL_WORKFLOW_READY = YES
DEEP_WORKFLOW_READY = YES
COLLABORATION_DIVERSITY_READY = YES

QA_TOOL_REGISTRY_READY = YES
QA_DATABASE_SELECTION_READY = YES
QA_TOOL_USE_VALIDATED = YES

ALFWORLD_REACT_READY = YES
WEBSHOP_REACT_READY = YES

CODING_AGENT_READY = YES
SWEBENCH_CODING_WORKFLOW_READY = NO

SKILL_END_TO_END_READY = NO
SKILL_SUMMARY_VALIDATED = NO

ALL_DATASETS_STABLE_ZERO_COMPLETE = NO
CORRECT_WRONG_DEMOS_COMPLETE = NO

MICRO_TRAINING_EXECUTED = NO
LEARNING_TREND_OBSERVED = NO

GITHUB_ARCHITECTURE_BACKUP = NOT_EVALUATED_BY_REPORT_GENERATOR

READY_FOR_FORMAL_MULTIDATASET_TRAINING = NO
```

判定说明：`DEEP_WORKFLOW_READY` 表示 search space 与 scheduler 支持 deep/parallel/fan-in/finite-reciprocal motif，不表示当前 canary 已普遍采用深图。`CORRECT_WRONG_DEMOS_COMPLETE` 只表示七个数据集是否都已有 evaluator-valid paired result；不要求为获得错例而人为制造失败。报告生成器不执行或验证 Git push，因此不对远端备份状态作结论。

## 报告索引

- [HotpotQA](reports/multidataset_stablezero/HOTPOTQA_ARCH_REPORT.md)
- [TriviaQA](reports/multidataset_stablezero/TRIVIAQA_ARCH_REPORT.md)
- [AIME-2025 Development（AIME 2026 目标适配）](reports/multidataset_stablezero/AIME2026_ARCH_REPORT.md)
- [HealthBench Professional（reference-judge diagnostic）](reports/multidataset_stablezero/HEALTHBENCH_PROFESSIONAL_ARCH_REPORT.md)
- [WebShop](reports/multidataset_stablezero/WEBSHOP_ARCH_REPORT.md)
- [ALFWorld](reports/multidataset_stablezero/ALFWORLD_ARCH_REPORT.md)
- [SWE-bench Regular Dev](reports/multidataset_stablezero/SWEBENCH_ARCH_REPORT.md)
