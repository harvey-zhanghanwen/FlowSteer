# 多数据集 Agent 架构 Stable Zero 报告

## 架构完成情况

控制路径保持为：本地 Qwen3.5-9B Flow-Director、one-atomic-edit progressive Canvas、execute-after-edit feedback、dynamic AgentGraph、显式 FINISH、数据集原生 evaluator 与完整 trajectory receipt。统一 AgentRuntime 分发 `reasoning`、Tool/ReAct、environment ReAct 和 `coding` execution adapter。Tool assignment、model selection、自由文本 contract、dependency、artifact type 与 completion condition 仍属于 Director search space。

本轮未执行大规模训练、GRPO、backward、optimizer update、LoRA publication 或新的 Skill activation。

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
| TriviaQA | PASS | 2 | 1 | 1 | exact_match: Direct=50.00%, AgentGraph=50.00%; token_f1: Direct=50.00%, AgentGraph=92.86% | 仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力 |
| AIME 2026 | PASS | 2 | 2 | 0 | exact_match: Direct=50.00%, AgentGraph=100.00% | 推理 + 有界 calculator/Python execution 能力 |
| HealthBench Professional | PASS | 2 | 0 | 2 | raw_score: Direct=0.2000, AgentGraph=0.2000 | 临床推理 + 冻结教材语料 MedRAG search 能力 |
| WebShop | PASS | 2 | 2 | 0 | success: Direct=100.00%, AgentGraph=100.00% | request-scoped SkillFlow/RAGEN environment ReAct |
| ALFWorld | PASS | 2 | 2 | 0 | success: Direct=50.00%, AgentGraph=100.00% | request-scoped SkillFlow/RAGEN environment ReAct |
| SWE-bench Verified | FAIL | 0 | 0 | 0 | 不可测 | detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness |

以上均为固定 2 题 Stable Zero 行为结果（AIME 从 30 个官方任务中固定选取），不是正式 benchmark 或 SOTA 估计。SWE-bench 标记为不可测，不以代理零分替代。

## Skill evidence gate

- `hotpotqa`: Skill=`jointqa.hotpotqa.exact_answer_handoff`, status=`candidate`, effective_pairs=40, paired_effect_mean=-0.0035714285714285726, calibrated_interval=[-0.10357142857142858, 0.1], harm_probability=0.51185; gate reasons=['calibrated lower bound does not exceed delta_min', 'negative-transfer probability exceeds harm limit', 'effect direction is inconsistent across task slices']
- `triviaqa`: Skill=`jointqa.triviaqa.exact_answer_handoff`, status=`candidate`, effective_pairs=40, paired_effect_mean=0.05361111111111111, calibrated_interval=[-0.04083333333333333, 0.15638888888888886], harm_probability=0.1322; gate reasons=['calibrated lower bound does not exceed delta_min', 'negative-transfer probability exceeds harm limit']

最新 evidence-gated `ACTIVE` Skill 数量：**0**。因此不存在可注入新多数据集 Director condition 的版本兼容 `ACTIVE` Skill，也不满足 Skill-on micro-training 的触发条件。`CANDIDATE` instruction 仍是候选，不作为已验证 Skill。

## 剩余问题分类

- `ENVIRONMENT_LIMITATION`：SWE-bench 官方 Docker harness 无法访问 Docker daemon，官方 `resolved_rate` 不可测。
- `SKILL_EVIDENCE_INSUFFICIENT`：最新独立 paired evidence 未满足 calibrated lower-bound/harm gate；`ACTIVE` Skill 数为 0。
- `POLICY_LEARNING_PROBLEM`：Tool 能力已经接线，但 HotpotQA、TriviaQA 与 AIME 的 Stable Zero graph 没有自然选择可选 retrieval/computation Tool。
- `MODEL_CAPABILITY_LIMIT`：HealthBench 的 2 题样本在 evaluator-valid execution 下 raw_score 仍低；该样本不足以证明更窄的架构缺陷。
- `ARCHITECTURE_DEFECT`（已修复）：WebShop JSON/native-action 不匹配和 ALFWorld 缺少 environment actor 的 terminal condition 已通过最小 executor/terminal adaptation 修正并独立复跑。

## 最终判定

```text
FLOWSTEER_CORE_PRESERVED = YES

MODEL_POOL_EXPANDED = YES
MULTI_MODEL_WORKFLOW_READY = YES
DEEP_WORKFLOW_READY = YES
COLLABORATION_DIVERSITY_READY = YES

QA_TOOL_REGISTRY_READY = YES
QA_DATABASE_SELECTION_READY = YES
QA_TOOL_USE_VALIDATED = NO

ALFWORLD_REACT_READY = YES
WEBSHOP_REACT_READY = YES

CODING_AGENT_READY = YES
SWEBENCH_CODING_WORKFLOW_READY = NO

SKILL_END_TO_END_READY = YES
SKILL_SUMMARY_VALIDATED = NO

ALL_DATASETS_STABLE_ZERO_COMPLETE = NO
CORRECT_WRONG_DEMOS_COMPLETE = NO

MICRO_TRAINING_EXECUTED = NO
LEARNING_TREND_OBSERVED = NO

GITHUB_ARCHITECTURE_BACKUP = NO

READY_FOR_FORMAL_MULTIDATASET_TRAINING = NO
```

`GITHUB_ARCHITECTURE_BACKUP = NO` 表示当前 branch 在本地可恢复，但生成报告时远端认证不可用；正常认证的 `git push` 成功前，不得表述为已推送。

## 报告索引

- [HotpotQA](reports/multidataset_stablezero/HOTPOTQA_ARCH_REPORT.md)
- [TriviaQA](reports/multidataset_stablezero/TRIVIAQA_ARCH_REPORT.md)
- [AIME 2026](reports/multidataset_stablezero/AIME2026_ARCH_REPORT.md)
- [HealthBench Professional](reports/multidataset_stablezero/HEALTHBENCH_PROFESSIONAL_ARCH_REPORT.md)
- [WebShop](reports/multidataset_stablezero/WEBSHOP_ARCH_REPORT.md)
- [ALFWorld](reports/multidataset_stablezero/ALFWORLD_ARCH_REPORT.md)
- [SWE-bench Verified](reports/multidataset_stablezero/SWEBENCH_ARCH_REPORT.md)
