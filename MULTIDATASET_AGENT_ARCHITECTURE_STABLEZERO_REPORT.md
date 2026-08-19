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
| SWE-bench Regular Dev | FAIL | 0 | 0 | 0 | resolved: Direct=不可测, AgentGraph=不可测 | detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness |

只有存在当前 evidence scope 下的 paired result 与原生 evaluator receipt 时才显示数值；缺失项显示“不可测”，不填 0。AIME 数值来自 AIME-2025 development canary，**不是 AIME 2026 benchmark 成绩**；WebShop v4 只报告当前 native-validation paired receipts。SWE-bench regular-dev 尚无 paired result，不以代理零分替代。

## Runtime receipts

- 显式 FINISH：**12** 条 trajectory
- ToolReceipt（含 environment action）：**26**
- QA / computation / MedRAG 自然策略 ToolReceipt：**2**
- Environment transition receipt：**24**
- Coding action receipt：**0**

## Natural Stable Zero workflow/model adoption

- Exclusive topology family：single=1, serial_2=8, serial_3_plus=3, parallel=0, fan_in=0, fan_out=0, reciprocal=0, verification=0, mixed=0
- Declared Executor node family：Qwen=8, DeepSeek=6, Gemini=0, GPT=11, MiniMax=0, Grok=0, GLM=1, Other=0
- Multi-model workflow：**5/12**

当前 12 条自然 AgentGraph trajectory 只观察到 single/serial topology；parallel、fan-in、fan-out 与 reciprocal 在独立 non-chain runtime diagnostic 中可执行，但尚未被当前 fixed-task Director policy 自然采用。因此 `DEEP_WORKFLOW_READY` 只表示 search space/runtime capability，不能解释为 policy adoption 或性能收益。

## Model capability Canary

| Exact catalog/model ID | Provider | Text | StructuredAction/ReAct | Coding format | Admitted Executor | Receipt |
|---|---|---:|---:|---:|---:|---|
| deepseek-v4-flash | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| glm-4.5-flash | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| gpt-4o-mini | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| MiniMax-M2.5 | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| MiniMax-M3 | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| qwen3.5-9b-local | local-director | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/local_qwen35_9b_nonthinking_20260820.json` |
| qwen3.5-flash | vectorengine | PASS | PASS | PASS | YES | `artifacts/model_capability_canary/cheap_fast_20260819.json` |
| grok-4-1-fast-non-reasoning | vectorengine | http_error | http_error | http_error | NO | `artifacts/model_capability_canary/cheap_fast_20260819.json` |

- Catalog entries with all three saved canaries：**7/7**
- `grok-4-1-fast-non-reasoning` 的三个 probe 均收到 HTTP 429，因此没有纳入 catalog；这不是把失败别名替换成另一个模型。
- `/v1/models` 与 canary 均未提供通过验证的 Gemini exact model ID，所以 Gemini 显式保持 0，不凭空加入。
- Flow-Director 仍固定为 local Qwen3.5-9B；表内远端模型只进入 Executor search space。


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

## 既有 bounded micro-training 证据

这组证据来自当前项目此前完成的 HotpotQA/TriviaQA joint-QA Flow-Director 训练闭环；它与本轮统一 Tool/Environment/Coding Stable Zero 的版本边界分开报告，不能证明新 Tool action-selection policy 已被训练。

| Manifest | Behavior policy | Updated policy | Optimizer updates | Trainable update L2 | Policy sync | Post-update canary |
|---|---|---|---:|---:|---:|---:|
| artifacts/joint_qa_micro/step_000001/training_manifest.json | qwen35-9b-hotpot-step-000000 | qwen35-9b-jointqa-step-000001 | 1 | 0.027050 | PASS | 2 |
| artifacts/joint_qa_micro/step_000002_attempt_04/training_manifest.json | qwen35-9b-jointqa-step-000001 | qwen35-9b-jointqa-step-000002 | 1 | 0.025979 | PASS | 2 |

- Receipt-valid optimizer updates：**2/2**
- Matched held-out curve receipt：`reports/joint_qa_curve/final/joint_qa_curve.json`；fixed task/evaluator/policy receipts verified=`true`

| Step | Policy version | HotpotQA/TriviaQA macro EM | Macro F1 |
|---|---|---:|---:|
| Step 0 | qwen35-9b-hotpot-step-000000 | 56.25% | 68.58% |
| Step 1 | qwen35-9b-jointqa-step-000001 | 56.25% | 66.41% |
| Step 2 | qwen35-9b-jointqa-step-000002 | 54.69% | 65.31% |

该证据证明 LoRA 参数更新、optimizer state、policy publication、route switch 和 post-update canary 的闭环可执行。固定 held-out 的最终宏平均没有超过 Step 0，因此不能声称观察到正向 learning trend；按方案应优先继续检查 architecture/search-space 与 evidence quality，而不是扩大训练规模。


## Current add_subgraph micro-training preflight

- Decision：`NO_GO`
- Receipt report：`reports/multidataset_stablezero/MICROTRAINING_PREFLIGHT.md`
- 当前没有新的冻结 schedule/cursor，也没有满足 evidence gate 的 `ACTIVE` Skill；现成 `add_subgraph` 模板是 Skill-on。GPU 3 被本任务之外的进程占用，GPU 4 rollout service 已关闭，provider credential 未配置。因而本轮没有启动服务或训练，也没有把旧 cursor 重放称为新更新。

## 剩余问题分类

- `ENVIRONMENT_LIMITATION`：SWE-bench regular-dev 当前尚无 official Docker harness evaluator receipt，因此 resolved_rate 不可测，不能提前归因为 Coding Agent 或 patch quality。
- `SKILL_EVIDENCE_INSUFFICIENT`：最新独立 paired evidence 未满足 calibrated lower-bound/harm gate；`ACTIVE` Skill 数为 0。
- `TRAINING_INSTABILITY`：既有 joint-QA bounded micro-training 完成了 2 次真实 optimizer update 与 policy sync，但固定 held-out 宏平均没有形成正向趋势；该证据不覆盖本轮新增 Tool/Environment/Coding action-selection policy。
- `TOOL_LIMITATION`：HotpotQA 与 TriviaQA v3 的当前 Stable Zero trajectories 各自然产生 1 条成功 retrieval `ToolReceipt`；AIME-2025 development 与 HealthBench v2 未自然选择其可选 Tool。两题 canary 只能验证自然工具调用链已经出现，不能估计 tool-use policy 的总体采用率、useful rate、wasted rate 或 Skill effect。
- `MODEL_CAPABILITY_LIMIT`：HealthBench 的 2 题 reference-judge diagnostic raw_score 较低；该 canary 不足以把差距唯一归因于模型、架构或缺少检索。
- `ARCHITECTURE_DEFECT`：当前没有新的 confirmed open defect。WebShop 的 action serialization / token-budget 缺陷已按 preserved failure receipt 修复；WebShop v4 已形成 native-validation paired receipt；旧 v2 native-test success 仍不进入当前指标，v3 仅保留为上下文预算失败诊断。
- `POLICY_LEARNING_PROBLEM`：尚未成立。当前自然 policy 没有采用非链式 topology，但 SWE Coding、ACTIVE Skill 和 Tool usefulness 闭环仍未齐全，不能先把缺口归因于 policy learning。

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

CODING_AGENT_IMPLEMENTED = YES
CODING_AGENT_READY = NO
SWEBENCH_CODING_WORKFLOW_READY = NO

SKILL_END_TO_END_READY = NO
SKILL_SUMMARY_VALIDATED = NO

ALL_DATASETS_STABLE_ZERO_COMPLETE = NO
CORRECT_WRONG_DEMOS_COMPLETE = NO

MICRO_TRAINING_EXECUTED = YES
LEARNING_TREND_OBSERVED = NO

LOCAL_RECOVERY_BACKUP = YES
GITHUB_ARCHITECTURE_BACKUP = NO

READY_FOR_FORMAL_MULTIDATASET_TRAINING = NO
```

判定说明：`DEEP_WORKFLOW_READY` 表示 search space 与 scheduler 支持 deep/parallel/fan-in/finite-reciprocal motif，不表示当前 canary 已普遍采用深图。`CORRECT_WRONG_DEMOS_COMPLETE` 要求每个数据集同时具有当前 evaluator-valid correct 与 wrong receipt；不会为凑数量制造失败。当前 branch/tag/patch/bundle 已形成可恢复本地备份，但当前环境对配置的 GitHub remote 没有可用 HTTPS/SSH 写认证，所以 `GITHUB_ARCHITECTURE_BACKUP=NO`，不伪称已推送。

## 报告索引

- [HotpotQA](reports/multidataset_stablezero/HOTPOTQA_ARCH_REPORT.md)
- [TriviaQA](reports/multidataset_stablezero/TRIVIAQA_ARCH_REPORT.md)
- [AIME-2025 Development（AIME 2026 目标适配）](reports/multidataset_stablezero/AIME2026_ARCH_REPORT.md)
- [HealthBench Professional（reference-judge diagnostic）](reports/multidataset_stablezero/HEALTHBENCH_PROFESSIONAL_ARCH_REPORT.md)
- [WebShop](reports/multidataset_stablezero/WEBSHOP_ARCH_REPORT.md)
- [ALFWorld](reports/multidataset_stablezero/ALFWORLD_ARCH_REPORT.md)
- [SWE-bench Regular Dev](reports/multidataset_stablezero/SWEBENCH_ARCH_REPORT.md)
