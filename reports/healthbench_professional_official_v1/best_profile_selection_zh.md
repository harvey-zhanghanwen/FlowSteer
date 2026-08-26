# HealthBench Professional AgentGraph Best Profile

## 选择结论

在同一 HealthBench Professional 官方 public `test`、同一 525 条任务、同一
OpenAI `simple-evals` reference evaluator 和同一 strict full-denominator
口径下，仓库中只有一个已完整收束的 AgentGraph 条件：
`healthbench_professional_official_v1`。因此它既是唯一合格候选，也是当前
已验证主指标最高的条件。这里没有把 Direct、128 条 validation、2 条
canary、prepared-only 或不同 Tool/evaluator protocol 的结果纳入排序。

## 已验证指标

本项目按 OpenAI `simple-evals` reference implementation 计算的
reference-compatible 主指标为 `overall_score_length_adjusted`，不是
Accuracy、EM 或 F1；它不冒充未公开的 production/private leaderboard
evaluator 分数。

| 条件 | 分母 | Evaluator valid | Strict length-adjusted | Strict raw |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 525 | 525 | 19.1728% | 18.9721% |
| AgentGraph best profile | 525 | 503 | **20.2395%** | **22.6451%** |
| AgentGraph − Direct | — | — | **+1.0667 pp** | **+3.6730 pp** |

AgentGraph 形成了 525/525 个最终 trajectory receipt，其中 503 个合法
`FINISH` 并通过 evaluator，22 个为 `max_rounds` terminal failure。Pending
evaluator retry 为 0，当前 operational/evaluator failure 为 0。仅在 503 个
evaluator-valid 样本上计算的 length-adjusted 指标为 21.1247%，它只作为辅助
指标，不替代 525 分母的 strict 主指标。

## Best profile

- condition：`healthbench_professional_official_v1`
- evaluated source commit：`3f1625642a5fd5fb284b37e158e943f13422f7ef`
- executable config：`config/evaluation_healthbench_professional_official_v1.yaml`
- versioned pointer：`config/healthbench_professional_best_profile_v1.yaml`
- current project pointer：`config/healthbench_professional_best_profile.yaml`
- Director：本地 Qwen3.5-9B base，无 LoRA adapter
- prompt：`agentgraph.director.minimal-neutral-scalar.v1`
- Canvas：incremental `execute_on_edit`
- search space：最多 8 Agent、20 rounds、two-bit relation、唯一 Output Agent
- recovery：`preserve_diagnose_repair_augment`
- Tool、Skill、GRPO、LoRA、MACE、Bayesian update：全部关闭

current project pointer 和版本化 descriptor 的 `next_run` 都已选择上述
executable config。仓库没有全局自动 default resolver，因此实际运行仍需
显式传入该 config；本轮未授权、也没有重跑 525 条正式付费评测。

## 排除项

| 条件 | 排除原因 |
| --- | --- |
| `healthbench_professional_round_01/evaluation` | validation/128，旧 raw-score protocol |
| `healthbench_professional_medrag_tool_stable_zero_v2/development` | development validation/128，仅 124 evaluator-valid，Tool protocol 不同 |
| `healthbench_professional_medrag_tool_stable_zero/canary` | 2 条 canary |
| `unified_architecture_v1/healthbench_professional_internal` | prepared-only，无正式 evaluator-valid 结果 |

## Evidence

- `reports/healthbench_professional_official_v1/evaluation_report.json`
- `reports/healthbench_professional_official_v1/evaluation_report_zh.md`
- `artifacts/healthbench_professional_official_v1/evaluation/run_manifest.json`
- `config/evaluation_healthbench_professional_official_v1.yaml`

本轮只建立可恢复的 best-profile 指针和证据约束；统一 AgentGraph core、正式
trajectory、candidate response、评分和模型权重均未改变。
