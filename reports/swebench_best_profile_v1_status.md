# SWE-bench best-profile v1 状态

## 结论

严格筛选后，当前不存在可切换的“已验证最高 AgentGraph 架构”。合格条件数为 **0**：
现有 SWE-bench 条件全部是 `prepared`、`failed_runtime_preflight`、小样本/不同 split，或没有
完整 official evaluator receipt。因而：

| 项目 | 结果 |
|---|---:|
| Best AgentGraph condition | **N/A（无合格候选）** |
| 固定任务数 | 128（目标协议；尚未完成） |
| AgentGraph Resolved / Resolved Rate | N/A / **N/A** |
| Direct Resolved / Resolved Rate | N/A / **N/A** |
| AgentGraph − Direct | **N/A** |
| 完整 official evaluator-valid AgentGraph 条件数 | **0** |

SWE-bench 的 N/A 不能写成 0%，也不能把 canary、local `run_tests`、prepared-only manifest、
proxy metric 或模型文本当作 Resolved。没有仅通过重命名配置制造 `best` 条件。

## 当前候选（不是 best）

- condition：`swebench_skillflow_v3_initial_v1`；
- config：`config/evaluation_swebench_skillflow_v3_initial_v1.yaml`；
- benchmark/split/分母：SWE-bench Verified / test / 128；
- protocol：SkillFlow v3 `code_generation` IID128，gold/test contract 只在 evaluator boundary；
- AgentGraph：自由 `agent_id + model_id + free-text contract`，统一 Canvas actions；
- Director prompt：`agentgraph.director.minimal-neutral.v10`；
- repository Tool：`skillflow.training.repository-tools.v1+flowsteer.workspace-diff.v1`；
- evaluator：SkillFlow `evaluate_patch` → official SWE-bench Docker harness；
- status：`failed_runtime_preflight`；
- official-evaluator-valid：0；model calls：0；official labels：0。

版本化 pointer `config/swebench_best_profile_v1.yaml` 的状态为
`no_evaluator_valid_candidate`，`activation_allowed=false`。evaluation runner 的 `--config`
是必填参数，因此不存在隐式默认配置；当前文档化
的 next-run 配置是 `config/evaluation_swebench_skillflow_v3_initial_v1.yaml`。本轮没有修改该入口，
也没有把它声明为 best-profile。当前候选源码版本为
`08c4252cba32e63f3d33260a56964b0745e3f916`。

## 排除的历史条件

| Condition | Split / denominator | Status | AgentGraph trajectory | Official evaluator receipt | 排除原因 |
|---|---:|---|---:|---:|---|
| `swebench_regular_dev_coding_agent_v2` | validation / 128 | `failed_runtime_preflight` | 0 | 0 | 未完成 |
| `swebench_regular_dev_coding_agent_stable_zero` | validation / 128 | `failed_runtime_preflight` | 0 | 0 | 未完成 |
| `swebench_regular_dev_unified_architecture_v1` | validation / 128 | `prepared` | 0 | 0 | prepared-only |
| `swebench_regular_dev_development_skill_off` | validation / 32 | `prepared` | 0 | 0 | 小样本且未运行 |
| historical Verified coding condition | validation / 128 | `failed_runtime_preflight` | 0 | 0 | 未完成/不同 split |
| historical Verified evaluation | validation / 128 | `prepared` | 0 | 0 | prepared-only/不同 split |
| `swebench_verified_evaluation_skill_off` | test / 128 | `prepared` | 0 | 0 | prepared-only |
| historical development condition | train / 32 | `failed_runtime_preflight` | 0 | 0 | 不同 split/分母且未完成 |
| `swebench_skillflow_v3_initial_v1` | test / 128 | `failed_runtime_preflight` | 0 | 0 | official evaluator coverage=0 |

## 当前阻塞与证据

- task-specific SkillFlow environment：1/128 ready，127 unavailable；
- official Docker harness：当前用户无 daemon socket 权限；
- 因此没有启动模型、没有 official evaluator task call，也没有生成 AgentGraph accuracy。

证据：

- `reports/swebench_skillflow_v3_initial_v1_preflight.md`
- `reports/swebench_skillflow_v3_initial_v1_preflight.json`
- `artifacts/swebench_skillflow_v3_initial_v1/run_manifest.json`
- `/ssd1/iclr/1/FlowSteer/reports/multidataset_stablezero/SWEBENCH_ARCH_REPORT.md`
- `/ssd1/iclr/1/FlowSteer/artifacts/eval128_current/swebench_regular_dev/run_manifest.json`

本轮没有训练、GRPO/LoRA/MACE 更新、模型/API 调用、全量付费 evaluation 或 official
harness rerun。
