# SWE-bench Verified 初版适配与评测报告（v19）

## 结论

SWE-bench Stable Zero 已跑通，固定 128 个 SWE-bench Verified task 的 Direct 与 AgentGraph 两个条件均完成官方 evaluator，最终状态为 `completed`。本轮没有训练、GRPO、MACE、Bayesian、LoRA 更新或 Skill evolution。

| 条件 | 完成数 | evaluator-valid | Resolved | Resolved Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 128 | 128 | 4 | 3.125% |
| AgentGraph | 128 | 128 | 4 | 3.125% |

AgentGraph 相对 Direct 的 Resolved Rate 差值为 **0.000 个百分点**。这是同任务、同 repository snapshot、同 repository Tool、同 task-global execution budget 下的描述性对比；AgentGraph 还包含 Director 与多 Agent 推理开销。

## 实际阻塞问题与修复

1. **Git 暂存区未进入最终补丁。** SkillFlow 的 repository `bash` action 允许 Agent 执行 `git add`，但旧实现使用不带 revision 的 `git diff`，只能得到 Git index 与 working tree 之间的差异。已改为 `git diff HEAD --binary`，最终补丁现在包含 `HEAD` 到最终 working tree 的 staged 与 unstaged 修改。
2. **Docker 容器无法解析 linked worktree 的 `.git` 文件。** `/testbed/.git` 指向宿主 source repository 的 Git common directory，但该目录之前没有挂载进容器，官方 harness 中的 `git apply`、`git diff`、`git reset`、`git checkout` 无法完整执行。现在把 `prepared.source_repository/.git` 挂载到容器内相同的绝对路径。
3. **Stable Zero 错误拒绝官方 `patch_apply_failed` 回执。** SkillFlow 官方 evaluator 将该状态作为 evaluator-valid 的 Failed 结果返回（`Resolved=0`），不是 evaluator infrastructure failure。Stable Zero 现在保留它的 0 分并接受其官方回执。
4. **SGLang `/server_info` 的 `max_running_requests` 为 `null`。** 当前 Supervisor 在 `internal_states[*].effective_max_running_requests_per_dp` 返回实际值 82。运行前检查现在优先使用显式 `max_running_requests`，为空时读取该实际字段。
5. **旧 Direct checkpoint 与新补丁生成实现混用。** v19 更新 Direct protocol，并取消 v17/v18 Direct 结果复用；128 条 Direct 全部重新生成。没有按 task ID 或官方答案做样本特定处理。

## Stable Zero 与完整运行

- Stable Zero：3/3 Direct 与 3/3 AgentGraph 均有完整 trajectory 和官方 evaluator receipt。
- 完整运行：Direct 128/128；AgentGraph 128/128。
- AgentGraph：`FINISH` 128/128，`max_rounds` 0，terminal failure 0，最终 operational failure 0。
- `sphinx-doc__sphinx-8056` 首次运行时 Director 返回截断 JSON，产生 `ReceiptValidationError`；同一 v19 condition 断点续跑后完成。该历史失败回执被保留，但最终任务有 evaluator-valid 结果。

## 官方 evaluator 结果分布

| 条件 | `resolved` | `unresolved` | `empty_patch` | `patch_apply_failed` |
|---|---:|---:|---:|---:|
| Direct | 4 | 35 | 89 | 0 |
| AgentGraph | 4 | 28 | 95 | 1 |

AgentGraph 额外解决了 `sympy__sympy-13480`，但没有解决 Direct 成功的 `django__django-10914`；其余 3 个 Resolved task 相同。因此 AgentGraph 没有获得净提升。当前最主要的正式失败结果是 `empty_patch`，其次是官方测试得到 `unresolved`；这说明执行链已经可用，但当前未训练 Director/AgentGraph 的 patch 产出能力仍弱。

## AgentGraph 与 repository Tool 使用

- Agent 数量：1 Agent 19 题、2 Agent 69 题、3 Agent 39 题、4 Agent 1 题。
- topology：single 19、serial 2-Agent 60、serial 3-Agent 以上 26、reciprocal 20、fan-in 2、mixed 1。
- Direct `search/view/edit/test/command`：132 / 65 / 76 / 9 / 2802。
- AgentGraph `search/view/edit/test/command`：204 / 167 / 86 / 24 / 2094。

这些 topology 由 Director 在自由 AgentGraph search space 中产生；配置没有预设 Coder、Reviewer、Tester，也没有固定 Coder→Reviewer→Tester 链。

## 实现来源

- SkillFlow：SWE-bench task identity、task-specific Conda environment、repository Tool action、workspace patch、官方 `evaluate_patch`。
- SWE-bench 官方 harness：`run_instance`、patch apply、test execution、grading 与 `Resolved`。
- FlowSteer：Canvas action、Agent runtime、execution feedback、trajectory 与 `FINISH`。
- 本项目必要适配：完整 Git diff、linked worktree Git common directory mount、official receipt admission 与配置接线；没有修改统一 orchestration core。

完整逐题 trajectory、Agent communication、Tool receipt、patch 和 evaluator receipt 保存在本机 `artifacts/swebench_skillflow_v3_stable_zero_v19/`。精简英文报告为 `reports/swebench_skillflow_v3_stable_zero_v19_report.md`。
