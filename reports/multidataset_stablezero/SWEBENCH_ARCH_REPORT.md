# SWE-bench Verified 架构报告

## Stable Zero

- 能力边界：detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness
- Protocol：唯一接受的指标是官方 resolved_rate；禁止使用代理评分。
- 冻结任务数：**128**
- Runtime status：`failed_runtime_preflight`
- 官方指标：`resolved_rate`
- 结果：**不可测**；`SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable`
- `STABLE_ZERO = FAIL`

两个固定 `astropy/astropy` base commit 的 repository/worktree preflight 已通过，但官方 Docker harness 不可用。fail-closed preflight 后没有执行模型/API 调用或 Coding Agent trajectory，也没有报告代理指标。

## Coding trace

不存在通过官方 evaluator 验证的 coding trajectory，因此不虚构 inspected files、edits、commands、tests、revisions 或 resolved status。

## 问题分类

`ENVIRONMENT_LIMITATION`：当前 runtime 中官方 SWE-bench Docker harness 无法连接 Docker daemon。
