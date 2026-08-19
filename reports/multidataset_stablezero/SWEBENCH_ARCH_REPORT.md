# SWE-bench Regular Dev 架构报告

## Stable Zero

- 能力边界：detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness
- 配置固定任务数：**128**
- Runtime status：`failed_runtime_preflight`
- 结果：**不可测**；SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable
- 显式 FINISH receipt：**0/0**
- Tool receipt：**0**
- Environment transition receipt：**0**
- Coding action receipt：**0**
- optimizer update：**0**
- `STABLE_ZERO = NOT_RUN`

| 原生指标 | Direct/Simple Baseline | AgentGraph |
|---|---:|---:|
| resolved | 不可测 | 不可测 |

缺少原生 evaluator-valid paired result 时不填 0、不使用代理指标，也不从旧条件迁移成绩。

## Evidence scope 与协议限制

- Evidence scope：SWE-bench regular-dev architecture development；Verified 完整保留给最终评测
- Protocol：架构适配只使用 SWE-bench regular dev；唯一接受的 terminal 指标是官方 Docker harness resolved/resolved_rate，禁止使用代理评分。

- 没有 official Docker harness receipt 时，Direct、AgentGraph 与 Stable Zero 均为不可测。
- worktree preflight、generated diff、LLM judgement 或 local proxy test 都不能替代 resolved。

### 明确排除的历史结果

- 旧 swebench_verified_* development/evaluation：Verified 曾被用于适配，按数据隔离规则排除；完整 Verified 只允许用于最终评测。


## Correct Demo

无。没有当前 evidence scope 下的 evaluator-valid paired result，不能复用旧 test 结果或构造 Correct Demo。

## Wrong / Failure Demo

### Runtime state

- 当前边界：`failed_runtime_preflight`
- 原生 evaluator receipt：无
- 记录的错误：`SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable`

FIRST ERROR：当前运行尚未形成可评分的 terminal receipt；不能归因为 Director、AgentGraph、Tool、environment action、Coding Agent 或模型能力。
