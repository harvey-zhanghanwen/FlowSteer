# SWE-bench Regular Dev 架构报告

## Stable Zero

- 能力边界：detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness
- 配置固定任务数：**128**
- Runtime status：`failed_runtime_preflight`
- 结果：**不可测**；SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable
- Frozen selection 与失败 receipt：manifest=`artifacts/eval128_current/swebench_regular_dev/run_manifest.json`; selected tasks=`artifacts/eval128_current/swebench_regular_dev/selected_tasks.jsonl`
- Repository/Tool receipts：repository preflight=`artifacts/eval128_current/swebench_regular_dev/repository_preflight_receipt.json`; no-model canary=`artifacts/eval128_current/swebench_regular_dev/coding_tool_canary_receipt.json`
- 显式 FINISH receipt：**0/0**
- Model trajectory Tool receipt：**0**；no-model Coding Tool canary：**1 passed**
- Environment transition receipt：**0**
- Coding action receipt：**0**
- optimizer update：**0**
- `STABLE_ZERO = FAIL`

| 原生指标 | Direct/Simple Baseline | AgentGraph |
|---|---:|---:|
| resolved | 不可测 | 不可测 |

缺少原生 evaluator-valid paired result 时不填 0、不使用代理指标，也不从旧条件迁移成绩。

## Runtime / Search-space capability 与 Director natural policy adoption

- Runtime / search-space capability：配置声明了 `detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness`；没有 evaluator-valid AgentGraph trajectory 时，这只表示接口与搜索空间边界，不能解释为该能力已经被自然策略采用。
- Director natural policy adoption：**不可测**。当前没有完成的 AgentGraph trajectory，因而不能从配置项推断 topology、Tool、environment 或 Coding action 的实际采用。



### Coding Agent contract status

- 已接线：SkillFlow-derived task-pinned detached worktree、bounded Coding/ReAct、`list_files`、`search_code`、`view_file`、`bash`、`str_replace_editor`、AST filemap、`run_tests`、`diff`、ToolReceipt 与 official `resolved` evaluator。`str_replace_editor` 支持 view/create/str_replace/insert/undo_edit；保留 `exact_edit` 兼容入口。
- `apply_patch` 由官方 Codex CLI `--codex-run-as-apply-patch` 执行；项目没有自行实现 patch parser。显式 `file_pattern` 可搜索 tests、docs 与非 Python 文件。
- 终止时序：最后一次成功 changed edit 之后必须有有效 `run_tests` observation，其后必须重新取得 changed `diff`，再执行 `complete`；旧 test/diff 不能跨 edit revision 提交。当前预算为 9 turns / 8 Tool calls。
- 代码导航边界：提供 Python AST document structure 与 textual reference search；没有把它声明为 LSP symbol/reference implementation。
- Repository readiness：regular-dev 的四个仓库均已准备；SQLFluff 50、pvlib 63、marshmallow 9、astroid 6，共 128/128 base commit 可用于 task-pinned worktree。
- 实际 no-model canary `sqlfluff__sqlfluff-4764` 已覆盖 `str_replace_editor → apply_patch → run_tests → diff → bash → cleanup`。该 receipt 只证明 Tool/worktree contract 可运行，不是 SWE-bench resolved 结果。


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

FIRST INFRASTRUCTURE BLOCKER：`SWEbenchHarnessUnavailable: official SWE-bench Docker harness is unavailable`；官方 Docker harness preflight 未通过，因此没有启动 Direct/Coding Agent、没有 workspace edit/test、也没有 evaluator-valid resolved receipt。

这里的“没有 workspace edit/test”仅指正式模型 evaluation trajectory；独立 no-model Coding Tool canary 已完成编辑、测试、diff 与清理，但不进入 `resolved_rate`。
