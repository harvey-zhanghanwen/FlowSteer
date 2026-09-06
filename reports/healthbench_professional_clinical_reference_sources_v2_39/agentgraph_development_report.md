# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 3；operational failures without trajectory：2
- evaluator valid：3；explicit FINISH：3；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.33333333 (3/5)；length-adjusted：0.33002093 (3/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：29；Director recorded phase calls：32；observed Tool calls（题内去重后求和）：11；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0 | 0.0286356 | true | true | false | 2 / 2 | node_1 → node_2 | 11 | 6 | 1 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | 0 | 0.048069 | true | true | false | 3 / 3 | node_1 → node_3; node_2 → node_3 | 8 | 2 | 0 |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | 1 | 0.9133582 | true | true | false | 2 / 2 | node_1 → node_2 | 10 | 3 | 0 |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 3/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"MiniMax-M3": 3, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 5, "unknown": 1}`；missing-source receipts=0
- Director turns=6；recorded phase calls=12；turn attempts=26 (6/6)；turn latency_ms=56285.259 (6/6)
- Tool calls：`{"healthbench-literature.search": 4, "healthbench-medrag.search": 1, "healthbench-trials.search": 1}`；raw receipts=12；duplicates=6；latency_ms=7513.4919 (6/6)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactGenerationError": 2, "TimeoutError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 47577 (8/11) | 5611 (8/11) | 53365 (8/11) | 59720.072 (8/11) |
| Director recorded phases | 245976 (12/12) | 2128 (12/12) | N/A (0/12) | 16946.365 (12/12) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 6501.516260206699, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1870, "output_reasoning_tokens": 63, "output_tokens": 225, "total_tokens": 2095}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

- Agent model calls：`{"MiniMax-M3": 1, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 5}`；missing-source receipts=1
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=66307.236 (5/5)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-trials.search": 1}`；raw receipts=2；duplicates=0；latency_ms=2590.0839 (2/2)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"CancelledError": 1, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 21751 (6/8) | 8235 (6/8) | 30045 (6/8) | 70817.715 (6/8) |
| Director recorded phases | 184393 (10/10) | 4008 (10/10) | N/A (0/10) | 30809.521 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 4144.355927594006, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2441, "output_reasoning_tokens": 85, "output_tokens": 329, "total_tokens": 2770}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

- Agent model calls：`{"MiniMax-M3": 6, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 3}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=52985.056 (5/5)
- Tool calls：`{"healthbench-knowledge.search": 1, "healthbench-literature.search": 2}`；raw receipts=9；duplicates=6；latency_ms=3282.517 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 41735 (9/10) | 6548 (9/10) | 48637 (9/10) | 70098.962 (9/10) |
| Director recorded phases | 183399 (10/10) | 2303 (10/10) | N/A (0/10) | 18864.116 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 4362.785450182855, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3408, "output_reasoning_tokens": 55, "output_tokens": 202, "total_tokens": 3610}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘。分数、valid/FINISH、图、调用、token、latency 均 N/A；完整执行不可恢复。原始 selected question、progress 与 collect failure 仅在 evaluator-private demo。

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘。分数、valid/FINISH、图、调用、token、latency 均 N/A；完整执行不可恢复。原始 selected question、progress 与 collect failure 仅在 evaluator-private demo。

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
