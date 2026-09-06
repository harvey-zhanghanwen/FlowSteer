# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 5；operational failures without trajectory：0
- evaluator valid：5；explicit FINISH：5；terminal failure：0
- 严格全五题 native mean raw：0.29333333；length-adjusted：0.24921569；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.29333333 (5/5)；length-adjusted：0.24921569 (5/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：58；Director recorded phase calls：58；observed Tool calls（题内去重后求和）：19；全五题 actual Tool calls：19

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0 | 0.0257838 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 11 | 3 | 1 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | 0 | -0.0359268 | true | true | false | 2 / 2 | node_1 → node_2 | 11 | 6 | 0 |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | 1 | 0.835213 | true | true | false | 1 / 1 | no directed relations | 6 | 3 | 0 |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | 0.46666667 | 0.48804047 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 19 | 4 | 0 |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | 0 | -0.067032 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 11 | 3 | 0 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 5/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"deepseek-v4-flash": 2, "qwen3.5-9b-local": 9}`；missing-source receipts=0
- Director turns=7；recorded phase calls=14；turn attempts=30 (7/7)；turn latency_ms=99081.481 (7/7)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-knowledge.search": 1}`；raw receipts=3；duplicates=0；latency_ms=7558.753 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 44903 (9/11) | 7994 (9/11) | 52897 (9/11) | 57222.98 (9/11) |
| Director recorded phases | 147348 (14/14) | 7110 (14/14) | N/A (0/14) | 53037.04 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7790.895499289036, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1910, "output_reasoning_tokens": 62, "output_tokens": 223, "total_tokens": 2133}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

- Agent model calls：`{"MiniMax-M3": 10, "deepseek-v4-flash": 1}`；missing-source receipts=0
- Director turns=4；recorded phase calls=8；turn attempts=18 (4/4)；turn latency_ms=46769.575 (4/4)
- Tool calls：`{"healthbench-knowledge.search": 1, "healthbench-literature.search": 3, "healthbench-trials.search": 2}`；raw receipts=6；duplicates=0；latency_ms=7579.0313 (6/6)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 45914 (10/11) | 6602 (10/11) | 53106 (10/11) | 150707.23 (10/11) |
| Director recorded phases | 79546 (8/8) | 2566 (8/8) | N/A (0/8) | 19741.09 (8/8) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7057.036176323891, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 4307, "output_reasoning_tokens": 190, "output_tokens": 447, "total_tokens": 4754}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

- Agent model calls：`{"MiniMax-M3": 5, "deepseek-v4-flash": 1}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=12 (3/3)；turn latency_ms=31711.586 (3/3)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-literature.search": 1, "healthbench-source.read": 1}`；raw receipts=3；duplicates=0；latency_ms=5219.9817 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 25888 (5/6) | 3653 (5/6) | 29836 (5/6) | 76801.743 (5/6) |
| Director recorded phases | 47137 (6/6) | 1114 (6/6) | N/A (0/6) | 8934.3873 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 6538.24049141258, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 5238, "output_reasoning_tokens": 92, "output_tokens": 272, "total_tokens": 5510}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

- Agent model calls：`{"MiniMax-M3": 13, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 5}`；missing-source receipts=0
- Director turns=8；recorded phase calls=16；turn attempts=38 (8/8)；turn latency_ms=105683.49 (8/8)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-computation.calculator": 1, "healthbench-literature.search": 1, "healthbench-source.read": 1}`；raw receipts=6；duplicates=2；latency_ms=7528.2691 (4/4)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactExecutionError": 2, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 111649 (18/19) | 19048 (18/19) | 131464 (18/19) | 275637.2 (18/19) |
| Director recorded phases | 189852 (16/16) | 4307 (16/16) | N/A (0/16) | 32233.048 (16/16) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 5683.009559288621, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1939, "output_reasoning_tokens": 127, "output_tokens": 285, "total_tokens": 2224}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

- Agent model calls：`{"MiniMax-M3": 1, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 8, "qwen3.5-flash": 1}`；missing-source receipts=1
- Director turns=7；recorded phase calls=14；turn attempts=30 (7/7)；turn latency_ms=74949.827 (7/7)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-literature.search": 1}`；raw receipts=3；duplicates=0；latency_ms=3611.1351 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"CompletionArtifactEmpty": 1, "ReactExecutionError": 1, "ReactGenerationError": 1}}`；tool errors=`{"ValueError": 1}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 41121 (10/11) | 19338 (10/11) | 60518 (10/11) | 146731.48 (10/11) |
| Director recorded phases | 151629 (14/14) | 3536 (14/14) | N/A (0/14) | 27032.259 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 8959.309480153024, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 6644, "output_reasoning_tokens": 215, "output_tokens": 589, "total_tokens": 7233}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
