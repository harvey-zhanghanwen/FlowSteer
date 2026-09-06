# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.11666667 (4/5)；length-adjusted：0.084503067 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：49；Director recorded phase calls：64；observed Tool calls（题内去重后求和）：17；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0 | 0.0075558 | true | true | false | 4 / 4 | node_1 → node_2; node_1 → node_3; node_2 → node_4; node_3 → node_4 | 10 | 3 | 2 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | 0 | 0.005439 | true | true | false | 3 / 3 | node_1 → node_3; node_2 → node_3 | 11 | 3 | 0 |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | 0 | -0.072177 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 19 | 8 | 0 |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | 0.46666667 | 0.39719447 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 9 | 3 | 0 |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"MiniMax-M3": 6, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 3}`；missing-source receipts=0
- Director turns=9；recorded phase calls=18；turn attempts=38 (9/9)；turn latency_ms=147852.3 (9/9)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-knowledge.search": 1}`；raw receipts=6；duplicates=3；latency_ms=5072.4429 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 2, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactExecutionError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 53153 (10/10) | 13746 (10/10) | 67253 (10/10) | 193596.94 (10/10) |
| Director recorded phases | 237356 (18/18) | 6762 (18/18) | N/A (0/18) | 53579.505 (18/18) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 9495.397264137864, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2218, "output_reasoning_tokens": 36, "output_tokens": 205, "total_tokens": 2423}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

- Agent model calls：`{"MiniMax-M3": 1, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 9}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=67381.475 (5/5)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-trials.search": 1}`；raw receipts=6；duplicates=3；latency_ms=7702.2147 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 53441 (10/11) | 6413 (10/11) | 59854 (10/11) | 55840.476 (10/11) |
| Director recorded phases | 115297 (10/10) | 4173 (10/10) | N/A (0/10) | 32249.837 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 9457.973989658058, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3296, "output_reasoning_tokens": 274, "output_tokens": 542, "total_tokens": 3838}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

- Agent model calls：`{"deepseek-v4-flash": 2, "qwen3.5-9b-local": 17}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=62602.622 (5/5)
- Tool calls：`{"healthbench-authoritative.search": 3, "healthbench-bookshelf.search": 2, "healthbench-computation.calculator": 1, "healthbench-literature.search": 2}`；raw receipts=8；duplicates=0；latency_ms=10782.354 (8/8)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 89969 (17/19) | 22962 (17/19) | 112931 (17/19) | 177050.63 (17/19) |
| Director recorded phases | 113391 (10/10) | 2523 (10/10) | N/A (0/10) | 19463.051 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 6568.19810718298, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3128, "output_reasoning_tokens": 63, "output_tokens": 209, "total_tokens": 3337}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

- Agent model calls：`{"MiniMax-M3": 6, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 2}`；missing-source receipts=0
- Director turns=13；recorded phase calls=26；turn attempts=54 (13/13)；turn latency_ms=211424.22 (13/13)
- Tool calls：`{"healthbench-knowledge.search": 1, "healthbench-literature.search": 1, "healthbench-source.read": 1}`；raw receipts=3；duplicates=0；latency_ms=3562.8352 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 47990 (8/9) | 9755 (8/9) | 58099 (8/9) | 139464.51 (8/9) |
| Director recorded phases | 321621 (26/26) | 10476 (26/26) | N/A (0/26) | 80911.384 (26/26) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 8166.3341941311955, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3311, "output_reasoning_tokens": 125, "output_tokens": 295, "total_tokens": 3606}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
