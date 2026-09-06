# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.096464646 (4/5)；length-adjusted：0.0065520965 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：59；Director recorded phase calls：44；observed Tool calls（题内去重后求和）：20；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0 | -0.0533904 | true | true | false | 1 / 1 | no directed relations | 8 | 3 | 0 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | 0.36363636 | 0.22392756 | true | true | false | 4 / 4 | node_1 → node_2; node_2 → node_3; node_3 → node_4 | 17 | 5 | 1 |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | 0.46666667 | 0.41227667 | true | true | false | 4 / 4 | node_1 → node_2; node_2 → node_3; node_3 → node_4 | 25 | 10 | 0 |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | -0.44444444 | -0.55660544 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 9 | 2 | 2 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"MiniMax-M3": 8}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=12 (3/3)；turn latency_ms=52607.968 (3/3)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-medrag.search": 1, "healthbench-source.read": 1}`；raw receipts=6；duplicates=3；latency_ms=1326.7668 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactExecutionError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 34292 (8/8) | 5925 (8/8) | 40689 (8/8) | 146217.36 (8/8) |
| Director recorded phases | 52191 (6/6) | 1903 (6/6) | N/A (0/6) | 14131.678 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 7406.459089368582, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3142, "output_reasoning_tokens": 88, "output_tokens": 252, "total_tokens": 3394}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

- Agent model calls：`{"MiniMax-M3": 10, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 5}`；missing-source receipts=0
- Director turns=6；recorded phase calls=12；turn attempts=26 (6/6)；turn latency_ms=78634.818 (6/6)
- Tool calls：`{"healthbench-literature.search": 4, "healthbench-trials.search": 1}`；raw receipts=8；duplicates=3；latency_ms=9399.9108 (5/5)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 79035 (15/17) | 15570 (15/17) | 95195 (15/17) | 210140.14 (15/17) |
| Director recorded phases | 162362 (12/12) | 4087 (12/12) | N/A (0/12) | 31200.217 (12/12) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 9169.600386172533, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 2560, "input_tokens": 6371, "output_reasoning_tokens": 420, "output_tokens": 707, "total_tokens": 7078}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

- Agent model calls：`{"MiniMax-M3": 17, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 7}`；missing-source receipts=0
- Director turns=8；recorded phase calls=16；turn attempts=34 (8/8)；turn latency_ms=80047.946 (8/8)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 2, "healthbench-literature.search": 7}`；raw receipts=10；duplicates=0；latency_ms=22512.619 (10/10)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 4, "runtime_failure_types": {"ReactExecutionError": 3, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 140411 (24/25) | 20031 (24/25) | 161445 (24/25) | 289593.45 (24/25) |
| Director recorded phases | 203744 (16/16) | 4814 (16/16) | N/A (0/16) | 35255.344 (16/16) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 5722.721378318965, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3005, "output_reasoning_tokens": 93, "output_tokens": 269, "total_tokens": 3274}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

- Agent model calls：`{"MiniMax-M3": 8, "deepseek-v4-flash": 1}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=59157.424 (5/5)
- Tool calls：`{"healthbench-literature.search": 1, "healthbench-trials.search": 1}`；raw receipts=2；duplicates=0；latency_ms=3168.9064 (2/2)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 2, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{"HTTPError": 1}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 34370 (8/9) | 24524 (8/9) | 59366 (8/9) | 342580.89 (8/9) |
| Director recorded phases | 107349 (10/10) | 1808 (10/10) | N/A (0/10) | 13973.751 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 6, "api_calls": 6, "latency_ms": 7706.032522022724, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 1280, "input_tokens": 8332, "output_reasoning_tokens": 298, "output_tokens": 667, "total_tokens": 8999}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
