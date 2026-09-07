# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.8 (4/5)；length-adjusted：0.77670785 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：77；Director recorded phase calls：54；observed Tool calls（题内去重后求和）：16；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:9566084de89c416408691006a6f06f9c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558 | 0.2 | 0.249098 | true | true | false | 1 / 1 | no directed relations | 7 | 3 | 1 |
| healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274 | 1 | 0.9219724 | true | true | false | 2 / 2 | node_1 → node_2 | 27 | 5 | 0 |
| healthbench-professional:e339f34a3a35f3f067422b5768287f7c | 1 | 0.9753628 | true | true | false | 4 / 4 | node_1 → node_2; node_2 → node_3; node_3 → node_4 | 33 | 6 | 0 |
| healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc | 1 | 0.9603982 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 10 | 2 | 0 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:9566084de89c416408691006a6f06f9c

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558

- Agent model calls：`{"deepseek-v4-flash": 1, "qwen3.5-9b-local": 6}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=20 (5/5)；turn latency_ms=58663.519 (5/5)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 2}`；raw receipts=9；duplicates=6；latency_ms=945.51857 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"CompletionArtifactQualityError": 1, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 20919 (6/7) | 8367 (6/7) | 29286 (6/7) | 59132.77 (6/7) |
| Director recorded phases | 77875 (10/10) | 3623 (10/10) | N/A (0/10) | 27771.1 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 6, "api_calls": 6, "latency_ms": 4959.72759835422, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 4202, "output_reasoning_tokens": 133, "output_tokens": 444, "total_tokens": 4646}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274

- Agent model calls：`{"MiniMax-M3": 21, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 4}`；missing-source receipts=0
- Director turns=8；recorded phase calls=16；turn attempts=32 (8/8)；turn latency_ms=76670.807 (8/8)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-literature.search": 1, "healthbench-medrag.search": 1, "healthbench-source.read": 1}`；raw receipts=7；duplicates=2；latency_ms=6174.5841 (5/5)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 5, "runtime_failure_types": {"ReactExecutionError": 3, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 138441 (25/27) | 26295 (25/27) | 165975 (25/27) | 307290.76 (25/27) |
| Director recorded phases | 183066 (16/16) | 5495 (16/16) | N/A (0/16) | 42434.083 (16/16) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 4353.78897562623, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1781, "output_reasoning_tokens": 49, "output_tokens": 139, "total_tokens": 1920}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:e339f34a3a35f3f067422b5768287f7c

- Agent model calls：`{"MiniMax-M3": 17, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 14}`；missing-source receipts=0
- Director turns=9；recorded phase calls=18；turn attempts=40 (9/9)；turn latency_ms=123184.04 (9/9)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-computation.calculator": 3, "healthbench-knowledge.search": 1}`；raw receipts=15；duplicates=9；latency_ms=2447.3271 (6/6)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 5, "runtime_failure_types": {"ReactExecutionError": 3, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 306355 (31/33) | 39845 (31/33) | 347203 (31/33) | 364227.85 (31/33) |
| Director recorded phases | 300579 (18/18) | 7344 (18/18) | N/A (0/18) | 55267.004 (18/18) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 4473.71032461524, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2787, "output_reasoning_tokens": 41, "output_tokens": 112, "total_tokens": 2899}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc

- Agent model calls：`{"MiniMax-M3": 7, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 2}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=68205.08 (5/5)
- Tool calls：`{"healthbench-literature.search": 1, "healthbench-medrag.search": 1}`；raw receipts=2；duplicates=0；latency_ms=1482.9666 (2/2)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 52442 (9/10) | 7179 (9/10) | 60034 (9/10) | 78511.056 (9/10) |
| Director recorded phases | 129959 (10/10) | 3354 (10/10) | N/A (0/10) | 26317.916 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 8542.765236459672, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2205, "output_reasoning_tokens": 0, "output_tokens": 81, "total_tokens": 2286}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
