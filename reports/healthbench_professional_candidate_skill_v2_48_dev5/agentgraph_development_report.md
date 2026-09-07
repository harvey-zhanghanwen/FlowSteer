# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.40796703 (4/5)；length-adjusted：0.29038173 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：42；Director recorded phase calls：40；observed Tool calls（题内去重后求和）：15；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781 | 0 | 0.0562422 | true | true | false | 1 / 1 | no directed relations | 1 | 0 | 0 |
| healthbench-professional:9a160f86c59743692e46fab89aae42f2 | 1 | 0.9392302 | true | true | false | 1 / 1 | no directed relations | 9 | 3 | 0 |
| healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65 | 0.34615385 | 0.12289025 | true | true | false | 2 / 2 | node_1 → node_2 | 7 | 3 | 1 |
| healthbench-professional:2014ab7a9d8865f0da483817843ccbc5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9 | 0.28571429 | 0.043164286 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 25 | 9 | 1 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781

- Agent model calls：`{"qwen3.5-9b-local": 1}`；missing-source receipts=0
- Director turns=2；recorded phase calls=4；turn attempts=8 (2/2)；turn latency_ms=59745.553 (2/2)
- Tool calls：`{}`；raw receipts=0；duplicates=0；latency_ms=N/A (0/0)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 3413 (1/1) | 742 (1/1) | 4155 (1/1) | 11443.101 (1/1) |
| Director recorded phases | 24508 (4/4) | 2101 (4/4) | N/A (0/4) | 32217.397 (4/4) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 5976.014074869454, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1934, "output_reasoning_tokens": 225, "output_tokens": 383, "total_tokens": 2317}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:9a160f86c59743692e46fab89aae42f2

- Agent model calls：`{"MiniMax-M3": 9}`；missing-source receipts=0
- Director turns=4；recorded phase calls=8；turn attempts=16 (4/4)；turn latency_ms=53123.674 (4/4)
- Tool calls：`{"healthbench-authoritative.search": 3}`；raw receipts=6；duplicates=3；latency_ms=3520.0897 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactExecutionError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 44441 (9/9) | 5170 (9/9) | 50142 (9/9) | 127907.97 (9/9) |
| Director recorded phases | 68100 (8/8) | 1509 (8/8) | N/A (0/8) | 24078.278 (8/8) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 5073.497301898897, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1680, "output_reasoning_tokens": 44, "output_tokens": 137, "total_tokens": 1817}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65

- Agent model calls：`{"MiniMax-M3": 5, "deepseek-v4-flash": 2}`；missing-source receipts=0
- Director turns=6；recorded phase calls=12；turn attempts=28 (6/6)；turn latency_ms=152058.5 (6/6)
- Tool calls：`{"healthbench-literature.search": 1, "healthbench-source.read": 2}`；raw receipts=3；duplicates=0；latency_ms=6564.3587 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 40714 (5/7) | 8873 (5/7) | 49882 (5/7) | 124521.95 (5/7) |
| Director recorded phases | 131572 (12/12) | 3500 (12/12) | N/A (0/12) | 54799.53 (12/12) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 6255.053269676864, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 2048, "input_tokens": 8712, "output_reasoning_tokens": 157, "output_tokens": 401, "total_tokens": 9113}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:2014ab7a9d8865f0da483817843ccbc5

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9

- Agent model calls：`{"MiniMax-M3": 16, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 7}`；missing-source receipts=0
- Director turns=8；recorded phase calls=16；turn attempts=34 (8/8)；turn latency_ms=194486.22 (8/8)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 1, "healthbench-literature.search": 5, "healthbench-source.read": 1, "healthbench-trials.search": 1}`；raw receipts=18；duplicates=9；latency_ms=13079.377 (9/9)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 5, "runtime_failure_types": {"ReactExecutionError": 3, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 246019 (23/25) | 36171 (23/25) | 283075 (23/25) | 559300.17 (23/25) |
| Director recorded phases | 251576 (16/16) | 5214 (16/16) | N/A (0/16) | 75671.5 (16/16) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 5794.210323132575, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 2432, "input_tokens": 9875, "output_reasoning_tokens": 132, "output_tokens": 380, "total_tokens": 10255}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
